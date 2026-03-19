"""
Optimizador de Parámetros de Capital — BTC/USDT
────────────────────────────────────────────────
Dado un conjunto fijo de parámetros de señal (EMA200 + Bollinger + Williams),
itera todas las combinaciones de:
  · USDT_PCT_TO_USE
  · BTC_PCT_TO_SELL
  · BTC_PCT_TO_ACCUMULATE

Calcula un score compuesto por combinación y produce:
  · Ranking en consola (Top 10)
  · optimizacion_capital.csv   (tabla completa)
  · optimizacion_capital.png   (heatmaps)

Los indicadores se calculan UNA SOLA VEZ antes del grid search.
"""

import sqlite3
import json
import os
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from config_EMA_BB_Williams import (
    DB_PATH, SALDO_USDT_INICIAL, FECHA_INICIO, FECHA_FIN,
    EMA_LENGTH, DIST_EMA_BUY, DIST_EMA_SELL,
    BB_LENGTH, BB_STD, BB_BUY, BB_SELL,
    WILLIAMS_LENGTH, WILLIAMS_BUY, WILLIAMS_SELL,
    COMMISSION_PCT,
)

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

# ─────────────────────────────────────────────────────────────
# GRIDS A EVALUAR  (modificá estos rangos según necesites)
# ─────────────────────────────────────────────────────────────
GRID_USDT_PCT      = [2, 5, 8, 10, 15, 20, 25, 30]
GRID_BTC_SELL      = [2, 5, 8, 10, 15, 20, 25, 30]
GRID_BTC_ACCUMULATE = [0.1, 0.5, 1, 2, 4]

# ─────────────────────────────────────────────────────────────
# PESOS DEL SCORE COMPUESTO  (deben sumar 1.0)
# ─────────────────────────────────────────────────────────────
W_PNL       = 0.35    # rentabilidad total en USDT
W_BTC       = 0.35    # BTC acumulado libre
W_USDT      = 0.15    # penaliza agotar el USDT (residual alto = mejor)
W_DRAWDOWN  = 0.10    # penaliza drawdowns grandes
W_BALANCE   = 0.05    # penaliza desbalance compras/ventas


# ─────────────────────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────────────────────

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_bollinger_pct_b(close, length, std_mult):
    mid   = close.rolling(length).mean()
    std   = close.rolling(length).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return (close - lower) / (upper - lower + 1e-10)

def calc_williams_r(high, low, close, length):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


# ─────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN (una sola vez)
# ─────────────────────────────────────────────────────────────

def cargar_y_preparar() -> pd.DataFrame:
    print(f"Cargando datos de {DB_PATH}...")
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, high, low, close
        FROM   {DB_TABLE}
        ORDER  BY timestamp ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if FECHA_INICIO:
        df = df[df["datetime"] >= pd.to_datetime(FECHA_INICIO)]
    if FECHA_FIN:
        df = df[df["datetime"] <= pd.to_datetime(FECHA_FIN)]
    df = df.reset_index(drop=True)
    print(f"Velas: {len(df):,}  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")

    print("Calculando indicadores (una sola vez)...")
    ema200             = calc_ema(df["close"], EMA_LENGTH)
    df["ema200_dist"]  = (df["close"] - ema200) / ema200 * 100
    df["bb_pct_b"]     = calc_bollinger_pct_b(df["close"], BB_LENGTH, BB_STD)
    df["williams_r"]   = calc_williams_r(df["high"], df["low"], df["close"], WILLIAMS_LENGTH)

    # Señales booleanas precalculadas
    df["sig_buy"]  = (
        (df["ema200_dist"] <= -DIST_EMA_BUY) &
        (df["bb_pct_b"]   <=  BB_BUY)        &
        (df["williams_r"] <=  WILLIAMS_BUY)
    )
    df["sig_sell"] = (
        (df["ema200_dist"] >=  DIST_EMA_SELL) &
        (df["bb_pct_b"]   >=  BB_SELL)        &
        (df["williams_r"] >=  WILLIAMS_SELL)
    )

    # Eliminar filas con NaN en indicadores (warm-up)
    df = df.dropna(subset=["ema200_dist", "bb_pct_b", "williams_r"]).reset_index(drop=True)

    n_buy  = df["sig_buy"].sum()
    n_sell = df["sig_sell"].sum()
    print(f"Señales → Compras: {n_buy:,}   Ventas: {n_sell:,}")
    return df


# ─────────────────────────────────────────────────────────────
# BACKTEST PARA UNA COMBINACIÓN
# ─────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame,
             usdt_pct: float,
             btc_sell_pct: float,
             btc_accum_pct: float) -> dict:

    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    positions_count   = 0
    usdt_invertido    = 0.0

    # Para drawdown: guardamos portfolio en cada trade
    portfolio_snapshots = [SALDO_USDT_INICIAL]

    price_final = float(df["close"].iloc[-1])

    sig_buy  = df["sig_buy"].values
    sig_sell = df["sig_sell"].values
    low_arr  = df["low"].values
    high_arr = df["high"].values
    close_arr = df["close"].values

    for i in range(len(df)):

        es_compra = sig_buy[i]
        es_venta  = sig_sell[i]

        if es_compra and usdt_balance > 0:
            usdt_a_usar   = usdt_balance * (usdt_pct / 100)
            comision      = usdt_a_usar  * (COMMISSION_PCT / 100)
            usdt_neto     = usdt_a_usar  - comision
            btc_adquirido = usdt_neto / low_arr[i]

            usdt_balance      -= usdt_a_usar
            btc_en_posiciones += btc_adquirido
            usdt_invertido    += usdt_a_usar
            positions_count   += 1

            snap = usdt_balance + (btc_balance + btc_en_posiciones) * close_arr[i]
            portfolio_snapshots.append(snap)

        elif es_venta and btc_en_posiciones > 0:
            btc_procesado  = btc_en_posiciones * (btc_sell_pct  / 100)
            btc_a_acumular = btc_procesado     * (btc_accum_pct / 100)
            btc_a_vender   = btc_procesado - btc_a_acumular

            usdt_bruto = btc_a_vender * high_arr[i]
            comision   = usdt_bruto   * (COMMISSION_PCT / 100)
            usdt_neto  = usdt_bruto   - comision

            proporcion         = btc_procesado / (btc_en_posiciones + btc_procesado)
            usdt_invertido    -= usdt_invertido * proporcion

            btc_en_posiciones -= btc_procesado
            btc_balance       += btc_a_acumular
            usdt_balance      += usdt_neto
            positions_count   -= 1

            snap = usdt_balance + (btc_balance + btc_en_posiciones) * close_arr[i]
            portfolio_snapshots.append(snap)

    # Métricas finales
    btc_total   = btc_balance + btc_en_posiciones
    portfolio_f = usdt_balance + btc_total * price_final
    pnl_pct     = (portfolio_f - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100

    # Drawdown máximo sobre snapshots de trades
    snaps   = np.array(portfolio_snapshots)
    peak    = np.maximum.accumulate(snaps)
    dd      = (snaps - peak) / (peak + 1e-10) * 100
    max_dd  = float(dd.min())

    usdt_residual_pct = usdt_balance / SALDO_USDT_INICIAL * 100

    return {
        "usdt_pct"          : usdt_pct,
        "btc_sell_pct"      : btc_sell_pct,
        "btc_accum_pct"     : btc_accum_pct,
        "portfolio_final"   : round(portfolio_f, 2),
        "pnl_pct"           : round(pnl_pct, 4),
        "btc_acumulado"     : round(btc_balance, 8),
        "btc_value_final"   : round(btc_balance * price_final, 2),
        "usdt_residual"     : round(usdt_balance, 4),
        "usdt_residual_pct" : round(usdt_residual_pct, 4),
        "positions_count"   : positions_count,
        "drawdown_max"      : round(max_dd, 4),
    }


# ─────────────────────────────────────────────────────────────
# SCORE COMPUESTO
# ─────────────────────────────────────────────────────────────

def calcular_scores(resultados: list) -> pd.DataFrame:
    df = pd.DataFrame(resultados)

    def norm(serie, invert=False):
        mn, mx = serie.min(), serie.max()
        if mx == mn:
            return pd.Series(0.5, index=serie.index)
        n = (serie - mn) / (mx - mn)
        return 1 - n if invert else n

    df["n_pnl"]      = norm(df["pnl_pct"])
    df["n_btc"]      = norm(df["btc_acumulado"])
    df["n_usdt"]     = norm(df["usdt_residual_pct"])
    df["n_drawdown"] = norm(df["drawdown_max"], invert=True)   # menor dd = mejor
    df["n_balance"]  = norm(df["positions_count"].abs(), invert=True)  # más cercano a 0 = mejor

    df["score"] = (
        df["n_pnl"]      * W_PNL      +
        df["n_btc"]      * W_BTC      +
        df["n_usdt"]     * W_USDT     +
        df["n_drawdown"] * W_DRAWDOWN +
        df["n_balance"]  * W_BALANCE
    )
    df["score"] = df["score"].round(6)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generar_graficos(df_res: pd.DataFrame):
    accum_vals = sorted(df_res["btc_accum_pct"].unique())
    n_accum    = len(accum_vals)

    metricas = [
        ("score",           "Score Compuesto",     "RdYlGn",  False),
        ("pnl_pct",         "PnL %",               "RdYlGn",  False),
        ("btc_acumulado",   "BTC Acumulado",        "RdYlGn",  False),
        ("usdt_residual_pct","USDT Residual %",     "RdYlGn",  False),
        ("drawdown_max",    "Drawdown Máx %",       "RdYlGn_r",True),
        ("positions_count", "positions_count final","coolwarm", False),
    ]

    fig, axes = plt.subplots(
        len(metricas), n_accum,
        figsize=(5 * n_accum, 4 * len(metricas)),
        squeeze=False
    )
    fig.patch.set_facecolor("#1a1a2e")

    for row, (metrica, titulo, cmap, _) in enumerate(metricas):
        for col, accum in enumerate(accum_vals):
            ax  = axes[row][col]
            sub = df_res[df_res["btc_accum_pct"] == accum]

            pivot = sub.pivot_table(
                index="usdt_pct", columns="btc_sell_pct", values=metrica
            )

            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                           origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v}%" for v in pivot.columns], fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v}%" for v in pivot.index], fontsize=7)

            # Anotar valores en cada celda
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        fmt = f"{val:.1f}" if abs(val) < 1000 else f"{val:.0f}"
                        ax.text(j, i, fmt, ha="center", va="center",
                                fontsize=6, color="white",
                                fontweight="bold")

            if row == 0:
                ax.set_title(f"BTC_ACCUM={accum}%\n{titulo}",
                             color="white", fontsize=9, fontweight="bold")
            else:
                ax.set_title(f"{titulo}  (accum={accum}%)",
                             color="white", fontsize=8)

            if col == 0:
                ax.set_ylabel("USDT_PCT_TO_USE", color="white", fontsize=8)
            ax.set_xlabel("BTC_PCT_TO_SELL", color="white", fontsize=8)

            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

    fig.suptitle(
        "Optimización de Parámetros de Capital — BTC/USDT\n"
        f"EMA{EMA_LENGTH} dist±{DIST_EMA_BUY}%  |  BB({BB_LENGTH},{BB_STD})  "
        f"|  Williams({WILLIAMS_LENGTH})",
        fontsize=13, fontweight="bold", color="white", y=1.005
    )

    plt.tight_layout()
    nombre = "optimizacion_capital.png"
    plt.savefig(nombre, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Gráfico guardado: {nombre}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────

def imprimir_reporte(df_res: pd.DataFrame):
    sep = "=" * 78

    print(f"\n{sep}")
    print("  TOP 10 — MEJORES COMBINACIONES POR SCORE COMPUESTO")
    print(f"  Pesos: PnL={W_PNL}  BTC={W_BTC}  USDT={W_USDT}  "
          f"DD={W_DRAWDOWN}  Balance={W_BALANCE}")
    print(sep)
    print(f"  {'#':>3}  {'USDT%':>6}  {'Sell%':>6}  {'Accum%':>7}  "
          f"{'Score':>7}  {'PnL%':>8}  {'BTC acum':>10}  "
          f"{'USDT res%':>10}  {'MaxDD%':>8}  {'PosCount':>9}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  "
          f"{'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}")

    for i, row in df_res.head(10).iterrows():
        print(f"  {i+1:>3}  {row['usdt_pct']:>6.1f}  {row['btc_sell_pct']:>6.1f}  "
              f"{row['btc_accum_pct']:>7.1f}  {row['score']:>7.4f}  "
              f"{row['pnl_pct']:>8.2f}  {row['btc_acumulado']:>10.6f}  "
              f"{row['usdt_residual_pct']:>10.2f}  {row['drawdown_max']:>8.2f}  "
              f"{row['positions_count']:>9}")

    # Ganador
    best = df_res.iloc[0]
    print(f"\n{sep}")
    print("  COMBINACIÓN GANADORA — pegá esto en config_EMA_BB_Williams.py")
    print(sep)
    print(f"\n  USDT_PCT_TO_USE       = {best['usdt_pct']}")
    print(f"  BTC_PCT_TO_SELL       = {best['btc_sell_pct']}")
    print(f"  BTC_PCT_TO_ACCUMULATE = {best['btc_accum_pct']}")
    print(f"\n  Portfolio final : ${best['portfolio_final']:,.2f}  "
          f"({best['pnl_pct']:+.2f}%)")
    print(f"  BTC acumulado   :  {best['btc_acumulado']:.8f} ₿  "
          f"(${best['btc_value_final']:,.2f})")
    print(f"  USDT residual   : ${best['usdt_residual']:,.2f}  "
          f"({best['usdt_residual_pct']:.2f}% del capital inicial)")
    print(f"  Drawdown máximo : {best['drawdown_max']:.2f}%")
    print(sep)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  OPTIMIZADOR DE PARÁMETROS DE CAPITAL — BTC/USDT")
    print("=" * 78)

    total_combinaciones = (len(GRID_USDT_PCT) *
                           len(GRID_BTC_SELL) *
                           len(GRID_BTC_ACCUMULATE))
    print(f"\n  Combinaciones a evaluar: {total_combinaciones}")
    print(f"  USDT_PCT      : {GRID_USDT_PCT}")
    print(f"  BTC_SELL_PCT  : {GRID_BTC_SELL}")
    print(f"  BTC_ACCUM_PCT : {GRID_BTC_ACCUMULATE}")

    # 1. Cargar datos y precalcular indicadores
    df = cargar_y_preparar()

    # 2. Grid search
    print(f"\nEjecutando {total_combinaciones} backtests...")
    t0         = time.time()
    resultados = []
    combinaciones = list(itertools.product(
        GRID_USDT_PCT, GRID_BTC_SELL, GRID_BTC_ACCUMULATE
    ))

    for idx, (usdt_pct, btc_sell, btc_accum) in enumerate(combinaciones, 1):
        r = backtest(df, usdt_pct, btc_sell, btc_accum)
        resultados.append(r)

        # Progreso cada 20 combinaciones
        if idx % 20 == 0 or idx == total_combinaciones:
            elapsed = time.time() - t0
            eta     = elapsed / idx * (total_combinaciones - idx)
            print(f"  {idx:>4}/{total_combinaciones}  "
                  f"({idx/total_combinaciones*100:.1f}%)  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    print(f"\nGrid search completado en {time.time()-t0:.1f}s")

    # 3. Calcular scores y ordenar
    df_res = calcular_scores(resultados)

    # 4. Reporte en consola
    imprimir_reporte(df_res)

    # 5. Guardar CSV
    cols_export = [
        "score", "usdt_pct", "btc_sell_pct", "btc_accum_pct",
        "portfolio_final", "pnl_pct", "btc_acumulado", "btc_value_final",
        "usdt_residual", "usdt_residual_pct", "positions_count", "drawdown_max"
    ]
    df_res[cols_export].to_csv("optimizacion_capital.csv", index=False)
    print("\nTabla completa guardada: optimizacion_capital.csv")

    # 6. Gráficos
    print("Generando heatmaps...")
    generar_graficos(df_res)


if __name__ == "__main__":
    main()
