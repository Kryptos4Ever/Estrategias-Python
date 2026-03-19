"""
Análisis de Comportamiento de Precios dentro de Rachas
═══════════════════════════════════════════════════════
Estrategia: Divergencia RSI · BTC/USDT · Velas Horarias
Parámetros: RSI_LENGTH=12, N=5

Para cada racha registra el precio de cada trade dentro de ella
(posición 1, 2, 3, ... N dentro de la racha) y analiza si los
precios suben o bajan a medida que la racha se alarga.

Preguntas que responde:
  · En rachas de COMPRA: ¿los precios bajan conforme avanza la racha?
    → Si sí: conviene aumentar el capital invertido a medida que avanza
  · En rachas de VENTA:  ¿los precios suben conforme avanza la racha?
    → Si sí: conviene vender más agresivamente al inicio y menos al final

Salida:
  · analisis_precios_rachas_RSI12_N5.png
  · analisis_precios_rachas_RSI12_N5.csv
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats as scipy_stats

# ─── Parámetros ──────────────────────────────────────────────────────────────
RSI_LENGTH = 12
N          = 5

try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL,
        USDT_PCT_TO_USE, BTC_PCT_TO_SELL, BTC_PCT_TO_ACCUMULATE,
        COMMISSION_PCT,
    )
except ImportError:
    DB_PATH               = r"btc_hourly.db"
    FECHA_INICIO          = '2021-11-10'
    FECHA_FIN             = '2025-10-06'
    SALDO_USDT_INICIAL    = 1000
    USDT_PCT_TO_USE       = 7
    BTC_PCT_TO_SELL       = 7
    BTC_PCT_TO_ACCUMULATE = 1
    COMMISSION_PCT        = 0.1

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]
OUT_PNG  = f"analisis_precios_rachas_RSI{RSI_LENGTH}_N{N}.png"
OUT_CSV  = f"analisis_precios_rachas_RSI{RSI_LENGTH}_N{N}.csv"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(series, length):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cargar_datos():
    conn  = sqlite3.connect(DB_PATH)
    df    = pd.read_sql(f"SELECT timestamp,open,high,low,close FROM {DB_TABLE} ORDER BY timestamp ASC", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if FECHA_INICIO: df = df[df["datetime"] >= pd.to_datetime(FECHA_INICIO)]
    if FECHA_FIN:    df = df[df["datetime"] <= pd.to_datetime(FECHA_FIN)]
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST CON REGISTRO DE POSICIÓN DENTRO DE RACHA
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_backtest(df):
    """
    Devuelve una lista de dicts, uno por trade, con:
      tipo          : BUY / SELL
      precio        : precio de ejecución
      pos_en_racha  : posición dentro de la racha (1 = primero, 2 = segundo, ...)
      precio_inicio : precio del primer trade de esa racha
      pct_vs_inicio : variación % vs el precio de inicio de la racha
      racha_id      : identificador único de la racha
    """
    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)
    rsi_l  = calcular_rsi(df["low"],  RSI_LENGTH).values.astype(float)
    rsi_h  = calcular_rsi(df["high"], RSI_LENGTH).values.astype(float)
    dts    = df["datetime"].values

    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    btc_balance       = 0.0

    trades           = []
    racha_tipo       = None
    racha_pos        = 0
    racha_precio_ini = None
    racha_id         = 0

    n = len(lows)
    for i in range(N, n):
        wl = lows[i-N:i]
        wh = highs[i-N:i]
        traded = False

        # ── COMPRA ───────────────────────────────────────────────────────────
        if usdt_balance > 0 and lows[i] < wl.min():
            idx_min = i - N + int(wl.argmin())
            if rsi_l[i] > rsi_l[idx_min]:
                price = lows[i]

                # Gestión racha
                if racha_tipo != "BUY":
                    racha_tipo = "BUY"
                    racha_pos  = 1
                    racha_precio_ini = price
                    racha_id  += 1
                else:
                    racha_pos += 1

                pct_vs_ini = (price - racha_precio_ini) / racha_precio_ini * 100

                trades.append({
                    "tipo"          : "BUY",
                    "datetime"      : str(pd.Timestamp(dts[i])),
                    "precio"        : round(price, 2),
                    "pos_en_racha"  : racha_pos,
                    "precio_inicio" : round(racha_precio_ini, 2),
                    "pct_vs_inicio" : round(pct_vs_ini, 4),
                    "racha_id"      : racha_id,
                })

                # Ejecutar trade
                usdt_a_usar   = usdt_balance * (USDT_PCT_TO_USE / 100)
                comision      = usdt_a_usar * (COMMISSION_PCT / 100)
                btc_adquirido = (usdt_a_usar - comision) / price
                usdt_balance      -= usdt_a_usar
                btc_en_posiciones += btc_adquirido
                usdt_invertido    += usdt_a_usar
                traded = True

        # ── VENTA ─────────────────────────────────────────────────────────────
        if not traded and btc_en_posiciones > 0 and highs[i] > wh.max():
            idx_max = i - N + int(wh.argmax())
            if rsi_h[i] < rsi_h[idx_max]:
                price = highs[i]

                # Gestión racha
                if racha_tipo != "SELL":
                    racha_tipo = "SELL"
                    racha_pos  = 1
                    racha_precio_ini = price
                    racha_id  += 1
                else:
                    racha_pos += 1

                pct_vs_ini = (price - racha_precio_ini) / racha_precio_ini * 100

                trades.append({
                    "tipo"          : "SELL",
                    "datetime"      : str(pd.Timestamp(dts[i])),
                    "precio"        : round(price, 2),
                    "pos_en_racha"  : racha_pos,
                    "precio_inicio" : round(racha_precio_ini, 2),
                    "pct_vs_inicio" : round(pct_vs_ini, 4),
                    "racha_id"      : racha_id,
                })

                btc_procesado  = btc_en_posiciones * (BTC_PCT_TO_SELL / 100)
                btc_a_acumular = btc_procesado * (BTC_PCT_TO_ACCUMULATE / 100)
                btc_a_vender   = btc_procesado - btc_a_acumular
                usdt_bruto     = btc_a_vender * price
                comision       = usdt_bruto * (COMMISSION_PCT / 100)
                proporcion         = btc_procesado / (btc_en_posiciones + btc_procesado)
                usdt_invertido    -= usdt_invertido * proporcion
                btc_en_posiciones -= btc_procesado
                btc_balance       += btc_a_acumular
                usdt_balance      += usdt_bruto - comision

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO
# ══════════════════════════════════════════════════════════════════════════════

def analizar_precios(df_trades: pd.DataFrame):
    """
    Para cada posición dentro de la racha (1, 2, 3, ...) calcula:
      · precio promedio, mediana, std
      · pct_vs_inicio promedio (variación acumulada desde el inicio de la racha)
      · tendencia de precios absolutos (¿sube o baja?)
    """
    resultados = {}
    for tipo in ["BUY", "SELL"]:
        sub = df_trades[df_trades["tipo"] == tipo].copy()

        # Normalizar precios dentro de cada racha para comparar entre rachas
        # (precio relativo al primer trade de la racha = precio_inicio)
        sub["precio_norm"] = sub["precio"] / sub["precio_inicio"]  # 1.0 = precio de inicio

        # Agrupar por posición dentro de la racha
        grp = sub.groupby("pos_en_racha").agg(
            n_obs          = ("precio",        "count"),
            precio_medio   = ("precio",        "mean"),
            precio_mediana = ("precio",        "median"),
            precio_std     = ("precio",        "std"),
            precio_norm_medio = ("precio_norm","mean"),
            precio_norm_std   = ("precio_norm","std"),
            pct_medio      = ("pct_vs_inicio", "mean"),
            pct_mediana    = ("pct_vs_inicio", "median"),
            pct_std        = ("pct_vs_inicio", "std"),
        ).reset_index()

        # Sólo posiciones con al menos 5 observaciones para que sean estadísticamente relevantes
        grp_sig = grp[grp["n_obs"] >= 5]

        # Tendencia lineal del precio normalizado vs posición
        if len(grp_sig) >= 2:
            slope, intercept, r, p, se = scipy_stats.linregress(
                grp_sig["pos_en_racha"], grp_sig["precio_norm_medio"]
            )
        else:
            slope, r, p = 0, 0, 1

        resultados[tipo] = {
            "tabla"    : grp,
            "tabla_sig": grp_sig,
            "slope"    : slope,
            "r_value"  : r,
            "p_value"  : p,
        }

        print(f"\n{'═'*65}")
        print(f"  Precios dentro de rachas de {tipo}")
        print(f"{'═'*65}")
        print(f"  Tendencia lineal del precio normalizado vs posición:")
        print(f"    slope   = {slope:+.6f}  ({'BAJA' if slope < 0 else 'SUBE'} por posición)")
        print(f"    R       = {r:.4f}  |  p-value = {p:.4f}  {'✓ SIGNIFICATIVO' if p < 0.05 else '✗ no significativo'}")
        print(f"\n  {'Pos':>4}  {'N':>5}  {'PrecNorm':>9}  {'Pct%Medio':>10}  {'Pct%Med':>9}  {'Pct%Std':>8}")
        print(f"  {'─'*55}")
        for _, r_row in grp.iterrows():
            marker = "◄" if r_row["n_obs"] < 5 else " "
            print(f"  {int(r_row['pos_en_racha']):>4}  "
                  f"{int(r_row['n_obs']):>5}  "
                  f"{r_row['precio_norm_medio']:>9.4f}  "
                  f"{r_row['pct_medio']:>+10.2f}%  "
                  f"{r_row['pct_mediana']:>+8.2f}%  "
                  f"{r_row['pct_std']:>8.2f}%  {marker}")
        print(f"  (◄ = menos de 5 obs, estadísticamente débil)")

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def graficar(df_trades: pd.DataFrame, resultados: dict):
    BUY_COLOR  = "#2196F3"
    SELL_COLOR = "#FF5722"
    BG_COLOR   = "#0D1117"
    PANEL_COLOR= "#161B22"
    TEXT_COLOR = "#E6EDF3"
    GRID_COLOR = "#21262D"
    ZERO_COLOR = "#555555"

    fig = plt.figure(figsize=(20, 16), facecolor=BG_COLOR)
    fig.suptitle(
        f"Comportamiento de Precios dentro de Rachas · RSI={RSI_LENGTH}  N={N}  |  {FECHA_INICIO} → {FECHA_FIN}",
        fontsize=13, color=TEXT_COLOR, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
    ax_pnorm_b, ax_pnorm_s  = axes[0], axes[1]  # precio normalizado medio por posición
    ax_pct_b,   ax_pct_s    = axes[2], axes[3]  # % vs inicio por posición
    ax_box_b,   ax_box_s    = axes[4], axes[5]  # boxplot por posición (máx posición 8)

    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    for tipo, ax_norm, ax_pct, ax_box, color, label in [
        ("BUY",  ax_pnorm_b, ax_pct_b, ax_box_b, BUY_COLOR,  "COMPRA"),
        ("SELL", ax_pnorm_s, ax_pct_s, ax_box_s, SELL_COLOR, "VENTA"),
    ]:
        res   = resultados[tipo]
        grp   = res["tabla"]
        grp_s = res["tabla_sig"]
        slope = res["slope"]
        r_val = res["r_value"]
        p_val = res["p_value"]
        sub   = df_trades[df_trades["tipo"] == tipo]

        # ── Panel 1: precio normalizado medio por posición ────────────────────
        ax_norm.bar(grp["pos_en_racha"], grp["precio_norm_medio"],
                    color=color, alpha=0.7, width=0.6, edgecolor=BG_COLOR)

        # Error bars solo donde hay suficientes obs
        ax_norm.errorbar(
            grp_s["pos_en_racha"],
            grp_s["precio_norm_medio"],
            yerr=grp_s["precio_norm_std"] / np.sqrt(grp_s["n_obs"]),
            fmt="none", color="white", capsize=4, linewidth=1.2, alpha=0.6
        )

        # Línea de regresión (solo sobre posiciones significativas)
        if len(grp_s) >= 2:
            x_fit = np.array([grp_s["pos_en_racha"].min(), grp_s["pos_en_racha"].max()])
            y_fit = res["slope"] * x_fit + (grp_s["precio_norm_medio"].iloc[0] - res["slope"] * grp_s["pos_en_racha"].iloc[0])
            ax_norm.plot(x_fit, y_fit, color="white", linewidth=1.5, linestyle="--", alpha=0.7, label="Tendencia")

        ax_norm.axhline(1.0, color=ZERO_COLOR, linewidth=1.0, linestyle="-")
        ax_norm.set_title(f"Precio Normalizado por Posición · {label}", fontsize=9, pad=6)
        ax_norm.set_xlabel("Posición en la racha")
        ax_norm.set_ylabel("Precio / Precio inicio racha")
        ax_norm.xaxis.set_major_locator(MaxNLocator(integer=True))
        sig_txt = "✓ Significativo" if p_val < 0.05 else "✗ No significativo"
        dir_txt = "↓ BAJA" if slope < 0 else "↑ SUBE"
        ax_norm.text(0.97, 0.04,
                     f"Slope: {slope:+.5f}  {dir_txt}\nR={r_val:.3f}  p={p_val:.3f}  {sig_txt}",
                     transform=ax_norm.transAxes, ha="right", va="bottom",
                     fontsize=7.5, color=TEXT_COLOR,
                     bbox=dict(facecolor="#2D333B", edgecolor=GRID_COLOR, boxstyle="round,pad=0.4"))

        # Anotar N de observaciones en cada barra
        for _, row in grp.iterrows():
            ax_norm.text(row["pos_en_racha"], row["precio_norm_medio"] + 0.001,
                         f"n={int(row['n_obs'])}", ha="center", va="bottom",
                         fontsize=6.5, color=TEXT_COLOR, alpha=0.8)

        # ── Panel 2: % vs inicio de racha por posición ────────────────────────
        bar_colors = [color if v <= 0 else "#4CAF50" for v in grp["pct_medio"]] if tipo == "BUY" \
                else [color if v >= 0 else "#2196F3" for v in grp["pct_medio"]]

        bars = ax_pct.bar(grp["pos_en_racha"], grp["pct_medio"],
                          color=bar_colors, alpha=0.8, width=0.6, edgecolor=BG_COLOR)

        # Banda de ±1 std (solo posiciones significativas)
        ax_pct.fill_between(
            grp_s["pos_en_racha"],
            grp_s["pct_medio"] - grp_s["pct_std"],
            grp_s["pct_medio"] + grp_s["pct_std"],
            color=color, alpha=0.15, label="±1 std"
        )

        ax_pct.axhline(0, color=ZERO_COLOR, linewidth=1.0)
        ax_pct.set_title(f"Variación % vs Precio Inicio de Racha · {label}", fontsize=9, pad=6)
        ax_pct.set_xlabel("Posición en la racha")
        ax_pct.set_ylabel("% cambio acumulado desde inicio")
        ax_pct.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_pct.legend(fontsize=7, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)

        for bar, (_, row) in zip(bars, grp.iterrows()):
            yoff = 0.05 if row["pct_medio"] >= 0 else -0.15
            ax_pct.text(bar.get_x() + bar.get_width()/2,
                        row["pct_medio"] + yoff,
                        f"{row['pct_medio']:+.1f}%",
                        ha="center", va="bottom", fontsize=6.5, color=TEXT_COLOR)

        # ── Panel 3: boxplot de pct_vs_inicio por posición (máx pos 8) ───────
        max_pos = min(int(sub["pos_en_racha"].max()), 8)
        data_box = [
            sub[sub["pos_en_racha"] == p]["pct_vs_inicio"].values
            for p in range(1, max_pos + 1)
            if len(sub[sub["pos_en_racha"] == p]) >= 3
        ]
        labels_box = [
            str(p)
            for p in range(1, max_pos + 1)
            if len(sub[sub["pos_en_racha"] == p]) >= 3
        ]

        if data_box:
            bp = ax_box.boxplot(
                data_box,
                labels=labels_box,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=TEXT_COLOR, linewidth=0.8),
                capprops=dict(color=TEXT_COLOR),
                flierprops=dict(markerfacecolor=TEXT_COLOR, marker=".", markersize=3, alpha=0.4),
                boxprops=dict(linewidth=0.8),
            )
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.6)

        ax_box.axhline(0, color=ZERO_COLOR, linewidth=1.0)
        ax_box.set_title(f"Dispersión % por Posición · {label}  (pos 1–{max_pos})", fontsize=9, pad=6)
        ax_box.set_xlabel("Posición en la racha")
        ax_box.set_ylabel("% vs precio inicio racha")

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"\n✓ Gráfico guardado: {OUT_PNG}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# RECOMENDACIÓN DE CAPITAL
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_recomendaciones(resultados: dict):
    print(f"\n{'═'*65}")
    print(f"  RECOMENDACIONES PARA GESTIÓN DINÁMICA DE CAPITAL")
    print(f"{'═'*65}")

    for tipo, label in [("BUY", "COMPRAS"), ("SELL", "VENTAS")]:
        res   = resultados[tipo]
        slope = res["slope"]
        p_val = res["p_value"]
        grp   = res["tabla_sig"]

        print(f"\n  {label}:")
        if p_val >= 0.05:
            print(f"    Los precios NO muestran tendencia estadísticamente")
            print(f"    significativa dentro de las rachas (p={p_val:.3f}).")
            print(f"    → Mantener porcentaje de capital fijo por operación.")
        else:
            if tipo == "BUY" and slope < 0:
                print(f"    ✓ Los precios BAJAN progresivamente dentro de la racha")
                print(f"      (slope={slope:+.5f}, p={p_val:.4f})")
                print(f"    → RECOMENDACIÓN: Aumentar gradualmente el capital")
                print(f"      usado en cada compra conforme avanza la racha.")
                print(f"      Ej: pos1=base%, pos2=base%×1.3, pos3=base%×1.6, ...")
            elif tipo == "BUY" and slope > 0:
                print(f"    ⚠ Los precios SUBEN dentro de la racha de compras")
                print(f"      (slope={slope:+.5f}, p={p_val:.4f})")
                print(f"    → RECOMENDACIÓN: Reducir capital en compras sucesivas.")
            elif tipo == "SELL" and slope > 0:
                print(f"    ✓ Los precios SUBEN progresivamente dentro de la racha")
                print(f"      (slope={slope:+.5f}, p={p_val:.4f})")
                print(f"    → RECOMENDACIÓN: Vender más agresivamente al inicio")
                print(f"      y reducir el % vendido conforme avanza la racha.")
                print(f"      Ej: pos1=base%×1.6, pos2=base%×1.3, pos3=base%, ...")
            elif tipo == "SELL" and slope < 0:
                print(f"    ⚠ Los precios BAJAN dentro de la racha de ventas")
                print(f"      (slope={slope:+.5f}, p={p_val:.4f})")
                print(f"    → RECOMENDACIÓN: Vender más tarde en la racha.")

        # Mostrar los % medios por posición de forma compacta
        if not grp.empty:
            pcts = "  ".join([f"pos{int(r['pos_en_racha'])}:{r['pct_medio']:+.1f}%"
                               for _, r in grp.iterrows()])
            print(f"    Δ% acumulado: {pcts}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ANÁLISIS PRECIOS EN RACHAS · RSI={RSI_LENGTH} · N={N}                  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    df = cargar_datos()
    print(f"✓ {len(df):,} velas cargadas")

    df_trades = ejecutar_backtest(df)
    print(f"✓ {len(df_trades):,} trades registrados")
    print(f"  BUY : {(df_trades['tipo']=='BUY').sum()}")
    print(f"  SELL: {(df_trades['tipo']=='SELL').sum()}")

    resultados = analizar_precios(df_trades)

    imprimir_recomendaciones(resultados)

    df_trades.to_csv(OUT_CSV, index=False)
    print(f"✓ CSV guardado: {OUT_CSV}")

    print("Generando gráficos...")
    graficar(df_trades, resultados)


if __name__ == "__main__":
    main()
