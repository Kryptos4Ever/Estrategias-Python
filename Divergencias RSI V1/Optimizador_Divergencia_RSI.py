"""
Optimizador — Divergencia RSI · Capital Dinámico ATH/ATL
═════════════════════════════════════════════════════════
BTC/USDT · Velas horarias · Backtesting + Grid Search

LÓGICA DE SEÑALES
─────────────────
COMPRA (divergencia alcista en mínimos locales):
  · low[i] < min(lows[i-N : i])       → nuevo mínimo local de precio
  · RSI(low[i]) > RSI(low[idx_min])   → RSI no confirma la caída
  → Presión vendedora agotada, probable reversión al alza

VENTA (divergencia bajista en máximos locales):
  · high[i] > max(highs[i-N : i])     → nuevo máximo local de precio
  · RSI(high[i]) < RSI(high[idx_max]) → RSI no confirma la suba
  → Presión compradora agotada, probable reversión a la baja

CAPITAL DINÁMICO POR RACHA (ATH / ATL)
────────────────────────────────────────
COMPRAS:
  caida%        = (ATH - precio_low) / ATH × 100
  pct_usdt      = clamp(caida / ATH_CAIDA_MAXIMA, 0, 1) ^ FACTOR_CAIDA × 100
  capital_racha = usdt_disponible × pct_usdt / 100
  → repartido en STREAK_COMPRAS slots con razón geométrica R_COMPRA

VENTAS:
  subida%    = (precio_high - ATL) / ATL × 100
  pct_btc    = clamp(subida / ATL_SUBIDA_MAXIMA, 0, 1) ^ FACTOR_SUBIDA × 100
  btc_racha  = btc_en_posiciones × pct_btc / 100
  → repartido en STREAK_VENTAS slots con razón geométrica R_VENTA

PARÁMETROS OPTIMIZADOS (grid search)
──────────────────────────────────────
  N            : ventana de búsqueda del extremo local  (5 → 50, paso 1)
  RSI_LENGTH   : período del RSI                        (7 → 21, paso 1)
  Total combos : ~690 backtests

El resto de parámetros se toman fijos desde config.py.

MÉTRICAS DE SALIDA (por combinación)
─────────────────────────────────────
  pnl_pct               : rentabilidad total del portfolio  ← ranking principal
  portfolio_final       : valor final en USDT
  total_trades          : compras + ventas ejecutadas
  total_compras/ventas  : conteo separado
  max_racha_compras/ventas : mayor racha consecutiva de cada tipo
  avg_racha_compras/ventas : promedio de rachas
  positions_count_final : sesgo acumulado al cierre (+ = más compras)

SALIDA
──────
  optimizacion_resultados.csv   → tabla completa rankeada por pnl_pct
  optimizacion_resultados.json  → misma data en JSON
  optimizacion_top10.txt        → resumen legible del top 10
"""

import sqlite3
import json
import os
import time
import numpy as np
import pandas as pd
from itertools import product

# ── Importar config ───────────────────────────────────────────────────────────
try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL,
        STREAK_COMPRAS, STREAK_VENTAS, R_COMPRA, R_VENTA,
        ATH_CAIDA_MAXIMA, FACTOR_CAIDA,
        ATL_SUBIDA_MAXIMA, FACTOR_SUBIDA,
        USDT_RESERVA_PCT,
        BTC_PCT_TO_ACCUMULATE,
        COMMISSION_PCT,
    )
    print("✓ config.py cargado correctamente")
except ImportError:
    print("⚠ config.py no encontrado — usando valores por defecto")
    DB_PATH               = r"btc_hourly.db"
    FECHA_INICIO          = '2021-11-10'
    FECHA_FIN             = '2025-10-06'
    SALDO_USDT_INICIAL    = 1000
    STREAK_COMPRAS        = 7
    STREAK_VENTAS         = 7
    R_COMPRA              = 1.5
    R_VENTA               = 1.5
    ATH_CAIDA_MAXIMA      = 80
    FACTOR_CAIDA          = 2.0
    ATL_SUBIDA_MAXIMA     = 300
    FACTOR_SUBIDA         = 2.0
    USDT_RESERVA_PCT      = 5
    BTC_PCT_TO_ACCUMULATE = 1
    COMMISSION_PCT        = 0.1

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100

# ── Espacio de búsqueda ───────────────────────────────────────────────────────
N_RANGE          = range(5,  51, 1)   # ventana local: 5 → 50
RSI_LENGTH_RANGE = range(7,  22, 1)   # período RSI  : 7 → 21

# ── Archivos de salida ────────────────────────────────────────────────────────
OUT_CSV  = "optimizacion_resultados.csv"
OUT_JSON = "optimizacion_resultados.json"
OUT_TXT  = "optimizacion_top10.txt"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI clásico de Wilder (EWM), idéntico al de TradingView."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_slots(capital: float, n: int, r: float) -> list:
    """
    n slots en progresión geométrica de razón r cuya suma = capital.
    Retorna lista vacía si capital <= 0.
    """
    if capital <= 0 or n <= 0:
        return []
    if abs(r - 1.0) < 1e-9:
        return [capital / n] * n
    s = (r ** n - 1) / (r - 1)
    a = capital / s
    return [a * (r ** i) for i in range(n)]


def pct_compra(precio_low: float, ath: float) -> float:
    """% [0-100] del USDT disponible a comprometer según caída desde ATH."""
    if ath <= 0:
        return 0.0
    caida = (ath - precio_low) / ath * 100
    ratio = max(0.0, min(1.0, caida / ATH_CAIDA_MAXIMA))
    return (ratio ** FACTOR_CAIDA) * 100.0


def pct_venta(precio_high: float, atl: float) -> float:
    """% [0-100] del BTC en posiciones a comprometer según subida desde ATL."""
    if atl <= 0:
        return 0.0
    subida = (precio_high - atl) / atl * 100
    ratio  = max(0.0, min(1.0, subida / ATL_SUBIDA_MAXIMA))
    return (ratio ** FACTOR_SUBIDA) * 100.0


def cargar_datos() -> pd.DataFrame:
    """Carga velas desde SQLite y aplica filtro de fechas."""
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, open, high, low, close
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
    print(f"✓ Velas cargadas  : {len(df):,}")
    print(f"  Desde           : {df['datetime'].iloc[0]}")
    print(f"  Hasta           : {df['datetime'].iloc[-1]}")
    return df


def calcular_rachas(trade_types: list) -> dict:
    """Métricas de rachas consecutivas de cada tipo."""
    if not trade_types:
        return {
            "max_racha_compras": 0, "max_racha_ventas": 0,
            "avg_racha_compras": 0, "avg_racha_ventas": 0,
        }

    rachas_buy, rachas_sell = [], []
    current_type  = trade_types[0]
    current_count = 1

    for t in trade_types[1:]:
        if t == current_type:
            current_count += 1
        else:
            (rachas_buy if current_type == "BUY" else rachas_sell).append(current_count)
            current_type  = t
            current_count = 1
    (rachas_buy if current_type == "BUY" else rachas_sell).append(current_count)

    return {
        "max_racha_compras": max(rachas_buy,  default=0),
        "max_racha_ventas" : max(rachas_sell, default=0),
        "avg_racha_compras": round(sum(rachas_buy)  / len(rachas_buy),  2) if rachas_buy  else 0,
        "avg_racha_ventas" : round(sum(rachas_sell) / len(rachas_sell), 2) if rachas_sell else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEO DEL BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_backtest(
    lows:     np.ndarray,
    highs:    np.ndarray,
    closes:   np.ndarray,
    rsi_low:  np.ndarray,
    rsi_high: np.ndarray,
    N:        int,
) -> dict:
    """
    Backtest completo con capital dinámico ATH/ATL y slots por racha.
    Recibe arrays pre-calculados para máxima velocidad en el grid search.
    """
    n_velas = len(lows)

    # ── Estado del portfolio ──────────────────────────────────────────────────
    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    positions_count   = 0

    # ── ATH / ATL ─────────────────────────────────────────────────────────────
    ath = float(highs[0])
    atl = float(lows[0])

    # ── Estado de la racha activa ─────────────────────────────────────────────
    racha_tipo  = None    # "BUY" | "SELL" | None
    racha_slots = []
    racha_idx   = 0

    trade_types = []

    for i in range(N, n_velas):

        # Actualizar ATH / ATL con la vela actual
        if highs[i] > ath: ath = float(highs[i])
        if lows[i]  < atl: atl = float(lows[i])

        if np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]):
            continue

        window_lows  = lows[i - N : i]
        window_highs = highs[i - N : i]

        # ── SEÑAL DE COMPRA ───────────────────────────────────────────────────
        señal_compra = False
        if lows[i] < window_lows.min():
            idx_min = i - N + int(window_lows.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                señal_compra = True

        # ── SEÑAL DE VENTA (nunca ambas en la misma vela) ────────────────────
        señal_venta = False
        if not señal_compra:
            if highs[i] > window_highs.max():
                idx_max = i - N + int(window_highs.argmax())
                if rsi_high[i] < rsi_high[idx_max]:
                    señal_venta = True

        # ── EJECUTAR COMPRA ───────────────────────────────────────────────────
        if señal_compra:
            usdt_disponible = usdt_balance - USDT_RESERVA
            if usdt_disponible <= 0:
                continue

            # Nueva racha → pre-calcular slots sobre el capital de la racha
            if racha_tipo != "BUY":
                capital_racha = usdt_disponible * pct_compra(lows[i], ath) / 100.0
                racha_slots   = calcular_slots(capital_racha, STREAK_COMPRAS, R_COMPRA)
                racha_idx     = 0
                racha_tipo    = "BUY"

            if racha_idx >= len(racha_slots):   # streak superado → ignorar
                continue

            usdt_a_usar = min(racha_slots[racha_idx], usdt_disponible)
            if usdt_a_usar <= 0:
                continue

            comision      = usdt_a_usar * (COMMISSION_PCT / 100)
            btc_adquirido = (usdt_a_usar - comision) / lows[i]

            usdt_balance      -= usdt_a_usar
            btc_en_posiciones += btc_adquirido
            usdt_invertido    += usdt_a_usar
            positions_count   += 1
            racha_idx         += 1
            trade_types.append("BUY")

        # ── EJECUTAR VENTA ────────────────────────────────────────────────────
        elif señal_venta and btc_en_posiciones > 0:

            # Nueva racha → pre-calcular slots sobre el BTC de la racha
            if racha_tipo != "SELL":
                btc_racha   = btc_en_posiciones * pct_venta(highs[i], atl) / 100.0
                racha_slots = calcular_slots(btc_racha, STREAK_VENTAS, R_VENTA)
                racha_idx   = 0
                racha_tipo  = "SELL"

            if racha_idx >= len(racha_slots):   # streak superado → ignorar
                continue

            btc_slot = min(racha_slots[racha_idx], btc_en_posiciones)
            if btc_slot <= 0:
                continue

            btc_a_acumular  = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_a_vender    = btc_slot - btc_a_acumular
            usdt_bruto      = btc_a_vender * highs[i]
            comision        = usdt_bruto * (COMMISSION_PCT / 100)
            usdt_neto       = usdt_bruto - comision

            costo_prop      = usdt_invertido * (btc_slot / btc_en_posiciones)
            usdt_invertido -= costo_prop
            usdt_invertido  = max(usdt_invertido, 0.0)

            btc_en_posiciones -= btc_slot
            btc_balance       += btc_a_acumular
            usdt_balance      += usdt_neto
            positions_count   -= 1
            racha_idx         += 1
            trade_types.append("SELL")

    # ── Métricas finales ──────────────────────────────────────────────────────
    precio_final    = float(closes[-1])
    portfolio_final = usdt_balance + (btc_balance + btc_en_posiciones) * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100
    rachas          = calcular_rachas(trade_types)

    return {
        "pnl_pct"              : round(pnl_pct,          4),
        "portfolio_final"      : round(portfolio_final,   4),
        "usdt_balance_final"   : round(usdt_balance,      4),
        "btc_balance_final"    : round(btc_balance,       8),
        "btc_en_posiciones"    : round(btc_en_posiciones, 8),
        "total_trades"         : len(trade_types),
        "total_compras"        : trade_types.count("BUY"),
        "total_ventas"         : trade_types.count("SELL"),
        "positions_count_final": positions_count,
        **rachas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame) -> pd.DataFrame:

    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)

    combos       = list(product(RSI_LENGTH_RANGE, N_RANGE))
    total_combos = len(combos)

    print(f"\n{'═'*64}")
    print(f"  GRID SEARCH — {total_combos} combinaciones")
    print(f"  RSI_LENGTH    : {min(RSI_LENGTH_RANGE)} → {max(RSI_LENGTH_RANGE)}")
    print(f"  N (ventana)   : {min(N_RANGE)} → {max(N_RANGE)}")
    print(f"  ATH_CAIDA_MAX : {ATH_CAIDA_MAXIMA}%   FACTOR_CAIDA  : {FACTOR_CAIDA}")
    print(f"  ATL_SUBIDA_MAX: {ATL_SUBIDA_MAXIMA}%  FACTOR_SUBIDA : {FACTOR_SUBIDA}")
    print(f"  STREAK        : {STREAK_COMPRAS}C / {STREAK_VENTAS}V   "
          f"R: {R_COMPRA}C / {R_VENTA}V")
    print(f"  USDT reserva  : {USDT_RESERVA_PCT}% (${USDT_RESERVA:.2f})")
    print(f"{'═'*64}\n")

    resultados = []
    t_inicio   = time.time()
    rsi_cache  = {}   # RSI_LENGTH → (rsi_low_arr, rsi_high_arr)

    for idx, (rsi_len, n) in enumerate(combos, 1):

        # Calcular RSI solo si no está cacheado para este rsi_len
        if rsi_len not in rsi_cache:
            rsi_l = calcular_rsi(df["low"],  rsi_len).values.astype(float)
            rsi_h = calcular_rsi(df["high"], rsi_len).values.astype(float)
            rsi_cache[rsi_len] = (rsi_l, rsi_h)
        else:
            rsi_l, rsi_h = rsi_cache[rsi_len]

        metricas           = ejecutar_backtest(lows, highs, closes, rsi_l, rsi_h, n)
        metricas["rsi_length"] = rsi_len
        metricas["N"]          = n
        resultados.append(metricas)

        if idx % 50 == 0 or idx == total_combos:
            elapsed  = time.time() - t_inicio
            eta      = elapsed / idx * (total_combos - idx)
            best_pnl = max(r["pnl_pct"] for r in resultados)
            print(f"  [{idx:>4}/{total_combos}]  "
                  f"elapsed: {elapsed:>6.1f}s  "
                  f"ETA: {eta:>5.1f}s  "
                  f"mejor PnL hasta ahora: {best_pnl:>+8.2f}%")

    print(f"\n✓ Grid search completado en {time.time() - t_inicio:.1f}s")

    df_res = pd.DataFrame(resultados)
    df_res = df_res.sort_values("pnl_pct", ascending=False).reset_index(drop=True)
    df_res.index += 1   # ranking desde 1

    cols_order = [
        "rsi_length", "N", "pnl_pct", "portfolio_final",
        "total_trades", "total_compras", "total_ventas",
        "max_racha_compras", "max_racha_ventas",
        "avg_racha_compras", "avg_racha_ventas",
        "positions_count_final",
        "usdt_balance_final", "btc_balance_final", "btc_en_posiciones",
    ]
    df_res = df_res[[c for c in cols_order if c in df_res.columns]]
    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados(df_res: pd.DataFrame):

    # ── CSV ───────────────────────────────────────────────────────────────────
    df_res.to_csv(OUT_CSV, index_label="rank")
    print(f"✓ CSV guardado    : {OUT_CSV}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "fecha_inicio"      : str(FECHA_INICIO),
            "fecha_fin"         : str(FECHA_FIN),
            "saldo_inicial"     : SALDO_USDT_INICIAL,
            "ath_caida_maxima"  : ATH_CAIDA_MAXIMA,
            "factor_caida"      : FACTOR_CAIDA,
            "atl_subida_maxima" : ATL_SUBIDA_MAXIMA,
            "factor_subida"     : FACTOR_SUBIDA,
            "streak_compras"    : STREAK_COMPRAS,
            "streak_ventas"     : STREAK_VENTAS,
            "r_compra"          : R_COMPRA,
            "r_venta"           : R_VENTA,
            "usdt_reserva_pct"  : USDT_RESERVA_PCT,
            "btc_pct_accumulate": BTC_PCT_TO_ACCUMULATE,
            "commission_pct"    : COMMISSION_PCT,
            "n_range"           : [min(N_RANGE), max(N_RANGE)],
            "rsi_length_range"  : [min(RSI_LENGTH_RANGE), max(RSI_LENGTH_RANGE)],
        },
        "resultados": df_res.reset_index().rename(
            columns={"index": "rank"}
        ).to_dict(orient="records"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ JSON guardado   : {OUT_JSON}")

    # ── Top 10 legible ────────────────────────────────────────────────────────
    lineas = []
    lineas.append("╔════════════════════════════════════════════════════════════════════╗")
    lineas.append("║   TOP 10 — DIVERGENCIA RSI · CAPITAL DINÁMICO ATH/ATL            ║")
    lineas.append("╠════════════════════════════════════════════════════════════════════╣")
    lineas.append(f"  Período          : {FECHA_INICIO}  →  {FECHA_FIN}")
    lineas.append(f"  Capital inicial  : ${SALDO_USDT_INICIAL:,}")
    lineas.append(f"  ATH_CAIDA_MAX    : {ATH_CAIDA_MAXIMA}%   FACTOR_CAIDA  : {FACTOR_CAIDA}")
    lineas.append(f"  ATL_SUBIDA_MAX   : {ATL_SUBIDA_MAXIMA}%  FACTOR_SUBIDA : {FACTOR_SUBIDA}")
    lineas.append(f"  STREAK           : {STREAK_COMPRAS}C / {STREAK_VENTAS}V   "
                  f"R: {R_COMPRA}C / {R_VENTA}V   "
                  f"Reserva: {USDT_RESERVA_PCT}% (${USDT_RESERVA:.0f})")
    lineas.append("")

    header = (f"{'#':>3}  {'RSI':>4}  {'N':>4}  {'PnL%':>8}  "
              f"{'Port.$':>9}  {'Trades':>6}  {'Buys':>5}  {'Sells':>5}  "
              f"{'MaxRchB':>7}  {'MaxRchS':>7}  {'AvgRchB':>7}  {'AvgRchS':>7}  "
              f"{'PosCount':>8}")
    lineas.append(header)
    lineas.append("─" * len(header))

    for rank, row in df_res.head(10).iterrows():
        lineas.append(
            f"{rank:>3}.  "
            f"{int(row['rsi_length']):>4}  "
            f"{int(row['N']):>4}  "
            f"{row['pnl_pct']:>+8.2f}%  "
            f"${row['portfolio_final']:>9,.2f}  "
            f"{int(row['total_trades']):>6}  "
            f"{int(row['total_compras']):>5}  "
            f"{int(row['total_ventas']):>5}  "
            f"{int(row['max_racha_compras']):>7}  "
            f"{int(row['max_racha_ventas']):>7}  "
            f"{row['avg_racha_compras']:>7.1f}  "
            f"{row['avg_racha_ventas']:>7.1f}  "
            f"{int(row['positions_count_final']):>+8}"
        )

    lineas.append("")
    lineas.append("Columnas: RSI=período RSI | N=ventana local | PnL%=rentabilidad total")
    lineas.append("  MaxRchB/S=máx racha BUY/SELL | AvgRchB/S=promedio de rachas")
    lineas.append("  PosCount=sesgo final al cierre (+ = más compras | - = más ventas)")
    lineas.append("╚════════════════════════════════════════════════════════════════════╝")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(f"✓ Top 10 guardado : {OUT_TXT}")
    print()
    print("\n".join(lineas))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   OPTIMIZADOR — DIVERGENCIA RSI · CAPITAL DINÁMICO ATH/ATL         ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    print("Cargando datos de la DB...")
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos en el rango especificado. Revisar config.py")
        return

    df_resultados = optimizar(df)

    print("\nGuardando resultados...")
    guardar_resultados(df_resultados)

    print("\n✓ Proceso completado.")


if __name__ == "__main__":
    main()