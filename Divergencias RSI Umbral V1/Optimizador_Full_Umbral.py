"""
Optimizador Completo — Divergencia RSI Umbral
══════════════════════════════════════════════
BTC/USDT · Velas Horarias · Grid Search Exhaustivo
Estrategia: Estrategia_Divergencia_RSI_Umbral

Variables optimizadas (10):
  Continuas  → especificar inicio / fin / paso
    RSI_LENGTH       : período del RSI
    N                : ventana de búsqueda del extremo local
    FLOOR_PCT        : piso del ciclo como % del ATH
    FACTOR_CAIDA     : curvatura del gradiente de compra
    FACTOR_SUBIDA    : curvatura del gradiente de venta
    RSI_BUY_TRIGGER  : RSI máximo de la vela âncla de compra (≤ trigger → señal válida)
    RSI_SELL_TRIGGER : RSI mínimo de la vela âncla de venta  (≥ trigger → señal válida)

  Booleanas  → siempre ambos valores
    GUARDIA_COMPRA        : bloqueo compras sobre PP
    GUARDIA_PRECIO_COMPRA : bloqueo compras sobre mínimo comprado
    GUARDIA_PRECIO_VENTA  : bloqueo ventas bajo máximo vendido

Variables fijas (desde config):
    USDT_RESERVA_PCT, BTC_PCT_TO_ACCUMULATE, COMMISSION_PCT,
    FECHA_INICIO, FECHA_FIN, SALDO_USDT_INICIAL

Métricas de ranking:
  1. PnL%           — retorno sobre capital, sin sesgo de tendencia
  2. BTC acumulado  — valor USD del BTC en posiciones al cierre
  3. Equilibrio     — media geométrica de las normas [0,1] de PnL% y BTC$

Métricas adicionales por combinación:
  tasa_aprobacion_compra — % divergencias alcistas que pasaron el filtro de umbral
  tasa_aprobacion_venta  — % divergencias bajistas que pasaron el filtro de umbral

Salidas (7 archivos):
  optimizacion_umbral.csv
  optimizacion_umbral_top.json
  optimizacion_umbral_ranking_pnl.png
  optimizacion_umbral_ranking_btc.png
  optimizacion_umbral_ranking_equilibrio.png
  optimizacion_umbral_guardias.png
  optimizacion_umbral_analisis.png
"""

import sqlite3
import json
import math
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from itertools import product

# ── Importar config ───────────────────────────────────────────────────────────
try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL,
        USDT_RESERVA_PCT,
        BTC_PCT_TO_ACCUMULATE,
        COMMISSION_PCT,
    )
    print("✓ config.py cargado")
except ImportError:
    print("⚠ config.py no encontrado — usando valores por defecto")
    DB_PATH               = r"btc_hourly.db"
    FECHA_INICIO          = '2021-11-10'
    FECHA_FIN             = '2022-11-22'
    SALDO_USDT_INICIAL    = 1000
    USDT_RESERVA_PCT      = 0
    BTC_PCT_TO_ACCUMULATE = 0
    COMMISSION_PCT        = 0.1

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100


# ══════════════════════════════════════════════════════════════════════════════
# ESPACIO DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════
# Modificá inicio / fin / paso para ajustar granularidad.
# paso=10 → 777k combos (~5 min)  |  paso=5 → 3.1M combos (~20 min)

# ── Variables continuas clásicas ─────────────────────────────────────────────
RSI_LENGTH_INICIO    = 5  ;  RSI_LENGTH_FIN    = 17  ;  RSI_LENGTH_PASO    = 3
N_INICIO             = 12  ;  N_FIN             = 36  ;  N_PASO             = 3
FLOOR_PCT_INICIO     = 10  ;  FLOOR_PCT_FIN     = 25  ;  FLOOR_PCT_PASO     = 5
FACTOR_CAIDA_INICIO  = 1.0 ;  FACTOR_CAIDA_FIN  = 4.0 ;  FACTOR_CAIDA_PASO  = 1.0
FACTOR_SUBIDA_INICIO = 0.5 ;  FACTOR_SUBIDA_FIN = 3.5 ;  FACTOR_SUBIDA_PASO = 1.0

# ── Triggers de umbral (nuevos) ───────────────────────────────────────────────
# RSI_BUY_TRIGGER  : âncla de compra debe tener RSI ≤ este valor
#   Rango típico: 20–45  (20 = muy restrictivo · 45 = permisivo)
RSI_BUY_TRIGGER_INICIO  = 15  ;  RSI_BUY_TRIGGER_FIN  = 35  ;  RSI_BUY_TRIGGER_PASO  = 2.5

# RSI_SELL_TRIGGER : âncla de venta debe tener RSI ≥ este valor
#   Rango típico: 55–80  (80 = muy restrictivo · 55 = permisivo)
RSI_SELL_TRIGGER_INICIO = 65  ;  RSI_SELL_TRIGGER_FIN = 75  ;  RSI_SELL_TRIGGER_PASO = 2.5

# ── Archivos de salida ────────────────────────────────────────────────────────
OUT_CSV        = "optimizacion_umbral.csv"
OUT_JSON       = "optimizacion_umbral_top.json"
OUT_TABLA_PNL  = "optimizacion_umbral_ranking_pnl.png"
OUT_TABLA_BTC  = "optimizacion_umbral_ranking_btc.png"
OUT_TABLA_EQ   = "optimizacion_umbral_ranking_equilibrio.png"
OUT_GUARDIAS   = "optimizacion_umbral_guardias.png"
OUT_ANALISIS   = "optimizacion_umbral_analisis.png"
TOP_N          = 25


# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE RANGOS
# ══════════════════════════════════════════════════════════════════════════════

def _rango(inicio, fin, paso, entero=False):
    vals, v = [], inicio
    while v <= fin + paso * 1e-9:
        vals.append(int(round(v)) if entero else round(v, 10))
        v += paso
    return vals


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datos() -> pd.DataFrame:
    conn  = sqlite3.connect(DB_PATH)
    query = f"SELECT timestamp, high, low, close FROM {DB_TABLE} ORDER BY timestamp ASC"
    df    = pd.read_sql(query, conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if FECHA_INICIO:
        df = df[df["datetime"] >= pd.to_datetime(FECHA_INICIO)]
    if FECHA_FIN:
        df = df[df["datetime"] <= pd.to_datetime(FECHA_FIN)]
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# RSI
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(series: pd.Series, length: int) -> np.ndarray:
    """RSI clásico de Wilder (EWM), idéntico al de TradingView."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values.astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENTES
# ══════════════════════════════════════════════════════════════════════════════

def _pct_compra(precio_low: float, ath: float,
                floor_pct: float, factor_caida: float) -> float:
    if ath <= 0 or floor_pct <= 0:
        return 0.0
    log_rango = math.log(100.0 / floor_pct)
    if log_rango <= 0:
        return 0.0
    pos = math.log(ath / precio_low) / log_rango
    return (max(0.0, min(1.0, pos)) ** factor_caida) * 100.0


def _pct_venta(precio_high: float, ath: float,
               precio_promedio: float, factor_subida: float) -> float:
    if ath <= 0 or precio_promedio <= 0 or precio_high <= precio_promedio:
        return 0.0
    log_amp = math.log(ath / precio_promedio)
    if log_amp <= 0:
        return 0.0
    pos = math.log(precio_high / precio_promedio) / log_amp
    return (max(0.0, min(1.0, pos)) ** factor_subida) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEO DEL BACKTEST  (optimizado para velocidad)
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_backtest(
    lows:                np.ndarray,
    highs:               np.ndarray,
    closes:              np.ndarray,
    rsi_low:             np.ndarray,
    rsi_high:            np.ndarray,
    N:                   int,
    floor_pct:           float,
    factor_caida:        float,
    factor_subida:       float,
    rsi_buy_trigger:     float,
    rsi_sell_trigger:    float,
    guardia_compra:      bool,
    guardia_prec_compra: bool,
    guardia_prec_venta:  bool,
) -> dict:

    n_velas           = len(lows)
    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    precio_min_comp   = math.inf
    precio_max_venta  = 0.0
    ath               = float(highs[0])
    compras = ventas  = 0

    # Contadores de divergencias para tasa de aprobación del filtro umbral
    div_compra_det = div_compra_apr = 0
    div_venta_det  = div_venta_apr  = 0

    port_max = float(SALDO_USDT_INICIAL)
    dd_max   = 0.0

    for i in range(N, n_velas):

        if highs[i] > ath:
            ath = float(highs[i])

        if np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]):
            continue

        wl = lows[i - N : i]
        wh = highs[i - N : i]
        pp = usdt_invertido / btc_en_posiciones if btc_en_posiciones > 0 else 0.0

        port_now = usdt_balance + btc_en_posiciones * closes[i]
        if port_now > port_max:
            port_max = port_now
        elif port_max > 0:
            dd = (port_max - port_now) / port_max * 100
            if dd > dd_max:
                dd_max = dd

        # ── SEÑAL DE COMPRA ───────────────────────────────────────────────────
        # Condiciones:
        #   1. lows[i] < mínimo de la ventana          → nuevo mínimo de precio
        #   2. rsi_low[i] > rsi_low[idx_min]           → divergencia alcista
        #   3. rsi_low[idx_min] <= rsi_buy_trigger      → âncla en sobreventa
        señal_compra = False
        if lows[i] < wl.min():
            idx_min = i - N + int(wl.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                div_compra_det += 1
                if rsi_low[idx_min] <= rsi_buy_trigger:
                    div_compra_apr += 1
                    señal_compra = True
                # else: divergencia rechazada por umbral — no genera orden

        # ── SEÑAL DE VENTA ────────────────────────────────────────────────────
        # Condiciones:
        #   1. highs[i] > máximo de la ventana         → nuevo máximo de precio
        #   2. rsi_high[i] < rsi_high[idx_max]         → divergencia bajista
        #   3. rsi_high[idx_max] >= rsi_sell_trigger    → âncla en sobrecompra
        señal_venta = False
        if not señal_compra and highs[i] > wh.max():
            idx_max = i - N + int(wh.argmax())
            if rsi_high[i] < rsi_high[idx_max]:
                div_venta_det += 1
                if rsi_high[idx_max] >= rsi_sell_trigger:
                    div_venta_apr += 1
                    señal_venta = True
                # else: rechazada por umbral

        # ── EJECUTAR COMPRA ───────────────────────────────────────────────────
        if señal_compra:
            ud = usdt_balance - USDT_RESERVA
            if ud <= 0:
                continue
            if guardia_compra and btc_en_posiciones > 0 and lows[i] >= pp:
                continue
            if guardia_prec_compra and precio_min_comp < math.inf and lows[i] >= precio_min_comp:
                continue

            pct = _pct_compra(lows[i], ath, floor_pct, factor_caida)
            ua  = ud * pct / 100.0
            if ua <= 0:
                continue

            com = ua * (COMMISSION_PCT / 100)
            ba  = (ua - com) / lows[i]
            usdt_balance      -= ua
            btc_en_posiciones += ba
            usdt_invertido    += ua
            compras           += 1
            if lows[i] < precio_min_comp:
                precio_min_comp = lows[i]

        # ── EJECUTAR VENTA ────────────────────────────────────────────────────
        elif señal_venta and btc_en_posiciones > 0:
            pct      = _pct_venta(highs[i], ath, pp, factor_subida)
            btc_slot = btc_en_posiciones * pct / 100.0
            if btc_slot <= 0:
                continue
            if guardia_prec_venta and precio_max_venta > 0 and highs[i] <= precio_max_venta:
                continue

            btc_acum   = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_vender = btc_slot - btc_acum
            ub         = btc_vender * highs[i]
            com        = ub * (COMMISSION_PCT / 100)
            un         = ub - com
            cp         = usdt_invertido * (btc_slot / btc_en_posiciones)
            usdt_invertido    = max(usdt_invertido - cp, 0.0)
            btc_en_posiciones -= btc_slot
            usdt_balance      += un
            ventas            += 1
            if highs[i] > precio_max_venta:
                precio_max_venta = highs[i]

    # ── Métricas finales ──────────────────────────────────────────────────────
    precio_final    = float(closes[-1])
    portfolio_final = usdt_balance + btc_en_posiciones * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100
    total_trades    = compras + ventas
    pnl_por_trade   = pnl_pct / total_trades if total_trades > 0 else 0.0
    pp_fin          = usdt_invertido / btc_en_posiciones if btc_en_posiciones > 0 else 0.0

    tasa_apr_compra = (div_compra_apr / div_compra_det * 100) if div_compra_det > 0 else 0.0
    tasa_apr_venta  = (div_venta_apr  / div_venta_det  * 100) if div_venta_det  > 0 else 0.0

    return {
        "pnl_pct"            : round(pnl_pct,          4),
        "portfolio_final"    : round(portfolio_final,   2),
        "usdt_final"         : round(usdt_balance,      2),
        "btc_posiciones"     : round(btc_en_posiciones, 8),
        "precio_prom_fin"    : round(pp_fin,            2),
        "total_trades"       : total_trades,
        "total_compras"      : compras,
        "total_ventas"       : ventas,
        "positions_count"    : compras - ventas,
        "pnl_por_trade"      : round(pnl_por_trade,     4),
        "max_drawdown"       : round(dd_max,            2),
        # Estadísticas del filtro umbral
        "div_compra_det"     : div_compra_det,
        "div_compra_apr"     : div_compra_apr,
        "tasa_apr_compra"    : round(tasa_apr_compra,   1),
        "div_venta_det"      : div_venta_det,
        "div_venta_apr"      : div_venta_apr,
        "tasa_apr_venta"     : round(tasa_apr_venta,    1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame) -> pd.DataFrame:

    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)

    rsi_lengths    = _rango(RSI_LENGTH_INICIO,       RSI_LENGTH_FIN,       RSI_LENGTH_PASO,       entero=True)
    ns             = _rango(N_INICIO,                N_FIN,                N_PASO,                entero=True)
    floor_pcts     = _rango(FLOOR_PCT_INICIO,        FLOOR_PCT_FIN,        FLOOR_PCT_PASO)
    factor_caidas  = _rango(FACTOR_CAIDA_INICIO,     FACTOR_CAIDA_FIN,     FACTOR_CAIDA_PASO)
    factor_subidas = _rango(FACTOR_SUBIDA_INICIO,    FACTOR_SUBIDA_FIN,    FACTOR_SUBIDA_PASO)
    buy_triggers   = _rango(RSI_BUY_TRIGGER_INICIO,  RSI_BUY_TRIGGER_FIN,  RSI_BUY_TRIGGER_PASO,  entero=True)
    sell_triggers  = _rango(RSI_SELL_TRIGGER_INICIO, RSI_SELL_TRIGGER_FIN, RSI_SELL_TRIGGER_PASO, entero=True)
    guardias_bool  = [True, False]

    total_combos = (len(rsi_lengths) * len(ns) * len(floor_pcts) *
                    len(factor_caidas) * len(factor_subidas) *
                    len(buy_triggers) * len(sell_triggers) * 8)

    print(f"\n{'═'*70}")
    print(f"  GRID SEARCH EXHAUSTIVO — {total_combos:,} combinaciones")
    print(f"{'═'*70}")
    print(f"  RSI_LENGTH       : {rsi_lengths[0]} → {rsi_lengths[-1]}  "
          f"(paso {RSI_LENGTH_PASO}, {len(rsi_lengths)} vals)")
    print(f"  N                : {ns[0]} → {ns[-1]}  "
          f"(paso {N_PASO}, {len(ns)} vals)")
    print(f"  FLOOR_PCT        : {floor_pcts[0]} → {floor_pcts[-1]}  "
          f"(paso {FLOOR_PCT_PASO}, {len(floor_pcts)} vals)")
    print(f"  FACTOR_CAIDA     : {factor_caidas[0]} → {factor_caidas[-1]}  "
          f"(paso {FACTOR_CAIDA_PASO}, {len(factor_caidas)} vals)")
    print(f"  FACTOR_SUBIDA    : {factor_subidas[0]} → {factor_subidas[-1]}  "
          f"(paso {FACTOR_SUBIDA_PASO}, {len(factor_subidas)} vals)")
    print(f"  RSI_BUY_TRIGGER  : {buy_triggers[0]} → {buy_triggers[-1]}  "
          f"(paso {RSI_BUY_TRIGGER_PASO}, {len(buy_triggers)} vals)  ← âncla compra ≤ trigger")
    print(f"  RSI_SELL_TRIGGER : {sell_triggers[0]} → {sell_triggers[-1]}  "
          f"(paso {RSI_SELL_TRIGGER_PASO}, {len(sell_triggers)} vals)  ← âncla venta ≥ trigger")
    print(f"  Guardias         : 2³ = 8 combinaciones booleanas")
    print(f"  ── Fijos desde config ──────────────────────────────────────────────")
    print(f"  Período          : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  USDT reserva     : {USDT_RESERVA_PCT}%  "
          f"BTC acum: {BTC_PCT_TO_ACCUMULATE}%  Comisión: {COMMISSION_PCT}%")
    print(f"{'═'*70}\n")

    # Caché RSI — solo depende de rsi_length (los triggers no afectan el cálculo del RSI)
    print("  Pre-calculando RSI...")
    rsi_cache = {}
    for rsi_len in rsi_lengths:
        rsi_cache[rsi_len] = (
            calcular_rsi(df["low"],  rsi_len),
            calcular_rsi(df["high"], rsi_len),
        )
    print(f"  ✓ {len(rsi_cache)} pares RSI calculados\n")

    resultados = []
    t0  = time.time()
    idx = 0

    for rsi_len, n, fp, fc, fs, bt, st, gc, gpc, gpv in product(
        rsi_lengths, ns, floor_pcts, factor_caidas, factor_subidas,
        buy_triggers, sell_triggers,
        guardias_bool, guardias_bool, guardias_bool,
    ):
        idx += 1
        rsi_l, rsi_h = rsi_cache[rsi_len]

        m = ejecutar_backtest(
            lows, highs, closes, rsi_l, rsi_h,
            n, fp, fc, fs, bt, st, gc, gpc, gpv,
        )
        m.update({
            "rsi_length"          : rsi_len,
            "N"                   : n,
            "floor_pct"           : fp,
            "factor_caida"        : fc,
            "factor_subida"       : fs,
            "rsi_buy_trigger"     : bt,
            "rsi_sell_trigger"    : st,
            "guardia_compra"      : gc,
            "guardia_prec_compra" : gpc,
            "guardia_prec_venta"  : gpv,
        })
        resultados.append(m)

        if idx % 20000 == 0 or idx == total_combos:
            elapsed  = time.time() - t0
            eta      = elapsed / idx * (total_combos - idx)
            best     = max(r["pnl_pct"] for r in resultados)
            pct_done = idx / total_combos * 100
            print(f"  [{idx:>8}/{total_combos:,}] {pct_done:5.1f}%  "
                  f"{elapsed:>7.1f}s  ETA:{eta:>6.1f}s  "
                  f"mejor PnL: {best:>+8.2f}%")

    print(f"\n✓ Completado en {time.time() - t0:.1f}s")

    df_res = pd.DataFrame(resultados)
    col_order = [
        "rsi_length", "N", "floor_pct", "factor_caida", "factor_subida",
        "rsi_buy_trigger", "rsi_sell_trigger",
        "guardia_compra", "guardia_prec_compra", "guardia_prec_venta",
        "pnl_pct", "portfolio_final", "usdt_final", "btc_posiciones",
        "precio_prom_fin", "total_trades", "total_compras", "total_ventas",
        "positions_count", "pnl_por_trade", "max_drawdown",
        "div_compra_det", "div_compra_apr", "tasa_apr_compra",
        "div_venta_det",  "div_venta_apr",  "tasa_apr_venta",
    ]
    df_res = df_res[col_order]

    # Verificación: 8 combinaciones de guardias con igual presencia
    n_g = df_res.groupby(["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]).ngroups
    print(f"\n  ✓ Verificación de guardias: {n_g}/8 combinaciones probadas")
    for (gc, gpc, gpv), grp in df_res.groupby(
        ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    ):
        print(f"    GC={gc} GPC={gpc} GPV={gpv}: {len(grp):,} combos "
              f"| mejor PnL: {grp['pnl_pct'].max():+.2f}%")

    # Verificación triggers
    print(f"\n  ✓ Verificación de triggers:")
    print(f"    RSI_BUY_TRIGGER  probados: {sorted(df_res['rsi_buy_trigger'].unique())}")
    print(f"    RSI_SELL_TRIGGER probados: {sorted(df_res['rsi_sell_trigger'].unique())}")

    df_res = df_res.sort_values("pnl_pct", ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _agregar_scores(df_res: pd.DataFrame, precio_final: float) -> pd.DataFrame:
    """
    btc_value       : btc_posiciones × precio_final
    equilibrio_score: √(pnl_norm × btc_norm)  — media geométrica de normas [0,1]
    """
    df = df_res.copy()
    df["btc_value"] = df["btc_posiciones"] * precio_final

    pnl_min, pnl_max = df["pnl_pct"].min(), df["pnl_pct"].max()
    df["pnl_norm"] = ((df["pnl_pct"] - pnl_min) / max(pnl_max - pnl_min, 1e-9)).clip(0, 1)

    btc_min, btc_max = df["btc_value"].min(), df["btc_value"].max()
    df["btc_norm"] = ((df["btc_value"] - btc_min) / max(btc_max - btc_min, 1e-9)).clip(0, 1)

    df["equilibrio_score"] = (df["pnl_norm"] * df["btc_norm"]).apply(math.sqrt)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CSV + JSON
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados(df_res: pd.DataFrame):
    df_res.to_csv(OUT_CSV, index_label="rank_pnl")
    print(f"  ✓ CSV: {OUT_CSV}  ({len(df_res):,} filas)")

    def _top(df, col):
        return (df.sort_values(col, ascending=False)
                  .head(TOP_N).reset_index(drop=True)
                  .reset_index().rename(columns={"index": "rank"})
                  .assign(rank=lambda d: d["rank"] + 1)
                  .to_dict(orient="records"))

    # Resumen por combinación de guardias
    guard_stats = []
    for (gc, gpc, gpv), grp in df_res.groupby(
        ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    ):
        guard_stats.append({
            "guardia_compra"      : gc,
            "guardia_prec_compra" : gpc,
            "guardia_prec_venta"  : gpv,
            "n_combos"            : len(grp),
            "mejor_pnl_pct"       : round(grp["pnl_pct"].max(),          4),
            "mediana_pnl_pct"     : round(grp["pnl_pct"].median(),        4),
            "mejor_btc_value"     : round(grp["btc_value"].max(),         2),
            "mejor_eq_score"      : round(grp["equilibrio_score"].max(),  6),
        })

    payload = {
        "meta": {
            "estrategia"                  : "Divergencia RSI Umbral",
            "fecha_inicio"                : FECHA_INICIO,
            "fecha_fin"                   : FECHA_FIN,
            "saldo_inicial"               : SALDO_USDT_INICIAL,
            "usdt_reserva_pct"            : USDT_RESERVA_PCT,
            "btc_pct_to_accumulate"       : BTC_PCT_TO_ACCUMULATE,
            "commission_pct"              : COMMISSION_PCT,
            "rsi_range"                   : [RSI_LENGTH_INICIO,       RSI_LENGTH_FIN,       RSI_LENGTH_PASO],
            "n_range"                     : [N_INICIO,                N_FIN,                N_PASO],
            "floor_pct_range"             : [FLOOR_PCT_INICIO,        FLOOR_PCT_FIN,        FLOOR_PCT_PASO],
            "factor_caida_range"          : [FACTOR_CAIDA_INICIO,     FACTOR_CAIDA_FIN,     FACTOR_CAIDA_PASO],
            "factor_subida_range"         : [FACTOR_SUBIDA_INICIO,    FACTOR_SUBIDA_FIN,    FACTOR_SUBIDA_PASO],
            "rsi_buy_trigger_range"       : [RSI_BUY_TRIGGER_INICIO,  RSI_BUY_TRIGGER_FIN,  RSI_BUY_TRIGGER_PASO],
            "rsi_sell_trigger_range"      : [RSI_SELL_TRIGGER_INICIO, RSI_SELL_TRIGGER_FIN, RSI_SELL_TRIGGER_PASO],
            "total_combos"                : len(df_res),
            "combos_por_config_guardias"  : len(df_res) // 8,
            "generado"                    : pd.Timestamp.now().isoformat(),
        },
        "guardias_analisis"  : guard_stats,
        "ranking_pnl"        : _top(df_res, "pnl_pct"),
        "ranking_btc"        : _top(df_res, "btc_value"),
        "ranking_equilibrio" : _top(df_res, "equilibrio_score"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  ✓ JSON: {OUT_JSON}  (3 rankings × top {TOP_N})")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN 1 — 3 TABLAS DE RANKING
# ══════════════════════════════════════════════════════════════════════════════

# Columnas de la tabla visual
# Índices:  0    1     2    3       4        5       6        7        8      9      10
_COLS = ["#", "RSI", "N", "BUY_T","SELL_T","FLOOR%","F_CAI","F_SUB","G_PP","G_P_C","G_P_V",
#          11       12           13       14      15      16       17         18          19
         "PnL %","Portfolio $","BTC $","Trades","C/V","Div.C%","Div.V%","Equilibrio","PnL/Trade",
#          20
         "MaxDD%"]

# (sort_col, highlight_col_idx, cmap, titulo, subtitulo, archivo)
_RANKINGS = [
    ("pnl_pct",          11, plt.cm.RdYlGn,
     "Ranking 1 — Máximo PnL%",
     "Prioriza el retorno total · Sin sesgo de tendencia",
     OUT_TABLA_PNL),
    ("btc_value",        13, plt.cm.YlOrRd,
     "Ranking 2 — Máxima Acumulación BTC",
     "Prioriza el valor USD del BTC en posiciones al cierre",
     OUT_TABLA_BTC),
    ("equilibrio_score", 18, plt.cm.PuBuGn,
     "Ranking 3 — Mejor Equilibrio PnL% × BTC",
     "Media geométrica de normas [0,1]  ·  Penaliza extremos en una sola dimensión",
     OUT_TABLA_EQ),
]


def _fig_tabla(df_res: pd.DataFrame, sort_col: str, highlight_col: int,
               cmap, titulo: str, subtitulo: str, filename: str):
    top = (df_res.sort_values(sort_col, ascending=False)
                 .head(TOP_N).reset_index(drop=True))

    def bs(v): return "Si" if v else "No"

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append([
            str(rank),
            str(r.rsi_length), str(r.N),
            str(int(r.rsi_buy_trigger)), str(int(r.rsi_sell_trigger)),
            f"{r.floor_pct:.0f}%",
            f"{r.factor_caida:.1f}", f"{r.factor_subida:.1f}",
            bs(r.guardia_compra), bs(r.guardia_prec_compra), bs(r.guardia_prec_venta),
            f"{r.pnl_pct:+.2f}%",
            f"${r.portfolio_final:,.2f}",
            f"${r.btc_value:,.2f}",
            str(int(r.total_trades)),
            f"{int(r.total_compras)}/{int(r.total_ventas)}",
            f"{r.tasa_apr_compra:.0f}%",
            f"{r.tasa_apr_venta:.0f}%",
            f"{r.equilibrio_score:.4f}",
            f"{r.pnl_por_trade:+.3f}%",
            f"{r.max_drawdown:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(26, TOP_N * 0.43 + 3.2))
    fig.patch.set_facecolor("#f4f6fa")
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=_COLS, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.52)

    # Cabecera
    for j in range(len(_COLS)):
        c = table[0, j]
        c.set_facecolor("#1a2540")
        c.set_text_props(color="white", fontweight="bold")

    # Cabeceras de las nuevas columnas de trigger en azul oscuro
    for j in [3, 4]:
        table[0, j].set_facecolor("#1a4a80")

    # Columna de ranking activo en naranja
    table[0, highlight_col].set_facecolor("#e67e22")

    # Cabeceras Div.C% y Div.V% en verde oscuro
    for j in [16, 17]:
        table[0, j].set_facecolor("#1e6b3c")

    # Filas alternas
    for i in range(1, len(rows) + 1):
        bg = "#eef2f9" if i % 2 == 0 else "#ffffff"
        for j in range(len(_COLS)):
            table[i, j].set_facecolor(bg)

    # Top 3 medallas
    for i, color in enumerate(["#ffd700", "#d8d8d8", "#cd7f32"][:min(3, len(rows))], 1):
        for j in range(len(_COLS)):
            table[i, j].set_facecolor(color)
            table[i, j].set_text_props(fontweight="bold")

    # Gradiente de color en la columna de ranking activo
    raw_vals = [float(rows[i][highlight_col]
                      .replace("$", "").replace(",", "")
                      .replace("%", "").replace("+", ""))
                for i in range(len(rows))]
    vmin, vmax = min(raw_vals), max(raw_vals)
    span = vmax - vmin if vmax > vmin else 1.0
    for i, v in enumerate(raw_vals, 1):
        norm  = (v - vmin) / span
        color = mcolors.to_hex(cmap(0.25 + 0.75 * norm))
        table[i, highlight_col].set_facecolor(color)

    ax.set_title(
        f"{titulo}\n{subtitulo}\n"
        f"Período: {FECHA_INICIO} → {FECHA_FIN}  |  "
        f"Total combinaciones: {len(df_res):,}  |  "
        f"G_PP=Guardia PP  ·  G_P_C=Guardia precio compra  ·  G_P_V=Guardia precio venta  |  "
        f"Div.C%=tasa aprobación divergencias compra  ·  Div.V%=venta",
        fontsize=9.5, fontweight="bold", color="#1a2540", pad=13,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ {titulo}: {filename}")


def fig_tres_tablas(df_res: pd.DataFrame):
    for sort_col, hl_col, cmap, titulo, subtitulo, filename in _RANKINGS:
        _fig_tabla(df_res, sort_col, hl_col, cmap, titulo, subtitulo, filename)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN 2 — ANÁLISIS DE GUARDIAS
# ══════════════════════════════════════════════════════════════════════════════

def fig_analisis_guardias(df_res: pd.DataFrame):
    GCOLS = ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]

    rows_tabla = []
    for (gc, gpc, gpv), grp in df_res.groupby(GCOLS):
        bp = grp.loc[grp["pnl_pct"].idxmax()]
        bb = grp.loc[grp["btc_value"].idxmax()]
        be = grp.loc[grp["equilibrio_score"].idxmax()]

        def params(r):
            return (f"RSI={int(r.rsi_length)} N={int(r.N)} "
                    f"BT={int(r.rsi_buy_trigger)} ST={int(r.rsi_sell_trigger)} "
                    f"FL={r.floor_pct:.0f}% FC={r.factor_caida:.1f} FS={r.factor_subida:.1f}")

        rows_tabla.append({
            "gc": gc, "gpc": gpc, "gpv": gpv,
            "n_combos"    : len(grp),
            "pnl_max"     : bp["pnl_pct"],     "pnl_port"   : bp["portfolio_final"],
            "pnl_params"  : params(bp),
            "btc_max_val" : bb["btc_value"],    "btc_btc"    : bb["btc_posiciones"],
            "btc_params"  : params(bb),
            "eq_score"    : be["equilibrio_score"],
            "eq_pnl"      : be["pnl_pct"],      "eq_btc"     : be["btc_value"],
            "eq_params"   : params(be),
            "pnl_median"  : grp["pnl_pct"].median(),
            "pnl_positive": (grp["pnl_pct"] > 0).sum(),
        })

    df_g = pd.DataFrame(rows_tabla).sort_values("pnl_max", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(28, 17))
    fig.patch.set_facecolor("#f4f6fa")
    gs  = GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[2.8, 1])

    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")

    def bs(v): return "Si" if v else "No"
    def lbl(gc, gpc, gpv): return f"GC={bs(gc)} GPC={bs(gpc)} GPV={bs(gpv)}"

    cols_t = [
        "Guardias\n(GC/GPC/GPV)", "Combos",
        "──── Mejor PnL% ────────────────────────────",
        "PnL%", "Portfolio $", "Params",
        "──── Mejor BTC ─────────────────────────────",
        "BTC $", "BTC ₿", "Params",
        "──── Mejor Equilibrio ───────────────────────",
        "Eq.Score", "PnL%", "BTC $", "Params",
        "Mediana\nPnL%", ">0%",
    ]

    rows_t = []
    for _, r in df_g.iterrows():
        rows_t.append([
            lbl(r.gc, r.gpc, r.gpv), f"{int(r.n_combos):,}",
            "",  f"{r.pnl_max:+.2f}%", f"${r.pnl_port:,.2f}", r.pnl_params,
            "",  f"${r.btc_max_val:,.2f}", f"{r.btc_btc:.6f} ₿", r.btc_params,
            "",  f"{r.eq_score:.4f}", f"{r.eq_pnl:+.2f}%", f"${r.eq_btc:,.2f}", r.eq_params,
            f"{r.pnl_median:+.2f}%", f"{int(r.pnl_positive):,}",
        ])

    table = ax_t.table(cellText=rows_t, colLabels=cols_t, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.7)

    for j in range(len(cols_t)):
        c = table[0, j]
        c.set_facecolor("#1a2540")
        c.set_text_props(color="white", fontweight="bold")
    for ci, color in zip([2, 6, 10], ["#1e6b3c", "#7b3f00", "#1a3a6b"]):
        table[0, ci].set_facecolor(color)

    for i in range(1, len(rows_t) + 1):
        bg = "#eef2f9" if i % 2 == 0 else "#ffffff"
        for j in range(len(cols_t)):
            table[i, j].set_facecolor(bg)
    for i, color in enumerate(["#ffd700", "#d8d8d8", "#cd7f32"][:min(3, len(rows_t))], 1):
        table[i, 3].set_facecolor(color)
        table[i, 3].set_text_props(fontweight="bold")
    table[1, 7].set_facecolor("#ffe4b5")
    table[1, 11].set_facecolor("#d4edda")

    for i, (_, r) in enumerate(df_g.iterrows(), 1):
        n_true    = sum([r.gc, r.gpc, r.gpv])
        intensity = n_true / 3
        color     = mcolors.to_hex(plt.cm.RdYlGn(0.3 + 0.7 * intensity))
        table[i, 0].set_facecolor(color)
        table[i, 0].set_text_props(fontweight="bold")

    ax_t.set_title(
        f"Análisis de Combinaciones de Guardias — {FECHA_INICIO} → {FECHA_FIN}\n"
        f"Cada fila = mejor resultado de las {len(df_res) // 8:,} combinaciones "
        f"con esa config de guardias  (Total: {len(df_res):,}  |  2³ = 8 configs)\n"
        f"Los params incluyen los triggers RSI — GC=GUARDIA_COMPRA · GPC=G_PRECIO_COMPRA · GPV=G_PRECIO_VENTA",
        fontsize=10, fontweight="bold", color="#1a2540", pad=12,
    )

    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor("#ffffff")
    x      = np.arange(8)
    labels = [lbl(r.gc, r.gpc, r.gpv) for _, r in df_g.iterrows()]
    pnl_m  = [r.pnl_max for _, r in df_g.iterrows()]
    btc_m  = [r.btc_max_val for _, r in df_g.iterrows()]
    btc_sc = [b / max(btc_m) * max(abs(p) for p in pnl_m) for b in btc_m]
    w = 0.35
    b1 = ax_b.bar(x - w/2, pnl_m,  w,
                  color=[plt.cm.RdYlGn(0.3 + 0.7 * (p - min(pnl_m)) /
                          max(max(pnl_m) - min(pnl_m), 0.001)) for p in pnl_m],
                  edgecolor="white", alpha=0.9, label="Mejor PnL%")
    ax_b.bar(x + w/2, btc_sc, w, color="#3498db", alpha=0.65,
             edgecolor="white", label="Mejor BTC$ (normalizado)")
    ax_b.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    ax_b.set_ylabel("Mejor PnL%", fontsize=9)
    ax_b.set_title("Comparativa entre las 8 combinaciones de guardias", fontsize=9)
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(b1, pnl_m):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + (0.3 if val >= 0 else -1.2),
                  f"{val:+.1f}%", ha="center", fontsize=7,
                  fontweight="bold", color="#1a2540")

    plt.savefig(OUT_GUARDIAS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Análisis guardias: {OUT_GUARDIAS}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN 3 — ANÁLISIS POR VARIABLE (10 paneles: 5×2)
# ══════════════════════════════════════════════════════════════════════════════

def fig_analisis_variables(df_res: pd.DataFrame):
    """
    Un panel por variable — mediana de PnL% ± IQR y máximo.
    Las dos columnas de triggers usan un eje secundario que muestra también
    la tasa de aprobación del filtro, para ver el trade-off entre señales
    generadas y calidad del filtro.
    """
    variables = [
        ("rsi_length",          "RSI_LENGTH",           False, None),
        ("N",                   "N (ventana)",           False, None),
        ("floor_pct",           "FLOOR_PCT (%)",         False, None),
        ("factor_caida",        "FACTOR_CAIDA",          False, None),
        ("factor_subida",       "FACTOR_SUBIDA",         False, None),
        ("rsi_buy_trigger",     "RSI_BUY_TRIGGER",       False, "tasa_apr_compra"),
        ("rsi_sell_trigger",    "RSI_SELL_TRIGGER",      False, "tasa_apr_venta"),
        ("guardia_compra",      "GUARDIA_COMPRA",        True,  None),
        ("guardia_prec_compra", "GUARDIA_PRECIO_COMPRA", True,  None),
        ("guardia_prec_venta",  "GUARDIA_PRECIO_VENTA",  True,  None),
    ]

    fig = plt.figure(figsize=(22, 25))
    fig.patch.set_facecolor("#f4f6fa")
    gs  = GridSpec(5, 2, figure=fig, hspace=0.60, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(5) for c in range(2)]

    for ax, (col, label, es_bool, tasa_col) in zip(axes, variables):
        ax.set_facecolor("#ffffff")
        grp    = df_res.groupby(col)["pnl_pct"]
        medians = grp.median().sort_index()
        q25     = grp.quantile(0.25).sort_index()
        q75     = grp.quantile(0.75).sort_index()
        tops    = grp.max().sort_index()
        x_vals  = list(medians.index)
        x_pos   = range(len(x_vals))

        if es_bool:
            colors   = ["#2ecc71" if v else "#e74c3c" for v in x_vals]
            x_labels = ["✓ True" if v else "✗ False" for v in x_vals]
        else:
            norm_v   = medians.values
            vmin, vmax = norm_v.min(), norm_v.max()
            span     = vmax - vmin if vmax > vmin else 1
            colors   = [mcolors.to_hex(plt.cm.RdYlGn(0.2 + 0.8 * (v - vmin) / span))
                        for v in norm_v]
            x_labels = [str(v) for v in x_vals]

        ax.bar(x_pos, medians.values, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.7, zorder=3)
        yerr_low  = medians.values - q25.values
        yerr_high = q75.values - medians.values
        ax.errorbar(x_pos, medians.values,
                    yerr=[yerr_low, yerr_high],
                    fmt="none", color="#555", linewidth=1.2, capsize=4, zorder=4)
        ax.scatter(x_pos, tops.values, marker="^", color="#e67e22",
                   s=45, zorder=5, label="Máx PnL%")
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)

        for xi, med in enumerate(medians.values):
            ax.text(xi, med + (q75.values[xi] - med) * 0.15,
                    f"{med:+.1f}%", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#1a2540")

        # Para las columnas de trigger: superponer tasa de aprobación en eje secundario
        if tasa_col is not None:
            ax2    = ax.twinx()
            tasas  = df_res.groupby(col)[tasa_col].mean().sort_index()
            ax2.plot(x_pos, tasas.values, color="#9b59b6", marker="o",
                     linewidth=1.5, markersize=5, zorder=6, alpha=0.85)
            ax2.set_ylabel("Tasa aprobación (%)", fontsize=7, color="#9b59b6")
            ax2.tick_params(axis="y", labelcolor="#9b59b6", labelsize=7)
            ax2.set_ylim(0, 105)
            ax.legend(["▲ Máx PnL%", "● Tasa aprob."], fontsize=6.5,
                      loc="upper right", handletextpad=0.3, borderpad=0.4)
            ax.set_title(f"{label}  (azul=PnL · violeta=tasa aprob.)",
                         fontsize=9.5, fontweight="bold", color="#1a2540", pad=6)
        else:
            if not es_bool:
                ax.legend(["▲ Máx PnL%"], fontsize=7, loc="upper right",
                          handletextpad=0.3, borderpad=0.4)
            ax.set_title(label, fontsize=10, fontweight="bold", color="#1a2540", pad=6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("PnL% (mediana ± IQR)", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, color="#dde3ef")

    fig.suptitle(
        f"Análisis de Impacto por Variable — Divergencia RSI Umbral\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  |  "
        f"Cada barra = mediana PnL% · barras de error = IQR (Q25–Q75) · ▲ = máximo\n"
        f"Total combinaciones: {len(df_res):,}  |  "
        f"Paneles BUY/SELL_TRIGGER incluyen tasa de aprobación del filtro (eje derecho)",
        fontsize=11, fontweight="bold", color="#1a2540", y=1.01,
    )
    plt.savefig(OUT_ANALISIS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Análisis variables: {OUT_ANALISIS}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(df_res: pd.DataFrame):
    sep = "═" * 78

    print(f"\n{sep}")
    print("  RESUMEN — OPTIMIZADOR COMPLETO UMBRAL")
    print(sep)
    print(f"  Período       : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Combinaciones : {len(df_res):,}")
    print(f"  PnL% rango    : {df_res['pnl_pct'].min():+.2f}%  →  {df_res['pnl_pct'].max():+.2f}%")
    print(f"  PnL% mediana  : {df_res['pnl_pct'].median():+.2f}%")
    print(f"  PnL% positivos: {(df_res['pnl_pct'] > 0).sum():,}  "
          f"({(df_res['pnl_pct'] > 0).mean()*100:.1f}%)")

    def bs(v): return "Si" if v else "No"

    hdr = (f"  {'#':>3}  {'RSI':>4}  {'N':>3}  {'BT':>4}  {'ST':>4}  "
           f"{'FL%':>4}  {'FC':>5}  {'FS':>5}  "
           f"{'GC':>3}  {'GPC':>4}  {'GPV':>4}  "
           f"{'PnL%':>8}  {'BTC$':>9}  {'Eq':>7}  "
           f"{'C':>3}  {'V':>3}  {'DC%':>5}  {'DV%':>5}  {'DD%':>6}")

    rankings = [
        ("RANKING 1 — MÁXIMO PnL%",     df_res.sort_values("pnl_pct",          ascending=False)),
        ("RANKING 2 — MÁXIMO BTC$",      df_res.sort_values("btc_value",         ascending=False)),
        ("RANKING 3 — MEJOR EQUILIBRIO", df_res.sort_values("equilibrio_score",  ascending=False)),
    ]

    for titulo, ranked in rankings:
        print(f"\n  {'─'*76}")
        print(f"  {titulo}")
        print(f"  {'─'*76}")
        print(hdr)
        print(f"  {'─'*76}")
        for rank, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
            marker = "★" if rank <= 3 else " "
            print(f"  {marker}{rank:>2}.  "
                  f"{r.rsi_length:>4}  {r.N:>3}  "
                  f"{int(r.rsi_buy_trigger):>4}  {int(r.rsi_sell_trigger):>4}  "
                  f"{r.floor_pct:>3.0f}%  {r.factor_caida:>5.1f}  {r.factor_subida:>5.1f}  "
                  f"{bs(r.guardia_compra):>3}  {bs(r.guardia_prec_compra):>4}  "
                  f"{bs(r.guardia_prec_venta):>4}  "
                  f"{r.pnl_pct:>+7.2f}%  ${r.btc_value:>8,.2f}  "
                  f"{r.equilibrio_score:>7.4f}  "
                  f"{int(r.total_compras):>3}  {int(r.total_ventas):>3}  "
                  f"{r.tasa_apr_compra:>4.0f}%  {r.tasa_apr_venta:>4.0f}%  "
                  f"{r.max_drawdown:>5.1f}%")

    # Valores dominantes por ranking
    for titulo, ranked in rankings:
        top10 = ranked.head(max(1, len(df_res) // 10))
        print(f"\n  Dominantes — {titulo}")
        for col, lbl_ in [("rsi_length","RSI"), ("N","N"),
                           ("rsi_buy_trigger","BUY_T"), ("rsi_sell_trigger","SELL_T"),
                           ("floor_pct","FLOOR%"), ("factor_caida","F_CAIDA"),
                           ("factor_subida","F_SUBIDA")]:
            val  = top10[col].mode().iloc[0]
            freq = (top10[col] == val).mean() * 100
            print(f"    {lbl_:<12}: {val}  ({freq:.0f}% del top 10%)")
        for col, lbl_ in [("guardia_compra","G_COMPRA"),
                           ("guardia_prec_compra","G_PREC_C"),
                           ("guardia_prec_venta","G_PREC_V")]:
            pct = top10[col].mean() * 100
            print(f"    {lbl_:<12}: {'✓ True' if pct >= 50 else '✗ False'}  ({pct:.0f}% True)")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMIZADOR COMPLETO — DIVERGENCIA RSI UMBRAL · TODAS LAS VARS     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    print("Cargando datos...")
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py")
        return
    print(f"  Velas  : {len(df):,}")
    print(f"  Desde  : {df['datetime'].iloc[0]}")
    print(f"  Hasta  : {df['datetime'].iloc[-1]}\n")

    df_res = optimizar(df)

    precio_final = float(df["close"].iloc[-1])
    df_res = _agregar_scores(df_res, precio_final)

    print("\nGuardando resultados...")
    guardar_resultados(df_res)

    imprimir_resumen(df_res)

    print("Generando visualizaciones...")
    fig_tres_tablas(df_res)
    fig_analisis_guardias(df_res)
    fig_analisis_variables(df_res)

    print(f"\n{'═'*70}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*70}")
    for f in [OUT_CSV, OUT_JSON, OUT_TABLA_PNL, OUT_TABLA_BTC,
              OUT_TABLA_EQ, OUT_GUARDIAS, OUT_ANALISIS]:
        print(f"  · {f}")
    print(f"{'═'*70}")
    print("✓ Proceso completado.\n")


if __name__ == "__main__":
    main()
