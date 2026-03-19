"""
Optimizador Completo — Divergencia RSI · Gradiente Asintótico  [VERSIÓN 2 VECTORIZADA]
══════════════════════════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Grid Search Exhaustivo
Estrategia: Estrategia_Divergencia_RSI_Umbral (curvas asintóticas de dos fases)

DIFERENCIAS RESPECTO AL OPTIMIZADOR v1 (Umbral)
────────────────────────────────────────────────────────────────────────────────
1. Curva de COMPRA — función asintótica de dos fases
   Antes : pct = pos ^ FACTOR_CAIDA          (potencia simple, 1 param)
   Ahora : pct = _gradiente(pos, NIVEL, INFL, K, FAC2)   (2 fases, 4 params)
     · Fase 1 [0, INFL]: exponencial que satura en NIVEL% del capital
     · Fase 2 [INFL, 1]: power que acelera desde NIVEL% hasta 100%

2. Curva de VENTA — anclada a PP con techo ATH_TEO dinámico
   Antes : techo = ATH real  →  log(precio/PP) / log(ATH/PP)
   Ahora : techo = ATH_TEO = ATL / (FLOOR_PCT/100)
             →  log(precio/PP) / log(ATH_TEO/PP)
   El ATH_TEO se recalibra con cada nuevo mínimo (ATL actualizable por señal).
   Misma función asintótica de dos fases que en compra.

3. Parámetros a optimizar por curva (×2 = compra + venta)
   NIVEL (% asíntota fase 1), INFL (pos inflexión), K (vel. exp.), FAC2 (curvatura)

4. Señales precomputadas
   Se añade `atl` por señal (necesario para calcular ATH_TEO en la venta).

5. Espacio de búsqueda default (conservador para rendimiento similar al v1)
   M por par (RSI/N) ≈ 660K  →  ~32M backtests totales
   Ajustar *_PASO para expandir / contraer.

ARQUITECTURA (idéntica al v1)
────────────────────────────────────────────────────────────────────────────────
· Señales precomputadas por (RSI_LENGTH, N) — sin re-escanear 9 000 velas
· Simulación vectorizada NumPy — estado de M combinaciones en paralelo
· Tres min-heaps de tamaño fijo — solo los mejores ~1 500 resultados a disco
· Sin Numba, sin threads, sin procesos
"""

import sqlite3
import json
import math
import os
import time
import heapq as _heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import itertools
from itertools import product
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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
#
# ── Señal de divergencia ─────────────────────────────────────────────────────
RSI_LENGTH_INICIO = 5  ;  RSI_LENGTH_FIN = 25  ;  RSI_LENGTH_PASO = 2.5
N_INICIO          = 10 ;  N_FIN          = 50  ;  N_PASO          = 5

# ── Referencia de ciclo ───────────────────────────────────────────────────────
FLOOR_PCT_INICIO  = 10 ;  FLOOR_PCT_FIN  = 25  ;  FLOOR_PCT_PASO  = 5
TOP_PCT_INICIO    = 400;  TOP_PCT_FIN    = 1000 ;  TOP_PCT_PASO    = 150  # [400, 1000]  p.ej. 400=4x, 750=7.5x, 1000=10x

# ── Umbrales RSI ─────────────────────────────────────────────────────────────
RSI_BUY_TRIGGER_INICIO  = 10 ;  RSI_BUY_TRIGGER_FIN  = 30 ;  RSI_BUY_TRIGGER_PASO  = 2.5
RSI_SELL_TRIGGER_INICIO = 70 ;  RSI_SELL_TRIGGER_FIN  = 90 ;  RSI_SELL_TRIGGER_PASO = 2.5

# ── Curva asintótica de COMPRA ────────────────────────────────────────────────
#   NIVEL  : % del capital disponible — asíntota de la fase 1   (rango típico: 5–30)
#   INFL   : posición de inflexión entre fase 1 y fase 2         (rango típico: 0.05–0.40)
#   K      : velocidad de saturación exponencial — fase 1        (rango típico: 2–12)
#   FAC2   : curvatura de la fase 2  (>1 = convexa, más agresiva al fondo)
#
#   Nota: con 2 valores por parámetro M ≈ 660K por par RSI/N → ~32M backtests totales.
#   Para expandir la búsqueda basta reducir el paso o añadir un valor intermedio.
NIVEL_C_INICIO = 5  ;  NIVEL_C_FIN = 20 ;  NIVEL_C_PASO = 5   # [5, 20]
INFL_C_INICIO  = 0.05; INFL_C_FIN = 0.30; INFL_C_PASO  = 0.05  # [0.05, 0.30]
K_C_INICIO     = 2  ;  K_C_FIN    = 8  ;  K_C_PASO     = 3     # [2, 8]
FAC2_C_INICIO  = 1  ;  FAC2_C_FIN = 5  ;  FAC2_C_PASO  = 2     # [1, 5]

# ── Curva asintótica de VENTA ─────────────────────────────────────────────────
#   Misma estructura; optimizada de forma independiente.
NIVEL_V_INICIO = 5  ;  NIVEL_V_FIN = 20 ;  NIVEL_V_PASO = 5   # [5, 20]
INFL_V_INICIO  = 0.05; INFL_V_FIN = 0.30; INFL_V_PASO  = 0.5  # [0.05, 0.30]
K_V_INICIO     = 2  ;  K_V_FIN    = 8  ;  K_V_PASO     = 3     # [2, 8]
FAC2_V_INICIO  = 1  ;  FAC2_V_FIN = 5  ;  FAC2_V_PASO  = 2     # [1, 5]

# ── Salida ────────────────────────────────────────────────────────────────────
OUT_CSV       = "optimizacion_asintotico.csv"
OUT_JSON      = "optimizacion_asintotico_top.json"
OUT_TABLA_PNL = "optimizacion_asintotico_ranking_pnl.png"
OUT_TABLA_BTC = "optimizacion_asintotico_ranking_btc.png"
OUT_TABLA_EQ  = "optimizacion_asintotico_ranking_equilibrio.png"
OUT_GUARDIAS  = "optimizacion_asintotico_guardias.png"
OUT_ANALISIS  = "optimizacion_asintotico_analisis.png"
TOP_N         = 25
TOP_HEAP      = 500


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _rango(inicio, fin, paso, entero=False):
    vals, v = [], inicio
    while v <= fin + paso * 1e-9:
        vals.append(int(round(v)) if entero else round(v, 10))
        v += paso
    return vals


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
    df = df.reset_index(drop=True)
    n_nan = df[["high", "low", "close"]].isna().sum().sum()
    if n_nan > 0:
        print(f"  ⚠ {n_nan} valores NaN — se eliminan filas afectadas")
        df = df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    return df


def calcular_rsi(series: pd.Series, length: int) -> np.ndarray:
    """RSI clásico de Wilder (EWM), idéntico al de TradingView."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values.astype(float)


def _gradiente_vec(pos_arr, nivel_arr, infl_arr, k_arr, fac2_arr):
    """
    Gradiente asintótico de dos fases, completamente vectorizado sobre M combos.

    Parámetros (todos shape (M,)):
      pos_arr   : posición normalizada ∈ [0, 1]
      nivel_arr : % asíntota de la fase 1
      infl_arr  : pos de inflexión entre fase 1 y fase 2
      k_arr     : velocidad de saturación exponencial (fase 1)
      fac2_arr  : curvatura de la aceleración (fase 2)

    Retorna pct ∈ [0, 100] — porcentaje de capital/BTC a usar.
    """
    # Fase 1: exponencial  →  satura exactamente en nivel% cuando pos = infl
    scale = 1.0 - np.exp(-k_arr * infl_arr)                # denominador normalizador
    ph1   = np.where(
        scale > 1e-12,
        nivel_arr * (1.0 - np.exp(-k_arr * pos_arr)) / np.maximum(scale, 1e-30),
        0.0,
    )
    # Fase 2: power  →  arranca en nivel%, llega a 100% en pos = 1
    x2  = (pos_arr - infl_arr) / np.maximum(1.0 - infl_arr, 1e-10)
    ph2 = nivel_arr + (100.0 - nivel_arr) * np.power(np.maximum(x2, 0.0), fac2_arr)

    return np.where(pos_arr <= 0.0, 0.0,
           np.where(pos_arr <= infl_arr, ph1, ph2))


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — PRECOMPUTO DE SEÑALES
# ══════════════════════════════════════════════════════════════════════════════
# Para un par (RSI_LENGTH, N) fijo, los eventos de divergencia son idénticos
# para todas las combinaciones restantes. Se detectan una sola vez.
#
# Cada señal almacena:
#   tipo  : 0=compra  1=venta
#   price : low (compra) o high (venta)
#   ath   : ATH vigente en ese punto (para la curva de compra)
#   atl   : ATL vigente en ese punto (para ATH_TEO en la curva de venta)
#   anc   : RSI del ancla (mínimo o máximo previo de la ventana)
#   close : close de esa vela (para mark-to-market del drawdown)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_signals(lows, highs, closes, rsi_low, rsi_high, N):
    n = len(lows)
    sig_tipo  = []
    sig_price = []
    sig_ath   = []
    sig_atl   = []      # ← nuevo: necesario para ATH_TEO en la venta
    sig_anc   = []
    sig_close = []

    # Inicializar ATH y ATL con las primeras N velas
    ath = highs[:N].max()
    atl = lows[:N].min()

    for i in range(N, n):
        if highs[i] > ath:
            ath = highs[i]
        if lows[i] < atl:
            atl = lows[i]

        rl_i = rsi_low[i]
        rh_i = rsi_high[i]
        if math.isnan(rl_i) or math.isnan(rh_i):
            continue

        wl = lows[i - N:i]
        wh = highs[i - N:i]
        buy_detected = False

        # Señal de COMPRA: divergencia alcista
        if lows[i] < wl.min():
            idx_min = i - N + int(wl.argmin())
            rsi_anc = rsi_low[idx_min]
            if not math.isnan(rsi_anc) and rl_i > rsi_anc:
                sig_tipo.append(0)
                sig_price.append(lows[i])
                sig_ath.append(ath)
                sig_atl.append(atl)
                sig_anc.append(rsi_anc)
                sig_close.append(closes[i])
                buy_detected = True

        # Señal de VENTA: divergencia bajista (solo si no hubo compra)
        if not buy_detected and highs[i] > wh.max():
            idx_max = i - N + int(wh.argmax())
            rsi_anc = rsi_high[idx_max]
            if not math.isnan(rsi_anc) and rh_i < rsi_anc:
                sig_tipo.append(1)
                sig_price.append(highs[i])
                sig_ath.append(ath)
                sig_atl.append(atl)
                sig_anc.append(rsi_anc)
                sig_close.append(closes[i])

    if not sig_tipo:
        return None

    return {
        "tipo"  : np.array(sig_tipo,  dtype=np.int8),
        "price" : np.array(sig_price, dtype=np.float64),
        "ath"   : np.array(sig_ath,   dtype=np.float64),
        "atl"   : np.array(sig_atl,   dtype=np.float64),   # ← nuevo
        "anc"   : np.array(sig_anc,   dtype=np.float64),
        "close" : np.array(sig_close, dtype=np.float64),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — SIMULACIÓN VECTORIZADA
# ══════════════════════════════════════════════════════════════════════════════
# Para cada señal, actualiza el estado de TODAS las M combinaciones en paralelo.
#
# Layout de combos_arr  (shape M × 15):
#   col  0 : floor_pct
#   col  1 : top_pct
#   col  2 : nivel_c     col  3 : infl_c    col  4 : k_c    col  5 : fac2_c
#   col  6 : nivel_v     col  7 : infl_v    col  8 : k_v    col  9 : fac2_v
#   col 10 : rsi_buy_trigger
#   col 11 : rsi_sell_trigger
#   col 12 : guardia_compra
#   col 13 : guardia_prec_compra
#   col 14 : guardia_prec_venta
# ══════════════════════════════════════════════════════════════════════════════

def simular_vectorizado(signals, combos_arr, usdt_ini, usdt_res, comm_pct, btc_acum):
    stype  = signals["tipo"]
    sprice = signals["price"]
    sath   = signals["ath"]
    satl   = signals["atl"]      # ← nuevo
    sanc   = signals["anc"]
    sclose = signals["close"]
    S = len(stype)
    M = len(combos_arr)

    fp_arr   = combos_arr[:, 0]               # floor_pct
    tp_arr   = combos_arr[:, 1]               # top_pct
    nc_arr   = combos_arr[:, 2]               # nivel_c
    ic_arr   = combos_arr[:, 3]               # infl_c
    kc_arr   = combos_arr[:, 4]               # k_c
    f2c_arr  = combos_arr[:, 5]               # fac2_c
    nv_arr   = combos_arr[:, 6]               # nivel_v
    iv_arr   = combos_arr[:, 7]               # infl_v
    kv_arr   = combos_arr[:, 8]               # k_v
    f2v_arr  = combos_arr[:, 9]               # fac2_v
    bt_arr   = combos_arr[:, 10]              # rsi_buy_trigger
    st_arr   = combos_arr[:, 11]              # rsi_sell_trigger
    gc_arr   = combos_arr[:, 12].astype(bool) # guardia_compra
    gpc_arr  = combos_arr[:, 13].astype(bool) # guardia_prec_compra
    gpv_arr  = combos_arr[:, 14].astype(bool) # guardia_prec_venta

    # Pre-calcular log_r para la curva de compra (solo depende de floor_pct)
    log_r_arr = np.log(100.0 / np.maximum(fp_arr, 1e-6))
    comm      = comm_pct / 100.0
    usdt_disp = usdt_ini - usdt_res

    # Estado: shape (M,)
    usdt     = np.full(M, usdt_disp, dtype=np.float64)
    btc_pos  = np.zeros(M,           dtype=np.float64)
    usdt_inv = np.zeros(M,           dtype=np.float64)
    pmin     = np.full(M, np.inf,    dtype=np.float64)
    pmax     = np.zeros(M,           dtype=np.float64)

    compras   = np.zeros(M, dtype=np.int32)
    ventas    = np.zeros(M, dtype=np.int32)
    div_c_det = np.zeros(M, dtype=np.int32)
    div_c_apr = np.zeros(M, dtype=np.int32)
    div_v_det = np.zeros(M, dtype=np.int32)
    div_v_apr = np.zeros(M, dtype=np.int32)
    pos_count = np.zeros(M, dtype=np.int32)
    peak_port = np.full(M, usdt_disp, dtype=np.float64)
    max_dd    = np.zeros(M, dtype=np.float64)

    for j in range(S):
        price = sprice[j]
        ath_j = sath[j]
        atl_j = satl[j]         # ← ATL en este punto
        anc   = sanc[j]

        has_pos = btc_pos > 0.0
        pp = np.where(has_pos, usdt_inv / np.maximum(btc_pos, 1e-30), 0.0)

        # Drawdown mark-to-market
        port      = usdt + btc_pos * sclose[j] + usdt_res
        peak_port = np.maximum(peak_port, port)
        dd        = np.where(peak_port > 0, (peak_port - port) / peak_port * 100.0, 0.0)
        max_dd    = np.maximum(max_dd, dd)

        # ════ COMPRA ════════════════════════════════════════════════════════
        if stype[j] == 0:
            div_c_det += 1

            trigger_mask = anc <= bt_arr
            div_c_apr   += trigger_mask.astype(np.int32)

            mask  = trigger_mask.copy()
            mask &= ~(gc_arr  & has_pos & (price >= pp))
            mask &= ~(gpc_arr & np.isfinite(pmin) & (price >= pmin))

            # Posición normalizada: log(ATH/precio) / log(100/FLOOR_PCT)
            log_ratio = np.log(np.maximum(ath_j / price, 1.0001))
            pos_c     = np.clip(log_ratio / log_r_arr, 0.0, 1.0)

            # Gradiente asintótico de dos fases → % capital
            pct_c = _gradiente_vec(pos_c, nc_arr, ic_arr, kc_arr, f2c_arr)
            ua    = usdt * (pct_c / 100.0)
            mask &= ua > 1e-8

            do        = mask.astype(np.float64)
            btc_pos  += do * (ua * (1.0 - comm)) / price
            usdt     -= do * ua
            usdt_inv += do * ua
            pmin      = np.where(mask & (price < pmin), price, pmin)
            compras   += mask.astype(np.int32)
            pos_count += mask.astype(np.int32)

        # ════ VENTA ═════════════════════════════════════════════════════════
        else:
            div_v_det += 1

            # ATH_TEO: techo teórico = ATL × TOP_PCT / 100  (independiente de FLOOR_PCT)
            ath_teo_arr = atl_j * (tp_arr / 100.0)

            can_sell     = has_pos & (pp > 0.0) & (price > pp) & (ath_teo_arr > pp)
            trigger_mask = can_sell & (anc >= st_arr)
            div_v_apr   += trigger_mask.astype(np.int32)

            mask = trigger_mask.copy()

            # Posición normalizada: log(precio/PP) / log(ATH_TEO/PP)
            safe_pp  = np.where(can_sell, pp, ath_teo_arr * 0.99)
            log_a    = np.where(can_sell,
                                np.log(np.maximum(ath_teo_arr / safe_pp, 1.0001)),
                                1.0)
            log_a    = np.maximum(log_a, 1e-10)
            log_num  = np.where(can_sell,
                                np.log(np.maximum(price / safe_pp, 1.0001)),
                                0.0)
            pos_v    = np.where(can_sell, np.clip(log_num / log_a, 0.0, 1.0), 0.0)

            # Gradiente asintótico de dos fases → % BTC a vender
            pct_v = _gradiente_vec(pos_v, nv_arr, iv_arr, kv_arr, f2v_arr)
            slot  = btc_pos * (pct_v / 100.0)
            mask &= slot > 1e-12
            mask &= ~(gpv_arr & (pmax > 0.0) & (price <= pmax))

            safe_btc  = np.maximum(btc_pos, 1e-30)
            cp        = np.where(mask, usdt_inv * slot / safe_btc, 0.0)
            do        = mask.astype(np.float64)
            usdt_inv  = np.maximum(usdt_inv - cp, 0.0)
            btc_pos  -= do * slot
            usdt     += do * slot * price * (1.0 - comm)
            pmax      = np.where(mask & (price > pmax), price, pmax)
            ventas    += mask.astype(np.int32)
            pos_count -= mask.astype(np.int32)

    # ── Métricas finales ─────────────────────────────────────────────────────
    last_close   = sclose[-1]
    btc_val      = btc_pos * last_close
    portfolio    = usdt + btc_val + usdt_res
    pnl_pct      = (portfolio - usdt_ini) / usdt_ini * 100.0
    total_trades = compras + ventas
    pp_final     = np.where(btc_pos > 0,
                            usdt_inv / np.maximum(btc_pos, 1e-30), 0.0)
    pnl_trade    = np.where(total_trades > 0,
                            pnl_pct / total_trades.astype(np.float64), 0.0)
    tasa_c = np.where(div_c_det > 0,
                      div_c_apr / np.maximum(div_c_det, 1).astype(np.float64) * 100.0, 0.0)
    tasa_v = np.where(div_v_det > 0,
                      div_v_apr / np.maximum(div_v_det, 1).astype(np.float64) * 100.0, 0.0)

    return {
        "pnl_pct"        : pnl_pct,
        "portfolio"      : portfolio,
        "usdt_final"     : usdt + usdt_res,
        "btc_posiciones" : btc_pos,
        "precio_prom_fin": pp_final,
        "total_trades"   : total_trades,
        "compras"        : compras,
        "ventas"         : ventas,
        "pos_count"      : pos_count,
        "pnl_por_trade"  : pnl_trade,
        "max_dd"         : max_dd,
        "div_c_det"      : div_c_det,
        "div_c_apr"      : div_c_apr,
        "tasa_c"         : tasa_c,
        "div_v_det"      : div_v_det,
        "div_v_apr"      : div_v_apr,
        "tasa_v"         : tasa_v,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

COL_ORDER = [
    "rsi_length", "N", "floor_pct", "top_pct",
    "nivel_c", "infl_c", "k_c", "fac2_c",
    "nivel_v", "infl_v", "k_v", "fac2_v",
    "rsi_buy_trigger", "rsi_sell_trigger",
    "guardia_compra", "guardia_prec_compra", "guardia_prec_venta",
    "pnl_pct", "portfolio_final", "usdt_final", "btc_posiciones",
    "precio_prom_fin", "total_trades", "total_compras", "total_ventas",
    "positions_count", "pnl_por_trade", "max_drawdown",
    "div_compra_det", "div_compra_apr", "tasa_apr_compra",
    "div_venta_det",  "div_venta_apr",  "tasa_apr_venta",
]


# ══════════════════════════════════════════════════════════════════════════════
# MIN-HEAP DE TAMAÑO FIJO
# ══════════════════════════════════════════════════════════════════════════════

class _TopHeap:
    """Min-heap de tamaño fijo que retiene los N mayores valores."""
    __slots__ = ("n", "_heap", "_counter")

    def __init__(self, n: int):
        self.n        = n
        self._heap    = []
        self._counter = 0

    @property
    def min_val(self) -> float:
        return self._heap[0][0] if self._heap else -math.inf

    def push_if_better(self, metric_val: float, row: dict):
        if len(self._heap) < self.n:
            _heapq.heappush(self._heap, (metric_val, self._counter, row))
            self._counter += 1
        elif metric_val > self.min_val:
            _heapq.heapreplace(self._heap, (metric_val, self._counter, row))
            self._counter += 1

    def to_list(self) -> list:
        return [entry[2] for entry in self._heap]


def optimizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grid search con tres min-heaps de tamaño TOP_HEAP.
    Solo los mejores resultados por cada métrica se conservan en RAM/disco.
    """
    lows   = df["low"].values.astype(np.float64)
    highs  = df["high"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    rsi_lengths   = _rango(RSI_LENGTH_INICIO, RSI_LENGTH_FIN, RSI_LENGTH_PASO, entero=True)
    ns            = _rango(N_INICIO,          N_FIN,          N_PASO,          entero=True)
    floor_pcts    = _rango(FLOOR_PCT_INICIO,  FLOOR_PCT_FIN,  FLOOR_PCT_PASO)
    top_pcts      = _rango(TOP_PCT_INICIO,    TOP_PCT_FIN,    TOP_PCT_PASO)
    nivel_c_vals  = _rango(NIVEL_C_INICIO, NIVEL_C_FIN, NIVEL_C_PASO)
    infl_c_vals   = _rango(INFL_C_INICIO,  INFL_C_FIN,  INFL_C_PASO)
    k_c_vals      = _rango(K_C_INICIO,     K_C_FIN,     K_C_PASO)
    fac2_c_vals   = _rango(FAC2_C_INICIO,  FAC2_C_FIN,  FAC2_C_PASO)
    nivel_v_vals  = _rango(NIVEL_V_INICIO, NIVEL_V_FIN, NIVEL_V_PASO)
    infl_v_vals   = _rango(INFL_V_INICIO,  INFL_V_FIN,  INFL_V_PASO)
    k_v_vals      = _rango(K_V_INICIO,     K_V_FIN,     K_V_PASO)
    fac2_v_vals   = _rango(FAC2_V_INICIO,  FAC2_V_FIN,  FAC2_V_PASO)
    buy_triggers  = _rango(RSI_BUY_TRIGGER_INICIO,  RSI_BUY_TRIGGER_FIN,  RSI_BUY_TRIGGER_PASO)
    sell_triggers = _rango(RSI_SELL_TRIGGER_INICIO, RSI_SELL_TRIGGER_FIN, RSI_SELL_TRIGGER_PASO)
    guardias_bool = [True, False]

    n_curve_c    = len(nivel_c_vals)*len(infl_c_vals)*len(k_c_vals)*len(fac2_c_vals)
    n_curve_v    = len(nivel_v_vals)*len(infl_v_vals)*len(k_v_vals)*len(fac2_v_vals)
    total_combos = (len(rsi_lengths) * len(ns) * len(floor_pcts) * len(top_pcts) *
                    n_curve_c * n_curve_v *
                    len(buy_triggers) * len(sell_triggers) * 8)
    rsi_n_pairs     = len(rsi_lengths) * len(ns)
    combos_per_pair = total_combos // rsi_n_pairs

    print(f"\n{'═'*72}")
    print(f"  GRID SEARCH — {total_combos:,} combinaciones")
    print(f"{'═'*72}")
    print(f"  RSI_LENGTH        : {rsi_lengths[0]} → {rsi_lengths[-1]}  ({len(rsi_lengths)} vals)")
    print(f"  N                 : {ns[0]} → {ns[-1]}  ({len(ns)} vals)")
    print(f"  FLOOR_PCT         : {floor_pcts[0]} → {floor_pcts[-1]}  ({len(floor_pcts)} vals)  [piso compra]")
    print(f"  TOP_PCT           : {top_pcts[0]} → {top_pcts[-1]}  ({len(top_pcts)} vals)  [techo venta, × ATL]")
    print(f"  Curva COMPRA")
    print(f"    NIVEL_C         : {nivel_c_vals}  ({len(nivel_c_vals)} vals)")
    print(f"    INFL_C          : {infl_c_vals}   ({len(infl_c_vals)} vals)")
    print(f"    K_C             : {k_c_vals}  ({len(k_c_vals)} vals)")
    print(f"    FAC2_C          : {fac2_c_vals}  ({len(fac2_c_vals)} vals)  → {n_curve_c} combos")
    print(f"  Curva VENTA")
    print(f"    NIVEL_V         : {nivel_v_vals}  ({len(nivel_v_vals)} vals)")
    print(f"    INFL_V          : {infl_v_vals}   ({len(infl_v_vals)} vals)")
    print(f"    K_V             : {k_v_vals}  ({len(k_v_vals)} vals)")
    print(f"    FAC2_V          : {fac2_v_vals}  ({len(fac2_v_vals)} vals)  → {n_curve_v} combos")
    print(f"  RSI_BUY_TRIGGER   : {buy_triggers[0]} → {buy_triggers[-1]}  ({len(buy_triggers)} vals)")
    print(f"  RSI_SELL_TRIGGER  : {sell_triggers[0]} → {sell_triggers[-1]}  ({len(sell_triggers)} vals)")
    print(f"  Guardias          : 2³ = 8 combinaciones booleanas")
    print(f"  Pares RSI/N       : {rsi_n_pairs}  ({combos_per_pair:,} combos/par)")
    print(f"  Top-K por métrica : {TOP_HEAP}  (máx ~{TOP_HEAP * 3:,} filas en disco)")
    print(f"{'═'*72}\n")

    # Pre-calcular RSI para cada RSI_LENGTH
    print("  Pre-calculando RSI...")
    rsi_cache = {}
    for rsi_len in rsi_lengths:
        rsi_cache[rsi_len] = (
            calcular_rsi(df["low"],  rsi_len),
            calcular_rsi(df["high"], rsi_len),
        )
    print(f"  ✓ {len(rsi_cache)} pares RSI pre-calculados\n")

    # ── Pre-construir arrays de parámetros para indexación numpy ─────────────
    # En lugar de iterar tuplas Python (lentísimo a >1M combos), representamos
    # el espacio cartesiano como 15 arrays 1-D y calculamos las columnas de
    # combos_arr con aritmética de índices pura numpy:
    #
    #   combo[k, j] = param_arrays[j][ k // stride[j] % size[j] ]
    #
    # Esto construye un batch de B×15 float64 con 15 operaciones numpy O(B)
    # en lugar de un loop Python O(B). Velocidad: ~1000× más rápida.
    _param_arrays = [
        np.array(floor_pcts,    dtype=np.float64),
        np.array(top_pcts,      dtype=np.float64),
        np.array(nivel_c_vals,  dtype=np.float64),
        np.array(infl_c_vals,   dtype=np.float64),
        np.array(k_c_vals,      dtype=np.float64),
        np.array(fac2_c_vals,   dtype=np.float64),
        np.array(nivel_v_vals,  dtype=np.float64),
        np.array(infl_v_vals,   dtype=np.float64),
        np.array(k_v_vals,      dtype=np.float64),
        np.array(fac2_v_vals,   dtype=np.float64),
        np.array(buy_triggers,  dtype=np.float64),
        np.array(sell_triggers, dtype=np.float64),
        np.array([1.0, 0.0]),   # guardia_compra
        np.array([1.0, 0.0]),   # guardia_prec_compra
        np.array([1.0, 0.0]),   # guardia_prec_venta
    ]
    _sizes   = np.array([len(a) for a in _param_arrays], dtype=np.int64)
    # stride[j] = producto de todos los tamaños posteriores a j
    _strides = np.ones(15, dtype=np.int64)
    for j in range(13, -1, -1):
        _strides[j] = _strides[j + 1] * _sizes[j + 1]

    def _build_batch_arr(start: int, end: int) -> np.ndarray:
        """
        Construye combos_arr[start:end] sin ningún loop Python.
        idx shape: (B,)  →  combos_arr shape: (B, 15)
        """
        idx = np.arange(start, end, dtype=np.int64)
        cols = [
            _param_arrays[j][(idx // _strides[j]) % _sizes[j]]
            for j in range(15)
        ]
        return np.column_stack(cols)   # (B, 15) float64

    def _decode_row(global_idx: int) -> tuple:
        """Devuelve la tupla de parámetros para el índice global dado."""
        return tuple(
            _param_arrays[j][int((global_idx // _strides[j]) % _sizes[j])]
            for j in range(15)
        )

    # Tamaño de batch: 500K × 15 cols × 8 bytes × ~20 arrays ≈ 1.1 GB pico
    BATCH_SIZE = 500_000
    n_batches  = math.ceil(combos_per_pair / BATCH_SIZE)
    ram_mb     = BATCH_SIZE * 15 * 8 * 20 / 1024**2

    print(f"  Modo    : numpy index-arithmetic, batches de {BATCH_SIZE:,}  (~{ram_mb:.0f} MB RAM/batch)")
    print(f"  Combos  : {combos_per_pair:,} por par RSI/N  →  {n_batches} batches/par")
    print(f"  Disco   : solo los mejores ~{TOP_HEAP*3:,} resultados\n")

    # Tres heaps — uno por métrica de ranking
    heap_pnl = _TopHeap(TOP_HEAP)
    heap_btc = _TopHeap(TOP_HEAP)
    heap_eq  = _TopHeap(TOP_HEAP)

    t0         = time.time()
    pares_done = 0
    best_pnl   = -999.0

    for rsi_len in rsi_lengths:
        rsi_low, rsi_high = rsi_cache[rsi_len]

        for n_val in ns:
            signals = precompute_signals(lows, highs, closes, rsi_low, rsi_high, n_val)
            n_sigs  = len(signals["tipo"]) if signals is not None else 0

            for b_start in range(0, combos_per_pair, BATCH_SIZE):
                b_end      = min(b_start + BATCH_SIZE, combos_per_pair)
                combos_arr = _build_batch_arr(b_start, b_end)

                if signals is not None:
                    arrs = simular_vectorizado(
                        signals, combos_arr,
                        float(SALDO_USDT_INICIAL), float(USDT_RESERVA),
                        float(COMMISSION_PCT), float(BTC_PCT_TO_ACCUMULATE),
                    )

                    pnl_arr = arrs["pnl_pct"]
                    btc_arr = arrs["btc_posiciones"]
                    eq_prx  = np.sqrt(np.maximum(pnl_arr, 0.0) * btc_arr)

                    min_pnl = heap_pnl.min_val
                    min_btc = heap_btc.min_val
                    min_eq  = heap_eq.min_val

                    cand_idx = np.where(
                        (pnl_arr > min_pnl) |
                        (btc_arr > min_btc) |
                        (eq_prx  > min_eq)
                    )[0]

                    for local_i in cand_idx:
                        global_i = b_start + int(local_i)
                        (fp, tp, nc, ic, kc, f2c,
                         nv, iv, kv, f2v,
                         bt, st, gc, gpc, gpv) = _decode_row(global_i)
                        row = {
                            "rsi_length"         : rsi_len,
                            "N"                  : n_val,
                            "floor_pct"          : fp,
                            "top_pct"            : tp,
                            "nivel_c"            : nc,
                            "infl_c"             : ic,
                            "k_c"                : kc,
                            "fac2_c"             : f2c,
                            "nivel_v"            : nv,
                            "infl_v"             : iv,
                            "k_v"                : kv,
                            "fac2_v"             : f2v,
                            "rsi_buy_trigger"    : bt,
                            "rsi_sell_trigger"   : st,
                            "guardia_compra"     : bool(gc),
                            "guardia_prec_compra": bool(gpc),
                            "guardia_prec_venta" : bool(gpv),
                            "pnl_pct"            : round(float(arrs["pnl_pct"][local_i]),          4),
                            "portfolio_final"    : round(float(arrs["portfolio"][local_i]),         2),
                            "usdt_final"         : round(float(arrs["usdt_final"][local_i]),        2),
                            "btc_posiciones"     : round(float(arrs["btc_posiciones"][local_i]),    8),
                            "precio_prom_fin"    : round(float(arrs["precio_prom_fin"][local_i]),   2),
                            "total_trades"       : int(arrs["total_trades"][local_i]),
                            "total_compras"      : int(arrs["compras"][local_i]),
                            "total_ventas"       : int(arrs["ventas"][local_i]),
                            "positions_count"    : int(arrs["pos_count"][local_i]),
                            "pnl_por_trade"      : round(float(arrs["pnl_por_trade"][local_i]),    4),
                            "max_drawdown"       : round(float(arrs["max_dd"][local_i]),            2),
                            "div_compra_det"     : int(arrs["div_c_det"][local_i]),
                            "div_compra_apr"     : int(arrs["div_c_apr"][local_i]),
                            "tasa_apr_compra"    : round(float(arrs["tasa_c"][local_i]),            1),
                            "div_venta_det"      : int(arrs["div_v_det"][local_i]),
                            "div_venta_apr"      : int(arrs["div_v_apr"][local_i]),
                            "tasa_apr_venta"     : round(float(arrs["tasa_v"][local_i]),            1),
                        }
                        heap_pnl.push_if_better(float(pnl_arr[local_i]), row)
                        heap_btc.push_if_better(float(btc_arr[local_i]), row)
                        heap_eq.push_if_better(float(eq_prx[local_i]),   row)

                    best_pnl = max(best_pnl, float(pnl_arr.max()))

            pares_done += 1
            elapsed = time.time() - t0
            done_c  = pares_done * combos_per_pair
            eta     = elapsed / done_c * (total_combos - done_c) if done_c > 0 else 0
            survivors = len({id(r) for h in (heap_pnl, heap_btc, heap_eq)
                             for r in h.to_list()})
            print(f"  [{pares_done:>3}/{rsi_n_pairs}] RSI={rsi_len:>2} N={n_val:>2} "
                  f"sigs={n_sigs:>4}  "
                  f"{elapsed:>6.1f}s  ETA:{eta:>5.1f}s  "
                  f"top:{survivors:>4}  mejor PnL: {best_pnl:>+8.2f}%")

    elapsed_total = time.time() - t0
    print(f"\n✓ Completado en {elapsed_total:.1f}s  "
          f"({total_combos / elapsed_total:,.0f} backtests/s)")

    # Unión deduplicada de los tres heaps
    seen     = set()
    all_rows = []
    for heap in (heap_pnl, heap_btc, heap_eq):
        for row in heap.to_list():
            key = (row["rsi_length"], row["N"], row["floor_pct"], row["top_pct"],
                   row["nivel_c"], row["infl_c"], row["k_c"], row["fac2_c"],
                   row["nivel_v"], row["infl_v"], row["k_v"], row["fac2_v"],
                   row["rsi_buy_trigger"], row["rsi_sell_trigger"],
                   row["guardia_compra"], row["guardia_prec_compra"],
                   row["guardia_prec_venta"])
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

    print(f"\n  ✓ Filas únicas conservadas: {len(all_rows):,}  "
          f"(máx teórico: {TOP_HEAP * 3:,})")

    df_res = pd.DataFrame(all_rows)[COL_ORDER]

    print(f"\n  ✓ Triggers verificados:")
    print(f"    RSI_BUY_TRIGGER  : {sorted(df_res['rsi_buy_trigger'].unique())}")
    print(f"    RSI_SELL_TRIGGER : {sorted(df_res['rsi_sell_trigger'].unique())}")

    return df_res.sort_values("pnl_pct", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _agregar_scores(df_res: pd.DataFrame, precio_final: float) -> pd.DataFrame:
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

    guard_stats = []
    for (gc, gpc, gpv), grp in df_res.groupby(
        ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    ):
        guard_stats.append({
            "guardia_compra"      : gc,
            "guardia_prec_compra" : gpc,
            "guardia_prec_venta"  : gpv,
            "n_combos"            : len(grp),
            "mejor_pnl_pct"       : round(grp["pnl_pct"].max(),         4),
            "mediana_pnl_pct"     : round(grp["pnl_pct"].median(),       4),
            "mejor_btc_value"     : round(grp["btc_value"].max(),        2),
            "mejor_eq_score"      : round(grp["equilibrio_score"].max(), 6),
        })

    payload = {
        "meta": {
            "estrategia"            : "Divergencia RSI — Gradiente Asintótico",
            "fecha_inicio"          : FECHA_INICIO,
            "fecha_fin"             : FECHA_FIN,
            "saldo_inicial"         : SALDO_USDT_INICIAL,
            "usdt_reserva_pct"      : USDT_RESERVA_PCT,
            "btc_pct_to_accumulate" : BTC_PCT_TO_ACCUMULATE,
            "commission_pct"        : COMMISSION_PCT,
            "rsi_range"             : [RSI_LENGTH_INICIO, RSI_LENGTH_FIN, RSI_LENGTH_PASO],
            "n_range"               : [N_INICIO,          N_FIN,          N_PASO],
            "floor_pct_range"       : [FLOOR_PCT_INICIO,  FLOOR_PCT_FIN,  FLOOR_PCT_PASO],
            "top_pct_range"         : [TOP_PCT_INICIO,    TOP_PCT_FIN,    TOP_PCT_PASO],
            "curva_compra": {
                "nivel" : [NIVEL_C_INICIO, NIVEL_C_FIN, NIVEL_C_PASO],
                "infl"  : [INFL_C_INICIO,  INFL_C_FIN,  INFL_C_PASO],
                "k"     : [K_C_INICIO,     K_C_FIN,     K_C_PASO],
                "fac2"  : [FAC2_C_INICIO,  FAC2_C_FIN,  FAC2_C_PASO],
            },
            "curva_venta": {
                "nivel" : [NIVEL_V_INICIO, NIVEL_V_FIN, NIVEL_V_PASO],
                "infl"  : [INFL_V_INICIO,  INFL_V_FIN,  INFL_V_PASO],
                "k"     : [K_V_INICIO,     K_V_FIN,     K_V_PASO],
                "fac2"  : [FAC2_V_INICIO,  FAC2_V_FIN,  FAC2_V_PASO],
            },
            "rsi_buy_range"         : [RSI_BUY_TRIGGER_INICIO,  RSI_BUY_TRIGGER_FIN,  RSI_BUY_TRIGGER_PASO],
            "rsi_sell_range"        : [RSI_SELL_TRIGGER_INICIO, RSI_SELL_TRIGGER_FIN, RSI_SELL_TRIGGER_PASO],
            "total_combos"          : len(df_res),
            "generado"              : pd.Timestamp.now().isoformat(),
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
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

# Columnas para las tablas de ranking
_COLS = ["#", "RSI", "N", "BUY_T", "SELL_T", "FL%", "TOP%",
         "NC", "IC", "KC", "F2C",    # curva compra
         "NV", "IV", "KV", "F2V",    # curva venta
         "G_PP", "G_PC", "G_PV",
         "PnL %", "Portfolio $", "BTC $", "Trades", "C/V",
         "Div.C%", "Div.V%", "Equilibrio", "PnL/Trade", "MaxDD%"]

_RANKINGS = [
    ("pnl_pct",          18, plt.cm.RdYlGn,
     "Ranking 1 — Máximo PnL%",
     "Prioriza el retorno total · Sin sesgo de tendencia",
     OUT_TABLA_PNL),
    ("btc_value",        20, plt.cm.YlOrRd,
     "Ranking 2 — Máxima Acumulación BTC",
     "Prioriza el valor USD del BTC en posiciones al cierre",
     OUT_TABLA_BTC),
    ("equilibrio_score", 25, plt.cm.PuBuGn,
     "Ranking 3 — Mejor Equilibrio PnL% × BTC",
     "Media geométrica de normas [0,1]  ·  Penaliza extremos en una sola dimensión",
     OUT_TABLA_EQ),
]


def _fig_tabla(df_res, sort_col, highlight_col, cmap, titulo, subtitulo, filename):
    top = df_res.sort_values(sort_col, ascending=False).head(TOP_N).reset_index(drop=True)
    def bs(v): return "✓" if v else "✗"
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append([
            str(rank),
            str(r.rsi_length), str(r.N),
            str(int(r.rsi_buy_trigger)), str(int(r.rsi_sell_trigger)),
            f"{r.floor_pct:.0f}%", f"{r.top_pct:.0f}%",
            f"{r.nivel_c:.0f}", f"{r.infl_c:.2f}", f"{r.k_c:.0f}", f"{r.fac2_c:.0f}",
            f"{r.nivel_v:.0f}", f"{r.infl_v:.2f}", f"{r.k_v:.0f}", f"{r.fac2_v:.0f}",
            bs(r.guardia_compra), bs(r.guardia_prec_compra), bs(r.guardia_prec_venta),
            f"{r.pnl_pct:+.2f}%", f"${r.portfolio_final:,.2f}", f"${r.btc_value:,.2f}",
            str(int(r.total_trades)), f"{int(r.total_compras)}/{int(r.total_ventas)}",
            f"{r.tasa_apr_compra:.0f}%", f"{r.tasa_apr_venta:.0f}%",
            f"{r.equilibrio_score:.4f}", f"{r.pnl_por_trade:+.3f}%", f"{r.max_drawdown:.1f}%",
        ])
    fig, ax = plt.subplots(figsize=(32, TOP_N * 0.43 + 3.5))
    fig.patch.set_facecolor("#f4f6fa"); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=_COLS, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.0); table.scale(1, 1.52)
    for j in range(len(_COLS)):
        c = table[0, j]
        c.set_facecolor("#1a2540"); c.set_text_props(color="white", fontweight="bold")
    # Colorear encabezados de curva compra (cols 6-9) y venta (cols 10-13)
    for j in range(7, 11):
        table[0, j].set_facecolor("#1e4a80")   # compra — azul
    for j in range(11, 15):
        table[0, j].set_facecolor("#1e6b3c")   # venta — verde
    table[0, highlight_col].set_facecolor("#e67e22")
    for j in [23, 24]:
        table[0, j].set_facecolor("#1e6b3c")
    for i in range(1, len(rows) + 1):
        bg = "#eef2f9" if i % 2 == 0 else "#ffffff"
        for j in range(len(_COLS)): table[i, j].set_facecolor(bg)
    for i, color in enumerate(["#ffd700", "#d8d8d8", "#cd7f32"][:min(3, len(rows))], 1):
        for j in range(len(_COLS)):
            table[i, j].set_facecolor(color); table[i, j].set_text_props(fontweight="bold")
    raw_vals = [float(rows[i][highlight_col].replace("$", "").replace(",", "")
                      .replace("%", "").replace("+", "")) for i in range(len(rows))]
    vmin, vmax = min(raw_vals), max(raw_vals)
    span = vmax - vmin if vmax > vmin else 1.0
    for i, v in enumerate(raw_vals, 1):
        table[i, highlight_col].set_facecolor(
            mcolors.to_hex(cmap(0.25 + 0.75 * (v - vmin) / span)))
    ax.set_title(
        f"{titulo}\n{subtitulo}\n"
        f"Período: {FECHA_INICIO} → {FECHA_FIN}  |  Total combinaciones: {len(df_res):,}  |  "
        f"FL%=FLOOR_PCT (piso compra)  ·  TOP%=TOP_PCT (techo venta)  |  "
        f"NC/IC/KC/F2C=curva compra  ·  NV/IV/KV/F2V=curva venta  |  "
        f"G_PP=Guardia precio prom.  ·  G_PC=precio mín. compra  ·  G_PV=precio máx. venta",
        fontsize=9.0, fontweight="bold", color="#1a2540", pad=13,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight", facecolor="#f4f6fa"); plt.close()
    print(f"  ✓ {titulo}: {filename}")


def fig_tres_tablas(df_res):
    for sort_col, hl_col, cmap, titulo, subtitulo, filename in _RANKINGS:
        _fig_tabla(df_res, sort_col, hl_col, cmap, titulo, subtitulo, filename)


def fig_analisis_guardias(df_res):
    GCOLS = ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    rows_tabla = []
    for (gc, gpc, gpv), grp in df_res.groupby(GCOLS):
        bp = grp.loc[grp["pnl_pct"].idxmax()]
        bb = grp.loc[grp["btc_value"].idxmax()]
        be = grp.loc[grp["equilibrio_score"].idxmax()]
        def params(r):
            return (f"RSI={int(r.rsi_length)} N={int(r.N)} "
                    f"BT={int(r.rsi_buy_trigger)} ST={int(r.rsi_sell_trigger)} "
                    f"FL={r.floor_pct:.0f}% TOP={r.top_pct:.0f}% "
                    f"NC={r.nivel_c:.0f}/IC={r.infl_c:.2f}/KC={r.k_c:.0f}/F2C={r.fac2_c:.0f} "
                    f"NV={r.nivel_v:.0f}/IV={r.infl_v:.2f}/KV={r.k_v:.0f}/F2V={r.fac2_v:.0f}")
        rows_tabla.append({
            "gc": gc, "gpc": gpc, "gpv": gpv,
            "n_combos"   : len(grp),
            "pnl_max"    : bp["pnl_pct"],        "pnl_port"    : bp["portfolio_final"],
            "pnl_params" : params(bp),
            "btc_max_val": bb["btc_value"],       "btc_btc"     : bb["btc_posiciones"],
            "btc_params" : params(bb),
            "eq_score"   : be["equilibrio_score"],
            "eq_pnl"     : be["pnl_pct"],         "eq_btc"      : be["btc_value"],
            "eq_params"  : params(be),
            "pnl_median" : grp["pnl_pct"].median(),
            "pnl_positive": (grp["pnl_pct"] > 0).sum(),
        })
    df_g = (pd.DataFrame(rows_tabla)
              .sort_values("pnl_max", ascending=False)
              .reset_index(drop=True))
    fig = plt.figure(figsize=(30, 17)); fig.patch.set_facecolor("#f4f6fa")
    gs = GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[2.8, 1])
    ax_t = fig.add_subplot(gs[0]); ax_t.axis("off")
    def bs(v): return "✓" if v else "✗"
    def lbl(gc, gpc, gpv): return f"GC={bs(gc)} GPC={bs(gpc)} GPV={bs(gpv)}"
    cols_t = ["Guardias\n(GC/GPC/GPV)", "Combos",
              "──── Mejor PnL% ──────────────────────────",
              "PnL%", "Portfolio $", "Params",
              "──── Mejor BTC ───────────────────────────",
              "BTC $", "BTC ₿", "Params",
              "──── Mejor Equilibrio ─────────────────────",
              "Eq.Score", "PnL%", "BTC $", "Params",
              "Mediana\nPnL%", ">0%"]
    rows_t = []
    for _, r in df_g.iterrows():
        rows_t.append([
            lbl(r.gc, r.gpc, r.gpv), f"{int(r.n_combos):,}",
            "", f"{r.pnl_max:+.2f}%", f"${r.pnl_port:,.2f}", r.pnl_params,
            "", f"${r.btc_max_val:,.2f}", f"{r.btc_btc:.6f} ₿", r.btc_params,
            "", f"{r.eq_score:.4f}", f"{r.eq_pnl:+.2f}%", f"${r.eq_btc:,.2f}", r.eq_params,
            f"{r.pnl_median:+.2f}%", f"{int(r.pnl_positive):,}",
        ])
    table = ax_t.table(cellText=rows_t, colLabels=cols_t, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.0); table.scale(1, 1.7)
    for j in range(len(cols_t)):
        c = table[0, j]
        c.set_facecolor("#1a2540"); c.set_text_props(color="white", fontweight="bold")
    for ci, color in zip([2, 6, 10], ["#1e6b3c", "#7b3f00", "#1a3a6b"]):
        table[0, ci].set_facecolor(color)
    for i in range(1, len(rows_t) + 1):
        bg = "#eef2f9" if i % 2 == 0 else "#ffffff"
        for j in range(len(cols_t)): table[i, j].set_facecolor(bg)
    for i, color in enumerate(["#ffd700", "#d8d8d8", "#cd7f32"][:min(3, len(rows_t))], 1):
        table[i, 3].set_facecolor(color); table[i, 3].set_text_props(fontweight="bold")
    table[1, 7].set_facecolor("#ffe4b5"); table[1, 11].set_facecolor("#d4edda")
    for i, (_, r) in enumerate(df_g.iterrows(), 1):
        n_true = sum([r.gc, r.gpc, r.gpv]); intensity = n_true / 3
        color = mcolors.to_hex(plt.cm.RdYlGn(0.3 + 0.7 * intensity))
        table[i, 0].set_facecolor(color); table[i, 0].set_text_props(fontweight="bold")
    ax_t.set_title(
        f"Análisis de Combinaciones de Guardias — {FECHA_INICIO} → {FECHA_FIN}\n"
        f"Cada fila = mejor resultado de las ~{len(df_res) // max(len(df_g), 1):,} combinaciones "
        f"con esa config de guardias  (Total rows: {len(df_res):,}  |  configs presentes: {len(df_g)}/8)\n"
        f"GC=GUARDIA_COMPRA · GPC=G_PRECIO_COMPRA · GPV=G_PRECIO_VENTA",
        fontsize=10, fontweight="bold", color="#1a2540", pad=12,
    )
    ax_b = fig.add_subplot(gs[1]); ax_b.set_facecolor("#ffffff")
    n_g_actual = len(df_g)
    x = np.arange(n_g_actual)
    labels = [lbl(r.gc, r.gpc, r.gpv) for _, r in df_g.iterrows()]
    pnl_m  = [r.pnl_max     for _, r in df_g.iterrows()]
    btc_m  = [r.btc_max_val for _, r in df_g.iterrows()]
    btc_sc = [b / max(btc_m) * max(abs(p) for p in pnl_m) for b in btc_m]
    w = 0.35
    b1 = ax_b.bar(x - w/2, pnl_m, w,
                  color=[plt.cm.RdYlGn(0.3 + 0.7 * (p - min(pnl_m)) /
                          max(max(pnl_m) - min(pnl_m), 0.001)) for p in pnl_m],
                  edgecolor="white", alpha=0.9, label="Mejor PnL%")
    ax_b.bar(x + w/2, btc_sc, w, color="#3498db", alpha=0.65,
             edgecolor="white", label="Mejor BTC$ (normalizado)")
    ax_b.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    ax_b.set_ylabel("Mejor PnL%", fontsize=9)
    ax_b.set_title(
        f"Comparativa entre las {n_g_actual} combinaciones de guardias presentes en el top-{TOP_HEAP}",
        fontsize=9)
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(b1, pnl_m):
        ax_b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + (0.3 if val >= 0 else -1.2),
                  f"{val:+.1f}%", ha="center", fontsize=7,
                  fontweight="bold", color="#1a2540")
    plt.savefig(OUT_GUARDIAS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa"); plt.close()
    print(f"  ✓ Análisis guardias: {OUT_GUARDIAS}")


def fig_analisis_variables(df_res):
    variables = [
        ("rsi_length",       "RSI_LENGTH",            False, None),
        ("N",                "N (ventana)",            False, None),
        ("floor_pct",        "FLOOR_PCT (%) — piso compra",   False, None),
        ("top_pct",          "TOP_PCT (%) — techo venta",     False, None),
        ("nivel_c",          "NIVEL_C  (asíntota compra %)",  False, None),
        ("infl_c",           "INFL_C   (inflexión compra)",   False, None),
        ("k_c",              "K_C      (vel. exp. compra)",   False, None),
        ("fac2_c",           "FAC2_C   (curvatura compra)",   False, None),
        ("nivel_v",          "NIVEL_V  (asíntota venta %)",   False, None),
        ("infl_v",           "INFL_V   (inflexión venta)",    False, None),
        ("k_v",              "K_V      (vel. exp. venta)",    False, None),
        ("fac2_v",           "FAC2_V   (curvatura venta)",    False, None),
        ("rsi_buy_trigger",  "RSI_BUY_TRIGGER",        False, "tasa_apr_compra"),
        ("rsi_sell_trigger", "RSI_SELL_TRIGGER",       False, "tasa_apr_venta"),
        ("guardia_compra",      "GUARDIA_COMPRA",      True,  None),
        ("guardia_prec_compra", "GUARDIA_PRECIO_COMPRA", True, None),
        ("guardia_prec_venta",  "GUARDIA_PRECIO_VENTA",  True, None),
    ]
    n_vars = len(variables)
    n_cols = 2
    n_rows = math.ceil(n_vars / n_cols)
    fig = plt.figure(figsize=(22, n_rows * 4.5)); fig.patch.set_facecolor("#f4f6fa")
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.65, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]
    for ax, (col, label, es_bool, tasa_col) in zip(axes, variables):
        ax.set_facecolor("#ffffff")
        grp     = df_res.groupby(col)["pnl_pct"]
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
            norm_v = medians.values; vmin, vmax = norm_v.min(), norm_v.max()
            span   = vmax - vmin if vmax > vmin else 1
            colors = [mcolors.to_hex(plt.cm.RdYlGn(0.2 + 0.8 * (v - vmin) / span))
                      for v in norm_v]
            x_labels = [str(v) for v in x_vals]
        ax.bar(x_pos, medians.values, color=colors,
               alpha=0.85, edgecolor="white", linewidth=0.7, zorder=3)
        yerr_low  = medians.values - q25.values
        yerr_high = q75.values - medians.values
        ax.errorbar(x_pos, medians.values,
                    yerr=[yerr_low, yerr_high],
                    fmt="none", color="#555", linewidth=1.2, capsize=4, zorder=4)
        ax.scatter(x_pos, tops.values, marker="^",
                   color="#e67e22", s=45, zorder=5, label="Máx PnL%")
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)
        for xi, med in enumerate(medians.values):
            ax.text(xi, med + (q75.values[xi] - med) * 0.15,
                    f"{med:+.1f}%", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#1a2540")
        if tasa_col is not None:
            ax2   = ax.twinx()
            tasas = df_res.groupby(col)[tasa_col].mean().sort_index()
            ax2.plot(x_pos, tasas.values, color="#9b59b6", marker="o",
                     linewidth=1.5, markersize=5, zorder=6, alpha=0.85)
            ax2.set_ylabel("Tasa aprobación (%)", fontsize=7, color="#9b59b6")
            ax2.tick_params(axis="y", labelcolor="#9b59b6", labelsize=7)
            ax2.set_ylim(0, 105)
            ax.legend(["▲ Máx PnL%", "● Tasa aprob."],
                      fontsize=6.5, loc="upper right",
                      handletextpad=0.3, borderpad=0.4)
            ax.set_title(f"{label}  (azul=PnL · violeta=tasa aprob.)",
                         fontsize=9.0, fontweight="bold", color="#1a2540", pad=6)
        else:
            if not es_bool:
                ax.legend(["▲ Máx PnL%"], fontsize=7, loc="upper right",
                          handletextpad=0.3, borderpad=0.4)
            ax.set_title(label, fontsize=9.5, fontweight="bold", color="#1a2540", pad=6)
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("PnL% (mediana ± IQR)", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, color="#dde3ef")
    # Ocultar ejes sobrantes si n_vars es impar
    for ax in axes[n_vars:]:
        ax.set_visible(False)
    fig.suptitle(
        f"Análisis de Impacto por Variable — Divergencia RSI · Gradiente Asintótico\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  |  "
        f"Cada barra = mediana PnL% · barras de error = IQR (Q25–Q75) · ▲ = máximo\n"
        f"Total combinaciones: {len(df_res):,}  |  "
        f"Paneles BUY/SELL_TRIGGER incluyen tasa de aprobación (eje derecho)",
        fontsize=11, fontweight="bold", color="#1a2540", y=1.01,
    )
    plt.savefig(OUT_ANALISIS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa"); plt.close()
    print(f"  ✓ Análisis variables: {OUT_ANALISIS}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(df_res):
    sep = "═" * 82
    print(f"\n{sep}")
    print("  RESUMEN — OPTIMIZADOR GRADIENTE ASINTÓTICO")
    print(sep)
    print(f"  Período       : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Combinaciones : {len(df_res):,}")
    print(f"  PnL% rango    : {df_res['pnl_pct'].min():+.2f}%  →  {df_res['pnl_pct'].max():+.2f}%")
    print(f"  PnL% mediana  : {df_res['pnl_pct'].median():+.2f}%")
    print(f"  PnL% positivos: {(df_res['pnl_pct'] > 0).sum():,}  "
          f"({(df_res['pnl_pct'] > 0).mean()*100:.1f}%)")

    def bs(v): return "✓" if v else "✗"

    hdr = (f"  {'#':>3}  {'RSI':>4}  {'N':>3}  {'BT':>4}  {'ST':>4}  {'FL%':>4}  {'TOP%':>5}  "
           f"{'NC':>4}  {'IC':>5}  {'KC':>4}  {'F2C':>4}  "
           f"{'NV':>4}  {'IV':>5}  {'KV':>4}  {'F2V':>4}  "
           f"{'G_PP':>4}  {'G_PC':>4}  {'G_PV':>4}  "
           f"{'PnL%':>8}  {'BTC$':>9}  {'Eq':>7}  {'C':>3}  {'V':>3}  "
           f"{'DC%':>5}  {'DV%':>5}  {'DD%':>6}")
    rankings = [
        ("RANKING 1 — MÁXIMO PnL%",     df_res.sort_values("pnl_pct",         ascending=False)),
        ("RANKING 2 — MÁXIMO BTC$",      df_res.sort_values("btc_value",        ascending=False)),
        ("RANKING 3 — MEJOR EQUILIBRIO", df_res.sort_values("equilibrio_score", ascending=False)),
    ]
    for titulo, ranked in rankings:
        print(f"\n  {'─'*80}")
        print(f"  {titulo}")
        print(f"  {'─'*80}")
        print(hdr)
        print(f"  {'─'*80}")
        for rank, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
            marker = "★" if rank <= 3 else " "
            print(f"  {marker}{rank:>2}.  "
                  f"{r.rsi_length:>4}  {r.N:>3}  "
                  f"{int(r.rsi_buy_trigger):>4}  {int(r.rsi_sell_trigger):>4}  "
                  f"{r.floor_pct:>3.0f}%  {r.top_pct:>4.0f}%  "
                  f"{r.nivel_c:>4.0f}  {r.infl_c:>5.2f}  {r.k_c:>4.0f}  {r.fac2_c:>4.0f}  "
                  f"{r.nivel_v:>4.0f}  {r.infl_v:>5.2f}  {r.k_v:>4.0f}  {r.fac2_v:>4.0f}  "
                  f"{bs(r.guardia_compra):>4}  {bs(r.guardia_prec_compra):>4}  "
                  f"{bs(r.guardia_prec_venta):>4}  "
                  f"{r.pnl_pct:>+7.2f}%  ${r.btc_value:>8,.2f}  "
                  f"{r.equilibrio_score:>7.4f}  "
                  f"{int(r.total_compras):>3}  {int(r.total_ventas):>3}  "
                  f"{r.tasa_apr_compra:>4.0f}%  {r.tasa_apr_venta:>4.0f}%  "
                  f"{r.max_drawdown:>5.1f}%")

    for titulo, ranked in rankings:
        top10 = ranked.head(max(1, len(df_res) // 10))
        print(f"\n  Dominantes — {titulo}")
        for col, lbl_ in [
            ("rsi_length",       "RSI"),
            ("N",                "N"),
            ("rsi_buy_trigger",  "BUY_T"),
            ("rsi_sell_trigger", "SELL_T"),
            ("floor_pct",        "FLOOR%"),
            ("top_pct",          "TOP%"),
            ("nivel_c",          "NIVEL_C"),
            ("infl_c",           "INFL_C"),
            ("k_c",              "K_C"),
            ("fac2_c",           "FAC2_C"),
            ("nivel_v",          "NIVEL_V"),
            ("infl_v",           "INFL_V"),
            ("k_v",              "K_V"),
            ("fac2_v",           "FAC2_V"),
        ]:
            val  = top10[col].mode().iloc[0]
            freq = (top10[col] == val).mean() * 100
            print(f"    {lbl_:<14}: {val}  ({freq:.0f}% del top 10%)")
        for col, lbl_ in [("guardia_compra",      "G_COMPRA"),
                           ("guardia_prec_compra", "G_PREC_C"),
                           ("guardia_prec_venta",  "G_PREC_V")]:
            pct = top10[col].mean() * 100
            print(f"    {lbl_:<14}: {'✓ True' if pct >= 50 else '✗ False'}  ({pct:.0f}% True)")
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMIZADOR — DIVERGENCIA RSI · GRADIENTE ASINTÓTICO  v3           ║")
    print("║  FLOOR_PCT (compra) · TOP_PCT (venta) independientes                ║")
    print("║  NumPy Vectorizado  ×  Señales Precomputadas  ×  Heaps top-K        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    t_total = time.time()

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

    print(f"\n{'═'*72}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*72}")
    for fname in [OUT_CSV, OUT_JSON, OUT_TABLA_PNL, OUT_TABLA_BTC,
                  OUT_TABLA_EQ, OUT_GUARDIAS, OUT_ANALISIS]:
        print(f"  · {fname}")
    print(f"{'═'*72}")
    print(f"✓ Proceso total completado en {time.time() - t_total:.1f}s\n")


if __name__ == "__main__":
    main()
