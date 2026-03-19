"""
Optimizador — Divergencia RSI Zona V2  [Multiprocessing]
══════════════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Grid Search Exhaustivo
Estrategia: Divergencia RSI · Zona de Orden en Tiempo Real (sin lookahead)

VARIABLES OPTIMIZADAS (11)
──────────────────────────────────────────────────────────────────────────────
  RSI_LENGTH            período del RSI de Wilder
  N                     ventana de búsqueda del extremo local
  RSI_BUY_TRIGGER       RSI máximo del âncla para divergencia de compra
  RSI_SELL_TRIGGER      RSI mínimo del âncla para divergencia de venta
  FLOOR_PCT             piso teórico del ciclo como % del ATH
  FACTOR_CAIDA          curvatura del gradiente de compra
  FACTOR_SUBIDA         curvatura del gradiente de venta
  GUARDIA_COMPRA        bloquear compras por encima del PP
  USDT_RESERVA_PCT      % del capital que nunca se opera
  BTC_PCT_TO_ACCUMULATE % del BTC de cada venta que se acumula
  PROF_ZONA_PCT         profundidad de la orden dentro de la zona válida

VARIABLES FIJAS
──────────────────────────────────────────────────────────────────────────────
  GUARDIA_PRECIO_COMPRA = False   GUARDIA_PRECIO_VENTA = False
  COMMISSION_PCT  (leído de config)
  DB_PATH / FECHA_INICIO / FECHA_FIN  (leídos de config)

ARQUITECTURA
──────────────────────────────────────────────────────────────────────────────
· Precomputación de RSI: los arrays rsi_low, avg_gain_low, avg_loss_low,
  rsi_high, avg_gain_high, avg_loss_high se calculan UNA SOLA VEZ por cada
  valor único de RSI_LENGTH, evitando el cálculo redundante más costoso.

· Paralelización: multiprocessing.Pool con todos los cores disponibles.
  Cada worker recibe los arrays precomputados por su RSI_LENGTH y ejecuta
  el backtest completo de forma independiente.

· Tres min-heaps: mejor PnL%, mayor BTC total, mayor portfolio.

NOTA: GUARDIA_PRECIO_COMPRA y GUARDIA_PRECIO_VENTA se fijan en False para
mantener el espacio en ~12k combinaciones. Pueden activarse editando
GUARDIA_PRECIO_VALS en la sección ESPACIO DE BÚSQUEDA.

ARCHIVOS DE SALIDA
──────────────────────────────────────────────────────────────────────────────
  optimizacion_div_rsi_v2.csv
  optimizacion_div_rsi_v2_top.json
  optimizacion_div_rsi_v2_ranking_pnl.png
  optimizacion_div_rsi_v2_ranking_btc.png
  optimizacion_div_rsi_v2_ranking_port.png
  optimizacion_div_rsi_v2_analisis.png
  optimizacion_div_rsi_v2_scatter.png
"""

import sqlite3, json, math, os, time
import heapq as _heapq
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from itertools import product
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL, COMMISSION_PCT,
    )
    print("✓ config.py cargado")
except ImportError:
    print("⚠  config.py no encontrado — usando valores por defecto")
    DB_PATH            = r"btc_hourly.db"
    FECHA_INICIO       = "2021-11-10"
    FECHA_FIN          = "2022-11-22"
    SALDO_USDT_INICIAL = 1000
    COMMISSION_PCT     = 0.1

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ══════════════════════════════════════════════════════════════════════════════
# ESPACIO DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════
#
#  Variable               valores                                vals
#  ──────────────────────────────────────────────────────────────────
#  RSI_LENGTH             3 · 5 · 7                               3
#  N                      10 · 15 · 20                            3
#  RSI_BUY_TRIGGER        5 · 10 · 20                             3
#  RSI_SELL_TRIGGER       50 · 60 · 70                            3
#  FLOOR_PCT              15 · 25                                 2
#  FACTOR_CAIDA           2.0 · 4.0                               2
#  FACTOR_SUBIDA          0.25 · 0.5 · 1.0                        3
#  PCT_ATH_PROYECTADO     100 · 200 · 300 · 400                   4
#  GUARDIA_COMPRA         True · False                            2
#  GUARDIA_PRECIO_COMPRA  True · False                            2
#  GUARDIA_PRECIO_VENTA   True · False                            2
#  USDT_RESERVA_PCT       0 · 5                                   2
#  BTC_PCT_TO_ACCUMULATE  0 · 5                                   2
#  PROF_ZONA_PCT          0 · 10 · 25                             3
#
#  Total: 3×3×3×3×2×2×3×4×2×2×2×2×2×3 = 373 248 combinaciones  (~4h)
# ══════════════════════════════════════════════════════════════════════════════

RSI_LENGTH_VALS       = [5, 7, 14]
N_VALS                = [10, 20, 30]
RSI_BUY_VALS          = [10, 20, 30]
RSI_SELL_VALS         = [70, 80, 90]
FLOOR_PCT_VALS        = [15, 25]
FACTOR_CAIDA_VALS     = [1, 2.5, 3.5]
FACTOR_SUBIDA_VALS    = [0.25, 0.5, 1.5]
PCT_ATH_PROY_VALS     = [350, 500, 750]
GUARDIA_COMPRA_VALS        = [True, False]
GUARDIA_PRECIO_COMPRA_VALS = [True, False]
GUARDIA_PRECIO_VENTA_VALS  = [True, False]
USDT_RESERVA_VALS          = [0]
BTC_ACUMULA_VALS      = [0, 1]
PROF_ZONA_VALS        = [0.0, 5.0]

# Configuración del optimizador
TOP_N    = 25
TOP_HEAP = 500

# Archivos de salida
OUT_CSV       = "optimizacion_div_rsi_v2.csv"
OUT_JSON      = "optimizacion_div_rsi_v2_top.json"
OUT_TABLA_PNL = "optimizacion_div_rsi_v2_ranking_pnl.png"
OUT_TABLA_BTC = "optimizacion_div_rsi_v2_ranking_btc.png"
OUT_TABLA_PRT = "optimizacion_div_rsi_v2_ranking_port.png"
OUT_ANALISIS  = "optimizacion_div_rsi_v2_analisis.png"
OUT_SCATTER   = "optimizacion_div_rsi_v2_scatter.png"

COL_ORDER = [
    "rsi_length", "n_ventana", "rsi_buy_trigger", "rsi_sell_trigger",
    "floor_pct", "factor_caida", "factor_subida", "pct_ath_proyectado",
    "guardia_compra", "guardia_precio_compra", "guardia_precio_venta",
    "usdt_reserva_pct", "btc_acumula_pct", "prof_zona_pct",
    "pnl_pct", "portfolio_final", "usdt_final",
    "btc_libre", "btc_en_pos", "btc_total",
    "n_pos_final", "total_compras", "total_ventas", "total_trades",
    "ordenes_canceladas", "max_drawdown",
]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datos() -> pd.DataFrame:
    conn  = sqlite3.connect(DB_PATH)
    query = (f"SELECT timestamp, open, high, low, close "
             f"FROM {DB_TABLE} ORDER BY timestamp ASC")
    df    = pd.read_sql(query, conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if FECHA_INICIO:
        df = df[df["datetime"] >= pd.to_datetime(FECHA_INICIO)]
    if FECHA_FIN:
        df = df[df["datetime"] <= pd.to_datetime(FECHA_FIN)]
    df = df.reset_index(drop=True)
    n_nan = df[["open", "high", "low", "close"]].isna().sum().sum()
    if n_nan:
        print(f"  ⚠  {n_nan} NaN — eliminando filas afectadas")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTACIÓN DE RSI
# ══════════════════════════════════════════════════════════════════════════════

def _calcular_rsi_con_estado(series: pd.Series, length: int):
    """Retorna (rsi, avg_gain, avg_loss) como arrays numpy."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    alpha    = 1.0 / length
    avg_gain = gain.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi.values, avg_gain.values, avg_loss.values


def precomputar_rsi(df: pd.DataFrame) -> dict:
    """
    Calcula los arrays RSI, avg_gain, avg_loss para cada RSI_LENGTH único.
    Retorna un dict: {length: (rsi_low, ag_low, al_low, rsi_high, ag_high, al_high)}
    """
    cache = {}
    for length in set(RSI_LENGTH_VALS):
        rl, agl, all_ = _calcular_rsi_con_estado(df["low"],  length)
        rh, agh, alh  = _calcular_rsi_con_estado(df["high"], length)
        cache[length] = (rl, agl, all_, rh, agh, alh)
        print(f"  RSI precomputado length={length}")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# MIN-HEAP DE TAMAÑO FIJO
# ══════════════════════════════════════════════════════════════════════════════

class _TopHeap:
    __slots__ = ("n", "_heap", "_counter")

    def __init__(self, n: int):
        self.n = n; self._heap = []; self._counter = 0

    @property
    def min_val(self) -> float:
        return self._heap[0][0] if self._heap else -math.inf

    def push_if_better(self, val: float, row: dict):
        if len(self._heap) < self.n:
            _heapq.heappush(self._heap, (val, self._counter, row))
            self._counter += 1
        elif val > self.min_val:
            _heapq.heapreplace(self._heap, (val, self._counter, row))
            self._counter += 1

    def to_list(self) -> list:
        return [e[2] for e in self._heap]


# ══════════════════════════════════════════════════════════════════════════════
# SIMULADOR POR COMBINACIÓN (función de nivel de módulo — necesario para mp)
# ══════════════════════════════════════════════════════════════════════════════

# Variables globales del worker (inicializadas en el proceso hijo)
_W_LOWS   = None
_W_HIGHS  = None
_W_CLOSES = None
_W_RSI    = None   # dict {length: (rl,agl,all_,rh,agh,alh)}
_W_SALDO  = None
_W_COMM   = None


def _init_worker(lows, highs, closes, rsi_cache, saldo, comm):
    global _W_LOWS, _W_HIGHS, _W_CLOSES, _W_RSI, _W_SALDO, _W_COMM
    _W_LOWS   = lows
    _W_HIGHS  = highs
    _W_CLOSES = closes
    _W_RSI    = rsi_cache
    _W_SALDO  = saldo
    _W_COMM   = comm


def _x_umbral_compra(G, L, p, Ra, alpha):
    if Ra <= 0:   return 0.0
    if Ra >= 100: return float('inf')
    k  = (1.0 - alpha) / alpha
    RS = Ra / (100.0 - Ra)
    return p + k * L - k * G / RS


def _x_umbral_venta(G, L, p, Ra, alpha):
    if Ra >= 100: return float('inf')
    if Ra <= 0:   return 0.0
    k  = (1.0 - alpha) / alpha
    RS = Ra / (100.0 - Ra)
    return p + k * RS * L - k * G


def _run_combo(combo: tuple) -> dict:
    """
    Ejecuta un backtest completo para la combinación dada.
    Usa los arrays precomputados en las variables globales del worker.
    """
    (rsi_length, n_vent, rsi_buy, rsi_sell,
     floor_pct, f_caida, f_subida, pct_ath_proy,
     guardia_compra, guardia_precio_compra, guardia_precio_venta,
     usdt_reserva_pct, btc_acumula_pct, prof_zona_pct) = combo

    lows   = _W_LOWS
    highs  = _W_HIGHS
    closes = _W_CLOSES
    saldo  = _W_SALDO
    comm   = _W_COMM / 100.0
    n      = len(lows)
    alpha  = 1.0 / rsi_length

    rsi_l, avg_gl, avg_ll, rsi_h, avg_gh, avg_lh = _W_RSI[rsi_length]

    usdt_reserva  = saldo * usdt_reserva_pct / 100.0
    log_rango_c   = math.log(100.0 / floor_pct) if floor_pct > 0 else 0.0

    # Estado financiero
    usdt_bal  = float(saldo)
    btc_libre = 0.0
    btc_pos   = 0.0
    usdt_inv  = 0.0
    pos_cnt   = 0

    precio_min_c = math.inf
    precio_max_v = 0.0
    ath = float(np.max(highs[:n_vent]))
    atl = float(np.min(lows[:n_vent]))

    total_compras   = 0
    total_ventas    = 0
    ord_canceladas  = 0
    orden_pendiente = None   # dict o None

    peak_port = float(saldo)
    max_dd    = 0.0

    for i in range(n_vent, n):
        # ATH/ATL con datos de la vela anterior
        if i > n_vent:
            if highs[i-1] > ath: ath = float(highs[i-1])
            if lows[i-1]  < atl: atl = float(lows[i-1])

        if math.isnan(rsi_l[i]) or math.isnan(rsi_h[i]):
            orden_pendiente = None
            continue

        precio_prom = usdt_inv / btc_pos if btc_pos > 0 else 0.0

        # ── PASO 1: Evaluar orden pendiente ──────────────────────────────────
        if orden_pendiente is not None:
            op = orden_pendiente
            orden_pendiente = None

            if op["tipo"] == "BUY":
                if lows[i] <= op["xu"]:
                    ord_canceladas += 1
                elif lows[i] <= op["po"]:
                    precio_ej = op["po"]
                    usdt_disp = usdt_bal - usdt_reserva
                    precio_min_ok = (not guardia_precio_compra or
                                    precio_min_c == math.inf or
                                    precio_ej < precio_min_c)
                    if usdt_disp > 0 and precio_min_ok and (not guardia_compra or btc_pos == 0 or precio_ej < precio_prom):
                        # Gradiente de compra
                        if log_rango_c > 0 and ath > 0 and precio_ej > 0:
                            pos = math.log(ath / precio_ej) / log_rango_c
                            pos = max(0.0, min(1.0, pos))
                            pct = (pos ** f_caida) * 100.0
                        else:
                            pct = 0.0
                        usdt_a = usdt_disp * pct / 100.0
                        if usdt_a > 0:
                            btc_adq    = (usdt_a - usdt_a * comm) / precio_ej
                            usdt_bal  -= usdt_a
                            btc_pos   += btc_adq
                            usdt_inv  += usdt_a
                            pos_cnt   += 1
                            precio_prom = usdt_inv / btc_pos
                            if precio_ej < precio_min_c:
                                precio_min_c = precio_ej
                            total_compras += 1

            elif op["tipo"] == "SELL" and btc_pos > 0:
                xu_v   = op["xu"]
                dentro = (highs[i] >= op["po"] and
                          (math.isinf(xu_v) or highs[i] < xu_v))
                if not math.isinf(xu_v) and highs[i] >= xu_v:
                    ord_canceladas += 1
                elif dentro:
                    precio_ej = op["po"]
                    precio_prom = usdt_inv / btc_pos if btc_pos > 0 else 0.0
                    ath_proy = atl * (1.0 + pct_ath_proy / 100.0)
                    if precio_ej > precio_prom and precio_prom > 0 and ath_proy > precio_prom:
                        log_amp = math.log(ath_proy / precio_prom)
                        pos_v   = math.log(precio_ej / precio_prom) / log_amp
                        pos_v   = max(0.0, min(1.0, pos_v))
                        pct_v   = (pos_v ** f_subida) * 100.0
                        btc_slot = btc_pos * pct_v / 100.0
                        precio_max_ok = (not guardia_precio_venta or
                                        precio_max_v == 0.0 or
                                        precio_ej > precio_max_v)
                        if btc_slot > 0 and precio_max_ok:
                            btc_acc    = btc_slot * btc_acumula_pct / 100.0
                            btc_vend   = btc_slot - btc_acc
                            usdt_bruto = btc_vend * precio_ej
                            usdt_neto  = usdt_bruto - usdt_bruto * comm
                            costo_prop = usdt_inv * (btc_slot / btc_pos)
                            btc_pos   -= btc_slot
                            btc_libre += btc_acc
                            usdt_bal  += usdt_neto
                            usdt_inv  -= costo_prop
                            usdt_inv   = max(usdt_inv, 0.0)
                            pos_cnt   -= 1
                            precio_prom = usdt_inv / btc_pos if btc_pos > 0 else 0.0
                            if precio_ej > precio_max_v:
                                precio_max_v = precio_ej
                            total_ventas += 1

        # ── PASO 2: Detectar nueva señal en vela i → orden para vela i+1 ────
        nueva_orden = None
        window_l = lows[i-n_vent+1  : i+1]
        window_h = highs[i-n_vent+1 : i+1]

        # COMPRA — divergencia alcista
        idx_min_rel = int(window_l.argmin())
        if idx_min_rel < len(window_l) - 1:
            idx_anc = i - n_vent + 1 + idx_min_rel
            Ra_c    = float(rsi_l[idx_anc])
            rsi_ic  = float(rsi_l[i])
            if not math.isnan(Ra_c) and rsi_ic > Ra_c and Ra_c <= rsi_buy:
                pa_c = float(lows[idx_anc])
                # Calcular zona de compra
                G_c  = float(avg_gl[i]); L_c = float(avg_ll[i])
                xu_c = _x_umbral_compra(G_c, L_c, float(lows[i]), Ra_c, alpha)
                if xu_c < pa_c and xu_c >= 0:
                    po_c = pa_c * (1.0 - prof_zona_pct / 100.0)
                    po_c = max(po_c, xu_c + 1e-8)
                    po_c = min(po_c, pa_c - 1e-8)
                    nueva_orden = {"tipo": "BUY",  "po": po_c, "xu": xu_c}

        # VENTA — divergencia bajista (solo si no hay señal de compra)
        if nueva_orden is None:
            idx_max_rel = int(window_h.argmax())
            if idx_max_rel < len(window_h) - 1:
                idx_anc = i - n_vent + 1 + idx_max_rel
                Ra_v    = float(rsi_h[idx_anc])
                rsi_iv  = float(rsi_h[i])
                if not math.isnan(Ra_v) and rsi_iv < Ra_v and Ra_v >= rsi_sell:
                    pa_v = float(highs[idx_anc])
                    G_v  = float(avg_gh[i]); L_v = float(avg_lh[i])
                    xu_v = _x_umbral_venta(G_v, L_v, float(highs[i]), Ra_v, alpha)
                    if xu_v > pa_v:
                        po_v = pa_v * (1.0 + prof_zona_pct / 100.0)
                        if not math.isinf(xu_v):
                            po_v = min(po_v, xu_v - 1e-8)
                        po_v = max(po_v, pa_v + 1e-8)
                        nueva_orden = {"tipo": "SELL", "po": po_v, "xu": xu_v}

        if nueva_orden is not None:
            orden_pendiente = nueva_orden

        # Mark-to-market y drawdown
        btc_total_i = btc_libre + btc_pos
        port_i      = usdt_bal + btc_total_i * float(closes[i])
        if port_i > peak_port: peak_port = port_i
        if peak_port > 0:
            dd = (peak_port - port_i) / peak_port * 100.0
            if dd > max_dd: max_dd = dd

    # Métricas finales
    precio_final  = float(closes[-1])
    btc_total_f   = btc_libre + btc_pos
    portfolio_f   = usdt_bal + btc_total_f * precio_final
    pnl_pct       = (portfolio_f - saldo) / saldo * 100.0

    return {
        "rsi_length"       : rsi_length,
        "n_ventana"        : n_vent,
        "rsi_buy_trigger"  : rsi_buy,
        "rsi_sell_trigger" : rsi_sell,
        "floor_pct"          : floor_pct,
        "factor_caida"       : f_caida,
        "factor_subida"      : f_subida,
        "pct_ath_proyectado" : pct_ath_proy,
        "guardia_compra"          : guardia_compra,
        "guardia_precio_compra"   : guardia_precio_compra,
        "guardia_precio_venta"    : guardia_precio_venta,
        "usdt_reserva_pct"        : usdt_reserva_pct,
        "btc_acumula_pct"  : btc_acumula_pct,
        "prof_zona_pct"    : prof_zona_pct,
        "pnl_pct"          : round(pnl_pct,       4),
        "portfolio_final"  : round(portfolio_f,    2),
        "usdt_final"       : round(usdt_bal,        2),
        "btc_libre"        : round(btc_libre,       8),
        "btc_en_pos"       : round(btc_pos,         8),
        "btc_total"        : round(btc_total_f,     8),
        "n_pos_final"      : pos_cnt,
        "total_compras"    : total_compras,
        "total_ventas"     : total_ventas,
        "total_trades"     : total_compras + total_ventas,
        "ordenes_canceladas": ord_canceladas,
        "max_drawdown"     : round(max_dd,          2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame, rsi_cache: dict) -> pd.DataFrame:

    combos = list(product(
        RSI_LENGTH_VALS, N_VALS, RSI_BUY_VALS, RSI_SELL_VALS,
        FLOOR_PCT_VALS, FACTOR_CAIDA_VALS, FACTOR_SUBIDA_VALS, PCT_ATH_PROY_VALS,
        GUARDIA_COMPRA_VALS, GUARDIA_PRECIO_COMPRA_VALS, GUARDIA_PRECIO_VENTA_VALS,
        USDT_RESERVA_VALS, BTC_ACUMULA_VALS, PROF_ZONA_VALS,
    ))
    M = len(combos)

    print(f"\n{'═'*72}")
    print(f"  GRID SEARCH — {M:,} combinaciones")
    print(f"{'═'*72}")
    print(f"  RSI_LENGTH          : {RSI_LENGTH_VALS}  ({len(RSI_LENGTH_VALS)} vals)")
    print(f"  N                   : {N_VALS}  ({len(N_VALS)} vals)")
    print(f"  RSI_BUY_TRIGGER     : {RSI_BUY_VALS}  ({len(RSI_BUY_VALS)} vals)")
    print(f"  RSI_SELL_TRIGGER    : {RSI_SELL_VALS}  ({len(RSI_SELL_VALS)} vals)")
    print(f"  FLOOR_PCT           : {FLOOR_PCT_VALS}  ({len(FLOOR_PCT_VALS)} vals)")
    print(f"  FACTOR_CAIDA        : {FACTOR_CAIDA_VALS}  ({len(FACTOR_CAIDA_VALS)} vals)")
    print(f"  FACTOR_SUBIDA       : {FACTOR_SUBIDA_VALS}  ({len(FACTOR_SUBIDA_VALS)} vals)")
    print(f"  PCT_ATH_PROYECTADO  : {PCT_ATH_PROY_VALS}  ({len(PCT_ATH_PROY_VALS)} vals)")
    print(f"  GUARDIA_COMPRA      : {GUARDIA_COMPRA_VALS}  ({len(GUARDIA_COMPRA_VALS)} vals)")
    print(f"  GUARDIA_PRECIO_C    : {GUARDIA_PRECIO_COMPRA_VALS}  ({len(GUARDIA_PRECIO_COMPRA_VALS)} vals)")
    print(f"  GUARDIA_PRECIO_V    : {GUARDIA_PRECIO_VENTA_VALS}  ({len(GUARDIA_PRECIO_VENTA_VALS)} vals)")
    print(f"  USDT_RESERVA_PCT    : {USDT_RESERVA_VALS}  ({len(USDT_RESERVA_VALS)} vals)")
    print(f"  BTC_PCT_ACCUMULATE  : {BTC_ACUMULA_VALS}  ({len(BTC_ACUMULA_VALS)} vals)")
    print(f"  PROF_ZONA_PCT       : {PROF_ZONA_VALS}  ({len(PROF_ZONA_VALS)} vals)")
    print(f"  Top-K por métrica   : {TOP_HEAP}  (máx ~{TOP_HEAP*3:,} filas en disco)")
    print(f"{'═'*72}\n")

    n_cores = mp.cpu_count()
    print(f"  Modo  : multiprocessing — {n_cores} cores")
    print(f"  RSI_LENGTH únicos precomputados: {sorted(rsi_cache.keys())}\n")

    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)

    heap_pnl  = _TopHeap(TOP_HEAP)
    heap_btc  = _TopHeap(TOP_HEAP)
    heap_port = _TopHeap(TOP_HEAP)

    t0          = time.time()
    combos_done = 0
    best_pnl    = -999.0
    all_pnl     = np.empty(M, dtype=np.float64)
    all_btc     = np.empty(M, dtype=np.float64)
    all_port    = np.empty(M, dtype=np.float64)

    # Procesar en lotes para mostrar progreso sin overhead excesivo
    BATCH    = max(1, M // 50)   # ~50 actualizaciones de progreso
    n_lotes  = math.ceil(M / BATCH)

    with mp.Pool(
        processes=n_cores,
        initializer=_init_worker,
        initargs=(lows, highs, closes, rsi_cache, SALDO_USDT_INICIAL, COMMISSION_PCT),
    ) as pool:
        for lote_idx in range(n_lotes):
            s = lote_idx * BATCH
            e = min(s + BATCH, M)
            lote = combos[s:e]

            resultados = pool.map(_run_combo, lote)

            for idx, r in enumerate(resultados):
                gi = s + idx
                pnl  = r["pnl_pct"]
                btc  = r["btc_total"]
                port = r["portfolio_final"]
                all_pnl[gi]  = pnl
                all_btc[gi]  = btc
                all_port[gi] = port

                heap_pnl.push_if_better(pnl,  r)
                heap_btc.push_if_better(btc,   r)
                heap_port.push_if_better(port, r)

            combos_done += len(lote)
            best_pnl     = max(best_pnl, float(max(r["pnl_pct"] for r in resultados)))
            elapsed      = time.time() - t0
            eta          = elapsed / combos_done * (M - combos_done) if combos_done else 0
            survivors    = len({id(r) for h in (heap_pnl, heap_btc, heap_port)
                                for r in h.to_list()})
            print(f"  [{lote_idx+1:>3}/{n_lotes}]  {combos_done:>7,}/{M:,}  "
                  f"{elapsed:>6.1f}s  ETA:{eta:>5.1f}s  "
                  f"top:{survivors:>4}  mejor PnL:{best_pnl:>+8.2f}%")

    elapsed_total = time.time() - t0
    print(f"\n✓ Completado en {elapsed_total:.1f}s  "
          f"({M / elapsed_total:,.0f} backtests/s)")

    # Unión deduplicada de los tres heaps
    seen, all_rows = set(), []
    for heap in (heap_pnl, heap_btc, heap_port):
        for row in heap.to_list():
            key = (row["rsi_length"], row["n_ventana"], row["rsi_buy_trigger"],
                   row["rsi_sell_trigger"], row["floor_pct"], row["factor_caida"],
                   row["factor_subida"], row["pct_ath_proyectado"],
                   row["guardia_compra"],
                   row["guardia_precio_compra"], row["guardia_precio_venta"],
                   row["usdt_reserva_pct"], row["btc_acumula_pct"],
                   row["prof_zona_pct"])
            if key not in seen:
                seen.add(key); all_rows.append(row)

    print(f"\n  ✓ Filas únicas: {len(all_rows):,}  (máx teórico: {TOP_HEAP*3:,})")
    print(f"\n  Estadísticas sobre {M:,} combinaciones:")
    print(f"    PnL%       : {all_pnl.min():+.2f}%  →  {all_pnl.max():+.2f}%"
          f"  (mediana {np.median(all_pnl):+.2f}%)")
    print(f"    BTC total  : {all_btc.min():.6f}  →  {all_btc.max():.6f} ₿")
    print(f"    Portfolio  : ${all_port.min():,.2f}  →  ${all_port.max():,.2f}")
    print(f"    PnL% > 0   : {(all_pnl > 0).sum():,}  "
          f"({(all_pnl > 0).mean()*100:.1f}%)")

    df_res = pd.DataFrame(all_rows)
    for col in COL_ORDER:
        if col not in df_res.columns:
            df_res[col] = None
    return df_res[COL_ORDER].sort_values("pnl_pct", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _agregar_scores(df_res: pd.DataFrame, precio_final: float) -> pd.DataFrame:
    df = df_res.copy()
    df["btc_total_usd"] = df["btc_total"] * precio_final
    for col, nc in [("pnl_pct","pnl_norm"),("btc_total","btc_norm"),("portfolio_final","port_norm")]:
        mn, mx = df[col].min(), df[col].max()
        sp = mx - mn if mx > mn else 1.0
        df[nc] = (df[col] - mn) / sp
    df["combo_score"] = df["pnl_norm"]*0.5 + df["btc_norm"]*0.3 + df["port_norm"]*0.2
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados(df_res: pd.DataFrame):
    df_res.to_csv(OUT_CSV, index=False)
    print(f"  ✓ CSV  : {OUT_CSV}  ({len(df_res):,} filas)")

    top_json = {
        "meta": {
            "fecha_inicio"    : FECHA_INICIO,
            "fecha_fin"       : FECHA_FIN,
            "total_filas"     : len(df_res),
            "rsi_length_vals" : RSI_LENGTH_VALS,
            "n_vals"          : N_VALS,
            "rsi_buy_vals"    : RSI_BUY_VALS,
            "rsi_sell_vals"   : RSI_SELL_VALS,
            "floor_pct_vals"  : FLOOR_PCT_VALS,
            "factor_caida_vals"   : FACTOR_CAIDA_VALS,
            "factor_subida_vals"      : FACTOR_SUBIDA_VALS,
            "pct_ath_proy_vals"          : PCT_ATH_PROY_VALS,
            "guardia_compra_vals"        : GUARDIA_COMPRA_VALS,
            "guardia_precio_compra_vals" : GUARDIA_PRECIO_COMPRA_VALS,
            "guardia_precio_venta_vals"  : GUARDIA_PRECIO_VENTA_VALS,
            "usdt_reserva_vals"          : USDT_RESERVA_VALS,
            "btc_acumula_vals"    : BTC_ACUMULA_VALS,
            "prof_zona_vals"      : PROF_ZONA_VALS,
            "top_heap"            : TOP_HEAP,
        },
        "ranking_pnl"  : df_res.sort_values("pnl_pct",         ascending=False).head(TOP_N).to_dict(orient="records"),
        "ranking_btc"  : df_res.sort_values("btc_total",        ascending=False).head(TOP_N).to_dict(orient="records"),
        "ranking_port" : df_res.sort_values("portfolio_final",  ascending=False).head(TOP_N).to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(top_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ JSON : {OUT_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

_BG   = "#f4f6fa"
_DARK = "#1a2540"

_COLS_TABLA = [
    "#", "RSI_L", "N", "BUY_T", "SELL_T", "FL%", "F_C", "F_S",
    "ATH%", "GC", "GPC", "GPV", "URES%", "BTC%", "PROF%",
    "PnL%", "Port $", "BTC Lib ₿", "BTC Tot ₿",
    "C", "V", "Canc", "DD%",
]

_RANKINGS = [
    ("pnl_pct",        15, plt.cm.RdYlGn,
     "Ranking 1 — Mejor PnL%",
     "Retorno total sobre portfolio valorizado al cierre",
     OUT_TABLA_PNL),
    ("btc_total",      18, plt.cm.YlOrRd,
     "Ranking 2 — Mayor BTC Total",
     "BTC libre + BTC en posiciones abiertas al cierre",
     OUT_TABLA_BTC),
    ("portfolio_final",16, plt.cm.PuBuGn,
     "Ranking 3 — Mayor Portfolio Final ($)",
     "USDT + BTC×precio_cierre",
     OUT_TABLA_PRT),
]


def _fila_tabla(rank: int, r) -> list:
    return [
        str(rank),
        str(int(r.rsi_length)),
        str(int(r.n_ventana)),
        str(int(r.rsi_buy_trigger)),
        str(int(r.rsi_sell_trigger)),
        f"{int(r.floor_pct)}%",
        f"{r.factor_caida:.1f}",
        f"{r.factor_subida:.2f}",
        f"{int(r.pct_ath_proyectado)}%",
        "T" if r.guardia_compra else "F",
        "T" if r.guardia_precio_compra else "F",
        "T" if r.guardia_precio_venta else "F",
        f"{int(r.usdt_reserva_pct)}%",
        f"{int(r.btc_acumula_pct)}%",
        f"{r.prof_zona_pct:.0f}%",
        f"{r.pnl_pct:+.2f}%",
        f"${r.portfolio_final:,.2f}",
        f"{r.btc_libre:.6f}",
        f"{r.btc_total:.6f}",
        str(int(r.total_compras)),
        str(int(r.total_ventas)),
        str(int(r.ordenes_canceladas)),
        f"{r.max_drawdown:.1f}%",
    ]


def _fig_tabla(df_res: pd.DataFrame, sort_col: str, hi_idx: int,
               cmap, titulo: str, subtitulo: str, out_path: str):
    ranked = df_res.sort_values(sort_col, ascending=False).head(TOP_N)
    rows   = [_fila_tabla(rk+1, r) for rk, (_, r) in enumerate(ranked.iterrows())]

    fig_h = max(5.0, 0.38 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(28, fig_h))
    fig.patch.set_facecolor(_BG); ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=_COLS_TABLA,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.8); tbl.scale(1, 1.35)

    for j in range(len(_COLS_TABLA)):
        tbl[0, j].set_facecolor(_DARK)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    norm = mcolors.Normalize(vmin=ranked[sort_col].min(), vmax=ranked[sort_col].max())
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for ii, (_, r) in enumerate(ranked.iterrows(), 1):
        base = "#f7f9fc" if ii % 2 == 0 else "#ffffff"
        hi   = mcolors.to_hex(sm.to_rgba(r[sort_col]))
        for j in range(len(_COLS_TABLA)):
            cell = tbl[ii, j]
            cell.set_facecolor(hi if j == hi_idx else base)
            if ii <= 3: cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#dde3ef")

    leyenda = ("RSI_L=RSI_LENGTH · N=ventana · BUY_T/SELL_T=umbrales · "
               "FL%=FLOOR_PCT · F_C=FACTOR_CAIDA · F_S=FACTOR_SUBIDA · "
               "GC=GUARDIA_COMPRA · URES%=USDT_RESERVA · BTC%=BTC_ACUMULA · "
               "PROF%=PROF_ZONA · C/V=compras/ventas · Canc=órdenes canceladas")
    fig.suptitle(
        f"{titulo}\n{subtitulo}\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  {leyenda}",
        fontsize=8.5, fontweight="bold", color=_DARK, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"  ✓ Tabla {sort_col}: {out_path}")


def fig_tres_tablas(df_res: pd.DataFrame):
    for sort_col, hi_idx, cmap, titulo, subtitulo, out_path in _RANKINGS:
        _fig_tabla(df_res, sort_col, hi_idx, cmap, titulo, subtitulo, out_path)


def fig_analisis_variables(df_res: pd.DataFrame):
    variables = [
        ("rsi_length",       "RSI_LENGTH",         lambda x: str(int(x))),
        ("n_ventana",        "N",                  lambda x: str(int(x))),
        ("rsi_buy_trigger",  "RSI_BUY_TRIGGER",    lambda x: str(int(x))),
        ("rsi_sell_trigger", "RSI_SELL_TRIGGER",   lambda x: str(int(x))),
        ("floor_pct",        "FLOOR_PCT",          lambda x: f"{int(x)}%"),
        ("factor_caida",     "FACTOR_CAIDA",       lambda x: f"{x:.1f}"),
        ("factor_subida",      "FACTOR_SUBIDA",      lambda x: f"{x:.2f}"),
        ("pct_ath_proyectado", "PCT_ATH_PROYECTADO", lambda x: f"{int(x)}%"),
        ("guardia_compra",     "GUARDIA_COMPRA",     lambda x: "T" if x else "F"),
        ("guardia_precio_compra",  "GUARDIA_PRECIO_C",  lambda x: "T" if x else "F"),
        ("guardia_precio_venta",   "GUARDIA_PRECIO_V",  lambda x: "T" if x else "F"),
        ("usdt_reserva_pct",       "USDT_RESERVA_PCT",  lambda x: f"{int(x)}%"),
        ("btc_acumula_pct",  "BTC_PCT_ACCUMULATE", lambda x: f"{int(x)}%"),
        ("prof_zona_pct",    "PROF_ZONA_PCT",      lambda x: f"{x:.0f}%"),
    ]
    n_vars = len(variables)
    cols   = 6
    rows_g = math.ceil(n_vars / cols)

    fig = plt.figure(figsize=(26, 7 * rows_g))
    fig.patch.set_facecolor(_BG)
    gs  = GridSpec(rows_g, cols, figure=fig, hspace=0.55, wspace=0.40)

    for vi, (col, label, fmt) in enumerate(variables):
        ax = fig.add_subplot(gs[vi // cols, vi % cols])
        df_g = df_res.copy()
        # Para guardia_compra (bool) convertir a 0/1 para agrupar
        if col in ("guardia_compra", "guardia_precio_compra", "guardia_precio_venta"):
            df_g[col] = df_g[col].astype(int)
        grp  = df_g.groupby(col)
        vals = sorted(df_g[col].unique())

        med_pnl = [grp.get_group(v)["pnl_pct"].median()          for v in vals]
        max_pnl = [grp.get_group(v)["pnl_pct"].max()             for v in vals]
        q25_pnl = [grp.get_group(v)["pnl_pct"].quantile(.25)     for v in vals]
        q75_pnl = [grp.get_group(v)["pnl_pct"].quantile(.75)     for v in vals]
        med_btc = [grp.get_group(v)["btc_total"].median()        for v in vals]

        x      = np.arange(len(vals))
        colors = ["#e74c3c" if m < 0 else "#27ae60" for m in med_pnl]
        yerr   = [np.array(med_pnl) - np.array(q25_pnl),
                  np.array(q75_pnl) - np.array(med_pnl)]

        ax.bar(x, med_pnl, color=colors, alpha=0.82, zorder=3,
               yerr=yerr, error_kw={"ecolor":"#555","capsize":3,"linewidth":0.9,"alpha":0.6})
        for xi, mp_ in zip(x, max_pnl):
            ax.annotate("▲", (xi, mp_), ha="center", va="bottom",
                        fontsize=7.5, color="#c0392b")

        ax2 = ax.twinx()
        ax2.plot(x, med_btc, color="#8e44ad", marker="o",
                 linewidth=1.5, markersize=4, zorder=4)
        ax2.tick_params(axis="y", labelsize=6.5, colors="#8e44ad")
        ax2.set_ylabel("BTC total ₿", fontsize=6.5, color="#8e44ad")

        ax.set_facecolor("#f8fafd")
        ax.set_xticks(x)
        # Para guardia_compra mostrar T/F en vez de 0/1
        if col in ("guardia_compra", "guardia_precio_compra", "guardia_precio_venta"):
            ax.set_xticklabels(["F", "T"], fontsize=8)
        else:
            ax.set_xticklabels([fmt(v) for v in vals], fontsize=7.5,
                               rotation=30 if len(vals) > 4 else 0)
        ax.set_title(label, fontsize=8.5, fontweight="bold", color=_DARK, pad=5)
        ax.set_xlabel("Valor", fontsize=7)
        ax.set_ylabel("PnL% mediano", fontsize=7)
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(axis="y", alpha=0.3, color="#dde3ef")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Impacto por Variable — Divergencia RSI Zona V2\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"Barras = mediana PnL%  ·  error = IQR (Q25–Q75)  ·  ▲ = máximo  ·  "
        f"Línea violeta = mediana BTC total (eje derecho)",
        fontsize=10, fontweight="bold", color=_DARK, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_ANALISIS, dpi=120, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"  ✓ Análisis de variables: {OUT_ANALISIS}")


def fig_scatter(df_res: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(_BG); ax.set_facecolor("#ffffff")

    pnl  = df_res["pnl_pct"].values
    btc  = df_res["btc_total"].values
    port = df_res["portfolio_final"].values
    dd   = df_res["max_drawdown"].values

    bmin, bmax = port.min(), port.max()
    dd_norm    = (dd - dd.min()) / max(dd.max() - dd.min(), 1)
    sizes      = 15 + (1 - dd_norm) * 60

    sc = ax.scatter(pnl, btc, c=port, cmap="RdYlGn",
                    s=sizes, alpha=0.55, linewidths=0,
                    vmin=bmin, vmax=bmax)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Portfolio final ($)", fontsize=10)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)

    for sort_col, color, lbl in [
        ("pnl_pct",       "#27ae60", "Top PnL%"),
        ("btc_total",     "#e67e22", "Top BTC Total"),
        ("portfolio_final","#2980b9","Top Portfolio"),
    ]:
        top3 = df_res.nlargest(3, sort_col)
        ax.scatter(top3["pnl_pct"], top3["btc_total"],
                   color=color, s=180, marker="*", zorder=10,
                   label=lbl, edgecolors="black", linewidths=0.5)
        for _, r in top3.iterrows():
            ax.annotate(
                f"RSI={int(r.rsi_length)} N={int(r.n_ventana)}\n"
                f"B={int(r.rsi_buy_trigger)} S={int(r.rsi_sell_trigger)}\n"
                f"FC={r.factor_caida:.1f} FS={r.factor_subida:.2f}",
                xy=(r.pnl_pct, r.btc_total),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6, color=_DARK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.78, edgecolor=color, linewidth=0.8),
            )

    ax.set_xlabel("PnL% sobre cartera total", fontsize=11)
    ax.set_ylabel("BTC total (₿)", fontsize=11)
    ax.set_title(
        f"Trade-off PnL% vs BTC Total — Divergencia RSI Zona V2\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"Color = Portfolio $  ·  Tamaño ∝ 1/MaxDD  ·  ★ = top-3",
        fontsize=10, fontweight="bold", color=_DARK,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, color="#dde3ef")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=120, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"  ✓ Scatter PnL% vs BTC total: {OUT_SCATTER}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(df_res: pd.DataFrame):
    sep = "═" * 96
    print(f"\n{sep}")
    print("  RESUMEN — OPTIMIZADOR DIVERGENCIA RSI ZONA V2")
    print(sep)
    print(f"  Período       : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Filas únicas  : {len(df_res):,}  (top-{TOP_HEAP} × 3 métricas)")
    print(f"  PnL%  rango   : {df_res['pnl_pct'].min():+.2f}%  →  {df_res['pnl_pct'].max():+.2f}%")
    print(f"  PnL%  mediana : {df_res['pnl_pct'].median():+.2f}%")
    print(f"  BTC total max : {df_res['btc_total'].max():.6f} ₿")
    print(f"  Portfolio max : ${df_res['portfolio_final'].max():,.2f}")

    hdr = (f"  {'#':>3}  {'RSL':>3}  {'N':>3}  {'BUY':>3}  {'SEL':>3}  "
           f"{'FL':>2}  {'FC':>4}  {'FS':>4}  {'ATH%':>4}  {'GC':>2}  {'GPC':>3}  {'GPV':>3}  "
           f"{'UR':>2}  {'BA':>2}  {'PZ':>3}  "
           f"{'PnL%':>8}  {'Port$':>10}  {'BTCtot':>9}  "
           f"{'C':>4}  {'V':>4}  {'Cn':>4}  {'DD%':>5}")

    rankings = [
        ("RANKING 1 — MEJOR PnL%",        df_res.sort_values("pnl_pct",        ascending=False)),
        ("RANKING 2 — MAYOR BTC TOTAL",    df_res.sort_values("btc_total",      ascending=False)),
        ("RANKING 3 — MAYOR PORTFOLIO $",  df_res.sort_values("portfolio_final",ascending=False)),
    ]

    for titulo, ranked in rankings:
        print(f"\n  {'─'*94}")
        print(f"  {titulo}")
        print(f"  {'─'*94}")
        print(hdr)
        print(f"  {'─'*94}")
        for rank, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
            mk = "★" if rank <= 3 else " "
            print(
                f"  {mk}{rank:>2}.  "
                f"{int(r.rsi_length):>3}  {int(r.n_ventana):>3}  "
                f"{int(r.rsi_buy_trigger):>3}  {int(r.rsi_sell_trigger):>3}  "
                f"{int(r.floor_pct):>2}  {r.factor_caida:>4.1f}  {r.factor_subida:>4.2f}  "
                f"{int(r.pct_ath_proyectado):>4}  "
                f"{'T' if r.guardia_compra else 'F':>2}  "
                f"{'T' if r.guardia_precio_compra else 'F':>3}  "
                f"{'T' if r.guardia_precio_venta else 'F':>3}  "
                f"{int(r.usdt_reserva_pct):>2}  {int(r.btc_acumula_pct):>2}  "
                f"{r.prof_zona_pct:>3.0f}  "
                f"{r.pnl_pct:>+7.2f}%  ${r.portfolio_final:>9,.2f}  "
                f"{r.btc_total:>9.6f}  "
                f"{int(r.total_compras):>4}  {int(r.total_ventas):>4}  "
                f"{int(r.ordenes_canceladas):>4}  {r.max_drawdown:>5.1f}%"
            )

    print(f"\n  {'─'*94}")
    print("  PARÁMETROS DOMINANTES EN EL TOP-10% DE CADA RANKING")
    print(f"  {'─'*94}")
    param_cols = [
        ("rsi_length",       "RSI_LENGTH",         lambda v: str(int(v))),
        ("n_ventana",        "N",                  lambda v: str(int(v))),
        ("rsi_buy_trigger",  "RSI_BUY_TRIGGER",    lambda v: str(int(v))),
        ("rsi_sell_trigger", "RSI_SELL_TRIGGER",   lambda v: str(int(v))),
        ("floor_pct",        "FLOOR_PCT",          lambda v: f"{int(v)}%"),
        ("factor_caida",     "FACTOR_CAIDA",       lambda v: f"{v:.1f}"),
        ("factor_subida",      "FACTOR_SUBIDA",      lambda v: f"{v:.2f}"),
        ("pct_ath_proyectado", "PCT_ATH_PROYECTADO", lambda v: f"{int(v)}%"),
        ("guardia_compra",     "GUARDIA_COMPRA",     lambda v: "True" if v else "False"),
        ("guardia_precio_compra", "GUARDIA_PRECIO_C",  lambda v: "True" if v else "False"),
        ("guardia_precio_venta",  "GUARDIA_PRECIO_V",  lambda v: "True" if v else "False"),
        ("usdt_reserva_pct",      "USDT_RESERVA_PCT",  lambda v: f"{int(v)}%"),
        ("btc_acumula_pct",  "BTC_PCT_ACCUMULATE", lambda v: f"{int(v)}%"),
        ("prof_zona_pct",    "PROF_ZONA_PCT",      lambda v: f"{v:.0f}%"),
    ]
    for titulo, ranked in rankings:
        top10 = ranked.head(max(1, len(ranked) // 10))
        print(f"\n  Dominantes — {titulo}")
        for col, lbl, fmt in param_cols:
            val  = top10[col].mode().iloc[0]
            freq = (top10[col] == val).mean() * 100
            print(f"    {lbl:<24}: {fmt(val)}  ({freq:.0f}% del top 10%)")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMIZADOR — DIVERGENCIA RSI ZONA V2                              ║")
    print("║  Multiprocessing · Sin lookahead · Órdenes límite reales            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    t_total = time.time()

    print("Cargando datos...")
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py"); return
    print(f"  Velas  : {len(df):,}")
    print(f"  Desde  : {df['datetime'].iloc[0]}")
    print(f"  Hasta  : {df['datetime'].iloc[-1]}\n")

    print("Precomputando RSI...")
    rsi_cache = precomputar_rsi(df)
    print()

    df_res = optimizar(df, rsi_cache)

    precio_final = float(df["close"].iloc[-1])
    df_res = _agregar_scores(df_res, precio_final)

    print("\nGuardando resultados...")
    guardar_resultados(df_res)

    imprimir_resumen(df_res)

    print("Generando visualizaciones...")
    fig_tres_tablas(df_res)
    fig_analisis_variables(df_res)
    fig_scatter(df_res)

    print(f"\n{'═'*72}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*72}")
    for fname in [OUT_CSV, OUT_JSON, OUT_TABLA_PNL, OUT_TABLA_BTC, OUT_TABLA_PRT,
                  OUT_ANALISIS, OUT_SCATTER]:
        print(f"  · {fname}")
    print(f"{'═'*72}")
    print(f"✓ Total completado en {time.time() - t_total:.1f}s\n")


if __name__ == "__main__":
    main()