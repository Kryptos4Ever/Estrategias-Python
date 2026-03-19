"""
Optimizador — Binance Spot Grid Bot  [Vectorizado NumPy]
══════════════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Grid Search Exhaustivo
Estrategia: Grid Fijo · Sizing Uniforme · Órdenes Límite

VARIABLES OPTIMIZADAS (6)
──────────────────────────────────────────────────────────────────────────────
  PCT_SUPERIOR      % por encima del primer close → PRECIO_SUPERIOR
  PCT_INF_RATIO     PRECIO_INFERIOR como fracción de PRECIO_SUPERIOR
  NUM_GRIDS         número de niveles del grid
  MODO_GRID         0 = geométrico  |  1 = aritmético
  STOP_LOSS_PCT     % por debajo de PRECIO_INFERIOR (-1 = desactivado)
  TAKE_PROFIT_PCT   % por encima de PRECIO_SUPERIOR (-1 = desactivado)

VARIABLES FIJAS (de config_binance_grid.py)
──────────────────────────────────────────────────────────────────────────────
  SALDO_USDT_INICIAL  USDT_RESERVA_PCT  COMMISSION_PCT
  DB_PATH / FECHA_INICIO / FECHA_FIN

ARQUITECTURA
──────────────────────────────────────────────────────────────────────────────
Vectorización NumPy pura sobre B combinaciones × MAX_GRIDS niveles.
Cada vela procesa TODOS los niveles de TODAS las combinaciones en una sola
pasada matricial — sin loops de cascada porque los niveles son independientes.

Arrays de estado (B, MAX_GRIDS):
  nivel_buy[B, MAX_GRIDS]   precio de compra de cada nivel
  nivel_sell[B, MAX_GRIDS]  precio de venta de cada nivel (= nivel[k+1])
  buy_activa[B, MAX_GRIDS]  orden de compra pendiente
  sell_activa[B, MAX_GRIDS] orden de venta pendiente
  btc_qty[B, MAX_GRIDS]     BTC en posición por nivel
  candle_buy[B, MAX_GRIDS]  vela de compra (filtro same-candle)

Same-candle fix idéntico a estrategia UPO:
  Posición comprada en vela i no puede venderse en vela i.

RANKINGS
──────────────────────────────────────────────────────────────────────────────
  1. Mejor PnL%           retorno total (USDT + BTC × precio cierre)
  2. Mayor ganancia ciclos USDT bloqueado por ciclos completos (sin BTC abierto)
  3. Mayor ciclos          más ciclos buy+sell ejecutados

ARCHIVOS DE SALIDA
──────────────────────────────────────────────────────────────────────────────
  optimizacion_binance_grid.csv
  optimizacion_binance_grid_top.json
  optimizacion_binance_grid_ranking_pnl.png
  optimizacion_binance_grid_ranking_ganancia.png
  optimizacion_binance_grid_ranking_ciclos.png
  optimizacion_binance_grid_analisis.png
  optimizacion_binance_grid_scatter.png
"""

import sqlite3, json, math, os, time
import heapq as _heapq
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
# CONFIG  (variables fijas leídas de config_binance_grid.py)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from config_binance_grid import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL, USDT_RESERVA_PCT, COMMISSION_PCT,
    )
    print("✓ config_binance_grid.py cargado")
except ImportError:
    print("⚠  config_binance_grid.py no encontrado — usando valores por defecto")
    DB_PATH            = r"btc_hourly.db"
    FECHA_INICIO       = "2021-11-10"
    FECHA_FIN          = "2022-11-22"
    SALDO_USDT_INICIAL = 1000
    USDT_RESERVA_PCT   = 0
    COMMISSION_PCT     = 0.1

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100


# ══════════════════════════════════════════════════════════════════════════════
# ESPACIO DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════
#
#  Variable          valores                                          vals
#  ──────────────────────────────────────────────────────────────────────────
#  PCT_SUPERIOR      [0%, 5%, 10%, 20%]  sobre primer close            4
#                    → PRECIO_SUPERIOR = first_close × (1 + pct_sup)
#
#  PCT_INF_RATIO     [10%, 15%, 20%, 25%, 30%, 40%]                    6
#                    → PRECIO_INFERIOR = PRECIO_SUPERIOR × pct_inf
#
#  NUM_GRIDS         [10, 20, 30, 50, 75, 100, 150]                    7
#
#  MODO_GRID         [0=geométrico, 1=aritmético]                      2
#
#  STOP_LOSS_PCT     [-1=off, 0.05, 0.10]   % bajo PRECIO_INFERIOR     3
#
#  TAKE_PROFIT_PCT   [-1=off, 0.10, 0.20]   % sobre PRECIO_SUPERIOR    3
#
#  Total: 4 × 6 × 7 × 2 × 3 × 3 = 9 072 combinaciones  (~3 min)
# ══════════════════════════════════════════════════════════════════════════════

PCT_SUP_VALS   = [0.00]
PCT_INF_VALS   = [0.20, 0.25, 0.30]
NUM_GRIDS_VALS = [10, 20, 30, 50, 75, 100, 150]
MODO_VALS      = [0, 1]          # 0 = geométrico, 1 = aritmético
SL_VALS        = [-1]
TP_VALS        = [-1]

MAX_GRIDS = 150   # máximo de niveles soportados por combinación

# Configuración del optimizador
TOP_N    = 25
TOP_HEAP = 500

# Archivos de salida
OUT_CSV       = "optimizacion_binance_grid.csv"
OUT_JSON      = "optimizacion_binance_grid_top.json"
OUT_TABLA_PNL = "optimizacion_binance_grid_ranking_pnl.png"
OUT_TABLA_GAN = "optimizacion_binance_grid_ranking_ganancia.png"
OUT_TABLA_CIC = "optimizacion_binance_grid_ranking_ciclos.png"
OUT_ANALISIS  = "optimizacion_binance_grid_analisis.png"
OUT_SCATTER   = "optimizacion_binance_grid_scatter.png"

COL_ORDER = [
    "pct_superior", "pct_inf_ratio", "num_grids", "modo_grid",
    "stop_loss_pct", "take_profit_pct",
    "pnl_pct", "portfolio_final", "usdt_final",
    "btc_en_pos", "n_niveles_abiertos",
    "ganancia_ciclos", "ciclos_completos", "total_compras", "total_trades",
    "max_drawdown", "bot_detenido",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
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
# SIMULADOR VECTORIZADO — Binance Spot Grid
# ══════════════════════════════════════════════════════════════════════════════
#
# combos_arr shape (B, 6):
#   col 0: pct_superior     col 1: pct_inf_ratio   col 2: num_grids
#   col 3: modo_grid        col 4: stop_loss_pct   col 5: take_profit_pct
#
# Diferencias clave respecto al simulador UPO:
#   · Grid FIJO desde inicio — niveles precomputados por combinación.
#   · Todos los niveles activos simultáneamente — un solo pase matricial por vela.
#   · Sin cascade loop — cada nivel es independiente.
#   · Stop loss / take profit global liquidan todas las posiciones y detienen el bot.
#   · Sizing uniforme: usdt_por_nivel = SALDO / NUM_GRIDS (mismo para todos los niveles).
#   · Same-candle fix idéntico al UPO: vela compra ≠ vela venta.
# ══════════════════════════════════════════════════════════════════════════════

def simular_vectorizado(df: pd.DataFrame, combos_arr: np.ndarray) -> dict:
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    n_velas = len(closes)
    B       = len(combos_arr)

    comm        = COMMISSION_PCT / 100.0
    factor_sell = 1.0 - comm
    capital     = float(SALDO_USDT_INICIAL - USDT_RESERVA)
    first_close = float(closes[0])

    # ── Parámetros shape (B,) ─────────────────────────────────────────────────
    p_sup     = combos_arr[:, 0]   # % sobre first_close para PRECIO_SUPERIOR
    p_inf_r   = combos_arr[:, 1]   # PRECIO_INFERIOR / PRECIO_SUPERIOR
    p_ng      = combos_arr[:, 2].astype(np.int32)   # num_grids
    p_modo    = combos_arr[:, 3].astype(np.int32)   # 0=geo, 1=ari
    p_sl      = combos_arr[:, 4]   # stop_loss_pct  (-1 = off)
    p_tp      = combos_arr[:, 5]   # take_profit_pct (-1 = off)

    precio_sup = first_close * (1.0 + p_sup)          # (B,)
    precio_inf = precio_sup  * p_inf_r                 # (B,)

    # Capital uniforme por nivel
    usdt_por_nivel = capital / np.maximum(p_ng.astype(np.float64), 1.0)  # (B,)

    # Precios de stop/take
    sl_enabled = (p_sl >= 0)
    sl_price   = np.where(sl_enabled, precio_inf * (1.0 - np.maximum(p_sl, 0.0)), -np.inf)
    tp_enabled = (p_tp >= 0)
    tp_price   = np.where(tp_enabled, precio_sup * (1.0 + np.maximum(p_tp, 0.0)),  np.inf)

    # ── Precomputar niveles del grid shape (B, MAX_GRIDS+1) ───────────────────
    k_all = np.arange(MAX_GRIDS + 1, dtype=np.float64)   # (MAX_GRIDS+1,)
    ng_f  = np.maximum(p_ng.astype(np.float64), 1.0)     # (B,)

    # Geométrico: nivel[k] = precio_inf × (precio_sup/precio_inf)^(k/num_grids)
    ratio_total  = precio_sup / np.maximum(precio_inf, 1e-30)         # (B,)
    exponent_all = k_all[None, :] / ng_f[:, None]                     # (B, MAX_GRIDS+1)
    nivel_ref_geo = precio_inf[:, None] * (ratio_total[:, None] ** exponent_all)

    # Aritmético: nivel[k] = precio_inf + step×k
    step_ari      = (precio_sup - precio_inf) / ng_f                   # (B,)
    nivel_ref_ari = precio_inf[:, None] + step_ari[:, None] * k_all[None, :]

    # Seleccionar según modo
    es_geo    = (p_modo == 0)[:, None]                                 # (B, 1)
    nivel_ref = np.where(es_geo, nivel_ref_geo, nivel_ref_ari)         # (B, MAX_GRIDS+1)

    nivel_buy  = nivel_ref[:, :MAX_GRIDS].copy()   # (B, MAX_GRIDS)  precio de compra
    nivel_sell = nivel_ref[:, 1:].copy()            # (B, MAX_GRIDS)  precio de venta

    # Máscara de niveles válidos (k < num_grids del combo)
    k_idx     = np.arange(MAX_GRIDS, dtype=np.int32)                   # (MAX_GRIDS,)
    grid_mask = (k_idx[None, :] < p_ng[:, None])                       # (B, MAX_GRIDS)

    # ── Estado shape (B, MAX_GRIDS) ───────────────────────────────────────────
    buy_activa  = grid_mask.copy()                                      # todas activas al inicio
    sell_activa = np.zeros((B, MAX_GRIDS), dtype=bool)
    btc_qty     = np.zeros((B, MAX_GRIDS), dtype=np.float64)
    candle_buy  = np.full((B, MAX_GRIDS), -1, dtype=np.int32)

    # ── Estado agregado shape (B,) ────────────────────────────────────────────
    usdt_balance    = np.full(B, capital, dtype=np.float64)
    ganancia_ciclos = np.zeros(B, dtype=np.float64)
    compras         = np.zeros(B, dtype=np.int32)
    ventas          = np.zeros(B, dtype=np.int32)
    bot_activo      = np.ones(B, dtype=bool)

    peak_port = np.full(B, float(SALDO_USDT_INICIAL), dtype=np.float64)
    max_dd    = np.zeros(B, dtype=np.float64)

    # Precompute btc a adquirir por nivel (uniforme dentro de combo — varía entre combos)
    # btc_por_compra[B, MAX_GRIDS] = usdt_por_nivel[B] × (1-comm) / nivel_buy[B, MAX_GRIDS]
    btc_por_compra = (usdt_por_nivel[:, None] * (1.0 - comm) /
                      np.maximum(nivel_buy, 1e-30))                    # (B, MAX_GRIDS)

    # ── Loop principal ─────────────────────────────────────────────────────────
    for i in range(n_velas):
        low_i   = float(lows[i])
        high_i  = float(highs[i])
        close_i = float(closes[i])
        alcista = closes[i] >= opens[i]

        # ── Stop Loss global ───────────────────────────────────────────────────
        sl_trig = bot_activo & sl_enabled & (low_i <= sl_price)
        if np.any(sl_trig):
            btc_total = (btc_qty * sell_activa)[sl_trig].sum(axis=1)
            usdt_sl   = btc_total * sl_price[sl_trig] * factor_sell
            usdt_balance[sl_trig] += usdt_sl
            buy_activa [sl_trig]   = False
            sell_activa[sl_trig]   = False
            btc_qty    [sl_trig]   = 0.0
            candle_buy [sl_trig]   = -1
            bot_activo [sl_trig]   = False

        # ── Take Profit global ─────────────────────────────────────────────────
        tp_trig = bot_activo & tp_enabled & (high_i >= tp_price)
        if np.any(tp_trig):
            btc_total = (btc_qty * sell_activa)[tp_trig].sum(axis=1)
            usdt_tp   = btc_total * tp_price[tp_trig] * factor_sell
            usdt_balance[tp_trig] += usdt_tp
            buy_activa [tp_trig]   = False
            sell_activa[tp_trig]   = False
            btc_qty    [tp_trig]   = 0.0
            candle_buy [tp_trig]   = -1
            bot_activo [tp_trig]   = False

        # ── COMPRAS ────────────────────────────────────────────────────────────
        # Un solo pase matricial: todos los niveles de todas las combos a la vez
        def _buy():
            active = bot_activo[:, None] & grid_mask & buy_activa  # (B, MAX_GRIDS)
            trig   = active & (low_i <= nivel_buy)                  # (B, MAX_GRIDS)
            if not np.any(trig):
                return
            n_b = trig.sum(axis=1).astype(np.float64)              # (B,) compras por combo
            usdt_balance.__isub__(usdt_por_nivel * n_b)
            btc_qty[trig]     = btc_por_compra[trig]
            candle_buy[trig]  = i
            buy_activa[trig]  = False
            sell_activa[trig] = True
            compras.__iadd__(trig.sum(axis=1).astype(np.int32))

        # ── VENTAS ─────────────────────────────────────────────────────────────
        # Un solo pase matricial: todos los niveles a la vez (sin cascade)
        # Same-candle fix: no vender si compró en esta misma vela
        def _sell():
            same_c = (candle_buy == i)                              # (B, MAX_GRIDS)
            active = bot_activo[:, None] & grid_mask & sell_activa & ~same_c
            trig   = active & (high_i >= nivel_sell)                # (B, MAX_GRIDS)
            if not np.any(trig):
                return

            # USDT recibido en cada venta
            usdt_sell_2d = btc_qty * nivel_sell * factor_sell       # (B, MAX_GRIDS)
            # Solo sumar los trigged
            usdt_from_sells    = (usdt_sell_2d * trig).sum(axis=1)  # (B,)
            # Ganancia = usdt_recibido - usdt_invertido
            ganancia_2d        = usdt_sell_2d - usdt_por_nivel[:, None]
            ganancia_this_sell = (ganancia_2d * trig).sum(axis=1)   # (B,)

            usdt_balance.__iadd__(usdt_from_sells)
            ganancia_ciclos.__iadd__(ganancia_this_sell)
            ventas.__iadd__(trig.sum(axis=1).astype(np.int32))

            # Reiniciar nivel: vuelve a buy_activa
            sell_activa[trig] = False
            buy_activa[trig]  = True
            btc_qty[trig]     = 0.0
            candle_buy[trig]  = -1

        if alcista:
            _buy(); _sell()
        else:
            _sell(); _buy()

        # Mark-to-market y drawdown
        btc_en_pos_i = (btc_qty * sell_activa).sum(axis=1)          # (B,)
        port         = usdt_balance + USDT_RESERVA + btc_en_pos_i * close_i
        peak_port    = np.maximum(peak_port, port)
        dd           = np.where(peak_port > 0,
                                (peak_port - port) / peak_port * 100.0, 0.0)
        max_dd       = np.maximum(max_dd, dd)

    # ── Métricas finales ───────────────────────────────────────────────────────
    last_close    = float(closes[-1])
    btc_en_pos_f  = (btc_qty * sell_activa).sum(axis=1)             # (B,)
    n_niveles_ab  = sell_activa.sum(axis=1).astype(np.int32)         # (B,)
    portfolio     = usdt_balance + USDT_RESERVA + btc_en_pos_f * last_close
    pnl_pct       = (portfolio - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100.0

    return {
        "pnl_pct"          : pnl_pct,
        "portfolio"        : portfolio,
        "usdt_final"       : usdt_balance + USDT_RESERVA,
        "btc_en_pos"       : btc_en_pos_f,
        "n_niveles_abiertos": n_niveles_ab,
        "ganancia_ciclos"  : ganancia_ciclos,
        "ciclos_completos" : ventas,
        "compras"          : compras,
        "total_trades"     : compras + ventas,
        "max_dd"           : max_dd,
        "bot_detenido"     : ~bot_activo,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame) -> pd.DataFrame:
    total_combos = (len(PCT_SUP_VALS) * len(PCT_INF_VALS) * len(NUM_GRIDS_VALS) *
                    len(MODO_VALS) * len(SL_VALS) * len(TP_VALS))

    print(f"\n{'═'*72}")
    print(f"  GRID SEARCH — {total_combos:,} combinaciones")
    print(f"{'═'*72}")
    print(f"  PCT_SUPERIOR    : {PCT_SUP_VALS}   ({len(PCT_SUP_VALS)} vals)")
    print(f"  PCT_INF_RATIO   : {PCT_INF_VALS}  ({len(PCT_INF_VALS)} vals)")
    print(f"  NUM_GRIDS       : {NUM_GRIDS_VALS}  ({len(NUM_GRIDS_VALS)} vals)")
    print(f"  MODO_GRID       : [0=geo, 1=ari]  (2 vals)")
    print(f"  STOP_LOSS_PCT   : {SL_VALS}   ({len(SL_VALS)} vals)")
    print(f"  TAKE_PROFIT_PCT : {TP_VALS}  ({len(TP_VALS)} vals)")
    print(f"  Top-K por métrica: {TOP_HEAP}  (máx ~{TOP_HEAP*3:,} filas)")
    print(f"{'═'*72}\n")

    param_combos = list(product(
        PCT_SUP_VALS, PCT_INF_VALS, NUM_GRIDS_VALS,
        MODO_VALS, SL_VALS, TP_VALS,
    ))
    combos_arr = np.array(param_combos, dtype=np.float64)            # (M, 6)
    M = len(combos_arr)

    heap_pnl = _TopHeap(TOP_HEAP)
    heap_gan = _TopHeap(TOP_HEAP)
    heap_cic = _TopHeap(TOP_HEAP)

    # Lote más pequeño que UPO: arrays (B, MAX_GRIDS) son más grandes
    BATCH   = 2000
    n_lotes = math.ceil(M / BATCH)
    ram_mb  = min(BATCH, M) * MAX_GRIDS * 5 * 8 / 1024**2   # ~5 arrays float64
    print(f"  Modo  : vectorizado NumPy — lotes de {BATCH:,}  ({n_lotes} lotes)")
    print(f"  RAM   : ~{ram_mb:.0f} MB por lote  (MAX_GRIDS={MAX_GRIDS})\n")

    t0          = time.time()
    combos_done = 0
    best_pnl    = -999.0
    all_pnl     = np.empty(M, dtype=np.float64)
    all_gan     = np.empty(M, dtype=np.float64)
    all_cic     = np.empty(M, dtype=np.float64)

    for lote_idx in range(n_lotes):
        s = lote_idx * BATCH
        e = min(s + BATCH, M)
        lote_arr    = combos_arr[s:e]
        lote_params = param_combos[s:e]

        arrs = simular_vectorizado(df, lote_arr)

        pnl_arr = arrs["pnl_pct"]
        gan_arr = arrs["ganancia_ciclos"]
        cic_arr = arrs["ciclos_completos"].astype(np.float64)
        all_pnl[s:e] = pnl_arr
        all_gan[s:e] = gan_arr
        all_cic[s:e] = cic_arr

        min_pnl = heap_pnl.min_val
        min_gan = heap_gan.min_val
        min_cic = heap_cic.min_val
        cand = np.where(
            (pnl_arr > min_pnl) |
            (gan_arr > min_gan) |
            (cic_arr > min_cic)
        )[0]

        for idx in cand:
            ps, pi, ng, mo, sl, tp = lote_params[idx]
            row = {
                "pct_superior"    : round(float(ps), 4),
                "pct_inf_ratio"   : round(float(pi), 4),
                "num_grids"       : int(ng),
                "modo_grid"       : "geo" if int(mo) == 0 else "ari",
                "stop_loss_pct"   : (round(float(sl), 4) if sl >= 0 else None),
                "take_profit_pct" : (round(float(tp), 4) if tp >= 0 else None),
                "pnl_pct"         : round(float(arrs["pnl_pct"][idx]),          4),
                "portfolio_final" : round(float(arrs["portfolio"][idx]),         2),
                "usdt_final"      : round(float(arrs["usdt_final"][idx]),        2),
                "btc_en_pos"      : round(float(arrs["btc_en_pos"][idx]),        8),
                "n_niveles_abiertos": int(arrs["n_niveles_abiertos"][idx]),
                "ganancia_ciclos" : round(float(arrs["ganancia_ciclos"][idx]),   4),
                "ciclos_completos": int(arrs["ciclos_completos"][idx]),
                "total_compras"   : int(arrs["compras"][idx]),
                "total_trades"    : int(arrs["total_trades"][idx]),
                "max_drawdown"    : round(float(arrs["max_dd"][idx]),            2),
                "bot_detenido"    : bool(arrs["bot_detenido"][idx]),
            }
            heap_pnl.push_if_better(float(pnl_arr[idx]), row)
            heap_gan.push_if_better(float(gan_arr[idx]), row)
            heap_cic.push_if_better(float(cic_arr[idx]), row)

        combos_done += len(lote_arr)
        best_pnl     = max(best_pnl, float(pnl_arr.max()))
        elapsed      = time.time() - t0
        eta          = elapsed / combos_done * (M - combos_done) if combos_done else 0
        survivors    = len({id(r) for h in (heap_pnl, heap_gan, heap_cic)
                            for r in h.to_list()})
        print(f"  [{lote_idx+1:>3}/{n_lotes}]  {combos_done:>6,}/{M:,}  "
              f"{elapsed:>6.1f}s  ETA:{eta:>5.1f}s  "
              f"top:{survivors:>4}  mejor PnL:{best_pnl:>+8.2f}%")

    elapsed_total = time.time() - t0
    print(f"\n✓ Completado en {elapsed_total:.1f}s  "
          f"({M / elapsed_total:,.0f} backtests/s)")

    seen, all_rows = set(), []
    for heap in (heap_pnl, heap_gan, heap_cic):
        for row in heap.to_list():
            key = (row["pct_superior"], row["pct_inf_ratio"], row["num_grids"],
                   row["modo_grid"], str(row["stop_loss_pct"]), str(row["take_profit_pct"]))
            if key not in seen:
                seen.add(key); all_rows.append(row)

    print(f"\n  ✓ Filas únicas: {len(all_rows):,}  (máx teórico: {TOP_HEAP*3:,})")
    print(f"\n  Estadísticas sobre {M:,} combinaciones:")
    print(f"    PnL%           : {all_pnl.min():+.2f}%  →  {all_pnl.max():+.2f}%"
          f"  (mediana {np.median(all_pnl):+.2f}%)")
    print(f"    Gan. ciclos    : ${all_gan.min():,.2f}  →  ${all_gan.max():,.2f}")
    print(f"    Ciclos compl.  : {all_cic.min():.0f}  →  {all_cic.max():.0f}")
    print(f"    PnL% > 0       : {(all_pnl > 0).sum():,}  "
          f"({(all_pnl > 0).mean()*100:.1f}%)")

    df_res = pd.DataFrame(all_rows)
    # Asegurar columnas presentes y ordenadas
    for col in COL_ORDER:
        if col not in df_res.columns:
            df_res[col] = None
    return df_res[COL_ORDER].sort_values("pnl_pct", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _agregar_scores(df_res: pd.DataFrame, precio_final: float) -> pd.DataFrame:
    df = df_res.copy()
    df["btc_en_pos_usd"] = df["btc_en_pos"] * precio_final
    for col, norm_col in [("pnl_pct", "pnl_norm"),
                          ("ganancia_ciclos", "gan_norm"),
                          ("ciclos_completos", "cic_norm")]:
        mn, mx = df[col].min(), df[col].max()
        span   = mx - mn if mx > mn else 1.0
        df[norm_col] = (df[col] - mn) / span
    df["combo_score"] = (df["pnl_norm"] * 0.5 +
                         df["gan_norm"] * 0.3 +
                         df["cic_norm"] * 0.2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados(df_res: pd.DataFrame):
    df_res.to_csv(OUT_CSV, index=False)
    print(f"  ✓ CSV  : {OUT_CSV}  ({len(df_res):,} filas)")

    top_json = {
        "meta": {
            "fecha_inicio"    : FECHA_INICIO,
            "fecha_fin"       : FECHA_FIN,
            "total_filas"     : len(df_res),
            "pct_sup_vals"    : PCT_SUP_VALS,
            "pct_inf_vals"    : PCT_INF_VALS,
            "num_grids_vals"  : NUM_GRIDS_VALS,
            "modo_vals"       : ["geo", "ari"],
            "sl_vals"         : SL_VALS,
            "tp_vals"         : TP_VALS,
            "top_heap"        : TOP_HEAP,
        },
        "ranking_pnl"     : df_res.sort_values("pnl_pct",          ascending=False)
                                   .head(TOP_N).to_dict(orient="records"),
        "ranking_ganancia": df_res.sort_values("ganancia_ciclos",   ascending=False)
                                   .head(TOP_N).to_dict(orient="records"),
        "ranking_ciclos"  : df_res.sort_values("ciclos_completos",  ascending=False)
                                   .head(TOP_N).to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(top_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ JSON : {OUT_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

_COLS = [
    "#", "SUP%", "INF%", "NG", "MODO", "SL%", "TP%",
    "PnL%", "Port $", "BTC Pos ₿", "Nivs",
    "GanCiclos $", "Ciclos", "Trades", "DD%", "Stop",
]

_RANKINGS = [
    ("pnl_pct",         7,  plt.cm.RdYlGn,
     "Ranking 1 — Mejor PnL% Total",
     "Retorno % sobre portfolio valorizado al precio de cierre",
     OUT_TABLA_PNL),
    ("ganancia_ciclos", 11, plt.cm.YlOrRd,
     "Ranking 2 — Mayor Ganancia de Ciclos (USDT)",
     "USDT bloqueados en ciclos buy+sell completados, neto de comisiones",
     OUT_TABLA_GAN),
    ("ciclos_completos",12, plt.cm.PuBuGn,
     "Ranking 3 — Mayor Número de Ciclos Completos",
     "Ciclos buy+sell ejecutados — proxy de actividad del grid",
     OUT_TABLA_CIC),
]


def _fmt_sl(v):
    return "off" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:.0f}%"

def _fmt_tp(v):
    return "off" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:.0f}%"


def _fig_tabla(df_res: pd.DataFrame, sort_col: str, hi_col_idx: int,
               cmap, titulo: str, subtitulo: str, out_path: str):
    ranked = df_res.sort_values(sort_col, ascending=False).head(TOP_N)
    rows   = []
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        rows.append([
            str(rank),
            f"{r.pct_superior*100:.0f}%",
            f"{r.pct_inf_ratio*100:.0f}%",
            str(int(r.num_grids)),
            str(r.modo_grid),
            _fmt_sl(r.stop_loss_pct),
            _fmt_tp(r.take_profit_pct),
            f"{r.pnl_pct:+.2f}%",
            f"${r.portfolio_final:,.2f}",
            f"{r.btc_en_pos:.6f}",
            str(int(r.n_niveles_abiertos)),
            f"${r.ganancia_ciclos:,.2f}",
            str(int(r.ciclos_completos)),
            str(int(r.total_trades)),
            f"{r.max_drawdown:.1f}%",
            "✓" if r.bot_detenido else "",
        ])

    fig_h = max(4.5, 0.36 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(24, fig_h))
    fig.patch.set_facecolor("#f4f6fa")
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=_COLS, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.35)

    for j in range(len(_COLS)):
        cell = table[0, j]
        cell.set_facecolor("#1a2540")
        cell.set_text_props(color="white", fontweight="bold")

    norm = mcolors.Normalize(vmin=ranked[sort_col].min(), vmax=ranked[sort_col].max())
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for ii, (_, r) in enumerate(ranked.iterrows(), 1):
        base   = "#f7f9fc" if ii % 2 == 0 else "#ffffff"
        hi_col = mcolors.to_hex(sm.to_rgba(r[sort_col]))
        for j in range(len(_COLS)):
            cell = table[ii, j]
            cell.set_facecolor(hi_col if j == hi_col_idx else base)
            if ii <= 3:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#dde3ef")

    fig.suptitle(
        f"{titulo}\n{subtitulo}\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"SUP%=pct sobre 1er close · INF%=inferior/superior · "
        f"NG=num_grids · MODO=geo/ari · SL/TP=stop/take profit",
        fontsize=9.5, fontweight="bold", color="#1a2540", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Tabla {sort_col}: {out_path}")


def fig_tres_tablas(df_res: pd.DataFrame):
    for sort_col, hi_idx, cmap, titulo, subtitulo, out_path in _RANKINGS:
        _fig_tabla(df_res, sort_col, hi_idx, cmap, titulo, subtitulo, out_path)


def fig_analisis_variables(df_res: pd.DataFrame):
    """Impacto mediano de cada variable sobre PnL% y ganancia de ciclos."""
    variables = [
        ("pct_superior",    "PCT_SUPERIOR",     lambda x: f"{x*100:.0f}%"),
        ("pct_inf_ratio",   "PCT_INF_RATIO",    lambda x: f"{x*100:.0f}%"),
        ("num_grids",       "NUM_GRIDS",         lambda x: str(int(x))),
        ("modo_grid",       "MODO_GRID",         lambda x: str(x)),
        ("stop_loss_pct",   "STOP_LOSS",
         lambda x: "off" if (x is None or (isinstance(x,float) and math.isnan(x)))
                   else f"{x*100:.0f}%"),
        ("take_profit_pct", "TAKE_PROFIT",
         lambda x: "off" if (x is None or (isinstance(x,float) and math.isnan(x)))
                   else f"{x*100:.0f}%"),
    ]

    # Para stop/take profit, None puede estar mezclado con floats; normalizar
    df_plot = df_res.copy()
    for col in ["stop_loss_pct", "take_profit_pct"]:
        df_plot[col] = df_plot[col].apply(lambda v: -1.0 if v is None else float(v))

    fig = plt.figure(figsize=(26, 9))
    fig.patch.set_facecolor("#f4f6fa")
    gs  = GridSpec(1, 6, figure=fig, hspace=0.5, wspace=0.42)
    axes = [fig.add_subplot(gs[0, c]) for c in range(6)]

    for ax, (col, label, fmt) in zip(axes, variables):
        grp     = df_plot.groupby(col)
        vals    = sorted(df_plot[col].unique())
        med_pnl = [grp.get_group(v)["pnl_pct"].median()          for v in vals]
        max_pnl = [grp.get_group(v)["pnl_pct"].max()             for v in vals]
        q25_pnl = [grp.get_group(v)["pnl_pct"].quantile(.25)     for v in vals]
        q75_pnl = [grp.get_group(v)["pnl_pct"].quantile(.75)     for v in vals]
        med_gan = [grp.get_group(v)["ganancia_ciclos"].median()   for v in vals]

        x      = np.arange(len(vals))
        colors = ["#e74c3c" if m < 0 else "#27ae60" for m in med_pnl]
        yerr   = [np.array(med_pnl) - np.array(q25_pnl),
                  np.array(q75_pnl) - np.array(med_pnl)]

        ax.bar(x, med_pnl, color=colors, alpha=0.82, zorder=3,
               yerr=yerr, error_kw={"ecolor": "#555", "capsize": 3,
                                    "linewidth": 0.9, "alpha": 0.6})
        for xi, mp in zip(x, max_pnl):
            ax.annotate("▲", (xi, mp), ha="center", va="bottom",
                        fontsize=8, color="#c0392b")

        ax2 = ax.twinx()
        ax2.plot(x, med_gan, color="#e67e22", marker="o",
                 linewidth=1.5, markersize=4, zorder=4)
        ax2.tick_params(axis="y", labelsize=7, colors="#e67e22")
        ax2.set_ylabel("Gan. ciclos $ mediana", fontsize=7, color="#e67e22")

        ax.set_facecolor("#f8fafd")
        ax.set_xticks(x)
        # Para stop/take profit, mapear -1 → fmt
        ax.set_xticklabels([fmt(v) for v in vals], fontsize=7.5,
                           rotation=35 if len(vals) > 5 else 0)
        ax.set_title(label, fontsize=9, fontweight="bold", color="#1a2540", pad=6)
        ax.set_xlabel("Valor", fontsize=7)
        ax.set_ylabel("PnL% mediano", fontsize=7)
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(axis="y", alpha=0.3, color="#dde3ef")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Impacto por Variable — Binance Spot Grid Bot\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"Barras = mediana PnL%  ·  error = IQR (Q25–Q75)  ·  ▲ = máximo  ·  "
        f"Línea naranja = mediana ganancia ciclos $ (eje derecho)",
        fontsize=10, fontweight="bold", color="#1a2540", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(OUT_ANALISIS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Análisis de variables: {OUT_ANALISIS}")


def fig_scatter(df_res: pd.DataFrame):
    """Scatter PnL% vs Ganancia Ciclos. Color = NUM_GRIDS. Tamaño ∝ 1/MaxDD."""
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#f4f6fa")
    ax.set_facecolor("#ffffff")

    pnl  = df_res["pnl_pct"].values
    gan  = df_res["ganancia_ciclos"].values
    ng   = df_res["num_grids"].values.astype(float)
    dd   = df_res["max_drawdown"].values

    dd_norm = (dd - dd.min()) / max(dd.max() - dd.min(), 1)
    sizes   = 15 + (1 - dd_norm) * 60

    sc = ax.scatter(pnl, gan, c=ng, cmap="viridis",
                    s=sizes, alpha=0.55, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("NUM_GRIDS", fontsize=10)

    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)

    for col, color, lbl in [
        ("pnl_pct",         "#27ae60", "Top PnL%"),
        ("ganancia_ciclos", "#e67e22", "Top Gan.Ciclos"),
        ("ciclos_completos","#8e44ad", "Top Ciclos"),
    ]:
        top3 = df_res.nlargest(3, col)
        ax.scatter(top3["pnl_pct"], top3["ganancia_ciclos"],
                   color=color, s=180, marker="*",
                   zorder=10, label=lbl, edgecolors="black", linewidths=0.5)
        for _, r in top3.iterrows():
            ax.annotate(
                f"SUP={r.pct_superior*100:.0f}% INF={r.pct_inf_ratio*100:.0f}%\n"
                f"NG={int(r.num_grids)} {r.modo_grid.upper()}\n"
                f"SL={_fmt_sl(r.stop_loss_pct)} TP={_fmt_tp(r.take_profit_pct)}",
                xy=(r.pnl_pct, r.ganancia_ciclos),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6.5, color="#1a2540",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.78, edgecolor=color, linewidth=0.8),
            )

    ax.set_xlabel("PnL% sobre cartera total", fontsize=11)
    ax.set_ylabel("Ganancia de ciclos ($)", fontsize=11)
    ax.set_title(
        f"Trade-off PnL% vs Ganancia Ciclos — Binance Spot Grid\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"Color = NUM_GRIDS  ·  Tamaño ∝ 1/MaxDD  ·  ★ = top-3",
        fontsize=10, fontweight="bold", color="#1a2540",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, color="#dde3ef")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Scatter PnL% vs Ganancia Ciclos: {OUT_SCATTER}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(df_res: pd.DataFrame):
    sep = "═" * 90
    print(f"\n{sep}")
    print("  RESUMEN — OPTIMIZADOR BINANCE SPOT GRID")
    print(sep)
    print(f"  Período          : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Filas únicas     : {len(df_res):,}  (top-{TOP_HEAP} × 3 métricas)")
    print(f"  PnL%   rango     : {df_res['pnl_pct'].min():+.2f}%  →  {df_res['pnl_pct'].max():+.2f}%")
    print(f"  PnL%   mediana   : {df_res['pnl_pct'].median():+.2f}%")
    print(f"  Gan.ciclos max   : ${df_res['ganancia_ciclos'].max():,.2f}")
    print(f"  Ciclos max       : {int(df_res['ciclos_completos'].max())}")

    hdr = (f"  {'#':>3}  {'SUP%':>4}  {'INF%':>4}  {'NG':>4}  {'MODO':>4}  "
           f"{'SL%':>4}  {'TP%':>4}  "
           f"{'PnL%':>8}  {'Port$':>10}  {'GanCiclos':>11}  "
           f"{'Ciclos':>6}  {'Nivs':>4}  {'DD%':>5}  {'Stop':>4}")

    rankings = [
        ("RANKING 1 — MEJOR PnL%",          df_res.sort_values("pnl_pct",          ascending=False)),
        ("RANKING 2 — MAYOR GANANCIA CICLOS",df_res.sort_values("ganancia_ciclos",  ascending=False)),
        ("RANKING 3 — MAYOR CICLOS",         df_res.sort_values("ciclos_completos", ascending=False)),
    ]

    for titulo, ranked in rankings:
        print(f"\n  {'─'*88}")
        print(f"  {titulo}")
        print(f"  {'─'*88}")
        print(hdr)
        print(f"  {'─'*88}")
        for rank, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
            marker = "★" if rank <= 3 else " "
            print(
                f"  {marker}{rank:>2}.  "
                f"{r.pct_superior*100:>3.0f}%  "
                f"{r.pct_inf_ratio*100:>3.0f}%  "
                f"{int(r.num_grids):>4}  "
                f"{str(r.modo_grid):>4}  "
                f"{_fmt_sl(r.stop_loss_pct):>4}  "
                f"{_fmt_tp(r.take_profit_pct):>4}  "
                f"{r.pnl_pct:>+7.2f}%  "
                f"${r.portfolio_final:>9,.2f}  "
                f"${r.ganancia_ciclos:>9,.2f}  "
                f"{int(r.ciclos_completos):>6}  "
                f"{int(r.n_niveles_abiertos):>4}  "
                f"{r.max_drawdown:>5.1f}%  "
                f"{'✓' if r.bot_detenido else '':>4}"
            )

    print(f"\n  {'─'*88}")
    print("  PARÁMETROS DOMINANTES EN EL TOP-10% DE CADA RANKING")
    print(f"  {'─'*88}")
    param_cols = [
        ("pct_superior",  "PCT_SUPERIOR",  lambda v: f"{v*100:.0f}%"),
        ("pct_inf_ratio", "PCT_INF_RATIO", lambda v: f"{v*100:.0f}%"),
        ("num_grids",     "NUM_GRIDS",     lambda v: str(int(v))),
        ("modo_grid",     "MODO_GRID",     lambda v: str(v)),
        ("stop_loss_pct", "STOP_LOSS",     _fmt_sl),
        ("take_profit_pct","TAKE_PROFIT",  _fmt_tp),
    ]
    df_plot = df_res.copy()
    for col in ["stop_loss_pct", "take_profit_pct"]:
        df_plot[col] = df_plot[col].apply(lambda v: -1.0 if v is None else float(v))

    for titulo, ranked in rankings:
        top10 = ranked.head(max(1, len(ranked) // 10))
        print(f"\n  Dominantes — {titulo}")
        for col, lbl, fmt in param_cols:
            col_data = df_plot.loc[top10.index, col]
            val      = col_data.mode().iloc[0]
            freq     = (col_data == val).mean() * 100
            print(f"    {lbl:<18}: {fmt(val)}  ({freq:.0f}% del top 10%)")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMIZADOR — BINANCE SPOT GRID BOT                                ║")
    print("║  NumPy Vectorizado · Grid Fijo · Sizing Uniforme                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    t_total = time.time()

    print("Cargando datos...")
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config_binance_grid.py")
        return
    print(f"  Velas      : {len(df):,}")
    print(f"  Desde      : {df['datetime'].iloc[0]}")
    print(f"  Hasta      : {df['datetime'].iloc[-1]}")
    print(f"  1er close  : ${float(df['close'].iloc[0]):,.2f}  (base de PCT_SUPERIOR)\n")

    df_res = optimizar(df)

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
    for fname in [OUT_CSV, OUT_JSON, OUT_TABLA_PNL, OUT_TABLA_GAN, OUT_TABLA_CIC,
                  OUT_ANALISIS, OUT_SCATTER]:
        print(f"  · {fname}")
    print(f"{'═'*72}")
    print(f"✓ Total completado en {time.time() - t_total:.1f}s\n")


if __name__ == "__main__":
    main()
