"""
Optimizador — Estrategia Grid UPO Acumulator  [Vectorizado NumPy]
══════════════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Grid Search Exhaustivo
Estrategia: Órdenes Límite por Precio · Progresión Lineal de Compra

VARIABLES OPTIMIZADAS (6)
──────────────────────────────────────────────────────────────────────────────
  PCT_CAIDA_ATH          % caída desde ATH para primera compra (Modo A)
  PCT_CAIDA              % caída desde last_op para DCA (Modo B)
  PCT_VENTA              % subida desde entrada para TP de venta
  FLOOR_PCT              piso teórico del ciclo como % del ATH
  PENDIENTE_COMPRA       velocidad de la progresión lineal de sizing
  LAST_OP_UPDATE_ON_SELL si True, last_op sube al TP tras cada venta

VARIABLES FIJAS
──────────────────────────────────────────────────────────────────────────────
  USDT_RESERVA_PCT = 0   COMMISSION_PCT = 0.1
  DB_PATH / FECHA_INICIO / FECHA_FIN  (leídos desde config.py)

ARQUITECTURA DE SIMULACIÓN
──────────────────────────────────────────────────────────────────────────────
Vectorización NumPy pura. Tracking exacto por posición individual (no modelo
agregado): cada combinación mantiene hasta MAX_POS posiciones con su
precio_entrada, usdt_invertido y btc_cantidad reales.

FIXES respecto a versiones anteriores:
  1. Same-candle: posición creada en vela i no puede vender en vela i.
  2. last_op SOLO se actualiza en compras (salvo que LAST_OP_ON_SELL=True).
  3. Cada venta recupera el usdt_invertido exacto de esa posición.

COLUMNAS DEL CSV / JSON
──────────────────────────────────────────────────────────────────────────────
  Params : ca%, c%, v%, fl, pend, lop
  Resultados: pnl_pct, portfolio_final, usdt_final,
              btc_libre, btc_en_pos, btc_total,
              n_pos_final, total_compras, total_ventas, total_trades,
              max_drawdown

RANKINGS GENERADOS
──────────────────────────────────────────────────────────────────────────────
  1. Mejor PnL%         — retorno total sobre cartera (USDT + BTC × precio)
  2. Mayor BTC libre    — BTC acumulado neto de ventas
  3. Mayor BTC total    — BTC libre + BTC en posiciones abiertas

ARCHIVOS DE SALIDA
──────────────────────────────────────────────────────────────────────────────
  optimizacion_grid_upo.csv               → resultados (union top-3 heaps)
  optimizacion_grid_upo_top.json          → 3 rankings × TOP_N en JSON
  optimizacion_grid_upo_ranking_pnl.png   → tabla ranking 1 (PnL%)
  optimizacion_grid_upo_ranking_btcl.png  → tabla ranking 2 (BTC libre)
  optimizacion_grid_upo_ranking_btct.png  → tabla ranking 3 (BTC total)
  optimizacion_grid_upo_analisis.png      → impacto mediano por variable
  optimizacion_grid_upo_scatter.png       → scatter PnL% vs BTC libre
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
# CONFIG  (solo las variables fijas se leen de config.py)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL, USDT_RESERVA_PCT, COMMISSION_PCT,
    )
    print("✓ config.py cargado")
except ImportError:
    print("⚠  config.py no encontrado — usando valores por defecto")
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
#  Variable               rango           paso    vals
#  ────────────────────────────────────────────────────────
#  PCT_CAIDA_ATH          1% → 5%         1.0pp     5
#  PCT_CAIDA              1% → 4%         0.5pp     7
#  PCT_VENTA              1% → 6%         1.0pp     6
#  FLOOR_PCT             15  → 30         5         4
#  PENDIENTE_COMPRA      50  → 300        25       11
#  LAST_OP_UPDATE_ON_SELL False / True              2
#
#  Total: 5 × 7 × 6 × 4 × 11 × 2 = 18 480 combinaciones  (~2 min)
# ══════════════════════════════════════════════════════════════════════════════

# Señales de precio — compra
PCT_CAIDA_ATH_I = 0.05;  PCT_CAIDA_ATH_F = 0.4;  PCT_CAIDA_ATH_P = 0.05
PCT_CAIDA_I     = 0.010; PCT_CAIDA_F     = 0.050; PCT_CAIDA_P     = 0.005
# Señal de precio — venta
PCT_VENTA_I     = 0.010;  PCT_VENTA_F    = 0.060;  PCT_VENTA_P     = 0.005
# Referencia de ciclo
FLOOR_PCT_I     = 15;    FLOOR_PCT_F     = 30;    FLOOR_PCT_P     = 5
# Progresión lineal de sizing
PENDIENTE_I     = 1;    PENDIENTE_F     = 100;   PENDIENTE_P     = 5
# Toggle last_op en ventas
LAST_OP_VALS    = [False, True]

# Configuración del optimizador
TOP_N         = 25    # filas mostradas en tablas visuales
TOP_HEAP      = 500   # mejores por métrica guardados en heap
MAX_POS       = 60    # máximo de posiciones abiertas por combinación

# Archivos de salida
OUT_CSV       = "optimizacion_grid_upo.csv"
OUT_JSON      = "optimizacion_grid_upo_top.json"
OUT_TABLA_PNL = "optimizacion_grid_upo_ranking_pnl.png"
OUT_TABLA_LIB = "optimizacion_grid_upo_ranking_btcl.png"
OUT_TABLA_TOT = "optimizacion_grid_upo_ranking_btct.png"
OUT_ANALISIS  = "optimizacion_grid_upo_analisis.png"
OUT_SCATTER   = "optimizacion_grid_upo_scatter.png"

COL_ORDER = [
    "pct_caida_ath", "pct_caida", "pct_venta",
    "floor_pct", "pendiente", "last_op_on_sell",
    "pnl_pct", "portfolio_final", "usdt_final",
    "btc_libre", "btc_en_pos", "btc_total",
    "n_pos_final", "total_compras", "total_ventas", "total_trades",
    "max_drawdown",
]


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


def _pct_capital_vec(limit_buy: np.ndarray, ath_i: float,
                     floor_pct: np.ndarray,
                     pendiente: np.ndarray) -> np.ndarray:
    """
    Progresión lineal de sizing, vectorizada sobre B combinaciones.

      pos         = log(ATH / limit_buy) / log(100 / FLOOR_PCT)  ∈ [0, 1]
      pct_capital = min(pos × pendiente, 100)

    Retorna array shape (B,) en [0, 100].
    """
    log_rango = np.log(100.0 / np.maximum(floor_pct, 1e-6))
    valid     = (log_rango > 0) & (limit_buy > 1e-30)
    pos       = np.where(valid,
                    np.clip(
                        np.log(np.maximum(ath_i / np.maximum(limit_buy, 1e-30), 1.0))
                        / np.maximum(log_rango, 1e-30),
                        0.0, 1.0),
                    0.0)
    return np.minimum(pos * pendiente, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# MIN-HEAP DE TAMAÑO FIJO
# ══════════════════════════════════════════════════════════════════════════════

class _TopHeap:
    """Mantiene los N mejores resultados por una métrica (mayor = mejor)."""
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
# SIMULADOR VECTORIZADO  —  tracking exacto por posición individual
# ══════════════════════════════════════════════════════════════════════════════
#
# combos_arr shape (B, 6):
#   col 0: pct_caida_ath   col 1: pct_caida     col 2: pct_venta
#   col 3: floor_pct       col 4: pendiente      col 5: last_op_on_sell (0/1)
#
# FIXES aplicados (idénticos a la estrategia real):
#   1. Same-candle: posición creada en vela i no puede vender en vela i.
#   2. last_op se actualiza en compras siempre; en ventas solo si col5 = 1.
#   3. Cada venta usa el usdt_invertido / btc_cantidad EXACTO de esa posición.
#
# Orden intracandle:
#   vela alcista (close >= open) : compra → ventas en cascada
#   vela bajista (close  < open) : ventas en cascada → compra
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

    # ── Parámetros shape (B,) ─────────────────────────────────────────────────
    p_caida_ath  = combos_arr[:, 0]
    p_caida      = combos_arr[:, 1]
    p_venta      = combos_arr[:, 2]
    p_floor      = combos_arr[:, 3]
    p_pendiente  = combos_arr[:, 4]
    p_lop_sell   = combos_arr[:, 5].astype(bool)   # last_op update on sell

    # ── Arrays de posiciones shape (B, MAX_POS) ───────────────────────────────
    pos_entry  = np.zeros((B, MAX_POS), dtype=np.float64)   # precio entrada
    pos_usdt   = np.zeros((B, MAX_POS), dtype=np.float64)   # USDT invertido
    pos_btc    = np.zeros((B, MAX_POS), dtype=np.float64)   # BTC en posición
    pos_candle = np.full((B, MAX_POS), -1, dtype=np.int32)  # vela de creación

    # ── Estado agregado shape (B,) ────────────────────────────────────────────
    usdt_ini_disp = float(SALDO_USDT_INICIAL - USDT_RESERVA)
    usdt        = np.full(B, usdt_ini_disp, dtype=np.float64)
    btc_libre   = np.zeros(B, dtype=np.float64)
    btc_en_pos  = np.zeros(B, dtype=np.float64)
    usdt_en_pos = np.zeros(B, dtype=np.float64)
    n_pos       = np.zeros(B, dtype=np.int32)
    last_op     = np.zeros(B, dtype=np.float64)   # solo actualiza en compras (y ventas si p_lop_sell)

    peak_port  = np.full(B, float(SALDO_USDT_INICIAL), dtype=np.float64)
    max_dd     = np.zeros(B, dtype=np.float64)
    compras    = np.zeros(B, dtype=np.int32)
    ventas     = np.zeros(B, dtype=np.int32)
    _idx       = np.arange(B)
    ath_cummax = np.maximum.accumulate(highs)

    # ── Loop principal ─────────────────────────────────────────────────────────
    for i in range(n_velas):
        low_i   = float(lows[i])
        high_i  = float(highs[i])
        close_i = float(closes[i])
        ath_i   = float(ath_cummax[i])
        alcista = closes[i] >= opens[i]

        # ── COMPRA ─────────────────────────────────────────────────────────────
        def _buy():
            # Modo A (sin pos abiertos): limit = ATH × (1 - pct_caida_ath)
            # Modo B (con pos abiertos): limit = last_op × (1 - pct_caida)
            lb      = np.where(n_pos == 0,
                               ath_i * (1.0 - p_caida_ath),
                               last_op * (1.0 - p_caida))
            pct_cap = _pct_capital_vec(lb, ath_i, p_floor, p_pendiente)
            ua      = usdt * pct_cap / 100.0
            btc_adq = (ua * (1.0 - comm)) / np.maximum(lb, 1e-30)
            mask    = (low_i <= lb) & (usdt > 1e-8) & (ua > 1e-8) & (n_pos < MAX_POS)
            bi = np.where(mask)[0]
            if len(bi):
                slots = n_pos[bi]
                pos_entry [bi, slots] = lb[bi]
                pos_usdt  [bi, slots] = ua[bi]
                pos_btc   [bi, slots] = btc_adq[bi]
                pos_candle[bi, slots] = i
                do_b = mask.astype(np.float64)
                usdt       .__isub__(do_b * ua)
                btc_en_pos .__iadd__(do_b * btc_adq)
                usdt_en_pos.__iadd__(do_b * ua)
                n_pos      .__iadd__(mask.astype(np.int32))
                np.copyto(last_op, lb, where=mask)   # siempre actualiza en compra
                compras.__iadd__(mask.astype(np.int32))

        # ── CASCADE SELL ────────────────────────────────────────────────────────
        # Orden LIFO: el último comprado es el más barato → TP más bajo → vende primero.
        # FIX 1: excluye posiciones creadas en esta misma vela (pos_candle < i).
        # FIX 2: last_op solo se actualiza si p_lop_sell=True para esa combinación.
        def _sell():
            for _ in range(MAX_POS):
                last_slot = np.maximum(n_pos - 1, 0)
                le   = pos_entry [_idx, last_slot]   # precio entrada posición
                lc   = pos_candle[_idx, last_slot]   # vela de creación
                lu   = pos_usdt  [_idx, last_slot]   # USDT invertido exacto
                lb_b = pos_btc   [_idx, last_slot]   # BTC exacto
                tp   = le * (1.0 + p_venta)          # TP de esta posición

                # FIX 1: excluir posición creada en esta vela
                sell_mask = (n_pos > 0) & (high_i >= tp) & (lc < i)
                if not np.any(sell_mask):
                    break

                # BTC a vender: recupera exactamente lu USDT neto (comm incluida)
                btc_v   = np.where(sell_mask,
                              np.minimum(lu / np.maximum(tp * factor_sell, 1e-30), lb_b),
                              0.0)
                btc_acc = np.where(sell_mask, lb_b - btc_v, 0.0)
                do_s    = sell_mask.astype(np.float64)

                usdt       .__iadd__(do_s * lu)
                btc_libre  .__iadd__(do_s * btc_acc)
                btc_en_pos .__isub__(do_s * lb_b)
                usdt_en_pos.__isub__(do_s * lu)
                n_pos.__isub__(sell_mask.astype(np.int32))
                ventas.__iadd__(sell_mask.astype(np.int32))

                # Limpiar slot vendido
                si = np.where(sell_mask)[0]
                if len(si):
                    ss = last_slot[si]
                    pos_entry [si, ss] = 0.0
                    pos_usdt  [si, ss] = 0.0
                    pos_btc   [si, ss] = 0.0
                    pos_candle[si, ss] = -1

                # FIX 2: last_op solo actualiza en ventas si la combinación lo pide
                lop_upd = sell_mask & p_lop_sell
                if np.any(lop_upd):
                    np.copyto(last_op, tp, where=lop_upd)

        if alcista:
            _buy(); _sell()
        else:
            _sell(); _buy()

        # Mark-to-market y drawdown
        btc_tot_i = btc_libre + btc_en_pos
        port      = usdt + USDT_RESERVA + btc_tot_i * close_i
        peak_port = np.maximum(peak_port, port)
        dd        = np.where(peak_port > 0,
                             (peak_port - port) / peak_port * 100.0, 0.0)
        max_dd    = np.maximum(max_dd, dd)

    # ── Métricas finales ───────────────────────────────────────────────────────
    last_close = float(closes[-1])
    btc_total  = btc_libre + btc_en_pos
    portfolio  = usdt + USDT_RESERVA + btc_total * last_close
    pnl_pct    = (portfolio - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100.0

    return {
        "pnl_pct"      : pnl_pct,
        "portfolio"    : portfolio,
        "usdt_final"   : usdt + USDT_RESERVA,
        "btc_libre"    : btc_libre,
        "btc_en_pos"   : btc_en_pos,
        "btc_total"    : btc_total,
        "n_pos_final"  : n_pos,
        "compras"      : compras,
        "ventas"       : ventas,
        "total_trades" : compras + ventas,
        "max_dd"       : max_dd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grid search exhaustivo con tres min-heaps (pnl, btc_libre, btc_total).
    Mantiene en RAM solo los TOP_HEAP mejores por cada métrica.
    La unión deduplicada → ≤ TOP_HEAP × 3 filas únicas en el CSV.
    """
    pct_caida_ath_vals = _rango(PCT_CAIDA_ATH_I, PCT_CAIDA_ATH_F, PCT_CAIDA_ATH_P)
    pct_caida_vals     = _rango(PCT_CAIDA_I,     PCT_CAIDA_F,     PCT_CAIDA_P)
    pct_venta_vals     = _rango(PCT_VENTA_I,     PCT_VENTA_F,     PCT_VENTA_P)
    floor_pct_vals     = _rango(FLOOR_PCT_I,     FLOOR_PCT_F,     FLOOR_PCT_P, entero=True)
    pendiente_vals     = _rango(PENDIENTE_I,     PENDIENTE_F,     PENDIENTE_P, entero=True)
    last_op_vals       = LAST_OP_VALS   # [False, True]

    total_combos = (len(pct_caida_ath_vals) * len(pct_caida_vals) *
                    len(pct_venta_vals) * len(floor_pct_vals) *
                    len(pendiente_vals) * len(last_op_vals))

    print(f"\n{'═'*72}")
    print(f"  GRID SEARCH — {total_combos:,} combinaciones")
    print(f"{'═'*72}")
    print(f"  PCT_CAIDA_ATH  : {pct_caida_ath_vals[0]*100:.1f}% → {pct_caida_ath_vals[-1]*100:.1f}%"
          f"  ({len(pct_caida_ath_vals)} vals)")
    print(f"  PCT_CAIDA      : {pct_caida_vals[0]*100:.1f}% → {pct_caida_vals[-1]*100:.1f}%"
          f"  ({len(pct_caida_vals)} vals)")
    print(f"  PCT_VENTA      : {pct_venta_vals[0]*100:.1f}% → {pct_venta_vals[-1]*100:.1f}%"
          f"  ({len(pct_venta_vals)} vals)")
    print(f"  FLOOR_PCT      : {floor_pct_vals[0]} → {floor_pct_vals[-1]}"
          f"  ({len(floor_pct_vals)} vals)")
    print(f"  PENDIENTE      : {pendiente_vals[0]} → {pendiente_vals[-1]}"
          f"  ({len(pendiente_vals)} vals)")
    print(f"  LAST_OP_ON_SELL: {last_op_vals}  (2 vals)")
    print(f"  Top-K por métrica: {TOP_HEAP}  (máx ~{TOP_HEAP*3:,} filas en disco)")
    print(f"{'═'*72}\n")

    # Construir array de parámetros shape (M, 6)
    # col5 = 0.0 (False) / 1.0 (True)
    param_combos = list(product(
        pct_caida_ath_vals, pct_caida_vals, pct_venta_vals,
        floor_pct_vals, pendiente_vals, last_op_vals,
    ))
    combos_arr = np.array(
        [[ca, pc, pv, float(fl), float(pend), float(lop)]
         for ca, pc, pv, fl, pend, lop in param_combos],
        dtype=np.float64
    )
    M = len(combos_arr)

    heap_pnl   = _TopHeap(TOP_HEAP)
    heap_libre = _TopHeap(TOP_HEAP)
    heap_total = _TopHeap(TOP_HEAP)

    # Tamaño de lote: sweet spot L3 cache (~8000 combos × 6 params × 8B ≈ 4MB)
    BATCH   = 8000
    n_lotes = math.ceil(M / BATCH)

    print(f"  Modo  : vectorizado NumPy — lotes de {BATCH:,}  ({n_lotes} lotes)")
    n_cols_por_pos = 4  # entry, usdt, btc, candle
    ram_mb = min(BATCH, M) * MAX_POS * n_cols_por_pos * 8 / 1024**2
    print(f"  RAM   : ~{ram_mb:.0f} MB por lote  (MAX_POS={MAX_POS})\n")

    t0          = time.time()
    combos_done = 0
    best_pnl    = -999.0
    all_pnl     = np.empty(M, dtype=np.float64)
    all_lib     = np.empty(M, dtype=np.float64)
    all_tot     = np.empty(M, dtype=np.float64)

    for lote_idx in range(n_lotes):
        s = lote_idx * BATCH
        e = min(s + BATCH, M)
        lote_arr    = combos_arr[s:e]
        lote_params = param_combos[s:e]

        arrs = simular_vectorizado(df, lote_arr)

        pnl_arr = arrs["pnl_pct"]
        lib_arr = arrs["btc_libre"]
        tot_arr = arrs["btc_total"]
        all_pnl[s:e] = pnl_arr
        all_lib[s:e] = lib_arr
        all_tot[s:e] = tot_arr

        # Umbral dinámico — solo construye dicts para candidatos reales
        min_pnl = heap_pnl.min_val
        min_lib = heap_libre.min_val
        min_tot = heap_total.min_val
        cand = np.where(
            (pnl_arr > min_pnl) |
            (lib_arr > min_lib) |
            (tot_arr > min_tot)
        )[0]

        for idx in cand:
            ca, pc, pv, fl, pend, lop = lote_params[idx]
            row = {
                "pct_caida_ath"  : round(float(ca),   6),
                "pct_caida"      : round(float(pc),   6),
                "pct_venta"      : round(float(pv),   6),
                "floor_pct"      : int(fl),
                "pendiente"      : int(pend),
                "last_op_on_sell": bool(lop),
                "pnl_pct"        : round(float(arrs["pnl_pct"][idx]),        4),
                "portfolio_final": round(float(arrs["portfolio"][idx]),       2),
                "usdt_final"     : round(float(arrs["usdt_final"][idx]),      2),
                "btc_libre"      : round(float(arrs["btc_libre"][idx]),       8),
                "btc_en_pos"     : round(float(arrs["btc_en_pos"][idx]),      8),
                "btc_total"      : round(float(arrs["btc_total"][idx]),       8),
                "n_pos_final"    : int(arrs["n_pos_final"][idx]),
                "total_compras"  : int(arrs["compras"][idx]),
                "total_ventas"   : int(arrs["ventas"][idx]),
                "total_trades"   : int(arrs["total_trades"][idx]),
                "max_drawdown"   : round(float(arrs["max_dd"][idx]),          2),
            }
            heap_pnl.push_if_better(float(pnl_arr[idx]), row)
            heap_libre.push_if_better(float(lib_arr[idx]), row)
            heap_total.push_if_better(float(tot_arr[idx]), row)

        combos_done += len(lote_arr)
        best_pnl     = max(best_pnl, float(pnl_arr.max()))
        elapsed      = time.time() - t0
        eta          = elapsed / combos_done * (M - combos_done) if combos_done else 0
        survivors    = len({id(r) for h in (heap_pnl, heap_libre, heap_total)
                            for r in h.to_list()})
        print(f"  [{lote_idx+1:>3}/{n_lotes}]  {combos_done:>7,}/{M:,}  "
              f"{elapsed:>6.1f}s  ETA:{eta:>5.1f}s  "
              f"top:{survivors:>4}  mejor PnL:{best_pnl:>+8.2f}%")

    elapsed_total = time.time() - t0
    print(f"\n✓ Completado en {elapsed_total:.1f}s  "
          f"({M / elapsed_total:,.0f} backtests/s)")

    # Unión deduplicada de los tres heaps
    seen, all_rows = set(), []
    for heap in (heap_pnl, heap_libre, heap_total):
        for row in heap.to_list():
            key = (row["pct_caida_ath"], row["pct_caida"], row["pct_venta"],
                   row["floor_pct"], row["pendiente"], row["last_op_on_sell"])
            if key not in seen:
                seen.add(key); all_rows.append(row)

    print(f"\n  ✓ Filas únicas conservadas: {len(all_rows):,}  "
          f"(máx teórico: {TOP_HEAP * 3:,})")

    print(f"\n  Estadísticas sobre {M:,} combinaciones:")
    print(f"    PnL%       : {all_pnl.min():+.2f}%  →  {all_pnl.max():+.2f}%"
          f"  (mediana {np.median(all_pnl):+.2f}%)")
    print(f"    BTC libre  : {all_lib.min():.6f}  →  {all_lib.max():.6f} ₿")
    print(f"    BTC total  : {all_tot.min():.6f}  →  {all_tot.max():.6f} ₿")
    print(f"    PnL% > 0   : {(all_pnl > 0).sum():,}  "
          f"({(all_pnl > 0).mean()*100:.1f}%)")

    df_res = pd.DataFrame(all_rows)[COL_ORDER]
    return df_res.sort_values("pnl_pct", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _agregar_scores(df_res: pd.DataFrame, precio_final: float) -> pd.DataFrame:
    df = df_res.copy()
    df["btc_libre_usd"] = df["btc_libre"] * precio_final
    df["btc_total_usd"] = df["btc_total"] * precio_final

    for col, norm_col in [("pnl_pct", "pnl_norm"),
                          ("btc_libre", "btcl_norm"),
                          ("btc_total", "btct_norm")]:
        mn, mx = df[col].min(), df[col].max()
        span   = mx - mn if mx > mn else 1.0
        df[norm_col] = (df[col] - mn) / span

    df["combo_score"] = (df["pnl_norm"] * 0.5 +
                         df["btcl_norm"] * 0.3 +
                         df["btct_norm"] * 0.2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados(df_res: pd.DataFrame):
    # CSV completo
    df_res.to_csv(OUT_CSV, index=False)
    print(f"  ✓ CSV  : {OUT_CSV}  ({len(df_res):,} filas)")

    # JSON con los top-N de cada ranking
    top_json = {
        "meta": {
            "fecha_inicio"        : FECHA_INICIO,
            "fecha_fin"           : FECHA_FIN,
            "total_filas"         : len(df_res),
            "pct_caida_ath_range" : [PCT_CAIDA_ATH_I, PCT_CAIDA_ATH_F, PCT_CAIDA_ATH_P],
            "pct_caida_range"     : [PCT_CAIDA_I,     PCT_CAIDA_F,     PCT_CAIDA_P],
            "pct_venta_range"     : [PCT_VENTA_I,     PCT_VENTA_F,     PCT_VENTA_P],
            "floor_pct_range"     : [FLOOR_PCT_I,     FLOOR_PCT_F,     FLOOR_PCT_P],
            "pendiente_range"     : [PENDIENTE_I,     PENDIENTE_F,     PENDIENTE_P],
            "last_op_vals"        : [str(v) for v in LAST_OP_VALS],
            "top_heap"            : TOP_HEAP,
        },
        "ranking_pnl"   : df_res.sort_values("pnl_pct",   ascending=False)
                                 .head(TOP_N).to_dict(orient="records"),
        "ranking_btclib": df_res.sort_values("btc_libre",  ascending=False)
                                 .head(TOP_N).to_dict(orient="records"),
        "ranking_btctot": df_res.sort_values("btc_total",  ascending=False)
                                 .head(TOP_N).to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(top_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ JSON : {OUT_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

# Columnas de la tabla visual
_COLS = [
    "#", "CA%", "C%", "V%", "FL%", "PEND", "LOP",
    "PnL%", "Port $", "BTC Lib ₿", "BTC Pos ₿", "BTC Tot ₿",
    "C", "V", "Pos", "DD%",
]

_RANKINGS = [
    ("pnl_pct",   7,  plt.cm.RdYlGn,
     "Ranking 1 — Mejor PnL% en Cartera Total",
     "Retorno % sobre portfolio valorizado al precio de cierre",
     OUT_TABLA_PNL),
    ("btc_libre", 9, plt.cm.YlOrRd,
     "Ranking 2 — Mayor BTC Libre Acumulado",
     "BTC neto acumulado en ventas (ganancia en BTC puro)",
     OUT_TABLA_LIB),
    ("btc_total", 11, plt.cm.PuBuGn,
     "Ranking 3 — Mayor BTC Total",
     "BTC libre + BTC en posiciones abiertas al cierre",
     OUT_TABLA_TOT),
]


def _fig_tabla(df_res: pd.DataFrame, sort_col: str, hi_col_idx: int,
               cmap, titulo: str, subtitulo: str, out_path: str):
    """Genera tabla visual PNG de los top-TOP_N resultados."""
    ranked = df_res.sort_values(sort_col, ascending=False).head(TOP_N)
    rows   = []
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        rows.append([
            str(rank),
            f"{r.pct_caida_ath*100:.1f}%",
            f"{r.pct_caida*100:.1f}%",
            f"{r.pct_venta*100:.1f}%",
            f"{int(r.floor_pct)}%",
            str(int(r.pendiente)),
            "T" if r.last_op_on_sell else "F",
            f"{r.pnl_pct:+.2f}%",
            f"${r.portfolio_final:,.2f}",
            f"{r.btc_libre:.6f}",
            f"{r.btc_en_pos:.6f}",
            f"{r.btc_total:.6f}",
            str(int(r.total_compras)),
            str(int(r.total_ventas)),
            str(int(r.n_pos_final)),
            f"{r.max_drawdown:.1f}%",
        ])

    n_rows = len(rows)
    fig_h  = max(4.5, 0.36 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(22, fig_h))
    fig.patch.set_facecolor("#f4f6fa")
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=_COLS, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.38)

    # Cabecera
    for j in range(len(_COLS)):
        cell = table[0, j]
        cell.set_facecolor("#1a2540"); cell.set_text_props(color="white", fontweight="bold")

    # Colores de filas
    norm = mcolors.Normalize(
        vmin=ranked[sort_col].min(),
        vmax=ranked[sort_col].max()
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        base   = "#f7f9fc" if i % 2 == 0 else "#ffffff"
        hi_col = mcolors.to_hex(sm.to_rgba(r[sort_col]))
        for j in range(len(_COLS)):
            cell = table[i, j]
            cell.set_facecolor(hi_col if j == hi_col_idx else base)
            if i <= 3:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#dde3ef")

    fig.suptitle(
        f"{titulo}\n"
        f"{subtitulo}\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"CA%=PCT_CAIDA_ATH · C%=PCT_CAIDA · V%=PCT_VENTA · "
        f"FL%=FLOOR_PCT · PEND=PENDIENTE · LOP=LAST_OP_ON_SELL (T/F)",
        fontsize=10, fontweight="bold", color="#1a2540", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Tabla {sort_col}: {out_path}")


def fig_tres_tablas(df_res: pd.DataFrame):
    for sort_col, hi_idx, cmap, titulo, subtitulo, out_path in _RANKINGS:
        _fig_tabla(df_res, sort_col, hi_idx, cmap, titulo, subtitulo, out_path)


def fig_analisis_variables(df_res: pd.DataFrame):
    """
    Barras de mediana PnL% + línea BTC libre para cada valor de cada variable.
    """
    variables = [
        ("pct_caida_ath",  "PCT_CAIDA_ATH",   lambda x: f"{x*100:.1f}%"),
        ("pct_caida",      "PCT_CAIDA",        lambda x: f"{x*100:.1f}%"),
        ("pct_venta",      "PCT_VENTA",        lambda x: f"{x*100:.1f}%"),
        ("floor_pct",      "FLOOR_PCT",        lambda x: f"{int(x)}%"),
        ("pendiente",      "PENDIENTE",        lambda x: str(int(x))),
        ("last_op_on_sell","LAST_OP_ON_SELL",  lambda x: "T" if x else "F"),
    ]

    fig = plt.figure(figsize=(24, 9))
    fig.patch.set_facecolor("#f4f6fa")
    gs  = GridSpec(1, 6, figure=fig, hspace=0.5, wspace=0.42)
    axes = [fig.add_subplot(gs[0, c]) for c in range(6)]

    for ax, (col, label, fmt) in zip(axes, variables):
        grp     = df_res.groupby(col)
        vals    = sorted(df_res[col].unique())
        med_pnl = [grp.get_group(v)["pnl_pct"].median()   for v in vals]
        max_pnl = [grp.get_group(v)["pnl_pct"].max()      for v in vals]
        q25_pnl = [grp.get_group(v)["pnl_pct"].quantile(.25) for v in vals]
        q75_pnl = [grp.get_group(v)["pnl_pct"].quantile(.75) for v in vals]
        med_lib = [grp.get_group(v)["btc_libre"].median() for v in vals]

        x      = np.arange(len(vals))
        colors = ["#e74c3c" if m < 0 else "#27ae60" for m in med_pnl]
        yerr   = [np.array(med_pnl) - np.array(q25_pnl),
                  np.array(q75_pnl) - np.array(med_pnl)]

        bars = ax.bar(x, med_pnl, color=colors, alpha=0.82, zorder=3,
                      yerr=yerr, error_kw={"ecolor": "#555", "capsize": 3,
                                           "linewidth": 0.9, "alpha": 0.6})

        for xi, mp in zip(x, max_pnl):
            ax.annotate("▲", (xi, mp), ha="center", va="bottom",
                        fontsize=8, color="#c0392b")

        ax2 = ax.twinx()
        ax2.plot(x, med_lib, color="#2980b9", marker="o", linewidth=1.5,
                 markersize=4, zorder=4, label="BTC libre mediana")
        ax2.tick_params(axis="y", labelsize=7, colors="#2980b9")
        ax2.set_ylabel("BTC libre ₿", fontsize=7, color="#2980b9")

        ax.set_facecolor("#f8fafd")
        ax.set_xticks(x)
        ax.set_xticklabels([fmt(v) for v in vals], fontsize=8,
                           rotation=30 if len(vals) > 6 else 0)
        ax.set_title(label, fontsize=9, fontweight="bold", color="#1a2540", pad=6)
        ax.set_xlabel("Valor", fontsize=7)
        ax.set_ylabel("PnL% mediano", fontsize=7)
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(axis="y", alpha=0.3, color="#dde3ef")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Impacto por Variable — Grid UPO Acumulator\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"Barras = mediana PnL%  ·  error = IQR (Q25–Q75)  ·  ▲ = máximo  ·  "
        f"Línea azul = mediana BTC libre (eje derecho)",
        fontsize=10.5, fontweight="bold", color="#1a2540", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(OUT_ANALISIS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Análisis de variables: {OUT_ANALISIS}")


def fig_scatter(df_res: pd.DataFrame):
    """Scatter PnL% vs BTC libre. Color = BTC total. Tamaño ∝ 1/MaxDD."""
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#f4f6fa")
    ax.set_facecolor("#ffffff")

    btc_tot = df_res["btc_total"].values
    pnl     = df_res["pnl_pct"].values
    btcl    = df_res["btc_libre"].values
    dd      = df_res["max_drawdown"].values

    bmin, bmax = btc_tot.min(), btc_tot.max()
    bspan      = bmax - bmin if bmax > bmin else 1
    dd_norm    = (dd - dd.min()) / max(dd.max() - dd.min(), 1)
    sizes      = 15 + (1 - dd_norm) * 60

    sc = ax.scatter(pnl, btcl, c=btc_tot, cmap="YlOrRd",
                    s=sizes, alpha=0.55, linewidths=0,
                    vmin=bmin, vmax=bmax)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("BTC total (₿)", fontsize=10)

    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)

    # Resaltar top-3 de cada ranking
    for col, color, lbl in [
        ("pnl_pct",   "#27ae60", "Top PnL%"),
        ("btc_libre", "#e67e22", "Top BTC Libre"),
        ("btc_total", "#8e44ad", "Top BTC Total"),
    ]:
        top3 = df_res.nlargest(3, col)
        ax.scatter(top3["pnl_pct"], top3["btc_libre"],
                   color=color, s=180, marker="*",
                   zorder=10, label=lbl, edgecolors="black", linewidths=0.5)
        for _, r in top3.iterrows():
            ax.annotate(
                f"CA={r.pct_caida_ath*100:.1f}% C={r.pct_caida*100:.1f}%\n"
                f"V={r.pct_venta*100:.1f}% FL={int(r.floor_pct)}%\n"
                f"PEND={int(r.pendiente)} LOP={'T' if r.last_op_on_sell else 'F'}",
                xy=(r.pnl_pct, r.btc_libre),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6.5, color="#1a2540",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.78, edgecolor=color, linewidth=0.8),
            )

    ax.set_xlabel("PnL% sobre cartera total", fontsize=11)
    ax.set_ylabel("BTC libre acumulado (₿)", fontsize=11)
    ax.set_title(
        f"Trade-off PnL% vs BTC Libre — Grid UPO Acumulator\n"
        f"{FECHA_INICIO} → {FECHA_FIN}  ·  "
        f"Color = BTC total  ·  Tamaño ∝ 1/MaxDD  ·  ★ = top-3 de cada ranking",
        fontsize=10, fontweight="bold", color="#1a2540",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, color="#dde3ef")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Scatter PnL% vs BTC libre: {OUT_SCATTER}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(df_res: pd.DataFrame):
    sep = "═" * 88
    print(f"\n{sep}")
    print("  RESUMEN — OPTIMIZADOR GRID UPO ACUMULATOR")
    print(sep)
    print(f"  Período       : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Filas únicas  : {len(df_res):,}  (top-{TOP_HEAP} × 3 métricas)")
    print(f"  PnL%  rango   : {df_res['pnl_pct'].min():+.2f}%  →  {df_res['pnl_pct'].max():+.2f}%")
    print(f"  PnL%  mediana : {df_res['pnl_pct'].median():+.2f}%")
    print(f"  BTC libre max : {df_res['btc_libre'].max():.6f} ₿")
    print(f"  BTC total max : {df_res['btc_total'].max():.6f} ₿")

    hdr = (f"  {'#':>3}  {'CA%':>5}  {'C%':>5}  {'V%':>5}  "
           f"{'FL':>3}  {'PEND':>4}  {'LOP':>3}  "
           f"{'PnL%':>8}  {'Port$':>10}  {'BTClib':>9}  {'BTCtot':>9}  "
           f"{'C':>4}  {'V':>4}  {'Pos':>3}  {'DD%':>5}")

    rankings = [
        ("RANKING 1 — MEJOR PnL%",      df_res.sort_values("pnl_pct",   ascending=False)),
        ("RANKING 2 — MAYOR BTC LIBRE",  df_res.sort_values("btc_libre", ascending=False)),
        ("RANKING 3 — MAYOR BTC TOTAL",  df_res.sort_values("btc_total", ascending=False)),
    ]

    for titulo, ranked in rankings:
        print(f"\n  {'─'*86}")
        print(f"  {titulo}")
        print(f"  {'─'*86}")
        print(hdr)
        print(f"  {'─'*86}")
        for rank, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
            marker = "★" if rank <= 3 else " "
            print(
                f"  {marker}{rank:>2}.  "
                f"{r.pct_caida_ath*100:>4.1f}%  "
                f"{r.pct_caida*100:>4.1f}%  "
                f"{r.pct_venta*100:>4.1f}%  "
                f"{int(r.floor_pct):>3}  "
                f"{int(r.pendiente):>4}  "
                f"{'T' if r.last_op_on_sell else 'F':>3}  "
                f"{r.pnl_pct:>+7.2f}%  "
                f"${r.portfolio_final:>9,.2f}  "
                f"{r.btc_libre:>9.6f}  "
                f"{r.btc_total:>9.6f}  "
                f"{int(r.total_compras):>4}  "
                f"{int(r.total_ventas):>4}  "
                f"{int(r.n_pos_final):>3}  "
                f"{r.max_drawdown:>5.1f}%"
            )

    print(f"\n  {'─'*86}")
    print("  PARÁMETROS DOMINANTES EN EL TOP-10% DE CADA RANKING")
    print(f"  {'─'*86}")
    param_cols = [
        ("pct_caida_ath",   "PCT_CAIDA_ATH",    lambda v: f"{v*100:.1f}%"),
        ("pct_caida",       "PCT_CAIDA",         lambda v: f"{v*100:.1f}%"),
        ("pct_venta",       "PCT_VENTA",         lambda v: f"{v*100:.1f}%"),
        ("floor_pct",       "FLOOR_PCT",         lambda v: f"{int(v)}%"),
        ("pendiente",       "PENDIENTE",         lambda v: str(int(v))),
        ("last_op_on_sell", "LAST_OP_ON_SELL",   lambda v: "True" if v else "False"),
    ]
    for titulo, ranked in rankings:
        top10pct = ranked.head(max(1, len(ranked) // 10))
        print(f"\n  Dominantes — {titulo}")
        for col, lbl, fmt in param_cols:
            val  = top10pct[col].mode().iloc[0]
            freq = (top10pct[col] == val).mean() * 100
            print(f"    {lbl:<20}: {fmt(val)}  ({freq:.0f}% del top 10%)")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMIZADOR — GRID UPO ACUMULATOR                                  ║")
    print("║  NumPy Vectorizado · Órdenes Límite · Progresión Lineal de Compra  ║")
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
    fig_analisis_variables(df_res)
    fig_scatter(df_res)

    print(f"\n{'═'*72}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*72}")
    for fname in [OUT_CSV, OUT_JSON, OUT_TABLA_PNL, OUT_TABLA_LIB,
                  OUT_TABLA_TOT, OUT_ANALISIS, OUT_SCATTER]:
        print(f"  · {fname}")
    print(f"{'═'*72}")
    print(f"✓ Total completado en {time.time() - t_total:.1f}s\n")


if __name__ == "__main__":
    main()