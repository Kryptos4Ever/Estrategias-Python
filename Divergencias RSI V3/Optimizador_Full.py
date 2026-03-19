"""
Optimizador Completo — Todas las Variables de Config
═════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Grid Search Exhaustivo

Variables optimizadas (8):
  Continuas  → especificar inicio / fin / paso
    RSI_LENGTH      : período del RSI
    N               : ventana de búsqueda del extremo local
    FLOOR_PCT       : piso del ciclo como % del ATH
    FACTOR_CAIDA    : curvatura del gradiente de compra
    FACTOR_SUBIDA   : curvatura del gradiente de venta

  Booleanas  → se prueban ambos valores siempre
    GUARDIA_COMPRA        : bloqueo compras sobre PP
    GUARDIA_PRECIO_COMPRA : bloqueo compras sobre mínimo comprado
    GUARDIA_PRECIO_VENTA  : bloqueo ventas bajo máximo vendido

Variables fijas (tomadas de config sin modificar):
    USDT_RESERVA_PCT, BTC_PCT_TO_ACCUMULATE, COMMISSION_PCT,
    FECHA_INICIO, FECHA_FIN, SALDO_USDT_INICIAL

Métrica de ranking: PnL%  (sin sesgo de tendencia)
  → mide retorno sobre capital puro, independiente de si el mercado
    sube o baja. No premia acumulación ni liquidación de forma explícita.
  Métricas secundarias incluidas en salida:
    portfolio_final, positions_count_final, total_trades,
    max_drawdown_aprox, pnl_por_trade

Salidas:
  optimizacion_full.csv            → todos los resultados
  optimizacion_full_top.json       → top N con config completa
  optimizacion_full_top_tabla.png  → tabla visual top 25
  optimizacion_full_analisis.png   → análisis por variable (8 paneles)
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
# Modificá los valores de inicio / fin / paso para ajustar la granularidad.
# Las variables booleanas siempre prueban ambos valores (True y False).

# ── Variables continuas ───────────────────────────────────────────────────────
RSI_LENGTH_INICIO    =  5  ;  RSI_LENGTH_FIN    = 21  ;  RSI_LENGTH_PASO    = 1
N_INICIO             =  5  ;  N_FIN             = 51  ;  N_PASO             = 1
FLOOR_PCT_INICIO     = 10  ;  FLOOR_PCT_FIN     = 25  ;  FLOOR_PCT_PASO     = 1
FACTOR_CAIDA_INICIO  = 1.0 ;  FACTOR_CAIDA_FIN  = 5.0 ;  FACTOR_CAIDA_PASO  = 1.0
FACTOR_SUBIDA_INICIO = 0.5 ;  FACTOR_SUBIDA_FIN = 3.0 ;  FACTOR_SUBIDA_PASO = 0.5

# ── Archivos de salida ────────────────────────────────────────────────────────
OUT_CSV        = "optimizacion_full.csv"
OUT_JSON       = "optimizacion_full_top.json"
OUT_TABLA_PNL  = "optimizacion_full_ranking_pnl.png"
OUT_TABLA_BTC  = "optimizacion_full_ranking_btc.png"
OUT_TABLA_EQ   = "optimizacion_full_ranking_equilibrio.png"
OUT_GUARDIAS   = "optimizacion_full_guardias.png"
OUT_ANALISIS   = "optimizacion_full_analisis.png"
TOP_N          = 25     # filas en cada tabla y en el JSON


# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE RANGOS
# ══════════════════════════════════════════════════════════════════════════════

def _rango(inicio, fin, paso, entero=False):
    """Genera lista [inicio, inicio+paso, ..., fin] con tolerancia float."""
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
    guardia_compra:      bool,
    guardia_prec_compra: bool,
    guardia_prec_venta:  bool,
) -> dict:

    n_velas           = len(lows)
    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    precio_min_comp   = math.inf   # ratchet de compra
    precio_max_venta  = 0.0        # ratchet de venta
    ath               = float(highs[0])
    compras = ventas  = 0

    # Para drawdown aproximado: mínimo portfolio visto durante la sesión
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

        # Drawdown aproximado (valorado a close)
        port_now = usdt_balance + btc_en_posiciones * closes[i]
        if port_now > port_max:
            port_max = port_now
        elif port_max > 0:
            dd = (port_max - port_now) / port_max * 100
            if dd > dd_max:
                dd_max = dd

        # ── SEÑAL DE COMPRA ───────────────────────────────────────────────────
        señal_compra = False
        if lows[i] < wl.min():
            idx_min = i - N + int(wl.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                señal_compra = True

        # ── SEÑAL DE VENTA ────────────────────────────────────────────────────
        señal_venta = False
        if not señal_compra and highs[i] > wh.max():
            idx_max = i - N + int(wh.argmax())
            if rsi_high[i] < rsi_high[idx_max]:
                señal_venta = True

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

            btc_acum    = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_vender  = btc_slot - btc_acum
            ub          = btc_vender * highs[i]
            com         = ub * (COMMISSION_PCT / 100)
            un          = ub - com
            cp          = usdt_invertido * (btc_slot / btc_en_posiciones)
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
    # positions_count: compras - ventas (sesgo al cierre; 0 = equilibrado)
    positions_count = compras - ventas

    return {
        "pnl_pct"           : round(pnl_pct,          4),
        "portfolio_final"   : round(portfolio_final,   2),
        "usdt_final"        : round(usdt_balance,      2),
        "btc_posiciones"    : round(btc_en_posiciones, 8),
        "precio_prom_fin"   : round(pp_fin,            2),
        "total_trades"      : total_trades,
        "total_compras"     : compras,
        "total_ventas"      : ventas,
        "positions_count"   : positions_count,
        "pnl_por_trade"     : round(pnl_por_trade,     4),
        "max_drawdown"      : round(dd_max,            2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame) -> pd.DataFrame:

    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)

    # Rangos
    rsi_lengths   = _rango(RSI_LENGTH_INICIO,    RSI_LENGTH_FIN,    RSI_LENGTH_PASO,    entero=True)
    ns            = _rango(N_INICIO,             N_FIN,             N_PASO,             entero=True)
    floor_pcts    = _rango(FLOOR_PCT_INICIO,     FLOOR_PCT_FIN,     FLOOR_PCT_PASO)
    factor_caidas = _rango(FACTOR_CAIDA_INICIO,  FACTOR_CAIDA_FIN,  FACTOR_CAIDA_PASO)
    factor_subidas= _rango(FACTOR_SUBIDA_INICIO, FACTOR_SUBIDA_FIN, FACTOR_SUBIDA_PASO)
    guardias_bool = [True, False]

    total_combos = (len(rsi_lengths) * len(ns) * len(floor_pcts) *
                    len(factor_caidas) * len(factor_subidas) * 8)  # 2^3 guardias

    print(f"\n{'═'*66}")
    print(f"  GRID SEARCH EXHAUSTIVO — {total_combos:,} combinaciones")
    print(f"{'═'*66}")
    print(f"  RSI_LENGTH    : {rsi_lengths[0]} → {rsi_lengths[-1]}  "
          f"(paso {RSI_LENGTH_PASO}, {len(rsi_lengths)} vals)")
    print(f"  N             : {ns[0]} → {ns[-1]}  "
          f"(paso {N_PASO}, {len(ns)} vals)")
    print(f"  FLOOR_PCT     : {floor_pcts[0]} → {floor_pcts[-1]}  "
          f"(paso {FLOOR_PCT_PASO}, {len(floor_pcts)} vals)")
    print(f"  FACTOR_CAIDA  : {factor_caidas[0]} → {factor_caidas[-1]}  "
          f"(paso {FACTOR_CAIDA_PASO}, {len(factor_caidas)} vals)")
    print(f"  FACTOR_SUBIDA : {factor_subidas[0]} → {factor_subidas[-1]}  "
          f"(paso {FACTOR_SUBIDA_PASO}, {len(factor_subidas)} vals)")
    print(f"  Guardias      : GUARDIA_COMPRA × GUARDIA_PRECIO_COMPRA × "
          f"GUARDIA_PRECIO_VENTA  = 2³ = 8")
    print(f"  ── Fijos desde config ──────────────────────────────────────────")
    print(f"  Período       : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  USDT reserva  : {USDT_RESERVA_PCT}%   BTC acum: {BTC_PCT_TO_ACCUMULATE}%   "
          f"Comisión: {COMMISSION_PCT}%")
    print(f"  Métrica rank  : PnL%  (sin sesgo de tendencia)")
    print(f"{'═'*66}\n")

    # Caché RSI: solo depende de rsi_length
    print("  Pre-calculando RSI...")
    rsi_cache = {}
    for rsi_len in rsi_lengths:
        rsi_cache[rsi_len] = (
            calcular_rsi(df["low"],  rsi_len),
            calcular_rsi(df["high"], rsi_len),
        )
    print(f"  ✓ {len(rsi_cache)} pares RSI calculados\n")

    resultados = []
    t0         = time.time()
    idx        = 0

    for rsi_len, n, fp, fc, fs, gc, gpc, gpv in product(
        rsi_lengths, ns, floor_pcts, factor_caidas, factor_subidas,
        guardias_bool, guardias_bool, guardias_bool
    ):
        idx += 1
        rsi_l, rsi_h = rsi_cache[rsi_len]

        m = ejecutar_backtest(
            lows, highs, closes, rsi_l, rsi_h,
            n, fp, fc, fs, gc, gpc, gpv,
        )
        m.update({
            "rsi_length"          : rsi_len,
            "N"                   : n,
            "floor_pct"           : fp,
            "factor_caida"        : fc,
            "factor_subida"       : fs,
            "guardia_compra"      : gc,
            "guardia_prec_compra" : gpc,
            "guardia_prec_venta"  : gpv,
        })
        resultados.append(m)

        if idx % 5000 == 0 or idx == total_combos:
            elapsed = time.time() - t0
            eta     = elapsed / idx * (total_combos - idx)
            best    = max(r["pnl_pct"] for r in resultados)
            pct_done = idx / total_combos * 100
            print(f"  [{idx:>6}/{total_combos:,}] {pct_done:5.1f}%  "
                  f"{elapsed:>6.1f}s  ETA:{eta:>5.1f}s  "
                  f"mejor PnL: {best:>+8.2f}%")

    print(f"\n✓ Completado en {time.time() - t0:.1f}s")

    df_res = pd.DataFrame(resultados)
    col_order = [
        "rsi_length", "N", "floor_pct", "factor_caida", "factor_subida",
        "guardia_compra", "guardia_prec_compra", "guardia_prec_venta",
        "pnl_pct", "portfolio_final", "usdt_final", "btc_posiciones",
        "precio_prom_fin", "total_trades", "total_compras", "total_ventas",
        "positions_count", "pnl_por_trade", "max_drawdown",
    ]
    df_res = df_res[col_order]

    # Verificación explícita: las 8 combinaciones de guardias deben estar presentes
    n_combos_guardias = df_res.groupby(
        ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    ).ngroups
    print(f"\n  ✓ Verificación de guardias: {n_combos_guardias}/8 combinaciones probadas")
    for (gc, gpc, gpv), grp in df_res.groupby(
        ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    ):
        print(f"    GC={gc} GPC={gpc} GPV={gpv}: {len(grp):,} combos "
              f"| mejor PnL: {grp['pnl_pct'].max():+.2f}%")

    # Ordenar por PnL% descendente (sort estable: no favorece ninguna combinación booleana)
    df_res = df_res.sort_values("pnl_pct", ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res


def _agregar_scores(df_res: pd.DataFrame, precio_final: float) -> pd.DataFrame:
    """
    Agrega columnas derivadas usadas para los 3 rankings:

    btc_value       : btc_posiciones × precio_final  (valor USD del BTC acumulado)

    equilibrio_score: media geométrica de las normas [0,1] de pnl_pct y btc_value.
      · Normalización min-max independiente para cada métrica.
      · score = sqrt(pnl_norm × btc_norm)
      · La media geométrica penaliza los extremos: una combinación excelente en
        una sola dimensión pero pésima en la otra obtiene score cercano a 0.
      · Solo considera valores donde pnl_pct > 0 y btc_value > 0 para la norma,
        pero calcula el score para todas las filas (los negativos quedan en 0).
    """
    df = df_res.copy()

    df["btc_value"] = df["btc_posiciones"] * precio_final

    # Normalizar pnl_pct a [0, 1]
    pnl_min, pnl_max = df["pnl_pct"].min(), df["pnl_pct"].max()
    pnl_span = pnl_max - pnl_min if pnl_max > pnl_min else 1.0
    df["pnl_norm"] = ((df["pnl_pct"] - pnl_min) / pnl_span).clip(0, 1)

    # Normalizar btc_value a [0, 1]
    btc_min, btc_max = df["btc_value"].min(), df["btc_value"].max()
    btc_span = btc_max - btc_min if btc_max > btc_min else 1.0
    df["btc_norm"] = ((df["btc_value"] - btc_min) / btc_span).clip(0, 1)

    # Media geométrica
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
                  .head(TOP_N)
                  .reset_index(drop=True)
                  .reset_index()
                  .rename(columns={"index": "rank"})
                  .assign(rank=lambda d: d["rank"] + 1)
                  .to_dict(orient="records"))

    # Resumen por combinación de guardias para el JSON
    guard_stats = []
    for (gc, gpc, gpv), grp in df_res.groupby(
        ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]
    ):
        guard_stats.append({
            "guardia_compra"      : gc,
            "guardia_prec_compra" : gpc,
            "guardia_prec_venta"  : gpv,
            "n_combos"            : len(grp),
            "mejor_pnl_pct"       : round(grp["pnl_pct"].max(), 4),
            "mediana_pnl_pct"     : round(grp["pnl_pct"].median(), 4),
            "mejor_btc_value"     : round(grp["btc_value"].max(), 2),
            "mejor_eq_score"      : round(grp["equilibrio_score"].max(), 6),
        })

    payload = {
        "meta": {
            "fecha_inicio"                : FECHA_INICIO,
            "fecha_fin"                   : FECHA_FIN,
            "saldo_inicial"               : SALDO_USDT_INICIAL,
            "usdt_reserva_pct"            : USDT_RESERVA_PCT,
            "btc_pct_to_accumulate"       : BTC_PCT_TO_ACCUMULATE,
            "commission_pct"              : COMMISSION_PCT,
            "rsi_range"                   : [RSI_LENGTH_INICIO, RSI_LENGTH_FIN, RSI_LENGTH_PASO],
            "n_range"                     : [N_INICIO, N_FIN, N_PASO],
            "floor_pct_range"             : [FLOOR_PCT_INICIO, FLOOR_PCT_FIN, FLOOR_PCT_PASO],
            "factor_caida_range"          : [FACTOR_CAIDA_INICIO, FACTOR_CAIDA_FIN, FACTOR_CAIDA_PASO],
            "factor_subida_range"         : [FACTOR_SUBIDA_INICIO, FACTOR_SUBIDA_FIN, FACTOR_SUBIDA_PASO],
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

# Configuración de cada ranking:
#   (sort_col, highlight_col_idx, cmap_highlight, título, subtítulo, archivo)
_RANKINGS = [
    (
        "pnl_pct",
        9,                          # índice de columna "PnL %" en la tabla
        plt.cm.RdYlGn,
        "Ranking 1 — Máximo PnL%",
        "Prioriza el retorno total sobre el capital inicial · Sin sesgo de tendencia",
        OUT_TABLA_PNL,
    ),
    (
        "btc_value",
        12,                         # índice de columna "BTC $" en la tabla
        plt.cm.YlOrRd,
        "Ranking 2 — Máxima Acumulación BTC",
        "Prioriza el valor USD del BTC mantenido en posiciones al cierre",
        OUT_TABLA_BTC,
    ),
    (
        "equilibrio_score",
        16,                         # índice de columna "Equilibrio" en la tabla
        plt.cm.PuBuGn,
        "Ranking 3 — Mejor Equilibrio PnL% × BTC",
        "Media geométrica de las normas [0,1] de PnL% y BTC acumulado  "
        "· Penaliza combinaciones excelentes en una sola dimensión",
        OUT_TABLA_EQ,
    ),
]


def _fig_tabla(df_res: pd.DataFrame, sort_col: str, highlight_col: int,
               cmap, titulo: str, subtitulo: str, filename: str):
    """Genera una tabla PNG con el top N ordenado por sort_col."""
    top = (df_res.sort_values(sort_col, ascending=False)
                 .head(TOP_N)
                 .reset_index(drop=True))

    cols = [
        "#", "RSI", "N", "FLOOR%", "F_CAIDA", "F_SUBIDA",
        "G_PP", "G_P_C", "G_P_V",
        "PnL %", "Portfolio $", "USDT $",
        "BTC pos.", "BTC $", "Trades", "C/V",
        "Equilibrio", "PnL/Trade", "MaxDD%",
    ]

    def bs(v): return "✓" if v else "✗"

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        rows.append([
            str(rank),
            str(r.rsi_length), str(r.N),
            f"{r.floor_pct:.0f}%",
            f"{r.factor_caida:.1f}", f"{r.factor_subida:.1f}",
            bs(r.guardia_compra),
            bs(r.guardia_prec_compra),
            bs(r.guardia_prec_venta),
            f"{r.pnl_pct:+.2f}%",
            f"${r.portfolio_final:,.2f}",
            f"${r.usdt_final:,.2f}",
            f"{r.btc_posiciones:.6f}",
            f"${r.btc_value:,.2f}",
            str(int(r.total_trades)),
            f"{int(r.total_compras)}/{int(r.total_ventas)}",
            f"{r.equilibrio_score:.4f}",
            f"{r.pnl_por_trade:+.3f}%",
            f"{r.max_drawdown:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(24, TOP_N * 0.42 + 3.2))
    fig.patch.set_facecolor("#f4f6fa")
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.52)

    # Cabecera
    for j in range(len(cols)):
        c = table[0, j]
        c.set_facecolor("#1a2540")
        c.set_text_props(color="white", fontweight="bold")

    # Resaltar cabecera de la columna de ranking activo
    table[0, highlight_col].set_facecolor("#e67e22")

    # Filas alternas
    for i in range(1, len(rows) + 1):
        bg = "#eef2f9" if i % 2 == 0 else "#ffffff"
        for j in range(len(cols)):
            table[i, j].set_facecolor(bg)

    # Top 3 medallas
    for i, color in enumerate(["#ffd700", "#d8d8d8", "#cd7f32"][:min(3, len(rows))], 1):
        for j in range(len(cols)):
            table[i, j].set_facecolor(color)
            table[i, j].set_text_props(fontweight="bold")

    # Gradiente de color en la columna destacada
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
        f"{titulo}\n"
        f"{subtitulo}\n"
        f"Período: {FECHA_INICIO} → {FECHA_FIN}  |  Total combinaciones: {len(df_res):,}  |  "
        f"G_PP=Guardia PP · G_P_C=Guardia precio compra · G_P_V=Guardia precio venta",
        fontsize=10, fontweight="bold", color="#1a2540", pad=13,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ {titulo}: {filename}")


def fig_tres_tablas(df_res: pd.DataFrame):
    for sort_col, hl_col, cmap, titulo, subtitulo, filename in _RANKINGS:
        _fig_tabla(df_res, sort_col, hl_col, cmap, titulo, subtitulo, filename)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN 3 — ANÁLISIS DE GUARDIAS (tabla + barras)
# ══════════════════════════════════════════════════════════════════════════════

def fig_analisis_guardias(df_res: pd.DataFrame):
    """
    Tabla con el MEJOR resultado de cada una de las 8 combinaciones de guardias
    (máximo PnL%, máximo BTC$ y máximo equilibrio_score).
    Permite verificar inequívocamente que las 3 guardias fueron probadas
    en ambos valores y ver qué combinación domina en cada criterio.
    """
    GCOLS = ["guardia_compra", "guardia_prec_compra", "guardia_prec_venta"]

    # ── Agrupar por combinación de guardias ───────────────────────────────────
    rows_tabla = []
    for (gc, gpc, gpv), grp in df_res.groupby(GCOLS):
        best_pnl = grp.loc[grp["pnl_pct"].idxmax()]
        best_btc = grp.loc[grp["btc_value"].idxmax()]
        best_eq  = grp.loc[grp["equilibrio_score"].idxmax()]
        rows_tabla.append({
            "gc": gc, "gpc": gpc, "gpv": gpv,
            "n_combos"     : len(grp),
            # Mejor PnL%
            "pnl_max"      : best_pnl["pnl_pct"],
            "pnl_port"     : best_pnl["portfolio_final"],
            "pnl_params"   : (f"RSI={int(best_pnl.rsi_length)} N={int(best_pnl.N)} "
                              f"FL={best_pnl.floor_pct:.0f}% "
                              f"FC={best_pnl.factor_caida:.1f} FS={best_pnl.factor_subida:.1f}"),
            # Mejor acumulación BTC
            "btc_max_val"  : best_btc["btc_value"],
            "btc_btc"      : best_btc["btc_posiciones"],
            "btc_params"   : (f"RSI={int(best_btc.rsi_length)} N={int(best_btc.N)} "
                              f"FL={best_btc.floor_pct:.0f}% "
                              f"FC={best_btc.factor_caida:.1f} FS={best_btc.factor_subida:.1f}"),
            # Mejor equilibrio
            "eq_score"     : best_eq["equilibrio_score"],
            "eq_pnl"       : best_eq["pnl_pct"],
            "eq_btc"       : best_eq["btc_value"],
            "eq_params"    : (f"RSI={int(best_eq.rsi_length)} N={int(best_eq.N)} "
                              f"FL={best_eq.floor_pct:.0f}% "
                              f"FC={best_eq.factor_caida:.1f} FS={best_eq.factor_subida:.1f}"),
            # Estadísticas de la distribución
            "pnl_median"   : grp["pnl_pct"].median(),
            "pnl_positive" : (grp["pnl_pct"] > 0).sum(),
        })

    df_g = pd.DataFrame(rows_tabla).sort_values("pnl_max", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor("#f4f6fa")
    gs = GridSpec(2, 1, figure=fig, hspace=0.45,
                  height_ratios=[2.8, 1])

    # ── Panel superior: tabla ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")

    def bs(v): return "✓" if v else "✗"
    def lbl(gc, gpc, gpv): return f"GC={bs(gc)} GPC={bs(gpc)} GPV={bs(gpv)}"

    cols_t = [
        "Guardias\n(GC/GPC/GPV)", "Combos\nprobadas",
        "── Mejor PnL% ──────────────────────",
        "PnL%", "Portfolio $", "Params",
        "── Mejor Acumulación BTC ───────────",
        "BTC $", "BTC ₿", "Params",
        "── Mejor Equilibrio ────────────────",
        "Eq.Score", "PnL%", "BTC $", "Params",
        "Mediana\nPnL%", "Combos\n>0%",
    ]

    rows_t = []
    for _, r in df_g.iterrows():
        rows_t.append([
            lbl(r.gc, r.gpc, r.gpv),
            f"{int(r.n_combos):,}",
            "", f"{r.pnl_max:+.2f}%", f"${r.pnl_port:,.2f}", r.pnl_params,
            "", f"${r.btc_max_val:,.2f}", f"{r.btc_btc:.6f} ₿", r.btc_params,
            "", f"{r.eq_score:.4f}", f"{r.eq_pnl:+.2f}%", f"${r.eq_btc:,.2f}", r.eq_params,
            f"{r.pnl_median:+.2f}%", f"{int(r.pnl_positive):,}",
        ])

    table = ax_t.table(cellText=rows_t, colLabels=cols_t,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.7)

    # Cabecera principal
    for j in range(len(cols_t)):
        c = table[0, j]
        c.set_facecolor("#1a2540")
        c.set_text_props(color="white", fontweight="bold")

    # Cabeceras de sección (columnas separadoras)
    SECTION_COLS = [2, 6, 10]  # índices de columnas "──"
    HDR_COLORS   = ["#1e6b3c", "#7b3f00", "#1a3a6b"]
    for ci, color in zip(SECTION_COLS, HDR_COLORS):
        table[0, ci].set_facecolor(color)

    # Colorear filas por ranking PnL
    pnl_vals = [r["pnl_max"] for r in rows_tabla]
    pnl_vals_sorted = sorted(pnl_vals, reverse=True)
    for i in range(1, len(rows_t) + 1):
        bg = "#eef2f9" if i % 2 == 0 else "#ffffff"
        for j in range(len(cols_t)):
            table[i, j].set_facecolor(bg)

    # Top 3 en columna PnL%
    for i, color in enumerate(["#ffd700", "#d8d8d8", "#cd7f32"][:min(3, len(rows_t))], 1):
        table[i, 3].set_facecolor(color)
        table[i, 3].set_text_props(fontweight="bold")
    # Top 1 en BTC y Equilibrio
    table[1, 7].set_facecolor("#ffe4b5")
    table[1, 11].set_facecolor("#d4edda")

    # Colorear columna guardias: verde si mayormente True
    for i, (_, r) in enumerate(df_g.iterrows(), 1):
        n_true = sum([r.gc, r.gpc, r.gpv])
        intensity = n_true / 3
        color = mcolors.to_hex(plt.cm.RdYlGn(0.3 + 0.7 * intensity))
        table[i, 0].set_facecolor(color)
        table[i, 0].set_text_props(fontweight="bold")

    ax_t.set_title(
        f"Análisis de Combinaciones de Guardias — {FECHA_INICIO} → {FECHA_FIN}\n"
        f"Cada fila = mejor resultado de las {df_res['guardia_compra'].shape[0] // 8:,} "
        f"combinaciones con esa configuración de guardias  "
        f"(Total combinaciones probadas: {len(df_res):,}  |  "
        f"Combinaciones de guardias: 2³ = 8)\n"
        f"GC=GUARDIA_COMPRA  ·  GPC=GUARDIA_PRECIO_COMPRA  ·  GPV=GUARDIA_PRECIO_VENTA",
        fontsize=10, fontweight="bold", color="#1a2540", pad=12,
    )

    # ── Panel inferior: barras comparativas ───────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor("#ffffff")

    x      = np.arange(8)
    labels = [lbl(r.gc, r.gpc, r.gpv) for _, r in df_g.iterrows()]
    pnl_m  = [r.pnl_max    for _, r in df_g.iterrows()]
    btc_m  = [r.btc_max_val for _, r in df_g.iterrows()]

    # Normalizar BTC a misma escala visual que PnL
    btc_scaled = [b / max(btc_m) * max(abs(p) for p in pnl_m) for b in btc_m]

    w = 0.35
    b1 = ax_b.bar(x - w/2, pnl_m,      w, label="Mejor PnL%",
                  color=[plt.cm.RdYlGn(0.3 + 0.7 * (p - min(pnl_m)) /
                          max(max(pnl_m) - min(pnl_m), 0.001)) for p in pnl_m],
                  edgecolor="white", alpha=0.9)
    b2 = ax_b.bar(x + w/2, btc_scaled, w, label="Mejor BTC$ (normalizado)",
                  color="#3498db", alpha=0.65, edgecolor="white")

    ax_b.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    ax_b.set_ylabel("Mejor PnL% por combinación", fontsize=9)
    ax_b.set_title("Comparativa entre las 8 combinaciones de guardias — "
                   "barras azules = BTC$ normalizado a la escala de PnL%",
                   fontsize=9)
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.grid(True, axis="y", alpha=0.3, color="#dde3ef")

    # Anotar valor de PnL sobre cada barra
    for bar, val in zip(b1, pnl_m):
        y_pos = bar.get_height() + (0.3 if val >= 0 else -1.2)
        ax_b.text(bar.get_x() + bar.get_width() / 2, y_pos,
                  f"{val:+.1f}%", ha="center", fontsize=7,
                  fontweight="bold", color="#1a2540")

    plt.savefig(OUT_GUARDIAS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Análisis guardias: {OUT_GUARDIAS}")

def fig_analisis_variables(df_res: pd.DataFrame):
    """
    Un panel por variable. Muestra la distribución del PnL%
    agrupando por cada valor posible de esa variable.
    Permite ver de un vistazo qué valores dominan el top.
    """
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_facecolor("#f4f6fa")
    gs  = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    variables = [
        ("rsi_length",          "RSI_LENGTH",          False),
        ("N",                   "N (ventana)",          False),
        ("floor_pct",           "FLOOR_PCT (%)",        False),
        ("factor_caida",        "FACTOR_CAIDA",         False),
        ("factor_subida",       "FACTOR_SUBIDA",        False),
        ("guardia_compra",      "GUARDIA_COMPRA",       True),
        ("guardia_prec_compra", "GUARDIA_PRECIO_COMPRA",True),
        ("guardia_prec_venta",  "GUARDIA_PRECIO_VENTA", True),
    ]

    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(2)]

    for ax, (col, label, es_bool) in zip(axes, variables):
        ax.set_facecolor("#ffffff")
        grp = df_res.groupby(col)["pnl_pct"]

        medians = grp.median().sort_index()
        q25     = grp.quantile(0.25).sort_index()
        q75     = grp.quantile(0.75).sort_index()
        tops    = grp.max().sort_index()

        x_vals  = list(medians.index)
        x_pos   = range(len(x_vals))

        if es_bool:
            colors = ["#2ecc71" if v else "#e74c3c" for v in x_vals]
            x_labels = ["✓ True" if v else "✗ False" for v in x_vals]
        else:
            # Gradiente de color por mediana
            norm_vals = medians.values
            vmin, vmax = norm_vals.min(), norm_vals.max()
            span = vmax - vmin if vmax > vmin else 1
            colors = [mcolors.to_hex(plt.cm.RdYlGn(0.2 + 0.8 * (v - vmin) / span))
                      for v in norm_vals]
            x_labels = [str(v) for v in x_vals]

        # Barras de mediana
        bars = ax.bar(x_pos, medians.values, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.7, zorder=3)

        # Rango IQR (Q25–Q75) como barras de error
        yerr_low  = medians.values - q25.values
        yerr_high = q75.values - medians.values
        ax.errorbar(x_pos, medians.values,
                    yerr=[yerr_low, yerr_high],
                    fmt="none", color="#555", linewidth=1.2,
                    capsize=4, zorder=4)

        # Máximos como puntos
        ax.scatter(x_pos, tops.values,
                   marker="^", color="#e67e22", s=45, zorder=5,
                   label="Máximo PnL%")

        # Línea de referencia en 0
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)

        # Anotar mediana sobre cada barra
        for xi, (med, top_v) in enumerate(zip(medians.values, tops.values)):
            ax.text(xi, med + (q75.values[xi] - med) * 0.15,
                    f"{med:+.1f}%", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#1a2540")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_ylabel("PnL% (mediana ± IQR)", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold", color="#1a2540", pad=6)
        ax.grid(True, axis="y", alpha=0.3, color="#dde3ef")

        if not es_bool:
            ax.legend(["▲ Máximo PnL%"], fontsize=7, loc="upper right",
                      handletextpad=0.3, borderpad=0.4)

    fig.suptitle(
        f"Análisis de Impacto por Variable  ·  {FECHA_INICIO} → {FECHA_FIN}\n"
        f"Cada barra = mediana PnL% de todas las combinaciones con ese valor  "
        f"|  Barras de error = IQR (Q25–Q75)  |  ▲ = máximo PnL%\n"
        f"Total combinaciones: {len(df_res):,}  |  Ranking: PnL% (sin sesgo de tendencia)",
        fontsize=11, fontweight="bold", color="#1a2540", y=1.01,
    )
    plt.savefig(OUT_ANALISIS, dpi=130, bbox_inches="tight", facecolor="#f4f6fa")
    plt.close()
    print(f"  ✓ Análisis: {OUT_ANALISIS}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(df_res: pd.DataFrame):
    sep = "═" * 70

    print(f"\n{sep}")
    print("  RESUMEN — OPTIMIZACIÓN COMPLETA")
    print(sep)
    print(f"  Período       : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Combinaciones : {len(df_res):,}")
    print(f"  PnL% rango    : {df_res['pnl_pct'].min():+.2f}%  →  {df_res['pnl_pct'].max():+.2f}%")
    print(f"  PnL% mediana  : {df_res['pnl_pct'].median():+.2f}%")
    print(f"  PnL% positivos: {(df_res['pnl_pct'] > 0).sum():,}  "
          f"({(df_res['pnl_pct'] > 0).mean()*100:.1f}%)")

    def bs(v): return "✓" if v else "✗"

    hdr = (f"  {'#':>3}  {'RSI':>4}  {'N':>3}  {'FL%':>4}  "
           f"{'FC':>5}  {'FS':>5}  {'GC':>3}  {'GPC':>4}  {'GPV':>4}  "
           f"{'PnL%':>8}  {'BTC $':>9}  {'Eq.Score':>9}  {'Trades':>6}  {'DD%':>6}")

    rankings = [
        ("RANKING PNL% — Máximo retorno sobre capital",
         df_res.sort_values("pnl_pct", ascending=False)),
        ("RANKING BTC — Máxima acumulación en posiciones",
         df_res.sort_values("btc_value", ascending=False)),
        ("RANKING EQUILIBRIO — Mejor balance PnL% × BTC acumulado",
         df_res.sort_values("equilibrio_score", ascending=False)),
    ]

    for titulo, ranked in rankings:
        print(f"\n  {'─'*68}")
        print(f"  {titulo}")
        print(f"  {'─'*68}")
        print(hdr)
        print(f"  {'─'*68}")
        for rank, (_, r) in enumerate(ranked.head(15).iterrows(), 1):
            marker = "★" if rank <= 3 else " "
            print(f"  {marker}{rank:>2}.  "
                  f"{r.rsi_length:>4}  {r.N:>3}  {r.floor_pct:>3.0f}%  "
                  f"{r.factor_caida:>5.1f}  {r.factor_subida:>5.1f}  "
                  f"{bs(r.guardia_compra):>3}  {bs(r.guardia_prec_compra):>4}  "
                  f"{bs(r.guardia_prec_venta):>4}  "
                  f"{r.pnl_pct:>+7.2f}%  ${r.btc_value:>8,.2f}  "
                  f"{r.equilibrio_score:>9.4f}  "
                  f"{int(r.total_trades):>6}  {r.max_drawdown:>5.1f}%")

    # Valores dominantes en el top 10% de cada ranking
    for titulo, ranked in rankings:
        top10 = ranked.head(max(1, len(df_res) // 10))
        print(f"\n  Dominantes — {titulo}")
        for col, lbl in [("rsi_length","RSI"), ("N","N"),
                          ("floor_pct","FLOOR%"), ("factor_caida","F_CAIDA"),
                          ("factor_subida","F_SUBIDA")]:
            val  = top10[col].mode().iloc[0]
            freq = (top10[col] == val).mean() * 100
            print(f"    {lbl:<12}: {val}  ({freq:.0f}% del top 10%)")
        for col, lbl in [("guardia_compra","G_COMPRA"),
                          ("guardia_prec_compra","G_PREC_C"),
                          ("guardia_prec_venta","G_PREC_V")]:
            pct  = top10[col].mean() * 100
            print(f"    {lbl:<12}: {'✓ True' if pct >= 50 else '✗ False'}  ({pct:.0f}% True)")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   OPTIMIZADOR COMPLETO — DIVERGENCIA RSI · TODAS LAS VARIABLES  ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    print("Cargando datos...")
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py")
        return
    print(f"  Velas  : {len(df):,}")
    print(f"  Desde  : {df['datetime'].iloc[0]}")
    print(f"  Hasta  : {df['datetime'].iloc[-1]}\n")

    df_res = optimizar(df)

    # Agregar columnas de scoring (btc_value, pnl_norm, btc_norm, equilibrio_score)
    precio_final = float(df["close"].iloc[-1])
    df_res = _agregar_scores(df_res, precio_final)

    print("\nGuardando resultados...")
    guardar_resultados(df_res)

    imprimir_resumen(df_res)

    print("Generando visualizaciones...")
    fig_tres_tablas(df_res)
    fig_analisis_guardias(df_res)
    fig_analisis_variables(df_res)

    print(f"\n{'═'*66}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*66}")
    for f in [OUT_CSV, OUT_JSON, OUT_TABLA_PNL, OUT_TABLA_BTC, OUT_TABLA_EQ,
              OUT_GUARDIAS, OUT_ANALISIS]:
        print(f"  · {f}")
    print(f"{'═'*66}")
    print("✓ Proceso completado.\n")


if __name__ == "__main__":
    main()