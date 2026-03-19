"""
Optimizador de Parámetros — EMA200 + Bollinger %B + Williams %R
────────────────────────────────────────────────────────────────
Objetivo: encontrar los umbrales óptimos de los 3 indicadores para que
las señales de compra coincidan con mínimos locales y las de venta
con máximos locales (según método tendencia zig-zag del Analizador).

Método:
  1. Carga precios de la DB y filtra por rango de fechas del config.
  2. Calcula EMA200_dist, BB_%B y Williams %R con los parámetros base.
  3. Detecta extremos locales reales (mínimos / máximos).
  4. Grid search sobre los umbrales de señal:
       - DIST_EMA_BUY/SELL   : cuánto % debe estar el precio bajo/sobre EMA
       - BB_BUY/SELL          : umbral de %B para compra/venta
       - WILLIAMS_BUY/SELL    : umbral de Williams para compra/venta
  5. Para cada combinación evalúa:
       · Precision_compra  : % de señales de compra dentro de ±TOLERANCIA velas de un mínimo
       · Precision_venta   : % de señales de venta dentro de ±TOLERANCIA velas de un máximo
       · Score             : media ponderada de ambas precisiones
       · n_compras / n_ventas: cantidad de señales (penaliza si hay < MIN_SEÑALES)
  6. Imprime top N combinaciones y genera:
       · optimizacion_resultados.json
       · optimizacion_resultados.png (heatmaps de score)
       · config_EMA_BB_Williams_OPTIMIZADO.py  listo para usar

Uso:
  python Optimizar_EMA_BB_Williams.py

Ajusta los rangos de búsqueda en la sección PARÁMETROS DE BÚSQUEDA.
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
from config_EMA_BB_Williams import (
    DB_PATH, FECHA_INICIO, FECHA_FIN,
    EMA_LENGTH, BB_LENGTH, BB_STD, WILLIAMS_LENGTH,
)

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

# ═══════════════════════════════════════════════════════════════
#  PARÁMETROS DE BÚSQUEDA  (ajusta a tu gusto)
# ═══════════════════════════════════════════════════════════════

# Detección de extremos por tendencia zig-zag
PCT_TENDENCIA = 0.05       # 5% de movimiento para confirmar extremo

# Tolerancia de coincidencia: ±N velas alrededor del extremo
TOLERANCIA_VELAS = 60      # ±60 min = ±1 hora

# Mínimo de señales para que la combinación sea válida
MIN_SEÑALES = 5

# Top N resultados a mostrar
TOP_N = 20

# ── Umbrales de EMA200 distancia (%) ─────────────────────────
# Positivo = precio BAJO la EMA (para compra) / SOBRE la EMA (para venta)
DIST_EMA_BUY_RANGE  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
DIST_EMA_SELL_RANGE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# ── Umbrales de Bollinger %B ──────────────────────────────────
BB_BUY_RANGE  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BB_SELL_RANGE = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# ── Umbrales de Williams %R ───────────────────────────────────
WILLIAMS_BUY_RANGE  = [-90, -85, -80, -75, -70, -60]
WILLIAMS_SELL_RANGE = [-30, -25, -20, -15, -10]

# ── Pesos del score final ─────────────────────────────────────
PESO_PRECISION_COMPRA = 0.5
PESO_PRECISION_VENTA  = 0.5

# ═══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    print(f"Conectando a {DB_PATH}...")
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, open, high, low, close, volume
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
    print(f"Velas cargadas : {len(df):,}  "
          f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
    return df


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
# DETECCIÓN DE EXTREMOS (zig-zag por tendencia)
# ─────────────────────────────────────────────────────────────

def extremos_por_tendencia(close: pd.Series, pct: float):
    """
    Zigzag: mínimo confirmado si desde ese punto sube pct% sin bajar pct%.
            máximo confirmado si desde ese punto baja pct% sin subir pct%.
    Devuelve dos arrays booleanos del tamaño de close.
    """
    arr  = close.values
    n    = len(arr)
    mins = np.zeros(n, dtype=bool)
    maxs = np.zeros(n, dtype=bool)

    # Estado: buscando mínimo (0) o máximo (1)
    estado   = 0    # empezamos buscando mínimo
    idx_ext  = 0
    val_ext  = arr[0]

    for i in range(1, n):
        v = arr[i]
        if estado == 0:          # buscando mínimo
            if v < val_ext:
                val_ext = v
                idx_ext = i
            elif v >= val_ext * (1 + pct):
                mins[idx_ext] = True
                estado  = 1
                val_ext = v
                idx_ext = i
        else:                    # buscando máximo
            if v > val_ext:
                val_ext = v
                idx_ext = i
            elif v <= val_ext * (1 - pct):
                maxs[idx_ext] = True
                estado  = 0
                val_ext = v
                idx_ext = i

    return mins, maxs


# ─────────────────────────────────────────────────────────────
# MÉTRICAS DE CALIDAD DE SEÑAL
# ─────────────────────────────────────────────────────────────

def precision_señal(señal_idx: np.ndarray, extremos_idx: np.ndarray,
                    tolerancia: int, n_total: int) -> float:
    """
    Para cada señal, comprueba si hay un extremo dentro de ±tolerancia velas.
    Retorna el % de señales que coinciden con un extremo.
    """
    if len(señal_idx) == 0:
        return 0.0
    # Construir set de velas "cerca de un extremo"
    extremo_set = set()
    for e in extremos_idx:
        for offset in range(-tolerancia, tolerancia + 1):
            extremo_set.add(e + offset)
    hits = sum(1 for s in señal_idx if s in extremo_set)
    return hits / len(señal_idx)


def cobertura_extremos(señal_idx: np.ndarray, extremos_idx: np.ndarray,
                       tolerancia: int) -> float:
    """
    % de extremos que tienen al menos una señal en ±tolerancia velas.
    """
    if len(extremos_idx) == 0:
        return 0.0
    extremo_cubierto = 0
    señal_set = set(señal_idx)
    for e in extremos_idx:
        ventana = range(max(0, e - tolerancia), e + tolerancia + 1)
        if any(v in señal_set for v in ventana):
            extremo_cubierto += 1
    return extremo_cubierto / len(extremos_idx)


# ─────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────

def grid_search(df: pd.DataFrame, ema_dist: np.ndarray,
                bb_pct_b: np.ndarray, williams: np.ndarray,
                mins_idx: np.ndarray, maxs_idx: np.ndarray) -> list:

    total_combos = (len(DIST_EMA_BUY_RANGE) * len(DIST_EMA_SELL_RANGE) *
                    len(BB_BUY_RANGE) * len(BB_SELL_RANGE) *
                    len(WILLIAMS_BUY_RANGE) * len(WILLIAMS_SELL_RANGE))

    print(f"\nGrid search: {total_combos:,} combinaciones...")
    print(f"  Extremos detectados → mínimos: {len(mins_idx):,}  "
          f"máximos: {len(maxs_idx):,}")

    resultados = []
    t0 = time.time()
    done = 0

    # Pre-computar índices válidos (sin NaN)
    valid_mask = (~np.isnan(ema_dist)) & (~np.isnan(bb_pct_b)) & (~np.isnan(williams))
    valid_idx  = np.where(valid_mask)[0]

    ema_v  = ema_dist[valid_idx]
    bb_v   = bb_pct_b[valid_idx]
    wil_v  = williams[valid_idx]

    for (deb, des, bbu, bse, wbu, wse) in itertools.product(
            DIST_EMA_BUY_RANGE, DIST_EMA_SELL_RANGE,
            BB_BUY_RANGE, BB_SELL_RANGE,
            WILLIAMS_BUY_RANGE, WILLIAMS_SELL_RANGE):

        done += 1
        if done % 5000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (total_combos - done)
            print(f"  {done:>7,}/{total_combos:,}  ({done/total_combos*100:.1f}%)  "
                  f"ETA {eta:.0f}s", end="\r")

        # Señales de compra
        mask_buy  = (ema_v  <= -deb) & (bb_v <= bbu) & (wil_v <= wbu)
        mask_sell = (ema_v  >=  des) & (bb_v >= bse) & (wil_v >= wse)

        buy_local  = np.where(mask_buy)[0]
        sell_local = np.where(mask_sell)[0]

        buy_global  = valid_idx[buy_local]
        sell_global = valid_idx[sell_local]

        n_buy  = len(buy_global)
        n_sell = len(sell_global)

        if n_buy < MIN_SEÑALES or n_sell < MIN_SEÑALES:
            continue

        prec_buy   = precision_señal(buy_global,  mins_idx, TOLERANCIA_VELAS, len(df))
        prec_sell  = precision_señal(sell_global, maxs_idx, TOLERANCIA_VELAS, len(df))
        cob_buy    = cobertura_extremos(buy_global,  mins_idx, TOLERANCIA_VELAS)
        cob_sell   = cobertura_extremos(sell_global, maxs_idx, TOLERANCIA_VELAS)

        score = (PESO_PRECISION_COMPRA * prec_buy +
                 PESO_PRECISION_VENTA  * prec_sell)

        resultados.append({
            "dist_ema_buy"   : deb,
            "dist_ema_sell"  : des,
            "bb_buy"         : bbu,
            "bb_sell"        : bse,
            "williams_buy"   : wbu,
            "williams_sell"  : wse,
            "n_compras"      : n_buy,
            "n_ventas"       : n_sell,
            "precision_buy"  : round(prec_buy,  4),
            "precision_sell" : round(prec_sell, 4),
            "cobertura_buy"  : round(cob_buy,   4),
            "cobertura_sell" : round(cob_sell,  4),
            "score"          : round(score, 4),
        })

    elapsed = time.time() - t0
    print(f"\n  Completado en {elapsed:.1f}s  |  "
          f"Combinaciones válidas: {len(resultados):,}")

    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados


# ─────────────────────────────────────────────────────────────
# REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────

def imprimir_top(resultados: list, top_n: int):
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  TOP {top_n} COMBINACIONES POR SCORE  "
          f"(score = prec_compra×{PESO_PRECISION_COMPRA} + prec_venta×{PESO_PRECISION_VENTA})")
    print(sep)
    hdr = (f"  {'#':>3}  {'dEMA_B':>7} {'dEMA_S':>7} {'BB_B':>6} {'BB_S':>6} "
           f"{'W_B':>5} {'W_S':>5}  {'nBuy':>6} {'nSell':>6}  "
           f"{'Prec_B':>7} {'Prec_S':>7} {'CobB':>6} {'CobS':>6}  {'SCORE':>7}")
    print(hdr)
    print(f"  {'-'*105}")
    for i, r in enumerate(resultados[:top_n], 1):
        print(f"  {i:>3}  {r['dist_ema_buy']:>7.2f} {r['dist_ema_sell']:>7.2f} "
              f"{r['bb_buy']:>6.2f} {r['bb_sell']:>6.2f} "
              f"{r['williams_buy']:>5}  {r['williams_sell']:>5}  "
              f"{r['n_compras']:>6,} {r['n_ventas']:>6,}  "
              f"{r['precision_buy']:>7.1%} {r['precision_sell']:>7.1%} "
              f"{r['cobertura_buy']:>6.1%} {r['cobertura_sell']:>6.1%}  "
              f"{r['score']:>7.4f}")
    print(sep)


# ─────────────────────────────────────────────────────────────
# GENERAR CONFIG OPTIMIZADO
# ─────────────────────────────────────────────────────────────

def guardar_config_optimo(mejor: dict, df: pd.DataFrame):
    nombre = "config_EMA_BB_Williams_OPTIMIZADO.py"
    contenido = f"""# ============================================================
#  Configuración OPTIMIZADA — EMA200 + Bollinger + Williams
#  Generada por Optimizar_EMA_BB_Williams.py
#  Score: {mejor['score']:.4f}
#  Precisión compra: {mejor['precision_buy']:.1%}  |  Precisión venta: {mejor['precision_sell']:.1%}
#  Cobertura mínimos: {mejor['cobertura_buy']:.1%}  |  Cobertura máximos: {mejor['cobertura_sell']:.1%}
#  Señales → compras: {mejor['n_compras']:,}  ventas: {mejor['n_ventas']:,}
#  Período analizado: {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}
# ============================================================

# ── Base de datos ────────────────────────────────────────────
DB_PATH      = "{DB_PATH}"

# ── Resultados ───────────────────────────────────────────────
RESULTS_JSON = "strategy_results.json"

# ── Capital inicial ──────────────────────────────────────────
SALDO_USDT_INICIAL = 1000.0

# ── Rango de fechas ──────────────────────────────────────────
FECHA_INICIO = '{FECHA_INICIO}'
FECHA_FIN    = '{FECHA_FIN}'

# ── EMA200 ───────────────────────────────────────────────────
EMA_LENGTH     = {EMA_LENGTH}
DIST_EMA_BUY   = {mejor['dist_ema_buy']}    # optimizado
DIST_EMA_SELL  = {mejor['dist_ema_sell']}    # optimizado

# ── Bollinger Bands ──────────────────────────────────────────
BB_LENGTH      = {BB_LENGTH}
BB_STD         = {BB_STD}
BB_BUY         = {mejor['bb_buy']}    # optimizado
BB_SELL        = {mejor['bb_sell']}    # optimizado

# ── Williams %R ──────────────────────────────────────────────
WILLIAMS_LENGTH = {WILLIAMS_LENGTH}
WILLIAMS_BUY   = {mejor['williams_buy']}    # optimizado
WILLIAMS_SELL  = {mejor['williams_sell']}    # optimizado

# ── Gestión de capital ───────────────────────────────────────
USDT_PCT_TO_USE      = 5.0
BTC_PCT_TO_SELL      = 5.0
BTC_PCT_TO_ACCUMULATE = 1.0

# ── Comisión ─────────────────────────────────────────────────
COMMISSION_PCT = 0.1

# ── Referencia RSI ───────────────────────────────────────────
RSI_LENGTH           = 14
LOW_RSI_BUY_TRIGGER  = 30
HI_RSI_SELL_TRIGGER  = 70
"""
    with open(nombre, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"\nConfig optimizado guardado: {nombre}")
    print(f"  → Cópialo como config_EMA_BB_Williams.py para usar con la estrategia.")


# ─────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generar_graficos(resultados: list, df: pd.DataFrame,
                     mins_idx: np.ndarray, maxs_idx: np.ndarray,
                     ema_dist_arr: np.ndarray, bb_arr: np.ndarray,
                     wil_arr: np.ndarray):
    mejor = resultados[0]
    fig   = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colores = {"score": "#00d4aa", "buy": "#26a69a", "sell": "#ef5350",
               "min": "#26a69a", "max": "#ef5350", "precio": "#b0bec5"}

    # ── Panel 1: Precio + extremos + señales óptimas ──────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("#161b22")
    ax0.plot(df.index, df["close"], color=colores["precio"], lw=0.4, alpha=0.6, label="BTC Close")
    ax0.scatter(mins_idx, df["close"].iloc[mins_idx], marker="^",
                color=colores["min"], s=30, zorder=5, label=f"Mínimos ({len(mins_idx)})")
    ax0.scatter(maxs_idx, df["close"].iloc[maxs_idx], marker="v",
                color=colores["max"], s=30, zorder=5, label=f"Máximos ({len(maxs_idx)})")

    # Señales óptimas
    valid_mask = (~np.isnan(ema_dist_arr)) & (~np.isnan(bb_arr)) & (~np.isnan(wil_arr))
    valid_idx  = np.where(valid_mask)[0]
    ema_v = ema_dist_arr[valid_idx]; bb_v = bb_arr[valid_idx]; wil_v = wil_arr[valid_idx]

    buy_local  = np.where((ema_v  <= -mejor["dist_ema_buy"])  &
                          (bb_v  <=  mejor["bb_buy"])         &
                          (wil_v <=  mejor["williams_buy"]))[0]
    sell_local = np.where((ema_v  >=  mejor["dist_ema_sell"]) &
                          (bb_v  >=  mejor["bb_sell"])         &
                          (wil_v >=  mejor["williams_sell"]))[0]
    buy_g  = valid_idx[buy_local]
    sell_g = valid_idx[sell_local]

    ax0.scatter(buy_g,  df["close"].iloc[buy_g],  marker="o", color="#ffeb3b",
                s=15, zorder=6, alpha=0.7, label=f"Señal compra ({len(buy_g)})")
    ax0.scatter(sell_g, df["close"].iloc[sell_g], marker="o", color="#ff9800",
                s=15, zorder=6, alpha=0.7, label=f"Señal venta ({len(sell_g)})")

    ax0.set_title(f"Precio BTC + Extremos + Señales Óptimas  "
                  f"[Score={mejor['score']:.4f}  PrecB={mejor['precision_buy']:.1%}  "
                  f"PrecS={mejor['precision_sell']:.1%}]",
                  color="white", fontsize=10)
    ax0.legend(fontsize=7, facecolor="#161b22", labelcolor="white",
               loc="upper left", ncol=3)

    # ── Panel 2: Score vs BB_BUY (promedio sobre el resto) ───
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor("#161b22")
    bb_buy_scores = {}
    for r in resultados:
        k = r["bb_buy"]
        if k not in bb_buy_scores:
            bb_buy_scores[k] = []
        bb_buy_scores[k].append(r["score"])
    xb = sorted(bb_buy_scores.keys())
    ax1.bar([str(x) for x in xb],
            [np.mean(bb_buy_scores[x]) for x in xb],
            color=colores["buy"], alpha=0.8)
    ax1.set_title("Score medio por BB_BUY", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_xlabel("BB_BUY umbral", color="white", fontsize=8)
    ax1.set_ylabel("Score medio", color="white", fontsize=8)

    # ── Panel 3: Score vs WILLIAMS_BUY ───────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor("#161b22")
    w_buy_scores = {}
    for r in resultados:
        k = r["williams_buy"]
        if k not in w_buy_scores:
            w_buy_scores[k] = []
        w_buy_scores[k].append(r["score"])
    xw = sorted(w_buy_scores.keys())
    ax2.bar([str(x) for x in xw],
            [np.mean(w_buy_scores[x]) for x in xw],
            color=colores["buy"], alpha=0.8)
    ax2.set_title("Score medio por WILLIAMS_BUY", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_xlabel("Williams BUY umbral", color="white", fontsize=8)

    # ── Panel 4: Score vs DIST_EMA_BUY ───────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor("#161b22")
    ema_buy_scores = {}
    for r in resultados:
        k = r["dist_ema_buy"]
        if k not in ema_buy_scores:
            ema_buy_scores[k] = []
        ema_buy_scores[k].append(r["score"])
    xe = sorted(ema_buy_scores.keys())
    ax3.bar([str(x) for x in xe],
            [np.mean(ema_buy_scores[x]) for x in xe],
            color=colores["buy"], alpha=0.8)
    ax3.set_title("Score medio por DIST_EMA_BUY", color="white", fontsize=9)
    ax3.tick_params(colors="white", labelsize=7)
    ax3.set_xlabel("Distancia EMA BUY (%)", color="white", fontsize=8)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.spines[:].set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    fig.suptitle("Optimización de Parámetros — EMA200 + Bollinger + Williams %R",
                 color="white", fontsize=13, fontweight="bold")

    nombre = "optimizacion_resultados.png"
    plt.savefig(nombre, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Gráfico guardado: {nombre}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  OPTIMIZADOR DE PARÁMETROS — EMA200 + BOLLINGER + WILLIAMS")
    print("=" * 72)

    # 1. Datos
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos en la DB para el rango indicado.")
        return

    # 2. Indicadores
    print("\nCalculando indicadores...")
    ema200           = calc_ema(df["close"], EMA_LENGTH)
    ema_dist_arr     = ((df["close"] - ema200) / ema200 * 100).values
    bb_arr           = calc_bollinger_pct_b(df["close"], BB_LENGTH, BB_STD).values
    wil_arr          = calc_williams_r(df["high"], df["low"], df["close"], WILLIAMS_LENGTH).values
    print("  Indicadores calculados ✓")

    # 3. Extremos reales
    print("\nDetectando extremos locales (zig-zag)...")
    mins_bool, maxs_bool = extremos_por_tendencia(df["close"], PCT_TENDENCIA)
    mins_idx = np.where(mins_bool)[0]
    maxs_idx = np.where(maxs_bool)[0]
    print(f"  Mínimos: {len(mins_idx):,}  |  Máximos: {len(maxs_idx):,}")

    # 4. Grid search
    resultados = grid_search(df, ema_dist_arr, bb_arr, wil_arr, mins_idx, maxs_idx)

    if not resultados:
        print("\n⚠️  No se encontró ninguna combinación con el mínimo de señales.")
        print(f"   Reduce MIN_SEÑALES (actual: {MIN_SEÑALES}) o amplía los rangos de búsqueda.")
        return

    # 5. Reporte
    imprimir_top(resultados, TOP_N)
    mejor = resultados[0]
    print(f"\n  ✦  MEJOR COMBINACIÓN:")
    print(f"     DIST_EMA_BUY  = {mejor['dist_ema_buy']}   DIST_EMA_SELL  = {mejor['dist_ema_sell']}")
    print(f"     BB_BUY        = {mejor['bb_buy']}      BB_SELL        = {mejor['bb_sell']}")
    print(f"     WILLIAMS_BUY  = {mejor['williams_buy']}      WILLIAMS_SELL  = {mejor['williams_sell']}")
    print(f"     Score         = {mejor['score']:.4f}")
    print(f"     Precisión compra: {mejor['precision_buy']:.1%}   "
          f"Precisión venta: {mejor['precision_sell']:.1%}")
    print(f"     Cobertura mínimos: {mejor['cobertura_buy']:.1%}   "
          f"Cobertura máximos: {mejor['cobertura_sell']:.1%}")
    print(f"     Señales → {mejor['n_compras']:,} compras / {mejor['n_ventas']:,} ventas")

    # 6. Guardar JSON
    with open("optimizacion_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "config_busqueda": {
                "pct_tendencia"   : PCT_TENDENCIA,
                "tolerancia_velas": TOLERANCIA_VELAS,
                "min_señales"     : MIN_SEÑALES,
                "peso_prec_buy"   : PESO_PRECISION_COMPRA,
                "peso_prec_sell"  : PESO_PRECISION_VENTA,
            },
            "n_extremos": {
                "minimos": int(len(mins_idx)),
                "maximos": int(len(maxs_idx)),
            },
            "resultados": resultados[:200],  # guardar top 200
        }, f, indent=2)
    print("\nResultados guardados: optimizacion_resultados.json  (top 200)")

    # 7. Config optimizado
    guardar_config_optimo(mejor, df)

    # 8. Gráficos
    print("\nGenerando gráficos...")
    generar_graficos(resultados, df, mins_idx, maxs_idx,
                     ema_dist_arr, bb_arr, wil_arr)


if __name__ == "__main__":
    main()
