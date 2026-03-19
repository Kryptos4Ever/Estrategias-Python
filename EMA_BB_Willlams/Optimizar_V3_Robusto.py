"""
Optimizador V3 — Parámetros Robustos entre Regímenes de Mercado
══════════════════════════════════════════════════════════════════════
Filosofía:
  En producción es imposible saber en qué régimen de mercado se está.
  Por eso los parámetros deben ser ROBUSTOS: funcionar razonablemente
  bien en cualquier condición, no excelentes en una y pésimos en otra.

Criterio de optimización — dos métricas complementarias:

  1. SCORE MAXIMIN (prioridad)
     score_robusto = min(score_regimen_1, score_regimen_2, ..., score_regimen_N)
     → Maximiza el peor caso. Una combinación que colapsa en un solo
       régimen queda automáticamente descartada.

  2. SCORE PENALIZADO POR VARIANZA
     score_penalizado = media(scores) - λ * std(scores)
     → Prioriza combinaciones consistentes. Una que da 0.8 en bull y
       0.2 en bear es peor que una que da 0.55 en ambos (con λ=1).

  El ranking final usa score_maximin como criterio principal y
  score_penalizado como desempate.

Score por régimen (igual que V2):
  score = 0.40*prec_buy + 0.40*prec_sell + 0.10*cob_buy + 0.10*cob_sell

Señal flexible (igual que V2):
  Cada indicador aporta un score parcial [0,1] proporcional a su
  intensidad. La señal se activa si la suma ponderada >= umbral.
  Esto aumenta cobertura sin sacrificar precisión.

Salidas:
  · optimizacion_v3_resultados.json
  · optimizacion_v3_robustez.png   (scatter score_min vs score_medio)
  · optimizacion_v3_regimenes.png  (precio + señales por régimen)
  · config_EMA_BB_Williams_V3_ROBUSTO.py

Uso:
  python Optimizar_V3_Robusto.py
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

# ══════════════════════════════════════════════════════════════
#  REGÍMENES DE MERCADO
#  El optimizador busca parámetros que funcionen bien en TODOS.
#  Ajustá las fechas según tu DB.
# ══════════════════════════════════════════════════════════════
REGIMENES = [
    {
        "nombre"      : "Bear_2021-2022",
        "fecha_inicio": "2021-11-10",
        "fecha_fin"   : "2022-11-22",
    },
    {
        "nombre"      : "Recuperacion_2023",
        "fecha_inicio": "2022-11-22",
        "fecha_fin"   : "2023-12-31",
    },
    {
        "nombre"      : "Bull_2024-2025",
        "fecha_inicio": "2024-01-01",
        "fecha_fin"   : "2025-10-06",
    },
]

# ══════════════════════════════════════════════════════════════
#  PARÁMETROS DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════

# ── Detección de extremos zig-zag ─────────────────────────────
PCT_TENDENCIA    = 0.05    # 5% de movimiento para confirmar extremo

# ── Tolerancia señal ↔ extremo ───────────────────────────────
TOLERANCIA_VELAS = 60     # ±60 velas (= ±1 hora en timeframe 1min)

# ── Mínimo de señales válidas por combinación y régimen ──────
MIN_SEÑALES      = 10

# ── Top N resultados a mostrar ────────────────────────────────
TOP_N = 20

# ── Penalización por varianza entre regímenes ────────────────
#    score_penalizado = media - LAMBDA * std
#    LAMBDA=0 → no penaliza varianza (= media pura)
#    LAMBDA=1 → penalización estándar
#    LAMBDA=2 → muy conservador (preferís consistencia sobre performance)
LAMBDA_VARIANZA = 1.0

# ── Pesos del score por régimen ───────────────────────────────
PESO_PREC_BUY  = 0.40
PESO_PREC_SELL = 0.40
PESO_COB_BUY   = 0.10
PESO_COB_SELL  = 0.10

# ── Señal flexible ────────────────────────────────────────────
UMBRAL_SCORE_SEÑAL_RANGE = [0.33, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0]

PESO_EMA = 0.40
PESO_BB  = 0.35
PESO_WIL = 0.25

NORM_EMA = 5.0    # rango de normalización de EMA_dist desde el umbral
NORM_BB  = 0.20   # rango de normalización de BB_%B desde el umbral
NORM_WIL = 20.0   # rango de normalización de Williams desde el umbral

# ── Rangos de umbrales ────────────────────────────────────────
DIST_EMA_BUY_RANGE  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
DIST_EMA_SELL_RANGE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

BB_BUY_RANGE   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
BB_SELL_RANGE  = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

WILLIAMS_BUY_RANGE  = [-95, -90, -85, -80, -75, -70, -60, -50]
WILLIAMS_SELL_RANGE = [-50, -40, -30, -25, -20, -15, -10]

# ══════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ────────────────────────────────────────────────────────────

def cargar_datos(fecha_inicio, fecha_fin) -> pd.DataFrame:
    conn  = sqlite3.connect(DB_PATH)
    df    = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM {DB_TABLE} ORDER BY timestamp ASC",
        conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if fecha_inicio:
        df = df[df["datetime"] >= pd.to_datetime(fecha_inicio)]
    if fecha_fin:
        df = df[df["datetime"] <= pd.to_datetime(fecha_fin)]
    return df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────
# INDICADORES
# ────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────
# DETECCIÓN DE EXTREMOS (zig-zag)
# ────────────────────────────────────────────────────────────

def extremos_por_tendencia(close: pd.Series, pct: float):
    arr     = close.values
    n       = len(arr)
    mins    = np.zeros(n, dtype=bool)
    maxs    = np.zeros(n, dtype=bool)
    estado  = 0
    idx_ext = 0
    val_ext = arr[0]
    for i in range(1, n):
        v = arr[i]
        if estado == 0:
            if v < val_ext:
                val_ext = v; idx_ext = i
            elif v >= val_ext * (1 + pct):
                mins[idx_ext] = True
                estado = 1; val_ext = v; idx_ext = i
        else:
            if v > val_ext:
                val_ext = v; idx_ext = i
            elif v <= val_ext * (1 - pct):
                maxs[idx_ext] = True
                estado = 0; val_ext = v; idx_ext = i
    return mins, maxs


# ────────────────────────────────────────────────────────────
# MÉTRICAS
# ────────────────────────────────────────────────────────────

def _build_extremo_set(extremos_idx, tolerancia, n_max):
    s = set()
    for e in extremos_idx:
        lo = max(0, e - tolerancia)
        hi = min(n_max, e + tolerancia + 1)
        s.update(range(lo, hi))
    return s

def precision_señal(señal_idx, extremo_set):
    if len(señal_idx) == 0:
        return 0.0
    return sum(1 for s in señal_idx if s in extremo_set) / len(señal_idx)

def cobertura_extremos(señal_idx, extremos_idx, tolerancia, n_max):
    if len(extremos_idx) == 0:
        return 0.0
    señal_set = set(señal_idx)
    cubiertos = 0
    for e in extremos_idx:
        lo = max(0, e - tolerancia)
        hi = min(n_max, e + tolerancia + 1)
        if any(v in señal_set for v in range(lo, hi)):
            cubiertos += 1
    return cubiertos / len(extremos_idx)

def score_regimen(buy_g, sell_g, mins_idx, maxs_idx,
                  mins_set, maxs_set, n_total):
    prec_b = precision_señal(buy_g,  mins_set)
    prec_s = precision_señal(sell_g, maxs_set)
    cob_b  = cobertura_extremos(buy_g,  mins_idx, TOLERANCIA_VELAS, n_total)
    cob_s  = cobertura_extremos(sell_g, maxs_idx, TOLERANCIA_VELAS, n_total)
    score  = (PESO_PREC_BUY  * prec_b +
              PESO_PREC_SELL * prec_s +
              PESO_COB_BUY   * cob_b  +
              PESO_COB_SELL  * cob_s)
    return score, prec_b, prec_s, cob_b, cob_s


# ────────────────────────────────────────────────────────────
# SEÑAL FLEXIBLE
# ────────────────────────────────────────────────────────────

def señales_flexibles(ema_v, bb_v, wil_v,
                       deb, des, bbu, bse, wbu, wse, umb):
    # Scores parciales compra
    ema_b = np.clip((-ema_v - deb) / max(NORM_EMA, 1e-9), 0.0, 1.0)
    bb_b  = np.clip((bbu - bb_v)   / max(NORM_BB,  1e-9), 0.0, 1.0)
    wil_b = np.clip((wbu - wil_v)  / max(NORM_WIL, 1e-9), 0.0, 1.0)
    score_buy = PESO_EMA * ema_b + PESO_BB * bb_b + PESO_WIL * wil_b

    # Scores parciales venta
    ema_s  = np.clip((ema_v  - des) / max(NORM_EMA, 1e-9), 0.0, 1.0)
    bb_s   = np.clip((bb_v  - bse)  / max(NORM_BB,  1e-9), 0.0, 1.0)
    wil_s  = np.clip((wil_v - wse)  / max(NORM_WIL, 1e-9), 0.0, 1.0)
    score_sell = PESO_EMA * ema_s + PESO_BB * bb_s + PESO_WIL * wil_s

    buy_local  = np.where(score_buy  >= umb)[0]
    sell_local = np.where(score_sell >= umb)[0]
    return buy_local, sell_local


# ────────────────────────────────────────────────────────────
# PREPARAR DATOS DE CADA RÉGIMEN (una sola vez)
# ────────────────────────────────────────────────────────────

def preparar_regimen(reg: dict) -> dict | None:
    nombre = reg["nombre"]
    fi     = reg["fecha_inicio"]
    ff     = reg["fecha_fin"]

    df = cargar_datos(fi, ff)
    if len(df) < 500:
        print(f"  ⚠️  [{nombre}] Muy pocas velas ({len(df)}), saltando.")
        return None

    ema200       = calc_ema(df["close"], EMA_LENGTH)
    ema_dist_arr = ((df["close"] - ema200) / ema200 * 100).values
    bb_arr       = calc_bollinger_pct_b(df["close"], BB_LENGTH, BB_STD).values
    wil_arr      = calc_williams_r(df["high"], df["low"], df["close"], WILLIAMS_LENGTH).values

    mins_bool, maxs_bool = extremos_por_tendencia(df["close"], PCT_TENDENCIA)
    mins_idx = np.where(mins_bool)[0]
    maxs_idx = np.where(maxs_bool)[0]

    if len(mins_idx) < 3 or len(maxs_idx) < 3:
        print(f"  ⚠️  [{nombre}] Insuficientes extremos, saltando.")
        return None

    valid_mask = (~np.isnan(ema_dist_arr)) & (~np.isnan(bb_arr)) & (~np.isnan(wil_arr))
    valid_idx  = np.where(valid_mask)[0]

    # Pre-construir sets de extremos (para métricas más rápidas)
    mins_set = _build_extremo_set(mins_idx, TOLERANCIA_VELAS, len(df))
    maxs_set = _build_extremo_set(maxs_idx, TOLERANCIA_VELAS, len(df))

    print(f"  [{nombre}]  velas: {len(df):,}  "
          f"mínimos: {len(mins_idx):,}  máximos: {len(maxs_idx):,}")

    return {
        "nombre"      : nombre,
        "fecha_inicio": fi,
        "fecha_fin"   : ff,
        "df"          : df,
        "ema_v"       : ema_dist_arr[valid_idx],
        "bb_v"        : bb_arr[valid_idx],
        "wil_v"       : wil_arr[valid_idx],
        "valid_idx"   : valid_idx,
        "mins_idx"    : mins_idx,
        "maxs_idx"    : maxs_idx,
        "mins_set"    : mins_set,
        "maxs_set"    : maxs_set,
        "n_total"     : len(df),
        "ema_dist_arr": ema_dist_arr,
        "bb_arr"      : bb_arr,
        "wil_arr"     : wil_arr,
    }


# ────────────────────────────────────────────────────────────
# GRID SEARCH ROBUSTO
# ────────────────────────────────────────────────────────────

def grid_search_robusto(regimenes_data: list) -> list:
    n_reg = len(regimenes_data)

    total_combos = (len(DIST_EMA_BUY_RANGE)  * len(DIST_EMA_SELL_RANGE) *
                    len(BB_BUY_RANGE)         * len(BB_SELL_RANGE)       *
                    len(WILLIAMS_BUY_RANGE)   * len(WILLIAMS_SELL_RANGE) *
                    len(UMBRAL_SCORE_SEÑAL_RANGE))

    print(f"\nGrid search robusto: {total_combos:,} combinaciones × {n_reg} regímenes")
    print(f"Criterio: maximin(scores) — desempate: media - {LAMBDA_VARIANZA}·std\n")

    resultados = []
    t0   = time.time()
    done = 0

    for (deb, des, bbu, bse, wbu, wse, umb) in itertools.product(
            DIST_EMA_BUY_RANGE,  DIST_EMA_SELL_RANGE,
            BB_BUY_RANGE,        BB_SELL_RANGE,
            WILLIAMS_BUY_RANGE,  WILLIAMS_SELL_RANGE,
            UMBRAL_SCORE_SEÑAL_RANGE):

        done += 1
        if done % 30000 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / done * (total_combos - done)
            print(f"  {done:>8,}/{total_combos:,}  ({done/total_combos*100:.1f}%)  "
                  f"ETA {eta:.0f}s  válidas: {len(resultados):,}", end="\r")

        scores_por_regimen = []
        detalle_regimenes  = []
        valida = True

        for rd in regimenes_data:
            buy_l, sell_l = señales_flexibles(
                rd["ema_v"], rd["bb_v"], rd["wil_v"],
                deb, des, bbu, bse, wbu, wse, umb)

            buy_g  = rd["valid_idx"][buy_l]
            sell_g = rd["valid_idx"][sell_l]

            if len(buy_g) < MIN_SEÑALES or len(sell_g) < MIN_SEÑALES:
                valida = False
                break

            sc, pb, ps, cb, cs = score_regimen(
                buy_g, sell_g,
                rd["mins_idx"], rd["maxs_idx"],
                rd["mins_set"], rd["maxs_set"],
                rd["n_total"])

            scores_por_regimen.append(sc)
            detalle_regimenes.append({
                "nombre"        : rd["nombre"],
                "score"         : round(sc,  4),
                "precision_buy" : round(pb,  4),
                "precision_sell": round(ps,  4),
                "cobertura_buy" : round(cb,  4),
                "cobertura_sell": round(cs,  4),
                "n_compras"     : int(len(buy_g)),
                "n_ventas"      : int(len(sell_g)),
            })

        if not valida:
            continue

        arr_scores     = np.array(scores_por_regimen)
        score_min      = float(np.min(arr_scores))       # maximin
        score_media    = float(np.mean(arr_scores))
        score_std      = float(np.std(arr_scores))
        score_penaliz  = score_media - LAMBDA_VARIANZA * score_std

        resultados.append({
            "dist_ema_buy"  : deb,
            "dist_ema_sell" : des,
            "bb_buy"        : bbu,
            "bb_sell"       : bse,
            "williams_buy"  : wbu,
            "williams_sell" : wse,
            "umbral_score"  : umb,
            "score_min"     : round(score_min,     4),
            "score_media"   : round(score_media,   4),
            "score_std"     : round(score_std,     4),
            "score_penaliz" : round(score_penaliz, 4),
            "regimenes"     : detalle_regimenes,
        })

    elapsed = time.time() - t0
    # Ordenar: primero maximin, desempate por score penalizado
    resultados.sort(key=lambda x: (x["score_min"], x["score_penaliz"]), reverse=True)
    print(f"\n\nCompletado en {elapsed:.1f}s  |  "
          f"Combinaciones válidas (todos los regímenes): {len(resultados):,}")
    return resultados


# ────────────────────────────────────────────────────────────
# REPORTE EN CONSOLA
# ────────────────────────────────────────────────────────────

def imprimir_top(resultados: list, nombres_regimenes: list):
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"  TOP {TOP_N} COMBINACIONES ROBUSTAS  "
          f"(orden: score_min desc, desempate score_penalizado)")
    print(sep)

    reg_cols = "  ".join(f"{n[:12]:>12}" for n in nombres_regimenes)
    print(f"  {'#':>3}  {'dEB':>5} {'dES':>5} {'BB_B':>5} {'BB_S':>5} "
          f"{'W_B':>5} {'W_S':>4} {'Umb':>5}  "
          f"{'ScMin':>6} {'ScMed':>6} {'ScStd':>6} {'ScPen':>6}  "
          f"{reg_cols}")
    print(f"  {'-'*125}")

    for i, r in enumerate(resultados[:TOP_N], 1):
        reg_scores = "  ".join(
            f"{d['score']:>12.4f}" for d in r["regimenes"])
        print(f"  {i:>3}  {r['dist_ema_buy']:>5.1f} {r['dist_ema_sell']:>5.1f} "
              f"{r['bb_buy']:>5.2f} {r['bb_sell']:>5.2f} "
              f"{r['williams_buy']:>5} {r['williams_sell']:>4} "
              f"{r['umbral_score']:>5.2f}  "
              f"{r['score_min']:>6.4f} {r['score_media']:>6.4f} "
              f"{r['score_std']:>6.4f} {r['score_penaliz']:>6.4f}  "
              f"{reg_scores}")
    print(sep)


# ────────────────────────────────────────────────────────────
# GRÁFICO 1: scatter robustez (score_min vs score_media)
# ────────────────────────────────────────────────────────────

def grafico_robustez(resultados: list):
    if not resultados:
        return

    score_min_arr = np.array([r["score_min"]    for r in resultados])
    score_med_arr = np.array([r["score_media"]  for r in resultados])
    score_std_arr = np.array([r["score_std"]    for r in resultados])
    umb_arr       = np.array([r["umbral_score"] for r in resultados])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")
    fig.suptitle("Robustez de Combinaciones — score_min vs score_media",
                 color="white", fontsize=12, fontweight="bold")

    # ── Scatter score_min vs score_media ─────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#161b22")
    sc = ax0.scatter(score_med_arr, score_min_arr, c=score_std_arr,
                     cmap="RdYlGn_r", s=4, alpha=0.5)
    cb = plt.colorbar(sc, ax=ax0)
    cb.set_label("std entre regímenes", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Marcar el top-1
    r0 = resultados[0]
    ax0.scatter([r0["score_media"]], [r0["score_min"]], marker="*",
                color="#ffeb3b", s=200, zorder=10,
                label=f"Top-1  min={r0['score_min']:.3f}")
    ax0.set_xlabel("Score medio (todos regímenes)", color="white", fontsize=9)
    ax0.set_ylabel("Score mínimo (peor régimen)", color="white", fontsize=9)
    ax0.set_title("Robustez: cada punto = 1 combinación\n"
                  "Arriba-derecha = alta media Y buen peor caso",
                  color="white", fontsize=8)
    ax0.legend(fontsize=8, facecolor="#161b22", labelcolor="white")

    # ── Histograma de score_min ────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#161b22")
    ax1.hist(score_min_arr, bins=50, color="#26a69a", alpha=0.8, edgecolor="#0d1117")
    ax1.axvline(r0["score_min"], color="#ffeb3b", lw=1.5,
                label=f"Top-1: {r0['score_min']:.4f}")
    ax1.set_xlabel("Score mínimo", color="white", fontsize=9)
    ax1.set_ylabel("Cantidad de combinaciones", color="white", fontsize=9)
    ax1.set_title("Distribución de score_min", color="white", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="white")

    # ── Score medio por umbral de señal ───────────────────────
    ax2 = axes[2]
    ax2.set_facecolor("#161b22")
    agrup = {}
    for r in resultados:
        k = r["umbral_score"]
        agrup.setdefault(k, []).append(r["score_min"])
    xk = sorted(agrup.keys())
    medias = [np.mean(agrup[k]) for k in xk]
    ax2.bar([str(k) for k in xk], medias, color="#00d4aa", alpha=0.85)
    ax2.set_xlabel("Umbral de señal flexible", color="white", fontsize=9)
    ax2.set_ylabel("Score_min medio", color="white", fontsize=9)
    ax2.set_title("Robustez por umbral de señal", color="white", fontsize=9)

    for ax in axes:
        ax.spines[:].set_edgecolor("#444")
        ax.tick_params(colors="white", labelsize=8)
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    plt.tight_layout()
    nombre = "optimizacion_v3_robustez.png"
    plt.savefig(nombre, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Gráfico robustez guardado: {nombre}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# GRÁFICO 2: precio + señales del top-1 por régimen
# ────────────────────────────────────────────────────────────

def grafico_señales_por_regimen(mejor: dict, regimenes_data: list):
    n_reg = len(regimenes_data)
    fig, axes = plt.subplots(n_reg, 1, figsize=(18, 5 * n_reg), facecolor="#0d1117")
    if n_reg == 1:
        axes = [axes]

    fig.suptitle(
        f"Señales del mejor parámetro robusto en cada régimen\n"
        f"EMA_B={mejor['dist_ema_buy']}%  EMA_S={mejor['dist_ema_sell']}%  "
        f"BB_B={mejor['bb_buy']}  BB_S={mejor['bb_sell']}  "
        f"W_B={mejor['williams_buy']}  W_S={mejor['williams_sell']}  "
        f"Umbral={mejor['umbral_score']}",
        color="white", fontsize=11, fontweight="bold")

    for ax, rd in zip(axes, regimenes_data):
        df  = rd["df"]
        det = next(d for d in mejor["regimenes"] if d["nombre"] == rd["nombre"])

        ax.set_facecolor("#161b22")
        ax.plot(df.index, df["close"], color="#b0bec5", lw=0.4, alpha=0.6, label="BTC")
        ax.scatter(rd["mins_idx"], df["close"].iloc[rd["mins_idx"]],
                   marker="^", color="#26a69a", s=35, zorder=5,
                   label=f"Mín ({len(rd['mins_idx'])})")
        ax.scatter(rd["maxs_idx"], df["close"].iloc[rd["maxs_idx"]],
                   marker="v", color="#ef5350", s=35, zorder=5,
                   label=f"Máx ({len(rd['maxs_idx'])})")

        # Reconstruir señales
        buy_l, sell_l = señales_flexibles(
            rd["ema_v"], rd["bb_v"], rd["wil_v"],
            mejor["dist_ema_buy"],  mejor["dist_ema_sell"],
            mejor["bb_buy"],         mejor["bb_sell"],
            mejor["williams_buy"],   mejor["williams_sell"],
            mejor["umbral_score"])
        buy_g  = rd["valid_idx"][buy_l]
        sell_g = rd["valid_idx"][sell_l]

        ax.scatter(buy_g,  df["close"].iloc[buy_g],  marker="o",
                   color="#ffeb3b", s=18, zorder=6, alpha=0.75,
                   label=f"Compra ({len(buy_g)})")
        ax.scatter(sell_g, df["close"].iloc[sell_g], marker="o",
                   color="#ff9800", s=18, zorder=6, alpha=0.75,
                   label=f"Venta ({len(sell_g)})")

        ax.set_title(
            f"{rd['nombre']}  |  "
            f"Score={det['score']:.4f}  "
            f"PrecB={det['precision_buy']:.1%}  PrecS={det['precision_sell']:.1%}  "
            f"CobB={det['cobertura_buy']:.1%}  CobS={det['cobertura_sell']:.1%}",
            color="white", fontsize=9)
        ax.legend(fontsize=7, facecolor="#161b22", labelcolor="white",
                  loc="upper left", ncol=4)
        ax.spines[:].set_edgecolor("#444")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    plt.tight_layout()
    nombre = "optimizacion_v3_regimenes.png"
    plt.savefig(nombre, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Gráfico regímenes guardado: {nombre}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# GUARDAR CONFIG
# ────────────────────────────────────────────────────────────

def guardar_config(mejor: dict):
    nombre = "config_EMA_BB_Williams_V3_ROBUSTO.py"
    reg_lines = "\n".join(
        f"#    {d['nombre']:<25} score={d['score']:.4f}  "
        f"PrecB={d['precision_buy']:.1%}  PrecS={d['precision_sell']:.1%}  "
        f"CobB={d['cobertura_buy']:.1%}  CobS={d['cobertura_sell']:.1%}  "
        f"compras={d['n_compras']:,}  ventas={d['n_ventas']:,}"
        for d in mejor["regimenes"])

    contenido = f"""# ============================================================
#  Configuración ROBUSTA V3 — EMA200 + Bollinger + Williams
#  Generada por Optimizar_V3_Robusto.py
#
#  Criterio: maximin entre regímenes + penalización por varianza
#  score_min (peor régimen) : {mejor['score_min']:.4f}
#  score_media (todos)      : {mejor['score_media']:.4f}
#  score_std  (varianza)    : {mejor['score_std']:.4f}
#  score_penalizado         : {mejor['score_penaliz']:.4f}  (media - {LAMBDA_VARIANZA}·std)
#
#  Detalle por régimen:
{reg_lines}
# ============================================================

# ── Base de datos ────────────────────────────────────────────
DB_PATH      = "{DB_PATH}"
RESULTS_JSON = "strategy_results.json"

# ── Capital inicial ──────────────────────────────────────────
SALDO_USDT_INICIAL = 1000.0

# ── Rango de fechas ──────────────────────────────────────────
FECHA_INICIO = '{FECHA_INICIO}'
FECHA_FIN    = '{FECHA_FIN}'

# ── EMA200 ───────────────────────────────────────────────────
EMA_LENGTH     = {EMA_LENGTH}
DIST_EMA_BUY   = {mejor['dist_ema_buy']}    # robusto entre regímenes
DIST_EMA_SELL  = {mejor['dist_ema_sell']}    # robusto entre regímenes

# ── Bollinger Bands ──────────────────────────────────────────
BB_LENGTH      = {BB_LENGTH}
BB_STD         = {BB_STD}
BB_BUY         = {mejor['bb_buy']}    # robusto entre regímenes
BB_SELL        = {mejor['bb_sell']}    # robusto entre regímenes

# ── Williams %R ──────────────────────────────────────────────
WILLIAMS_LENGTH = {WILLIAMS_LENGTH}
WILLIAMS_BUY   = {mejor['williams_buy']}    # robusto entre regímenes
WILLIAMS_SELL  = {mejor['williams_sell']}    # robusto entre regímenes

# ── Señal flexible ───────────────────────────────────────────
#  La estrategia debe usar calcular_scores_señal() en lugar de AND.
UMBRAL_SCORE_SEÑAL = {mejor['umbral_score']}
PESO_EMA_SEÑAL     = {PESO_EMA}
PESO_BB_SEÑAL      = {PESO_BB}
PESO_WIL_SEÑAL     = {PESO_WIL}
NORM_EMA_SEÑAL     = {NORM_EMA}
NORM_BB_SEÑAL      = {NORM_BB}
NORM_WIL_SEÑAL     = {NORM_WIL}

# ── Gestión de capital ───────────────────────────────────────
USDT_PCT_TO_USE       = 5.0
BTC_PCT_TO_SELL       = 5.0
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
    print(f"\n  Config V3 guardado: {nombre}")


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  OPTIMIZADOR V3 — PARÁMETROS ROBUSTOS ENTRE REGÍMENES")
    print("  Criterio: maximizar el peor caso (maximin)")
    print("  Desempate: media - λ·std  (λ =", LAMBDA_VARIANZA, ")")
    print("=" * 72)

    # 1. Preparar datos de cada régimen
    print(f"\nCargando y preparando {len(REGIMENES)} regímenes...")
    regimenes_data = []
    for reg in REGIMENES:
        rd = preparar_regimen(reg)
        if rd:
            regimenes_data.append(rd)

    if len(regimenes_data) < 2:
        print("\n⚠️  Se necesitan al menos 2 regímenes válidos para el análisis robusto.")
        print("   Revisá las fechas en REGIMENES o reducí PCT_TENDENCIA.")
        return

    nombres_reg = [rd["nombre"] for rd in regimenes_data]
    print(f"\nRegímenes válidos: {', '.join(nombres_reg)}")

    # 2. Grid search robusto
    resultados = grid_search_robusto(regimenes_data)

    if not resultados:
        print("\n⚠️  Sin combinaciones válidas en todos los regímenes simultáneamente.")
        print(f"   Reducí MIN_SEÑALES (actual: {MIN_SEÑALES}) o ampliá los rangos.")
        return

    # 3. Reporte
    imprimir_top(resultados, nombres_reg)
    mejor = resultados[0]

    print(f"\n{'═'*72}")
    print(f"  ✦  PARÁMETROS ROBUSTOS ÓPTIMOS")
    print(f"{'═'*72}")
    print(f"  DIST_EMA_BUY  = {mejor['dist_ema_buy']}%     DIST_EMA_SELL  = {mejor['dist_ema_sell']}%")
    print(f"  BB_BUY        = {mejor['bb_buy']}        BB_SELL        = {mejor['bb_sell']}")
    print(f"  WILLIAMS_BUY  = {mejor['williams_buy']}        WILLIAMS_SELL  = {mejor['williams_sell']}")
    print(f"  UMBRAL_SEÑAL  = {mejor['umbral_score']}")
    print(f"\n  score_min     = {mejor['score_min']:.4f}  (peor régimen — lo que maximizamos)")
    print(f"  score_media   = {mejor['score_media']:.4f}  (promedio entre regímenes)")
    print(f"  score_std     = {mejor['score_std']:.4f}  (consistencia — menor = mejor)")
    print(f"  score_penaliz = {mejor['score_penaliz']:.4f}  (media - {LAMBDA_VARIANZA}·std)")
    print()
    for d in mejor["regimenes"]:
        print(f"  [{d['nombre']:<25}]  "
              f"score={d['score']:.4f}  "
              f"PrecB={d['precision_buy']:.1%}  PrecS={d['precision_sell']:.1%}  "
              f"CobB={d['cobertura_buy']:.1%}  CobS={d['cobertura_sell']:.1%}  "
              f"compras={d['n_compras']:,}  ventas={d['n_ventas']:,}")
    print(f"{'═'*72}")

    # 4. Guardar JSON
    with open("optimizacion_v3_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "pct_tendencia"    : PCT_TENDENCIA,
                "tolerancia_velas" : TOLERANCIA_VELAS,
                "min_señales"      : MIN_SEÑALES,
                "lambda_varianza"  : LAMBDA_VARIANZA,
                "peso_prec_buy"    : PESO_PREC_BUY,
                "peso_prec_sell"   : PESO_PREC_SELL,
                "peso_cob_buy"     : PESO_COB_BUY,
                "peso_cob_sell"    : PESO_COB_SELL,
                "peso_ema"         : PESO_EMA,
                "peso_bb"          : PESO_BB,
                "peso_wil"         : PESO_WIL,
                "norm_ema"         : NORM_EMA,
                "norm_bb"          : NORM_BB,
                "norm_wil"         : NORM_WIL,
            },
            "regimenes": nombres_reg,
            "resultados": resultados[:200],
        }, f, indent=2, default=str)
    print("\n  Resultados guardados: optimizacion_v3_resultados.json  (top 200)")

    # 5. Config
    guardar_config(mejor)

    # 6. Gráficos
    print("\n  Generando gráficos...")
    grafico_robustez(resultados)
    grafico_señales_por_regimen(mejor, regimenes_data)
    print("\n  ✓ Listo.")


if __name__ == "__main__":
    main()
