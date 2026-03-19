"""
Optimizador de Variables de Capital — Bear Market
══════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Análisis de Acumulación

Objetivo del Bear Market:
  · Maximizar BTC acumulado en posiciones
  · Mantener portfolio saludable (PnL > 0)
  · USDT nunca llega a cero

Variables optimizadas:
  · FLOOR_PCT      : 5 → 40  (piso del ciclo como % del ATH)
  · FACTOR_CAIDA   : 0.5 → 5.0  (curvatura del gradiente de compra)
  · FACTOR_SUBIDA  : 0.5 → 3.0  (curvatura del gradiente de venta)

Señales RSI fijas (ya optimizadas):
  · RSI_LENGTH = config.RSI_LENGTH
  · N          = config.N

Métricas de salida:
  · portfolio_final     : valor total en USD al cierre
  · pnl_pct             : rentabilidad total %
  · btc_acumulado       : BTC en posiciones al cierre (₿)
  · btc_value           : valor USD del BTC acumulado
  · btc_ratio           : btc_value / portfolio_final  → fracción BTC del portfolio
  · bear_score          : btc_acumulado × (portfolio / inicial) → métrica compuesta
  · min_usdt            : mínimo USDT alcanzado (constraint de supervivencia)
  · sobrevivio          : True si min_usdt > MIN_USDT_THRESHOLD en todo momento

Salidas:
  · optimizacion_capital.csv
  · optimizacion_capital_heatmaps.png
  · optimizacion_capital_pareto.png
  · optimizacion_capital_top20.png
"""

import sqlite3
import math
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from itertools import product
from matplotlib.ticker import FuncFormatter

# ── Importar config ───────────────────────────────────────────────────────────
try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL,
        RSI_LENGTH, N,
        GUARDIA_COMPRA,
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
    RSI_LENGTH            = 7
    N                     = 13
    GUARDIA_COMPRA        = True
    USDT_RESERVA_PCT      = 0
    BTC_PCT_TO_ACCUMULATE = 0
    COMMISSION_PCT        = 0.1

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100

# ══════════════════════════════════════════════════════════════════════════════
# ESPACIO DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuración del espacio de búsqueda ────────────────────────────────────
# Especificá inicio, fin y paso para cada variable.
# El valor de fin está INCLUIDO si cae exactamente en la grilla.

def _rango(inicio, fin, paso):
    """Genera lista de valores [inicio, inicio+paso, ..., fin] con precisión float."""
    valores, v = [], inicio
    while v <= fin + paso * 1e-9:
        valores.append(round(v, 10))
        v += paso
    return valores

FLOOR_PCT_INICIO    = 10  ;  FLOOR_PCT_FIN    = 25  ;  FLOOR_PCT_PASO    = 5
FACTOR_CAIDA_INICIO = 0.5 ;  FACTOR_CAIDA_FIN = 5.0 ;  FACTOR_CAIDA_PASO = 0.5
FACTOR_SUBIDA_INICIO= 0.5 ;  FACTOR_SUBIDA_FIN= 3.0 ;  FACTOR_SUBIDA_PASO= 0.5

FLOOR_PCT_RANGE     = _rango(FLOOR_PCT_INICIO,    FLOOR_PCT_FIN,    FLOOR_PCT_PASO)
FACTOR_CAIDA_RANGE  = _rango(FACTOR_CAIDA_INICIO, FACTOR_CAIDA_FIN, FACTOR_CAIDA_PASO)
FACTOR_SUBIDA_RANGE = _rango(FACTOR_SUBIDA_INICIO,FACTOR_SUBIDA_FIN,FACTOR_SUBIDA_PASO)

# Constraint: USDT nunca debe bajar de este umbral
MIN_USDT_THRESHOLD = 1.0   # $1 — prácticamente cero

# Archivos de salida
OUT_CSV        = "optimizacion_capital.csv"
OUT_HEATMAPS   = "optimizacion_capital_heatmaps.png"
OUT_PARETO     = "optimizacion_capital_pareto.png"
OUT_TOP20      = "optimizacion_capital_top20.png"
OUT_JSON       = "optimizacion_capital.json"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS RSI Y GRADIENTES
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(series: pd.Series, length: int) -> np.ndarray:
    """RSI clásico de Wilder (EWM)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values.astype(float)


def pct_compra(precio_low: float, ath: float, floor_pct: float, factor_caida: float) -> float:
    """% [0-100] USDT disponible — gradiente logarítmico de compra."""
    if ath <= 0 or floor_pct <= 0:
        return 0.0
    log_rango = math.log(100.0 / floor_pct)
    if log_rango <= 0:
        return 0.0
    pos = math.log(ath / precio_low) / log_rango
    pos = max(0.0, min(1.0, pos))
    return (pos ** factor_caida) * 100.0


def pct_venta(precio_high: float, ath: float, precio_promedio: float, factor_subida: float) -> float:
    """% [0-100] BTC en posiciones — gradiente log anclado al PP."""
    if ath <= 0 or precio_promedio <= 0 or precio_high <= precio_promedio:
        return 0.0
    log_amp = math.log(ath / precio_promedio)
    if log_amp <= 0:
        return 0.0
    pos = math.log(precio_high / precio_promedio) / log_amp
    pos = max(0.0, min(1.0, pos))
    return (pos ** factor_subida) * 100.0


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
# NÚCLEO DEL BACKTEST (optimizado para velocidad)
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_backtest(
    lows:         np.ndarray,
    highs:        np.ndarray,
    closes:       np.ndarray,
    rsi_low:      np.ndarray,
    rsi_high:     np.ndarray,
    floor_pct:    float,
    factor_caida: float,
    factor_subida:float,
) -> dict:
    """
    Backtest con gradientes logarítmicos parametrizados.
    Trackea min_usdt para enforcement del constraint de supervivencia.
    """
    n_velas = len(lows)

    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    min_usdt          = float(SALDO_USDT_INICIAL)
    compras           = 0
    ventas            = 0

    ath = float(highs[0])

    for i in range(N, n_velas):

        if highs[i] > ath:
            ath = float(highs[i])

        if np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]):
            continue

        window_lows  = lows[i - N : i]
        window_highs = highs[i - N : i]

        precio_promedio = usdt_invertido / btc_en_posiciones if btc_en_posiciones > 0 else 0.0

        # ── SEÑAL DE COMPRA ───────────────────────────────────────────────────
        señal_compra = False
        if lows[i] < window_lows.min():
            idx_min = i - N + int(window_lows.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                señal_compra = True

        # ── SEÑAL DE VENTA ────────────────────────────────────────────────────
        señal_venta = False
        if not señal_compra and highs[i] > window_highs.max():
            idx_max = i - N + int(window_highs.argmax())
            if rsi_high[i] < rsi_high[idx_max]:
                señal_venta = True

        # ── EJECUTAR COMPRA ───────────────────────────────────────────────────
        if señal_compra:
            usdt_disponible = usdt_balance - USDT_RESERVA
            if usdt_disponible <= 0:
                continue
            if GUARDIA_COMPRA and btc_en_posiciones > 0 and lows[i] >= precio_promedio:
                continue

            pct         = pct_compra(lows[i], ath, floor_pct, factor_caida)
            usdt_a_usar = usdt_disponible * pct / 100.0
            if usdt_a_usar <= 0:
                continue

            comision      = usdt_a_usar * (COMMISSION_PCT / 100)
            btc_adquirido = (usdt_a_usar - comision) / lows[i]

            usdt_balance      -= usdt_a_usar
            btc_en_posiciones += btc_adquirido
            usdt_invertido    += usdt_a_usar
            compras           += 1

            if usdt_balance < min_usdt:
                min_usdt = usdt_balance

        # ── EJECUTAR VENTA ────────────────────────────────────────────────────
        elif señal_venta and btc_en_posiciones > 0:

            pct      = pct_venta(highs[i], ath, precio_promedio, factor_subida)
            btc_slot = btc_en_posiciones * pct / 100.0
            if btc_slot <= 0:
                continue

            btc_a_acumular  = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_a_vender    = btc_slot - btc_a_acumular
            usdt_bruto      = btc_a_vender * highs[i]
            comision        = usdt_bruto * (COMMISSION_PCT / 100)
            usdt_neto       = usdt_bruto - comision

            costo_prop      = usdt_invertido * (btc_slot / btc_en_posiciones)
            usdt_invertido  = max(usdt_invertido - costo_prop, 0.0)
            btc_en_posiciones -= btc_slot
            usdt_balance      += usdt_neto
            ventas            += 1

    # ── Métricas finales ──────────────────────────────────────────────────────
    precio_final    = float(closes[-1])
    btc_value       = btc_en_posiciones * precio_final
    portfolio_final = usdt_balance + btc_value
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100
    btc_ratio       = btc_value / portfolio_final if portfolio_final > 0 else 0.0
    bear_score      = btc_en_posiciones * (portfolio_final / SALDO_USDT_INICIAL)
    sobrevivio      = min_usdt >= MIN_USDT_THRESHOLD

    precio_promedio_final = usdt_invertido / btc_en_posiciones if btc_en_posiciones > 0 else 0.0

    return {
        "floor_pct"           : floor_pct,
        "factor_caida"        : factor_caida,
        "factor_subida"       : factor_subida,
        "portfolio_final"     : round(portfolio_final, 2),
        "pnl_pct"             : round(pnl_pct, 3),
        "btc_acumulado"       : round(btc_en_posiciones, 8),
        "btc_value"           : round(btc_value, 2),
        "btc_ratio"           : round(btc_ratio, 4),
        "bear_score"          : round(bear_score, 6),
        "usdt_final"          : round(usdt_balance, 2),
        "min_usdt"            : round(min_usdt, 2),
        "sobrevivio"          : sobrevivio,
        "total_compras"       : compras,
        "total_ventas"        : ventas,
        "precio_prom_final"   : round(precio_promedio_final, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def optimizar(df: pd.DataFrame) -> pd.DataFrame:

    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)

    # Pre-calcular RSI (único, parámetros fijos)
    print(f"  Calculando RSI (length={RSI_LENGTH}, N={N})...")
    rsi_low  = calcular_rsi(df["low"],  RSI_LENGTH)
    rsi_high = calcular_rsi(df["high"], RSI_LENGTH)

    combos       = list(product(FLOOR_PCT_RANGE, FACTOR_CAIDA_RANGE, FACTOR_SUBIDA_RANGE))
    total_combos = len(combos)

    print(f"\n{'═'*64}")
    print(f"  GRID SEARCH — {total_combos} combinaciones")
    print(f"  FLOOR_PCT     : {FLOOR_PCT_RANGE}")
    print(f"  FACTOR_CAIDA  : {FACTOR_CAIDA_RANGE[0]} → {FACTOR_CAIDA_RANGE[-1]}")
    print(f"  FACTOR_SUBIDA : {FACTOR_SUBIDA_RANGE[0]} → {FACTOR_SUBIDA_RANGE[-1]}")
    print(f"  RSI_LENGTH={RSI_LENGTH}  N={N}  (fijos)")
    print(f"  Constraint    : USDT mínimo ≥ ${MIN_USDT_THRESHOLD}")
    print(f"{'═'*64}\n")

    resultados = []
    t_inicio   = time.time()

    for idx, (fp, fc, fs) in enumerate(combos, 1):
        r = ejecutar_backtest(lows, highs, closes, rsi_low, rsi_high, fp, fc, fs)
        resultados.append(r)

        if idx % 60 == 0 or idx == total_combos:
            elapsed  = time.time() - t_inicio
            eta      = elapsed / idx * (total_combos - idx)
            validos  = sum(1 for r in resultados if r["sobrevivio"])
            best     = max((r["bear_score"] for r in resultados if r["sobrevivio"]), default=0)
            print(f"  [{idx:>3}/{total_combos}]  {elapsed:>5.1f}s  ETA: {eta:>4.1f}s  "
                  f"válidos: {validos}  mejor bear_score: {best:.5f}")

    print(f"\n✓ Grid search completado en {time.time() - t_inicio:.1f}s")

    df_res = pd.DataFrame(resultados)
    df_res = df_res.sort_values("bear_score", ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

STYLE = {
    "fig_bg"   : "#f4f6fa",
    "ax_bg"    : "#ffffff",
    "grid_c"   : "#e2e6ef",
    "title_c"  : "#1a2540",
    "label_c"  : "#2d3a55",
    "valid_c"  : "#2ecc71",
    "invalid_c": "#e74c3c",
    "cmap_port": "RdYlGn",
    "cmap_btc" : "YlOrRd",
    "cmap_bear": "PuBuGn",
    "cmap_ratio": "Blues",
}


def _pivot_metric(df_valid: pd.DataFrame, fp: float, metric: str) -> pd.DataFrame:
    """Pivot FACTOR_CAIDA × FACTOR_SUBIDA para un FLOOR_PCT dado."""
    sub = df_valid[df_valid["floor_pct"] == fp].copy()
    return sub.pivot(index="factor_caida", columns="factor_subida", values=metric)


def fig_heatmaps(df_res: pd.DataFrame):
    """
    4 figuras de heatmaps (una por métrica), cada una con 6 subplots (uno por FLOOR_PCT).
    Solo muestra combinaciones que sobrevivieron (min_usdt ≥ threshold).
    """
    df_v = df_res[df_res["sobrevivio"]].copy()
    df_i = df_res[~df_res["sobrevivio"]].copy()

    metrics = [
        ("portfolio_final", "Portfolio Final (USD)",  STYLE["cmap_port"],  "${:.0f}"),
        ("btc_acumulado",   "BTC Acumulado (₿)",      STYLE["cmap_btc"],   "{:.5f} ₿"),
        ("btc_ratio",       "Ratio BTC/Portfolio",    STYLE["cmap_ratio"], "{:.2%}"),
        ("bear_score",      "Bear Score\n(BTC × PnL-ratio)", STYLE["cmap_bear"], "{:.4f}"),
    ]

    fig_files = []

    for metric, title, cmap, fmt in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor(STYLE["fig_bg"])
        fig.suptitle(
            f"Optimización Capital Bear Market  ·  {FECHA_INICIO} → {FECHA_FIN}\n"
            f"Métrica: {title}  |  RSI_LENGTH={RSI_LENGTH}  N={N}  "
            f"(solo combinaciones que sobrevivieron — USDT ≥ ${MIN_USDT_THRESHOLD})",
            fontsize=12, fontweight="bold", color=STYLE["title_c"], y=1.01
        )

        for ax, fp in zip(axes.flat, FLOOR_PCT_RANGE):
            ax.set_facecolor(STYLE["ax_bg"])

            if len(df_v) > 0:
                piv = _pivot_metric(df_v, fp, metric)
                if not piv.empty:
                    im = ax.imshow(
                        piv.values,
                        aspect="auto", cmap=cmap,
                        origin="lower",
                        interpolation="nearest"
                    )
                    # Anotaciones
                    vmax = piv.values.max()
                    vmin = piv.values.min()
                    span = vmax - vmin if vmax > vmin else 1
                    for ri, fc_val in enumerate(piv.index):
                        for ci, fs_val in enumerate(piv.columns):
                            val = piv.loc[fc_val, fs_val]
                            if np.isnan(val):
                                continue
                            rel = (val - vmin) / span
                            color = "white" if rel > 0.6 else STYLE["label_c"]
                            ax.text(ci, ri, fmt.format(val),
                                    ha="center", va="center", fontsize=6.5,
                                    color=color, fontweight="bold")
                    ax.set_xticks(range(len(piv.columns)))
                    ax.set_yticks(range(len(piv.index)))
                    ax.set_xticklabels([f"{v}" for v in piv.columns], fontsize=7)
                    ax.set_yticklabels([f"{v}" for v in piv.index],   fontsize=7)
                    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

            # Marcar inválidos como X grises
            df_inv_fp = df_i[df_i["floor_pct"] == fp]
            if len(df_inv_fp) > 0 and not piv.empty:
                for _, row in df_inv_fp.iterrows():
                    try:
                        ci = list(piv.columns).index(row["factor_subida"])
                        ri = list(piv.index).index(row["factor_caida"])
                        ax.plot(ci, ri, "x", color="#cc2222", markersize=9, markeredgewidth=2, zorder=5)
                    except (ValueError, UnboundLocalError):
                        pass

            ax.set_title(f"FLOOR_PCT = {fp}%  (ATL_REF = {fp}% ATH)", 
                         fontsize=9, fontweight="bold", color=STYLE["title_c"])
            ax.set_xlabel("FACTOR_SUBIDA →", fontsize=8, color=STYLE["label_c"])
            ax.set_ylabel("← FACTOR_CAIDA", fontsize=8, color=STYLE["label_c"])
            ax.grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f"optimizacion_capital_{metric}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=STYLE["fig_bg"])
        plt.close()
        fig_files.append(fname)
        print(f"  ✓ Heatmap guardado: {fname}")

    return fig_files


def fig_pareto(df_res: pd.DataFrame):
    """
    Scatter Pareto: Portfolio Final vs BTC Acumulado.
    Color = FLOOR_PCT | Tamaño = FACTOR_CAIDA | Forma = FACTOR_SUBIDA
    Resalta la frontera de Pareto (dominancia doble).
    """
    df_v = df_res[df_res["sobrevivio"]].copy()
    df_i = df_res[~df_res["sobrevivio"]].copy()

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(STYLE["fig_bg"])
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_main  = fig.add_subplot(gs[0, :2])   # scatter principal
    ax_ratio = fig.add_subplot(gs[0, 2])    # BTC ratio distribution
    ax_floor = fig.add_subplot(gs[1, 0])    # por FLOOR_PCT
    ax_fc    = fig.add_subplot(gs[1, 1])    # por FACTOR_CAIDA
    ax_fs    = fig.add_subplot(gs[1, 2])    # por FACTOR_SUBIDA

    fig.suptitle(
        f"Análisis Pareto — Acumulación BTC vs Portfolio  ·  {FECHA_INICIO} → {FECHA_FIN}\n"
        f"Bear Market | RSI_LENGTH={RSI_LENGTH}  N={N}  "
        f"| Solo combinaciones sobrevivientes (USDT ≥ ${MIN_USDT_THRESHOLD})",
        fontsize=12, fontweight="bold", color=STYLE["title_c"]
    )

    # ── Scatter principal ──────────────────────────────────────────────────────
    cmap    = plt.cm.get_cmap("plasma", len(FLOOR_PCT_RANGE))
    fp_to_c = {fp: cmap(i) for i, fp in enumerate(FLOOR_PCT_RANGE)}

    # Inválidos en gris
    if len(df_i) > 0:
        ax_main.scatter(df_i["portfolio_final"], df_i["btc_acumulado"],
                        c="#cccccc", alpha=0.25, s=15, marker="x",
                        label=f"No sobrevivió ({len(df_i)})", zorder=1)

    for fp in FLOOR_PCT_RANGE:
        sub = df_v[df_v["floor_pct"] == fp]
        if len(sub) == 0:
            continue
        sizes = sub["factor_caida"] * 12
        ax_main.scatter(sub["portfolio_final"], sub["btc_acumulado"],
                        c=[fp_to_c[fp]] * len(sub),
                        s=sizes, alpha=0.75, edgecolors="white", linewidths=0.4,
                        label=f"FLOOR={fp}%", zorder=3)

    # Frontera de Pareto
    if len(df_v) > 0:
        pts = df_v[["portfolio_final", "btc_acumulado"]].values
        pareto_mask = np.ones(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            if pareto_mask[i]:
                dominated = (pts[:, 0] >= p[0]) & (pts[:, 1] >= p[1])
                dominated[i] = False
                pareto_mask[dominated] = False

        pareto_df = df_v[pareto_mask].sort_values("portfolio_final")
        ax_main.scatter(pareto_df["portfolio_final"], pareto_df["btc_acumulado"],
                        c="gold", s=120, edgecolors="#333", linewidths=1.2,
                        marker="*", zorder=6, label=f"Frontera Pareto ({pareto_mask.sum()})")
        ax_main.plot(pareto_df["portfolio_final"], pareto_df["btc_acumulado"],
                     color="gold", alpha=0.55, linewidth=1.2, zorder=5)

    ax_main.axvline(SALDO_USDT_INICIAL, color="red", linestyle="--", alpha=0.5,
                    linewidth=1.2, label=f"Capital inicial (${SALDO_USDT_INICIAL:,})")
    ax_main.set_xlabel("Portfolio Final (USD)", fontsize=10, color=STYLE["label_c"])
    ax_main.set_ylabel("BTC Acumulado en Posiciones (₿)", fontsize=10, color=STYLE["label_c"])
    ax_main.set_title("Frontera de Pareto: Portfolio vs BTC Acumulado\n"
                      "(tamaño = FACTOR_CAIDA, color = FLOOR_PCT)", fontsize=10)
    ax_main.legend(fontsize=8, ncol=4, loc="upper left")
    ax_main.grid(True, alpha=0.3, color=STYLE["grid_c"])
    ax_main.set_facecolor(STYLE["ax_bg"])
    ax_main.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── BTC Ratio distribution ─────────────────────────────────────────────────
    if len(df_v) > 0:
        ax_ratio.hist(df_v["btc_ratio"] * 100, bins=20,
                      color="#3498db", alpha=0.75, edgecolor="white")
        ax_ratio.axvline(df_v["btc_ratio"].median() * 100, color="red",
                         linestyle="--", linewidth=1.5,
                         label=f"Mediana: {df_v['btc_ratio'].median()*100:.1f}%")
        ax_ratio.set_xlabel("BTC Ratio (%)", fontsize=9, color=STYLE["label_c"])
        ax_ratio.set_ylabel("Frecuencia", fontsize=9)
        ax_ratio.set_title("Distribución BTC/Portfolio", fontsize=9)
        ax_ratio.legend(fontsize=8)
        ax_ratio.grid(True, alpha=0.3, color=STYLE["grid_c"])
        ax_ratio.set_facecolor(STYLE["ax_bg"])

    # ── Por FLOOR_PCT ──────────────────────────────────────────────────────────
    if len(df_v) > 0:
        grp_fp = df_v.groupby("floor_pct").agg(
            btc_med=("btc_acumulado", "median"),
            port_med=("portfolio_final", "median"),
            count=("bear_score", "count"),
            bear_max=("bear_score", "max"),
        ).reset_index()

        colors_fp = [fp_to_c[fp] for fp in grp_fp["floor_pct"]]
        bars = ax_floor.bar(grp_fp["floor_pct"].astype(str) + "%",
                            grp_fp["btc_med"], color=colors_fp, edgecolor="white",
                            linewidth=0.7, alpha=0.9)
        ax_floor.set_xlabel("FLOOR_PCT", fontsize=9, color=STYLE["label_c"])
        ax_floor.set_ylabel("BTC Acumulado Mediano (₿)", fontsize=8)
        ax_floor.set_title("BTC Acumulado por FLOOR_PCT", fontsize=9)
        ax_floor.grid(True, axis="y", alpha=0.3, color=STYLE["grid_c"])
        ax_floor.set_facecolor(STYLE["ax_bg"])
        for bar, n in zip(bars, grp_fp["count"]):
            ax_floor.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f"n={n}", ha="center", va="bottom", fontsize=7)

    # ── Por FACTOR_CAIDA ───────────────────────────────────────────────────────
    if len(df_v) > 0:
        grp_fc = df_v.groupby("factor_caida").agg(
            btc_med=("btc_acumulado", "median"),
            bear_max=("bear_score", "max"),
        ).reset_index()
        ax_fc.plot(grp_fc["factor_caida"], grp_fc["btc_med"],
                   "o-", color="#e74c3c", linewidth=2, markersize=5, alpha=0.85)
        ax_fc.fill_between(grp_fc["factor_caida"], 0, grp_fc["btc_med"],
                           alpha=0.2, color="#e74c3c")
        ax_fc.set_xlabel("FACTOR_CAIDA", fontsize=9, color=STYLE["label_c"])
        ax_fc.set_ylabel("BTC Acumulado Mediano (₿)", fontsize=8)
        ax_fc.set_title("BTC Acumulado por FACTOR_CAIDA", fontsize=9)
        ax_fc.grid(True, alpha=0.3, color=STYLE["grid_c"])
        ax_fc.set_facecolor(STYLE["ax_bg"])

    # ── Por FACTOR_SUBIDA ──────────────────────────────────────────────────────
    if len(df_v) > 0:
        grp_fs = df_v.groupby("factor_subida").agg(
            btc_med=("btc_acumulado", "median"),
            port_med=("portfolio_final", "median"),
        ).reset_index()
        ax2_fs = ax_fs.twinx()
        ax_fs.plot(grp_fs["factor_subida"], grp_fs["btc_med"],
                   "s-", color="#8e44ad", linewidth=2, markersize=5, alpha=0.85,
                   label="BTC Acum. (₿)")
        ax2_fs.plot(grp_fs["factor_subida"], grp_fs["port_med"],
                    "^--", color="#27ae60", linewidth=2, markersize=5, alpha=0.85,
                    label="Portfolio ($)")
        ax_fs.set_xlabel("FACTOR_SUBIDA", fontsize=9, color=STYLE["label_c"])
        ax_fs.set_ylabel("BTC Acumulado Mediano (₿)", fontsize=8, color="#8e44ad")
        ax2_fs.set_ylabel("Portfolio Mediano ($)", fontsize=8, color="#27ae60")
        ax_fs.set_title("BTC vs Portfolio por FACTOR_SUBIDA", fontsize=9)
        ax_fs.grid(True, alpha=0.3, color=STYLE["grid_c"])
        ax_fs.set_facecolor(STYLE["ax_bg"])
        lines1, labels1 = ax_fs.get_legend_handles_labels()
        lines2, labels2 = ax2_fs.get_legend_handles_labels()
        ax_fs.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    plt.savefig(OUT_PARETO, dpi=130, bbox_inches="tight", facecolor=STYLE["fig_bg"])
    plt.close()
    print(f"  ✓ Pareto guardado: {OUT_PARETO}")


def fig_top20(df_res: pd.DataFrame):
    """Tabla visual con las top 20 combinaciones sobrevivientes."""
    df_v = df_res[df_res["sobrevivio"]].copy().head(20)

    if len(df_v) == 0:
        print("  ⚠ No hay combinaciones sobrevivientes para el Top 20")
        return

    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor(STYLE["fig_bg"])
    ax.axis("off")

    cols = ["#", "FLOOR%", "F_CAIDA", "F_SUBIDA",
            "Portfolio $", "PnL %",
            "BTC Acum.", "BTC Value $", "BTC Ratio",
            "Bear Score", "USDT Min $",
            "Compras", "Ventas", "PP Final $"]

    rows = []
    for rank, (_, row) in enumerate(df_v.iterrows(), 1):
        rows.append([
            str(rank),
            f"{row.floor_pct:.0f}%",
            f"{row.factor_caida:.1f}",
            f"{row.factor_subida:.1f}",
            f"${row.portfolio_final:,.2f}",
            f"{row.pnl_pct:+.2f}%",
            f"{row.btc_acumulado:.6f} ₿",
            f"${row.btc_value:,.2f}",
            f"{row.btc_ratio*100:.1f}%",
            f"{row.bear_score:.5f}",
            f"${row.min_usdt:,.2f}",
            str(int(row.total_compras)),
            str(int(row.total_ventas)),
            f"${row.precio_prom_final:,.0f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.55)

    # Estilo cabecera
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#1a2540")
        cell.set_text_props(color="white", fontweight="bold")

    # Colorear filas
    for i in range(1, len(rows) + 1):
        bg = "#eaf6ee" if i % 2 == 0 else "#f8f9fa"
        for j in range(len(cols)):
            table[i, j].set_facecolor(bg)

    # Destacar Top 3
    colors_top = ["#ffd700", "#c0c0c0", "#cd7f32"]
    for i, c in enumerate(colors_top[:min(3, len(rows))], 1):
        for j in range(len(cols)):
            table[i, j].set_facecolor(c)
            table[i, j].set_text_props(fontweight="bold")

    # Destacar columna Bear Score
    col_bear = cols.index("Bear Score")
    max_bs = df_v["bear_score"].max()
    for i in range(1, len(rows) + 1):
        bs_val = df_v.iloc[i - 1]["bear_score"]
        intensity = bs_val / max_bs if max_bs > 0 else 0
        table[i, col_bear].set_facecolor(
            mcolors.to_hex(plt.cm.Greens(0.3 + 0.6 * intensity))
        )

    ax.set_title(
        f"Top 20 Combinaciones — Bear Market {FECHA_INICIO} → {FECHA_FIN}\n"
        f"Ordenadas por Bear Score = BTC_acumulado × (portfolio / capital_inicial)\n"
        f"RSI_LENGTH={RSI_LENGTH}  N={N}  |  Constraint: USDT mínimo ≥ ${MIN_USDT_THRESHOLD}  |  "
        f"Total sobrevivientes: {len(df_res[df_res['sobrevivio']])} / {len(df_res)}",
        fontsize=11, fontweight="bold", color=STYLE["title_c"], pad=14
    )

    plt.tight_layout()
    plt.savefig(OUT_TOP20, dpi=130, bbox_inches="tight", facecolor=STYLE["fig_bg"])
    plt.close()
    print(f"  ✓ Top 20 guardado: {OUT_TOP20}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_csv_json(df_res: pd.DataFrame):
    df_res.to_csv(OUT_CSV, index_label="rank")
    print(f"  ✓ CSV guardado: {OUT_CSV}")

    top20 = df_res[df_res["sobrevivio"]].head(20).to_dict(orient="records")
    config_info = {
        "fecha_inicio"   : FECHA_INICIO,
        "fecha_fin"      : FECHA_FIN,
        "rsi_length"     : RSI_LENGTH,
        "N"              : N,
        "guardia_compra" : GUARDIA_COMPRA,
        "commission_pct" : COMMISSION_PCT,
        "floor_range"    : FLOOR_PCT_RANGE,
        "fc_range"       : FACTOR_CAIDA_RANGE,
        "fs_range"       : FACTOR_SUBIDA_RANGE,
        "total_combos"   : len(df_res),
        "sobrevivientes" : int(df_res["sobrevivio"].sum()),
        "generado"       : pd.Timestamp.now().isoformat(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"config": config_info, "top20": top20}, f, indent=2, default=str)
    print(f"  ✓ JSON guardado: {OUT_JSON}")


def imprimir_resumen_consola(df_res: pd.DataFrame):
    df_v = df_res[df_res["sobrevivio"]]
    df_i = df_res[~df_res["sobrevivio"]]

    print(f"\n{'═'*70}")
    print("  RESUMEN DE OPTIMIZACIÓN")
    print(f"{'═'*70}")
    print(f"  Período         : {FECHA_INICIO}  →  {FECHA_FIN}")
    print(f"  Total combos    : {len(df_res)}")
    print(f"  Sobrevivientes  : {len(df_v)}  ({len(df_v)/len(df_res)*100:.1f}%)")
    print(f"  Eliminados      : {len(df_i)}  (USDT llegó a 0)")

    if len(df_v) == 0:
        print("\n  ⚠ Ninguna combinación sobrevivió el constraint de USDT")
        return

    print(f"\n  Rangos en válidas:")
    print(f"    Portfolio     : ${df_v['portfolio_final'].min():,.2f}  →  ${df_v['portfolio_final'].max():,.2f}")
    print(f"    BTC Acum.     : {df_v['btc_acumulado'].min():.6f} ₿  →  {df_v['btc_acumulado'].max():.6f} ₿")
    print(f"    BTC Ratio     : {df_v['btc_ratio'].min()*100:.1f}%  →  {df_v['btc_ratio'].max()*100:.1f}%")
    print(f"    Bear Score    : {df_v['bear_score'].min():.5f}  →  {df_v['bear_score'].max():.5f}")

    print(f"\n  {'─'*68}")
    print(f"  {'#':>3}  {'FLOOR':>6}  {'F_CAIDA':>8}  {'F_SUBIDA':>9}  "
          f"{'Portfolio':>11}  {'PnL':>7}  {'BTC ₿':>10}  {'Bear Score':>11}  {'USDT_min':>9}")
    print(f"  {'─'*68}")
    for rank, (_, row) in enumerate(df_v.head(15).iterrows(), 1):
        marker = "★" if rank <= 3 else " "
        print(f"  {marker}{rank:>2}.  "
              f"{row.floor_pct:>4.0f}%  "
              f"{row.factor_caida:>8.1f}  "
              f"{row.factor_subida:>9.1f}  "
              f"${row.portfolio_final:>10,.2f}  "
              f"{row.pnl_pct:>+6.2f}%  "
              f"{row.btc_acumulado:>10.6f}  "
              f"{row.bear_score:>11.5f}  "
              f"${row.min_usdt:>8,.2f}")
    print(f"  {'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   OPTIMIZADOR CAPITAL — BEAR MARKET  ·  BTC ACCUMULATION        ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    print("Cargando datos...")
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py")
        return
    print(f"  Velas   : {len(df):,}")
    print(f"  Desde   : {df['datetime'].iloc[0]}")
    print(f"  Hasta   : {df['datetime'].iloc[-1]}")

    print("\nEjecutando grid search...")
    df_res = optimizar(df)

    print("\nGuardando CSV y JSON...")
    guardar_csv_json(df_res)

    imprimir_resumen_consola(df_res)

    print("Generando visualizaciones...")
    heat_files = fig_heatmaps(df_res)
    fig_pareto(df_res)
    fig_top20(df_res)

    print(f"\n{'═'*64}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*64}")
    for f in [OUT_CSV, OUT_JSON, OUT_PARETO, OUT_TOP20] + heat_files:
        print(f"  · {f}")
    print(f"{'═'*64}")
    print("✓ Proceso completado.\n")


if __name__ == "__main__":
    main()