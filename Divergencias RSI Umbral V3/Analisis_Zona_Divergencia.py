"""
Análisis Estadístico — Profundidad Óptima de Entrada en Zona de Divergencia RSI
════════════════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias

PREGUNTA:
  Al detectarse una divergencia alcista válida al cierre de la vela i,
  el precio ya superó el low del ancla hacia abajo.
  ¿Cuánto más puede seguir bajando dentro de la zona válida antes de rebotar?
  ¿A qué precio conviene colocar la orden límite para maximizar la probabilidad
  de ejecución sin subaprovechar la oportunidad?

METODOLOGÍA:
  Para cada señal de divergencia detectada en el histórico:

  COMPRA:
    1. Registra precio_ancla_low, x_umbral, lows[i] (precio real en la vela señal)
    2. Mide la profundidad real: pct = (precio_ancla - lows[i]) / precio_ancla
    3. Mide la profundidad máxima disponible: pct_zona = (precio_ancla - x_umbral) / precio_ancla
    4. Calcula la fracción de zona utilizada: fill_ratio = pct / pct_zona
    5. Registra si la orden al ancla se hubiera llenado (siempre sí)
    6. Registra la ganancia post-señal en N velas siguientes

  Para distintos precios de orden (10%, 25%, 50%, 75%, 90% de la zona):
    · Tasa de fill: % de señales donde lows[i] ≤ precio_orden
    · PnL promedio a distintos horizontes (12h, 24h, 48h, 72h)

  VENTA: misma lógica simétrica.

ARCHIVOS DE SALIDA:
  analisis_zona_divergencia_compra.png
  analisis_zona_divergencia_venta.png
  analisis_zona_divergencia_resultados.json
"""

import sqlite3, json, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
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
        RSI_LENGTH, N,
        RSI_BUY_TRIGGER, RSI_SELL_TRIGGER,
    )
    print("✓ config.py cargado")
except ImportError:
    print("⚠  config.py no encontrado — usando valores por defecto")
    DB_PATH          = r"btc_hourly.db"
    FECHA_INICIO     = "2021-11-10"
    FECHA_FIN        = "2022-11-22"
    RSI_LENGTH       = 5
    N                = 15
    RSI_BUY_TRIGGER  = 10
    RSI_SELL_TRIGGER = 60

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

# Horizontes de análisis post-señal (en velas horarias)
HORIZONTES   = [6, 12, 24, 48, 72, 168]   # 6h, 12h, 1d, 2d, 3d, 7d
OUT_PNG_C    = "analisis_zona_divergencia_compra.png"
OUT_PNG_V    = "analisis_zona_divergencia_venta.png"
OUT_JSON     = "analisis_zona_divergencia_resultados.json"

# Niveles de profundidad a evaluar como % del rango (ancla → x_umbral)
NIVELES_PCT  = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]


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
    print(f"✓ Velas cargadas  : {len(df):,}")
    print(f"  Desde           : {df['datetime'].iloc[0]}")
    print(f"  Hasta           : {df['datetime'].iloc[-1]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# RSI DE WILDER
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi_wilder(series: pd.Series, length: int):
    """Retorna (rsi, avg_gain, avg_loss) — arrays completos."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    alpha    = 1.0 / length
    avg_gain = gain.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi.values, avg_gain.values, avg_loss.values


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE UMBRALES
# ══════════════════════════════════════════════════════════════════════════════

def x_umbral_compra(G: float, L: float, p: float, Ra: float,
                    alpha: float = None) -> float:
    """
    Precio mínimo tal que RSI_low(x) = Ra cuando x < p.
      x = p + k*L - k*G / RS
    Si Ra <= 0: zona ilimitada → retorna 0.
    """
    if alpha is None:
        alpha = 1.0 / RSI_LENGTH
    if Ra <= 0:
        return 0.0
    if Ra >= 100:
        return float('inf')
    k  = (1.0 - alpha) / alpha
    RS = Ra / (100.0 - Ra)
    return p + k * L - k * G / RS


def x_umbral_venta(G: float, L: float, p: float, Ra: float,
                   alpha: float = None) -> float:
    """
    Precio máximo tal que RSI_high(x) = Ra cuando x > p.
      x = p + k*RS*L - k*G
    Si Ra >= 100: zona ilimitada → retorna inf.
    """
    if alpha is None:
        alpha = 1.0 / RSI_LENGTH
    if Ra >= 100:
        return float('inf')
    if Ra <= 0:
        return 0.0
    k  = (1.0 - alpha) / alpha
    RS = Ra / (100.0 - Ra)
    return p + k * RS * L - k * G


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR DE SEÑALES — COMPRA
# ══════════════════════════════════════════════════════════════════════════════

def detectar_señales_compra(df: pd.DataFrame) -> list[dict]:
    """
    Detecta todas las divergencias alcistas válidas con filtro RSI_BUY_TRIGGER.
    Para cada señal registra la geometría de la zona y métricas post-señal.
    """
    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(lows)

    rsi_l, avg_g, avg_l = calcular_rsi_wilder(df["low"], RSI_LENGTH)

    señales = []

    for i in range(N, n):
        # NaN guard
        if np.isnan(rsi_l[i]) or np.isnan(avg_g[i-1]) or np.isnan(avg_l[i-1]):
            continue

        window   = lows[i-N:i]
        if lows[i] >= window.min():
            continue                     # no es nuevo mínimo local

        idx_rel  = int(window.argmin())
        idx_anc  = i - N + idx_rel
        Ra       = float(rsi_l[idx_anc])
        rsi_i    = float(rsi_l[i])

        if rsi_i <= Ra:
            continue                     # RSI no diverge
        if Ra > RSI_BUY_TRIGGER:
            continue                     # ancla fuera de zona de sobreventa

        pa = float(lows[idx_anc])        # precio del ancla (low)
        G  = float(avg_g[i-1])
        L  = float(avg_l[i-1])
        p  = float(lows[i-1])

        xu = x_umbral_compra(G, L, p, Ra)

        # Profundidad real: cuánto cayó respecto al ancla
        low_real   = float(lows[i])
        prof_abs   = pa - low_real                      # USD
        prof_pct   = prof_abs / pa * 100 if pa > 0 else 0

        # Amplitud total de la zona
        if xu > 0 and xu < pa:
            zona_abs = pa - xu
            zona_pct = zona_abs / pa * 100
            fill_ratio = prof_abs / zona_abs if zona_abs > 0 else 0
        else:
            zona_abs   = pa                              # zona ilimitada → usar pa como proxy
            zona_pct   = 100.0
            fill_ratio = 0.0

        # Retornos post-señal a distintos horizontes
        retornos = {}
        for h in HORIZONTES:
            if i + h < n:
                # Mejor precio alcanzado en el horizonte (máximo high)
                best_high = float(np.max(highs[i+1:i+h+1]))
                # Precio de cierre al horizonte
                close_h   = float(closes[i+h])
                # Retorno desde el precio real de ejecución (low[i])
                retornos[f"max_h{h}"]   = (best_high - low_real) / low_real * 100
                retornos[f"close_h{h}"] = (close_h   - low_real) / low_real * 100

        señales.append({
            "idx"          : i,
            "datetime"     : str(df["datetime"].iloc[i]),
            "precio_ancla" : pa,
            "rsi_ancla"    : Ra,
            "rsi_señal"    : rsi_i,
            "low_real"     : low_real,
            "x_umbral"     : xu,
            "prof_abs"     : prof_abs,
            "prof_pct"     : prof_pct,
            "zona_abs"     : zona_abs,
            "zona_pct"     : zona_pct,
            "fill_ratio"   : fill_ratio,
            "zona_ilimitada": xu <= 0,
            **retornos,
        })

    return señales


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR DE SEÑALES — VENTA
# ══════════════════════════════════════════════════════════════════════════════

def detectar_señales_venta(df: pd.DataFrame) -> list[dict]:
    """
    Detecta todas las divergencias bajistas válidas con filtro RSI_SELL_TRIGGER.
    """
    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(highs)

    rsi_h, avg_g, avg_l = calcular_rsi_wilder(df["high"], RSI_LENGTH)

    señales = []

    for i in range(N, n):
        if np.isnan(rsi_h[i]) or np.isnan(avg_g[i-1]) or np.isnan(avg_l[i-1]):
            continue

        window = highs[i-N:i]
        if highs[i] <= window.max():
            continue

        idx_rel  = int(window.argmax())
        idx_anc  = i - N + idx_rel
        Ra       = float(rsi_h[idx_anc])
        rsi_i    = float(rsi_h[i])

        if rsi_i >= Ra:
            continue
        if Ra < RSI_SELL_TRIGGER:
            continue

        pa = float(highs[idx_anc])
        G  = float(avg_g[i-1])
        L  = float(avg_l[i-1])
        p  = float(highs[i-1])

        xu = x_umbral_venta(G, L, p, Ra)

        high_real  = float(highs[i])
        prof_abs   = high_real - pa
        prof_pct   = prof_abs / pa * 100 if pa > 0 else 0

        if not math.isinf(xu) and xu > pa:
            zona_abs   = xu - pa
            zona_pct   = zona_abs / pa * 100
            fill_ratio = prof_abs / zona_abs if zona_abs > 0 else 0
        else:
            zona_abs   = pa * 0.30       # proxy 30% para zona ilimitada
            zona_pct   = 30.0
            fill_ratio = 0.0

        retornos = {}
        for h in HORIZONTES:
            if i + h < n:
                best_low  = float(np.min(lows[i+1:i+h+1]))
                close_h   = float(closes[i+h])
                retornos[f"min_h{h}"]   = (high_real - best_low) / high_real * 100
                retornos[f"close_h{h}"] = (high_real - close_h)  / high_real * 100

        señales.append({
            "idx"           : i,
            "datetime"      : str(df["datetime"].iloc[i]),
            "precio_ancla"  : pa,
            "rsi_ancla"     : Ra,
            "rsi_señal"     : rsi_i,
            "high_real"     : high_real,
            "x_umbral"      : xu,
            "prof_abs"      : prof_abs,
            "prof_pct"      : prof_pct,
            "zona_abs"      : zona_abs,
            "zona_pct"      : zona_pct,
            "fill_ratio"    : fill_ratio,
            "zona_ilimitada": math.isinf(xu),
            **retornos,
        })

    return señales


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE NIVELES DE ORDEN
# ══════════════════════════════════════════════════════════════════════════════

def analizar_niveles_compra(señales: list[dict]) -> pd.DataFrame:
    """
    Para cada nivel de profundidad (% de zona), calcula:
    · tasa_fill     : % de señales donde la vela alcanzó ese precio
    · pnl_*         : PnL promedio a cada horizonte (dado fill)
    · esperado_*    : PnL esperado = tasa_fill × pnl_promedio
    """
    if not señales:
        return pd.DataFrame()
    df = pd.DataFrame(señales)
    rows = []

    for nivel_pct in NIVELES_PCT:
        # Precio de la orden = precio_ancla - nivel_pct * zona_abs
        # fill si lows[i] <= precio_orden
        # precio_orden = precio_ancla - nivel_pct * zona_abs
        # como lows[i] = precio_ancla - prof_abs:
        # fill si prof_pct >= nivel_pct * zona_pct  (en términos relativos)
        # → simplificado: fill si fill_ratio >= nivel_pct
        fills = df["fill_ratio"] >= nivel_pct

        # Para zona ilimitada, nivel_pct de zona es arbitrario → incluir siempre
        fills = fills | df["zona_ilimitada"]

        tasa_fill = fills.mean() * 100

        row = {
            "nivel_pct"  : nivel_pct,
            "tasa_fill"  : round(tasa_fill, 1),
            "n_fills"    : int(fills.sum()),
            "n_total"    : len(df),
        }

        df_fill = df[fills]
        for h in HORIZONTES:
            max_col   = f"max_h{h}"
            close_col = f"close_h{h}"
            if max_col in df_fill.columns:
                pnl_max   = df_fill[max_col].mean()
                pnl_close = df_fill[close_col].mean()
                row[f"pnl_max_h{h}"]      = round(pnl_max,   2)
                row[f"pnl_close_h{h}"]    = round(pnl_close, 2)
                row[f"esp_max_h{h}"]      = round(tasa_fill/100 * pnl_max,   2)
                row[f"esp_close_h{h}"]    = round(tasa_fill/100 * pnl_close, 2)
        rows.append(row)

    return pd.DataFrame(rows)


def analizar_niveles_venta(señales: list[dict]) -> pd.DataFrame:
    if not señales:
        return pd.DataFrame()
    df = pd.DataFrame(señales)
    rows = []

    for nivel_pct in NIVELES_PCT:
        fills = df["fill_ratio"] >= nivel_pct
        fills = fills | df["zona_ilimitada"]
        tasa_fill = fills.mean() * 100
        row = {
            "nivel_pct"  : nivel_pct,
            "tasa_fill"  : round(tasa_fill, 1),
            "n_fills"    : int(fills.sum()),
            "n_total"    : len(df),
        }
        df_fill = df[fills]
        for h in HORIZONTES:
            min_col   = f"min_h{h}"
            close_col = f"close_h{h}"
            if min_col in df_fill.columns:
                pnl_min   = df_fill[min_col].mean()
                pnl_close = df_fill[close_col].mean()
                row[f"pnl_min_h{h}"]      = round(pnl_min,   2)
                row[f"pnl_close_h{h}"]    = round(pnl_close, 2)
                row[f"esp_min_h{h}"]      = round(tasa_fill/100 * pnl_min,   2)
                row[f"esp_close_h{h}"]    = round(tasa_fill/100 * pnl_close, 2)
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN — COMPRA
# ══════════════════════════════════════════════════════════════════════════════

AZUL   = "#2980b9"
VERDE  = "#27ae60"
ROJO   = "#e74c3c"
NARANJ = "#e67e22"
GRIS   = "#95a5a6"
BG     = "#f4f6fa"
DARK   = "#1a2540"


def fig_compra(señales: list[dict], df_niv: pd.DataFrame):
    if not señales:
        print("  ⚠  Sin señales de compra — no se genera figura"); return
    df_s = pd.DataFrame(señales)

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── 1. Histograma profundidad real (prof_pct) ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vals = df_s["prof_pct"].dropna()
    ax1.hist(vals, bins=20, color=AZUL, alpha=0.82, edgecolor="white", linewidth=0.6)
    for q, ls in [(25, "--"), (50, "-"), (75, ":")]:
        qv = float(np.percentile(vals, q))
        ax1.axvline(qv, color=ROJO, linestyle=ls, linewidth=1.4,
                    label=f"P{q}: {qv:.2f}%")
    ax1.set_xlabel("Profundidad real (% bajo ancla)", fontsize=9)
    ax1.set_ylabel("Señales", fontsize=9)
    ax1.set_title("Distribución: caída desde ancla\n(cuánto baja la vela señal bajo el low del ancla)",
                  fontsize=9, fontweight="bold", color=DARK)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_facecolor("#f8fafd")

    # ── 2. Histograma fill_ratio ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fr  = df_s["fill_ratio"].dropna()
    ax2.hist(fr, bins=20, color=VERDE, alpha=0.82, edgecolor="white", linewidth=0.6)
    for q, ls in [(25, "--"), (50, "-"), (75, ":")]:
        qv = float(np.percentile(fr, q))
        ax2.axvline(qv, color=ROJO, linestyle=ls, linewidth=1.4,
                    label=f"P{q}: {qv:.2f}")
    ax2.set_xlabel("Fill ratio (fracción de zona usada)", fontsize=9)
    ax2.set_ylabel("Señales", fontsize=9)
    ax2.set_title("Fracción de zona aprovechada\n(0 = ancla, 1 = x_umbral)",
                  fontsize=9, fontweight="bold", color=DARK)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_facecolor("#f8fafd")

    # ── 3. Tasa de fill vs nivel ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    niv = df_niv["nivel_pct"].values * 100
    tf  = df_niv["tasa_fill"].values
    bars = ax3.bar(range(len(niv)), tf, color=AZUL, alpha=0.82, edgecolor="white")
    ax3.set_xticks(range(len(niv)))
    ax3.set_xticklabels([f"{v:.0f}%" for v in niv], fontsize=8)
    ax3.set_ylim(0, 110)
    for b, v in zip(bars, tf):
        ax3.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}%",
                 ha="center", fontsize=8, fontweight="bold", color=DARK)
    ax3.set_xlabel("Nivel de profundidad en zona (%)", fontsize=9)
    ax3.set_ylabel("Tasa de fill (%)", fontsize=9)
    ax3.set_title("Probabilidad de fill\npor nivel de precio de orden",
                  fontsize=9, fontweight="bold", color=DARK)
    ax3.grid(axis="y", alpha=0.3); ax3.set_facecolor("#f8fafd")

    # ── 4-6. PnL esperado (tasa×pnl) por horizonte para 24h, 48h, 72h ────────
    h_show = [h for h in [24, 48, 72] if f"pnl_close_h{h}" in df_niv.columns]
    colors_h = [VERDE, AZUL, NARANJ]
    for col_idx, (h, color) in enumerate(zip(h_show, colors_h)):
        ax = fig.add_subplot(gs[1, col_idx])
        esp_close = df_niv[f"esp_close_h{h}"].values
        esp_max   = df_niv[f"esp_max_h{h}"].values
        x = np.arange(len(niv))
        ax.bar(x - 0.2, esp_max,   0.38, label=f"Esp. best-high", color=color, alpha=0.8)
        ax.bar(x + 0.2, esp_close, 0.38, label=f"Esp. close",     color=GRIS,  alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.0f}%" for v in niv], fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        opt_idx = int(np.argmax(esp_close))
        ax.axvline(opt_idx, color=ROJO, linestyle="--", linewidth=1.5,
                   label=f"Óptimo: {niv[opt_idx]:.0f}%")
        ax.set_title(f"PnL esperado — horizonte {h}h\n(fill_rate × PnL promedio dado fill)",
                     fontsize=9, fontweight="bold", color=DARK)
        ax.set_xlabel("Nivel de profundidad en zona", fontsize=9)
        ax.set_ylabel("PnL esperado (%)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_facecolor("#f8fafd")

    # ── 7. Tabla resumen ──────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    col_h = 48 if 48 in h_show else h_show[0]
    tabla_cols = ["Nivel %", "Fill %", "Nº fills",
                  f"PnL close {col_h}h (dado fill)", f"PnL max {col_h}h (dado fill)",
                  f"PnL esp. close {col_h}h", f"PnL esp. max {col_h}h",
                  "SCORE (esp.max + esp.close)"]
    tabla_rows = []
    for _, r in df_niv.iterrows():
        pnl_c = r.get(f"pnl_close_h{col_h}", float("nan"))
        pnl_m = r.get(f"pnl_max_h{col_h}",   float("nan"))
        esp_c = r.get(f"esp_close_h{col_h}",  float("nan"))
        esp_m = r.get(f"esp_max_h{col_h}",    float("nan"))
        score = (esp_m + esp_c) if not (math.isnan(esp_m) or math.isnan(esp_c)) else 0
        tabla_rows.append([
            f"{r.nivel_pct*100:.0f}%",
            f"{r.tasa_fill:.0f}%",
            str(int(r.n_fills)),
            f"{pnl_c:+.2f}%",
            f"{pnl_m:+.2f}%",
            f"{esp_c:+.2f}%",
            f"{esp_m:+.2f}%",
            f"{score:+.2f}",
        ])

    tbl = ax7.table(cellText=tabla_rows, colLabels=tabla_cols,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.5)
    for j in range(len(tabla_cols)):
        tbl[0, j].set_facecolor(DARK)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Resaltar fila óptima por score
    scores_num = []
    for r in tabla_rows:
        try: scores_num.append(float(r[-1]))
        except: scores_num.append(-999)
    opt_row = int(np.argmax(scores_num)) + 1
    for j in range(len(tabla_cols)):
        tbl[opt_row, j].set_facecolor("#d5f5e3")
        tbl[opt_row, j].set_text_props(fontweight="bold")
    for i_r in range(1, len(tabla_rows) + 1):
        base = "#f7f9fc" if i_r % 2 == 0 else "#ffffff"
        for j in range(len(tabla_cols)):
            if i_r != opt_row:
                tbl[i_r, j].set_facecolor(base)

    # Estadísticas generales
    p25 = float(np.percentile(df_s["prof_pct"], 25))
    p50 = float(np.percentile(df_s["prof_pct"], 50))
    p75 = float(np.percentile(df_s["prof_pct"], 75))
    stats_txt = (f"Señales totales: {len(df_s)}  |  "
                 f"Profundidad real: P25={p25:.2f}%  P50={p50:.2f}%  P75={p75:.2f}%  |  "
                 f"Fill ratio: P50={float(np.percentile(df_s['fill_ratio'],50)):.2f}  |  "
                 f"★ Fila verde = nivel con mayor PnL esperado (horizonte {col_h}h)")
    fig.text(0.5, 0.02, stats_txt, ha="center", fontsize=8.5,
             color=DARK, bbox=dict(facecolor="white", alpha=0.8, edgecolor="#dde3ef"))

    fig.suptitle(
        f"Análisis de Zona de Divergencia RSI — COMPRA\n"
        f"RSI_BUY_TRIGGER ≤ {RSI_BUY_TRIGGER}  |  RSI_LENGTH={RSI_LENGTH}  |  N={N}  |  "
        f"{FECHA_INICIO} → {FECHA_FIN}\n"
        f"Zona válida: (x_umbral, precio_ancla_low)  —  Profundidad 0% = comprar en ancla, "
        f"100% = comprar en x_umbral",
        fontsize=11, fontweight="bold", color=DARK, y=0.99,
    )
    plt.savefig(OUT_PNG_C, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {OUT_PNG_C}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN — VENTA
# ══════════════════════════════════════════════════════════════════════════════

def fig_venta(señales: list[dict], df_niv: pd.DataFrame):
    if not señales:
        print("  ⚠  Sin señales de venta — no se genera figura"); return
    df_s = pd.DataFrame(señales)

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    vals = df_s["prof_pct"].dropna()
    ax1.hist(vals, bins=20, color=NARANJ, alpha=0.82, edgecolor="white", linewidth=0.6)
    for q, ls in [(25,"--"),(50,"-"),(75,":")]:
        qv = float(np.percentile(vals, q))
        ax1.axvline(qv, color=ROJO, linestyle=ls, linewidth=1.4, label=f"P{q}: {qv:.2f}%")
    ax1.set_xlabel("Profundidad real (% sobre ancla)", fontsize=9)
    ax1.set_ylabel("Señales", fontsize=9)
    ax1.set_title("Distribución: subida desde ancla\n(cuánto sube la vela señal sobre el high del ancla)",
                  fontsize=9, fontweight="bold", color=DARK)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_facecolor("#f8fafd")

    ax2 = fig.add_subplot(gs[0, 1])
    fr  = df_s["fill_ratio"].dropna()
    ax2.hist(fr, bins=20, color=VERDE, alpha=0.82, edgecolor="white", linewidth=0.6)
    for q, ls in [(25,"--"),(50,"-"),(75,":")]:
        qv = float(np.percentile(fr, q))
        ax2.axvline(qv, color=ROJO, linestyle=ls, linewidth=1.4, label=f"P{q}: {qv:.2f}")
    ax2.set_xlabel("Fill ratio (fracción de zona usada)", fontsize=9)
    ax2.set_ylabel("Señales", fontsize=9)
    ax2.set_title("Fracción de zona aprovechada\n(0 = ancla, 1 = x_umbral_venta)",
                  fontsize=9, fontweight="bold", color=DARK)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_facecolor("#f8fafd")

    ax3 = fig.add_subplot(gs[0, 2])
    niv = df_niv["nivel_pct"].values * 100
    tf  = df_niv["tasa_fill"].values
    bars = ax3.bar(range(len(niv)), tf, color=NARANJ, alpha=0.82, edgecolor="white")
    ax3.set_xticks(range(len(niv)))
    ax3.set_xticklabels([f"{v:.0f}%" for v in niv], fontsize=8)
    ax3.set_ylim(0, 110)
    for b, v in zip(bars, tf):
        ax3.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}%",
                 ha="center", fontsize=8, fontweight="bold", color=DARK)
    ax3.set_xlabel("Nivel de profundidad en zona (%)", fontsize=9)
    ax3.set_ylabel("Tasa de fill (%)", fontsize=9)
    ax3.set_title("Probabilidad de fill\npor nivel de precio de orden de venta",
                  fontsize=9, fontweight="bold", color=DARK)
    ax3.grid(axis="y", alpha=0.3); ax3.set_facecolor("#f8fafd")

    h_show = [h for h in [24, 48, 72] if f"pnl_close_h{h}" in df_niv.columns]
    for col_idx, (h, color) in enumerate(zip(h_show, [VERDE, AZUL, NARANJ])):
        ax = fig.add_subplot(gs[1, col_idx])
        esp_close = df_niv[f"esp_close_h{h}"].values
        esp_min   = df_niv[f"esp_min_h{h}"].values
        x = np.arange(len(niv))
        ax.bar(x - 0.2, esp_min,   0.38, label="Esp. best-drop", color=color, alpha=0.8)
        ax.bar(x + 0.2, esp_close, 0.38, label="Esp. close",     color=GRIS,  alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}%" for v in niv], fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        opt_idx = int(np.argmax(esp_close))
        ax.axvline(opt_idx, color=ROJO, linestyle="--", linewidth=1.5,
                   label=f"Óptimo: {niv[opt_idx]:.0f}%")
        ax.set_title(f"PnL esperado — horizonte {h}h\n(drop máximo desde precio de venta)",
                     fontsize=9, fontweight="bold", color=DARK)
        ax.set_xlabel("Nivel de profundidad en zona", fontsize=9)
        ax.set_ylabel("PnL esperado (%)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_facecolor("#f8fafd")

    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    col_h = 48 if 48 in h_show else h_show[0]
    tabla_cols = ["Nivel %", "Fill %", "Nº fills",
                  f"PnL close {col_h}h (dado fill)", f"PnL min {col_h}h (dado fill)",
                  f"PnL esp. close {col_h}h", f"PnL esp. min {col_h}h",
                  "SCORE"]
    tabla_rows = []
    for _, r in df_niv.iterrows():
        pnl_c = r.get(f"pnl_close_h{col_h}", float("nan"))
        pnl_m = r.get(f"pnl_min_h{col_h}",   float("nan"))
        esp_c = r.get(f"esp_close_h{col_h}",  float("nan"))
        esp_m = r.get(f"esp_min_h{col_h}",    float("nan"))
        score = (esp_m + esp_c) if not (math.isnan(esp_m) or math.isnan(esp_c)) else 0
        tabla_rows.append([
            f"{r.nivel_pct*100:.0f}%",
            f"{r.tasa_fill:.0f}%",
            str(int(r.n_fills)),
            f"{pnl_c:+.2f}%",
            f"{pnl_m:+.2f}%",
            f"{esp_c:+.2f}%",
            f"{esp_m:+.2f}%",
            f"{score:+.2f}",
        ])
    tbl = ax7.table(cellText=tabla_rows, colLabels=tabla_cols,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.5)
    for j in range(len(tabla_cols)):
        tbl[0, j].set_facecolor(DARK)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    scores_num = []
    for r in tabla_rows:
        try: scores_num.append(float(r[-1]))
        except: scores_num.append(-999)
    opt_row = int(np.argmax(scores_num)) + 1
    for j in range(len(tabla_cols)):
        tbl[opt_row, j].set_facecolor("#fde8cc")
        tbl[opt_row, j].set_text_props(fontweight="bold")
    for i_r in range(1, len(tabla_rows) + 1):
        base = "#f7f9fc" if i_r % 2 == 0 else "#ffffff"
        for j in range(len(tabla_cols)):
            if i_r != opt_row:
                tbl[i_r, j].set_facecolor(base)

    p25 = float(np.percentile(df_s["prof_pct"], 25))
    p50 = float(np.percentile(df_s["prof_pct"], 50))
    p75 = float(np.percentile(df_s["prof_pct"], 75))
    fig.text(0.5, 0.02,
             f"Señales totales: {len(df_s)}  |  "
             f"Profundidad real: P25={p25:.2f}%  P50={p50:.2f}%  P75={p75:.2f}%  |  "
             f"★ Fila naranja = nivel con mayor PnL esperado (horizonte {col_h}h)",
             ha="center", fontsize=8.5, color=DARK,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="#dde3ef"))

    fig.suptitle(
        f"Análisis de Zona de Divergencia RSI — VENTA\n"
        f"RSI_SELL_TRIGGER ≥ {RSI_SELL_TRIGGER}  |  RSI_LENGTH={RSI_LENGTH}  |  N={N}  |  "
        f"{FECHA_INICIO} → {FECHA_FIN}\n"
        f"Zona válida: (precio_ancla_high, x_umbral_venta)  —  "
        f"Profundidad 0% = vender en ancla, 100% = vender en x_umbral_venta",
        fontsize=11, fontweight="bold", color=DARK, y=0.99,
    )
    plt.savefig(OUT_PNG_V, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {OUT_PNG_V}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ANÁLISIS ZONA DIVERGENCIA RSI — Profundidad Óptima          ║")
    print("║  BTC/USDT · Estadística de fill y PnL post-señal             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    df = cargar_datos()
    if df.empty:
        print("ERROR: sin datos."); return

    print("\nDetectando señales de COMPRA...")
    s_compra = detectar_señales_compra(df)
    print(f"  Señales encontradas: {len(s_compra)}")

    print("\nDetectando señales de VENTA...")
    s_venta  = detectar_señales_venta(df)
    print(f"  Señales encontradas: {len(s_venta)}")

    if not s_compra and not s_venta:
        print("Sin señales. Revisar parámetros en config.py"); return

    print("\nAnalizando niveles de orden...")
    df_niv_c = analizar_niveles_compra(s_compra)
    df_niv_v = analizar_niveles_venta(s_venta)

    # ── Consola compra ────────────────────────────────────────────────────────
    if not df_niv_c.empty:
        print(f"\n{'═'*72}")
        print(f"  COMPRA — Profundidad óptima")
        print(f"  RSI_BUY_TRIGGER ≤ {RSI_BUY_TRIGGER}  |  {len(s_compra)} señales")
        print(f"{'═'*72}")
        df_s_c = pd.DataFrame(s_compra)
        print(f"  Profundidad real bajo ancla:")
        print(f"    P25: {float(np.percentile(df_s_c['prof_pct'],25)):.3f}%")
        print(f"    P50: {float(np.percentile(df_s_c['prof_pct'],50)):.3f}%  ← mediana")
        print(f"    P75: {float(np.percentile(df_s_c['prof_pct'],75)):.3f}%")
        print(f"    Máx: {float(df_s_c['prof_pct'].max()):.3f}%")
        print(f"  Fill ratio (% de zona usada):")
        print(f"    P50: {float(np.percentile(df_s_c['fill_ratio'],50)):.3f}")
        print(f"    P75: {float(np.percentile(df_s_c['fill_ratio'],75)):.3f}")
        print()
        h_ref = 48
        print(f"  {'Nivel':>6}  {'Fill%':>6}  {'N':>4}  "
              f"{'PnL_close_{h}h':>14}  {'PnL_max_{h}h':>12}  "
              f"{'Esp_close':>10}  {'Esp_max':>8}  {'Score':>7}".format(h=h_ref))
        print(f"  {'-'*70}")
        for _, r in df_niv_c.iterrows():
            pc  = r.get(f"pnl_close_h{h_ref}", float("nan"))
            pm  = r.get(f"pnl_max_h{h_ref}",   float("nan"))
            ec  = r.get(f"esp_close_h{h_ref}",  float("nan"))
            em  = r.get(f"esp_max_h{h_ref}",    float("nan"))
            sc  = ec + em if not (math.isnan(ec) or math.isnan(em)) else float("nan")
            print(f"  {r.nivel_pct*100:>5.0f}%  {r.tasa_fill:>5.0f}%  {int(r.n_fills):>4}  "
                  f"{pc:>+13.2f}%  {pm:>+11.2f}%  {ec:>+9.2f}%  {em:>+7.2f}%  {sc:>+6.2f}")

    # ── Consola venta ─────────────────────────────────────────────────────────
    if not df_niv_v.empty:
        print(f"\n{'═'*72}")
        print(f"  VENTA — Profundidad óptima")
        print(f"  RSI_SELL_TRIGGER ≥ {RSI_SELL_TRIGGER}  |  {len(s_venta)} señales")
        print(f"{'═'*72}")
        df_s_v = pd.DataFrame(s_venta)
        print(f"  Profundidad real sobre ancla:")
        print(f"    P25: {float(np.percentile(df_s_v['prof_pct'],25)):.3f}%")
        print(f"    P50: {float(np.percentile(df_s_v['prof_pct'],50)):.3f}%  ← mediana")
        print(f"    P75: {float(np.percentile(df_s_v['prof_pct'],75)):.3f}%")
        print(f"    Máx: {float(df_s_v['prof_pct'].max()):.3f}%")
        print()
        h_ref = 48
        print(f"  {'Nivel':>6}  {'Fill%':>6}  {'N':>4}  "
              f"{'PnL_close_{h}h':>14}  {'PnL_min_{h}h':>12}  "
              f"{'Esp_close':>10}  {'Esp_min':>8}  {'Score':>7}".format(h=h_ref))
        print(f"  {'-'*70}")
        for _, r in df_niv_v.iterrows():
            pc  = r.get(f"pnl_close_h{h_ref}", float("nan"))
            pm  = r.get(f"pnl_min_h{h_ref}",   float("nan"))
            ec  = r.get(f"esp_close_h{h_ref}",  float("nan"))
            em  = r.get(f"esp_min_h{h_ref}",    float("nan"))
            sc  = ec + em if not (math.isnan(ec) or math.isnan(em)) else float("nan")
            print(f"  {r.nivel_pct*100:>5.0f}%  {r.tasa_fill:>5.0f}%  {int(r.n_fills):>4}  "
                  f"{pc:>+13.2f}%  {pm:>+11.2f}%  {ec:>+9.2f}%  {em:>+7.2f}%  {sc:>+6.2f}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    resultado = {
        "config": {
            "rsi_length": RSI_LENGTH, "N": N,
            "rsi_buy_trigger": RSI_BUY_TRIGGER,
            "rsi_sell_trigger": RSI_SELL_TRIGGER,
            "fecha_inicio": FECHA_INICIO, "fecha_fin": FECHA_FIN,
            "horizontes_h": HORIZONTES,
        },
        "compra": {
            "n_señales"     : len(s_compra),
            "niveles"       : df_niv_c.to_dict(orient="records") if not df_niv_c.empty else [],
            "distribucion"  : {
                "prof_pct_p25" : float(np.percentile(pd.DataFrame(s_compra)["prof_pct"],25)) if s_compra else None,
                "prof_pct_p50" : float(np.percentile(pd.DataFrame(s_compra)["prof_pct"],50)) if s_compra else None,
                "prof_pct_p75" : float(np.percentile(pd.DataFrame(s_compra)["prof_pct"],75)) if s_compra else None,
            },
        },
        "venta": {
            "n_señales"     : len(s_venta),
            "niveles"       : df_niv_v.to_dict(orient="records") if not df_niv_v.empty else [],
            "distribucion"  : {
                "prof_pct_p25" : float(np.percentile(pd.DataFrame(s_venta)["prof_pct"],25)) if s_venta else None,
                "prof_pct_p50" : float(np.percentile(pd.DataFrame(s_venta)["prof_pct"],50)) if s_venta else None,
                "prof_pct_p75" : float(np.percentile(pd.DataFrame(s_venta)["prof_pct"],75)) if s_venta else None,
            },
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, default=str)
    print(f"\n  ✓ {OUT_JSON}")

    print("\nGenerando visualizaciones...")
    fig_compra(s_compra, df_niv_c)
    fig_venta (s_venta,  df_niv_v)

    print(f"\n{'═'*62}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*62}")
    for f in [OUT_PNG_C, OUT_PNG_V, OUT_JSON]:
        print(f"  · {f}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
