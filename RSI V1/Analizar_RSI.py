"""
Analizador de Distribución del RSI — BTC/USDT
──────────────────────────────────────────────
Calcula RSI(low) y RSI(high) para cada vela de la DB y construye
estadísticas detalladas para orientar la elección de triggers.

Salida:
  · Reporte en consola
  · rsi_distribution_analysis.json  (datos completos)
  · rsi_distribution_analysis.png   (gráficos)
"""

import sqlite3
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from config import DB_PATH, RSI_LENGTH, FECHA_INICIO, FECHA_FIN

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def calcular_rsi(series: pd.Series, length: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cargar_datos() -> pd.DataFrame:
    print(f"Conectando a {DB_PATH}...")
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, high, low, close
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
    print(f"Velas cargadas  : {len(df):,}")
    print(f"Desde           : {df['datetime'].iloc[0]}")
    print(f"Hasta           : {df['datetime'].iloc[-1]}")
    return df


# ─────────────────────────────────────────────────────────────
# ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────

def estadisticas_zona(serie: pd.Series, label: str, umbral: float, zona: str) -> dict:
    """
    Calcula estadísticas completas para los valores de RSI
    que están en la zona de interés (< umbral para compra, > umbral para venta).
    """
    if zona == "compra":
        muestra = serie[serie <= umbral].dropna()
    else:
        muestra = serie[serie >= umbral].dropna()

    total_velas = len(serie.dropna())

    if len(muestra) == 0:
        return {}

    moda_result = stats.mode(muestra.round(1), keepdims=True)
    percentiles = np.percentile(muestra, [5, 10, 25, 50, 75, 90, 95])

    return {
        "label"              : label,
        "zona"               : zona,
        "umbral_referencia"  : umbral,
        "total_velas_rsi"    : total_velas,
        "velas_en_zona"      : len(muestra),
        "frecuencia_pct"     : round(len(muestra) / total_velas * 100, 4),
        "media"              : round(float(muestra.mean()), 4),
        "mediana"            : round(float(muestra.median()), 4),
        "moda"               : round(float(moda_result.mode[0]), 4),
        "desvio_std"         : round(float(muestra.std()), 4),
        "minimo"             : round(float(muestra.min()), 4),
        "maximo"             : round(float(muestra.max()), 4),
        "p05"                : round(float(percentiles[0]), 4),
        "p10"                : round(float(percentiles[1]), 4),
        "p25"                : round(float(percentiles[2]), 4),
        "p50"                : round(float(percentiles[3]), 4),
        "p75"                : round(float(percentiles[4]), 4),
        "p90"                : round(float(percentiles[5]), 4),
        "p95"                : round(float(percentiles[6]), 4),
    }


def frecuencia_por_nivel(serie: pd.Series, zona: str, paso: float = 1.0) -> dict:
    """
    Cuenta cuántas velas tienen RSI en cada nivel (resolución = paso).
    Devuelve dict {nivel: conteo} ordenado.
    """
    total = len(serie.dropna())
    if zona == "compra":
        muestra = serie[serie <= 50].dropna()
        niveles = np.arange(0, 51, paso)
    else:
        muestra = serie[serie >= 50].dropna()
        niveles = np.arange(50, 101, paso)

    conteos = {}
    for n in niveles:
        n = round(n, 1)
        count = int(((muestra >= n) & (muestra < n + paso)).sum())
        conteos[n] = {
            "conteo" : count,
            "pct_total": round(count / total * 100, 4)
        }
    return conteos


def analisis_por_ciclo(df: pd.DataFrame, rsi_low: pd.Series, rsi_high: pd.Series) -> list:
    """
    Divide el historial en ciclos de mercado conocidos y calcula
    la frecuencia de señales en cada uno.
    """
    ciclos = [
        ("Bull 2017",      "2017-01-01", "2017-12-17"),
        ("Bear 2018",      "2017-12-18", "2018-12-15"),
        ("Recuperación",   "2018-12-16", "2020-03-12"),
        ("COVID + Bull",   "2020-03-13", "2021-04-14"),
        ("Corrección",     "2021-04-15", "2021-07-20"),
        ("Bull 2021 Q4",   "2021-07-21", "2021-11-10"),
        ("Bear 2022",      "2021-11-11", "2022-11-22"),
        ("Recuperación 2", "2022-11-23", "2024-01-10"),
        ("Bull 2024-2025", "2024-01-11", "2026-12-31"),
    ]

    df = df.copy()
    df["rsi_low"]  = rsi_low.values
    df["rsi_high"] = rsi_high.values

    resultados = []
    for nombre, inicio, fin in ciclos:
        mask = (df["datetime"] >= inicio) & (df["datetime"] <= fin)
        sub  = df[mask].dropna(subset=["rsi_low", "rsi_high"])
        if len(sub) == 0:
            continue

        señales_compra = (sub["rsi_low"]  <= 30).sum()
        señales_venta  = (sub["rsi_high"] >= 70).sum()
        total          = len(sub)

        resultados.append({
            "ciclo"             : nombre,
            "periodo"           : f"{inicio} → {fin}",
            "velas"             : total,
            "señales_compra_30" : int(señales_compra),
            "señales_venta_70"  : int(señales_venta),
            "frec_compra_pct"   : round(señales_compra / total * 100, 4),
            "frec_venta_pct"    : round(señales_venta  / total * 100, 4),
            "ratio_cv"          : round(señales_compra / max(señales_venta, 1), 4),
        })

    return resultados


# ─────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generar_graficos(rsi_low: pd.Series, rsi_high: pd.Series,
                     stats_compra: dict, stats_venta: dict,
                     freq_low: dict, freq_high: dict):

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    color_compra = "#2196F3"
    color_venta  = "#F44336"
    color_zona   = "#4CAF50"

    # ── 1. Histograma completo RSI(low) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(rsi_low.dropna(), bins=200, color=color_compra, alpha=0.7, edgecolor="none")
    ax1.axvline(30, color=color_zona,  linewidth=2, linestyle="--", label="Umbral 30")
    ax1.axvline(stats_compra["media"],  color="orange", linewidth=1.5, linestyle="-",  label=f"Media zona ({stats_compra['media']:.1f})")
    ax1.axvline(stats_compra["mediana"],color="yellow", linewidth=1.5, linestyle="-.", label=f"Mediana zona ({stats_compra['mediana']:.1f})")
    ax1.set_title("Distribución RSI(low) — Completa", fontsize=12, fontweight="bold")
    ax1.set_xlabel("RSI(low)")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 2. Histograma completo RSI(high) ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rsi_high.dropna(), bins=200, color=color_venta, alpha=0.7, edgecolor="none")
    ax2.axvline(70, color=color_zona,  linewidth=2, linestyle="--", label="Umbral 70")
    ax2.axvline(stats_venta["media"],  color="orange", linewidth=1.5, linestyle="-",  label=f"Media zona ({stats_venta['media']:.1f})")
    ax2.axvline(stats_venta["mediana"],color="yellow", linewidth=1.5, linestyle="-.", label=f"Mediana zona ({stats_venta['mediana']:.1f})")
    ax2.set_title("Distribución RSI(high) — Completa", fontsize=12, fontweight="bold")
    ax2.set_xlabel("RSI(high)")
    ax2.set_ylabel("Frecuencia")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. Zoom zona de compra (RSI low <= 50) ───────────────
    ax3 = fig.add_subplot(gs[1, 0])
    niveles_c = [k for k in freq_low.keys() if k <= 50]
    conteos_c = [freq_low[k]["pct_total"] for k in niveles_c]
    bars = ax3.bar(niveles_c, conteos_c, width=0.8, color=color_compra, alpha=0.6, edgecolor="none")
    # Resaltar zona <= 30
    for bar, nivel in zip(bars, niveles_c):
        if nivel <= 30:
            bar.set_color(color_zona)
            bar.set_alpha(0.9)
    ax3.axvline(30, color="white", linewidth=1.5, linestyle="--")
    ax3.set_title("RSI(low) por nivel — Zona compra (0–50)\n% de velas totales", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Nivel RSI(low)")
    ax3.set_ylabel("% de velas")
    ax3.grid(True, alpha=0.3, axis="y")

    # Anotar percentiles clave
    for p_label, p_val in [("P10", stats_compra["p10"]), ("P25", stats_compra["p25"])]:
        ax3.axvline(p_val, color="yellow", linewidth=1, linestyle=":")
        ax3.text(p_val + 0.3, ax3.get_ylim()[1] * 0.8, p_label, color="yellow", fontsize=8)

    # ── 4. Zoom zona de venta (RSI high >= 50) ───────────────
    ax4 = fig.add_subplot(gs[1, 1])
    niveles_v = [k for k in freq_high.keys() if k >= 50]
    conteos_v = [freq_high[k]["pct_total"] for k in niveles_v]
    bars = ax4.bar(niveles_v, conteos_v, width=0.8, color=color_venta, alpha=0.6, edgecolor="none")
    for bar, nivel in zip(bars, niveles_v):
        if nivel >= 70:
            bar.set_color(color_zona)
            bar.set_alpha(0.9)
    ax4.axvline(70, color="white", linewidth=1.5, linestyle="--")
    ax4.set_title("RSI(high) por nivel — Zona venta (50–100)\n% de velas totales", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Nivel RSI(high)")
    ax4.set_ylabel("% de velas")
    ax4.grid(True, alpha=0.3, axis="y")

    for p_label, p_val in [("P75", stats_venta["p75"]), ("P90", stats_venta["p90"])]:
        ax4.axvline(p_val, color="yellow", linewidth=1, linestyle=":")
        ax4.text(p_val + 0.3, ax4.get_ylim()[1] * 0.8, p_label, color="yellow", fontsize=8)

    # ── 5. Frecuencia acumulada RSI(low) ─────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    umbrales_c = np.arange(1, 51, 1)
    frec_acum_c = [(rsi_low <= u).sum() / len(rsi_low.dropna()) * 100 for u in umbrales_c]
    ax5.plot(umbrales_c, frec_acum_c, color=color_compra, linewidth=2)
    ax5.fill_between(umbrales_c, frec_acum_c, alpha=0.2, color=color_compra)
    ax5.axvline(30, color=color_zona, linewidth=2, linestyle="--", label="Umbral 30")
    # Marcar el valor de frecuencia en el umbral 30
    frec_30 = (rsi_low <= 30).sum() / len(rsi_low.dropna()) * 100
    ax5.scatter([30], [frec_30], color="white", zorder=5, s=60)
    ax5.annotate(f"{frec_30:.2f}% de velas\ntienen RSI(low)≤30",
                 xy=(30, frec_30), xytext=(33, frec_30 - 1),
                 fontsize=9, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1))
    ax5.set_title("% acumulado de velas con RSI(low) ≤ umbral", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Umbral RSI(low)")
    ax5.set_ylabel("% de velas")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ── 6. Frecuencia acumulada RSI(high) ────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    umbrales_v = np.arange(50, 100, 1)
    frec_acum_v = [(rsi_high >= u).sum() / len(rsi_high.dropna()) * 100 for u in umbrales_v]
    ax6.plot(umbrales_v, frec_acum_v, color=color_venta, linewidth=2)
    ax6.fill_between(umbrales_v, frec_acum_v, alpha=0.2, color=color_venta)
    ax6.axvline(70, color=color_zona, linewidth=2, linestyle="--", label="Umbral 70")
    frec_70 = (rsi_high >= 70).sum() / len(rsi_high.dropna()) * 100
    ax6.scatter([70], [frec_70], color="white", zorder=5, s=60)
    ax6.annotate(f"{frec_70:.2f}% de velas\ntienen RSI(high)≥70",
                 xy=(70, frec_70), xytext=(73, frec_70 + 1),
                 fontsize=9, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1))
    ax6.set_title("% acumulado de velas con RSI(high) ≥ umbral", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Umbral RSI(high)")
    ax6.set_ylabel("% de velas")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle(f"Análisis de Distribución RSI — BTC/USDT  (RSI length={RSI_LENGTH})",
                 fontsize=15, fontweight="bold", color="white", y=1.01)

    nombre = "rsi_distribution_analysis.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nGráfico guardado: {nombre}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────

def imprimir_reporte(stats_c: dict, stats_v: dict, ciclos: list):

    sep = "=" * 62

    print(f"\n{sep}")
    print("  ANÁLISIS DE DISTRIBUCIÓN RSI — ZONA DE COMPRA (low ≤ 30)")
    print(sep)
    print(f"  Velas con RSI(low) ≤ 30 : {stats_c['velas_en_zona']:>12,}  ({stats_c['frecuencia_pct']:.4f}% del total)")
    print(f"  Media                   : {stats_c['media']:>12.4f}")
    print(f"  Mediana                 : {stats_c['mediana']:>12.4f}")
    print(f"  Moda                    : {stats_c['moda']:>12.4f}")
    print(f"  Desvío estándar         : {stats_c['desvio_std']:>12.4f}")
    print(f"  Mínimo                  : {stats_c['minimo']:>12.4f}")
    print(f"  Máximo                  : {stats_c['maximo']:>12.4f}")
    print(f"\n  Percentiles:")
    print(f"    P05={stats_c['p05']:.2f}  P10={stats_c['p10']:.2f}  P25={stats_c['p25']:.2f}  "
          f"P50={stats_c['p50']:.2f}  P75={stats_c['p75']:.2f}  P90={stats_c['p90']:.2f}  P95={stats_c['p95']:.2f}")

    print(f"\n{sep}")
    print("  ANÁLISIS DE DISTRIBUCIÓN RSI — ZONA DE VENTA (high ≥ 70)")
    print(sep)
    print(f"  Velas con RSI(high) ≥ 70 : {stats_v['velas_en_zona']:>11,}  ({stats_v['frecuencia_pct']:.4f}% del total)")
    print(f"  Media                    : {stats_v['media']:>12.4f}")
    print(f"  Mediana                  : {stats_v['mediana']:>12.4f}")
    print(f"  Moda                     : {stats_v['moda']:>12.4f}")
    print(f"  Desvío estándar          : {stats_v['desvio_std']:>12.4f}")
    print(f"  Mínimo                   : {stats_v['minimo']:>12.4f}")
    print(f"  Máximo                   : {stats_v['maximo']:>12.4f}")
    print(f"\n  Percentiles:")
    print(f"    P05={stats_v['p05']:.2f}  P10={stats_v['p10']:.2f}  P25={stats_v['p25']:.2f}  "
          f"P50={stats_v['p50']:.2f}  P75={stats_v['p75']:.2f}  P90={stats_v['p90']:.2f}  P95={stats_v['p95']:.2f}")

    print(f"\n{sep}")
    print("  FRECUENCIA DE SEÑALES POR CICLO DE MERCADO")
    print(sep)
    print(f"  {'Ciclo':<20} {'Velas':>10} {'Compras%':>10} {'Ventas%':>10} {'Ratio C/V':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for c in ciclos:
        print(f"  {c['ciclo']:<20} {c['velas']:>10,} {c['frec_compra_pct']:>10.4f} "
              f"{c['frec_venta_pct']:>10.4f} {c['ratio_cv']:>10.4f}")

    print(f"\n{sep}")
    print("  SUGERENCIAS BASADAS EN LA DISTRIBUCIÓN")
    print(sep)

    # Sugerencia compra: P10 de la zona (10% más extremo)
    sug_compra_conservador = stats_c["p25"]
    sug_compra_agresivo    = stats_c["p10"]

    # Sugerencia venta: P75 de la zona (10% más extremo)
    sug_venta_conservador  = stats_c["p75"] + (70 - stats_c["p75"])  # simetría respecto a 70
    sug_venta_conservador  = stats_v["p25"]
    sug_venta_agresivo     = stats_v["p90"]

    print(f"  Trigger COMPRA conservador (P25 zona): {sug_compra_conservador:.1f}")
    print(f"  Trigger COMPRA agresivo    (P10 zona): {sug_compra_agresivo:.1f}")
    print(f"  Trigger VENTA conservador  (P25 zona): {sug_venta_conservador:.1f}")
    print(f"  Trigger VENTA agresivo     (P90 zona): {sug_venta_agresivo:.1f}")
    print(f"\n  Nota: 'conservador' = más señales, más operaciones")
    print(f"        'agresivo'     = menos señales, más selectivo")
    print(sep)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  ANALIZADOR DE DISTRIBUCIÓN RSI — BTC/USDT")
    print("=" * 62)

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Calcular RSI
    print("\nCalculando RSI(low) y RSI(high)...")
    rsi_low  = calcular_rsi(df["low"],  RSI_LENGTH)
    rsi_high = calcular_rsi(df["high"], RSI_LENGTH)
    print(f"RSI calculado sobre {len(rsi_low.dropna()):,} velas válidas")

    # 3. Estadísticas zona de compra y venta
    print("Calculando estadísticas...")
    stats_c = estadisticas_zona(rsi_low,  "RSI(low)",  30, "compra")
    stats_v = estadisticas_zona(rsi_high, "RSI(high)", 70, "venta")

    # 4. Frecuencia por nivel (para los gráficos de barras)
    freq_low  = frecuencia_por_nivel(rsi_low,  "compra")
    freq_high = frecuencia_por_nivel(rsi_high, "venta")

    # 5. Análisis por ciclo de mercado
    print("Analizando ciclos de mercado...")
    ciclos = analisis_por_ciclo(df, rsi_low, rsi_high)

    # 6. Reporte en consola
    imprimir_reporte(stats_c, stats_v, ciclos)

    # 7. Guardar JSON
    resultado = {
        "config": {
            "rsi_length" : RSI_LENGTH,
            "db_path"    : DB_PATH,
            "fecha_inicio": str(FECHA_INICIO),
            "fecha_fin"  : str(FECHA_FIN),
            "total_velas": len(df),
        },
        "zona_compra" : stats_c,
        "zona_venta"  : stats_v,
        "frecuencia_por_nivel_low" : {str(k): v for k, v in freq_low.items()},
        "frecuencia_por_nivel_high": {str(k): v for k, v in freq_high.items()},
        "ciclos_mercado": ciclos,
    }

    with open("rsi_distribution_analysis.json", "w") as f:
        json.dump(resultado, f, indent=2, default=str)
    print("\nDatos guardados: rsi_distribution_analysis.json")

    # 8. Gráficos
    print("Generando gráficos...")
    generar_graficos(rsi_low, rsi_high, stats_c, stats_v, freq_low, freq_high)


if __name__ == "__main__":
    try:
        from scipy import stats as _
    except ImportError:
        print("Instalando scipy...")
        os.system("pip install scipy")

    main()
