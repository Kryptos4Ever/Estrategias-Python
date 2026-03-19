"""
Analizador de Rachas RSI — BTC/USDT
─────────────────────────────────────
Detecta rachas consecutivas de señales de compra y venta
(ignorando velas sin señal) para orientar la elección de
USDT_PCT_TO_USE y BTC_PCT_TO_SELL.

Una "racha de compra" es una secuencia de N señales de compra
consecutivas antes de que aparezca la primera señal de venta,
y viceversa.

Salida:
  · Reporte en consola con estadísticas y sugerencias
  · rsi_streak_analysis.png  (gráficos)
  · rsi_streak_analysis.json (datos completos)
"""

import sqlite3
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    DB_PATH, RSI_LENGTH,
    LOW_RSI_BUY_TRIGGER, HI_RSI_SELL_TRIGGER,
    FECHA_INICIO, FECHA_FIN,
)

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
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, high, low
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

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# DETECCIÓN DE RACHAS
# ─────────────────────────────────────────────────────────────

def detectar_rachas(señales: pd.Series) -> dict:
    """
    Recibe una serie con valores: 'BUY', 'SELL' o None.
    Filtra solo las velas con señal y detecta rachas consecutivas.

    Devuelve:
      rachas_compra : lista con la longitud de cada racha de compra
      rachas_venta  : lista con la longitud de cada racha de venta
      secuencia     : lista de tuplas (tipo, longitud) en orden cronológico
    """
    # Filtrar solo velas con señal
    solo_señales = señales[señales.notna()].values

    if len(solo_señales) == 0:
        return {"rachas_compra": [], "rachas_venta": [], "secuencia": []}

    rachas_compra = []
    rachas_venta  = []
    secuencia     = []

    tipo_actual  = solo_señales[0]
    count        = 1

    for señal in solo_señales[1:]:
        if señal == tipo_actual:
            count += 1
        else:
            # Cerrar racha anterior
            if tipo_actual == "BUY":
                rachas_compra.append(count)
            else:
                rachas_venta.append(count)
            secuencia.append((tipo_actual, count))

            # Iniciar nueva racha
            tipo_actual = señal
            count       = 1

    # Cerrar última racha
    if tipo_actual == "BUY":
        rachas_compra.append(count)
    else:
        rachas_venta.append(count)
    secuencia.append((tipo_actual, count))

    return {
        "rachas_compra": rachas_compra,
        "rachas_venta" : rachas_venta,
        "secuencia"    : secuencia,
    }


def estadisticas_rachas(rachas: list, tipo: str) -> dict:
    if not rachas:
        return {}

    arr = np.array(rachas)
    conteo_por_longitud = {}
    for v in sorted(set(arr)):
        conteo_por_longitud[int(v)] = int((arr == v).sum())

    # Percentiles clave
    percentiles = np.percentile(arr, [50, 75, 90, 95, 99])

    return {
        "tipo"                : tipo,
        "total_rachas"        : len(rachas),
        "total_señales"       : int(arr.sum()),
        "media"               : round(float(arr.mean()), 4),
        "mediana"             : round(float(np.median(arr)), 4),
        "moda"                : int(np.bincount(arr).argmax()),
        "maximo"              : int(arr.max()),
        "minimo"              : int(arr.min()),
        "desvio_std"          : round(float(arr.std()), 4),
        "p50"                 : round(float(percentiles[0]), 1),
        "p75"                 : round(float(percentiles[1]), 1),
        "p90"                 : round(float(percentiles[2]), 1),
        "p95"                 : round(float(percentiles[3]), 1),
        "p99"                 : round(float(percentiles[4]), 1),
        "conteo_por_longitud" : conteo_por_longitud,
    }


def calcular_pct_sugerido(percentil: float) -> float:
    """
    Dado que en una racha de N señales se usan N veces el X% del saldo,
    el saldo residual después de N compras es: saldo * (1 - pct)^N

    Para que quede al menos un 10% del saldo original después de la racha P90:
      (1 - pct)^N = 0.10  →  pct = 1 - 0.10^(1/N)
    """
    if percentil <= 0:
        return 0.0
    pct = (1 - 0.10 ** (1 / percentil)) * 100
    return round(pct, 2)


# ─────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generar_graficos(stats_c: dict, stats_v: dict, secuencia: list):

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    color_c = "#2196F3"
    color_v = "#F44336"
    bg_dark = "#16213e"

    rachas_c = list(stats_c["conteo_por_longitud"].keys())
    counts_c = list(stats_c["conteo_por_longitud"].values())
    rachas_v = list(stats_v["conteo_por_longitud"].keys())
    counts_v = list(stats_v["conteo_por_longitud"].values())

    # ── 1. Histograma rachas de compra ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(rachas_c, counts_c, color=color_c, alpha=0.7, edgecolor="none", width=0.8)
    for p_label, p_val in [("P75", stats_c["p75"]), ("P90", stats_c["p90"]), ("P99", stats_c["p99"])]:
        ax1.axvline(p_val, color="yellow", linewidth=1.5, linestyle="--")
        ax1.text(p_val + 0.2, ax1.get_ylim()[1] * 0.85, p_label,
                 color="yellow", fontsize=8, rotation=90)
    ax1.set_title("Distribución de rachas de COMPRA\n(señales BUY consecutivas)", fontweight="bold")
    ax1.set_xlabel("Longitud de racha (N compras seguidas)")
    ax1.set_ylabel("Frecuencia")
    ax1.set_xlim(0, min(max(rachas_c) + 1, stats_c["p99"] * 1.5))
    ax1.grid(True, alpha=0.3, axis="y")

    # ── 2. Histograma rachas de venta ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(rachas_v, counts_v, color=color_v, alpha=0.7, edgecolor="none", width=0.8)
    for p_label, p_val in [("P75", stats_v["p75"]), ("P90", stats_v["p90"]), ("P99", stats_v["p99"])]:
        ax2.axvline(p_val, color="yellow", linewidth=1.5, linestyle="--")
        ax2.text(p_val + 0.2, ax2.get_ylim()[1] * 0.85, p_label,
                 color="yellow", fontsize=8, rotation=90)
    ax2.set_title("Distribución de rachas de VENTA\n(señales SELL consecutivas)", fontweight="bold")
    ax2.set_xlabel("Longitud de racha (N ventas seguidas)")
    ax2.set_ylabel("Frecuencia")
    ax2.set_xlim(0, min(max(rachas_v) + 1, stats_v["p99"] * 1.5))
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 3. Frecuencia acumulada compra ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    max_n_c = int(stats_c["p99"] * 1.2)
    arr_c   = np.array(list(stats_c["conteo_por_longitud"].keys()))
    cnt_c   = np.array(list(stats_c["conteo_por_longitud"].values()))
    total_c = cnt_c.sum()
    ns_c    = np.arange(1, max_n_c + 1)
    acum_c  = [((arr_c <= n) * cnt_c).sum() / total_c * 100 for n in ns_c]
    ax3.plot(ns_c, acum_c, color=color_c, linewidth=2)
    ax3.fill_between(ns_c, acum_c, alpha=0.2, color=color_c)
    for p_label, p_val in [("P75", stats_c["p75"]), ("P90", stats_c["p90"]), ("P95", stats_c["p95"])]:
        y_val = next((a for n, a in zip(ns_c, acum_c) if n >= p_val), acum_c[-1])
        ax3.axvline(p_val, color="yellow", linewidth=1.2, linestyle="--")
        ax3.axhline(y_val, color="yellow", linewidth=0.8, linestyle=":", alpha=0.5)
        ax3.text(p_val + 0.2, 5, f"{p_label}={p_val:.0f}", color="yellow", fontsize=8, rotation=90)
    ax3.set_title("% acumulado de rachas de COMPRA ≤ N", fontweight="bold")
    ax3.set_xlabel("Longitud de racha N")
    ax3.set_ylabel("% de rachas")
    ax3.grid(True, alpha=0.3)

    # ── 4. Frecuencia acumulada venta ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    max_n_v = int(stats_v["p99"] * 1.2)
    arr_v   = np.array(list(stats_v["conteo_por_longitud"].keys()))
    cnt_v   = np.array(list(stats_v["conteo_por_longitud"].values()))
    total_v = cnt_v.sum()
    ns_v    = np.arange(1, max_n_v + 1)
    acum_v  = [((arr_v <= n) * cnt_v).sum() / total_v * 100 for n in ns_v]
    ax4.plot(ns_v, acum_v, color=color_v, linewidth=2)
    ax4.fill_between(ns_v, acum_v, alpha=0.2, color=color_v)
    for p_label, p_val in [("P75", stats_v["p75"]), ("P90", stats_v["p90"]), ("P95", stats_v["p95"])]:
        y_val = next((a for n, a in zip(ns_v, acum_v) if n >= p_val), acum_v[-1])
        ax4.axvline(p_val, color="yellow", linewidth=1.2, linestyle="--")
        ax4.axhline(y_val, color="yellow", linewidth=0.8, linestyle=":", alpha=0.5)
        ax4.text(p_val + 0.2, 5, f"{p_label}={p_val:.0f}", color="yellow", fontsize=8, rotation=90)
    ax4.set_title("% acumulado de rachas de VENTA ≤ N", fontweight="bold")
    ax4.set_xlabel("Longitud de racha N")
    ax4.set_ylabel("% de rachas")
    ax4.grid(True, alpha=0.3)

    # ── 5. Curva de PCT sugerido según cobertura deseada ─────
    ax5 = fig.add_subplot(gs[2, 0])
    coberturas = np.arange(1, int(stats_c["p99"]) + 1)
    pcts       = [calcular_pct_sugerido(n) for n in coberturas]
    ax5.plot(coberturas, pcts, color=color_c, linewidth=2)
    ax5.fill_between(coberturas, pcts, alpha=0.2, color=color_c)
    for p_label, p_val in [("P75", stats_c["p75"]), ("P90", stats_c["p90"]), ("P95", stats_c["p95"])]:
        idx  = int(p_val) - 1
        if 0 <= idx < len(pcts):
            ax5.scatter([p_val], [pcts[idx]], color="yellow", zorder=5, s=60)
            ax5.annotate(f"{p_label}: {pcts[idx]:.1f}%",
                         xy=(p_val, pcts[idx]),
                         xytext=(p_val + 1, pcts[idx] + 0.5),
                         color="yellow", fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=1))
    ax5.set_title("USDT_PCT_TO_USE sugerido\npara cubrir racha de N compras (dejando 10% reserva)",
                  fontweight="bold")
    ax5.set_xlabel("Longitud de racha cubierta (N compras)")
    ax5.set_ylabel("% USDT por compra")
    ax5.grid(True, alpha=0.3)

    # ── 6. Alternancia de rachas en el tiempo (primeras 500) ─
    ax6 = fig.add_subplot(gs[2, 1])
    muestra   = secuencia[:500]
    colores   = [color_c if t == "BUY" else color_v for t, _ in muestra]
    longitudes = [l for _, l in muestra]
    ax6.bar(range(len(muestra)), longitudes, color=colores, alpha=0.7, edgecolor="none", width=1.0)
    ax6.set_title("Alternancia de rachas (primeras 500)\nAzul=compra  Rojo=venta", fontweight="bold")
    ax6.set_xlabel("Índice de racha")
    ax6.set_ylabel("Longitud")
    ax6.grid(True, alpha=0.3, axis="y")

    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor(bg_dark)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle(
        f"Análisis de Rachas RSI — BTC/USDT  "
        f"(RSI length={RSI_LENGTH}  BUY≤{LOW_RSI_BUY_TRIGGER}  SELL≥{HI_RSI_SELL_TRIGGER})",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )

    nombre = "rsi_streak_analysis.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Gráfico guardado: {nombre}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────

def imprimir_reporte(stats_c: dict, stats_v: dict):
    sep = "=" * 62

    for stats, emoji in [(stats_c, "COMPRA  📈"), (stats_v, "VENTA   📉")]:
        print(f"\n{sep}")
        print(f"  RACHAS DE {emoji}")
        print(sep)
        print(f"  Total de rachas        : {stats['total_rachas']:>10,}")
        print(f"  Total de señales       : {stats['total_señales']:>10,}")
        print(f"  Longitud media         : {stats['media']:>10.2f}")
        print(f"  Longitud mediana       : {stats['mediana']:>10.1f}")
        print(f"  Moda (más frecuente)   : {stats['moda']:>10}")
        print(f"  Longitud máxima        : {stats['maximo']:>10}")
        print(f"  Desvío estándar        : {stats['desvio_std']:>10.2f}")
        print(f"\n  Percentiles:")
        print(f"    P50={stats['p50']:.0f}  P75={stats['p75']:.0f}  "
              f"P90={stats['p90']:.0f}  P95={stats['p95']:.0f}  P99={stats['p99']:.0f}")

    # Sugerencias de porcentajes
    print(f"\n{sep}")
    print("  SUGERENCIAS DE PORCENTAJES")
    print(sep)

    for label, p_val, tipo in [
        ("P75  compra", stats_c["p75"], "USDT"),
        ("P90  compra", stats_c["p90"], "USDT"),
        ("P95  compra", stats_c["p95"], "USDT"),
        ("P75  venta",  stats_v["p75"], "BTC"),
        ("P90  venta",  stats_v["p90"], "BTC"),
        ("P95  venta",  stats_v["p95"], "BTC"),
    ]:
        pct = calcular_pct_sugerido(p_val)
        print(f"  Cubrir racha {label} (N={p_val:.0f})  →  "
              f"usar {pct:>5.2f}% de {tipo} por operación")

    print(f"\n  Fórmula: pct = (1 - 0.10^(1/N)) × 100")
    print(f"  Garantiza que tras N operaciones consecutivas")
    print(f"  queda al menos el 10% del saldo original.")
    print(sep)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  ANALIZADOR DE RACHAS RSI — BTC/USDT")
    print(f"  RSI({RSI_LENGTH})  BUY≤{LOW_RSI_BUY_TRIGGER}  SELL≥{HI_RSI_SELL_TRIGGER}")
    print("=" * 62)

    # 1. Cargar y calcular RSI
    print("\nCargando datos...")
    df = cargar_datos()
    print(f"Velas: {len(df):,}  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")

    print("Calculando RSI...")
    rsi_low  = calcular_rsi(df["low"],  RSI_LENGTH)
    rsi_high = calcular_rsi(df["high"], RSI_LENGTH)

    # 2. Construir serie de señales (solo BUY / SELL / None)
    señales = pd.Series(index=df.index, dtype=object)
    señales[rsi_low  <= LOW_RSI_BUY_TRIGGER]  = "BUY"
    señales[rsi_high >= HI_RSI_SELL_TRIGGER]  = "SELL"
    # Si una vela dispara ambos (raro pero posible), BUY tiene prioridad
    ambos = (rsi_low <= LOW_RSI_BUY_TRIGGER) & (rsi_high >= HI_RSI_SELL_TRIGGER)
    señales[ambos] = "BUY"

    total_señales = señales.notna().sum()
    print(f"Señales totales: {total_señales:,}  "
          f"({(señales=='BUY').sum():,} BUY  /  {(señales=='SELL').sum():,} SELL)")

    # 3. Detectar rachas
    print("Detectando rachas...")
    resultado = detectar_rachas(señales)

    # 4. Estadísticas
    stats_c = estadisticas_rachas(resultado["rachas_compra"], "BUY")
    stats_v = estadisticas_rachas(resultado["rachas_venta"],  "SELL")

    # 5. Reporte
    imprimir_reporte(stats_c, stats_v)

    # 6. Guardar JSON
    with open("rsi_streak_analysis.json", "w") as f:
        json.dump({
            "config": {
                "rsi_length"          : RSI_LENGTH,
                "low_rsi_buy_trigger" : LOW_RSI_BUY_TRIGGER,
                "hi_rsi_sell_trigger" : HI_RSI_SELL_TRIGGER,
            },
            "stats_compra": stats_c,
            "stats_venta" : stats_v,
        }, f, indent=2)
    print("\nDatos guardados: rsi_streak_analysis.json")

    # 7. Gráficos
    print("Generando gráficos...")
    generar_graficos(stats_c, stats_v, resultado["secuencia"])


if __name__ == "__main__":
    main()