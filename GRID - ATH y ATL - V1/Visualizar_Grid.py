"""
Visualizador de Grids ATH/ATL — BTC/USDT
==========================================
Lee strategy_results.json y btc_minutes.db y genera un grafico
a pantalla completa con:
  · Precio BTC como linea de fondo (escala logaritmica)
  · Lineas horizontales rojas  = niveles de compra activos (ultima tabla)
  · Lineas horizontales verdes = niveles de venta activos  (ultima tabla)
  · Puntos azules  = compras ejecutadas
  · Puntos naranja = ventas ejecutadas
  · Lineas de referencia ATH y ATL historicos
  · Intensidad de color proporcional al monto de la orden
"""

import sqlite3
import json
import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from config import (
    DB_PATH, RESULTS_JSON,
    PASO_PCT_COMPRA, CAIDA_MAXIMA,
    PASO_PCT_VENTA,  SUBIDA_MAXIMA,
    MIN_COMPRA_PCT,  MAX_COMPRA_PCT,
    MIN_VENTA_PCT,   MAX_VENTA_PCT,
    MIN_ACUMULAR_PCT, MAX_ACUMULAR_PCT,
)

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def cargar_precio(muestra: int = 10) -> pd.DataFrame:
    """Carga el precio de cierre desde la DB, submuestreado cada N velas."""
    print(f"Cargando precios (cada {muestra} velas)...")
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, close
        FROM   {DB_TABLE}
        ORDER  BY timestamp ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.iloc[::muestra].reset_index(drop=True)


def cargar_resultados() -> dict:
    print(f"Cargando {RESULTS_JSON}...")
    with open(RESULTS_JSON) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# RECONSTRUCCION DE TABLAS FINALES
# ─────────────────────────────────────────────────────────────

def reconstruir_tabla_compras(ath: float, usdt_ref: float) -> list:
    """Reconstruye los niveles de compra usando el ATH final."""
    factor  = 1.0 - PASO_PCT_COMPRA / 100.0
    piso    = 1.0 - CAIDA_MAXIMA    / 100.0
    n_pasos = int(math.log(piso) / math.log(factor))
    niveles = []
    for i in range(1, n_pasos + 1):
        precio         = ath * (factor ** i)
        t_lineal       = (i - 1) / (n_pasos - 1) if n_pasos > 1 else 0.0
        t              = t_lineal ** 0.5
        pct_nivel      = MIN_COMPRA_PCT + t * (MAX_COMPRA_PCT - MIN_COMPRA_PCT)
        monto_objetivo = usdt_ref * (pct_nivel / 100.0)
        niveles.append({
            "precio"         : precio,
            "pct_nivel"      : pct_nivel,
            "monto_objetivo" : monto_objetivo,
            "distancia_pct"  : (1.0 - factor**i) * 100,
        })
    return niveles


def reconstruir_tabla_ventas(atl: float, btc_ref: float) -> list:
    """Reconstruye los niveles de venta usando el ATL final."""
    factor  = 1.0 + PASO_PCT_VENTA / 100.0
    techo   = 1.0 + SUBIDA_MAXIMA  / 100.0
    n_pasos = int(math.log(techo) / math.log(factor))
    niveles = []
    for i in range(1, n_pasos + 1):
        precio       = atl * (factor ** i)
        t_lineal     = (i - 1) / (n_pasos - 1) if n_pasos > 1 else 0.0
        t            = t_lineal ** 0.5
        venta_pct    = MIN_VENTA_PCT    + t * (MAX_VENTA_PCT    - MIN_VENTA_PCT)
        acumular_pct = MAX_ACUMULAR_PCT - t * (MAX_ACUMULAR_PCT - MIN_ACUMULAR_PCT)
        btc_objetivo = btc_ref * (venta_pct / 100.0)
        niveles.append({
            "precio"       : precio,
            "venta_pct"    : venta_pct,
            "acumular_pct" : acumular_pct,
            "btc_objetivo" : btc_objetivo,
            "distancia_pct": (factor**i - 1.0) * 100,
        })
    return niveles


# ─────────────────────────────────────────────────────────────
# GRAFICO
# ─────────────────────────────────────────────────────────────

def graficar(df_precio: pd.DataFrame, resultados: dict):

    summary      = resultados["summary"]
    trades       = resultados["trade_history"]
    compras      = [t for t in trades if t["type"] == "BUY"]
    ventas       = [t for t in trades if t["type"] == "SELL"]

    ath_final    = summary["ath_final"]
    atl_final    = summary["atl_final"]
    usdt_final   = summary["usdt_balance_final"]
    btc_pos_final = summary["btc_en_posiciones_final"]

    # Reconstruir tablas con valores finales
    print("Reconstruyendo tablas de ordenes finales...")
    niveles_compra = reconstruir_tabla_compras(ath_final, usdt_final)
    niveles_venta  = reconstruir_tabla_ventas(atl_final, btc_pos_final if btc_pos_final > 0 else 1.0)

    print(f"  Niveles de compra: {len(niveles_compra):,}")
    print(f"  Niveles de venta : {len(niveles_venta):,}")
    print(f"  Compras ejecutadas: {len(compras):,}")
    print(f"  Ventas ejecutadas : {len(ventas):,}")

    # Rango de precio visible
    precio_min = df_precio["close"].min()
    precio_max = df_precio["close"].max()
    fecha_ini  = df_precio["datetime"].iloc[0]
    fecha_fin  = df_precio["datetime"].iloc[-1]

    # ── Figura pantalla completa ──────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 12))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.06)

    # ── Precio BTC ───────────────────────────────────────────
    ax.plot(df_precio["datetime"], df_precio["close"],
            color="#4a9eff", linewidth=0.6, alpha=0.7, zorder=2, label="Precio BTC")

    # ── Niveles de COMPRA ────────────────────────────────────
    # Intensidad proporcional al % del nivel (mayor % = mas opaco)
    pcts_c = [n["pct_nivel"] for n in niveles_compra]
    pct_c_min, pct_c_max = min(pcts_c), max(pcts_c)

    for nivel in niveles_compra:
        precio = nivel["precio"]
        # Solo dibujar si esta en el rango visible
        if precio_min * 0.5 <= precio <= precio_max * 1.5:
            alpha = 0.15 + 0.55 * (nivel["pct_nivel"] - pct_c_min) / (pct_c_max - pct_c_min + 1e-10)
            ax.axhline(precio, color="#ff4444", linewidth=0.5,
                       alpha=alpha, zorder=1)

    # ── Niveles de VENTA ─────────────────────────────────────
    pcts_v = [n["venta_pct"] for n in niveles_venta]
    pct_v_min, pct_v_max = min(pcts_v), max(pcts_v)

    for nivel in niveles_venta:
        precio = nivel["precio"]
        if precio_min * 0.5 <= precio <= precio_max * 1.5:
            alpha = 0.15 + 0.55 * (nivel["venta_pct"] - pct_v_min) / (pct_v_max - pct_v_min + 1e-10)
            ax.axhline(precio, color="#44ff88", linewidth=0.5,
                       alpha=alpha, zorder=1)

    # ── ATH y ATL ────────────────────────────────────────────
    ax.axhline(ath_final, color="#ffd700", linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=3)
    ax.text(fecha_fin, ath_final * 1.01,
            f"ATH ${ath_final:,.0f}", color="#ffd700",
            fontsize=8, va="bottom", ha="right")

    ax.axhline(atl_final, color="#ff8c00", linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=3)
    ax.text(fecha_fin, atl_final * 0.99,
            f"ATL ${atl_final:,.2f}", color="#ff8c00",
            fontsize=8, va="top", ha="right")

    # ── Compras ejecutadas ───────────────────────────────────
    if compras:
        fechas_c  = pd.to_datetime([t["datetime"] for t in compras])
        precios_c = [t["price"] for t in compras]
        montos_c  = [t["usdt_spent"] or 0 for t in compras]
        # Tamaño del punto proporcional al monto
        m_min, m_max = min(montos_c), max(montos_c) if max(montos_c) > 0 else 1
        sizes_c = [8 + 40 * (m - m_min) / (m_max - m_min + 1e-10) for m in montos_c]
        ax.scatter(fechas_c, precios_c, s=sizes_c,
                   color="#00aaff", alpha=0.7, zorder=5,
                   label=f"Compras ejecutadas ({len(compras):,})")

    # ── Ventas ejecutadas ────────────────────────────────────
    if ventas:
        fechas_v  = pd.to_datetime([t["datetime"] for t in ventas])
        precios_v = [t["price"] for t in ventas]
        montos_v  = [t["usdt_received"] or 0 for t in ventas]
        m_min, m_max = min(montos_v), max(montos_v) if max(montos_v) > 0 else 1
        sizes_v = [8 + 40 * (m - m_min) / (m_max - m_min + 1e-10) for m in montos_v]
        ax.scatter(fechas_v, precios_v, s=sizes_v,
                   color="#ff8800", alpha=0.7, zorder=5,
                   label=f"Ventas ejecutadas ({len(ventas):,})")

    # ── Escala y ejes ────────────────────────────────────────
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}" if x >= 1 else f"${x:.2f}"
    ))
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(True, which="major", color="#222", linewidth=0.5, alpha=0.8)
    ax.grid(True, which="minor", color="#1a1a1a", linewidth=0.3, alpha=0.5)

    # ── Leyenda ──────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="#4a9eff", linewidth=1.5, label="Precio BTC"),
        Line2D([0], [0], color="#ff4444", linewidth=1.5,
               label=f"Niveles compra ({len(niveles_compra):,})  PASO={PASO_PCT_COMPRA}%  MAX=-{CAIDA_MAXIMA:.0f}%"),
        Line2D([0], [0], color="#44ff88", linewidth=1.5,
               label=f"Niveles venta  ({len(niveles_venta):,})  PASO={PASO_PCT_VENTA}%  MAX=+{SUBIDA_MAXIMA:.0f}%"),
        Line2D([0], [0], color="#ffd700", linewidth=1.5,
               linestyle="--", label=f"ATH ${ath_final:,.0f}"),
        Line2D([0], [0], color="#ff8c00", linewidth=1.5,
               linestyle="--", label=f"ATL ${atl_final:,.4f}"),
        plt.scatter([], [], s=25, color="#00aaff",
                    label=f"Compras ejecutadas ({len(compras):,})"),
        plt.scatter([], [], s=25, color="#ff8800",
                    label=f"Ventas ejecutadas ({len(ventas):,})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", fontsize=8)

    # ── Titulo ───────────────────────────────────────────────
    ax.set_title(
        f"Grid ATH/ATL — Niveles de Ordenes y Operaciones Ejecutadas  |  "
        f"{summary['fecha_inicio'][:10]} → {summary['fecha_fin'][:10]}  |  "
        f"PnL: {summary['pnl_pct']:+.2f}%  |  "
        f"Trades: {summary['total_trades']:,}",
        color="white", fontsize=11, fontweight="bold", pad=12
    )

    # Anotacion de intensidad de lineas
    ax.text(0.99, 0.02,
            "Intensidad de linea proporcional al % de capital por orden",
            transform=ax.transAxes, color="#888", fontsize=7,
            ha="right", va="bottom")

    nombre = "grid_ordenes_visualizacion.png"
    plt.savefig(nombre, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nGrafico guardado: {nombre}")

    # Pantalla completa
    try:
        manager = plt.get_current_fig_manager()
        manager.window.state("zoomed")   # Windows
    except Exception:
        try:
            manager.full_screen_toggle()  # Linux / Mac
        except Exception:
            pass

    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  VISUALIZADOR DE GRID ATH/ATL — BTC/USDT")
    print("=" * 62)

    df_precio  = cargar_precio(muestra=10)
    resultados = cargar_resultados()

    print(f"\nResumen del backtest:")
    s = resultados["summary"]
    print(f"  Periodo    : {s['fecha_inicio'][:10]} → {s['fecha_fin'][:10]}")
    print(f"  PnL        : {s['pnl_pct']:+.2f}%")
    print(f"  ATH final  : ${s['ath_final']:,.2f}")
    print(f"  ATL final  : ${s['atl_final']:,.4f}")
    print(f"  Compras    : {s['total_compras']:,}")
    print(f"  Ventas     : {s['total_ventas']:,}")

    graficar(df_precio, resultados)


if __name__ == "__main__":
    main()
