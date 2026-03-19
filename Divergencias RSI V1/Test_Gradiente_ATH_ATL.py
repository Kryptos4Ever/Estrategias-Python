"""
Test Gradiente ATH/ATL — Verifica la asignación dinámica de capital
====================================================================
Muestra tablas y gráficos de las curvas de capital para compras (ATH)
y ventas (ATL) con los parámetros actuales del config.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    SALDO_USDT_INICIAL,
    ATH_CAIDA_MAXIMA, FACTOR_CAIDA,
    ATL_SUBIDA_MAXIMA, FACTOR_SUBIDA,
    STREAK_COMPRAS, STREAK_VENTAS,
    R_COMPRA, R_VENTA,
)

# ── Parámetros del test ───────────────────────────────────────────────────────
ATH_TEST          = 69_000.0
ATL_TEST          = 15_476.0
USDT_TEST         = SALDO_USDT_INICIAL
BTC_TEST          = 0.0285

# Precios de ejemplo para simular señales de compra y venta
PRECIOS_COMPRA_TEST = [65_000, 55_000, 45_000, 35_000, 25_000, 20_000, 14_000]
PRECIOS_VENTA_TEST  = [18_000, 25_000, 35_000, 50_000, 62_000, 75_000, 90_000]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNCIONES DE GRADIENTE (idénticas a la estrategia)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pct_capital_compra(precio: float, ath: float) -> float:
    if ath <= 0: return 0.0
    caida = (ath - precio) / ath * 100
    ratio = max(0.0, min(1.0, caida / ATH_CAIDA_MAXIMA))
    return (ratio ** FACTOR_CAIDA) * 100.0

def pct_capital_venta(precio: float, atl: float) -> float:
    if atl <= 0: return 0.0
    subida = (precio - atl) / atl * 100
    ratio  = max(0.0, min(1.0, subida / ATL_SUBIDA_MAXIMA))
    return (ratio ** FACTOR_SUBIDA) * 100.0

def calcular_slots(capital: float, n: int, r: float) -> list:
    if capital <= 0 or n <= 0: return []
    if abs(r - 1.0) < 1e-9: return [capital / n] * n
    s = (r ** n - 1) / (r - 1)
    a = capital / s
    return [a * (r ** i) for i in range(n)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABLAS DE TEXTO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def imprimir_tabla_compras():
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  GRADIENTE DE COMPRAS  ·  ATH=${ATH_TEST:,.0f}  ·  "
          f"CAIDA_MAXIMA={ATH_CAIDA_MAXIMA}%  ·  FACTOR_CAIDA={FACTOR_CAIDA}"
          f"  ·  R_COMPRA={R_COMPRA}  ·  STREAK={STREAK_COMPRAS}")
    print(sep)
    print(f"  {'Precio':>10}  {'Caída%':>7}  {'pct_USDT':>9}  {'Capital_racha':>14}  "
          f"  {'Slot1':>10}  {'Slot2':>10}  {'SlotN':>10}  {'Suma':>10}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for precio in PRECIOS_COMPRA_TEST:
        pct     = pct_capital_compra(precio, ATH_TEST)
        capital = USDT_TEST * pct / 100
        slots   = calcular_slots(capital, STREAK_COMPRAS, R_COMPRA)
        caida   = (ATH_TEST - precio) / ATH_TEST * 100
        s1      = slots[0]  if slots else 0
        s2      = slots[1]  if len(slots) > 1 else 0
        sn      = slots[-1] if slots else 0
        suma    = sum(slots)
        flag    = "  ◄ CLAMPED" if caida >= ATH_CAIDA_MAXIMA else ""
        print(f"  ${precio:>9,.0f}  {caida:>6.1f}%  {pct:>8.2f}%  ${capital:>13,.4f}  "
              f"  ${s1:>9,.4f}  ${s2:>9,.4f}  ${sn:>9,.4f}  ${suma:>9,.4f}{flag}")

    print(sep)

def imprimir_tabla_ventas():
    sep = "=" * 96
    print(f"\n{sep}")
    print(f"  GRADIENTE DE VENTAS  ·  ATL=${ATL_TEST:,.0f}  ·  "
          f"SUBIDA_MAXIMA={ATL_SUBIDA_MAXIMA}%  ·  FACTOR_SUBIDA={FACTOR_SUBIDA}"
          f"  ·  R_VENTA={R_VENTA}  ·  STREAK={STREAK_VENTAS}")
    print(sep)
    print(f"  {'Precio':>10}  {'Subida%':>8}  {'pct_BTC':>8}  {'BTC_racha':>12}  "
          f"  {'Slot1':>12}  {'Slot2':>12}  {'SlotN':>12}  {'Suma':>12}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for precio in PRECIOS_VENTA_TEST:
        pct   = pct_capital_venta(precio, ATL_TEST)
        btc_r = BTC_TEST * pct / 100
        slots = calcular_slots(btc_r, STREAK_VENTAS, R_VENTA)
        subida = (precio - ATL_TEST) / ATL_TEST * 100
        s1     = slots[0]  if slots else 0
        s2     = slots[1]  if len(slots) > 1 else 0
        sn     = slots[-1] if slots else 0
        suma   = sum(slots)
        flag   = "  ◄ CLAMPED" if subida >= ATL_SUBIDA_MAXIMA else ""
        print(f"  ${precio:>9,.0f}  {subida:>7.1f}%  {pct:>7.2f}%  {btc_r:>12.7f}  "
              f"  {s1:>12.7f}  {s2:>12.7f}  {sn:>12.7f}  {suma:>12.7f}{flag}")

    print(sep)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generar_graficos():
    # ── Paleta ────────────────────────────────────────────────────────────────
    COLOR_BG    = "#0e1117"
    COLOR_PANEL = "#161b22"
    COLOR_BUY   = "#2ecc71"
    COLOR_SELL  = "#e74c3c"
    COLOR_GRID  = "#2a2d35"
    COLOR_TEXT  = "#c9d1d9"
    COLOR_ATH   = "#f39c12"
    COLOR_ATL   = "#9b59b6"

    fig = plt.figure(figsize=(18, 14), facecolor=COLOR_BG)
    fig.suptitle(
        f"Test Gradiente ATH/ATL  ·  ATH=${ATH_TEST:,.0f}  ·  ATL=${ATL_TEST:,.0f}\n"
        f"CAIDA_MAXIMA={ATH_CAIDA_MAXIMA}%  FACTOR_CAIDA={FACTOR_CAIDA}  "
        f"SUBIDA_MAXIMA={ATL_SUBIDA_MAXIMA}%  FACTOR_SUBIDA={FACTOR_SUBIDA}  "
        f"R_COMPRA={R_COMPRA}  R_VENTA={R_VENTA}  STREAK={STREAK_COMPRAS}/{STREAK_VENTAS}",
        color=COLOR_TEXT, fontsize=12, y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.06)

    def estilo_ax(ax, titulo):
        ax.set_facecolor(COLOR_PANEL)
        ax.set_title(titulo, color=COLOR_TEXT, fontsize=10, pad=8)
        ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        ax.spines[:].set_color(COLOR_GRID)
        ax.grid(True, color=COLOR_GRID, linewidth=0.5, alpha=0.7)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(COLOR_TEXT)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 1 — Curva pct_USDT vs precio de compra
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax1 = fig.add_subplot(gs[0, 0])
    precios_c = np.linspace(ATH_TEST * 0.05, ATH_TEST, 400)
    pcts_c    = [pct_capital_compra(p, ATH_TEST) for p in precios_c]
    ax1.plot(precios_c / 1000, pcts_c, color=COLOR_BUY, linewidth=2.2)
    ax1.fill_between(precios_c / 1000, pcts_c, alpha=0.18, color=COLOR_BUY)
    ax1.axvline(ATH_TEST / 1000, color=COLOR_ATH, linewidth=1.2,
                linestyle="--", label=f"ATH ${ATH_TEST/1000:.0f}k")
    # Marcar los precios de test
    for p in PRECIOS_COMPRA_TEST:
        if p <= ATH_TEST:
            pct = pct_capital_compra(p, ATH_TEST)
            ax1.scatter(p / 1000, pct, color=COLOR_BUY, s=50, zorder=5)
            ax1.annotate(f"${p//1000}k\n{pct:.1f}%",
                         (p / 1000, pct), textcoords="offset points",
                         xytext=(6, 4), fontsize=7, color=COLOR_TEXT)
    ax1.set_xlabel("Precio BTC (miles $)", color=COLOR_TEXT, fontsize=8)
    ax1.set_ylabel("% USDT comprometido", color=COLOR_TEXT, fontsize=8)
    ax1.legend(fontsize=8, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXT,
               edgecolor=COLOR_GRID)
    estilo_ax(ax1, "Gradiente COMPRAS — % USDT vs Precio")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 2 — Curva pct_BTC vs precio de venta
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax2 = fig.add_subplot(gs[0, 1])
    p_max_v   = ATL_TEST * (1 + ATL_SUBIDA_MAXIMA / 100)
    precios_v = np.linspace(ATL_TEST, p_max_v * 1.2, 400)
    pcts_v    = [pct_capital_venta(p, ATL_TEST) for p in precios_v]
    ax2.plot(precios_v / 1000, pcts_v, color=COLOR_SELL, linewidth=2.2)
    ax2.fill_between(precios_v / 1000, pcts_v, alpha=0.18, color=COLOR_SELL)
    ax2.axvline(ATL_TEST / 1000, color=COLOR_ATL, linewidth=1.2,
                linestyle="--", label=f"ATL ${ATL_TEST/1000:.1f}k")
    ax2.axvline(p_max_v / 1000, color=COLOR_ATH, linewidth=1.2,
                linestyle=":", label=f"SUBIDA_MAX ${p_max_v/1000:.0f}k")
    for p in PRECIOS_VENTA_TEST:
        pct = pct_capital_venta(p, ATL_TEST)
        ax2.scatter(p / 1000, pct, color=COLOR_SELL, s=50, zorder=5)
        ax2.annotate(f"${p//1000}k\n{pct:.1f}%",
                     (p / 1000, pct), textcoords="offset points",
                     xytext=(6, 4), fontsize=7, color=COLOR_TEXT)
    ax2.set_xlabel("Precio BTC (miles $)", color=COLOR_TEXT, fontsize=8)
    ax2.set_ylabel("% BTC comprometido", color=COLOR_TEXT, fontsize=8)
    ax2.legend(fontsize=8, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXT,
               edgecolor=COLOR_GRID)
    estilo_ax(ax2, "Gradiente VENTAS — % BTC vs Precio")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 3 — Capital USDT absoluto por racha de compra
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax3 = fig.add_subplot(gs[1, 0])
    caps_c = [USDT_TEST * pct_capital_compra(p, ATH_TEST) / 100 for p in precios_c]
    ax3.plot(precios_c / 1000, caps_c, color=COLOR_BUY, linewidth=2.2)
    ax3.fill_between(precios_c / 1000, caps_c, alpha=0.18, color=COLOR_BUY)
    # Slots del STREAK para cada precio de test
    for p in PRECIOS_COMPRA_TEST:
        if p <= ATH_TEST:
            cap   = USDT_TEST * pct_capital_compra(p, ATH_TEST) / 100
            slots = calcular_slots(cap, STREAK_COMPRAS, R_COMPRA)
            if slots:
                ax3.scatter(p / 1000, cap, color=COLOR_BUY, s=60, zorder=5)
                # Barras de slots
                offset = p / 1000
                for j, s in enumerate(slots):
                    ax3.bar(offset, s, bottom=sum(slots[:j]),
                            width=700, color=COLOR_BUY,
                            alpha=0.12 + 0.11 * j, edgecolor="none")
    ax3.set_xlabel("Precio BTC (miles $)", color=COLOR_TEXT, fontsize=8)
    ax3.set_ylabel("USDT capital de racha ($)", color=COLOR_TEXT, fontsize=8)
    estilo_ax(ax3, "Capital USDT por Racha de Compra (barras = slots)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 4 — BTC absoluto por racha de venta
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax4 = fig.add_subplot(gs[1, 1])
    caps_v = [BTC_TEST * pct_capital_venta(p, ATL_TEST) / 100 for p in precios_v]
    ax4.plot(precios_v / 1000, caps_v, color=COLOR_SELL, linewidth=2.2)
    ax4.fill_between(precios_v / 1000, caps_v, alpha=0.18, color=COLOR_SELL)
    for p in PRECIOS_VENTA_TEST:
        btc_r = BTC_TEST * pct_capital_venta(p, ATL_TEST) / 100
        slots = calcular_slots(btc_r, STREAK_VENTAS, R_VENTA)
        if slots:
            ax4.scatter(p / 1000, btc_r, color=COLOR_SELL, s=60, zorder=5)
            offset = p / 1000
            for j, s in enumerate(slots):
                ax4.bar(offset, s, bottom=sum(slots[:j]),
                        width=(p_max_v * 1.2 - ATL_TEST) / 400 * 30,
                        color=COLOR_SELL, alpha=0.12 + 0.11 * j,
                        edgecolor="none")
    ax4.set_xlabel("Precio BTC (miles $)", color=COLOR_TEXT, fontsize=8)
    ax4.set_ylabel("BTC capital de racha (₿)", color=COLOR_TEXT, fontsize=8)
    estilo_ax(ax4, "BTC por Racha de Venta (barras = slots)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 5 — Comparativa de factores FACTOR_CAIDA (sensibilidad)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax5 = fig.add_subplot(gs[2, 0])
    caidas = np.linspace(0, ATH_CAIDA_MAXIMA, 300)
    factores_test = [0.5, 1.0, FACTOR_CAIDA, 3.0, 4.0]
    colores_f = ["#aaaaaa", "#5dade2", COLOR_BUY, "#f39c12", "#e74c3c"]
    for fc, col in zip(factores_test, colores_f):
        pcts = [(c / ATH_CAIDA_MAXIMA) ** fc * 100 for c in caidas]
        lw   = 2.8 if fc == FACTOR_CAIDA else 1.4
        ls   = "-" if fc == FACTOR_CAIDA else "--"
        label = f"factor={fc}  ← ACTIVO" if fc == FACTOR_CAIDA else f"factor={fc}"
        ax5.plot(caidas, pcts, color=col, linewidth=lw, linestyle=ls, label=label)
    ax5.set_xlabel("Caída desde ATH (%)", color=COLOR_TEXT, fontsize=8)
    ax5.set_ylabel("% USDT comprometido", color=COLOR_TEXT, fontsize=8)
    ax5.legend(fontsize=7.5, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXT,
               edgecolor=COLOR_GRID)
    estilo_ax(ax5, "Sensibilidad FACTOR_CAIDA — Compras")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 6 — Comparativa de factores FACTOR_SUBIDA (sensibilidad)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax6 = fig.add_subplot(gs[2, 1])
    subidas = np.linspace(0, ATL_SUBIDA_MAXIMA, 300)
    factores_test_v = [0.5, 1.0, FACTOR_SUBIDA, 3.0, 4.0]
    for fs, col in zip(factores_test_v, colores_f):
        pcts = [(s / ATL_SUBIDA_MAXIMA) ** fs * 100 for s in subidas]
        lw   = 2.8 if fs == FACTOR_SUBIDA else 1.4
        ls   = "-" if fs == FACTOR_SUBIDA else "--"
        label = f"factor={fs}  ← ACTIVO" if fs == FACTOR_SUBIDA else f"factor={fs}"
        ax6.plot(subidas, pcts, color=col, linewidth=lw, linestyle=ls, label=label)
    ax6.set_xlabel("Subida desde ATL (%)", color=COLOR_TEXT, fontsize=8)
    ax6.set_ylabel("% BTC comprometido", color=COLOR_TEXT, fontsize=8)
    ax6.legend(fontsize=7.5, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXT,
               edgecolor=COLOR_GRID)
    estilo_ax(ax6, "Sensibilidad FACTOR_SUBIDA — Ventas")

    plt.savefig("test_gradiente_ath_atl.png", dpi=150, bbox_inches="tight",
                facecolor=COLOR_BG)
    print("\n✓ Gráfico guardado: test_gradiente_ath_atl.png")
    plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    sep = "=" * 90
    print(sep)
    print("  TEST GRADIENTE ATH/ATL — ASIGNACIÓN DINÁMICA DE CAPITAL")
    print(sep)
    print(f"  ATH test        : ${ATH_TEST:>10,.0f}")
    print(f"  ATL test        : ${ATL_TEST:>10,.0f}")
    print(f"  USDT disponible : ${USDT_TEST:>10,.2f}")
    print(f"  BTC posiciones  :  {BTC_TEST:>10.4f} ₿")
    print(f"  CAIDA_MAXIMA    : {ATH_CAIDA_MAXIMA}%   FACTOR_CAIDA  : {FACTOR_CAIDA}")
    print(f"  SUBIDA_MAXIMA   : {ATL_SUBIDA_MAXIMA}%  FACTOR_SUBIDA : {FACTOR_SUBIDA}")
    print(f"  STREAK          : {STREAK_COMPRAS} compras / {STREAK_VENTAS} ventas")
    print(f"  R               : {R_COMPRA} compras / {R_VENTA} ventas")

    imprimir_tabla_compras()
    imprimir_tabla_ventas()
    generar_graficos()


if __name__ == "__main__":
    main()
