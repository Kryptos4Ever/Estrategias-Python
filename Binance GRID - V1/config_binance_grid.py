"""
Configuración — Estrategia Binance Spot Grid Bot
══════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Simulación local fiel al bot de Binance

Parámetros idénticos a los que expone Binance en su interfaz.
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000       # inversión total del bot

# ── Rango de precios del grid ─────────────────────────────────────────────────
#
#   El bot opera SOLO dentro de este rango.
#   Si el precio sale por debajo de PRECIO_INFERIOR: el bot queda 100% en BTC
#   (todas las buy orders ejecutadas, ninguna sell activa dentro del rango).
#   Si el precio sale por encima de PRECIO_SUPERIOR: el bot queda 100% en USDT
#   (todas las sell orders ejecutadas).
#
PRECIO_SUPERIOR = 69000         # límite superior del grid (upper price)
PRECIO_INFERIOR = 15620         # límite inferior del grid (lower price)

# ── Número de grids ───────────────────────────────────────────────────────────
#
#   NUM_GRIDS niveles de compra → NUM_GRIDS + 1 precios de referencia.
#   Capital por nivel = SALDO_USDT_INICIAL / NUM_GRIDS  (uniforme).
#
NUM_GRIDS = 100                  # número de intervalos (niveles de compra)

# ── Modo del grid ─────────────────────────────────────────────────────────────
#
#   "aritmetico"  → diferencia de precio constante entre niveles.
#                   Más órdenes concentradas en precios altos (misma distancia $ en todos).
#
#   "geometrico"  → ratio de precio constante entre niveles (igual % de diferencia).
#                   Más órdenes en precios bajos; equivale a PCT_CAIDA constante.
#
MODO_GRID = "geometrico"        # "aritmetico" | "geometrico"

# ── Stop Loss global (opcional) ───────────────────────────────────────────────
#
#   Si el precio de cierre de una vela cae por debajo de STOP_LOSS:
#     - Se venden TODAS las posiciones abiertas al precio STOP_LOSS.
#     - El bot se detiene definitivamente.
#   Poner None para desactivar.
#
STOP_LOSS = None                # ej: 14000.0 | None

# ── Take Profit global (opcional) ─────────────────────────────────────────────
#
#   Si el precio de cierre de una vela sube por encima de TAKE_PROFIT:
#     - Se venden TODAS las posiciones abiertas al precio TAKE_PROFIT.
#     - El bot se detiene definitivamente.
#   Poner None para desactivar.
#
TAKE_PROFIT = None              # ej: 80000.0 | None

# ── Comisión ──────────────────────────────────────────────────────────────────
COMMISSION_PCT = 0.1            # % sobre el monto de cada operación

# ── Reserva de capital ────────────────────────────────────────────────────────
USDT_RESERVA_PCT = 0            # % del saldo inicial que nunca se opera

# ── Rutas ─────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results_binance_grid.json"

# ── Rango de fechas ───────────────────────────────────────────────────────────
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2024-03-04'


def mostrar_configuracion():
    import math
    usdt_reserva    = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100
    capital_operable = SALDO_USDT_INICIAL - usdt_reserva
    capital_por_grid = capital_operable / NUM_GRIDS

    if MODO_GRID == "geometrico":
        ratio = (PRECIO_INFERIOR / PRECIO_SUPERIOR) ** (1 / NUM_GRIDS)
        pct_spacing = (1 - ratio) * 100
        spacing_info = f"ratio={ratio:.6f}  ({pct_spacing:.2f}% entre niveles)"
    else:
        spacing = (PRECIO_SUPERIOR - PRECIO_INFERIOR) / NUM_GRIDS
        spacing_info = f"${spacing:,.2f} entre niveles"

    print("CONFIGURACIÓN — BINANCE SPOT GRID BOT")
    print("=" * 66)
    print(f"  Saldo inicial          : ${SALDO_USDT_INICIAL:,}")
    print(f"  Período                : {FECHA_INICIO or 'inicio'}  →  {FECHA_FIN or 'fin'}")
    print(f"  ── Rango del grid ────────────────────────────────────────")
    print(f"  PRECIO_SUPERIOR        : ${PRECIO_SUPERIOR:,}")
    print(f"  PRECIO_INFERIOR        : ${PRECIO_INFERIOR:,}")
    print(f"  NUM_GRIDS              : {NUM_GRIDS}  ({NUM_GRIDS+1} precios de referencia)")
    print(f"  MODO_GRID              : {MODO_GRID.upper()}")
    print(f"  Spacing                : {spacing_info}")
    print(f"  Capital por nivel      : ${capital_por_grid:,.4f}")
    print(f"  ── Protecciones ──────────────────────────────────────────")
    print(f"  STOP_LOSS              : {'desactivado' if STOP_LOSS is None else f'${STOP_LOSS:,}'}")
    print(f"  TAKE_PROFIT            : {'desactivado' if TAKE_PROFIT is None else f'${TAKE_PROFIT:,}'}")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  USDT reserva           : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f}")
    print(f"  Comisión               : {COMMISSION_PCT}%")
    print(f"  JSON salida            : {RESULTS_JSON}")
    print()
