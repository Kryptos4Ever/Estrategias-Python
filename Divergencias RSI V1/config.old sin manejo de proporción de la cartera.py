"""
Configuración centralizada — Estrategia Divergencia RSI · Capital Dinámico por Racha
BTC/USDT · Velas Horarias
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000

# ── Señal de divergencia RSI ──────────────────────────────────────────────────
RSI_LENGTH = 12       # Período del RSI (validado: mejor correlación con divergencias)
N          = 5        # Ventana de búsqueda del extremo local en velas anteriores

# ── Gestión de capital por racha (progresión geométrica) ─────────────────────
#
# Al inicio de cada nueva racha se pre-calculan STREAK slots distribuyendo
# el capital disponible según una progresión geométrica de razón R:
#
#   slots = [a, a·r, a·r², ..., a·r^(n-1)]
#   donde a = Capital · (r-1) / (r^n - 1)
#
# COMPRAS  — capital base: usdt_balance disponible
#   R > 1 → más USDT en trades tardíos (precio más bajo dentro de la racha)
#   R = 1 → distribución uniforme
#   R < 1 → más USDT en trades tempranos
#
# VENTAS   — capital base: btc_en_posiciones
#   R > 1 → más BTC vendido en trades tardíos (precio más alto dentro de la racha)
#   R = 1 → distribución uniforme
#   R < 1 → más BTC vendido en trades tempranos
#
STREAK_COMPRAS = 7    # Slots máximos por racha de compra  (señales > STREAK se ignoran)
STREAK_VENTAS  = 7    # Slots máximos por racha de venta

R_COMPRA = 1.2        # Razón geométrica compras
                      # (1.5 drena USDT muy rápido en bear markets — 1.2 más conservador)
R_VENTA  = 1.5        # Razón geométrica ventas

# ── Reserva mínima de USDT ────────────────────────────────────────────────────
# Nunca se compromete USDT por debajo de este umbral.
# Protege la capacidad de compra en caídas prolongadas.
# Expresado como % del SALDO_USDT_INICIAL.
# Ej: 10 → siempre se reservan $100 si el capital inicial fue $1000.
USDT_RESERVA_PCT = 10   # % del saldo inicial (0 = sin reserva)

# ── Acumulación de BTC ────────────────────────────────────────────────────────
# % del slot de BTC procesado en cada venta que se desvía a btc_balance libre
# (nunca se vende, se acumula permanentemente)
BTC_PCT_TO_ACCUMULATE = 0

# ── Comisión del exchange ─────────────────────────────────────────────────────
COMMISSION_PCT = 0.1    # % por operación (0.1 = estándar Binance spot)

# ── Rutas ─────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results_div_rsi.json"   # ← nombre unificado para Graficador.py

# ── Rango de fechas ───────────────────────────────────────────────────────────
# Formato: 'YYYY-MM-DD'  |  None = desde el inicio / hasta el final
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2022-11-22'

# Referencias útiles de fechas históricas:
# Rango de fechas (None = todos los datos)
# Fecha Botom Bear 2018 : '2018-12-10'
# Fecha pre COVID       : '2019-06-27'
# Fecha Inicio Bull 2020: '2020-03-17'
# Fecha TOP1 2021       : '2021-04-14'
# Fecha TOP2 2021       : '2021-11-10'
# Fecha Botom Bear 2022 : '2022-11-22'
USDT_PCT_TO_USE      = 1
BTC_PCT_TO_SELL      = 1


def mostrar_configuracion():
    """Imprime la configuración activa en consola."""
    usdt_reserva = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100
    print("CONFIGURACIÓN DE LA ESTRATEGIA")
    print("=" * 52)
    print(f"  Saldo inicial      : ${SALDO_USDT_INICIAL:,}")
    print(f"  Fecha inicio       : {FECHA_INICIO or 'Desde el inicio'}")
    print(f"  Fecha fin          : {FECHA_FIN    or 'Hasta el final'}")
    print(f"  RSI_LENGTH         : {RSI_LENGTH}")
    print(f"  N (ventana)        : {N}")
    print(f"  STREAK_COMPRAS     : {STREAK_COMPRAS}   R_COMPRA: {R_COMPRA}")
    print(f"  STREAK_VENTAS      : {STREAK_VENTAS}   R_VENTA : {R_VENTA}")
    print(f"  USDT reserva       : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f} intocables")
    print(f"  BTC acumulación    : {BTC_PCT_TO_ACCUMULATE}% por venta")
    print(f"  Comisión           : {COMMISSION_PCT}%")
    print(f"  Resultados JSON    : {RESULTS_JSON}")
    print()