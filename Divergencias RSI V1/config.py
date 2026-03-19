"""
Configuración centralizada
Estrategia: Divergencia RSI · Capital Dinámico ATH/ATL por Racha
BTC/USDT · Velas Horarias
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000

# ── Señal de divergencia RSI ──────────────────────────────────────────────────
RSI_LENGTH = 15       # Período del RSI
N          = 5        # Ventana de búsqueda del extremo local (velas anteriores)

# ── Gestión de capital por racha (progresión geométrica) ─────────────────────
#
# Al inicio de cada nueva racha se pre-calculan STREAK slots distribuyendo
# el capital de la racha según progresión geométrica de razón R:
#
#   slots = [a, a·r, a·r², ..., a·r^(n-1)]
#   donde: a = Capital_racha · (r-1) / (r^n - 1)   si r ≠ 1
#          a = Capital_racha / n                      si r = 1
#
# El Capital_racha queda determinado por los gradientes ATH/ATL (ver abajo).
# Si la racha supera STREAK → señal ignorada.
#
STREAK_COMPRAS = 7    # Slots máximos por racha de compra
STREAK_VENTAS  = 7    # Slots máximos por racha de venta
R_COMPRA       = 1  # Razón geométrica dentro de la racha de compra
                      #   r > 1 → más USDT en slots tardíos (precio más bajo)
R_VENTA        = 1  # Razón geométrica dentro de la racha de venta
                      #   r > 1 → más BTC en slots tardíos (precio más alto)

# ── Gradiente de capital de compra — basado en caída desde ATH ───────────────
#
#   caida_actual%  = (ATH - precio_low) / ATH × 100
#   pct_usdt       = clamp(caida_actual / ATH_CAIDA_MAXIMA, 0, 1) ^ FACTOR_CAIDA × 100
#   capital_racha  = usdt_disponible × pct_usdt / 100
#
# Con ATH_CAIDA_MAXIMA=80, FACTOR_CAIDA=2:
#   caída  10% → pct_usdt =  1.6%
#   caída  30% → pct_usdt = 14.1%
#   caída  50% → pct_usdt = 39.1%
#   caída  70% → pct_usdt = 76.6%
#   caída  80% → pct_usdt =100.0%  ← usa todo el USDT disponible
#
ATH_CAIDA_MAXIMA = 80    # % caída desde ATH donde pct_usdt = 100%
FACTOR_CAIDA     = 4.0   # Curvatura  (>1 convexo · =1 lineal · <1 cóncavo)

# ── Gradiente de BTC de venta — basado en subida desde ATL ───────────────────
#
#   subida_actual% = (precio_high - ATL) / ATL × 100
#   pct_btc        = clamp(subida_actual / ATL_SUBIDA_MAXIMA, 0, 1) ^ FACTOR_SUBIDA × 100
#   btc_racha      = btc_en_posiciones × pct_btc / 100
#
# Con ATL=$15,500, ATL_SUBIDA_MAXIMA=300, FACTOR_SUBIDA=2:
#   precio $20,000 → subida  29% → pct_btc =  0.9%  ← fondo, casi no vende
#   precio $30,000 → subida  94% → pct_btc =  9.8%
#   precio $50,000 → subida 223% → pct_btc = 55.3%
#   precio $62,000 → subida 300% → pct_btc =100.0%  ← usa todo el BTC en posiciones
#
ATL_SUBIDA_MAXIMA = 750  # % subida desde ATL donde pct_btc = 100%
FACTOR_SUBIDA     = 4.0  # Curvatura  (>1 convexo · =1 lineal · <1 cóncavo)

# ── Reserva mínima de USDT ────────────────────────────────────────────────────
# Piso absoluto nunca comprometido en compras (segunda línea de defensa).
# Expresado como % del SALDO_USDT_INICIAL.
USDT_RESERVA_PCT = 0     # % del saldo inicial (0 = sin reserva)

# ── Acumulación de BTC ────────────────────────────────────────────────────────
# % del slot BTC procesado en cada venta que va a btc_balance libre
# (nunca se vende, acumulación permanente).
BTC_PCT_TO_ACCUMULATE = 0

# ── Comisión del exchange ─────────────────────────────────────────────────────
COMMISSION_PCT = 0.1

# ── Rutas ─────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results_div_rsi.json"

# ── Rango de fechas ───────────────────────────────────────────────────────────
# Formato: 'YYYY-MM-DD'  |  None = desde el inicio / hasta el final
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2022-11-22'

# Referencias útiles:
# Rango de fechas (None = todos los datos)
# Fecha Botom Bear 2018 : '2018-12-10'
# Fecha pre COVID       : '2019-06-27'
# Fecha Inicio Bull 2020: '2020-03-17'
# Fecha TOP1 2021       : '2021-04-14'
# Fecha TOP2 2021       : '2021-11-10'
# Fecha Botom Bear 2022 : '2022-11-22'


def mostrar_configuracion():
    usdt_reserva = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100
    print("CONFIGURACIÓN DE LA ESTRATEGIA")
    print("=" * 56)
    print(f"  Saldo inicial        : ${SALDO_USDT_INICIAL:,}")
    print(f"  Período              : {FECHA_INICIO or 'inicio'}  →  {FECHA_FIN or 'fin'}")
    print(f"  RSI_LENGTH / N       : {RSI_LENGTH} / {N}")
    print(f"  STREAK  BUY / SELL   : {STREAK_COMPRAS} / {STREAK_VENTAS}")
    print(f"  R       BUY / SELL   : {R_COMPRA} / {R_VENTA}")
    print(f"  ATH_CAIDA_MAXIMA     : {ATH_CAIDA_MAXIMA}%   FACTOR_CAIDA   : {FACTOR_CAIDA}")
    print(f"  ATL_SUBIDA_MAXIMA    : {ATL_SUBIDA_MAXIMA}%  FACTOR_SUBIDA  : {FACTOR_SUBIDA}")
    print(f"  USDT reserva         : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f}")
    print(f"  BTC acumulación      : {BTC_PCT_TO_ACCUMULATE}% por venta")
    print(f"  Comisión             : {COMMISSION_PCT}%")
    print(f"  JSON salida          : {RESULTS_JSON}")
    print()
