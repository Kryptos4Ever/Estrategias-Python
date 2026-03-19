# ============================================================
#  Configuracion - Estrategia Grid ATH/ATL
#  BTC/USDT Spot | Backtesting
#  Todos los porcentajes son valores directos (50.0 = 50%)
# ============================================================

DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results.json"

SALDO_USDT_INICIAL = 1000.0

# Rango de fechas (None = todos los datos)
# Fecha Botom Bear 2018 : '2018-12-10'
# Fecha pre COVID       : '2019-06-27'
# Fecha Inicio Bull 2020: '2020-03-17'
# Fecha TOP1 2021       : '2021-04-14'
# Fecha TOP2 2021       : '2021-11-10'
# Fecha Botom Bear 2022 : '2022-11-22'
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2025-10-06'

# ── COMPRAS ───────────────────────────────────────────────────
PASO_PCT_COMPRA = 3.0    # % entre niveles (escala logaritmica)
CAIDA_MAXIMA    = 77.47   # % maximo de caida desde ATH

FACTOR_COMPRA   = 140.0    # relacion entre el monto de la ultima y la primera orden
                          # 1.0 = todas las ordenes iguales
                          # 4.0 = la ultima orden compra 4x mas que la primera
                          # el ratio r se deriva automaticamente segun n de niveles

# ── VENTAS ────────────────────────────────────────────────────
PASO_PCT_VENTA  = 5.0    # % entre niveles (escala logaritmica)
SUBIDA_MAXIMA   = 715.45  # % maximo de subida desde precio promedio de posiciones

FACTOR_VENTA    = 1500.0    # mismo criterio que FACTOR_COMPRA pero para ventas

# ── ACUMULACION DE BTC ────────────────────────────────────────
MIN_ACUMULAR_PCT = 0.0   # % de cada venta que se acumula como BTC libre
MAX_ACUMULAR_PCT = 0.0   # (0.0 = vender todo, sin acumulacion)

# ── COMISION ──────────────────────────────────────────────────
COMMISSION_PCT  = 0.1