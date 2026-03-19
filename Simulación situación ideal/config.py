# ══════════════════════════════════════════════════════════════════════════════
# SALIDA Y GRAFICADOR
# ══════════════════════════════════════════════════════════════════════════════

DARK_MODE         = True   # True = tema oscuro en el gráfico
OUTPUT_PNG        = "Gráfico_estrategia.png"
DPI               = 150

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL — BACKTEST IRREAL LOCAL BOTTOMS / LOCAL TOPS
# ══════════════════════════════════════════════════════════════════════════════

# ── Base de datos ─────────────────────────────────────────────────────────────
DB_PATH             = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"        # Ruta al archivo SQLite generado por el downloader
RESULTS_JSON = "backtest_results.json"   # archivo de salida del backtesting

# ── Rango de fechas (formato "YYYY-MM-DD") ────────────────────────────────────
FECHA_INICIO        = '2021-11-10'
FECHA_FIN           = '2022-11-22'

# Referencias útiles para testear períodos específicos:
#   Bottom Bear 2018  : '2018-12-10'
#   Pre COVID         : '2019-06-27'
#   Inicio Bull 2020  : '2020-03-17'
#   TOP1 2021         : '2021-04-14'
#   TOP2 2021         : '2021-11-10'
#   Bottom Bear 2022  : '2022-11-22'
#   TOP 2025          : '2025-10-06'

# ── Saldo inicial ─────────────────────────────────────────────────────────────
SALDO_USDT_INICIAL  = 1000.0

# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE LOCALES (oráculo perfecto)
# ══════════════════════════════════════════════════════════════════════════════
 
# Número de velas a cada lado que debe superar un mínimo/máximo local.
# Mayor valor → ciclos más grandes y menos señales.
# Rango recomendado: 5–30
VENTANA_LOCAL       = 10
 
# Precio de ejecución por tipo de operación:
#   Compra → "low"  | "close" | "open"   (irreal puro: "low")
#   Venta  → "high" | "close" | "open"   (irreal puro: "high")
PRECIO_COMPRA       = "low"
PRECIO_VENTA        = "high"
 
# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE GESTIÓN DE CAPITAL
# ══════════════════════════════════════════════════════════════════════════════
 
# ── Slots de posición ─────────────────────────────────────────────────────────
# Define cuántas posiciones simultáneas puede haber en cartera.
#
# El capital se divide en N slots iguales SÓLO cuando las posiciones abiertas
# llegan a 0 (todo el BTC vendido). Ese monto por slot queda fijo hasta el
# próximo reinicio a cero posiciones.
#
#   slot_usdt = usdt_balance / MAX_POSICIONES   (solo cuando positions == 0)
#
# Cada BUY consume exactamente 1 slot de USDT.
# Cada SELL vende  btc_en_posiciones / positions_count  BTC,
#   recalculado luego de cada compra y congelado entre ventas.
#
# Ejemplo: saldo $1 000, MAX_POSICIONES = 10 → slot = $100 por compra.
# Rango recomendado: 4–15
MAX_POSICIONES      = 5
 
# ── Comisión de exchange ──────────────────────────────────────────────────────
# Porcentaje sobre el monto operado (Binance spot maker/taker: 0.1)
COMMISSION_PCT      = 0.1     # 0.10 %
 