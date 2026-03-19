"""
Configuración centralizada de la estrategia BTC Accumulation
"""
from datetime import datetime


# Parámetros de capital
SALDO_USDT_INICIAL = 1000
# ── Parámetros de la estrategia RSI ─────────────────────────
RSI_LENGTH          = 14          # Longitud del RSI
LOW_RSI_BUY_TRIGGER  = 25       # Compra si RSI(low)  <= este valor
HI_RSI_SELL_TRIGGER  = 75        # Vende si RSI(high) >= este valor

# ── Gestión de capital ───────────────────────────────────────
USDT_PCT_TO_USE = 7            # % del USDT disponible usado en cada compra  (ej: 10 = 10%)
BTC_PCT_TO_SELL = 7            # % del BTC disponible vendido en cada señal  (ej: 100 = todo)
BTC_PCT_TO_ACCUMULATE = 1  # % del BTC calculado por BTC_PCT_TO_SELL que se envía para acumular

# ── Comisión del exchange ────────────────────────────────────
COMMISSION_PCT = 0.1              # % por operación (0.1 = estándar Binance spot)


# Archivos y rutas
DB_PATH = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results.json"

# Configuración de rangos de fechas
# Formato: 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM:SS'
# Dejar None para usar desde el inicio o hasta el final
FECHA_INICIO = '2021-11-10'     # None = desde el primer dato disponible
FECHA_FIN = '2025-10-06'               # None = hasta el último dato disponible

# Ejemplos de configuración:
# Fecha Botom Bear 2018: '2018-12-10'
# Fecha Recuperación pre COVID: '2019-06-27'
# Fecha Inicio Bull 2020: '2020-03-17'
# Fecha Primer TOP 2021: '2021-04-14'
# Fecha Segundo TOP 2021: '2021-11-10'
# Fecha Botom Bear 2022: '2022-11-22'

def mostrar_configuracion():
    """Muestra la configuración actual"""
    print("CONFIGURACIÓN DE LA ESTRATEGIA")
    print("=" * 50)
    print(f"Saldo inicial     : ${SALDO_USDT_INICIAL:,}")
    print(f"Fecha inicio      : {FECHA_INICIO or 'Desde el inicio'}")
    print(f"Fecha fin         : {FECHA_FIN or 'Hasta el final'}")
    print(f"RSI buy trigger   : {LOW_RSI_BUY_TRIGGER}")
    print(f"RSI sell trigger  : {HI_RSI_SELL_TRIGGER}")
    print(f"pct_to_accumulate : {BTC_PCT_TO_ACCUMULATE}")
    print()