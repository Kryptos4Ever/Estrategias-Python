"""
Configuración centralizada de la estrategia DCA (Dollar Cost Averaging)
"""
from datetime import datetime

# Parámetros de trading
PCT_CAIDA_ATH = 0.1              # % de caída desde ATH que dispara compras
PCT_CAIDA_LAST_ATH_BUY = 0.1    # % de caída desde la última compra por ATH que dispara compras adicionales

# Parámetros de capital
SALDO_USDT_INICIAL = 1000

# Parámetros aumento progresivo del monto
MIN_COMPRA_PCT = 0.01       # % del saldo para la compra mínima
MAX_COMPRA_PCT = 0.2      # % del saldo para la máxima caída
CAIDA_MAXIMA = 0.99

# Archivos y rutas
DB_PATH = "btc_minutes.db"
RESULTS_JSON = "strategy_results.json"

# Configuración de rangos de fechas
# Formato: 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM:SS'
# Dejar None para usar desde el inicio o hasta el final
FECHA_INICIO = '2017-12-17 00:00:00'      # None = desde el primer dato disponible
FECHA_FIN = '2018-12-10 23:59:59'         # None = hasta el último dato disponible

# Ejemplos de configuración:
# Fecha Botom Bear 2018: '2018-12-10'
# Fecha Recuperación pre COVID: '2019-06-27'
# Fecha Inicio Bull 2020: '2020-03-17'
# Fecha Primer TOP 2021: '2021-04-14'
# Fecha Segundo TOP 2021: '2021-11-10'
# Fecha Botom Bear 2022: '2022-11-22'

def mostrar_configuracion():
    """Muestra la configuración actual"""
    print("CONFIGURACIÓN DE LA ESTRATEGIA DCA")
    print("=" * 50)
    print(f"Saldo inicial: ${SALDO_USDT_INICIAL:,}")
    print(f"% Caída desde ATH que dispara compras: {PCT_CAIDA_ATH*100}%")
    print(f"% Caída desde última compra ATH que dispara compras adicionales: {PCT_CAIDA_LAST_ATH_BUY*100}%")
    print(f"% Mínimo de compra: {MIN_COMPRA_PCT*100}% del saldo inicial")
    print(f"% Máximo de compra: {MAX_COMPRA_PCT*100}% del saldo inicial")
    print(f"Fecha inicio: {FECHA_INICIO or 'Desde el inicio'}")
    print(f"Fecha fin: {FECHA_FIN or 'Hasta el final'}")
    print()