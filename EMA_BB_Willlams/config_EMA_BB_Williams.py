# ============================================================
#  Configuración — Estrategia EMA200 + Bollinger + Williams
#  BTC/USDT Spot | Backtesting
# ============================================================

# ── Base de datos ────────────────────────────────────────────
DB_PATH      = "btc_minutes.db"

# ── Resultados ───────────────────────────────────────────────
RESULTS_JSON = "strategy_results.json"

# ── Capital inicial ──────────────────────────────────────────
SALDO_USDT_INICIAL = 1000.0

# ── Rango de fechas ──────────────────────────────────────────
#    Formato: "YYYY-MM-DD HH:MM:SS"  |  None = todos los datos
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2025-10-06'

# Ejemplos de configuración:
# Fecha Botom Bear 2018: '2018-12-10'
# Fecha Recuperación pre COVID: '2019-06-27'
# Fecha Inicio Bull 2020: '2020-03-17'
# Fecha Primer TOP 2021: '2021-04-14'
# Fecha Segundo TOP 2021: '2021-11-10'
# Fecha Botom Bear 2022: '2022-11-22'

# ── EMA200 ───────────────────────────────────────────────────
EMA_LENGTH     = 200
DIST_EMA_BUY   = 1.52    # compra si precio está >= X% BAJO la EMA200  (positivo = bajo)
DIST_EMA_SELL  = 1.51    # vende si precio está >= X% SOBRE la EMA200  (positivo = sobre)

# ── Bollinger Bands ──────────────────────────────────────────
BB_LENGTH      = 20
BB_STD         = 2.0
BB_BUY         = 0   # compra si %B <= este valor  (0 = banda inf, 1 = banda sup)
BB_SELL        = 1   # vende si %B >= este valor

# ── Williams %R ──────────────────────────────────────────────
WILLIAMS_LENGTH = 14
WILLIAMS_BUY   = -90    # compra si Williams %R <= este valor  (sobreventa)
WILLIAMS_SELL  = -10    # vende si Williams %R >= este valor   (sobrecompra)

# ── Gestión de capital ───────────────────────────────────────
USDT_PCT_TO_USE      = 5.0   # % del usdt_balance usado en cada compra
BTC_PCT_TO_SELL      = 5.0   # % del btc_en_posiciones procesado en cada venta
BTC_PCT_TO_ACCUMULATE = 1  # % del BTC procesado que se acumula (no se vende)
 
# ── Comisión ─────────────────────────────────────────────────
COMMISSION_PCT = 0.1

# ── Referencia para otros módulos ────────────────────────────
#    (usado por Analizar_RSI y Analizar_Rachas_RSI si se ejecutan)
RSI_LENGTH           = 14
LOW_RSI_BUY_TRIGGER  = 30
HI_RSI_SELL_TRIGGER  = 70
