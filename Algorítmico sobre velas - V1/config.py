"""
config.py — Configuración del Backtester de Señal Compuesta BTC/USDT
══════════════════════════════════════════════════════════════════════
Editar este archivo para ajustar parámetros y ejecutar optimizaciones.
Luego correr:  python backtest_compuesto.py
El resultado JSON se pasa al Graficador_v2.py para visualizar.
"""
# ══════════════════════════════════════════════════════════════════════════════
# SALIDA Y GRAFICADOR
# ══════════════════════════════════════════════════════════════════════════════

DARK_MODE         = True   # True = tema oscuro en el gráfico
OUTPUT_PNG        = "Gráfico_estrategia.png"
DPI               = 150

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL — BACKTEST IRREAL LOCAL BOTTOMS / LOCAL TOPS
# ══════════════════════════════════════════════════════════════════════════════

# ── Comisión del exchange ──────────────────────────────────────────────────────
COMMISSION_PCT = 0.1          # % comisión por operación (Binance Spot = 0.1)


# ── Rutas ──────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "backtest_compuesto_results.json"   # archivo de salida del backtesting


# ── Rango de fechas ────────────────────────────────────────────────────────────
# Formato: 'YYYY-MM-DD'  |  None = desde el inicio / hasta el final del dataset
FECHA_INICIO = '2025-10-06'
FECHA_FIN    = None

# Referencias útiles para testear períodos específicos:
#   Bottom Bear 2018  : '2018-12-10'
#   Pre COVID         : '2019-06-27'
#   Inicio Bull 2020  : '2020-03-17'
#   TOP1 2021         : '2021-04-14'
#   TOP2 2021         : '2021-11-10'
#   Bottom Bear 2022  : '2022-11-22'
#   Inicio Bull 2023  : '2023-01-01'
#   Inicio Bull 2024  : '2024-01-01'


# ── Capital inicial ────────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA SEÑAL COMPUESTA
# ══════════════════════════════════════════════════════════════════════════════

# ── Umbrales de activación (0–100) ────────────────────────────────────────────
# Mayor umbral → menos señales, mayor precisión, menor recall
# Recomendado: 65–85
THR_BOT = 65.0        # umbral de score para abrir posición (señal BOTTOM)
THR_TOP = 65.0        # umbral de score para cerrar posición (señal TOP)

# ── Cooldown entre señales del mismo tipo (en horas/velas) ────────────────────
# Evita abrir múltiples posiciones en el mismo evento de reversal
# Recomendado: 8–24
COOLDOWN_VELAS = 24

# ── Ventana del score adaptativo (en velas = horas) ───────────────────────────
# El score se normaliza contra esta ventana histórica deslizante
# Más corta → más reactivo; más larga → más estable
# Recomendado: 300–700
VENTANA_SCORE = 500

# ── Suavizado del score (media móvil en horas) ────────────────────────────────
# Reduce falsas señales por ruido de velas individuales
# Recomendado: 2–6
SUAVIZADO_SCORE = 6


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE GESTIÓN DE CAPITAL
# ══════════════════════════════════════════════════════════════════════════════

# ── Tamaño de posición ────────────────────────────────────────────────────────
# % del USDT disponible a invertir por cada señal de compra
# Con sizing adaptativo, se escala entre 75%–100% de este valor según intensidad
# Recomendado: 0.15–0.35
PCT_USDT_POR_SEÑAL = 0.30     # ej: 0.20 = 20% del USDT libre por señal

# % del BTC total en posiciones a vender por cada señal de venta
# Recomendado: 0.25–0.60
PCT_BTC_POR_SEÑAL  = 0.55     # ej: 0.35 = 35% del BTC en posiciones por señal

# ── Límite de exposición simultánea ──────────────────────────────────────────
# Máximo de posiciones abiertas al mismo tiempo
# Recomendado: 4–10
MAX_POSICIONES = 10

# ── Reserva de capital ────────────────────────────────────────────────────────
# % del capital inicial que siempre se mantiene en USDT (no se invierte)
# Protege contra quedarse sin liquidez en caídas extremas
# Recomendado: 0.03–0.10
USDT_RESERVA_PCT = 0.05       # ej: 0.05 = 5% del capital inicial

# ── Sizing adaptativo ─────────────────────────────────────────────────────────
# Si True: el tamaño de la orden escala con la intensidad del score
#   score=75 → usa 75% del PCT_USDT_POR_SEÑAL
#   score=100 → usa 100% del PCT_USDT_POR_SEÑAL
# Si False: tamaño fijo igual a PCT_USDT_POR_SEÑAL
SIZING_ADAPTATIVO = False

# ── Orden de liquidación de posiciones (SELL) ─────────────────────────────────
# 'fifo'        → primero las posiciones más antiguas
# 'lifo'        → primero las posiciones más recientes
# 'mejor_pnl'   → primero las que tienen mayor ganancia no realizada
# 'peor_pnl'    → primero las que tienen mayor pérdida (stop-loss suave)
ORDEN_LIQUIDACION = 'mejor_pnl'


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL DNA DE VELAS (Feature Engineering)
# ══════════════════════════════════════════════════════════════════════════════

# Ventana para normalizar range relativo y trade density
VENTANA_DNA = 72      # horas (recomendado: 24–72)

# Clip de outliers en features DNA
CLIP_RANGE_REL   = 5.0   # máximo valor de range_rel (múltiplos del promedio)
CLIP_TRADE_DENS  = 5.0   # máximo valor de trade_density


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL MÉTODO ③ — ESPACIO DE FASE (Lyapunov + HFD)
# ══════════════════════════════════════════════════════════════════════════════

# Delay embedding (Teorema de Takens)
TAU_EMBEDDING = 4     # delay en horas (recomendado: 2–8)
DIM_EMBEDDING = 5     # dimensión del embedding (recomendado: 4–7)

# Lyapunov local
W_LYAPUNOV    = 8     # ventana de divergencia en horas (recomendado: 4–16)
K_VECINOS     = 5     # vecinos más cercanos para Lyapunov (recomendado: 3–8)

# Dimensión fractal de Higuchi
WIN_HFD       = 64    # ventana para HFD (recomendado: 32–128)
KMAX_HFD      = 8     # parámetro kmax de Higuchi (recomendado: 6–12)

# Ventana de normalización (percentil rolling) para Lyapunov
WIN_LYAP_NORM = 500   # recomendado: 300–700


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL MÉTODO ④ — ENTROPÍA DE PERMUTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Orden de permutación (número de elementos comparados)
PE_ORDER      = 4     # recomendado: 3–5 (4 es el sweet spot)
PE_DELAY      = 1     # delay entre elementos del patrón
PE_VENTANA    = 64    # ventana para calcular distribución de patrones

# Pesos de los 4 canales para el PE compuesto
PE_PESO_CLOSE   = 0.40
PE_PESO_DELTA   = 0.30
PE_PESO_LWK     = 0.15
PE_PESO_TRADE   = 0.15

# Ventana de normalización para PE tensión
WIN_PE_NORM   = 500


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL RANDOM FOREST (Detector morfológico)
# ══════════════════════════════════════════════════════════════════════════════

RF_N_ESTIMATORS   = 300
RF_MAX_DEPTH      = 12
RF_MIN_SAMPLES    = 10
RF_N_SPLITS_CV    = 5    # folds para TimeSeriesSplit

# Labeling de local tops/bottoms (multi-escala)
LABEL_ORDERS      = [6, 12, 24, 48, 96]   # ventanas en horas
LABEL_MIN_SWING   = 0.015                  # swing mínimo = 1.5%

# Ratio de submuestreo de neutros para balancear el dataset
NEUTROS_RATIO     = 3    # neutros = señales × NEUTROS_RATIO


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA REGRESIÓN LOGÍSTICA (pesos del score compuesto)
# ══════════════════════════════════════════════════════════════════════════════

LR_C              = 1.0   # regularización (mayor C = menos regularización)
LR_N_SPLITS_CV    = 5     # folds para optimización de pesos



