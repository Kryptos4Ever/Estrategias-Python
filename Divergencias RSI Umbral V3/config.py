"""
Configuración centralizada
Estrategia: Divergencia RSI · Gradientes Logarítmicos ATH/ATL · Zona de Orden V2
BTC/USDT · Velas Horarias
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000

# ── Señal de divergencia RSI ──────────────────────────────────────────────────
RSI_LENGTH = 14       # Período del RSI
N          = 48      # Ventana de búsqueda del extremo local (velas anteriores)

# ── Umbrales RSI para divergencia (Estrategia_Divergencia_RSI_Umbral) ─────────
#
# RSI_BUY_TRIGGER  : la vela âncla de la divergencia alcista (la de precio más
#   bajo en la ventana) debe haber tenido RSI ≤ este valor.
#   → Filtra divergencias de compra que no vienen de zona de sobreventa.
#   → Valor típico: 30  (zona de sobreventa clásica)
#   → Cuanto más bajo, más restrictivo (menos señales, mayor calidad)
#
# RSI_SELL_TRIGGER : la vela âncla de la divergencia bajista (la de precio más
#   alto en la ventana) debe haber tenido RSI ≥ este valor.
#   → Filtra divergencias de venta que no vienen de zona de sobrecompra.
#   → Valor típico: 70  (zona de sobrecompra clásica)
#   → Cuanto más alto, más restrictivo (menos señales, mayor calidad)
#
RSI_BUY_TRIGGER  = 30   # umbral RSI máximo de la vela âncla de compra
RSI_SELL_TRIGGER = 70   # umbral RSI mínimo de la vela âncla de venta

# ── Profundidad de orden dentro de la zona válida (solo para V2) ──────────────
#
#   La zona válida de compra es (x_umbral, precio_ancla_low).
#   La zona válida de venta  es (precio_ancla_high, x_umbral_venta).
#
#   PROF_ZONA_PCT controla a qué profundidad dentro de esa zona se coloca
#   la orden límite, expresado como % de la amplitud total de la zona:
#
#     precio_orden_compra = precio_ancla_low  × (1 − PROF_ZONA_PCT/100)
#     precio_orden_venta  = precio_ancla_high × (1 + PROF_ZONA_PCT/100)
#
#     0.0  → orden en el borde del ancla (fill más probable, precio menos agresivo)
#    25.0  → orden al 25% de la zona (fill menos probable, precio más agresivo)
#
#   Calibrar con Analisis_Zona_Divergencia.py antes de ajustar.
#   Default conservador: 0.0
#
PROF_ZONA_PCT = 0.0     # % de la zona  (rango útil: 0–50)

# ── Gradiente logarítmico de compra — posición en el ciclo ATH→ATL ───────────
#
#   ATL_REF    = ath × FLOOR_PCT / 100              ← FIJO, no depende del ATL actual
#   log_rango  = log(100 / FLOOR_PCT)               ← CONSTANTE del ciclo
#   pos_compra = log(ATH / precio_low) / log_rango  ∈ [0, 1]
#   pct_usdt   = clamp(pos_compra, 0, 1) ^ FACTOR_CAIDA × 100
#   usdt_trade = usdt_disponible × pct_usdt / 100
#
#   log_rango es CONSTANTE: no depende del ATL dinámico sino solo del FLOOR_PCT.
#   Esto evita que la señal de compra (que actualiza ATL = lows[i]) colapse
#   el denominador a log(ATH/precio) = numerador → pos = 1.0 siempre.
#
#   FLOOR_PCT: precio mínimo esperado expresado como % del ATH.
#     → ATL_REF = ATH × FLOOR_PCT / 100  (precio piso del ciclo)
#     → log_rango = log(100 / FLOOR_PCT)  (amplitud logarítmica esperada)
#     → FLOOR_PCT=15 → ATL_REF = 15% del ATH → caída máxima esperada: 85%
#     → Si el precio cae más que lo esperado: pos > 1 → clampea a 100%
#
#   Ejemplos con ATH=$69k, ATL_REF=$10,350 (FLOOR=15%), FACTOR_CAIDA=2:
#     precio $60,000 → pos=0.059 → pct_usdt =  0.4%
#     precio $50,000 → pos=0.135 → pct_usdt =  1.8%
#     precio $40,000 → pos=0.228 → pct_usdt =  5.2%
#     precio $30,000 → pos=0.349 → pct_usdt = 12.2%
#     precio $20,000 → pos=0.520 → pct_usdt = 27.0%
#     precio $15,000 → pos=0.652 → pct_usdt = 42.5%
#     precio $10,350 → pos=1.000 → pct_usdt =100.0%
#
FLOOR_PCT    = 1     # % del ATH usado como ATL_REF mínimo  (ej: 15 → ratio 6.7x)
FACTOR_CAIDA = 2    # Curvatura  (>1 convexo · =1 lineal en log · <1 cóncavo)

# ── Gradiente logarítmico de venta — posición sobre el precio promedio ────────
#
#   ATH_PROY   = ATL_actual × (1 + PCT_ATH_PROYECTADO / 100)
#   pos_venta  = log(precio / PP) / log(ATH_PROY / PP)   ∈ [0, 1]
#   pct_btc    = clamp(pos_venta, 0, 1) ^ FACTOR_SUBIDA × 100
#   btc_trade  = btc_en_posiciones × pct_btc / 100
#
#   PP      = precio promedio de las posiciones abiertas.
#   ATH_PROY = techo dinámico calculado como % de subida esperada desde el
#              ATL registrado hasta ese momento.
#
#   La curva vende 0% en el PP, 100% en ATH_PROY.
#
#   VENTAJA vs. usar ATH histórico:
#     · En un bear market el ATH real puede estar 70-80% por encima del mercado.
#       Usar ATH histórico hace que log(ATH/PP) sea enorme → la curva queda casi
#       plana → vende muy poco incluso con buenos rebotes.
#     · ATH_PROY = ATL × (1 + PCT/100) es un techo realista y dinámico.
#       A medida que el ATL baja (el mercado encuentra nuevos mínimos), el techo
#       también baja → la estrategia ajusta su agresividad de ventas en tiempo real.
#
#   PCT_ATH_PROYECTADO = 200 → ATH_PROY = ATL × 3.0
#     Si ATL=$15,600 → ATH_PROY = $46,800 (objetivo razonable post-bear)
#   PCT_ATH_PROYECTADO = 100 → ATH_PROY = ATL × 2.0  (más conservador)
#   PCT_ATH_PROYECTADO = 300 → ATH_PROY = ATL × 4.0  (más agresivo / menos ventas)
#
#   Ejemplos con PP=$22k, ATL=$15,600, PCT=200% → ATH_PROY=$46,800, FACTOR_SUBIDA=0.5:
#     precio $25,000 (+14% vs PP) → pos=0.122 → pct_btc = 35.0%
#     precio $30,000 (+36% vs PP) → pos=0.321 → pct_btc = 56.7%
#     precio $35,000 (+59% vs PP) → pos=0.504 → pct_btc = 71.0%
#     precio $40,000 (+82% vs PP) → pos=0.675 → pct_btc = 82.2%
#     precio $46,800(+113% vs PP) → pos=1.000 → pct_btc =100.0%
#
PCT_ATH_PROYECTADO = 9000  # % de subida sobre el ATL para definir el techo de ventas
                           # rango útil: 100–400
FACTOR_SUBIDA = 1.5        # Curvatura  (>1 convexo · =1 lineal en log · <1 cóncavo)

# ── Guardia de precio promedio — COMPRA ───────────────────────────────────────
# Bloquea compras cuando precio_low ≥ PP actual (solo si ya hay posiciones).
# → Evita subir el PP comprando por encima del costo promedio.
# Nota: la guardia de VENTA está incorporada en el gradiente logarítmico.
#
GUARDIA_COMPRA = False   # True = activa | False = desactiva

# ── Guardia de precio mínimo comprado — COMPRA ────────────────────────────────
# Bloquea compras cuando precio_low ≥ precio mínimo al que se ejecutó alguna compra.
# → Asegura que cada compra sea a un precio menor que todas las anteriores.
# → Efecto: el precio promedio nunca puede subir por nuevas compras.
# Nota: sin efecto hasta que se ejecuta la primera compra.
#
GUARDIA_PRECIO_COMPRA = False  # True = activa | False = desactiva

# ── Guardia de precio máximo vendido — VENTA ──────────────────────────────────
# Bloquea ventas cuando precio_high ≤ precio máximo al que se ejecutó alguna venta.
# → Asegura que cada venta sea a un precio mayor que todas las anteriores.
# → Efecto: el sistema solo vende cuando el mercado supera su propio récord de venta.
# Nota: sin efecto hasta que se ejecuta la primera venta.
#
GUARDIA_PRECIO_VENTA  = False  # True = activa | False = desactiva

# ── Reserva mínima de USDT ────────────────────────────────────────────────────
# Piso absoluto de USDT que nunca se compromete en compras.
# Expresado como % del SALDO_USDT_INICIAL.
USDT_RESERVA_PCT = 0     # % del saldo inicial (0 = sin reserva)

# ── Acumulación de BTC ────────────────────────────────────────────────────────
# % del BTC procesado en cada venta que se desvía a btc_balance libre
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
FECHA_FIN    = '2025-10-07'

# Referencias útiles:
#   Bottom Bear 2018 : '2018-12-10'
#   Pre COVID        : '2019-06-27'
#   Inicio Bull 2020 : '2020-03-17'
#   TOP1 2021        : '2021-04-14'
#   TOP2 2021        : '2021-11-10'
#   Bottom Bear 2022 : '2022-11-22'


def mostrar_configuracion():
    usdt_reserva = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100
    print("CONFIGURACIÓN DE LA ESTRATEGIA")
    print("=" * 56)
    print(f"  Saldo inicial        : ${SALDO_USDT_INICIAL:,}")
    print(f"  Período              : {FECHA_INICIO or 'inicio'}  →  {FECHA_FIN or 'fin'}")
    print(f"  RSI_LENGTH / N       : {RSI_LENGTH} / {N}")
    print(f"  FLOOR_PCT            : {FLOOR_PCT}%  (ATL_REF = ATH × {FLOOR_PCT/100:.2f})")
    print(f"  FACTOR_CAIDA         : {FACTOR_CAIDA}   (compras  — curvatura log)")
    print(f"  PCT_ATH_PROYECTADO   : {PCT_ATH_PROYECTADO}%  (ATH_PROY = ATL × {1 + PCT_ATH_PROYECTADO/100:.1f})")
    print(f"  FACTOR_SUBIDA        : {FACTOR_SUBIDA}   (ventas   — curvatura log)")
    print(f"  Guardia compra       : {'✓ activa' if GUARDIA_COMPRA else '✗ desactivada'}")
    print(f"  Guardia precio compra: {'✓ activa' if GUARDIA_PRECIO_COMPRA else '✗ desactivada'}  (no comprar sobre mínimo comprado)")
    print(f"  Guardia precio venta : {'✓ activa' if GUARDIA_PRECIO_VENTA  else '✗ desactivada'}  (no vender bajo máximo vendido)")
    print(f"  RSI_BUY_TRIGGER      : ≤ {RSI_BUY_TRIGGER}  (umbral âncla divergencia compra)")
    print(f"  RSI_SELL_TRIGGER     : ≥ {RSI_SELL_TRIGGER}  (umbral âncla divergencia venta)")
    print(f"  PROF_ZONA_PCT        : {PROF_ZONA_PCT}%  (0% = ancla, >0% = más profundo en zona)")
    print(f"  USDT reserva         : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f}")
    print(f"  BTC acumulación      : {BTC_PCT_TO_ACCUMULATE}% por venta")
    print(f"  Comisión             : {COMMISSION_PCT}%")
    print(f"  JSON salida          : {RESULTS_JSON}")
    print()