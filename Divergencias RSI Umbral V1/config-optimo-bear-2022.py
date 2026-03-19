"""
Configuración centralizada
Estrategia: Divergencia RSI · Gradientes Logarítmicos ATH/ATL
BTC/USDT · Velas Horarias
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000

# ── Señal de divergencia RSI ──────────────────────────────────────────────────
RSI_LENGTH = 5       # Período del RSI
N          = 15      # Ventana de búsqueda del extremo local (velas anteriores)

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
RSI_BUY_TRIGGER  = 10   # umbral RSI máximo de la vela âncla de compra
RSI_SELL_TRIGGER = 60   # umbral RSI mínimo de la vela âncla de venta

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
FLOOR_PCT    = 25     # % del ATH usado como ATL_REF mínimo  (ej: 15 → ratio 6.7x)
FACTOR_CAIDA = 4.0    # Curvatura  (>1 convexo · =1 lineal en log · <1 cóncavo)

# ── Gradiente logarítmico de venta — posición sobre el precio promedio ────────
#
#   pos_venta  = log(precio_high / PP) / log(ATH / PP)   ∈ [0, 1]
#   pct_btc    = clamp(pos_venta, 0, 1) ^ FACTOR_SUBIDA × 100
#   btc_trade  = btc_en_posiciones × pct_btc / 100
#
#   PP = precio promedio de las posiciones abiertas.
#   La curva se ancla al PP: vende 0% en el PP, 100% en el ATH.
#   El denominador log(ATH/PP) es el espacio de ganancia disponible.
#
#   Propiedad clave: la guardia de venta está INCORPORADA matemáticamente.
#   Cuando precio ≤ PP → pos_venta ≤ 0 → vende 0% sin parámetro extra.
#
#   El gradiente se AUTOAJUSTA con cada compra:
#     · Compramos más abajo → PP baja → log(ATH/PP) crece → curva más sensible
#       (empieza a vender antes y en mayor volumen para cada precio dado)
#     · Compramos arriba   → PP sube → curva más restrictiva
#
#   Ejemplos con PP=$22k, ATH=$69k, FACTOR_SUBIDA=1.5:
#     precio $25,000 (+14%) → pos=0.112 → pct_btc =  3.7%
#     precio $30,000 (+36%) → pos=0.301 → pct_btc = 16.5%
#     precio $35,000 (+59%) → pos=0.462 → pct_btc = 31.4%
#     precio $40,000 (+82%) → pos=0.594 → pct_btc = 45.8%
#     precio $50,000(+127%) → pos=0.800 → pct_btc = 71.6%
#     precio $69,000(+214%) → pos=1.000 → pct_btc =100.0%
#
FACTOR_SUBIDA = 0.5   # Curvatura  (>1 convexo · =1 lineal en log · <1 cóncavo)

# ── Guardia de precio promedio — COMPRA ───────────────────────────────────────
# Bloquea compras cuando precio_low ≥ PP actual (solo si ya hay posiciones).
# → Evita subir el PP comprando por encima del costo promedio.
# Nota: la guardia de VENTA está incorporada en el gradiente logarítmico.
#
GUARDIA_COMPRA = True   # True = activa | False = desactiva

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
FECHA_FIN    = '2022-11-22'

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
    print(f"  FACTOR_SUBIDA        : {FACTOR_SUBIDA}   (ventas   — curvatura log)")
    print(f"  Guardia compra       : {'✓ activa' if GUARDIA_COMPRA else '✗ desactivada'}")
    print(f"  Guardia precio compra: {'✓ activa' if GUARDIA_PRECIO_COMPRA else '✗ desactivada'}  (no comprar sobre mínimo comprado)")
    print(f"  Guardia precio venta : {'✓ activa' if GUARDIA_PRECIO_VENTA  else '✗ desactivada'}  (no vender bajo máximo vendido)")
    print(f"  RSI_BUY_TRIGGER      : ≤ {RSI_BUY_TRIGGER}  (umbral âncla divergencia compra)")
    print(f"  RSI_SELL_TRIGGER     : ≥ {RSI_SELL_TRIGGER}  (umbral âncla divergencia venta)")
    print(f"  USDT reserva         : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f}")
    print(f"  BTC acumulación      : {BTC_PCT_TO_ACCUMULATE}% por venta")
    print(f"  Comisión             : {COMMISSION_PCT}%")
    print(f"  JSON salida          : {RESULTS_JSON}")
    print()