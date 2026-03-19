"""
Configuración centralizada
Estrategia: Divergencia RSI · Gradiente Asintótico · ATH_TEO dinámico
BTC/USDT · Velas Horarias
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000

# ── Señal de divergencia RSI ──────────────────────────────────────────────────
RSI_LENGTH = 5
N          = 10

# ── Umbrales RSI para divergencia ─────────────────────────────────────────────
RSI_BUY_TRIGGER  = 10
RSI_SELL_TRIGGER = 70

# ── FLOOR_PCT — referencia del lado de COMPRA ─────────────────────────────────
#
#   Define el piso teórico del ciclo bajista como fracción del ATH.
#
#   COMPRA  →  ATL_REF = ATH × FLOOR_PCT / 100
#              log_rango = log(100 / FLOOR_PCT)
#              pos_compra = 0  cuando precio = ATH
#              pos_compra = 1  cuando precio = ATL_REF
#
#   ATL_REF se recalcula con cada nuevo ATH que se registra.
#
#   Ejemplo con FLOOR_PCT=25:
#     ATH=$69k → ATL_REF = $17,250   (caída esperada del 75%)
#
FLOOR_PCT = 25      # % del ATH usado como piso teórico de compra

# ── TOP_PCT — referencia del lado de VENTA ────────────────────────────────────
#
#   Define el techo teórico del ciclo alcista como múltiplo del ATL.
#   Es independiente de FLOOR_PCT, lo que permite calibrar cada lado
#   del ciclo por separado.
#
#   VENTA   →  ATH_TEO = ATL × TOP_PCT / 100
#              log_amp  = log(ATH_TEO / PP)
#              pos_venta = 0  cuando precio = PP  (precio promedio posiciones)
#              pos_venta = 1  cuando precio = ATH_TEO
#
#   ATH_TEO se recalcula con cada nuevo ATL que se registra.
#
#   Ejemplos:
#     TOP_PCT=400  →  ATH_TEO = ATL × 4.0   (subida esperada del 300%)
#     TOP_PCT=750  →  ATH_TEO = ATL × 7.5   (subida esperada del 650%)
#     TOP_PCT=1000 →  ATH_TEO = ATL × 10.0  (subida esperada del 900%)
#
#   Con ATL=$15k:
#     TOP_PCT=400  →  ATH_TEO = $60,000
#     TOP_PCT=750  →  ATH_TEO = $112,500
#     TOP_PCT=1000 →  ATH_TEO = $150,000
#
TOP_PCT = 750       # % del ATL usado como techo teórico de venta  (ATH_TEO = ATL × TOP_PCT/100)

# ── Curva asintótica de COMPRA ────────────────────────────────────────────────
#
#   La posición normalizada (pos ∈ [0,1]) se transforma en % de capital
#   mediante una función de dos fases:
#
#   Fase 1  [0, COMPRA_INFL]:
#     Exponencial que satura en COMPRA_NIVEL%.
#     Captura retrocesos iniciales del bear market con una fracción
#     acotada del capital. La asíntota impide gastar todo el saldo
#     antes de llegar al verdadero fondo.
#
#   Fase 2  [COMPRA_INFL, 1]:
#     Power con curvatura COMPRA_FAC2 que acelera desde COMPRA_NIVEL%
#     hasta 100% al llegar a ATL_REF.
#     Concentra el grueso del capital en los precios más bajos del ciclo.
#
#   Parámetros:
#     COMPRA_NIVEL : % máximo de capital en la fase 1  (la "asíntota")
#     COMPRA_INFL  : pos donde termina la fase 1 y empieza la aceleración
#     COMPRA_K     : velocidad de saturación exponencial (mayor = más abrupto)
#     COMPRA_FAC2  : curvatura de la fase 2  (>1 = convexa, más agresiva al final)
#
CURVA_COMPRA_NIVEL = 5      # % del capital disponible — techo fase 1
CURVA_COMPRA_INFL  = 0.30   # posición de inflexión  (0.30 = 30% del recorrido ATH→ATL_REF)
CURVA_COMPRA_K     = 2      # velocidad de la exp     (rango útil: 2–12)
CURVA_COMPRA_FAC2  = 5      # curvatura aceleración   (rango útil: 1.0–5.0)

# ── Curva asintótica de VENTA ─────────────────────────────────────────────────
#
#   Misma estructura de dos fases, pero sobre el eje de ventas:
#     pos = 0 cuando precio = PP  (precio promedio posiciones)
#     pos = 1 cuando precio = ATH_TEO = ATL × TOP_PCT / 100
#
#   Fase 1  [0, VENTA_INFL]:
#     Cosecha temprana acotada: vende hasta VENTA_NIVEL% del BTC
#     en los primeros rebotes por encima del PP. Asegura liquidez
#     y realización parcial de ganancias en rallies intermedios.
#
#   Fase 2  [VENTA_INFL, 1]:
#     Distribución agresiva concentrada hacia el ATH_TEO.
#     Reserva el grueso del BTC para la fase final del bull market.
#
#   Nota: pos_venta = 0 cuando precio ≤ PP → guardia de venta
#   incorporada matemáticamente.
#
CURVA_VENTA_NIVEL = 20      # % del BTC en posiciones — techo fase 1
CURVA_VENTA_INFL  = 0.05    # posición de inflexión  (0.05 = 5% del recorrido PP→ATH_TEO)
CURVA_VENTA_K     = 8       # velocidad de la exp     (rango útil: 2–12)
CURVA_VENTA_FAC2  = 1       # curvatura aceleración   (rango útil: 1.0–5.0)

# ── Guardia de precio promedio — COMPRA ───────────────────────────────────────
GUARDIA_COMPRA        = True
GUARDIA_PRECIO_COMPRA = True
GUARDIA_PRECIO_VENTA  = False

# ── Reserva de capital ────────────────────────────────────────────────────────
USDT_RESERVA_PCT      = 0
BTC_PCT_TO_ACCUMULATE = 0

# ── Comisión ──────────────────────────────────────────────────────────────────
COMMISSION_PCT = 0.1

# ── Rutas ─────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results_div_rsi.json"

# ── Rango de fechas ───────────────────────────────────────────────────────────
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2022-11-22'


def mostrar_configuracion():
    usdt_reserva = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100
    import math
    atl_ref_ejemplo = f"ATH × {FLOOR_PCT/100:.2f}  (caída esperada del {100 - FLOOR_PCT}%)"
    ath_teo_ejemplo = f"ATL × {TOP_PCT/100:.2f}  (subida esperada del {TOP_PCT - 100}%)"
    print("CONFIGURACIÓN DE LA ESTRATEGIA")
    print("=" * 66)
    print(f"  Saldo inicial          : ${SALDO_USDT_INICIAL:,}")
    print(f"  Período                : {FECHA_INICIO or 'inicio'}  →  {FECHA_FIN or 'fin'}")
    print(f"  RSI_LENGTH / N         : {RSI_LENGTH} / {N}")
    print(f"  ── Referencia de ciclo ───────────────────────────────────")
    print(f"  FLOOR_PCT  (compra)    : {FLOOR_PCT}%  →  ATL_REF = {atl_ref_ejemplo}")
    print(f"  TOP_PCT    (venta)     : {TOP_PCT}%  →  ATH_TEO = {ath_teo_ejemplo}")
    print(f"  ── Curva compra ──────────────────────────────────────────")
    print(f"  COMPRA nivel/infl/k/f2 : {CURVA_COMPRA_NIVEL}% / {CURVA_COMPRA_INFL} / {CURVA_COMPRA_K} / {CURVA_COMPRA_FAC2}")
    print(f"  ── Curva venta ───────────────────────────────────────────")
    print(f"  VENTA  nivel/infl/k/f2 : {CURVA_VENTA_NIVEL}% / {CURVA_VENTA_INFL} / {CURVA_VENTA_K} / {CURVA_VENTA_FAC2}")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Guardia compra         : {'✓ activa' if GUARDIA_COMPRA else '✗ desactivada'}")
    print(f"  Guardia precio compra  : {'✓ activa' if GUARDIA_PRECIO_COMPRA else '✗ desactivada'}")
    print(f"  Guardia precio venta   : {'✓ activa' if GUARDIA_PRECIO_VENTA  else '✗ desactivada'}")
    print(f"  RSI_BUY_TRIGGER        : ≤ {RSI_BUY_TRIGGER}")
    print(f"  RSI_SELL_TRIGGER       : ≥ {RSI_SELL_TRIGGER}")
    print(f"  USDT reserva           : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f}")
    print(f"  BTC acumulación        : {BTC_PCT_TO_ACCUMULATE}% por venta")
    print(f"  Comisión               : {COMMISSION_PCT}%")
    print(f"  JSON salida            : {RESULTS_JSON}")
    print()
