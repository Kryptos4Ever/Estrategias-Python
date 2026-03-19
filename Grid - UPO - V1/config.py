"""
Configuración centralizada
Estrategia: Órdenes Límite por Precio · Progresión Lineal de Compra
BTC/USDT · Velas Horarias
"""

# ── Capital inicial ───────────────────────────────────────────────────────────
SALDO_USDT_INICIAL = 1000

# ── Señal de precio — COMPRA ──────────────────────────────────────────────────
#
#   Modo A (sin posiciones abiertas):
#     limit_buy = ATH × (1 − PCT_CAIDA_ATH)
#     Se ejecuta si low ≤ limit_buy
#
#   Modo B (con posiciones abiertas):
#     limit_buy = last_op_price × (1 − PCT_CAIDA)
#     Se ejecuta si low ≤ limit_buy
#
PCT_CAIDA_ATH = 0.4    # % de caída desde ATH para la primera compra (Modo A)
PCT_CAIDA     = 0.02    # % de caída desde last_op_price para DCA      (Modo B)

# ── Señal de precio — VENTA ───────────────────────────────────────────────────
#
#   Cada posición tiene su TP individual:
#     precio_tp = limit_buy_de_esa_posicion × (1 + PCT_VENTA)
#   Se ejecuta si high ≥ precio_tp.
#
#   BTC vendido = mínimo para recuperar usdt_invertido en esa posición
#                 + cubrir comisión de compra y de venta.
#   El resto del BTC de la posición se acumula como BTC libre.
#
PCT_VENTA = 0.06        # % de subida desde precio de entrada → TP de venta

# ── FLOOR_PCT — referencia del lado de COMPRA ─────────────────────────────────
#
#   Define el piso teórico del ciclo bajista como fracción del ATH.
#   Se usa para normalizar la posición dentro del rango de compra:
#
#     ATL_REF   = ATH × FLOOR_PCT / 100
#     pos       = log(ATH / precio) / log(100 / FLOOR_PCT)   ∈ [0, 1]
#
#   pos = 0 cuando precio = ATH      →  gradiente devuelve 0%
#   pos = 1 cuando precio = ATL_REF  →  gradiente devuelve PENDIENTE% (cap 100%)
#
FLOOR_PCT = 25          # % del ATH usado como piso teórico de compra

# ── Progresión lineal de COMPRA ───────────────────────────────────────────────
#
#   pct_capital = min(pos × PENDIENTE_COMPRA, 100)
#
#   PENDIENTE_COMPRA controla la velocidad de asignación de capital:
#     100  →  línea recta: 0% en ATH, 100% al llegar al piso (FLOOR)
#      50  →  más suave:   0% en ATH,  50% en el piso (nunca llega a 100%)
#     200  →  más agresiva: satura al 100% a mitad del camino al piso
#     300  →  muy agresiva: satura al 100% al primer tercio del camino
#
PENDIENTE_COMPRA = 96  # pendiente de la progresión lineal (rango útil: 50–300)

# ── Comportamiento de last_op_price en ventas ─────────────────────────────────
#
#   False (recomendado): last_op_price SOLO se actualiza en compras.
#     La cadena DCA desciende siempre desde el último precio de compra.
#
#   True:  last_op_price también se actualiza al ejecutar una venta (= precio_tp).
#     El próximo Modo B se ancla en el TP de la última venta, lo que
#     puede generar compras en niveles más altos de lo esperado cuando
#     PCT_VENTA > PCT_CAIDA.
#
LAST_OP_UPDATE_ON_SELL = True

# ── Reserva de capital ────────────────────────────────────────────────────────
USDT_RESERVA_PCT = 0        # % del saldo inicial que nunca se opera

# ── Comisión ──────────────────────────────────────────────────────────────────
COMMISSION_PCT = 0.1        # % sobre el monto de cada operación

# ── Rutas ─────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Users\Bernardo\Documents\CRYPTO\Estrategias de trading automatizado\DB\btc_hourly.db"
RESULTS_JSON = "strategy_results_precio.json"

# ── Rango de fechas ───────────────────────────────────────────────────────────
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2024-03-04'

# Referencias útiles:
#   Bottom Bear 2018 : '2018-12-10'
#   Pre COVID        : '2019-06-27'
#   Inicio Bull 2020 : '2020-03-17'
#   TOP1 2021        : '2021-04-14'
#   TOP2 2021        : '2021-11-10'
#   Bottom Bear 2022 : '2022-11-22'
#   Recuperación ATH : '2024-03-04'

def mostrar_configuracion():
    usdt_reserva = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100
    import math
    atl_ref_ejemplo = f"ATH × {FLOOR_PCT/100:.2f}  (caída esperada del {100 - FLOOR_PCT}%)"
    print("CONFIGURACIÓN DE LA ESTRATEGIA")
    print("=" * 66)
    print(f"  Saldo inicial          : ${SALDO_USDT_INICIAL:,}")
    print(f"  Período                : {FECHA_INICIO or 'inicio'}  →  {FECHA_FIN or 'fin'}")
    print(f"  ── Señales de precio ─────────────────────────────────────")
    print(f"  PCT_CAIDA_ATH          : {PCT_CAIDA_ATH*100:.2f}%  (primera compra desde ATH)")
    print(f"  PCT_CAIDA              : {PCT_CAIDA*100:.2f}%  (DCA desde last_op_price)")
    print(f"  PCT_VENTA              : {PCT_VENTA*100:.2f}%  (TP sobre precio de entrada)")
    print(f"  ── Referencia de ciclo ───────────────────────────────────")
    print(f"  FLOOR_PCT  (compra)    : {FLOOR_PCT}%  →  ATL_REF = {atl_ref_ejemplo}")
    print(f"  ── Progresión lineal de compra ───────────────────────────")
    print(f"  PENDIENTE_COMPRA       : {PENDIENTE_COMPRA}")
    print(f"  LAST_OP_UPDATE_ON_SELL : {LAST_OP_UPDATE_ON_SELL}")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  USDT reserva           : {USDT_RESERVA_PCT}% → ${usdt_reserva:,.2f}")
    print(f"  Comisión               : {COMMISSION_PCT}%")
    print(f"  JSON salida            : {RESULTS_JSON}")
    print()
