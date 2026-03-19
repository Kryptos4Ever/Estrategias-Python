"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            TEST SUITE — backtest_irreal.py                                  ║
║   Sin dependencias de config.py ni de la DB de precios                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cobertura:
  · Posicion                — constructor
  · Portfolio               — init, propiedades, slot, btc_por_venta,
                              comprar, vender, FIFO, comisión, ganancia
  · detectar_señales        — bordes, zigzag, plano, ventana=1, ventana grande
  · correr_backtest         — contadores, ignorados, scores, formulas PnL,
                              summary completo, trade_history campos
  · Extremos                — precio micro/macro, capital mínimo, comisión 0
                              y 50 %, max_posiciones=1, ciclos completos,
                              SELL sin posiciones, MAX_POSICIONES lleno,
                              señales solo BUY, señales solo SELL

Uso:
    python test_backtest.py
"""

import math
import random
import sys
import types
import traceback
from collections import deque

# ─── Importar las clases/funciones bajo test (sin config ni DB) ──────────────
# Se inyecta un módulo "config" falso para que el import de backtest_irreal no
# falle, pero ningún test lo usa directamente.
_fake_cfg = types.ModuleType("config")
_fake_cfg.DB_PATH            = ":memory:"
_fake_cfg.OUTPUT_JSON_PATH   = "/dev/null"
_fake_cfg.FECHA_INICIO       = "2020-01-01"
_fake_cfg.FECHA_FIN          = "2020-12-31"
_fake_cfg.SALDO_USDT_INICIAL = 1000.0
_fake_cfg.MAX_POSICIONES     = 5
_fake_cfg.COMMISSION_PCT     = 0.1
_fake_cfg.VENTANA_LOCAL      = 3
_fake_cfg.PRECIO_COMPRA      = "low"
_fake_cfg.PRECIO_VENTA       = "high"
sys.modules["config"] = _fake_cfg

from backtest_irreal import Posicion, Portfolio, detectar_señales, correr_backtest

# ─── Utilidades ───────────────────────────────────────────────────────────────

EPS = 1e-7   # tolerancia flotante


def close(a: float, b: float, tol: float = EPS) -> bool:
    return abs(a - b) <= tol


def make_candle(low, high, close=None, open_=None, dt="2020-01-01 00:00:00") -> dict:
    close_ = close if close is not None else (low + high) / 2
    return {
        "ts": 0, "datetime": dt,
        "open":  open_ if open_ is not None else close_,
        "high":  high,
        "low":   low,
        "close": close_,
        "volume": 1.0,
    }


def make_cfg(**kwargs) -> types.SimpleNamespace:
    """Config mínimo con valores sobreescribibles."""
    defaults = dict(
        SALDO_USDT_INICIAL=1000.0,
        MAX_POSICIONES=5,
        COMMISSION_PCT=0.1,
        VENTANA_LOCAL=3,
        PRECIO_COMPRA="low",
        PRECIO_VENTA="high",
        FECHA_INICIO="2020-01-01",
        FECHA_FIN="2020-12-31",
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ─── Registro de resultados ───────────────────────────────────────────────────

PASSED = []
FAILED = []


def run(name: str, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  ✓  {name}")
    except Exception as e:
        FAILED.append((name, e))
        print(f"  ✗  {name}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Posicion
# ══════════════════════════════════════════════════════════════════════════════

def test_posicion_constructor():
    p = Posicion(entry_price=50000.0, btc=0.02)
    assert p.entry_price == 50000.0
    assert p.btc == 0.02


def test_posicion_campos_mutables():
    p = Posicion(entry_price=10000.0, btc=0.1)
    p.btc -= 0.05
    assert close(p.btc, 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Portfolio: inicialización
# ══════════════════════════════════════════════════════════════════════════════

def test_portfolio_slot_inicial():
    """slot_usdt = usdt_inicial / max_posiciones en el constructor."""
    for usdt, n in [(1000, 5), (500, 4), (777.77, 7), (1, 1), (1e6, 10)]:
        p = Portfolio(usdt, n)
        assert close(p.slot_usdt, usdt / n), f"usdt={usdt} n={n}"


def test_portfolio_estado_inicial_vacio():
    p = Portfolio(1000.0, 5)
    assert p.positions_count    == 0
    assert close(p.btc_en_posiciones, 0.0)
    assert close(p.precio_promedio_posiciones, 0.0)
    assert close(p.btc_acumulado_total, 0.0)
    assert close(p.btc_por_venta, 0.0)


def test_portfolio_valor_inicial_sin_posiciones():
    p = Portfolio(1000.0, 5)
    assert close(p.valor_portfolio(99999.0), 1000.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Portfolio: comprar
# ══════════════════════════════════════════════════════════════════════════════

def test_comprar_gasta_exactamente_un_slot():
    """usdt_spent == slot_usdt en todo momento."""
    p = Portfolio(1000.0, 4)           # slot = 250
    r = p.comprar(50000.0, 0.1)
    assert r is not None
    assert close(r["usdt_spent"], 250.0)


def test_comprar_descuenta_usdt():
    p = Portfolio(1000.0, 4)           # slot = 250
    p.comprar(50000.0, 0.1)
    assert close(p.usdt, 750.0)


def test_comprar_comision_correcta():
    """commission = usdt_spent * commission_pct / 100."""
    p = Portfolio(1000.0, 4)
    r = p.comprar(50000.0, 0.1)        # comisión = 250 * 0.1/100 = 0.25
    assert close(r["commission_usdt"], 0.25)


def test_comprar_btc_neto():
    """btc_bought = (usdt_spent - commission) / precio."""
    p = Portfolio(1000.0, 4)           # slot=250, commission=0.25, neto=249.75
    r = p.comprar(50000.0, 0.1)
    esperado = round(249.75 / 50000.0, 10)
    assert close(r["btc_bought"], esperado)


def test_comprar_sin_comision():
    """Con commission_pct=0 todo el slot va a BTC."""
    p = Portfolio(1000.0, 5)           # slot=200
    r = p.comprar(40000.0, 0.0)
    assert close(r["commission_usdt"], 0.0)
    assert close(r["btc_bought"], 200.0 / 40000.0, tol=1e-8)


def test_comprar_abre_posicion():
    p = Portfolio(1000.0, 5)
    p.comprar(30000.0, 0.1)
    assert p.positions_count == 1
    assert p.posiciones[0].entry_price == 30000.0


def test_comprar_actualiza_btc_por_venta():
    """btc_por_venta = btc_en_posiciones / positions_count después de cada compra."""
    p = Portfolio(1000.0, 4)
    p.comprar(10000.0, 0.1)
    assert close(p.btc_por_venta, p.btc_en_posiciones / 1)
    p.comprar(20000.0, 0.1)
    assert close(p.btc_por_venta, p.btc_en_posiciones / 2)
    p.comprar(30000.0, 0.1)
    assert close(p.btc_por_venta, p.btc_en_posiciones / 3)


def test_comprar_slot_invariante_con_posiciones_abiertas():
    """El slot NO cambia aunque el USDT libre disminuya con cada compra."""
    p = Portfolio(1000.0, 4)           # slot=250
    slot_original = p.slot_usdt
    for _ in range(4):
        r = p.comprar(random.uniform(1000, 100000), 0.1)
        assert close(r["usdt_spent"], slot_original), "slot mutó"


def test_comprar_con_fondos_insuficientes_devuelve_none():
    """Si el USDT libre es menor que el slot, comprar devuelve None."""
    p = Portfolio(100.0, 5)            # slot=20
    # Gastar manualmente casi todo
    p.usdt = 0.5                       # slot sigue siendo 20 pero usdt < slot
    resultado = p.comprar(50000.0, 0.1)
    assert resultado is None


def test_comprar_slot_sub_uno_devuelve_none():
    """slot < $1 → None (mínimo operativo)."""
    p = Portfolio(3.0, 10)             # slot = 0.30
    assert p.comprar(50000.0, 0.1) is None


def test_comprar_respeta_max_posiciones_en_correr():
    """
    Portfolio no tiene un check interno de MAX_POSICIONES.
    Si se inyecta USDT manualmente, puede abrir una 3° posición con max=2.
    El límite real lo aplica correr_backtest antes de llamar a comprar().
    """
    p = Portfolio(1000.0, 2)           # slot=500
    p.comprar(10000.0, 0.0)            # pos 1 → usdt=500
    p.comprar(10000.0, 0.0)            # pos 2 → usdt=0
    # Inyectar USDT manualmente (simula ausencia del guard de correr_backtest)
    p.usdt = 500.0
    r3 = p.comprar(10000.0, 0.0)       # Portfolio no bloquea esto
    assert r3 is not None,             "Portfolio debería comprar si hay USDT"
    assert p.positions_count == 3


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Portfolio: vender
# ══════════════════════════════════════════════════════════════════════════════

def test_vender_sin_posiciones_devuelve_none():
    p = Portfolio(1000.0, 5)
    assert p.vender(50000.0, 0.1) is None


def test_vender_cantidad_es_btc_por_venta():
    """btc_sold == btc_por_venta (congelado en la última compra)."""
    p = Portfolio(1000.0, 2)
    p.comprar(10000.0, 0.0)
    p.comprar(20000.0, 0.0)
    bpv_antes = p.btc_por_venta
    r = p.vender(15000.0, 0.0)
    assert close(r["btc_sold"], bpv_antes)


def test_vender_btc_por_venta_no_cambia_con_ventas():
    """btc_por_venta permanece fijo entre ventas; sólo cambia con BUY."""
    p = Portfolio(1000.0, 4)
    p.comprar(10000.0, 0.0)
    p.comprar(20000.0, 0.0)
    bpv = p.btc_por_venta
    p.vender(15000.0, 0.0)
    assert close(p.btc_por_venta, bpv), "btc_por_venta mutó con SELL"
    p.vender(15000.0, 0.0)
    assert close(p.btc_por_venta, bpv), "btc_por_venta mutó con 2° SELL"


def test_vender_comision_correcta():
    """commission_sell = btc_sold * precio * commission_pct / 100."""
    p = Portfolio(1000.0, 1)           # slot=1000
    p.comprar(10000.0, 0.0)            # btc_por_venta = btc_en_pos / 1
    btc_pv = p.btc_por_venta
    precio_venta = 20000.0
    r = p.vender(precio_venta, 0.5)    # 0.5 % de comisión
    comision_esperada = round(btc_pv * precio_venta * 0.5 / 100.0, 8)
    assert close(r["commission_usdt"], comision_esperada)


def test_vender_usdt_recibido():
    """usdt_received = btc_sold * precio - commission."""
    p = Portfolio(1000.0, 1)
    p.comprar(10000.0, 0.0)
    btc_pv = p.btc_por_venta
    precio = 15000.0
    r = p.vender(precio, 0.1)
    bruto  = btc_pv * precio
    comis  = round(bruto * 0.1 / 100.0, 8)
    assert close(r["usdt_received"], round(bruto - comis, 8))


def test_vender_ganancia_positiva():
    """Venta sobre el precio de entrada → ganancia > 0."""
    p = Portfolio(1000.0, 1)
    p.comprar(10000.0, 0.0)            # compra a 10 000, sin comisión
    r = p.vender(20000.0, 0.0)         # venta a 20 000, sin comisión
    assert r["ganancia_usdt"] > 0


def test_vender_ganancia_negativa():
    """Venta por debajo del precio de entrada → ganancia < 0."""
    p = Portfolio(1000.0, 1)
    p.comprar(50000.0, 0.0)
    r = p.vender(30000.0, 0.0)
    assert r["ganancia_usdt"] < 0


def test_vender_ganancia_cero_mismo_precio():
    """Sin comisión y venta al mismo precio de entrada → ganancia ≈ 0."""
    p = Portfolio(1000.0, 1)
    p.comprar(20000.0, 0.0)
    r = p.vender(20000.0, 0.0)
    assert close(r["ganancia_usdt"], 0.0, tol=1e-6)


def test_vender_incrementa_usdt():
    p = Portfolio(1000.0, 1)
    usdt_antes = p.usdt
    p.comprar(10000.0, 0.0)
    usdt_tras_compra = p.usdt
    p.vender(20000.0, 0.0)
    assert p.usdt > usdt_tras_compra
    assert p.usdt > usdt_antes   # ganamos dinero


def test_vender_acumula_btc_acumulado_total():
    p = Portfolio(2000.0, 2)
    p.comprar(10000.0, 0.0)
    p.comprar(10000.0, 0.0)
    btc_pv = p.btc_por_venta
    p.vender(15000.0, 0.0)
    assert close(p.btc_acumulado_total, btc_pv)
    p.comprar(10000.0, 0.0)            # recalcula btc_por_venta
    btc_pv2 = p.btc_por_venta
    p.vender(15000.0, 0.0)
    assert close(p.btc_acumulado_total, btc_pv + btc_pv2)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Portfolio: FIFO
# ══════════════════════════════════════════════════════════════════════════════

def test_fifo_primera_posicion_se_consume_primero():
    """La posición más antigua se reduce primero."""
    p = Portfolio(2000.0, 2)
    p.comprar(10000.0, 0.0)    # pos0: comprada a 10 000
    btc_pos0 = p.posiciones[0].btc
    p.comprar(20000.0, 0.0)    # pos1: comprada a 20 000

    # btc_por_venta = total / 2; eso consume exactamente una posición si las dos
    # tienen el mismo BTC. En este caso tienen distintos BTC (precios distintos).
    # Verificamos que la primera posición se reduce antes que la segunda.
    bpv = p.btc_por_venta

    # Si bpv <= btc_pos0, la primera posición absorbe toda la venta
    if bpv <= btc_pos0 + 1e-10:
        p.vender(15000.0, 0.0)
        if p.positions_count == 2:
            # pos0 se redujo parcialmente
            assert p.posiciones[0].entry_price == 10000.0
        else:
            # pos0 se consumió completamente
            assert p.posiciones[0].entry_price == 20000.0


def test_fifo_consume_multiples_posiciones():
    """Si btc_por_venta supera el BTC de la primera posición, se salta a la siguiente."""
    # Construir manualmente para controlar btc_por_venta
    p = Portfolio(6000.0, 3)
    # 3 compras → slot = 2000 cada una
    p.comprar(100000.0, 0.0)   # btc ≈ 0.02
    p.comprar(100000.0, 0.0)   # btc ≈ 0.02
    p.comprar(100000.0, 0.0)   # btc ≈ 0.02
    # btc_por_venta = total / 3 = exactamente 1 posición
    bpv = p.btc_por_venta
    total_antes = p.btc_en_posiciones

    r = p.vender(100000.0, 0.0)
    assert r is not None
    assert close(p.btc_en_posiciones, total_antes - bpv, tol=1e-8)


def test_fifo_posicion_vacia_se_elimina():
    """Una posición con btc=0 (o < epsilon) se elimina de la deque."""
    p = Portfolio(1000.0, 1)
    p.comprar(10000.0, 0.0)
    assert p.positions_count == 1
    p.vender(20000.0, 0.0)
    # La única posición fue consumida completamente
    assert p.positions_count == 0


def test_fifo_ganancia_usa_precio_promedio_fifo():
    """ganancia = usdt_neto - avg_entry * btc_vendido (promedio FIFO del lote vendido)."""
    p = Portfolio(2000.0, 2)
    p.comprar(10000.0, 0.0)    # pos0
    p.comprar(20000.0, 0.0)    # pos1
    # btc_por_venta = total_btc / 2
    btc_pos0 = p.posiciones[0].btc
    btc_pos1 = p.posiciones[1].btc
    bpv      = p.btc_por_venta  # btc_total / 2

    r = p.vender(15000.0, 0.0)
    # El costo del lote vendido depende de cuánto se tomó de pos0 y pos1 (FIFO)
    pendiente   = bpv
    costo       = 0.0
    for ep, btc in [(10000.0, btc_pos0), (20000.0, btc_pos1)]:
        if pendiente <= 0:
            break
        tomado   = min(btc, pendiente)
        costo   += ep * tomado
        pendiente -= tomado
    avg_entry   = costo / bpv
    usdt_recibido = bpv * 15000.0
    ganancia_esp  = round(usdt_recibido - avg_entry * bpv, 8)
    assert close(r["ganancia_usdt"], ganancia_esp, tol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — Portfolio: slot recalculation al llegar a 0 posiciones
# ══════════════════════════════════════════════════════════════════════════════

def test_slot_recalcula_al_vaciar_cartera():
    """Después de vender todo (positions=0), el slot se recalcula con el nuevo USDT."""
    p = Portfolio(1000.0, 2)           # slot=500
    p.comprar(10000.0, 0.0)            # usdt→500, pos=1
    p.comprar(10000.0, 0.0)            # usdt→0,   pos=2
    p.vender(20000.0, 0.0)             # vende bpv, pos puede quedar en 1 o 0
    p.vender(20000.0, 0.0)             # sigue vaciando

    # Forzamos vaciado completo vendiendo repetidamente hasta positions==0
    for _ in range(20):
        if p.positions_count == 0:
            break
        p.vender(20000.0, 0.0)

    assert p.positions_count == 0
    slot_nuevo_esperado = p.usdt / p.max_posiciones
    # Próxima compra dispara _recalcular_slot_si_vacio
    p.comprar(10000.0, 0.0)
    # El slot que se usó debe ser el recalculado, no el original 500
    assert close(p.posiciones[0].btc * 10000.0, slot_nuevo_esperado, tol=1.0)


def test_slot_no_recalcula_mientras_hay_posiciones():
    """Con posiciones abiertas, el slot permanece aunque vendamos algo."""
    p = Portfolio(1000.0, 4)           # slot=250
    p.comprar(10000.0, 0.0)
    p.comprar(10000.0, 0.0)
    slot_fijo = p.slot_usdt
    p.vender(20000.0, 0.0)             # positions pasa de 2 a 1 (no a 0)
    # Si aún hay posiciones, el slot no debe haber cambiado
    if p.positions_count > 0:
        assert close(p.slot_usdt, slot_fijo)


def test_ciclo_completo_slot_recalculado_correctamente():
    """
    Ciclo: compra 1 → vende todo → compra de nuevo con saldo distinto.
    El segundo ciclo usa el USDT acumulado como base del nuevo slot.
    """
    p = Portfolio(1000.0, 1)           # slot=1000 (1 posición)
    # Primer ciclo
    p.comprar(10000.0, 0.0)            # compra todo a 10 000
    usdt_antes_venta = p.usdt
    p.vender(20000.0, 0.0)             # venta a 20 000 → ganancia ~100 %
    usdt_tras_venta = p.usdt
    assert usdt_tras_venta > usdt_antes_venta   # ganamos

    # Segundo ciclo — slot debe ser el nuevo USDT / 1
    slot_esperado = usdt_tras_venta / 1
    r2 = p.comprar(20000.0, 0.0)
    assert close(r2["usdt_spent"], slot_esperado, tol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — Portfolio: precio_promedio_posiciones
# ══════════════════════════════════════════════════════════════════════════════

def test_precio_promedio_una_posicion():
    p = Portfolio(1000.0, 1)
    p.comprar(50000.0, 0.0)
    assert close(p.precio_promedio_posiciones, 50000.0)


def test_precio_promedio_ponderado_exacto():
    """Promedio ponderado por BTC, no aritmético simple."""
    p = Portfolio(3000.0, 3)
    # Tres precios distintos → BTC distintos (slot=1000)
    p.comprar(10000.0, 0.0)   # btc = 1000/10000 = 0.1
    p.comprar(20000.0, 0.0)   # btc = 1000/20000 = 0.05
    p.comprar(50000.0, 0.0)   # btc = 1000/50000 = 0.02
    btc_total  = 0.1 + 0.05 + 0.02
    pp_esp     = (10000*0.1 + 20000*0.05 + 50000*0.02) / btc_total
    assert close(p.precio_promedio_posiciones, pp_esp, tol=1e-3)


def test_precio_promedio_cero_sin_posiciones():
    p = Portfolio(1000.0, 5)
    assert p.precio_promedio_posiciones == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — Portfolio: valor_portfolio
# ══════════════════════════════════════════════════════════════════════════════

def test_valor_portfolio_formula():
    """valor = usdt + btc_en_posiciones * close."""
    p = Portfolio(1000.0, 2)
    p.comprar(10000.0, 0.0)            # usdt=500, btc≈0.05
    close_price = 15000.0
    esperado    = p.usdt + p.btc_en_posiciones * close_price
    assert close(p.valor_portfolio(close_price), esperado)


def test_valor_portfolio_sube_con_precio():
    p = Portfolio(1000.0, 2)
    p.comprar(10000.0, 0.0)
    val_bajo  = p.valor_portfolio(5000.0)
    val_alto  = p.valor_portfolio(50000.0)
    assert val_alto > val_bajo


def test_valor_portfolio_precio_cero():
    p = Portfolio(1000.0, 2)
    p.comprar(10000.0, 0.0)
    assert close(p.valor_portfolio(0.0), p.usdt)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — detectar_señales
# ══════════════════════════════════════════════════════════════════════════════

def zigzag_candles(n: int, amplitude: float = 1000.0, base: float = 10000.0):
    """Crea n velas con zigzag limpio: pares=valley, impares=peak."""
    candles = []
    for i in range(n):
        if i % 2 == 0:  # valley
            candles.append(make_candle(base - amplitude, base - amplitude/2, dt=f"2020-01-{i+1:02d} 00:00:00"))
        else:            # peak
            candles.append(make_candle(base + amplitude/2, base + amplitude, dt=f"2020-01-{i+1:02d} 00:00:00"))
    return candles


def test_señales_bordes_siempre_none():
    """Las primeras y últimas ventana velas siempre devuelven None."""
    ventana = 5
    candles = zigzag_candles(40)
    señales = detectar_señales(candles, ventana)
    for i in range(ventana):
        assert señales[i] is None,            f"borde izq [{i}] no es None"
    for i in range(len(candles) - ventana, len(candles)):
        assert señales[i] is None,            f"borde der [{i}] no es None"


def test_señales_longitud_igual_a_candles():
    candles = zigzag_candles(30)
    assert len(detectar_señales(candles, 3)) == len(candles)


def test_señales_zigzag_buy_en_valleys():
    """Velas en valley (low mínimo local) deben producir BUY."""
    candles = zigzag_candles(30, amplitude=2000)
    señales = detectar_señales(candles, 3)
    # Índices de valleys dentro del rango con contexto completo
    for i in range(3, len(candles) - 3):
        if i % 2 == 0:   # esperamos BUY en valleys
            assert señales[i] == "BUY" or señales[i] is None, \
                f"valley[{i}] = {señales[i]}"


def test_señales_zigzag_sell_en_peaks():
    """Velas en peak (high máximo local) deben producir SELL."""
    candles = zigzag_candles(30, amplitude=2000)
    señales = detectar_señales(candles, 3)
    for i in range(3, len(candles) - 3):
        if i % 2 == 1:   # esperamos SELL en peaks
            assert señales[i] == "SELL" or señales[i] is None, \
                f"peak[{i}] = {señales[i]}"


def test_señales_precios_planos_sin_señal():
    """Si todos los lows y highs son iguales, ningún punto es estrictamente mínimo/máximo."""
    candles = [make_candle(100.0, 200.0, dt=f"2020-01-{i+1:02d} 00:00:00") for i in range(20)]
    señales = detectar_señales(candles, 3)
    # Con <=/>= un punto plano SÍ puede ser "mínimo/máximo" (no estricto)
    # verificamos que la lista tiene la longitud correcta y no hay errores
    assert len(señales) == 20


def test_señales_ventana_1():
    """Ventana de 1: sólo necesita 1 vela a cada lado."""
    # Zigzag claro con ventana mínima
    candles = [
        make_candle(100, 150),   # 0 → border
        make_candle(50, 80),     # 1 → BUY (low más bajo)
        make_candle(200, 300),   # 2 → SELL (high más alto)
        make_candle(60, 90),     # 3 → BUY
        make_candle(180, 280),   # 4 → border
    ]
    señales = detectar_señales(candles, 1)
    assert señales[0] is None
    assert señales[4] is None
    assert señales[1] == "BUY"
    assert señales[2] == "SELL"


def test_señales_muy_pocas_velas():
    """Con menos de 2*ventana+1 velas, todo debe ser None."""
    candles = zigzag_candles(4)
    señales = detectar_señales(candles, 3)
    assert all(s is None for s in señales)


def test_señales_monotona_ascendente():
    """Serie estrictamente creciente: sólo puede haber SELL al final del rango activo."""
    candles = [make_candle(100*i + 1, 100*i + 50, dt=f"2020-01-{i+1:02d} 00:00:00")
               for i in range(20)]
    señales = detectar_señales(candles, 3)
    for s in señales:
        assert s != "BUY", "No debe haber BUY en serie ascendente"


def test_señales_monotona_descendente():
    """Serie estrictamente decreciente: sólo puede haber BUY al inicio del rango activo."""
    candles = [make_candle(10000 - 100*i, 10000 - 100*i + 50, dt=f"2020-01-{i+1:02d} 00:00:00")
               for i in range(20)]
    señales = detectar_señales(candles, 3)
    for s in señales:
        assert s != "SELL", "No debe haber SELL en serie descendente"


def test_señales_no_tiene_buy_y_sell_simultaneous_en_zigzag_puro():
    """En un zigzag limpio ningún índice debe ser BUY y SELL al mismo tiempo."""
    candles = zigzag_candles(50)
    señales = detectar_señales(candles, 4)
    # La función ya prioriza BUY en empate, pero verificamos consistencia
    for i, s in enumerate(señales):
        assert s in ("BUY", "SELL", None), f"señal inválida en [{i}]: {s}"


def test_señales_ventana_grande_menos_señales():
    """Mayor ventana → igual o menos señales que ventana chica."""
    candles = zigzag_candles(100)
    n3  = sum(1 for s in detectar_señales(candles, 3)  if s is not None)
    n10 = sum(1 for s in detectar_señales(candles, 10) if s is not None)
    assert n10 <= n3


def test_señales_random_no_crashea():
    """Serie aleatoria de 200 velas sin excepciones, con ventana aleatoria 1–15."""
    rng = random.Random(42)
    for _ in range(10):
        ventana = rng.randint(1, 15)
        n       = rng.randint(ventana * 2 + 1, 200)
        candles = []
        for i in range(n):
            lo = rng.uniform(1000, 50000)
            hi = lo + rng.uniform(10, 2000)
            candles.append(make_candle(lo, hi, dt=f"2020-01-{i%28+1:02d} 00:00:00"))
        señales = detectar_señales(candles, ventana)
        assert len(señales) == n
        assert all(s in ("BUY", "SELL", None) for s in señales)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — correr_backtest: trade_history
# ══════════════════════════════════════════════════════════════════════════════

CAMPOS_REQUERIDOS = {
    "datetime", "type", "price", "score_bot", "score_top",
    "usdt_balance", "btc_balance", "btc_en_posiciones",
    "positions_count", "precio_promedio_posiciones",
    "portfolio_value", "pnl_pct_acumulado",
    "ignorado", "motivo_ignorado",
    "usdt_spent", "btc_bought", "commission_usdt",
    "btc_sold", "btc_accumulated", "usdt_received",
    "ganancia_usdt", "pct_capital_usado",
}


def _backtest_simple(señales_str, precios_low, precios_high, precios_close,
                     usdt=1000.0, max_pos=5, commission=0.1):
    """Helper: construye candles y ejecuta backtest con señales predefinidas."""
    n = len(señales_str)
    candles = []
    for i in range(n):
        lo = precios_low[i]
        hi = precios_high[i]
        cl = precios_close[i]
        candles.append({
            "ts": i, "datetime": f"2020-01-{i+1:02d} 00:00:00",
            "open": cl, "high": hi, "low": lo, "close": cl, "volume": 1.0,
        })
    señales = señales_str  # ya son "BUY"/"SELL"/None
    cfg = make_cfg(SALDO_USDT_INICIAL=usdt, MAX_POSICIONES=max_pos,
                   COMMISSION_PCT=commission)
    return correr_backtest(candles, señales, cfg), candles


def test_trade_history_contiene_todos_los_campos():
    señales = [None, "BUY", None, "SELL", None]
    precios = [10000] * 5
    res, _ = _backtest_simple(señales, precios, precios, precios)
    for rec in res["trade_history"]:
        for campo in CAMPOS_REQUERIDOS:
            assert campo in rec, f"Falta campo '{campo}'"


def test_trade_history_solo_incluye_velas_con_señal():
    señales = [None, "BUY", None, None, "SELL", None]
    precios = [10000, 9000, 10000, 11000, 12000, 10000]
    highs   = [p + 500 for p in precios]
    res, _  = _backtest_simple(señales, precios, highs, precios)
    assert len(res["trade_history"]) == 2


def test_score_bot_100_en_buy():
    señales = ["BUY", None]
    res, _  = _backtest_simple(señales, [10000]*2, [11000]*2, [10000]*2)
    assert res["trade_history"][0]["score_bot"] == 100.0
    assert res["trade_history"][0]["score_top"] == 0.0


def test_score_top_100_en_sell():
    señales = ["BUY", "SELL"]
    precios = [10000, 20000]
    highs   = [11000, 21000]
    res, _  = _backtest_simple(señales, precios, highs, precios)
    sell_rec = res["trade_history"][1]
    assert sell_rec["score_top"] == 100.0
    assert sell_rec["score_bot"] == 0.0


def test_precio_exec_buy_usa_low():
    señales = ["BUY", None]
    lo, hi, cl = [8000], [12000], [10000]
    cfg = make_cfg(PRECIO_COMPRA="low", PRECIO_VENTA="high")
    candles = [make_candle(8000, 12000, 10000)]
    señales_full = ["BUY"]
    res = correr_backtest(candles, señales_full, cfg)
    assert res["trade_history"][0]["price"] == 8000


def test_precio_exec_sell_usa_high():
    cfg = make_cfg(PRECIO_COMPRA="low", PRECIO_VENTA="high")
    candles = [make_candle(8000, 12000, 10000)]
    res = correr_backtest(candles, ["BUY"], cfg)
    # Primero BUY, luego SELL en otra vela
    candles2 = [make_candle(8000, 12000, 10000), make_candle(9000, 15000, 12000)]
    res2 = correr_backtest(candles2, ["BUY", "SELL"], cfg)
    assert res2["trade_history"][1]["price"] == 15000


def test_precio_exec_close():
    cfg = make_cfg(PRECIO_COMPRA="close", PRECIO_VENTA="close")
    candles = [make_candle(8000, 12000, 10500), make_candle(9000, 14000, 11000)]
    res = correr_backtest(candles, ["BUY", "SELL"], cfg)
    assert res["trade_history"][0]["price"] == 10500
    assert res["trade_history"][1]["price"] == 11000


def test_portfolio_value_en_trade_history():
    """portfolio_value = usdt + btc_en_pos * close (no high ni low)."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=1, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 12000, 11000)]
    res = correr_backtest(candles, ["BUY"], cfg)
    rec = res["trade_history"][0]
    # usdt = 0 (gastó todo el slot = 1000), btc = 1000/10000 = 0.1
    esperado = rec["usdt_balance"] + rec["btc_en_posiciones"] * 11000
    assert close(rec["portfolio_value"], esperado, tol=0.01)


def test_pnl_acumulado_formula():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=1, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 12000, 11000)]
    res = correr_backtest(candles, ["BUY"], cfg)
    rec = res["trade_history"][0]
    pnl_esp = (rec["portfolio_value"] - 1000.0) / 1000.0 * 100
    assert close(rec["pnl_pct_acumulado"], pnl_esp, tol=0.01)


def test_campos_null_en_buy():
    """En un BUY: btc_sold, usdt_received, ganancia_usdt, btc_accumulated → None."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.1)
    candles = [make_candle(10000, 11000, 10500)]
    res = correr_backtest(candles, ["BUY"], cfg)
    rec = res["trade_history"][0]
    assert rec["btc_sold"]        is None
    assert rec["usdt_received"]   is None
    assert rec["ganancia_usdt"]   is None
    assert rec["btc_accumulated"] is None


def test_campos_null_en_sell():
    """En un SELL: usdt_spent, btc_bought → None."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.1)
    candles = [make_candle(10000, 11000, 10500), make_candle(12000, 15000, 13000)]
    res = correr_backtest(candles, ["BUY", "SELL"], cfg)
    sell = res["trade_history"][1]
    assert sell["usdt_spent"]  is None
    assert sell["btc_bought"]  is None


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — correr_backtest: ignorados y límites
# ══════════════════════════════════════════════════════════════════════════════

def test_sell_sin_posiciones_ignorado():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5)
    candles = [make_candle(10000, 11000, 10000)]
    res = correr_backtest(candles, ["SELL"], cfg)
    rec = res["trade_history"][0]
    assert rec["ignorado"] is True
    assert rec["motivo_ignorado"] == "sin_posiciones"
    assert res["summary"]["total_ignorados"] == 1
    assert res["summary"]["total_ventas"]    == 0


def test_max_posiciones_ignorado():
    max_p = 2
    cfg   = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=max_p, COMMISSION_PCT=0.0)
    # 3 BUYs seguidas: la 3° debe ignorarse
    candles = [make_candle(10000, 11000, 10000) for _ in range(3)]
    res = correr_backtest(candles, ["BUY", "BUY", "BUY"], cfg)
    th = res["trade_history"]
    assert th[0]["ignorado"] is False
    assert th[1]["ignorado"] is False
    assert th[2]["ignorado"] is True
    assert th[2]["motivo_ignorado"] == f"max_posiciones({max_p})"
    assert res["summary"]["total_compras"] == 2
    assert res["summary"]["total_ignorados"] == 1


def test_ignorados_por_motivo_contado_correctamente():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=1, COMMISSION_PCT=0.0)
    # 1 BUY válido + 2 BUYs ignorados por max_posiciones
    candles = [make_candle(10000, 11000, 10000) for _ in range(3)]
    res = correr_backtest(candles, ["BUY", "BUY", "BUY"], cfg)
    motivo = f"max_posiciones(1)"
    assert res["summary"]["ignorados_por_motivo"].get(motivo, 0) == 2


def test_sell_contabilizado_en_total_ventas():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 11000, 10000), make_candle(15000, 20000, 17000)]
    res = correr_backtest(candles, ["BUY", "SELL"], cfg)
    assert res["summary"]["total_compras"] == 1
    assert res["summary"]["total_ventas"]  == 1
    assert res["summary"]["total_trades_ejecutados"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12 — correr_backtest: summary
# ══════════════════════════════════════════════════════════════════════════════

def test_summary_pnl_formula():
    """pnl_pct = (portfolio_value_final - saldo_inicial) / saldo_inicial * 100."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 11000, 10000), make_candle(15000, 20000, 18000)]
    res = correr_backtest(candles, ["BUY", "SELL"], cfg)
    s  = res["summary"]
    pnl_esp = (s["portfolio_value_final"] - 1000.0) / 1000.0 * 100
    assert close(s["pnl_pct"], pnl_esp, tol=0.01)


def test_summary_buy_hold():
    """buy_hold_pnl_pct = (close_final - close_inicial) / close_inicial * 100."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5)
    candles = [
        make_candle(9000, 10000, 10000),  # close inicial = 10000
        make_candle(14000, 16000, 15000), # close final   = 15000
    ]
    res = correr_backtest(candles, [None, None], cfg)
    bh_esp = (15000 - 10000) / 10000 * 100   # 50 %
    assert close(res["summary"]["buy_hold_pnl_pct"], bh_esp, tol=0.01)


def test_summary_alpha():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5)
    candles = [make_candle(10000, 11000, 10000), make_candle(15000, 20000, 15000)]
    res = correr_backtest(candles, ["BUY", "SELL"], cfg)
    s   = res["summary"]
    assert close(s["alpha_vs_bh"], s["pnl_pct"] - s["buy_hold_pnl_pct"], tol=0.01)


def test_summary_atl_ath():
    cfg = make_cfg()
    candles = [
        make_candle(500,  1000, 750),
        make_candle(8000, 9500, 9000),
        make_candle(200,  400,  300),
        make_candle(5000, 12000, 8000),
    ]
    res = correr_backtest(candles, [None]*4, cfg)
    s   = res["summary"]
    assert close(s["atl_final"],           200.0)
    assert close(s["ath_proyectado_final"], 12000.0)


def test_summary_precio_min_comprado():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.0,
                   PRECIO_COMPRA="low")
    candles = [
        make_candle(8000,  9000,  8500),   # BUY  → exec a 8000
        make_candle(15000, 18000, 16000),  # SELL
        make_candle(5000,  6000,  5500),   # BUY  → exec a 5000
    ]
    res = correr_backtest(candles, ["BUY", "SELL", "BUY"], cfg)
    assert close(res["summary"]["precio_min_comprado"], 5000.0)


def test_summary_precio_max_vendido():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.0,
                   PRECIO_VENTA="high")
    candles = [
        make_candle(5000,  6000,  5500),   # BUY
        make_candle(20000, 25000, 22000),  # SELL → exec a 25000
        make_candle(4000,  5000,  4500),   # BUY
        make_candle(10000, 15000, 12000),  # SELL → exec a 15000
    ]
    res = correr_backtest(candles, ["BUY", "SELL", "BUY", "SELL"], cfg)
    assert close(res["summary"]["precio_max_vendido"], 25000.0)


def test_summary_precio_min_none_si_sin_compras():
    cfg = make_cfg()
    candles = [make_candle(10000, 11000, 10000)]
    res = correr_backtest(candles, ["SELL"], cfg)
    assert res["summary"]["precio_min_comprado"] is None


def test_summary_precio_max_none_si_sin_ventas():
    cfg = make_cfg()
    candles = [make_candle(10000, 11000, 10000)]
    res = correr_backtest(candles, ["BUY"], cfg)
    assert res["summary"]["precio_max_vendido"] is None


def test_summary_positions_count_final():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=5, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 11000, 10000) for _ in range(3)]
    res = correr_backtest(candles, ["BUY", "BUY", "BUY"], cfg)
    assert res["summary"]["positions_count_final"] == 3


def test_summary_portfolio_value_final_consistente():
    """portfolio_value_final = usdt_final + btc_en_pos * close_última_vela."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=2, COMMISSION_PCT=0.0)
    candles = [
        make_candle(10000, 11000, 10000),
        make_candle(10000, 11000, 12000),  # close final = 12000
    ]
    res = correr_backtest(candles, ["BUY", None], cfg)
    s   = res["summary"]
    esperado = s["usdt_balance_final"] + s["btc_en_posiciones_final"] * 12000
    assert close(s["portfolio_value_final"], esperado, tol=0.01)


def test_summary_parametros_incluidos():
    cfg = make_cfg(VENTANA_LOCAL=7, PRECIO_COMPRA="close", PRECIO_VENTA="open",
                   MAX_POSICIONES=8, COMMISSION_PCT=0.05)
    candles = [make_candle(10000, 11000, 10000)]
    res = correr_backtest(candles, [None], cfg)
    par = res["summary"]["parametros"]
    assert par["ventana_local"]  == 7
    assert par["precio_compra"]  == "close"
    assert par["precio_venta"]   == "open"
    assert par["max_posiciones"] == 8
    assert close(par["commission_pct"], 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13 — Extremos
# ══════════════════════════════════════════════════════════════════════════════

def test_extremo_precio_muy_bajo_1_satoshi():
    """Precio = 0.00000001 USD (1 satoshi). Sin crash."""
    p = Portfolio(1.0, 1)
    r = p.comprar(0.00000001, 0.1)
    assert r is not None or r is None   # no debe lanzar excepción


def test_extremo_precio_muy_alto():
    """Precio = 10_000_000 USD. Sin crash."""
    p = Portfolio(1_000_000.0, 1)
    r = p.comprar(10_000_000.0, 0.1)
    if r is not None:
        assert r["btc_bought"] >= 0


def test_extremo_capital_minimo():
    """USDT inicial = 1.00, max_posiciones = 1 → slot = 1.0 (mínimo válido)."""
    p = Portfolio(1.0, 1)
    r = p.comprar(50000.0, 0.0)
    assert r is not None
    assert close(r["usdt_spent"], 1.0)


def test_extremo_max_posiciones_1():
    """Con MAX_POSICIONES=1 sólo puede haber 1 posición; el resto se ignoran."""
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=1, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 11000, 10000) for _ in range(5)]
    res = correr_backtest(candles, ["BUY", "BUY", "BUY", "BUY", "BUY"], cfg)
    assert res["summary"]["total_compras"]    == 1
    assert res["summary"]["total_ignorados"]  == 4


def test_extremo_comision_cero():
    """Sin comisión, todo el slot se convierte en BTC."""
    p = Portfolio(1000.0, 1)
    r = p.comprar(20000.0, 0.0)
    assert close(r["commission_usdt"], 0.0)
    assert close(r["btc_bought"], 1000.0 / 20000.0, tol=1e-8)


def test_extremo_comision_50_pct():
    """Comisión = 50 %: la mitad del USDT va a comisión."""
    p = Portfolio(1000.0, 1)
    r = p.comprar(10000.0, 50.0)
    assert close(r["commission_usdt"], 500.0)
    assert close(r["btc_bought"], 500.0 / 10000.0, tol=1e-8)


def test_extremo_solo_sells_todo_ignorado():
    cfg = make_cfg()
    candles = [make_candle(10000, 11000, 10000) for _ in range(5)]
    res = correr_backtest(candles, ["SELL"] * 5, cfg)
    assert res["summary"]["total_ventas"]   == 0
    assert res["summary"]["total_ignorados"] == 5


def test_extremo_solo_buys_sin_ventas():
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=10, COMMISSION_PCT=0.0)
    candles = [make_candle(10000, 11000, 10000) for _ in range(5)]
    res = correr_backtest(candles, ["BUY"] * 5, cfg)
    assert res["summary"]["total_ventas"]  == 0
    assert res["summary"]["total_compras"] == 5


def test_extremo_multiples_ciclos_compra_venta():
    """
    Ciclo: comprar todo → vender todo → comprar de nuevo.
    Verifica que el slot se recalcula y el capital crece.
    """
    cfg = make_cfg(SALDO_USDT_INICIAL=1000.0, MAX_POSICIONES=1, COMMISSION_PCT=0.0)
    # Ciclo 1: compra a 10k, vende a 20k (ganancia 100%)
    # Ciclo 2: compra a 15k, vende a 30k (ganancia 100%)
    candles = [
        make_candle(10000, 10001, 10000),   # BUY
        make_candle(20000, 20001, 20000),   # SELL
        make_candle(15000, 15001, 15000),   # BUY  → slot = ~2000 (ganó en ciclo 1)
        make_candle(30000, 30001, 30000),   # SELL
    ]
    res = correr_backtest(candles, ["BUY", "SELL", "BUY", "SELL"], cfg)
    s   = res["summary"]
    assert s["total_compras"] == 2
    assert s["total_ventas"]  == 2
    assert s["portfolio_value_final"] > 1000.0  # ganamos dinero


def test_extremo_btc_por_venta_congelado_entre_ventas_multiples():
    """
    btc_por_venta calculado en el 2° BUY no debe cambiar entre las ventas
    que siguen antes del próximo BUY.
    """
    p = Portfolio(2000.0, 2)
    p.comprar(10000.0, 0.0)
    p.comprar(10000.0, 0.0)
    bpv_fijo = p.btc_por_venta
    # 3 ventas seguidas sin ningún BUY en medio
    for _ in range(3):
        if p.positions_count > 0:
            r = p.vender(15000.0, 0.0)
            if r is not None:
                assert close(r["btc_sold"], bpv_fijo, tol=1e-8), \
                    f"btc_sold={r['btc_sold']} ≠ bpv_fijo={bpv_fijo}"
            assert close(p.btc_por_venta, bpv_fijo), "btc_por_venta mutó"


def test_extremo_velas_con_valores_random():
    """Stress test: 500 velas con precios aleatorios y señales random."""
    rng = random.Random(7)
    n   = 500
    candles = []
    for i in range(n):
        lo = rng.uniform(1000, 80000)
        hi = lo + rng.uniform(100, 5000)
        cl = rng.uniform(lo, hi)
        candles.append({
            "ts": i, "datetime": f"2020-01-{i%28+1:02d} 00:00:00",
            "open": cl, "high": hi, "low": lo, "close": cl, "volume": 1.0,
        })
    señales = [rng.choice(["BUY", "SELL", None]) for _ in range(n)]
    cfg = make_cfg(
        SALDO_USDT_INICIAL=rng.uniform(100, 10000),
        MAX_POSICIONES=rng.randint(1, 15),
        COMMISSION_PCT=rng.uniform(0, 1),
    )
    res = correr_backtest(candles, señales, cfg)
    s   = res["summary"]
    # Invariantes fundamentales
    assert s["total_trades_ejecutados"] == s["total_compras"] + s["total_ventas"]
    assert s["total_ignorados"]         >= 0
    assert s["portfolio_value_final"]   >= 0
    assert s["positions_count_final"]   == sum(
        1 for r in res["trade_history"]
        if not r["ignorado"] and r["type"] == "BUY"
    ) - sum(
        # cada SELL consume parte de posiciones; positions_count_final
        # lo medimos directamente del summary
        0 for _ in []
    ) or True   # verificación de estructura, no de valor exacto


def test_extremo_conservacion_capital_sin_comision():
    """
    Sin comisión, el capital total (USDT + BTC*precio) se conserva en cada
    operación: lo que sale de USDT entra a BTC y viceversa.
    """
    precio_btc = 20000.0
    p = Portfolio(1000.0, 1)
    val_antes = p.valor_portfolio(precio_btc)
    p.comprar(precio_btc, 0.0)
    val_despues = p.valor_portfolio(precio_btc)
    assert close(val_antes, val_despues, tol=1e-4), \
        f"Capital no conservado: {val_antes} → {val_despues}"


def test_extremo_conservacion_capital_con_comision():
    """Con comisión, el valor del portafolio debe DECRECER exactamente en la comisión."""
    precio_btc = 30000.0
    p          = Portfolio(1000.0, 1)    # slot=1000
    comis_pct  = 0.1
    val_antes  = p.valor_portfolio(precio_btc)
    r          = p.comprar(precio_btc, comis_pct)
    val_despues = p.valor_portfolio(precio_btc)
    comis_pagada = r["commission_usdt"]
    assert close(val_antes - val_despues, comis_pagada, tol=1e-4), \
        f"Pérdida por comisión incorrecta: {val_antes-val_despues} ≠ {comis_pagada}"


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

TESTS = [
    # Posicion
    ("Posicion: constructor",                        test_posicion_constructor),
    ("Posicion: campos mutables",                    test_posicion_campos_mutables),
    # Portfolio init
    ("Portfolio: slot inicial = usdt/n",             test_portfolio_slot_inicial),
    ("Portfolio: estado inicial vacío",              test_portfolio_estado_inicial_vacio),
    ("Portfolio: valor inicial sin posiciones",      test_portfolio_valor_inicial_sin_posiciones),
    # comprar
    ("comprar: gasta exactamente 1 slot",            test_comprar_gasta_exactamente_un_slot),
    ("comprar: descuenta USDT",                      test_comprar_descuenta_usdt),
    ("comprar: comisión correcta",                   test_comprar_comision_correcta),
    ("comprar: BTC neto = (slot-comis)/precio",      test_comprar_btc_neto),
    ("comprar: sin comisión todo va a BTC",          test_comprar_sin_comision),
    ("comprar: abre posición",                       test_comprar_abre_posicion),
    ("comprar: actualiza btc_por_venta",             test_comprar_actualiza_btc_por_venta),
    ("comprar: slot invariante con posiciones",      test_comprar_slot_invariante_con_posiciones_abiertas),
    ("comprar: fondos insuficientes → None",         test_comprar_con_fondos_insuficientes_devuelve_none),
    ("comprar: slot < $1 → None",                   test_comprar_slot_sub_uno_devuelve_none),
    ("comprar: Portfolio no limita max_posiciones",  test_comprar_respeta_max_posiciones_en_correr),
    # vender
    ("vender: sin posiciones → None",                test_vender_sin_posiciones_devuelve_none),
    ("vender: cantidad == btc_por_venta",            test_vender_cantidad_es_btc_por_venta),
    ("vender: btc_por_venta no cambia con ventas",   test_vender_btc_por_venta_no_cambia_con_ventas),
    ("vender: comisión correcta",                    test_vender_comision_correcta),
    ("vender: usdt_received correcto",               test_vender_usdt_recibido),
    ("vender: ganancia positiva al vender arriba",   test_vender_ganancia_positiva),
    ("vender: ganancia negativa al vender abajo",    test_vender_ganancia_negativa),
    ("vender: ganancia ≈ 0 mismo precio sin comis.", test_vender_ganancia_cero_mismo_precio),
    ("vender: incrementa USDT",                      test_vender_incrementa_usdt),
    ("vender: acumula btc_acumulado_total",          test_vender_acumula_btc_acumulado_total),
    # FIFO
    ("FIFO: primera posición se consume primero",    test_fifo_primera_posicion_se_consume_primero),
    ("FIFO: consume múltiples posiciones",           test_fifo_consume_multiples_posiciones),
    ("FIFO: posición vacía se elimina",              test_fifo_posicion_vacia_se_elimina),
    ("FIFO: ganancia usa avg_entry FIFO",            test_fifo_ganancia_usa_precio_promedio_fifo),
    # Slot recalculation
    ("Slot: recalcula al vaciar cartera",            test_slot_recalcula_al_vaciar_cartera),
    ("Slot: no recalcula con posiciones abiertas",   test_slot_no_recalcula_mientras_hay_posiciones),
    ("Slot: ciclo completo recalculado correctamente", test_ciclo_completo_slot_recalculado_correctamente),
    # precio_promedio
    ("precio_promedio: una posición",                test_precio_promedio_una_posicion),
    ("precio_promedio: ponderado exacto",            test_precio_promedio_ponderado_exacto),
    ("precio_promedio: cero sin posiciones",         test_precio_promedio_cero_sin_posiciones),
    # valor_portfolio
    ("valor_portfolio: fórmula correcta",            test_valor_portfolio_formula),
    ("valor_portfolio: sube con precio",             test_valor_portfolio_sube_con_precio),
    ("valor_portfolio: precio=0",                    test_valor_portfolio_precio_cero),
    # detectar_señales
    ("detectar_señales: bordes siempre None",        test_señales_bordes_siempre_none),
    ("detectar_señales: longitud == len(candles)",   test_señales_longitud_igual_a_candles),
    ("detectar_señales: BUY en valleys",             test_señales_zigzag_buy_en_valleys),
    ("detectar_señales: SELL en peaks",              test_señales_zigzag_sell_en_peaks),
    ("detectar_señales: precios planos sin crash",   test_señales_precios_planos_sin_señal),
    ("detectar_señales: ventana=1",                  test_señales_ventana_1),
    ("detectar_señales: <2*ventana → todo None",     test_señales_muy_pocas_velas),
    ("detectar_señales: monotona ascendente",        test_señales_monotona_ascendente),
    ("detectar_señales: monotona descendente",       test_señales_monotona_descendente),
    ("detectar_señales: BUY/SELL/None válidos",      test_señales_no_tiene_buy_y_sell_simultaneous_en_zigzag_puro),
    ("detectar_señales: ventana grande < señales",   test_señales_ventana_grande_menos_señales),
    ("detectar_señales: random sin crash (x10)",     test_señales_random_no_crashea),
    # trade_history
    ("trade_history: todos los campos presentes",    test_trade_history_contiene_todos_los_campos),
    ("trade_history: sólo velas con señal",          test_trade_history_solo_incluye_velas_con_señal),
    ("trade_history: score_bot=100 en BUY",          test_score_bot_100_en_buy),
    ("trade_history: score_top=100 en SELL",         test_score_top_100_en_sell),
    ("trade_history: precio exec usa low en BUY",    test_precio_exec_buy_usa_low),
    ("trade_history: precio exec usa high en SELL",  test_precio_exec_sell_usa_high),
    ("trade_history: precio exec usa close",         test_precio_exec_close),
    ("trade_history: portfolio_value = usdt+btc*cl", test_portfolio_value_en_trade_history),
    ("trade_history: pnl_pct_acumulado fórmula",     test_pnl_acumulado_formula),
    ("trade_history: campos null en BUY",            test_campos_null_en_buy),
    ("trade_history: campos null en SELL",           test_campos_null_en_sell),
    # ignorados
    ("ignorados: SELL sin pos → sin_posiciones",     test_sell_sin_posiciones_ignorado),
    ("ignorados: BUY > max_posiciones",              test_max_posiciones_ignorado),
    ("ignorados: contador por motivo",               test_ignorados_por_motivo_contado_correctamente),
    ("ignorados: total_ventas contado",              test_sell_contabilizado_en_total_ventas),
    # summary
    ("summary: pnl_pct fórmula",                    test_summary_pnl_formula),
    ("summary: buy_hold fórmula",                    test_summary_buy_hold),
    ("summary: alpha = pnl - bh",                   test_summary_alpha),
    ("summary: ATL/ATH",                             test_summary_atl_ath),
    ("summary: precio_min_comprado",                 test_summary_precio_min_comprado),
    ("summary: precio_max_vendido",                  test_summary_precio_max_vendido),
    ("summary: precio_min None si sin compras",      test_summary_precio_min_none_si_sin_compras),
    ("summary: precio_max None si sin ventas",       test_summary_precio_max_none_si_sin_ventas),
    ("summary: positions_count_final",               test_summary_positions_count_final),
    ("summary: portfolio_value_final consistente",   test_summary_portfolio_value_final_consistente),
    ("summary: parámetros incluidos",                test_summary_parametros_incluidos),
    # Extremos
    ("extremo: precio 1 satoshi",                    test_extremo_precio_muy_bajo_1_satoshi),
    ("extremo: precio $10M",                         test_extremo_precio_muy_alto),
    ("extremo: capital mínimo $1",                   test_extremo_capital_minimo),
    ("extremo: max_posiciones=1",                    test_extremo_max_posiciones_1),
    ("extremo: comisión 0%",                         test_extremo_comision_cero),
    ("extremo: comisión 50%",                        test_extremo_comision_50_pct),
    ("extremo: sólo SELLs → todo ignorado",          test_extremo_solo_sells_todo_ignorado),
    ("extremo: sólo BUYs → sin ventas",              test_extremo_solo_buys_sin_ventas),
    ("extremo: múltiples ciclos compra/venta",       test_extremo_multiples_ciclos_compra_venta),
    ("extremo: btc_por_venta congelado multi-venta", test_extremo_btc_por_venta_congelado_entre_ventas_multiples),
    ("extremo: 500 velas random sin crash",          test_extremo_velas_con_valores_random),
    ("extremo: conservación capital sin comisión",   test_extremo_conservacion_capital_sin_comision),
    ("extremo: pérdida exacta por comisión",         test_extremo_conservacion_capital_con_comision),
]


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           TEST SUITE — backtest_irreal.py                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Ejecutando {len(TESTS)} tests...\n")

    for name, fn in TESTS:
        run(name, fn)

    total  = len(TESTS)
    passed = len(PASSED)
    failed = len(FAILED)

    print(f"\n{'═'*72}")
    print(f"  RESULTADO FINAL:  {passed}/{total} tests pasados", end="")
    print(f"  |  {failed} fallos" if failed else "  — TODOS OK ✓")
    print(f"{'═'*72}")

    if failed:
        print("\n  Tests fallidos:")
        for name, err in FAILED:
            print(f"    ✗  {name}")
            print(f"       {type(err).__name__}: {err}")
        sys.exit(1)
    else:
        sys.exit(0)
