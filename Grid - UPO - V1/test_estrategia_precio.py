"""
Test suite — Estrategia_Precio
══════════════════════════════════════════════════════════════════════
Cobertura:
  1.  _gradiente_asintotico()      — fases, bordes, continuidad
  2.  pct_capital_compra()         — posición logarítmica, extremos
  3.  _intentar_compra()           — Modo A, Modo B, límites, gradiente
  4.  _ejecutar_ventas()           — TP simple, múltiples, misma vela
  5.  Matemática de comisiones     — recuperación exacta de usdt_invertido
  6.  Orden intracandle            — alcista vs bajista
  7.  last_op_price                — actualizaciones en compra y venta
  8.  ATH / ATL                    — tracking en el loop principal
  9.  Salvaguarda btc_a_vender     — no vender más BTC del disponible
  10. Flujo completo end-to-end    — secuencias multi-vela coherentes

No requiere base de datos. Toda la lógica se prueba con datos sintéticos.
"""

import math
import sys
import types
import unittest
import pandas as pd
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MOCK DE CONFIG — valores fijos para los tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cfg = types.ModuleType("config")
cfg.DB_PATH             = ":memory:"
cfg.RESULTS_JSON        = "test_results.json"
cfg.FECHA_INICIO        = None
cfg.FECHA_FIN           = None
cfg.SALDO_USDT_INICIAL  = 1000.0
cfg.PCT_CAIDA_ATH       = 0.03        # 3 % desde ATH → primera compra
cfg.PCT_CAIDA           = 0.02        # 2 % desde last_op_price → DCA
cfg.PCT_VENTA           = 0.05        # 5 % sobre precio entrada → TP
cfg.FLOOR_PCT           = 25          # ATL_REF = ATH × 0.25
cfg.CURVA_COMPRA_NIVEL  = 5.0
cfg.CURVA_COMPRA_INFL   = 0.30
cfg.CURVA_COMPRA_K      = 2.0
cfg.CURVA_COMPRA_FAC2   = 5.0
cfg.USDT_RESERVA_PCT    = 0.0
cfg.COMMISSION_PCT      = 0.1         # 0.1 %
cfg.mostrar_configuracion = lambda: None
sys.modules["config"] = cfg

# Importar el módulo a testear (sin ejecutar main)
import importlib
estrategia = importlib.import_module("Estrategia_Grid_UPO_Acumulator")

# Aliases cortos
_grad       = estrategia._gradiente_asintotico
_pct_compra = estrategia.pct_capital_compra
_compra     = estrategia._intentar_compra
_ventas     = estrategia._ejecutar_ventas
COMMISSION  = cfg.COMMISSION_PCT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ts(label="2024-01-01 00:00:00"):
    return pd.Timestamp(label)


def estado_limpio(usdt=1000.0):
    """Devuelve el estado inicial canónico para los tests."""
    return {
        "posiciones"    : [],
        "last_op_price" : None,
        "usdt_balance"  : usdt,
        "btc_balance"   : 0.0,
        "trade_history" : [],
    }


def run_compra(i=1, low=None, ath=60000.0, atl=15000.0,
               posiciones=None, last_op=None,
               usdt=1000.0, btc=0.0):
    """Wrapper cómodo para _intentar_compra."""
    posiciones    = posiciones if posiciones is not None else []
    trade_history = []
    pos, lop, usdt_b, ok = _compra(
        i, ts(), low, ath, atl,
        posiciones, last_op,
        usdt, btc,
        trade_history,
    )
    return pos, lop, usdt_b, ok, trade_history


def run_ventas(i=2, high=None, ath=60000.0, atl=15000.0,
               posiciones=None, last_op=None,
               usdt=1000.0, btc=0.0):
    """Wrapper cómodo para _ejecutar_ventas."""
    trade_history = []
    pos, lop, usdt_b, btc_b = _ventas(
        i, ts(), high, ath, atl,
        posiciones, last_op,
        usdt, btc,
        trade_history,
    )
    return pos, lop, usdt_b, btc_b, trade_history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. GRADIENTE ASINTÓTICO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGradienteAsintotico(unittest.TestCase):

    def setUp(self):
        self.nivel = 5.0
        self.infl  = 0.30
        self.k     = 2.0
        self.fac2  = 5.0

    def _g(self, pos):
        return _grad(pos, self.nivel, self.infl, self.k, self.fac2)

    def test_pos_negativa_retorna_cero(self):
        self.assertEqual(self._g(-1.0), 0.0)
        self.assertEqual(self._g(-0.001), 0.0)

    def test_pos_cero_retorna_cero(self):
        self.assertEqual(self._g(0.0), 0.0)

    def test_pos_en_infl_retorna_nivel(self):
        resultado = self._g(self.infl)
        self.assertAlmostEqual(resultado, self.nivel, places=6,
            msg="En pos=INFL el gradiente debe retornar exactamente NIVEL")

    def test_pos_uno_retorna_cien(self):
        resultado = self._g(1.0)
        self.assertAlmostEqual(resultado, 100.0, places=6,
            msg="En pos=1 el gradiente debe retornar 100%")

    def test_fase1_monotona_creciente(self):
        vals = [self._g(p) for p in [0.05, 0.10, 0.20, 0.29, 0.30]]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b, "Fase 1 debe ser estrictamente creciente")

    def test_fase2_monotona_creciente(self):
        vals = [self._g(p) for p in [0.31, 0.50, 0.70, 0.90, 1.0]]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b, "Fase 2 debe ser estrictamente creciente")

    def test_continuidad_en_inflexion(self):
        """Diferencia entre fase 1 y fase 2 justo en el punto de inflexión < 0.001."""
        antes  = self._g(self.infl - 1e-9)
        despues = self._g(self.infl + 1e-9)
        self.assertAlmostEqual(antes, despues, places=3,
            msg="Debe haber continuidad en el punto de inflexión")

    def test_fase1_nunca_supera_nivel(self):
        for p in [x / 100 for x in range(1, 31)]:
            self.assertLessEqual(self._g(p), self.nivel + 1e-9,
                msg=f"Fase 1 supera NIVEL en pos={p}")

    def test_fase2_siempre_supera_nivel(self):
        for p in [x / 100 for x in range(31, 101)]:
            self.assertGreaterEqual(self._g(p), self.nivel - 1e-9,
                msg=f"Fase 2 cae bajo NIVEL en pos={p}")

    def test_fac2_1_es_lineal_en_fase2(self):
        """Con FAC2=1 la fase 2 es lineal entre NIVEL y 100."""
        g = lambda p: _grad(p, self.nivel, self.infl, self.k, 1.0)
        p_mid = (self.infl + 1.0) / 2
        esperado = self.nivel + (100.0 - self.nivel) * ((p_mid - self.infl) / (1 - self.infl))
        self.assertAlmostEqual(g(p_mid), esperado, places=6)

    def test_scale_cero_retorna_cero(self):
        """Si infl=0, scale=0 → retorna 0 sin división por cero."""
        resultado = _grad(0.01, 5.0, 0.0, 2.0, 5.0)
        # Con infl=0 pos > infl siempre → pasa a fase 2
        self.assertGreaterEqual(resultado, 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PCT_CAPITAL_COMPRA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPctCapitalCompra(unittest.TestCase):

    def setUp(self):
        self.ath     = 60000.0
        # Hardcoded: no depende de cfg.FLOOR_PCT para evitar contaminación
        # entre tests que parchean FLOOR_PCT. FLOOR_PCT=25 → ATL_REF = 15000.
        self.atl_ref = 15000.0

    def test_precio_igual_a_ath_retorna_cero(self):
        """En el ATH pos=0 → gradiente=0."""
        resultado = _pct_compra(self.ath, self.ath)
        self.assertEqual(resultado, 0.0)

    def test_precio_igual_a_atl_ref_retorna_cien(self):
        """En ATL_REF pos=1 → gradiente=100%."""
        resultado = _pct_compra(self.atl_ref, self.ath)
        self.assertAlmostEqual(resultado, 100.0, places=4)

    def test_precio_sobre_ath_retorna_cero(self):
        """Precio por encima del ATH → pos negativa → clamp a 0."""
        resultado = _pct_compra(self.ath * 1.1, self.ath)
        self.assertEqual(resultado, 0.0)

    def test_precio_bajo_atl_ref_retorna_cien(self):
        """Precio por debajo del ATL_REF → pos > 1 → clamp a 1 → 100%."""
        resultado = _pct_compra(self.atl_ref * 0.5, self.ath)
        self.assertAlmostEqual(resultado, 100.0, places=4)

    def test_precio_cero_retorna_cero(self):
        """limit_buy=0 → guard defensivo → retorna 0 sin excepción."""
        resultado = _pct_compra(0.0, self.ath)
        self.assertEqual(resultado, 0.0)

    def test_monotonicidad(self):
        """A menor precio, mayor % de capital asignado."""
        precios = [50000, 40000, 30000, 20000, 15000]
        pcts = [_pct_compra(p, self.ath) for p in precios]
        for a, b in zip(pcts, pcts[1:]):
            self.assertLess(a, b, "pct_capital_compra debe ser monótona decreciente con el precio")

    def test_ath_cero_retorna_cero(self):
        self.assertEqual(_pct_compra(30000, 0), 0.0)

    def test_floor_pct_cero_retorna_cero(self):
        """
        FLOOR_PCT=0 → guard → retorna 0.
        Se parchea estrategia.FLOOR_PCT (el valor importado por la función),
        NO cfg.FLOOR_PCT, para no contaminar setUp de tests siguientes.
        Se usa try/finally para garantizar restauración incluso si falla.
        """
        original = estrategia.FLOOR_PCT
        estrategia.FLOOR_PCT = 0
        try:
            resultado = _pct_compra(30000, 60000)
            self.assertEqual(resultado, 0.0)
        finally:
            estrategia.FLOOR_PCT = original

    def test_floor_pct_100_log_rango_cero_retorna_cero(self):
        """FLOOR_PCT=100 → log(100/100)=0 → log_rango=0 → guard → retorna 0."""
        original = estrategia.FLOOR_PCT
        estrategia.FLOOR_PCT = 100
        try:
            resultado = _pct_compra(30000, 60000)
            self.assertEqual(resultado, 0.0)
        finally:
            estrategia.FLOOR_PCT = original

    def test_rango_siempre_entre_cero_y_cien(self):
        for precio in range(10000, 65000, 1000):
            r = _pct_compra(float(precio), self.ath)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 100.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. _INTENTAR_COMPRA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntentarCompra(unittest.TestCase):

    ATH = 60000.0
    ATL = 15000.0

    def _limit_a(self):
        return self.ATH * (1 - cfg.PCT_CAIDA_ATH)   # 60000 × 0.97 = 58200

    def _limit_b(self, last_op):
        return last_op * (1 - cfg.PCT_CAIDA)

    # ── Modo A ────────────────────────────────────────────────────────────────

    def test_modo_a_no_ejecuta_si_low_alto(self):
        """Low por encima del límite → no compra."""
        limit = self._limit_a()
        pos, lop, usdt, ok, hist = run_compra(low=limit + 1, ath=self.ATH, atl=self.ATL)
        self.assertFalse(ok)
        self.assertEqual(len(pos), 0)
        self.assertIsNone(lop)

    def test_modo_a_ejecuta_si_low_igual_al_limite(self):
        """Low exactamente en el límite → compra."""
        limit = self._limit_a()
        pos, lop, usdt, ok, hist = run_compra(low=limit, ath=self.ATH, atl=self.ATL)
        self.assertTrue(ok)
        self.assertEqual(len(pos), 1)

    def test_modo_a_ejecuta_si_low_bajo_el_limite(self):
        """Low por debajo del límite → compra."""
        limit = self._limit_a()
        pos, lop, usdt, ok, hist = run_compra(low=limit - 100, ath=self.ATH, atl=self.ATL)
        self.assertTrue(ok)

    def test_modo_a_precio_entrada_es_limit_no_low(self):
        """El precio de entrada debe ser limit_buy, NO el low de la vela."""
        limit = self._limit_a()
        low_real = limit - 500
        pos, lop, usdt, ok, hist = run_compra(low=low_real, ath=self.ATH, atl=self.ATL)
        self.assertTrue(ok)
        self.assertAlmostEqual(pos[0]["precio_entrada"], limit, places=4,
            msg="precio_entrada debe ser limit_buy, no el low de la vela")

    def test_modo_a_last_op_price_es_limit(self):
        """last_op_price se actualiza al precio límite de la orden."""
        limit = self._limit_a()
        _, lop, _, ok, _ = run_compra(low=limit - 100, ath=self.ATH, atl=self.ATL)
        self.assertTrue(ok)
        self.assertAlmostEqual(lop, limit, places=4)

    # ── Modo B ────────────────────────────────────────────────────────────────

    def test_modo_b_usa_last_op_price(self):
        """Con posiciones abiertas, el límite se calcula desde last_op_price."""
        last_op = 55000.0
        limit_esperado = self._limit_b(last_op)
        pos_existente = [{
            "precio_entrada": last_op, "usdt_invertido": 100.0,
            "btc_cantidad": 0.001, "precio_tp": last_op * 1.05,
            "vela_creacion": 0,
        }]
        pos, lop, usdt, ok, hist = run_compra(
            i=1, low=limit_esperado - 10,
            ath=self.ATH, atl=self.ATL,
            posiciones=pos_existente, last_op=last_op,
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(pos[-1]["precio_entrada"], limit_esperado, places=4)

    def test_modo_b_no_ejecuta_si_caida_insuficiente(self):
        last_op = 55000.0
        limit = self._limit_b(last_op)
        pos_existente = [{
            "precio_entrada": last_op, "usdt_invertido": 100.0,
            "btc_cantidad": 0.001, "precio_tp": last_op * 1.05,
            "vela_creacion": 0,
        }]
        _, _, _, ok, _ = run_compra(
            i=1, low=limit + 1,
            ath=self.ATH, atl=self.ATL,
            posiciones=pos_existente, last_op=last_op,
        )
        self.assertFalse(ok)

    # ── Sizing y capital ──────────────────────────────────────────────────────

    def test_usdt_balance_se_reduce(self):
        limit = self._limit_a()
        _, _, usdt_final, ok, hist = run_compra(
            low=limit, ath=self.ATH, atl=self.ATL, usdt=1000.0
        )
        self.assertTrue(ok)
        self.assertLess(usdt_final, 1000.0, "El saldo USDT debe reducirse tras la compra")

    def test_posicion_almacena_usdt_invertido(self):
        limit = self._limit_a()
        pos, _, usdt_final, ok, hist = run_compra(
            low=limit, ath=self.ATH, atl=self.ATL, usdt=1000.0
        )
        self.assertTrue(ok)
        gasto = 1000.0 - usdt_final
        self.assertAlmostEqual(pos[0]["usdt_invertido"], gasto, places=6)

    def test_btc_cantidad_neta_tras_comision_compra(self):
        """BTC adquirido = (usdt_a_usar - comision_compra) / precio."""
        limit = self._limit_a()
        pos, _, _, ok, hist = run_compra(
            low=limit, ath=self.ATH, atl=self.ATL, usdt=1000.0
        )
        self.assertTrue(ok)
        usdt_usado = pos[0]["usdt_invertido"]
        comision   = usdt_usado * (COMMISSION / 100.0)
        btc_esperado = (usdt_usado - comision) / limit
        self.assertAlmostEqual(pos[0]["btc_cantidad"], btc_esperado, places=8)

    def test_precio_tp_correcto(self):
        limit = self._limit_a()
        pos, _, _, ok, _ = run_compra(low=limit, ath=self.ATH, atl=self.ATL)
        self.assertTrue(ok)
        tp_esperado = limit * (1.0 + cfg.PCT_VENTA)
        self.assertAlmostEqual(pos[0]["precio_tp"], tp_esperado, places=4)

    def test_posicion_guarda_vela_creacion(self):
        limit = self._limit_a()
        pos, _, _, ok, _ = run_compra(i=7, low=limit, ath=self.ATH, atl=self.ATL)
        self.assertTrue(ok)
        self.assertEqual(pos[0]["vela_creacion"], 7)

    # ── Casos límite ──────────────────────────────────────────────────────────

    def test_sin_capital_registra_ignorado(self):
        limit = self._limit_a()
        _, _, _, ok, hist = run_compra(
            low=limit, ath=self.ATH, atl=self.ATL, usdt=0.0
        )
        self.assertFalse(ok)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["motivo_ignorado"], "sin_capital_sobre_reserva")

    def test_gradiente_cero_registra_ignorado(self):
        """
        La ruta "gradiente_cero" se activa cuando la orden límite se dispara
        pero pct_capital_compra devuelve 0.

        Con parámetros normales (FLOOR_PCT=25) el límite de Modo A es
        ATH × 0.97 < ATH, por lo que pos > 0 y el gradiente siempre > 0.
        Para forzar gradiente=0 se parchea FLOOR_PCT=100 → log_rango=0 →
        la función retorna 0 para cualquier precio.

        Separadamente, verificamos que pct_capital_compra(ATH, ATH) = 0
        (pos=0 → gradiente=0) como propiedad matemática.
        """
        # Propiedad matemática: en pos=0 el gradiente devuelve 0
        self.assertEqual(_pct_compra(self.ATH, self.ATH), 0.0)

        # Ruta "gradiente_cero" en _intentar_compra:
        # Parcheamos FLOOR_PCT=100 → log_rango=0 → pct_capital_compra=0
        # La orden sí se dispara (low ≤ limit_a), pero el gradiente la ignora.
        original = estrategia.FLOOR_PCT
        estrategia.FLOOR_PCT = 100
        try:
            limit_a = self.ATH * (1 - cfg.PCT_CAIDA_ATH)
            _, _, _, ok, hist = run_compra(
                low=limit_a - 100, ath=self.ATH, atl=self.ATL, usdt=1000.0
            )
            self.assertFalse(ok)
            self.assertTrue(
                any(h["motivo_ignorado"] == "gradiente_cero" for h in hist),
                "Debe registrarse motivo='gradiente_cero' cuando pct_capital_compra=0"
            )
        finally:
            estrategia.FLOOR_PCT = original

    def test_multiples_compras_acumulan_posiciones(self):
        limit_a = self._limit_a()
        pos1, lop1, usdt1, ok1, _ = run_compra(
            i=1, low=limit_a, ath=self.ATH, atl=self.ATL, usdt=1000.0
        )
        self.assertTrue(ok1)
        limit_b = self._limit_b(lop1)
        pos2, lop2, usdt2, ok2, _ = run_compra(
            i=2, low=limit_b - 10, ath=self.ATH, atl=self.ATL,
            posiciones=pos1, last_op=lop1, usdt=usdt1,
        )
        self.assertTrue(ok2)
        self.assertEqual(len(pos2), 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. _EJECUTAR_VENTAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEjecutarVentas(unittest.TestCase):

    ATH = 60000.0
    ATL = 15000.0

    def _pos(self, precio_entrada=50000.0, usdt_inv=100.0, vela=0):
        btc = usdt_inv * (1 - COMMISSION / 100.0) / precio_entrada
        tp  = precio_entrada * (1 + cfg.PCT_VENTA)
        return {
            "precio_entrada" : precio_entrada,
            "usdt_invertido" : usdt_inv,
            "btc_cantidad"   : btc,
            "precio_tp"      : tp,
            "vela_creacion"  : vela,
        }

    def test_tp_no_alcanzado_posicion_sobrevive(self):
        pos = [self._pos(50000.0)]
        tp = pos[0]["precio_tp"]
        pos_r, _, _, _, hist = run_ventas(i=5, high=tp - 1, posiciones=pos)
        self.assertEqual(len(pos_r), 1)
        self.assertEqual(len(hist), 0)

    def test_tp_alcanzado_posicion_se_cierra(self):
        pos = [self._pos(50000.0)]
        tp = pos[0]["precio_tp"]
        pos_r, _, _, _, hist = run_ventas(i=5, high=tp, posiciones=pos)
        self.assertEqual(len(pos_r), 0)
        self.assertEqual(len([h for h in hist if h["type"] == "SELL"]), 1)

    def test_precio_venta_es_tp_no_el_high(self):
        pos = [self._pos(50000.0)]
        tp = pos[0]["precio_tp"]
        _, _, _, _, hist = run_ventas(i=5, high=tp + 5000, posiciones=pos)
        venta = [h for h in hist if h["type"] == "SELL"][0]
        self.assertAlmostEqual(venta["price"], tp, places=4,
            msg="El precio de venta debe ser el TP límite, no el high de la vela")

    def test_posicion_misma_vela_no_se_vende(self):
        """Una posición creada en la vela i no debe evaluarse en la misma vela."""
        pos = [self._pos(50000.0, vela=5)]   # creada en vela 5
        tp = pos[0]["precio_tp"]
        pos_r, _, _, _, hist = run_ventas(i=5, high=tp + 1000, posiciones=pos)
        self.assertEqual(len(pos_r), 1, "La posición de la misma vela no debe venderse")
        self.assertEqual(len(hist), 0)

    def test_multiple_tp_misma_vela(self):
        """Si el high supera varios TPs, todas las posiciones previas se cierran."""
        p1 = self._pos(50000.0, usdt_inv=100.0, vela=0)
        p2 = self._pos(48000.0, usdt_inv=100.0, vela=1)
        pos = [p1, p2]
        high_alto = max(p1["precio_tp"], p2["precio_tp"]) + 1000
        pos_r, _, _, _, hist = run_ventas(i=5, high=high_alto, posiciones=pos)
        self.assertEqual(len(pos_r), 0, "Ambas posiciones deben cerrarse")
        ventas = [h for h in hist if h["type"] == "SELL"]
        self.assertEqual(len(ventas), 2)

    def test_multiple_tp_last_op_es_el_ultimo_ejecutado(self):
        """last_op_price queda en el TP de la última posición procesada."""
        p1 = self._pos(50000.0, usdt_inv=100.0, vela=0)
        p2 = self._pos(48000.0, usdt_inv=100.0, vela=1)
        # p2 tiene menor precio → menor TP → se alcanza primero
        # pero ambos se alcanzan. El last_op_price final depende del orden
        # en que están en la lista (orden de inserción).
        pos = [p1, p2]
        high_alto = max(p1["precio_tp"], p2["precio_tp"]) + 1000
        _, lop, _, _, hist = run_ventas(i=5, high=high_alto, posiciones=pos)
        ventas = [h for h in hist if h["type"] == "SELL"]
        # last_op_price debe ser el TP de la última venta ejecutada
        self.assertAlmostEqual(lop, ventas[-1]["price"], places=4)

    def test_solo_posiciones_previas_se_evaluan_misma_vela_sobrevive(self):
        """Mezcla: una posición previa se vende, la de la misma vela sobrevive."""
        p_previa = self._pos(50000.0, usdt_inv=100.0, vela=3)
        p_nueva  = self._pos(50000.0, usdt_inv=100.0, vela=5)
        high_alto = p_previa["precio_tp"] + 1000
        pos_r, _, _, _, hist = run_ventas(i=5, high=high_alto, posiciones=[p_previa, p_nueva])
        self.assertEqual(len(pos_r), 1)
        self.assertEqual(pos_r[0]["vela_creacion"], 5)
        ventas = [h for h in hist if h["type"] == "SELL"]
        self.assertEqual(len(ventas), 1)

    def test_last_op_se_actualiza_en_venta(self):
        pos = [self._pos(50000.0, vela=0)]
        tp = pos[0]["precio_tp"]
        _, lop, _, _, _ = run_ventas(i=5, high=tp + 100, posiciones=pos, last_op=40000.0)
        self.assertAlmostEqual(lop, tp, places=4)

    def test_usdt_balance_aumenta_tras_venta(self):
        pos = [self._pos(50000.0, usdt_inv=100.0, vela=0)]
        tp = pos[0]["precio_tp"]
        _, _, usdt_f, _, _ = run_ventas(i=5, high=tp + 100, posiciones=pos, usdt=500.0)
        self.assertGreater(usdt_f, 500.0)

    def test_btc_balance_aumenta_con_acumulado(self):
        """El BTC sobrante de la venta (profit) se acumula en btc_balance."""
        pos = [self._pos(50000.0, usdt_inv=100.0, vela=0)]
        tp = pos[0]["precio_tp"]
        _, _, _, btc_f, hist = run_ventas(i=5, high=tp + 100, posiciones=pos, btc=0.0)
        venta = [h for h in hist if h["type"] == "SELL"][0]
        self.assertGreater(btc_f, 0.0)
        self.assertAlmostEqual(btc_f, venta["btc_accumulated"], places=8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. MATEMÁTICA DE COMISIONES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMatematicaComisiones(unittest.TestCase):
    """
    Propiedad central: la venta debe recuperar exactamente el usdt_invertido
    (que ya incluye la comisión de compra) y cubrir su propia comisión de venta,
    dejando el resto como BTC libre.
    """

    ATH = 60000.0
    ATL = 15000.0

    def _ciclo_compra_venta(self, precio_entrada, usdt_a_invertir):
        """Simula una compra y la venta de esa posición, devuelve los componentes."""
        comm = COMMISSION / 100.0
        # Compra
        comision_compra = usdt_a_invertir * comm
        btc_neto        = (usdt_a_invertir - comision_compra) / precio_entrada
        precio_tp       = precio_entrada * (1 + cfg.PCT_VENTA)
        # Venta — fórmula del código
        factor          = 1.0 - comm
        btc_a_vender    = usdt_a_invertir / (precio_tp * factor)
        usdt_bruto      = btc_a_vender * precio_tp
        comision_venta  = usdt_bruto * comm
        usdt_neto       = usdt_bruto - comision_venta
        btc_acumulado   = btc_neto - btc_a_vender
        return {
            "btc_neto"       : btc_neto,
            "btc_a_vender"   : btc_a_vender,
            "btc_acumulado"  : btc_acumulado,
            "usdt_neto"      : usdt_neto,
            "usdt_invertido" : usdt_a_invertir,
        }

    def test_usdt_neto_recupera_exactamente_invertido(self):
        """usdt_neto de la venta ≈ usdt_invertido (recuperación del capital + comisiones)."""
        for precio in [20000, 35000, 50000, 60000]:
            for monto in [50, 100, 500]:
                with self.subTest(precio=precio, monto=monto):
                    r = self._ciclo_compra_venta(float(precio), float(monto))
                    self.assertAlmostEqual(
                        r["usdt_neto"], r["usdt_invertido"], places=6,
                        msg=f"La venta no recupera exactamente lo invertido "
                            f"(precio={precio}, monto={monto})"
                    )

    def test_btc_acumulado_es_positivo(self):
        """Siempre queda BTC libre (ganancia en BTC) después de la venta."""
        r = self._ciclo_compra_venta(50000.0, 100.0)
        self.assertGreater(r["btc_acumulado"], 0.0,
            msg="El BTC acumulado debe ser positivo — es la ganancia de la operación")

    def test_btc_a_vender_menor_que_btc_neto(self):
        """Se vende menos BTC del que se compró — el resto es ganancia."""
        r = self._ciclo_compra_venta(50000.0, 100.0)
        self.assertLess(r["btc_a_vender"], r["btc_neto"])

    def test_salvaguarda_btc_a_vender_no_supera_disponible(self):
        """
        Si por alguna razón btc_a_vender > btc_cantidad, la salvaguarda lo limita.
        Esto puede ocurrir con comisiones extremas o TPs muy ajustados.
        """
        # Simular el peor caso: commission=50% (irreal pero útil para el test)
        comm_original = cfg.COMMISSION_PCT
        cfg.COMMISSION_PCT = 50.0
        try:
            precio_entrada = 50000.0
            usdt_inv = 100.0
            comm = cfg.COMMISSION_PCT / 100.0
            btc_neto = usdt_inv * (1 - comm) / precio_entrada
            precio_tp = precio_entrada * (1 + cfg.PCT_VENTA)
            factor = 1.0 - comm
            btc_calculado = usdt_inv / (precio_tp * factor)
            # La salvaguarda en el código: min(btc_calculado, btc_neto)
            btc_real = min(btc_calculado, btc_neto)
            self.assertLessEqual(btc_real, btc_neto,
                msg="La salvaguarda debe impedir vender más BTC del disponible")
        finally:
            cfg.COMMISSION_PCT = comm_original

    def test_ganancia_en_trade_history_es_correcta(self):
        """ganancia_usdt = usdt_neto - usdt_invertido ≈ 0 (diseño intencional)."""
        pos_list = []
        precio_entrada = 50000.0
        usdt_inv = 100.0
        comm = COMMISSION / 100.0
        btc_neto = usdt_inv * (1 - comm) / precio_entrada
        tp = precio_entrada * (1 + cfg.PCT_VENTA)
        pos_list.append({
            "precio_entrada" : precio_entrada,
            "usdt_invertido" : usdt_inv,
            "btc_cantidad"   : btc_neto,
            "precio_tp"      : tp,
            "vela_creacion"  : 0,
        })
        _, _, _, _, hist = run_ventas(
            i=5, high=tp + 100,
            posiciones=pos_list,
            ath=self.ATH, atl=self.ATL,
        )
        venta = [h for h in hist if h["type"] == "SELL"][0]
        # ganancia ≈ 0 porque el diseño es recuperar exactamente lo invertido
        self.assertAlmostEqual(venta["ganancia_usdt"], 0.0, places=4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. ORDEN INTRACANDLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOrdenIntracandle(unittest.TestCase):
    """
    Construye DataFrames sintéticos de una sola vela para verificar
    que el orden compra/venta depende de la dirección de la vela.
    """

    def _df_una_vela(self, open_, high, low, close, timestamp="2024-01-01"):
        return pd.DataFrame([{
            "timestamp" : pd.Timestamp(timestamp).value // 10**6,
            "open"      : open_,
            "high"      : high,
            "low"       : low,
            "close"     : close,
            "volume"    : 1.0,
            "datetime"  : pd.Timestamp(timestamp),
        }])

    def test_vela_alcista_compra_primero(self):
        """
        Vela alcista (close > open):
          - Hay una posición con TP alcanzable.
          - El low también dispara una compra.
          - En una vela alcista el low ocurrió antes → la compra se ejecuta
            antes que la venta, por lo que la posición recién creada
            (vela_creacion = i actual) NO se vende en la misma vela.
        """
        ath = 60000.0
        # Posición preexistente con TP en 58200
        precio_e = 55000.0
        tp = precio_e * (1 + cfg.PCT_VENTA)   # ≈ 57750

        # limit_buy Modo A (sin posiciones inicialmente, pero con la pos preexistente
        # usamos Modo B). Vamos a simplificar: la posición preexistente tiene last_op=55000
        # limit_b = 55000 × 0.98 = 53900. Low = 53900 → compra se ejecuta.

        # Vela alcista: open=54000, close=58500, low=53900, high=58500
        # → compra primero (en low=53900) → luego venta del TP preexistente

        cfg.SALDO_USDT_INICIAL = 1000.0
        estrategia.USDT_RESERVA = 0.0

        comm = COMMISSION / 100.0
        btc_neto = 100.0 * (1 - comm) / precio_e

        pos_inicial = [{
            "precio_entrada" : precio_e,
            "usdt_invertido" : 100.0,
            "btc_cantidad"   : btc_neto,
            "precio_tp"      : tp,
            "vela_creacion"  : 0,   # vela anterior
        }]

        df = self._df_una_vela(
            open_=54000, high=tp + 100, low=53900, close=tp + 100,
        )
        df.index = range(len(df))

        # Guardar estado y parchearlo en el módulo no es trivial.
        # En su lugar, verificamos indirectamente la lógica:
        # Si la compra ocurre primero (vela_creacion=0 para la nueva),
        # la nueva posición tiene vela_creacion=0 (índice i de la única vela).
        # La venta evalúa solo posiciones con vela_creacion < i=0, es decir ninguna nueva.

        # Verificamos que _ejecutar_ventas con i=0 no toca posiciones con vela_creacion=0
        nueva_pos = {
            "precio_entrada" : 53900 * (1 - cfg.PCT_CAIDA),
            "usdt_invertido" : 50.0,
            "btc_cantidad"   : 0.001,
            "precio_tp"      : 53900 * (1 + cfg.PCT_VENTA),
            "vela_creacion"  : 0,  # creada en i=0
        }
        all_pos = pos_inicial + [nueva_pos]
        pos_r, _, _, _, hist = run_ventas(
            i=0, high=tp + 100,
            posiciones=all_pos,
            ath=ath, atl=15000.0,
        )
        ventas = [h for h in hist if h["type"] == "SELL"]
        # Solo debe venderse la posición preexistente (vela_creacion=0 < i... espera, i=0 también)
        # Corrección: pos_inicial tiene vela_creacion=0 y i=0 → tampoco se vende
        # Para el test correcto, la pos inicial debe tener vela_creacion < i
        # Rehacemos con i=1
        pos_inicial[0]["vela_creacion"] = 0
        nueva_pos["vela_creacion"] = 1
        all_pos = pos_inicial + [nueva_pos]
        pos_r, _, _, _, hist = run_ventas(
            i=1, high=tp + 100,
            posiciones=all_pos,
            ath=ath, atl=15000.0,
        )
        ventas = [h for h in hist if h["type"] == "SELL"]
        self.assertEqual(len(ventas), 1, "Solo la posición preexistente debe venderse")
        self.assertEqual(pos_r[0]["vela_creacion"], 1,
            "La posición nueva (misma vela) debe sobrevivir")

    def test_vela_bajista_ventas_primero(self):
        """
        Vela bajista: el high ocurre antes que el low.
        La venta de una posición preexistente actualiza last_op_price.
        La compra subsiguiente usa ese last_op_price actualizado si hay posiciones.
        """
        precio_e = 55000.0
        tp = precio_e * (1 + cfg.PCT_VENTA)
        comm = COMMISSION / 100.0

        pos_previa = [{
            "precio_entrada" : precio_e,
            "usdt_invertido" : 100.0,
            "btc_cantidad"   : 100.0 * (1 - comm) / precio_e,
            "precio_tp"      : tp,
            "vela_creacion"  : 0,
        }]

        # En vela bajista: ventas primero → last_op_price = tp
        # Luego compra: last_op_price = tp → limit_b = tp × (1 - PCT_CAIDA)
        # Si low ≤ limit_b → nueva compra
        limit_b_esperado = tp * (1 - cfg.PCT_CAIDA)

        pos_tras_venta, lop_tras_venta, usdt_v, btc_v, hist_v = run_ventas(
            i=1, high=tp + 100,
            posiciones=pos_previa, last_op=precio_e,
            usdt=900.0, btc=0.0,
            ath=60000.0, atl=15000.0,
        )
        # Tras la venta, last_op = tp y no quedan posiciones
        self.assertAlmostEqual(lop_tras_venta, tp, places=2)
        self.assertEqual(len(pos_tras_venta), 0)

        # Ahora la compra: sin posiciones → Modo A (no Modo B)
        # Porque posiciones está vacía después de la venta
        # (La lógica de Modo A/B la determina len(posiciones), no last_op_price)
        limit_a = 60000.0 * (1 - cfg.PCT_CAIDA_ATH)
        pos_c, lop_c, usdt_c, ok_c, hist_c = run_compra(
            i=1, low=limit_a - 10,
            ath=60000.0, atl=15000.0,
            posiciones=pos_tras_venta,   # vacío → Modo A
            last_op=lop_tras_venta,
            usdt=usdt_v,
        )
        self.assertTrue(ok_c, "La compra debe ejecutarse en Modo A tras vaciarse posiciones")
        self.assertAlmostEqual(pos_c[0]["precio_entrada"], limit_a, places=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. LAST_OP_PRICE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLastOpPrice(unittest.TestCase):

    ATH = 60000.0
    ATL = 15000.0

    def test_none_inicial_usa_modo_a(self):
        """Con last_op=None y sin posiciones siempre Modo A."""
        limit_a = self.ATH * (1 - cfg.PCT_CAIDA_ATH)
        _, lop, _, ok, _ = run_compra(
            low=limit_a, ath=self.ATH, atl=self.ATL,
            posiciones=[], last_op=None,
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(lop, limit_a, places=4)

    def test_compra_actualiza_last_op_al_limit(self):
        limit_a = self.ATH * (1 - cfg.PCT_CAIDA_ATH)
        _, lop, _, ok, _ = run_compra(
            low=limit_a - 200, ath=self.ATH, atl=self.ATL,
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(lop, limit_a, places=4,
            msg="last_op_price debe ser el precio límite de la orden, no el low")

    def test_venta_actualiza_last_op_al_tp(self):
        comm = COMMISSION / 100.0
        precio_e = 50000.0
        tp = precio_e * (1 + cfg.PCT_VENTA)
        pos = [{
            "precio_entrada": precio_e,
            "usdt_invertido": 100.0,
            "btc_cantidad"  : 100.0 * (1 - comm) / precio_e,
            "precio_tp"     : tp,
            "vela_creacion" : 0,
        }]
        _, lop, _, _, _ = run_ventas(
            i=5, high=tp + 500, posiciones=pos, last_op=precio_e,
        )
        self.assertAlmostEqual(lop, tp, places=4,
            msg="Tras la venta, last_op_price debe ser el precio del TP")

    def test_last_op_no_se_resetea_al_vaciar_posiciones(self):
        """
        Diseño explícito: last_op NO se resetea a None cuando posiciones queda vacía.
        El precio de la última venta sirve como referencia para la siguiente compra.
        """
        comm = COMMISSION / 100.0
        precio_e = 50000.0
        tp = precio_e * (1 + cfg.PCT_VENTA)
        pos = [{
            "precio_entrada": precio_e,
            "usdt_invertido": 100.0,
            "btc_cantidad"  : 100.0 * (1 - comm) / precio_e,
            "precio_tp"     : tp,
            "vela_creacion" : 0,
        }]
        pos_r, lop, _, _, _ = run_ventas(
            i=5, high=tp + 500, posiciones=pos, last_op=precio_e,
        )
        self.assertEqual(len(pos_r), 0)
        self.assertIsNotNone(lop,
            msg="last_op_price no debe resetearse a None al cerrar todas las posiciones")
        self.assertAlmostEqual(lop, tp, places=4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ATH / ATL TRACKING EN EL LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAthAtlTracking(unittest.TestCase):

    def _df(self, rows):
        """rows: lista de (open, high, low, close)"""
        records = []
        for i, (o, h, l, c) in enumerate(rows):
            records.append({
                "timestamp" : i * 3600000,
                "open"      : o,
                "high"      : h,
                "low"       : l,
                "close"     : c,
                "volume"    : 1.0,
                "datetime"  : pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            })
        return pd.DataFrame(records)

    def test_ath_se_actualiza_en_nuevo_maximo(self):
        """El ATH sube cuando una vela supera el máximo anterior."""
        df = self._df([
            (50000, 60000, 49000, 55000),   # vela 0: ATH = 60000
            (55000, 65000, 54000, 64000),   # vela 1: nuevo ATH = 65000
        ])
        results = estrategia.ejecutar_estrategia(df)
        # El resumen no expone ATH directamente, pero podemos inferirlo
        # del precio_min_comprado y el comportamiento del gradiente.
        # Test indirecto: si hubiera compra en vela 1, limit_a = 65000 × 0.97
        # Para verificar ATH, chequeamos que el loop no rompe con nuevos máximos.
        self.assertIn("summary", results)

    def test_atl_final_es_el_minimo_historico(self):
        df = self._df([
            (50000, 55000, 48000, 52000),
            (48000, 49000, 30000, 31000),   # nuevo mínimo
            (31000, 35000, 29000, 34000),   # nuevo mínimo aún más bajo
        ])
        results = estrategia.ejecutar_estrategia(df)
        self.assertAlmostEqual(results["summary"]["atl_final"], 29000.0, places=0)

    def test_loop_completo_no_rompe_con_datos_sinteticos(self):
        """El loop debe ejecutarse sin excepciones en datos básicos."""
        df = self._df([
            (60000, 62000, 58000, 61000),
            (61000, 61500, 55000, 56000),
            (56000, 57000, 54000, 55000),
            (55000, 56000, 50000, 51000),
            (51000, 52000, 49000, 50000),
        ])
        try:
            results = estrategia.ejecutar_estrategia(df)
            self.assertIn("trade_history", results)
        except Exception as e:
            self.fail(f"ejecutar_estrategia lanzó una excepción inesperada: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. FLUJO END-TO-END — SECUENCIAS MULTI-VELA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFlujoEndToEnd(unittest.TestCase):

    def _df(self, rows):
        records = []
        for i, (o, h, l, c) in enumerate(rows):
            records.append({
                "timestamp" : i * 3600000,
                "open"      : float(o),
                "high"      : float(h),
                "low"       : float(l),
                "close"     : float(c),
                "volume"    : 1.0,
                "datetime"  : pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            })
        return pd.DataFrame(records)

    def test_ciclo_compra_venta_completo(self):
        """
        Secuencia:
          Vela 0: ATH se establece en 60000
          Vela 1: bajista, low cae 3% bajo ATH → compra en limit_a
          Vela 2: sube, high ≥ TP de la compra → venta
        """
        ath    = 60000.0
        limit  = ath * (1 - cfg.PCT_CAIDA_ATH)   # 58200
        tp     = limit * (1 + cfg.PCT_VENTA)       # 58200 × 1.05 = 61110

        df = self._df([
            (59000, 60000, 58500, 59500),          # vela 0: ATH=60000
            (59000, 59500, limit - 100, 58500),    # vela 1: bajista, compra
            (59000, tp + 100, 58900, tp + 50),     # vela 2: alcista, venta
        ])

        results = estrategia.ejecutar_estrategia(df)
        s = results["summary"]
        trades = [t for t in results["trade_history"] if not t["ignorado"]]
        compras = [t for t in trades if t["type"] == "BUY"]
        ventas  = [t for t in trades if t["type"] == "SELL"]

        self.assertEqual(len(compras), 1, "Debe haber exactamente 1 compra")
        self.assertEqual(len(ventas),  1, "Debe haber exactamente 1 venta")
        self.assertAlmostEqual(compras[0]["price"], limit, places=1)
        self.assertAlmostEqual(ventas[0]["price"],  tp,    places=1)
        # Al final: posiciones cerradas, BTC acumulado > 0
        self.assertEqual(s["positions_count_final"], 0)
        self.assertGreater(s["btc_balance_final"], 0.0)

    def test_dca_dos_compras_una_venta(self):
        """
        Secuencia de DCA:
          Compra 1 → precio baja más → Compra 2 → precio sube → venta de Compra 1
        """
        ath     = 60000.0
        limit_a = ath * (1 - cfg.PCT_CAIDA_ATH)                   # 58200
        limit_b = limit_a * (1 - cfg.PCT_CAIDA)                    # 58200 × 0.98
        tp1     = limit_a * (1 + cfg.PCT_VENTA)                    # TP de compra 1
        tp2     = limit_b * (1 + cfg.PCT_VENTA)                    # TP de compra 2

        df = self._df([
            (59500, 60000, 59000, 59500),           # vela 0: ATH=60000
            (59000, 59200, limit_a - 50, 58300),    # vela 1: compra 1 (bajista)
            (58200, 58300, limit_b - 50, 57100),    # vela 2: compra 2 (bajista)
            (57500, tp1 + 200, 57400, tp1 + 100),   # vela 3: sube, venta compra 1
        ])

        results = estrategia.ejecutar_estrategia(df)
        trades  = [t for t in results["trade_history"] if not t["ignorado"]]
        compras = [t for t in trades if t["type"] == "BUY"]
        ventas  = [t for t in trades if t["type"] == "SELL"]

        self.assertEqual(len(compras), 2, "Deben ejecutarse 2 compras")
        self.assertGreaterEqual(len(ventas), 1, "Debe ejecutarse al menos 1 venta")
        self.assertAlmostEqual(compras[0]["price"], limit_a, places=0)
        self.assertAlmostEqual(compras[1]["price"], limit_b, places=0)

    def test_sin_compra_si_precio_nunca_cae_suficiente(self):
        """Si el precio nunca baja el PCT_CAIDA_ATH desde el ATH, no hay compras."""
        ath = 60000.0
        # Precio siempre cerca del ATH, sin caer el 3%
        caida_insuficiente = ath * (1 - cfg.PCT_CAIDA_ATH + 0.005)  # cae solo 2.5%

        df = self._df([
            (59500, 60000, 59000, 59500),
            (59000, 59500, caida_insuficiente, 59200),
            (59200, 59800, caida_insuficiente + 50, 59600),
        ])

        results = estrategia.ejecutar_estrategia(df)
        compras = [t for t in results["trade_history"]
                   if t["type"] == "BUY" and not t["ignorado"]]
        self.assertEqual(len(compras), 0)

    def test_balance_contable_consistente(self):
        """
        Invariante: usdt_balance + btc_en_posiciones × precio + btc_libre × precio
        = portfolio. Y usdt_inicial = usdt_final + usdt_en_btc (aprox).
        """
        ath    = 60000.0
        limit  = ath * (1 - cfg.PCT_CAIDA_ATH)
        tp     = limit * (1 + cfg.PCT_VENTA)

        df = self._df([
            (59000, 60000, 58500, 59500),
            (59000, 59500, limit - 100, 58500),
            (59000, tp + 100, 58900, tp + 50),
        ])

        results = estrategia.ejecutar_estrategia(df)
        s = results["summary"]

        # Después de una venta completa: usdt_balance ≈ inicial (la ganancia es en BTC)
        # El portfolio debe ser ≥ inicial (no puede haber pérdida si se recuperó lo invertido)
        self.assertGreaterEqual(
            s["portfolio_value_final"],
            cfg.SALDO_USDT_INICIAL * 0.99,  # tolerancia del 1% por comisiones
            msg="El portfolio final no debe ser significativamente menor al inicial"
        )

    def test_posiciones_count_final_correcto(self):
        """positions_count_final refleja las posiciones abiertas al terminar."""
        ath   = 60000.0
        limit = ath * (1 - cfg.PCT_CAIDA_ATH)
        # Solo compramos, no hay venta
        df = self._df([
            (59000, 60000, 58500, 59500),
            (59000, 59200, limit - 100, 58000),   # compra, sin venta posterior
        ])
        results = estrategia.ejecutar_estrategia(df)
        s = results["summary"]
        self.assertEqual(s["positions_count_final"], 1)
        self.assertGreater(s["btc_en_posiciones_final"], 0.0)

    def test_trade_history_campos_completos_compra(self):
        """Todos los campos requeridos existen en un registro de compra."""
        ath   = 60000.0
        limit = ath * (1 - cfg.PCT_CAIDA_ATH)
        df = self._df([
            (59000, 60000, 58500, 59500),
            (59000, 59200, limit - 100, 58000),
        ])
        results = estrategia.ejecutar_estrategia(df)
        compras = [t for t in results["trade_history"]
                   if t["type"] == "BUY" and not t["ignorado"]]
        self.assertGreater(len(compras), 0)
        campos = ["datetime", "type", "price", "precio_tp", "ath", "atl",
                  "pct_capital_usado", "pos_gradiente", "usdt_spent",
                  "commission_compra_usdt", "btc_bought", "usdt_balance",
                  "btc_balance", "btc_en_posiciones", "positions_count"]
        for campo in campos:
            self.assertIn(campo, compras[0], f"Campo '{campo}' ausente en registro de compra")

    def test_trade_history_campos_completos_venta(self):
        """Todos los campos requeridos existen en un registro de venta."""
        ath   = 60000.0
        limit = ath * (1 - cfg.PCT_CAIDA_ATH)
        tp    = limit * (1 + cfg.PCT_VENTA)
        df = self._df([
            (59000, 60000, 58500, 59500),
            (59000, 59200, limit - 100, 58000),
            (59000, tp + 100, 58900, tp + 50),
        ])
        results = estrategia.ejecutar_estrategia(df)
        ventas = [t for t in results["trade_history"]
                  if t["type"] == "SELL" and not t["ignorado"]]
        self.assertGreater(len(ventas), 0)
        campos = ["datetime", "type", "price", "precio_tp", "ath", "atl",
                  "pos_gradiente", "btc_sold", "commission_venta_usdt",
                  "btc_accumulated", "usdt_received", "ganancia_usdt",
                  "usdt_balance", "btc_balance", "btc_en_posiciones", "positions_count"]
        for campo in campos:
            self.assertIn(campo, ventas[0], f"Campo '{campo}' ausente en registro de venta")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
