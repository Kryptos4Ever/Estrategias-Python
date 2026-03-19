"""
Tests — Estrategia_Divergencia_RSI_Umbral
==========================================
Corre con:  python test_estrategia.py
No requiere la DB real: usa DataFrames sintéticos construidos a mano
para que cada invariante sea verificable con valores exactos conocidos.

Organización
------------
  Bloque A — Funciones matemáticas  (gradientes, RSI)
  Bloque B — Señales de divergencia
  Bloque C — Ejecución de trades    (compras, ventas, guardias)
  Bloque D — Invariantes financieras (capital, contadores, PP)
  Bloque E — Consistencia del trade_history
  Bloque F — ATH / ATL tracking
  Bloque G — Casos borde / inputs degenerados
"""

import sys, os, math, json
import numpy as np
import pandas as pd

# ── Patch config antes de importar la estrategia ─────────────────────────────
# Usamos valores conocidos para que los asserts sean deterministas
import types
cfg = types.ModuleType("config")
cfg.DB_PATH               = ":memory:"
cfg.RESULTS_JSON          = "test_results.json"
cfg.FECHA_INICIO          = None
cfg.FECHA_FIN             = None
cfg.SALDO_USDT_INICIAL    = 1000.0
cfg.RSI_LENGTH            = 7
cfg.N                     = 5
cfg.FLOOR_PCT             = 20        # ATL_REF = ATH * 0.20
cfg.FACTOR_CAIDA          = 2.0
cfg.FACTOR_SUBIDA         = 1.0
cfg.GUARDIA_COMPRA        = False
cfg.GUARDIA_PRECIO_COMPRA = False
cfg.GUARDIA_PRECIO_VENTA  = False
cfg.RSI_BUY_TRIGGER       = 30
cfg.RSI_SELL_TRIGGER      = 70
cfg.USDT_RESERVA_PCT      = 0
cfg.BTC_PCT_TO_ACCUMULATE = 0
cfg.COMMISSION_PCT        = 0.1
cfg.mostrar_configuracion = lambda: None
sys.modules["config"] = cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(
    "/mnt/user-data/uploads/Estrategia_Divergencia_RSI_Umbral.py")))
sys.path.insert(0, "/mnt/user-data/uploads")

from Estrategia_Divergencia_RSI_Umbral import (
    calcular_rsi,
    pct_capital_compra,
    pct_capital_venta,
    ejecutar_estrategia,
)

COMM   = cfg.COMMISSION_PCT / 100
SALDO  = cfg.SALDO_USDT_INICIAL
EPS    = 1e-9   # tolerancia para comparaciones float

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_df(highs, lows, closes=None, opens=None):
    """Construye el mínimo DataFrame que espera ejecutar_estrategia."""
    n = len(highs)
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    if opens is None:
        opens = closes
    return pd.DataFrame({
        "timestamp": range(n),
        "open"     : opens,
        "high"     : highs,
        "low"      : lows,
        "close"    : closes,
        "datetime" : pd.date_range("2021-01-01", periods=n, freq="h"),
    })


def _trades(results):
    """Filtra solo los trades ejecutados (no ignorados)."""
    return [t for t in results["trade_history"] if not t.get("ignorado", False)]


def _buys(results):
    return [t for t in _trades(results) if t["type"] == "BUY"]


def _sells(results):
    return [t for t in _trades(results) if t["type"] == "SELL"]


passed = []
failed = []

def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  ✓  {name}")
    else:
        failed.append(name)
        print(f"  ✗  {name}  {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE A — GRADIENTES Y RSI
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque A: Gradientes y RSI ──────────────────────────────────────")

ATH   = 60_000.0
FLOOR = cfg.FLOOR_PCT      # 20
ATL_R = ATH * FLOOR / 100  # 12_000  ← precio al que pct=100

# A1 — Output siempre ∈ [0, 100]
for price in [ATH * 2, ATH, ATH * 0.9, ATH * 0.5, ATL_R, ATL_R * 0.5, 1.0]:
    v = pct_capital_compra(price, ATH)
    check(f"A1 compra_clamp  price={price:,.0f}", 0 <= v <= 100,
          f"got {v:.4f}")

# A2 — Precio >= ATH → pct = 0
v = pct_capital_compra(ATH, ATH)
check("A2 compra=0 @ ATH", v == 0.0, f"got {v}")

# A3 — Precio = ATL_REF → pct = 100
v = pct_capital_compra(ATL_R, ATH)
check("A3 compra=100 @ ATL_REF", abs(v - 100.0) < 1e-6, f"got {v}")

# A4 — Precio < ATL_REF → pct sigue siendo 100 (clamp)
v = pct_capital_compra(ATL_R * 0.5, ATH)
check("A4 compra=100 bajo ATL_REF (clamp)", abs(v - 100.0) < 1e-6, f"got {v}")

# A5 — Monotonicidad: a menor precio, mayor pct
prices = [ATH * f for f in [0.9, 0.7, 0.5, 0.3]]
pcts   = [pct_capital_compra(p, ATH) for p in prices]
check("A5 compra monotona decreciente", all(pcts[i] < pcts[i+1] for i in range(len(pcts)-1)),
      f"pcts={[round(p,2) for p in pcts]}")

# A6 — FACTOR_CAIDA=1 → pos lineal en log → pct = pos * 100 exactamente
# Hay que patchear el módulo directamente (ya importó FACTOR_CAIDA al cargarse)
import Estrategia_Divergencia_RSI_Umbral as _est_a6
_est_a6.FACTOR_CAIDA = 1.0
precio_mid = math.exp((math.log(ATH) + math.log(ATL_R)) / 2)  # media geométrica
v = _est_a6.pct_capital_compra(precio_mid, ATH)
check("A6 compra FC=1 media_geom=50%", abs(v - 50.0) < 1e-4, f"got {v:.4f}")
_est_a6.FACTOR_CAIDA = 2.0  # restaurar

# A7 — Venta: precio <= PP → pct = 0
PP = 30_000.0
check("A7 venta=0 @ precio=PP",   pct_capital_venta(PP, ATH, PP)       == 0.0)
check("A8 venta=0 @ precio<PP",   pct_capital_venta(PP * 0.9, ATH, PP) == 0.0)

# A9 — Venta: precio = ATH → pct = 100
v = pct_capital_venta(ATH, ATH, PP)
check("A9 venta=100 @ ATH", abs(v - 100.0) < 1e-6, f"got {v}")

# A10 — Venta: PP >= ATH → pct = 0 (log_amp <= 0)
check("A10 venta=0 cuando PP>=ATH", pct_capital_venta(ATH * 1.1, ATH, ATH) == 0.0)

# A11 — Venta monotona: a mayor precio, mayor pct
# Nota: PP*2.0 == ATH cuando PP=30k, ATH=60k → ambos clampean a 100%, empate no es error.
# Usamos ATH*0.95 como cuarto precio para tener 4 valores estrictamente distintos.
venta_pcts = [pct_capital_venta(p, ATH, PP) for p in [PP*1.1, PP*1.5, ATH*0.95, ATH]]
check("A11 venta monotona creciente",
      all(venta_pcts[i] < venta_pcts[i+1] for i in range(len(venta_pcts)-1)),
      f"pcts={[round(p,2) for p in venta_pcts]}")

# A12 — RSI entre 0 y 100
serie = pd.Series([100.0, 98, 97, 96, 95, 97, 98, 96, 94, 93, 95, 97, 99, 98, 97])
rsi = calcular_rsi(serie, 7)
valid = rsi.dropna()
check("A12 RSI ∈ [0,100]",
      len(valid) > 0 and float(valid.min()) >= 0 and float(valid.max()) <= 100,
      f"min={float(valid.min()):.2f} max={float(valid.max()):.2f}")

# A13 — Serie constante → delta=0 siempre → avg_loss=0 → RS=nan → RSI=nan
# (0/0 es indeterminado: no hay pérdidas NI ganancias, el RSI de Wilder no está definido)
serie_flat = pd.Series([50000.0] * 30)
rsi_flat   = calcular_rsi(serie_flat, 7)
check("A13 RSI serie constante = NaN",
      rsi_flat.dropna().empty,
      f"valores no-nan: {rsi_flat.dropna().tolist()[:3]}")

# A14 — Serie mayormente subiendo (con ruido mínimo) → RSI alto
# Nota: serie estrictamente monotónica → avg_loss=0 → RSI=nan.
# Se necesita algún pullback para que el RSI sea calculable.
# Para RSI calculable necesitamos caídas ocasionales (avg_loss > 0)
serie_up = pd.Series([100.0 + i*3 - (5 if i%4==0 else 0) for i in range(50)])
rsi_up   = calcular_rsi(serie_up, 7).dropna()
check("A14 RSI serie mayormente alcista > 70",
      len(rsi_up) > 0 and float(rsi_up.values[-1]) > 70,
      f"got {float(rsi_up.values[-1]):.2f}" if len(rsi_up) > 0 else "empty")

# A15 — Serie mayormente bajando (con ruido mínimo) → RSI bajo
serie_dn = pd.Series([100.0 - i*2 - (1 if i%5==0 else 0) for i in range(50) if 100.0 - i*2 > 0])
rsi_dn   = calcular_rsi(serie_dn, 7).dropna()
check("A15 RSI serie mayormente bajista < 30",
      len(rsi_dn) > 0 and float(rsi_dn.values[-1]) < 30,
      f"got {float(rsi_dn.values[-1]):.2f}" if len(rsi_dn) > 0 else "empty")

# A16 — Inputs degenerados en gradientes
check("A16 compra ATH=0 → 0", pct_capital_compra(1000, 0) == 0.0)
check("A17 venta PP=0  → 0", pct_capital_venta(1000, 60000, 0) == 0.0)
check("A18 venta ATH=0 → 0", pct_capital_venta(1000, 0, 500) == 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE B — SEÑALES DE DIVERGENCIA (via ejecutar_estrategia)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque B: Señales de divergencia ────────────────────────────────")

# Construimos una serie donde conocemos exactamente cuándo hay divergencia
# N=5, RSI_LENGTH=7
# Necesitamos:
#   · lows[i] < min(lows[i-5:i])
#   · RSI(low[i]) > RSI(low[idx_min])
#   · RSI(low[idx_min]) <= 30
# Fabricamos: caída sostenida (RSI bajo) luego un mínimo de precio MÁS bajo
# pero con RSI más alto (la divergencia).

def _build_divergence_series(n_warmup=30, n_fall=15, n_div=3):
    """
    Devuelve (highs, lows) donde:
      - warmup: precio lateral (permite que RSI se estabilice)
      - fall: caída brusca (RSI llega a zona oversold)
      - div: sigue cayendo en precio pero el RSI sube ligeramente → divergencia
    """
    lows  = []
    highs = []
    # warmup lateral
    for i in range(n_warmup):
        lows.append(50000.0)
        highs.append(51000.0)
    # caída fuerte: RSI baja hacia oversold
    for i in range(n_fall):
        lows.append(50000.0 - i * 600)
        highs.append(50500.0 - i * 600)
    anchor_low = lows[-1]
    # divergencia: precio más bajo, pero caída más suave (RSI sube un poco)
    for i in range(n_div):
        lows.append(anchor_low - (i + 1) * 50)   # precio sigue bajando poco
        highs.append(anchor_low + 200)             # high se recupera un poco
    return highs, lows

# B1 — Sin ninguna señal: serie totalmente plana → 0 trades
highs_flat = [50000.0] * 50
lows_flat  = [49900.0] * 50
df_flat = _make_df(highs_flat, lows_flat)
r_flat  = ejecutar_estrategia(df_flat)
buys_flat = _buys(r_flat)
check("B1 serie plana → 0 compras ejecutadas", len(buys_flat) == 0)

# B2 — BUY y SELL no pueden ocurrir en el mismo trade_history entry
r2 = ejecutar_estrategia(_make_df(highs_flat, lows_flat))
both = [t for t in r2["trade_history"]
        if not t.get("ignorado") and t["type"] not in ("BUY", "SELL")]
check("B2 todo trade es BUY o SELL", len(both) == 0)

# B3 — Trigger RSI: divergencia con ancla fuera de zona → debe ser IGNORADO
# Forzamos RSI_BUY_TRIGGER muy bajo para que nunca pase el filtro
cfg.RSI_BUY_TRIGGER = 5   # casi imposible de satisfacer
highs_b3, lows_b3 = _build_divergence_series()
r_b3 = ejecutar_estrategia(_make_df(highs_b3, lows_b3))
check("B3 trigger muy bajo → divergencias rechazadas",
      r_b3["summary"]["umbral_filtro"]["divergencias_compra_aprobadas"] == 0
      or len(_buys(r_b3)) == 0)
cfg.RSI_BUY_TRIGGER = 30  # restaurar

# B4 — Trigger RSI muy alto (= 100): todas las divergencias pasan el filtro
cfg.RSI_BUY_TRIGGER = 100
r_b4  = ejecutar_estrategia(_make_df(highs_b3, lows_b3))
det4  = r_b4["summary"]["umbral_filtro"]["divergencias_compra_detectadas"]
apr4  = r_b4["summary"]["umbral_filtro"]["divergencias_compra_aprobadas"]
check("B4 trigger=100 → detectadas == aprobadas", det4 == apr4,
      f"det={det4} apr={apr4}")
cfg.RSI_BUY_TRIGGER = 30


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE C — EJECUCIÓN DE TRADES Y GUARDIAS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque C: Trades y guardias ─────────────────────────────────────")

# Usamos una serie que genera al menos una compra — caída clara con divergencia
# Para tests de guardias necesitamos poder controlar el estado con precisión.
# Empleamos RSI_BUY_TRIGGER=100 para que el filtro no interfiera.
cfg.RSI_BUY_TRIGGER  = 100
cfg.RSI_SELL_TRIGGER = 0

highs_c, lows_c = _build_divergence_series(n_warmup=40, n_fall=20, n_div=5)
df_c = _make_df(highs_c, lows_c)
r_c  = ejecutar_estrategia(df_c)
buys_c = _buys(r_c)

# C1 — Cada BUY: usdt_spent > 0
check("C1 BUY usdt_spent > 0",
      all(t["usdt_spent"] > 0 for t in buys_c), f"{len(buys_c)} buys")

# C2 — Cada BUY: btc_bought > 0
check("C2 BUY btc_bought > 0",
      all(t["btc_bought"] > 0 for t in buys_c))

# C3 — BUY: comisión = usdt_spent * COMM_PCT/100
for t in buys_c:
    expected_comm = t["usdt_spent"] * COMM
    check(f"C3 BUY comision correcta @ {t['price']:,.0f}",
          abs(t["commission_usdt"] - expected_comm) < 0.01,
          f"got {t['commission_usdt']:.4f} expected {expected_comm:.4f}")

# C4 — BUY: btc_bought = (usdt_spent - comision) / price
for t in buys_c:
    expected_btc = (t["usdt_spent"] - t["commission_usdt"]) / t["price"]
    check(f"C4 BUY btc_bought formula @ {t['price']:,.0f}",
          abs(t["btc_bought"] - expected_btc) < 1e-8,
          f"got {t['btc_bought']:.8f} expected {expected_btc:.8f}")

# C5 — BUY: price = low de la vela (compra siempre al low)
lows_arr = df_c["low"].values
for t in buys_c:
    dt = pd.Timestamp(t["datetime"])
    idx = df_c[df_c["datetime"] == dt].index
    if len(idx):
        expected_price = float(lows_arr[idx[0]])
        check(f"C5 BUY price = low @ {t['price']:,.0f}",
              abs(t["price"] - expected_price) < EPS)

# C6 — GUARDIA_COMPRA: ningún BUY con precio >= PP (cuando hay posiciones)
cfg.GUARDIA_COMPRA = True
r_gc = ejecutar_estrategia(df_c)
buys_gc = _buys(r_gc)
violations = []
running_pp = 0.0
running_btc = 0.0
running_inv = 0.0
for t in _trades(r_gc):
    if t["type"] == "BUY":
        if running_btc > 0:
            violations.append(t["price"] >= running_pp)
        running_btc += t["btc_bought"]
        running_inv += t["usdt_spent"]
        running_pp   = running_inv / running_btc if running_btc > 0 else 0
    elif t["type"] == "SELL":
        prop = t["btc_sold"] / running_btc if running_btc > 0 else 0
        running_inv  = max(running_inv - running_inv * prop, 0)
        running_btc -= t["btc_sold"]
        running_pp   = running_inv / running_btc if running_btc > 0 else 0
check("C6 GUARDIA_COMPRA: nunca compra por encima del PP",
      not any(violations), f"{sum(violations)} violaciones")
cfg.GUARDIA_COMPRA = False

# C7 — GUARDIA_PRECIO_COMPRA: precios de compra estrictamente decrecientes
cfg.GUARDIA_PRECIO_COMPRA = True
r_gpc = ejecutar_estrategia(df_c)
buy_prices = [t["price"] for t in _buys(r_gpc)]
check("C7 GUARDIA_PRECIO_COMPRA: precios decrecientes",
      all(buy_prices[i] > buy_prices[i+1] for i in range(len(buy_prices)-1)),
      f"prices={buy_prices[:5]}")
cfg.GUARDIA_PRECIO_COMPRA = False

# C8 — GUARDIA_PRECIO_VENTA: precios de venta estrictamente crecientes
cfg.GUARDIA_PRECIO_VENTA = True
r_gpv = ejecutar_estrategia(df_c)
sell_prices = [t["price"] for t in _sells(r_gpv)]
check("C8 GUARDIA_PRECIO_VENTA: precios crecientes",
      all(sell_prices[i] < sell_prices[i+1] for i in range(len(sell_prices)-1))
      if len(sell_prices) >= 2 else True,
      f"prices={sell_prices[:5]}")
cfg.GUARDIA_PRECIO_VENTA = False

cfg.RSI_BUY_TRIGGER  = 30
cfg.RSI_SELL_TRIGGER = 70


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE D — INVARIANTES FINANCIERAS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque D: Invariantes financieras ───────────────────────────────")

cfg.RSI_BUY_TRIGGER = 100
df_d = _make_df(*_build_divergence_series(n_warmup=40, n_fall=20, n_div=5))
r_d  = ejecutar_estrategia(df_d)
s    = r_d["summary"]

# D1 — usdt_balance_final >= 0
check("D1 usdt_balance_final >= 0", s["usdt_balance_final"] >= 0,
      f"got {s['usdt_balance_final']}")

# D2 — btc_en_posiciones_final >= 0
check("D2 btc_en_posiciones_final >= 0", s["btc_en_posiciones_final"] >= 0)

# D3 — portfolio = usdt + (btc_pos + btc_libre) * precio_cierre
precio_cierre = float(df_d["close"].iloc[-1])
portfolio_calc = (s["usdt_balance_final"]
                  + (s["btc_en_posiciones_final"] + s["btc_balance_final"]) * precio_cierre)
check("D3 portfolio_final consistente",
      abs(portfolio_calc - s["portfolio_value_final"]) < 0.01,
      f"calc={portfolio_calc:.4f} stored={s['portfolio_value_final']:.4f}")

# D4 — PnL% consistente con portfolio
pnl_calc = (s["portfolio_value_final"] - SALDO) / SALDO * 100
check("D4 pnl_pct consistente",
      abs(pnl_calc - s["pnl_pct"]) < 0.001,
      f"calc={pnl_calc:.4f} stored={s['pnl_pct']:.4f}")

# D5 — positions_count final = compras - ventas
check("D5 positions_count = compras - ventas",
      s["positions_count_final"] == s["total_compras"] - s["total_ventas"],
      f"count={s['positions_count_final']} C={s['total_compras']} V={s['total_ventas']}")

# D6 — positions_count final >= 0
check("D6 positions_count_final >= 0", s["positions_count_final"] >= 0)

# D7 — total_trades = compras + ventas
check("D7 total_trades = compras + ventas",
      s["total_trades_ejecutados"] == s["total_compras"] + s["total_ventas"])

# D8 — total_ignorados = suma de motivos
motivos_sum = sum(s["ignorados_por_motivo"].values())
check("D8 total_ignorados = sum(motivos)",
      s["total_ignorados"] == motivos_sum,
      f"total={s['total_ignorados']} sum={motivos_sum}")

# D9 — PP final = usdt_invertido / btc_en_posiciones (cuando hay BTC)
#      Verificamos reconstruyendo desde el trade history
usdt_inv_rec = 0.0
btc_rec      = 0.0
for t in _trades(r_d):
    if t["type"] == "BUY":
        usdt_inv_rec += t["usdt_spent"]
        btc_rec      += t["btc_bought"]
    elif t["type"] == "SELL":
        prop          = t["btc_sold"] / btc_rec if btc_rec > 0 else 0
        usdt_inv_rec  = max(usdt_inv_rec - usdt_inv_rec * prop, 0)
        btc_rec      -= t["btc_sold"]
pp_rec = usdt_inv_rec / btc_rec if btc_rec > 1e-12 else 0.0
pp_sto = s["precio_promedio_final"]
check("D9 PP final consistente con trade history",
      abs(pp_rec - pp_sto) < 0.01 or (btc_rec < 1e-12 and pp_sto == 0.0),
      f"recalc={pp_rec:.2f} stored={pp_sto:.2f}")

# D10 — Sin trades: portfolio = saldo inicial, pnl = 0
r_notrades = ejecutar_estrategia(_make_df([50000.0]*30, [49900.0]*30))
check("D10 sin trades: pnl=0.0",
      abs(r_notrades["summary"]["pnl_pct"]) < EPS)
check("D11 sin trades: portfolio=saldo_inicial",
      abs(r_notrades["summary"]["portfolio_value_final"] - SALDO) < 0.01)

cfg.RSI_BUY_TRIGGER = 30


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE E — CONSISTENCIA DEL TRADE HISTORY
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque E: Consistencia del trade_history ────────────────────────")

cfg.RSI_BUY_TRIGGER = 100
r_e = ejecutar_estrategia(_make_df(*_build_divergence_series(n_warmup=40, n_fall=20, n_div=5)))

for t in r_e["trade_history"]:
    ignorado = t.get("ignorado", False)
    tipo     = t["type"]

    # E1 — Todo trade: ignorado=True XOR campos de ejecución definidos
    if not ignorado:
        if tipo == "BUY":
            check(f"E1 BUY ejecutado: usdt_spent != None",
                  t["usdt_spent"] is not None)
            check(f"E1 BUY ejecutado: btc_bought != None",
                  t["btc_bought"] is not None)
            check(f"E1 BUY ejecutado: btc_sold = None",
                  t["btc_sold"] is None)
            check(f"E1 BUY ejecutado: usdt_received = None",
                  t["usdt_received"] is None)
        elif tipo == "SELL":
            check(f"E1 SELL ejecutado: btc_sold != None",
                  t["btc_sold"] is not None)
            check(f"E1 SELL ejecutado: usdt_received != None",
                  t["usdt_received"] is not None)
            check(f"E1 SELL ejecutado: usdt_spent = None",
                  t["usdt_spent"] is None)
            check(f"E1 SELL ejecutado: btc_bought = None",
                  t["btc_bought"] is None)
    else:
        # E2 — Ignorado: motivo siempre informado
        check(f"E2 ignorado: motivo != None",
              t.get("motivo_ignorado") is not None)

# E3 — Saldo USDT en el trade history nunca negativo
usdt_neg = [t for t in r_e["trade_history"] if t.get("usdt_balance", 0) < -EPS]
check("E3 usdt_balance nunca negativo en history",
      len(usdt_neg) == 0, f"{len(usdt_neg)} negativos")

# E4 — BTC en posiciones en el trade history nunca negativo
btc_neg = [t for t in r_e["trade_history"] if t.get("btc_en_posiciones", 0) < -EPS]
check("E4 btc_en_posiciones nunca negativo en history",
      len(btc_neg) == 0, f"{len(btc_neg)} negativos")

# E5 — positions_count en history es consistente paso a paso
pc = 0
pc_errors = 0
for t in r_e["trade_history"]:
    if not t.get("ignorado"):
        pc += (1 if t["type"] == "BUY" else -1)
    if t.get("positions_count", pc) != pc:
        pc_errors += 1
check("E5 positions_count consistente en cada trade",
      pc_errors == 0, f"{pc_errors} inconsistencias")

cfg.RSI_BUY_TRIGGER = 30


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE F — ATH / ATL
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque F: ATH / ATL ─────────────────────────────────────────────")

# Serie con ATH y ATL conocidos exactamente
highs_f = [50000.0] * 10 + [80000.0] + [50000.0] * 20   # ATH = 80k en candle 10
lows_f  = [49000.0] * 15 + [10000.0] + [49000.0] * 15   # ATL = 10k en candle 15
df_f    = _make_df(highs_f, lows_f)
r_f     = ejecutar_estrategia(df_f)

# F1 — ATL final = mínimo de todos los lows
atl_real = min(lows_f)
check("F1 atl_final = min(lows)",
      abs(r_f["summary"]["atl_final"] - atl_real) < EPS,
      f"got {r_f['summary']['atl_final']} expected {atl_real}")

# F2 — ATH en cada trade siempre >= high de la vela en ese momento
#      (lo verificamos usando el campo ath en el trade_history)
highs_arr = df_f["high"].values
dts_arr   = df_f["datetime"].values
for t in r_f["trade_history"]:
    dt    = pd.Timestamp(t["datetime"])
    idx   = df_f[df_f["datetime"] == dt].index
    if len(idx):
        max_high_so_far = float(highs_arr[:idx[0]+1].max())
        check(f"F2 ath en trade >= max_high_hasta_ese_momento",
              t["ath"] >= max_high_so_far - EPS,
              f"ath={t['ath']} max_so_far={max_high_so_far}")


# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE G — CASOS BORDE
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Bloque G: Casos borde ───────────────────────────────────────────")

# G1 — DataFrame con exactamente N velas (loop nunca ejecuta)
df_min = _make_df([50000.0] * cfg.N, [49900.0] * cfg.N)
r_min  = ejecutar_estrategia(df_min)
check("G1 df con solo N velas → 0 trades, sin error",
      r_min["summary"]["total_trades_ejecutados"] == 0)

# G2 — DataFrame con N+1 velas (una sola iteración)
df_n1  = _make_df([50000.0] * (cfg.N + 1), [49900.0] * (cfg.N + 1))
r_n1   = ejecutar_estrategia(df_n1)
check("G2 df con N+1 velas → sin error",
      isinstance(r_n1["summary"]["pnl_pct"], float))

# G3 — USDT_RESERVA_PCT = 100 → sin capital disponible → 0 compras
cfg.USDT_RESERVA_PCT = 100
cfg.RSI_BUY_TRIGGER  = 100
# Necesitamos re-importar USDT_RESERVA con el nuevo valor
import importlib
import Estrategia_Divergencia_RSI_Umbral as _est
_est.USDT_RESERVA = SALDO * 100 / 100  # patch directo
r_g3  = _est.ejecutar_estrategia(df_d)
check("G3 reserva=100% → 0 compras ejecutadas",
      r_g3["summary"]["total_compras"] == 0,
      f"compras={r_g3['summary']['total_compras']}")
_est.USDT_RESERVA = 0.0
cfg.USDT_RESERVA_PCT = 0
cfg.RSI_BUY_TRIGGER  = 30

# G4 — BTC_PCT_TO_ACCUMULATE = 0 → btc_acumulado = 0 en todos los SELL
r_g4 = ejecutar_estrategia(_make_df(*_build_divergence_series()))
bad_acum = [t for t in _sells(r_g4) if t.get("btc_accumulated") not in (None, 0, 0.0)]
check("G4 acumulacion=0 → ningún SELL acumula BTC",
      len(bad_acum) == 0, f"{len(bad_acum)} con btc_accumulated != 0")

# G5 — BTC_PCT_TO_ACCUMULATE = 50 → btc_acumulado = 50% del slot en cada SELL
cfg.BTC_PCT_TO_ACCUMULATE = 50
import Estrategia_Divergencia_RSI_Umbral as _est2
_est2.BTC_PCT_TO_ACCUMULATE = 50   # patch módulo
cfg.RSI_BUY_TRIGGER = 100
r_g5    = _est2.ejecutar_estrategia(df_d)
sells_g5 = _sells(r_g5)
acum_err = []
for t in sells_g5:
    btc_slot = (t["btc_sold"] or 0) + (t["btc_accumulated"] or 0)
    expected = btc_slot * 0.50
    if abs((t["btc_accumulated"] or 0) - expected) > 1e-8:
        acum_err.append(t)
check("G5 acumulacion=50% → btc_accumulated = slot * 0.50",
      len(acum_err) == 0, f"{len(acum_err)} con acumulación incorrecta")
_est2.BTC_PCT_TO_ACCUMULATE = 0
cfg.BTC_PCT_TO_ACCUMULATE = 0
cfg.RSI_BUY_TRIGGER = 30


# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════
total = len(passed) + len(failed)
print(f"\n{'═'*65}")
print(f"  RESULTADO FINAL:  {len(passed)}/{total} tests pasados")
if failed:
    print(f"\n  FALLARON ({len(failed)}):")
    for f in failed:
        print(f"    ✗  {f}")
print(f"{'═'*65}")
sys.exit(0 if not failed else 1)
