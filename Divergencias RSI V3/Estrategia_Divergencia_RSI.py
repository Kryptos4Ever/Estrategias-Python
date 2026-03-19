"""
Estrategia Divergencia RSI — Gradientes Logarítmicos ATH/ATL
═════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Backtesting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEÑALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPRA (divergencia alcista):
  · low[i]      < min(low[i-N : i])     → nuevo mínimo local de precio
  · RSI(low[i]) > RSI(low[idx_min])     → RSI no confirma ese mínimo

VENTA (divergencia bajista):
  · high[i]      > max(high[i-N : i])   → nuevo máximo local de precio
  · RSI(high[i]) < RSI(high[idx_max])   → RSI no confirma ese máximo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENTE LOGARÍTMICO DE COMPRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ATL_REF    = ath × FLOOR_PCT / 100     ← FIJO (no depende del ATL actual)
  log_rango  = log(100 / FLOOR_PCT)       ← CONSTANTE del ciclo
  pos_compra = log(ATH / precio_low) / log(ATH / ATL_REF)
  pct_usdt   = clamp(pos_compra, 0, 1) ^ FACTOR_CAIDA × 100
  usdt_trade = usdt_disponible × pct_usdt / 100

  log_rango es constante: log(100/FLOOR_PCT) = amplitud logarítmica esperada.
  Si el precio cae más de lo esperado (pos > 1), se clampea a 100%.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENTE LOGARÍTMICO DE VENTA — ANCLADO AL PP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pos_venta  = log(precio_high / PP) / log(ATH / PP)
  pct_btc    = clamp(pos_venta, 0, 1) ^ FACTOR_SUBIDA × 100
  btc_trade  = btc_en_posiciones × pct_btc / 100

  PP = precio promedio de posiciones abiertas (se actualiza con cada trade).
  La guardia de venta está incorporada: precio ≤ PP → pos ≤ 0 → vende 0%.
  Autoajuste: PP más bajo → log(ATH/PP) mayor → curva más sensible.
"""

import sqlite3
import json
import math
import os
import numpy as np
import pandas as pd

from config import (
    DB_PATH, RESULTS_JSON, FECHA_INICIO, FECHA_FIN,
    SALDO_USDT_INICIAL,
    RSI_LENGTH, N,
    FLOOR_PCT, FACTOR_CAIDA, FACTOR_SUBIDA,
    GUARDIA_COMPRA,
    GUARDIA_PRECIO_COMPRA,
    GUARDIA_PRECIO_VENTA,
    USDT_RESERVA_PCT,
    BTC_PCT_TO_ACCUMULATE,
    COMMISSION_PCT,
    mostrar_configuracion,
)

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI clásico de Wilder (EWM), idéntico al de TradingView."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def pct_capital_compra(precio_low: float, ath: float) -> float:
    """
    Retorna % [0, 100] del USDT disponible a usar en la compra.

    Gradiente logarítmico con ATL_REF FIJO:
      log_rango = log(100 / FLOOR_PCT)          ← CONSTANTE, independiente del ATL actual
      pos       = log(ATH / precio) / log_rango
      pct       = clamp(pos, 0, 1) ^ FACTOR_CAIDA × 100

    IMPORTANTE: NO usar max(atl, floor). La señal de compra dispara en
    lows[i] < window_min, actualizando ATL = lows[i] en la misma vela.
    max(atl, floor) colapsaría a atl = precio → pos = 1.0 → 100% siempre.
    Con log_rango = log(100/FLOOR_PCT) el denominador es constante.
    Si el mercado cae más que FLOOR_PCT, pos > 1 se clampea a 1 → 100%.
    """
    if ath <= 0 or FLOOR_PCT <= 0:
        return 0.0
    log_rango = math.log(100.0 / FLOOR_PCT)
    if log_rango <= 0:
        return 0.0
    pos = math.log(ath / precio_low) / log_rango
    pos = max(0.0, min(1.0, pos))
    return (pos ** FACTOR_CAIDA) * 100.0


def pct_capital_venta(precio_high: float, ath: float,
                      precio_promedio: float) -> float:
    """
    Retorna % [0, 100] del BTC en posiciones a vender.

    Gradiente logarítmico anclado al PP:
      pos_venta = log(precio / PP) / log(ATH / PP)
      pct       = clamp(pos_venta, 0, 1) ^ FACTOR_SUBIDA × 100

    Retorna 0 si precio ≤ PP (guardia incorporada).
    Retorna 0 si no hay PP válido (sin posiciones abiertas).
    """
    if ath <= 0 or precio_promedio <= 0:
        return 0.0
    if precio_high <= precio_promedio:
        return 0.0
    log_amp = math.log(ath / precio_promedio)
    if log_amp <= 0:
        return 0.0
    pos  = math.log(precio_high / precio_promedio) / log_amp
    pos  = max(0.0, min(1.0, pos))
    return (pos ** FACTOR_SUBIDA) * 100.0


def cargar_datos() -> pd.DataFrame:
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM   {DB_TABLE}
        ORDER  BY timestamp ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if FECHA_INICIO:
        df = df[df["datetime"] >= pd.to_datetime(FECHA_INICIO)]
    if FECHA_FIN:
        df = df[df["datetime"] <= pd.to_datetime(FECHA_FIN)]

    df = df.reset_index(drop=True)
    print(f"✓ Velas cargadas  : {len(df):,}")
    print(f"  Desde           : {df['datetime'].iloc[0]}")
    print(f"  Hasta           : {df['datetime'].iloc[-1]}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGISTRO DE TRADE IGNORADO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _registro_ignorado(ts, tipo, precio, rsi_l, rsi_h,
                        usdt_bal, btc_bal, btc_pos,
                        pos_count, ath, atl, motivo: str) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : tipo,
        "price"                      : precio,
        "rsi_low"                    : round(rsi_l, 4),
        "rsi_high"                   : round(rsi_h, 4),
        "ath"                        : round(ath, 4),
        "atl"                        : round(atl, 4),
        "pct_capital_usado"          : None,
        "pos_gradiente"              : None,
        "usdt_spent"                 : None,
        "btc_bought"                 : None,
        "commission_usdt"            : None,
        "btc_sold"                   : None,
        "btc_accumulated"            : None,
        "usdt_received"              : None,
        "ganancia_usdt"              : None,
        "usdt_balance"               : round(usdt_bal, 4),
        "btc_balance"                : round(btc_bal,  8),
        "btc_en_posiciones"          : round(btc_pos,  8),
        "positions_count"            : pos_count,
        "precio_promedio_posiciones" : None,
        "ignorado"                   : True,
        "motivo_ignorado"            : motivo,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ESTRATEGIA PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ejecutar_estrategia(df: pd.DataFrame) -> dict:

    rsi_low  = calcular_rsi(df["low"],  RSI_LENGTH).values.astype(float)
    rsi_high = calcular_rsi(df["high"], RSI_LENGTH).values.astype(float)
    lows     = df["low"].values.astype(float)
    highs    = df["high"].values.astype(float)
    closes   = df["close"].values.astype(float)
    dts      = df["datetime"].values

    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    positions_count   = 0

    precio_min_comprado = math.inf   # mínimo precio al que se ejecutó una compra
    precio_max_vendido  = 0.0        # máximo precio al que se ejecutó una venta

    ath = float(highs[0])
    atl = float(lows[0])

    trade_history = []
    n_velas = len(lows)

    for i in range(N, n_velas):

        if highs[i] > ath: ath = float(highs[i])
        if lows[i]  < atl: atl = float(lows[i])

        if np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]):
            continue

        window_lows  = lows[i - N : i]
        window_highs = highs[i - N : i]
        ts           = pd.Timestamp(dts[i])

        precio_promedio = (usdt_invertido / btc_en_posiciones
                           if btc_en_posiciones > 0 else 0.0)

        # ── Detectar señales ──────────────────────────────────────────────────
        señal_compra = False
        if lows[i] < window_lows.min():
            idx_min = i - N + int(window_lows.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                señal_compra = True

        señal_venta = False
        if not señal_compra:
            if highs[i] > window_highs.max():
                idx_max = i - N + int(window_highs.argmax())
                if rsi_high[i] < rsi_high[idx_max]:
                    señal_venta = True

        # ─────────────────────────────────────────────────────────────────────
        # COMPRA
        # ─────────────────────────────────────────────────────────────────────
        if señal_compra:

            usdt_disponible = usdt_balance - USDT_RESERVA
            if usdt_disponible <= 0:
                trade_history.append(_registro_ignorado(
                    ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "sin_capital_sobre_reserva"
                ))
                continue

            # Guardia: no comprar si el precio está sobre el PP actual
            if GUARDIA_COMPRA and btc_en_posiciones > 0 and lows[i] >= precio_promedio:
                trade_history.append(_registro_ignorado(
                    ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "precio_sobre_promedio"
                ))
                continue

            # Guardia: no comprar por encima del mínimo precio comprado histórico
            if GUARDIA_PRECIO_COMPRA and precio_min_comprado < math.inf and lows[i] >= precio_min_comprado:
                trade_history.append(_registro_ignorado(
                    ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "precio_sobre_min_comprado"
                ))
                continue

            pct         = pct_capital_compra(lows[i], ath)
            usdt_a_usar = usdt_disponible * pct / 100.0

            if usdt_a_usar <= 0:
                trade_history.append(_registro_ignorado(
                    ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "gradiente_cero"
                ))
                continue

            # pos_gradiente para registro (informativo)
            log_rango = math.log(100.0 / FLOOR_PCT)
            pos_grad  = round(math.log(ath / lows[i]) / log_rango, 6) if log_rango > 0 else 0.0

            price         = lows[i]
            comision      = usdt_a_usar * (COMMISSION_PCT / 100)
            btc_adquirido = (usdt_a_usar - comision) / price

            usdt_balance      -= usdt_a_usar
            btc_en_posiciones += btc_adquirido
            usdt_invertido    += usdt_a_usar
            positions_count   += 1
            precio_promedio    = usdt_invertido / btc_en_posiciones
            if price < precio_min_comprado:
                precio_min_comprado = price

            trade_history.append({
                "datetime"                   : ts.isoformat(),
                "type"                       : "BUY",
                "price"                      : price,
                "rsi_low"                    : round(rsi_low[i],  4),
                "rsi_high"                   : round(rsi_high[i], 4),
                "ath"                        : round(ath,  4),
                "atl"                        : round(atl,  4),
                "pct_capital_usado"          : round(pct,        4),
                "pos_gradiente"              : pos_grad,
                "usdt_spent"                 : round(usdt_a_usar,    4),
                "btc_bought"                 : round(btc_adquirido,  8),
                "commission_usdt"            : round(comision,       4),
                "btc_sold"                   : None,
                "btc_accumulated"            : None,
                "usdt_received"              : None,
                "ganancia_usdt"              : None,
                "usdt_balance"               : round(usdt_balance,        4),
                "btc_balance"                : round(btc_balance,          8),
                "btc_en_posiciones"          : round(btc_en_posiciones,    8),
                "positions_count"            : positions_count,
                "precio_promedio_posiciones" : round(precio_promedio,      4),
                "ignorado"                   : False,
                "motivo_ignorado"            : None,
            })

        # ─────────────────────────────────────────────────────────────────────
        # VENTA
        # ─────────────────────────────────────────────────────────────────────
        elif señal_venta and btc_en_posiciones > 0:

            pct      = pct_capital_venta(highs[i], ath, precio_promedio)
            btc_slot = btc_en_posiciones * pct / 100.0

            if btc_slot <= 0:
                # precio ≤ PP (guardia incorporada) o gradiente cero
                motivo = ("precio_bajo_promedio"
                          if precio_promedio > 0 and highs[i] <= precio_promedio
                          else "gradiente_cero")
                trade_history.append(_registro_ignorado(
                    ts, "SELL", highs[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, motivo
                ))
                continue

            # Guardia: no vender por debajo del máximo precio vendido histórico
            if GUARDIA_PRECIO_VENTA and precio_max_vendido > 0 and highs[i] <= precio_max_vendido:
                trade_history.append(_registro_ignorado(
                    ts, "SELL", highs[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "precio_bajo_max_vendido"
                ))
                continue

            # pos_gradiente para registro
            log_amp  = math.log(ath / precio_promedio) if precio_promedio > 0 else 0
            pos_grad = round(math.log(highs[i] / precio_promedio) / log_amp, 6) \
                       if log_amp > 0 and highs[i] > precio_promedio else 0.0

            btc_a_acumular     = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_a_vender       = btc_slot - btc_a_acumular
            price              = highs[i]
            usdt_bruto         = btc_a_vender * price
            comision           = usdt_bruto * (COMMISSION_PCT / 100)
            usdt_neto          = usdt_bruto - comision

            costo_proporcional = usdt_invertido * (btc_slot / btc_en_posiciones)
            ganancia           = usdt_neto - (costo_proporcional
                                              * (1 - BTC_PCT_TO_ACCUMULATE / 100))

            btc_en_posiciones -= btc_slot
            btc_balance       += btc_a_acumular
            usdt_balance      += usdt_neto
            usdt_invertido    -= costo_proporcional
            usdt_invertido     = max(usdt_invertido, 0.0)
            positions_count   -= 1
            precio_promedio    = (usdt_invertido / btc_en_posiciones
                                  if btc_en_posiciones > 0 else 0.0)
            if price > precio_max_vendido:
                precio_max_vendido = price

            trade_history.append({
                "datetime"                   : ts.isoformat(),
                "type"                       : "SELL",
                "price"                      : price,
                "rsi_low"                    : round(rsi_low[i],  4),
                "rsi_high"                   : round(rsi_high[i], 4),
                "ath"                        : round(ath,  4),
                "atl"                        : round(atl,  4),
                "pct_capital_usado"          : round(pct,        4),
                "pos_gradiente"              : pos_grad,
                "usdt_spent"                 : None,
                "btc_bought"                 : None,
                "commission_usdt"            : round(comision,       4),
                "btc_sold"                   : round(btc_a_vender,   8),
                "btc_accumulated"            : round(btc_a_acumular, 8),
                "usdt_received"              : round(usdt_neto,      4),
                "ganancia_usdt"              : round(ganancia,        4),
                "usdt_balance"               : round(usdt_balance,        4),
                "btc_balance"                : round(btc_balance,          8),
                "btc_en_posiciones"          : round(btc_en_posiciones,    8),
                "positions_count"            : positions_count,
                "precio_promedio_posiciones" : round(precio_promedio,      4),
                "ignorado"                   : False,
                "motivo_ignorado"            : None,
            })

    # ── Resumen final ─────────────────────────────────────────────────────────
    precio_final    = float(closes[-1])
    btc_total_final = btc_balance + btc_en_posiciones
    portfolio_final = usdt_balance + btc_total_final * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100
    precio_prom_fin = (usdt_invertido / btc_en_posiciones
                       if btc_en_posiciones > 0 else 0.0)

    trades_activos = [t for t in trade_history if not t.get("ignorado", False)]
    compras        = [t for t in trades_activos if t["type"] == "BUY"]
    ventas         = [t for t in trades_activos if t["type"] == "SELL"]
    ignorados      = [t for t in trade_history  if t.get("ignorado", False)]

    motivos = {}
    for t in ignorados:
        m = t.get("motivo_ignorado", "desconocido")
        motivos[m] = motivos.get(m, 0) + 1

    btc_acumulado_total = sum(
        t["btc_accumulated"] for t in ventas
        if t.get("btc_accumulated") is not None
    )

    summary = {
        "estrategia"              : "Divergencia RSI — Gradientes Logarítmicos",
        "fecha_inicio"            : str(df["datetime"].iloc[0]),
        "fecha_fin"               : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"      : SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(usdt_balance,        4),
        "btc_balance_final"       : round(btc_balance,          8),
        "btc_acumulado_total"     : round(btc_acumulado_total,  8),
        "btc_en_posiciones_final" : round(btc_en_posiciones,    8),
        "precio_promedio_final"   : round(precio_prom_fin,      4),
        "portfolio_value_final"   : round(portfolio_final,      4),
        "pnl_pct"                 : round(pnl_pct,              4),
        "precio_min_comprado"     : round(precio_min_comprado if precio_min_comprado < math.inf else 0.0, 4),
        "precio_max_vendido"      : round(precio_max_vendido, 4),
        "ath_final"               : round(ath, 4),        
        "atl_final"               : round(atl, 4),
        "total_trades_ejecutados" : len(trades_activos),
        "total_compras"           : len(compras),
        "total_ventas"            : len(ventas),
        "total_ignorados"         : len(ignorados),
        "ignorados_por_motivo"    : motivos,
        "positions_count_final"   : positions_count,
        "usdt_reserva_aplicada"   : round(USDT_RESERVA, 4),
        "parametros": {
            "rsi_length"           : RSI_LENGTH,
            "N"                    : N,
            "floor_pct"            : FLOOR_PCT,
            "factor_caida"         : FACTOR_CAIDA,
            "factor_subida"        : FACTOR_SUBIDA,
            "guardia_compra"       : GUARDIA_COMPRA,
            "guardia_precio_compra": GUARDIA_PRECIO_COMPRA,
            "guardia_precio_venta" : GUARDIA_PRECIO_VENTA,
            "usdt_reserva_pct"     : USDT_RESERVA_PCT,
            "btc_pct_to_accumulate": BTC_PCT_TO_ACCUMULATE,
            "commission_pct"       : COMMISSION_PCT,
        },
    }

    return {"summary": summary, "trade_history": trade_history}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DIVERGENCIA RSI — GRADIENTES LOGARÍTMICOS                   ║")
    print("║  BTC/USDT · Backtesting                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    mostrar_configuracion()

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py")
        return

    print("Ejecutando estrategia...")
    results = ejecutar_estrategia(df)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Resultados guardados en: {RESULTS_JSON}")

    s = results["summary"]
    print("\n" + "═" * 62)
    print("  RESUMEN FINAL")
    print("═" * 62)
    print(f"  Período              : {s['fecha_inicio'][:10]}  →  {s['fecha_fin'][:10]}")
    print(f"  Capital inicial      : ${s['saldo_inicial_usdt']:>10,.2f}")
    print(f"  Portfolio final      : ${s['portfolio_value_final']:>10,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre           : ${s['usdt_balance_final']:>10,.4f}   (reserva: ${s['usdt_reserva_aplicada']:,.2f})")
    print(f"  BTC libre (acum.)    :  {s['btc_balance_final']:>.8f} ₿  ({s['btc_acumulado_total']:.8f} ₿)")
    print(f"  BTC en posiciones    :  {s['btc_en_posiciones_final']:>.8f} ₿")
    pp = s['precio_promedio_final']
    if pp > 0:
        print(f"  Precio promedio BTC  : ${pp:>10,.2f}")
    print(f"  ATH registrado       : ${s['ath_final']:>10,.2f}")
    print(f"  ATL registrado       : ${s['atl_final']:>10,.2f}")
    pmc = s.get('precio_min_comprado', 0)
    pmv = s.get('precio_max_vendido',  0)
    if pmc > 0: print(f"  Precio mín. comprado : ${pmc:>10,.2f}  (guardia compra precio)")
    if pmv > 0: print(f"  Precio máx. vendido  : ${pmv:>10,.2f}  (guardia venta precio)")
    atl_ref = s['ath_final'] * FLOOR_PCT / 100
    print(f"  ATL_REF (FLOOR {FLOOR_PCT}%)  : ${atl_ref:>10,.2f}  (log_rango={math.log(100/FLOOR_PCT):.4f})")
    print(f"  Trades ejecutados    :  {s['total_trades_ejecutados']}  "
          f"(compras: {s['total_compras']}  |  ventas: {s['total_ventas']})")
    print(f"  Señales ignoradas    :  {s['total_ignorados']}")
    for motivo, cnt in s.get("ignorados_por_motivo", {}).items():
        print(f"    · {motivo:<35}: {cnt}")
    print(f"  positions_count      :  {s['positions_count_final']:+d}  "
          f"{'(más compras)' if s['positions_count_final'] > 0 else '(más ventas)' if s['positions_count_final'] < 0 else '(equilibrado)'}")
    print("═" * 62)


if __name__ == "__main__":
    main()