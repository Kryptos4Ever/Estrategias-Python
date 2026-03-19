"""
Estrategia Divergencia RSI — Umbral de Zona
════════════════════════════════════════════
BTC/USDT · Velas Horarias · Backtesting

Variante de Estrategia_Divergencia_RSI que añade un filtro de zona RSI
sobre la vela âncla de la divergencia.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEÑALES (vs. versión base)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPRA (divergencia alcista con filtro de sobreventa):
  · window_lows  = low[i-N : i]
  · idx_min      = índice del mínimo de precio en la ventana
  · rsi_low[idx_min] ≤ RSI_BUY_TRIGGER          ← NUEVO: âncla en sobreventa
  · low[i]       < window_lows.min()             → nuevo mínimo local de precio
  · RSI(low[i])  > RSI(low[idx_min])             → RSI no confirma ese mínimo

  La divergencia solo es válida si la vela âncla (el mínimo de precio de la
  ventana) tenía RSI ≤ RSI_BUY_TRIGGER en ese momento. Esto asegura que el
  punto de referencia viene de zona de sobreventa real, filtrando divergencias
  en la mitad de una tendencia bajista donde el RSI nunca toca niveles bajos.

VENTA (divergencia bajista con filtro de sobrecompra):
  · window_highs = high[i-N : i]
  · idx_max      = índice del máximo de precio en la ventana
  · rsi_high[idx_max] ≥ RSI_SELL_TRIGGER         ← NUEVO: âncla en sobrecompra
  · high[i]      > window_highs.max()             → nuevo máximo local de precio
  · RSI(high[i]) < RSI(high[idx_max])             → RSI no confirma ese máximo

  Misma lógica inversa: solo vende si el máximo de referencia estaba en zona
  de sobrecompra (RSI ≥ RSI_SELL_TRIGGER).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENTE LOGARÍTMICO DE COMPRA  (idéntico a versión base)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ATL_REF    = ath × FLOOR_PCT / 100     ← FIJO (no depende del ATL actual)
  log_rango  = log(100 / FLOOR_PCT)       ← CONSTANTE del ciclo
  pos_compra = log(ATH / precio_low) / log(ATH / ATL_REF)
  pct_usdt   = clamp(pos_compra, 0, 1) ^ FACTOR_CAIDA × 100
  usdt_trade = usdt_disponible × pct_usdt / 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENTE LOGARÍTMICO DE VENTA — ANCLADO AL PP (idéntico)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pos_venta  = log(precio_high / PP) / log(ATH / PP)
  pct_btc    = clamp(pos_venta, 0, 1) ^ FACTOR_SUBIDA × 100
  btc_trade  = btc_en_posiciones × pct_btc / 100
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
    RSI_BUY_TRIGGER,
    RSI_SELL_TRIGGER,
    USDT_RESERVA_PCT,
    BTC_PCT_TO_ACCUMULATE,
    COMMISSION_PCT,
    mostrar_configuracion,
)

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100

RESULTS_JSON_UMBRAL = RESULTS_JSON.replace(".json", "_umbral.json")


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
    Gradiente logarítmico con ATL_REF FIJO.
      log_rango = log(100 / FLOOR_PCT)   ← CONSTANTE, independiente del ATL actual
      pos       = log(ATH / precio) / log_rango
      pct       = clamp(pos, 0, 1) ^ FACTOR_CAIDA × 100
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
    Gradiente logarítmico anclado al PP.
      pos_venta = log(precio / PP) / log(ATH / PP)
      pct       = clamp(pos_venta, 0, 1) ^ FACTOR_SUBIDA × 100
    Retorna 0 si precio ≤ PP (guardia incorporada).
    """
    if ath <= 0 or precio_promedio <= 0:
        return 0.0
    if precio_high <= precio_promedio:
        return 0.0
    log_amp = math.log(ath / precio_promedio)
    if log_amp <= 0:
        return 0.0
    pos = math.log(precio_high / precio_promedio) / log_amp
    pos = max(0.0, min(1.0, pos))
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

    precio_min_comprado = math.inf
    precio_max_vendido  = 0.0

    # Inicializar ATH/ATL con el máximo/mínimo de las primeras N velas
    # (el loop empieza en i=N, por lo que highs[1..N-1] quedarían invisibles
    # si solo inicializamos con highs[0])
    ath = float(np.max(highs[:N]))
    atl = float(np.min(lows[:N]))

    # Contadores de señales filtradas por umbral RSI (estadística)
    n_div_compra_sin_umbral  = 0   # divergencias alcistas encontradas antes del filtro
    n_div_compra_con_umbral  = 0   # cuántas pasaron el filtro RSI_BUY_TRIGGER
    n_div_venta_sin_umbral   = 0
    n_div_venta_con_umbral   = 0

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

        # ── Detectar señal de COMPRA ──────────────────────────────────────────
        señal_compra = False
        if lows[i] < window_lows.min():
            idx_min = i - N + int(window_lows.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                # Divergencia alcista detectada — aplicar filtro de umbral
                n_div_compra_sin_umbral += 1
                if rsi_low[idx_min] <= RSI_BUY_TRIGGER:
                    # La vela âncla estaba en zona de sobreventa → señal válida
                    n_div_compra_con_umbral += 1
                    señal_compra = True
                else:
                    # Divergencia rechazada: âncla fuera de zona de sobreventa
                    trade_history.append(_registro_ignorado(
                        ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                        usdt_balance, btc_balance, btc_en_posiciones,
                        positions_count, ath, atl,
                        f"rsi_ancla_compra_fuera_umbral"
                        f"(rsi_ancla={rsi_low[idx_min]:.1f}>{RSI_BUY_TRIGGER})"
                    ))
                    continue

        # ── Detectar señal de VENTA ───────────────────────────────────────────
        señal_venta = False
        if not señal_compra:
            if highs[i] > window_highs.max():
                idx_max = i - N + int(window_highs.argmax())
                if rsi_high[i] < rsi_high[idx_max]:
                    # Divergencia bajista detectada — aplicar filtro de umbral
                    n_div_venta_sin_umbral += 1
                    if rsi_high[idx_max] >= RSI_SELL_TRIGGER:
                        # La vela âncla estaba en zona de sobrecompra → señal válida
                        n_div_venta_con_umbral += 1
                        señal_venta = True
                    else:
                        # Divergencia rechazada: âncla fuera de zona de sobrecompra
                        trade_history.append(_registro_ignorado(
                            ts, "SELL", highs[i], rsi_low[i], rsi_high[i],
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl,
                            f"rsi_ancla_venta_fuera_umbral"
                            f"(rsi_ancla={rsi_high[idx_max]:.1f}<{RSI_SELL_TRIGGER})"
                        ))
                        continue

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

            if GUARDIA_COMPRA and btc_en_posiciones > 0 and lows[i] >= precio_promedio:
                trade_history.append(_registro_ignorado(
                    ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "precio_sobre_promedio"
                ))
                continue

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
                motivo = ("precio_bajo_promedio"
                          if precio_promedio > 0 and highs[i] <= precio_promedio
                          else "gradiente_cero")
                trade_history.append(_registro_ignorado(
                    ts, "SELL", highs[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, motivo
                ))
                continue

            if GUARDIA_PRECIO_VENTA and precio_max_vendido > 0 and highs[i] <= precio_max_vendido:
                trade_history.append(_registro_ignorado(
                    ts, "SELL", highs[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl, "precio_bajo_max_vendido"
                ))
                continue

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
    ignorados      = [t for t in trade_history  if t.get("ignorado",  False)]

    motivos = {}
    for t in ignorados:
        # Agrupar los motivos de umbral con texto fijo para estadística limpia
        m = t.get("motivo_ignorado", "desconocido")
        clave = (m.split("(")[0] if m.startswith("rsi_ancla") else m)
        motivos[clave] = motivos.get(clave, 0) + 1

    btc_acumulado_total = sum(
        t["btc_accumulated"] for t in ventas
        if t.get("btc_accumulated") is not None
    )

    # Tasa de filtrado por umbral RSI
    tasa_filtro_compra = (
        round((1 - n_div_compra_con_umbral / n_div_compra_sin_umbral) * 100, 1)
        if n_div_compra_sin_umbral > 0 else 0.0
    )
    tasa_filtro_venta = (
        round((1 - n_div_venta_con_umbral / n_div_venta_sin_umbral) * 100, 1)
        if n_div_venta_sin_umbral > 0 else 0.0
    )

    summary = {
        "estrategia"              : "Divergencia RSI — Umbral de Zona",
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
        "atl_final"               : round(atl, 4),
        "total_trades_ejecutados" : len(trades_activos),
        "total_compras"           : len(compras),
        "total_ventas"            : len(ventas),
        "total_ignorados"         : len(ignorados),
        "ignorados_por_motivo"    : motivos,
        "positions_count_final"   : positions_count,
        "usdt_reserva_aplicada"   : round(USDT_RESERVA, 4),
        # Estadísticas del filtro umbral RSI
        "umbral_filtro": {
            "rsi_buy_trigger"               : RSI_BUY_TRIGGER,
            "rsi_sell_trigger"              : RSI_SELL_TRIGGER,
            "divergencias_compra_detectadas": n_div_compra_sin_umbral,
            "divergencias_compra_aprobadas" : n_div_compra_con_umbral,
            "tasa_rechazo_compra_pct"       : tasa_filtro_compra,
            "divergencias_venta_detectadas" : n_div_venta_sin_umbral,
            "divergencias_venta_aprobadas"  : n_div_venta_con_umbral,
            "tasa_rechazo_venta_pct"        : tasa_filtro_venta,
        },
        "parametros": {
            "rsi_length"           : RSI_LENGTH,
            "N"                    : N,
            "floor_pct"            : FLOOR_PCT,
            "factor_caida"         : FACTOR_CAIDA,
            "factor_subida"        : FACTOR_SUBIDA,
            "guardia_compra"       : GUARDIA_COMPRA,
            "guardia_precio_compra": GUARDIA_PRECIO_COMPRA,
            "guardia_precio_venta" : GUARDIA_PRECIO_VENTA,
            "rsi_buy_trigger"      : RSI_BUY_TRIGGER,
            "rsi_sell_trigger"     : RSI_SELL_TRIGGER,
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
    print("║  DIVERGENCIA RSI — UMBRAL DE ZONA                            ║")
    print("║  BTC/USDT · Backtesting                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    mostrar_configuracion()

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py")
        return

    print("Ejecutando estrategia...")
    results = ejecutar_estrategia(df)

    with open(RESULTS_JSON_UMBRAL, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Resultados guardados en: {RESULTS_JSON_UMBRAL}")

    s = results["summary"]
    u = s["umbral_filtro"]

    print("\n" + "═" * 62)
    print("  RESUMEN FINAL")
    print("═" * 62)
    print(f"  Período              : {s['fecha_inicio'][:10]}  →  {s['fecha_fin'][:10]}")
    print(f"  Capital inicial      : ${s['saldo_inicial_usdt']:>10,.2f}")
    print(f"  Portfolio final      : ${s['portfolio_value_final']:>10,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre           : ${s['usdt_balance_final']:>10,.4f}   (reserva: ${s['usdt_reserva_aplicada']:,.2f})")
    print(f"  BTC libre (acum.)    :  {s['btc_balance_final']:>.8f} ₿  ({s['btc_acumulado_total']:.8f} ₿)")
    print(f"  BTC en posiciones    :  {s['btc_en_posiciones_final']:>.8f} ₿")
    pp = s["precio_promedio_final"]
    if pp > 0:
        print(f"  Precio promedio BTC  : ${pp:>10,.2f}")
    print(f"  ATL registrado       : ${s['atl_final']:>10,.2f}")
    pmc = s.get("precio_min_comprado", 0)
    pmv = s.get("precio_max_vendido",  0)
    if pmc > 0: print(f"  Precio mín. comprado : ${pmc:>10,.2f}")
    if pmv > 0: print(f"  Precio máx. vendido  : ${pmv:>10,.2f}")
    print(f"  Trades ejecutados    :  {s['total_trades_ejecutados']}  "
          f"(compras: {s['total_compras']}  |  ventas: {s['total_ventas']})")
    print(f"  Señales ignoradas    :  {s['total_ignorados']}")
    for motivo, cnt in s.get("ignorados_por_motivo", {}).items():
        print(f"    · {motivo:<42}: {cnt}")

    print(f"\n  {'─'*60}")
    print(f"  FILTRO UMBRAL RSI")
    print(f"  {'─'*60}")
    print(f"  RSI_BUY_TRIGGER      : ≤ {u['rsi_buy_trigger']}")
    print(f"  RSI_SELL_TRIGGER     : ≥ {u['rsi_sell_trigger']}")
    print(f"  Divergencias compra  : {u['divergencias_compra_detectadas']} detectadas → "
          f"{u['divergencias_compra_aprobadas']} aprobadas  "
          f"({u['tasa_rechazo_compra_pct']:.1f}% rechazadas por umbral)")
    print(f"  Divergencias venta   : {u['divergencias_venta_detectadas']} detectadas → "
          f"{u['divergencias_venta_aprobadas']} aprobadas  "
          f"({u['tasa_rechazo_venta_pct']:.1f}% rechazadas por umbral)")
    print(f"  positions_count      :  {s['positions_count_final']:+d}  "
          f"{'(más compras)' if s['positions_count_final'] > 0 else '(más ventas)' if s['positions_count_final'] < 0 else '(equilibrado)'}")
    print("═" * 62)


if __name__ == "__main__":
    main()