"""
Estrategia EMA200 + Bollinger Bands + Williams %R — BTC/USDT Spot
──────────────────────────────────────────────────────────────────
Señal de COMPRA (las 3 deben cumplirse simultáneamente):
  · EMA200_dist  <= -DIST_EMA_BUY    precio >= X% BAJO la EMA200
  · Bollinger %B <=  BB_BUY          precio cerca/bajo banda inferior
  · Williams %R  <=  WILLIAMS_BUY    sobreventa confirmada

Señal de VENTA (las 3 deben cumplirse simultáneamente):
  · EMA200_dist  >=  DIST_EMA_SELL   precio >= X% SOBRE la EMA200
  · Bollinger %B >=  BB_SELL         precio cerca/sobre banda superior
  · Williams %R  >=  WILLIAMS_SELL   sobrecompra confirmada

Si ambas señales ocurren en la misma vela → prioridad COMPRA.

Precio de ejecución:
  · Compra → low de la vela
  · Venta  → high de la vela

Balance:
  · btc_en_posiciones: BTC acumulado en compras aún no procesadas
  · btc_balance:       BTC libre acumulado permanentemente (nunca se vende)
  · positions_count:   delta +1 por compra / -1 por venta
                       (alto = relajar trigger venta / bajo = relajar trigger compra)

Salida: strategy_results.json  (compatible con Graficador.py)
"""

import sqlite3
import json
import os
import numpy as np
import pandas as pd

from config_EMA_BB_Williams import (
    DB_PATH, RESULTS_JSON,
    SALDO_USDT_INICIAL, FECHA_INICIO, FECHA_FIN,
    EMA_LENGTH, DIST_EMA_BUY, DIST_EMA_SELL,
    BB_LENGTH, BB_STD, BB_BUY, BB_SELL,
    WILLIAMS_LENGTH, WILLIAMS_BUY, WILLIAMS_SELL,
    USDT_PCT_TO_USE, BTC_PCT_TO_SELL, BTC_PCT_TO_ACCUMULATE,
    COMMISSION_PCT,
)

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ─────────────────────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def calc_bollinger_pct_b(close: pd.Series, length: int, std_mult: float) -> pd.Series:
    mid   = close.rolling(length).mean()
    std   = close.rolling(length).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return (close - lower) / (upper - lower + 1e-10)


def calc_williams_r(high: pd.Series, low: pd.Series,
                    close: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

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
    print(f"Velas cargadas : {len(df):,}")
    print(f"Desde          : {df['datetime'].iloc[0]}")
    print(f"Hasta          : {df['datetime'].iloc[-1]}")
    return df


# ─────────────────────────────────────────────────────────────
# ESTRATEGIA
# ─────────────────────────────────────────────────────────────

def ejecutar_estrategia(df: pd.DataFrame) -> dict:

    # ── Calcular indicadores ─────────────────────────────────
    print("  Calculando EMA200...")
    ema200           = calc_ema(df["close"], EMA_LENGTH)
    df["ema200_dist"] = (df["close"] - ema200) / ema200 * 100   # % distancia

    print("  Calculando Bollinger %B...")
    df["bb_pct_b"]   = calc_bollinger_pct_b(df["close"], BB_LENGTH, BB_STD)

    print("  Calculando Williams %R...")
    df["williams_r"] = calc_williams_r(df["high"], df["low"], df["close"], WILLIAMS_LENGTH)

    # ── Señales ──────────────────────────────────────────────
    # Compra: precio bajo EMA200 (dist negativa), BB cerca inf, Williams sobreventa
    cond_compra = (
        (df["ema200_dist"] <= -DIST_EMA_BUY)  &
        (df["bb_pct_b"]   <=  BB_BUY)         &
        (df["williams_r"] <=  WILLIAMS_BUY)
    )

    # Venta: precio sobre EMA200 (dist positiva), BB cerca sup, Williams sobrecompra
    cond_venta = (
        (df["ema200_dist"] >=  DIST_EMA_SELL) &
        (df["bb_pct_b"]   >=  BB_SELL)        &
        (df["williams_r"] >=  WILLIAMS_SELL)
    )

    total_compras_posibles = cond_compra.sum()
    total_ventas_posibles  = cond_venta.sum()
    print(f"  Señales potenciales → Compras: {total_compras_posibles:,}  "
          f"Ventas: {total_ventas_posibles:,}")

    # ── Estado del portfolio ─────────────────────────────────
    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    positions_count   = 0
    usdt_invertido_en_posiciones = 0.0

    trade_history = []

    # ── Iterar velas ─────────────────────────────────────────
    for i, row in df.iterrows():

        # Saltar hasta que todos los indicadores tengan valor válido
        if pd.isna(row["ema200_dist"]) or pd.isna(row["bb_pct_b"]) or pd.isna(row["williams_r"]):
            continue

        es_compra = cond_compra.iloc[i]
        es_venta  = cond_venta.iloc[i]

        price_low  = float(row["low"])
        price_high = float(row["high"])
        ts         = row["datetime"]

        # ── COMPRA ───────────────────────────────────────────
        if es_compra and usdt_balance > 0:

            usdt_a_usar   = usdt_balance * (USDT_PCT_TO_USE / 100)
            comision      = usdt_a_usar  * (COMMISSION_PCT  / 100)
            usdt_neto     = usdt_a_usar  - comision
            btc_adquirido = usdt_neto / price_low

            usdt_balance                 -= usdt_a_usar
            btc_en_posiciones            += btc_adquirido
            usdt_invertido_en_posiciones += usdt_a_usar
            positions_count              += 1

            precio_promedio = usdt_invertido_en_posiciones / btc_en_posiciones

            trade_history.append({
                "datetime"                  : ts.isoformat(),
                "type"                      : "BUY",
                "price"                     : price_low,
                "ema200_dist"               : round(float(row["ema200_dist"]), 4),
                "bb_pct_b"                  : round(float(row["bb_pct_b"]), 4),
                "williams_r"                : round(float(row["williams_r"]), 4),
                "usdt_spent"                : round(usdt_a_usar, 4),
                "btc_bought"                : round(btc_adquirido, 8),
                "commission_usdt"           : round(comision, 4),
                "btc_sold"                  : None,
                "btc_accumulated"           : None,
                "usdt_received"             : None,
                "ganancia_usdt"             : None,
                "usdt_balance"              : round(usdt_balance, 4),
                "btc_balance"               : round(btc_balance, 8),
                "btc_en_posiciones"         : round(btc_en_posiciones, 8),
                "positions_count"           : positions_count,
                "precio_promedio_posiciones": round(precio_promedio, 4),
            })

        # ── VENTA ────────────────────────────────────────────
        elif es_venta and btc_en_posiciones > 0:

            btc_procesado  = btc_en_posiciones * (BTC_PCT_TO_SELL      / 100)
            btc_a_acumular = btc_procesado     * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_a_vender   = btc_procesado - btc_a_acumular

            usdt_bruto     = btc_a_vender * price_high
            comision       = usdt_bruto   * (COMMISSION_PCT / 100)
            usdt_neto      = usdt_bruto   - comision

            # Actualizar balances
            btc_en_posiciones -= btc_procesado
            btc_balance       += btc_a_acumular
            usdt_balance      += usdt_neto
            positions_count   -= 1

            # Ganancia vs costo proporcional
            proporcion_procesada         = btc_procesado / (btc_en_posiciones + btc_procesado)
            costo_procesado              = usdt_invertido_en_posiciones * proporcion_procesada
            usdt_invertido_en_posiciones -= costo_procesado
            ganancia = usdt_neto - (costo_procesado * (1 - BTC_PCT_TO_ACCUMULATE / 100))

            precio_promedio = (usdt_invertido_en_posiciones / btc_en_posiciones
                               if btc_en_posiciones > 0 else 0.0)

            trade_history.append({
                "datetime"                  : ts.isoformat(),
                "type"                      : "SELL",
                "price"                     : price_high,
                "ema200_dist"               : round(float(row["ema200_dist"]), 4),
                "bb_pct_b"                  : round(float(row["bb_pct_b"]), 4),
                "williams_r"                : round(float(row["williams_r"]), 4),
                "usdt_spent"                : None,
                "btc_bought"                : None,
                "commission_usdt"           : round(comision, 4),
                "btc_sold"                  : round(btc_a_vender, 8),
                "btc_accumulated"           : round(btc_a_acumular, 8),
                "usdt_received"             : round(usdt_neto, 4),
                "ganancia_usdt"             : round(ganancia, 4),
                "usdt_balance"              : round(usdt_balance, 4),
                "btc_balance"               : round(btc_balance, 8),
                "btc_en_posiciones"         : round(btc_en_posiciones, 8),
                "positions_count"           : positions_count,
                "precio_promedio_posiciones": round(precio_promedio, 4),
            })

    # ── Resumen final ─────────────────────────────────────────
    btc_total_final = btc_balance + btc_en_posiciones
    precio_final    = float(df["close"].iloc[-1])
    portfolio_final = usdt_balance + btc_total_final * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100

    compras = [t for t in trade_history if t["type"] == "BUY"]
    ventas  = [t for t in trade_history if t["type"] == "SELL"]
    btc_acumulado_total = sum(t["btc_accumulated"] for t in ventas
                              if t["btc_accumulated"] is not None)

    summary = {
        "estrategia"                : "EMA200 + Bollinger + Williams BTC/USDT",
        "fecha_inicio"              : str(df["datetime"].iloc[0]),
        "fecha_fin"                 : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"        : SALDO_USDT_INICIAL,
        "usdt_balance_final"        : round(usdt_balance, 4),
        "btc_balance_final"         : round(btc_balance, 8),
        "btc_acumulado_total"       : round(btc_acumulado_total, 8),
        "btc_en_posiciones_final"   : round(btc_en_posiciones, 8),
        "portfolio_value_final"     : round(portfolio_final, 4),
        "pnl_pct"                   : round(pnl_pct, 4),
        "total_trades"              : len(trade_history),
        "total_compras"             : len(compras),
        "total_ventas"              : len(ventas),
        "positions_count_final"     : positions_count,
        "parametros": {
            "ema_length"             : EMA_LENGTH,
            "dist_ema_buy"           : DIST_EMA_BUY,
            "dist_ema_sell"          : DIST_EMA_SELL,
            "bb_length"              : BB_LENGTH,
            "bb_std"                 : BB_STD,
            "bb_buy"                 : BB_BUY,
            "bb_sell"                : BB_SELL,
            "williams_length"        : WILLIAMS_LENGTH,
            "williams_buy"           : WILLIAMS_BUY,
            "williams_sell"          : WILLIAMS_SELL,
            "usdt_pct_to_use"        : USDT_PCT_TO_USE,
            "btc_pct_to_sell"        : BTC_PCT_TO_SELL,
            "btc_pct_to_accumulate"  : BTC_PCT_TO_ACCUMULATE,
            "commission_pct"         : COMMISSION_PCT,
        },
    }

    return {"summary": summary, "trade_history": trade_history}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  ESTRATEGIA EMA200 + BOLLINGER + WILLIAMS — BACKTESTING")
    print("=" * 62)
    print(f"\n  Parámetros de señal:")
    print(f"  COMPRA  → EMA200_dist ≤ -{DIST_EMA_BUY}%  "
          f"AND BB_%B ≤ {BB_BUY}  AND Williams ≤ {WILLIAMS_BUY}")
    print(f"  VENTA   → EMA200_dist ≥ +{DIST_EMA_SELL}%  "
          f"AND BB_%B ≥ {BB_SELL}  AND Williams ≥ {WILLIAMS_SELL}")

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos en la DB para el rango indicado.")
        return

    print("\nEjecutando estrategia...")
    results = ejecutar_estrategia(df)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en: {RESULTS_JSON}")

    s = results["summary"]
    print("\n" + "=" * 62)
    print("  RESUMEN FINAL")
    print("=" * 62)
    print(f"  Período           : {s['fecha_inicio']}  →  {s['fecha_fin']}")
    print(f"  Capital inicial   : ${s['saldo_inicial_usdt']:>12,.2f}")
    print(f"  Portfolio final   : ${s['portfolio_value_final']:>12,.2f}"
          f"   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre        : ${s['usdt_balance_final']:>12,.4f}")
    print(f"  BTC libre         :  {s['btc_balance_final']:>14.8f} ₿"
          f"  (acumulado: {s['btc_acumulado_total']:.8f} ₿)")
    print(f"  BTC en posiciones :  {s['btc_en_posiciones_final']:>14.8f} ₿")
    print(f"  Compras           :  {s['total_compras']:,}")
    print(f"  Ventas            :  {s['total_ventas']:,}")
    print(f"  positions_count   :  {s['positions_count_final']:+d}  "
          f"{'(más compras que ventas)' if s['positions_count_final'] > 0 else '(más ventas que compras)' if s['positions_count_final'] < 0 else '(equilibrado)'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
