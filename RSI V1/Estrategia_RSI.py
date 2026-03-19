"""
Estrategia RSI Bot — BTC/USDT Spot | Backtesting
─────────────────────────────────────────────────
Lógica de compra:
  · Trigger: RSI(low, RSI_LENGTH) <= LOW_RSI_BUY_TRIGGER
  · Usa USDT_PCT_TO_USE % del usdt_balance disponible
  · El BTC adquirido se acumula en btc_en_posiciones
  · positions_count += 1

Lógica de venta:
  · Trigger: RSI(high, RSI_LENGTH) >= HI_RSI_SELL_TRIGGER
  · Se procesa BTC_PCT_TO_SELL % del btc_en_posiciones
  · De ese BTC procesado:
      - BTC_PCT_TO_ACCUMULATE % → pasa a btc_balance libre (nunca se vende)
      - El resto               → se vende y vuelve como USDT
  · btc_en_posiciones se reduce pero nunca se fuerza a cero
  · positions_count -= 1

positions_count como indicador de sesgo:
  · Valor alto (+)  → muchas más compras que ventas → relajar HI_RSI_SELL_TRIGGER
  · Valor bajo (-)  → muchas más ventas que compras → relajar LOW_RSI_BUY_TRIGGER

Salida: strategy_results.json  (compatible con Graficador.py)
"""

import sqlite3
import json
import pandas as pd
import numpy as np

from config import (
    DB_PATH, RESULTS_JSON,
    SALDO_USDT_INICIAL, FECHA_INICIO, FECHA_FIN,
    RSI_LENGTH, LOW_RSI_BUY_TRIGGER, HI_RSI_SELL_TRIGGER,
    USDT_PCT_TO_USE, BTC_PCT_TO_SELL, BTC_PCT_TO_ACCUMULATE,
    COMMISSION_PCT,
)

import os
DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def calcular_rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI clásico de Wilder (EMA suavizada), idéntico al de TradingView."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cargar_datos() -> pd.DataFrame:
    """Carga las velas desde la DB SQLite y aplica el filtro de fechas."""
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

    # ── Calcular RSI sobre low y sobre high ──────────────────
    df["rsi_low"]  = calcular_rsi(df["low"],  RSI_LENGTH)
    df["rsi_high"] = calcular_rsi(df["high"], RSI_LENGTH)

    # ── Estado del portfolio ─────────────────────────────────
    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0    # BTC libre acumulado permanentemente
    btc_en_posiciones = 0.0    # BTC comprado aún no procesado en ventas
    positions_count   = 0      # Delta acumulado: +1 por compra, -1 por venta

    # Para precio promedio ponderado de posiciones abiertas
    usdt_invertido_en_posiciones = 0.0

    trade_history = []

    # ── Iterar velas ─────────────────────────────────────────
    for _, row in df.iterrows():

        rsi_l = row["rsi_low"]
        rsi_h = row["rsi_high"]

        if pd.isna(rsi_l) or pd.isna(rsi_h):
            continue

        price_low  = float(row["low"])
        price_high = float(row["high"])
        ts         = row["datetime"]

        # ── SEÑAL DE COMPRA ───────────────────────────────────
        if rsi_l <= LOW_RSI_BUY_TRIGGER and usdt_balance > 0:

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
                "rsi_low"                   : round(rsi_l, 4),
                "rsi_high"                  : round(rsi_h, 4),
                # compra
                "usdt_spent"                : round(usdt_a_usar, 4),
                "btc_bought"                : round(btc_adquirido, 8),
                "commission_usdt"           : round(comision, 4),
                # venta (vacío en compras)
                "btc_sold"                  : None,
                "btc_accumulated"           : None,
                "usdt_received"             : None,
                "ganancia_usdt"             : None,
                # estado del portfolio
                "usdt_balance"              : round(usdt_balance, 4),
                "btc_balance"               : round(btc_balance, 8),
                "btc_en_posiciones"         : round(btc_en_posiciones, 8),
                "positions_count"           : positions_count,
                "precio_promedio_posiciones": round(precio_promedio, 4),
            })

        # ── SEÑAL DE VENTA ────────────────────────────────────
        elif rsi_h >= HI_RSI_SELL_TRIGGER and btc_en_posiciones > 0:

            # 1. BTC a procesar en este trigger
            btc_procesado = btc_en_posiciones * (BTC_PCT_TO_SELL / 100)

            # 2. Fracción que se acumula permanentemente (no se vende)
            btc_a_acumular = btc_procesado * (BTC_PCT_TO_ACCUMULATE / 100)

            # 3. Fracción que efectivamente se vende
            btc_a_vender   = btc_procesado - btc_a_acumular
            usdt_bruto     = btc_a_vender * price_high
            comision       = usdt_bruto   * (COMMISSION_PCT / 100)
            usdt_neto      = usdt_bruto   - comision

            # 4. Actualizar balances
            btc_en_posiciones -= btc_procesado      # reduce en todo lo procesado
            btc_balance       += btc_a_acumular     # el % acumulado pasa a libre
            usdt_balance      += usdt_neto
            positions_count   -= 1

            # 5. Ganancia vs costo proporcional de las posiciones procesadas
            proporcion_procesada = btc_procesado / (btc_en_posiciones + btc_procesado)
            costo_procesado      = usdt_invertido_en_posiciones * proporcion_procesada
            usdt_invertido_en_posiciones -= costo_procesado
            ganancia             = usdt_neto - (costo_procesado * (1 - BTC_PCT_TO_ACCUMULATE / 100))

            # Precio promedio: recalcular si quedan posiciones abiertas
            precio_promedio = (usdt_invertido_en_posiciones / btc_en_posiciones
                               if btc_en_posiciones > 0 else 0.0)

            trade_history.append({
                "datetime"                  : ts.isoformat(),
                "type"                      : "SELL",
                "price"                     : price_high,
                "rsi_low"                   : round(rsi_l, 4),
                "rsi_high"                  : round(rsi_h, 4),
                # compra (vacío en ventas)
                "usdt_spent"                : None,
                "btc_bought"                : None,
                # venta
                "commission_usdt"           : round(comision, 4),
                "btc_sold"                  : round(btc_a_vender, 8),
                "btc_accumulated"           : round(btc_a_acumular, 8),
                "usdt_received"             : round(usdt_neto, 4),
                "ganancia_usdt"             : round(ganancia, 4),
                # estado del portfolio
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

    # BTC total acumulado libre a lo largo de toda la estrategia
    btc_acumulado_total = sum(t["btc_accumulated"] for t in ventas if t["btc_accumulated"])

    summary = {
        "estrategia"                : "RSI Bot BTC/USDT",
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
            "rsi_length"             : RSI_LENGTH,
            "low_rsi_buy_trigger"    : LOW_RSI_BUY_TRIGGER,
            "hi_rsi_sell_trigger"    : HI_RSI_SELL_TRIGGER,
            "usdt_pct_to_use"        : USDT_PCT_TO_USE,
            "btc_pct_to_sell"        : BTC_PCT_TO_SELL,
            "btc_pct_to_accumulate"  : BTC_PCT_TO_ACCUMULATE,
            "commission_pct"         : COMMISSION_PCT,
        },
    }

    return {
        "summary"      : summary,
        "trade_history": trade_history,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RSI BOT — BACKTESTING")
    print("=" * 60)

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
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Período           : {s['fecha_inicio']}  →  {s['fecha_fin']}")
    print(f"  Capital inicial   : ${s['saldo_inicial_usdt']:>12,.2f}")
    print(f"  Portfolio final   : ${s['portfolio_value_final']:>12,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre        : ${s['usdt_balance_final']:>12,.4f}")
    print(f"  BTC libre         :  {s['btc_balance_final']:>14.8f} ₿  (acumulado: {s['btc_acumulado_total']:.8f} ₿)")
    print(f"  BTC en posiciones :  {s['btc_en_posiciones_final']:>14.8f} ₿")
    print(f"  Compras           :  {s['total_compras']}")
    print(f"  Ventas            :  {s['total_ventas']}")
    print(f"  positions_count   :  {s['positions_count_final']:+d}  {'(más compras que ventas)' if s['positions_count_final'] > 0 else '(más ventas que compras)' if s['positions_count_final'] < 0 else '(equilibrado)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
