"""
Estrategia RSI Bot — BTC/USDT Spot | Backtesting
─────────────────────────────────────────────────
Lógica:
  · Compra cuando RSI(low,  RSI_LENGTH) <= LOW_RSI_BUY_TRIGGER
  · Vende cuando RSI(high, RSI_LENGTH) >= HI_RSI_SELL_TRIGGER

Salida: strategy_results.json  (compatible con Graficador.py)
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    DB_PATH, RESULTS_JSON,
    SALDO_USDT_INICIAL, FECHA_INICIO, FECHA_FIN,
    RSI_LENGTH, LOW_RSI_BUY_TRIGGER, HI_RSI_SELL_TRIGGER,
    USDT_PCT_TO_USE, BTC_PCT_TO_SELL,
    COMMISSION_PCT,
)

import os
DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def calcular_rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI clásico de Wilder (EMA suavizada)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
    print(f"Velas cargadas: {len(df):,}  "
          f"({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")
    return df


# ─────────────────────────────────────────────────────────────
# ESTRATEGIA
# ─────────────────────────────────────────────────────────────

def ejecutar_estrategia(df: pd.DataFrame) -> dict:
    """
    Itera vela a vela simulando el bot.
    Devuelve el dict listo para serializar como strategy_results.json.
    """

    # ── Calcular RSI sobre low y sobre high ──────────────────
    df["rsi_low"]  = calcular_rsi(df["low"],  RSI_LENGTH)
    df["rsi_high"] = calcular_rsi(df["high"], RSI_LENGTH)

    # ── Estado del portfolio ─────────────────────────────────
    usdt_balance = float(SALDO_USDT_INICIAL)
    btc_balance  = 0.0          # BTC disponible (libre)
    btc_en_posiciones = 0.0     # BTC recién comprado aún no vendido

    trade_history = []

    # Tracking para estadísticas
    total_usdt_invertido = 0.0
    total_ganancia_usdt  = 0.0

    # ── Iterar velas ─────────────────────────────────────────
    for i, row in df.iterrows():

        rsi_l = row["rsi_low"]
        rsi_h = row["rsi_high"]

        # Saltar velas sin RSI válido (warm-up)
        if pd.isna(rsi_l) or pd.isna(rsi_h):
            continue

        price    = float(row["close"])
        ts       = row["datetime"]

        # ── SEÑAL DE COMPRA ───────────────────────────────────
        if rsi_l <= LOW_RSI_BUY_TRIGGER and usdt_balance > 0:

            usdt_a_usar   = usdt_balance * (USDT_PCT_TO_USE / 100)
            comision      = usdt_a_usar  * (COMMISSION_PCT  / 100)
            usdt_neto     = usdt_a_usar  - comision
            btc_adquirido = usdt_neto / price

            usdt_balance      -= usdt_a_usar
            btc_en_posiciones += btc_adquirido
            total_usdt_invertido += usdt_a_usar

            trade_history.append({
                "datetime"               : ts.isoformat(),
                "type"                   : "BUY",
                "price"                  : price,
                "rsi_low"                : round(rsi_l, 4),
                "rsi_high"               : round(rsi_h, 4),
                "usdt_spent"             : round(usdt_a_usar, 4),
                "btc_bought"             : round(btc_adquirido, 8),
                "commission_usdt"        : round(comision, 4),
                "usdt_balance"           : round(usdt_balance, 4),
                "btc_balance"            : round(btc_balance, 8),
                "btc_en_posiciones"      : round(btc_en_posiciones, 8),
                "positions_count"        : 1 if btc_en_posiciones > 0 else 0,
                "precio_promedio_posiciones": price,
                "ganancia_usdt"          : None,
            })

        # ── SEÑAL DE VENTA ────────────────────────────────────
        elif rsi_h >= HI_RSI_SELL_TRIGGER and btc_en_posiciones > 0:

            # Sólo vendemos el BTC que está en posiciones abiertas
            btc_a_vender  = btc_en_posiciones * (BTC_PCT_TO_SELL / 100)
            usdt_bruto    = btc_a_vender * price
            comision      = usdt_bruto * (COMMISSION_PCT / 100)
            usdt_neto     = usdt_bruto - comision

            # El BTC vendido sale de posiciones; la ganancia va a USDT
            # El remanente (si BTC_PCT_TO_SELL < 100) pasa a btc_balance libre
            btc_remanente       = btc_en_posiciones - btc_a_vender
            btc_en_posiciones   = 0.0
            btc_balance        += btc_remanente
            usdt_balance       += usdt_neto

            ganancia = usdt_neto - (btc_a_vender * price / (1 + COMMISSION_PCT / 100))
            total_ganancia_usdt += usdt_neto
            total_usdt_invertido = max(total_usdt_invertido - usdt_neto, 0)

            trade_history.append({
                "datetime"               : ts.isoformat(),
                "type"                   : "SELL",
                "price"                  : price,
                "rsi_low"                : round(rsi_l, 4),
                "rsi_high"               : round(rsi_h, 4),
                "usdt_spent"             : None,
                "btc_bought"             : None,
                "commission_usdt"        : round(comision, 4),
                "btc_sold"               : round(btc_a_vender, 8),
                "usdt_received"          : round(usdt_neto, 4),
                "usdt_balance"           : round(usdt_balance, 4),
                "btc_balance"            : round(btc_balance, 8),
                "btc_en_posiciones"      : round(btc_en_posiciones, 8),
                "positions_count"        : 1 if btc_en_posiciones > 0 else 0,
                "precio_promedio_posiciones": 0.0,
                "ganancia_usdt"          : round(usdt_neto, 4),
            })

    # ── Resumen final ─────────────────────────────────────────
    btc_total_final  = btc_balance + btc_en_posiciones
    precio_final     = float(df["close"].iloc[-1])
    portfolio_final  = usdt_balance + btc_total_final * precio_final
    pnl_pct          = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100

    compras = [t for t in trade_history if t["type"] == "BUY"]
    ventas  = [t for t in trade_history if t["type"] == "SELL"]

    summary = {
        "estrategia"          : "RSI Bot BTC/USDT",
        "fecha_inicio"        : str(df["datetime"].iloc[0]),
        "fecha_fin"           : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"  : SALDO_USDT_INICIAL,
        "usdt_balance_final"  : round(usdt_balance, 4),
        "btc_balance_final"   : round(btc_balance, 8),
        "btc_en_posiciones_final": round(btc_en_posiciones, 8),
        "portfolio_value_final": round(portfolio_final, 4),
        "pnl_pct"             : round(pnl_pct, 4),
        "total_trades"        : len(trade_history),
        "total_compras"       : len(compras),
        "total_ventas"        : len(ventas),
        # Parámetros usados (útil para comparar corridas)
        "parametros": {
            "rsi_length"          : RSI_LENGTH,
            "low_rsi_buy_trigger" : LOW_RSI_BUY_TRIGGER,
            "hi_rsi_sell_trigger" : HI_RSI_SELL_TRIGGER,
            "usdt_pct_to_use"     : USDT_PCT_TO_USE,
            "btc_pct_to_sell"     : BTC_PCT_TO_SELL,
            "commission_pct"      : COMMISSION_PCT,
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

    # 1. Cargar datos
    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos en la DB para el rango indicado.")
        return

    # 2. Ejecutar estrategia
    print("\nEjecutando estrategia...")
    results = ejecutar_estrategia(df)

    # 3. Guardar JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en: {RESULTS_JSON}")

    # 4. Imprimir resumen
    s = results["summary"]
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Período        : {s['fecha_inicio']}  →  {s['fecha_fin']}")
    print(f"  Capital inicial: ${s['saldo_inicial_usdt']:>12,.2f}")
    print(f"  Portfolio final: ${s['portfolio_value_final']:>12,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre     : ${s['usdt_balance_final']:>12,.4f}")
    print(f"  BTC libre      :  {s['btc_balance_final']:>14.8f} ₿")
    print(f"  BTC posiciones :  {s['btc_en_posiciones_final']:>14.8f} ₿")
    print(f"  Compras        :  {s['total_compras']}")
    print(f"  Ventas         :  {s['total_ventas']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
