"""
Estrategia Divergencia RSI — Capital Dinámico por Racha
════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Backtesting

Todos los parámetros se leen desde config.py.
Salida: RESULTS_JSON definido en config.py (compatible con Graficador.py)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEÑALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPRA (divergencia alcista):
  · low[i]      < min(low[i-N : i])       → nuevo mínimo local de precio
  · RSI(low[i]) > RSI(low[idx_min])       → RSI no confirma ese mínimo
  → El precio cae más pero el momentum se agota → reversión al alza

VENTA (divergencia bajista):
  · high[i]      > max(high[i-N : i])     → nuevo máximo local de precio
  · RSI(high[i]) < RSI(high[idx_max])     → RSI no confirma ese máximo
  → El precio sube más pero el momentum se agota → reversión a la baja

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GESTIÓN DE CAPITAL POR RACHA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Al detectarse el PRIMER trade de una nueva racha se pre-calculan
STREAK slots sobre el capital disponible usando progresión geométrica
de razón R. Ver config.py para la fórmula y parámetros.

Si la racha supera STREAK → señal ignorada (registrada en el JSON).
Si la racha termina antes → capital no utilizado queda disponible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESERVA MÍNIMA DE USDT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El capital base para slots de compra se limita a:
  usdt_disponible = usdt_balance - usdt_reserva
donde usdt_reserva = SALDO_USDT_INICIAL × USDT_RESERVA_PCT / 100

Si usdt_disponible ≤ 0 → señal de compra ignorada.
La reserva garantiza capacidad de compra en caídas extremas prolongadas.
"""

import sqlite3
import json
import os
import numpy as np
import pandas as pd

# ── Importar TODOS los parámetros desde config.py ────────────────────────────
from config import (
    DB_PATH, RESULTS_JSON, FECHA_INICIO, FECHA_FIN,
    SALDO_USDT_INICIAL,
    RSI_LENGTH, N,
    STREAK_COMPRAS, STREAK_VENTAS, R_COMPRA, R_VENTA,
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


def calcular_slots(capital: float, n: int, r: float) -> list:
    """
    Calcula n slots en progresión geométrica de razón r
    tal que su suma sea exactamente igual a capital.

      a · S(r,n) = capital
      S(r,n) = (r^n - 1)/(r-1)   si r ≠ 1
             = n                   si r = 1

    Retorna: [a, a·r, a·r², ..., a·r^(n-1)]
    """
    if capital <= 0 or n <= 0:
        return []
    if abs(r - 1.0) < 1e-9:
        return [capital / n] * n
    sum_factor = (r ** n - 1) / (r - 1)
    slot_base  = capital / sum_factor
    return [slot_base * (r ** i) for i in range(n)]


def cargar_datos() -> pd.DataFrame:
    """Carga las velas desde la DB SQLite y aplica filtro de fechas."""
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
                        pos_count, motivo: str) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : tipo,
        "price"                      : precio,
        "rsi_low"                    : round(rsi_l, 4),
        "rsi_high"                   : round(rsi_h, 4),
        "usdt_spent"                 : None,
        "btc_bought"                 : None,
        "commission_usdt"            : None,
        "btc_sold"                   : None,
        "btc_accumulated"            : None,
        "usdt_received"              : None,
        "ganancia_usdt"              : None,
        "racha_pos"                  : None,
        "racha_slot_total"           : None,
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

    racha_tipo  = None
    racha_slots = []
    racha_idx   = 0

    trade_history = []
    n_velas = len(lows)

    for i in range(N, n_velas):

        if np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]):
            continue

        window_lows  = lows[i - N : i]
        window_highs = highs[i - N : i]
        ts           = pd.Timestamp(dts[i])

        # ── Detectar señal de COMPRA ──────────────────────────────────────────
        señal_compra = False
        if lows[i] < window_lows.min():
            idx_min = i - N + int(window_lows.argmin())
            if rsi_low[i] > rsi_low[idx_min]:
                señal_compra = True

        # ── Detectar señal de VENTA ───────────────────────────────────────────
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
                    positions_count, "sin_capital_sobre_reserva"
                ))
                continue

            if racha_tipo != "BUY":
                racha_slots = calcular_slots(usdt_disponible, STREAK_COMPRAS, R_COMPRA)
                racha_idx   = 0
                racha_tipo  = "BUY"

            if racha_idx >= len(racha_slots):
                trade_history.append(_registro_ignorado(
                    ts, "BUY", lows[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, "streak_superado"
                ))
                continue

            usdt_a_usar   = min(racha_slots[racha_idx], usdt_disponible)
            comision      = usdt_a_usar * (COMMISSION_PCT / 100)
            usdt_neto     = usdt_a_usar - comision
            price         = lows[i]
            btc_adquirido = usdt_neto / price

            usdt_balance      -= usdt_a_usar
            btc_en_posiciones += btc_adquirido
            usdt_invertido    += usdt_a_usar
            positions_count   += 1
            precio_promedio    = usdt_invertido / btc_en_posiciones

            trade_history.append({
                "datetime"                   : ts.isoformat(),
                "type"                       : "BUY",
                "price"                      : price,
                "rsi_low"                    : round(rsi_low[i],  4),
                "rsi_high"                   : round(rsi_high[i], 4),
                "usdt_spent"                 : round(usdt_a_usar,    4),
                "btc_bought"                 : round(btc_adquirido,  8),
                "commission_usdt"            : round(comision,       4),
                "btc_sold"                   : None,
                "btc_accumulated"            : None,
                "usdt_received"              : None,
                "ganancia_usdt"              : None,
                "racha_pos"                  : racha_idx + 1,
                "racha_slot_total"           : round(sum(racha_slots), 4),
                "usdt_balance"               : round(usdt_balance,        4),
                "btc_balance"                : round(btc_balance,          8),
                "btc_en_posiciones"          : round(btc_en_posiciones,    8),
                "positions_count"            : positions_count,
                "precio_promedio_posiciones" : round(precio_promedio,      4),
                "ignorado"                   : False,
                "motivo_ignorado"            : None,
            })
            racha_idx += 1

        # ─────────────────────────────────────────────────────────────────────
        # VENTA
        # ─────────────────────────────────────────────────────────────────────
        elif señal_venta and btc_en_posiciones > 0:

            if racha_tipo != "SELL":
                racha_slots = calcular_slots(btc_en_posiciones, STREAK_VENTAS, R_VENTA)
                racha_idx   = 0
                racha_tipo  = "SELL"

            if racha_idx >= len(racha_slots):
                trade_history.append(_registro_ignorado(
                    ts, "SELL", highs[i], rsi_low[i], rsi_high[i],
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, "streak_superado"
                ))
                continue

            btc_slot       = min(racha_slots[racha_idx], btc_en_posiciones)
            btc_a_acumular = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100)
            btc_a_vender   = btc_slot - btc_a_acumular
            price          = highs[i]
            usdt_bruto     = btc_a_vender * price
            comision       = usdt_bruto * (COMMISSION_PCT / 100)
            usdt_neto      = usdt_bruto - comision

            costo_proporcional = usdt_invertido * (btc_slot / btc_en_posiciones)
            ganancia           = usdt_neto - (costo_proporcional * (1 - BTC_PCT_TO_ACCUMULATE / 100))

            btc_en_posiciones -= btc_slot
            btc_balance       += btc_a_acumular
            usdt_balance      += usdt_neto
            usdt_invertido    -= costo_proporcional
            usdt_invertido     = max(usdt_invertido, 0.0)
            positions_count   -= 1
            precio_promedio    = (usdt_invertido / btc_en_posiciones
                                  if btc_en_posiciones > 0 else 0.0)

            trade_history.append({
                "datetime"                   : ts.isoformat(),
                "type"                       : "SELL",
                "price"                      : price,
                "rsi_low"                    : round(rsi_low[i],  4),
                "rsi_high"                   : round(rsi_high[i], 4),
                "usdt_spent"                 : None,
                "btc_bought"                 : None,
                "commission_usdt"            : round(comision,       4),
                "btc_sold"                   : round(btc_a_vender,   8),
                "btc_accumulated"            : round(btc_a_acumular, 8),
                "usdt_received"              : round(usdt_neto,      4),
                "ganancia_usdt"              : round(ganancia,        4),
                "racha_pos"                  : racha_idx + 1,
                "racha_slot_total"           : round(sum(racha_slots), 4),
                "usdt_balance"               : round(usdt_balance,        4),
                "btc_balance"                : round(btc_balance,          8),
                "btc_en_posiciones"          : round(btc_en_posiciones,    8),
                "positions_count"            : positions_count,
                "precio_promedio_posiciones" : round(precio_promedio,      4),
                "ignorado"                   : False,
                "motivo_ignorado"            : None,
            })
            racha_idx += 1

    # ── Resumen final ─────────────────────────────────────────────────────────
    precio_final    = float(closes[-1])
    btc_total_final = btc_balance + btc_en_posiciones
    portfolio_final = usdt_balance + btc_total_final * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100

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
        "estrategia"              : "Divergencia RSI — Capital Dinámico por Racha",
        "fecha_inicio"            : str(df["datetime"].iloc[0]),
        "fecha_fin"               : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"      : SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(usdt_balance,        4),
        "btc_balance_final"       : round(btc_balance,          8),
        "btc_acumulado_total"     : round(btc_acumulado_total,  8),
        "btc_en_posiciones_final" : round(btc_en_posiciones,    8),
        "portfolio_value_final"   : round(portfolio_final,      4),
        "pnl_pct"                 : round(pnl_pct,              4),
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
            "streak_compras"       : STREAK_COMPRAS,
            "streak_ventas"        : STREAK_VENTAS,
            "r_compra"             : R_COMPRA,
            "r_venta"              : R_VENTA,
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
    print("║  DIVERGENCIA RSI — CAPITAL DINÁMICO POR RACHA               ║")
    print("║  BTC/USDT · Backtesting                                     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    mostrar_configuracion()

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config.py")
        return

    print("\nEjecutando estrategia...")
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
    print(f"  Trades ejecutados    :  {s['total_trades_ejecutados']}  "
          f"(compras: {s['total_compras']}  |  ventas: {s['total_ventas']})")
    print(f"  Señales ignoradas    :  {s['total_ignorados']}")
    for motivo, cnt in s.get("ignorados_por_motivo", {}).items():
        print(f"    · {motivo:<35}: {cnt}")
    print(f"  positions_count      :  {s['positions_count_final']:+d}  "
          f"{'(más compras que ventas)' if s['positions_count_final'] > 0 else '(más ventas que compras)' if s['positions_count_final'] < 0 else '(equilibrado)'}")
    print("═" * 62)


if __name__ == "__main__":
    main()