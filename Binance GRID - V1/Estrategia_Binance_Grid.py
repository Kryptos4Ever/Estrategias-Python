"""
Estrategia Binance Spot Grid Bot — Simulación local
══════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Backtesting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MECÁNICA (idéntica al bot de Binance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ESTRUCTURA DEL GRID:
  Se generan NUM_GRIDS+1 precios de referencia entre
  PRECIO_INFERIOR y PRECIO_SUPERIOR (aritmético o geométrico).
  Esto crea NUM_GRIDS intervalos, cada uno con:
    · Una orden de COMPRA en el precio inferior del intervalo.
    · Una orden de VENTA en el precio superior del intervalo.
        (= el precio inferior del intervalo siguiente)

SIZING:
  Capital uniforme: SALDO_USDT_INICIAL / NUM_GRIDS por nivel.
  Cada compra usa exactamente esa fracción del capital.

CICLO POR NIVEL:
  1. Orden de compra activa en precio_nivel[i].
  2. Si low ≤ precio_nivel[i] → compra ejecutada.
  3. Orden de venta se activa en precio_nivel[i+1].
  4. Si high ≥ precio_nivel[i+1] → venta ejecutada.
  5. Orden de compra reaparece en precio_nivel[i] → ciclo reinicia.

  El nivel i puede ciclar múltiples veces si el precio oscila
  repetidamente entre nivel[i] y nivel[i+1].

ESTADO INICIAL:
  Todas las órdenes de compra de todos los niveles se colocan
  simultáneamente al inicio. Las que están por debajo del precio
  actual se ejecutan en las primeras velas.

FUERA DE RANGO:
  · Precio < PRECIO_INFERIOR: todas las compras ejecutadas,
    el bot espera sin operar hasta que el precio suba.
  · Precio > PRECIO_SUPERIOR: todas las ventas ejecutadas,
    el bot espera sin operar hasta que el precio baje.

STOP LOSS / TAKE PROFIT GLOBAL:
  Si el precio cierra por debajo de STOP_LOSS o por encima de
  TAKE_PROFIT, todas las posiciones se liquidan a ese precio
  y el bot se detiene.

ORDEN INTRACANDLE:
  Vela alcista (close ≥ open): compras primero → ventas después.
  Vela bajista (close  < open): ventas primero → compras después.
  Las posiciones abiertas en esta vela NO evalúan su venta
  en la misma vela (idéntico a la estrategia UPO).

COMISIONES:
  Compra: se descuenta del BTC recibido.
    btc_neto = (usdt_gastado × (1 - COMMISSION_PCT/100)) / precio
  Venta: se descuenta del USDT recibido.
    usdt_neto = btc_vendido × precio × (1 - COMMISSION_PCT/100)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIFERENCIAS CON LA ESTRATEGIA UPO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  · Niveles fijos desde el inicio (no dinámicos desde ATH).
  · Múltiples órdenes activas simultáneamente (no 1 sola).
  · Sizing uniforme (no escalonado por profundidad de caída).
  · Cada nivel es independiente — no hay "posición global" con PP.
  · Stop Loss / Take Profit globales opcionales.
"""

import sqlite3
import json
import math
import os
import numpy as np
import pandas as pd

from config_binance_grid import (
    DB_PATH, RESULTS_JSON, FECHA_INICIO, FECHA_FIN,
    SALDO_USDT_INICIAL, USDT_RESERVA_PCT, COMMISSION_PCT,
    PRECIO_SUPERIOR, PRECIO_INFERIOR, NUM_GRIDS, MODO_GRID,
    STOP_LOSS, TAKE_PROFIT,
    mostrar_configuracion,
)

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERACIÓN DE NIVELES DEL GRID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generar_niveles() -> list[float]:
    """
    Genera NUM_GRIDS + 1 precios de referencia en [PRECIO_INFERIOR, PRECIO_SUPERIOR].

    Aritmético : precio[i] = PRECIO_INFERIOR + i × step
                 step = (PRECIO_SUPERIOR − PRECIO_INFERIOR) / NUM_GRIDS
    Geométrico : precio[i] = PRECIO_INFERIOR × ratio^i
                 ratio = (PRECIO_SUPERIOR / PRECIO_INFERIOR)^(1/NUM_GRIDS)

    Los niveles se devuelven en orden ASCENDENTE.
    nivel[i] es el precio de COMPRA del intervalo i.
    nivel[i+1] es el precio de VENTA del intervalo i (= TP de nivel[i]).
    """
    if MODO_GRID == "aritmetico":
        step   = (PRECIO_SUPERIOR - PRECIO_INFERIOR) / NUM_GRIDS
        niveles = [PRECIO_INFERIOR + i * step for i in range(NUM_GRIDS + 1)]
    else:  # geometrico
        ratio  = (PRECIO_SUPERIOR / PRECIO_INFERIOR) ** (1.0 / NUM_GRIDS)
        niveles = [PRECIO_INFERIOR * (ratio ** i) for i in range(NUM_GRIDS + 1)]

    return niveles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CARGA DE DATOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# HELPERS DE REGISTRO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _registro_compra(ts, nivel_idx, precio_compra, precio_tp,
                     usdt_gastado, comision, btc_neto,
                     usdt_balance, btc_libre, btc_en_pos,
                     n_pos_abiertas) -> dict:
    return {
        "datetime"            : ts.isoformat(),
        "type"                : "BUY",
        "nivel_idx"           : nivel_idx,
        "price"               : round(precio_compra, 4),
        "precio_tp"           : round(precio_tp,     4),
        "usdt_spent"          : round(usdt_gastado,  4),
        "commission_usdt"     : round(comision,       4),
        "btc_bought"          : round(btc_neto,       8),
        "btc_sold"            : None,
        "usdt_received"       : None,
        "ganancia_usdt"       : None,
        "usdt_balance"        : round(usdt_balance,   4),
        "btc_libre"           : round(btc_libre,      8),
        "btc_en_posiciones"   : round(btc_en_pos,     8),
        "positions_count"     : n_pos_abiertas,
        "ignorado"            : False,
        "motivo_ignorado"     : None,
    }


def _registro_venta(ts, nivel_idx, precio_venta, precio_compra_origen,
                    btc_vendido, comision, usdt_neto, ganancia,
                    usdt_balance, btc_libre, btc_en_pos,
                    n_pos_abiertas) -> dict:
    return {
        "datetime"            : ts.isoformat(),
        "type"                : "SELL",
        "nivel_idx"           : nivel_idx,
        "price"               : round(precio_venta,        4),
        "precio_tp"           : round(precio_venta,        4),
        "usdt_spent"          : None,
        "commission_usdt"     : round(comision,            4),
        "btc_bought"          : None,
        "btc_sold"            : round(btc_vendido,         8),
        "usdt_received"       : round(usdt_neto,           4),
        "ganancia_usdt"       : round(ganancia,            4),
        "usdt_balance"        : round(usdt_balance,        4),
        "btc_libre"           : round(btc_libre,           8),
        "btc_en_posiciones"   : round(btc_en_pos,          8),
        "positions_count"     : n_pos_abiertas,
        "ignorado"            : False,
        "motivo_ignorado"     : None,
    }


def _registro_liquidacion(ts, tipo, precio, btc_vendido, usdt_neto,
                           usdt_balance, btc_libre, motivo) -> dict:
    return {
        "datetime"            : ts.isoformat(),
        "type"                : tipo,    # "STOP_LOSS" | "TAKE_PROFIT"
        "nivel_idx"           : -1,
        "price"               : round(precio,      4),
        "precio_tp"           : round(precio,      4),
        "usdt_spent"          : None,
        "commission_usdt"     : None,
        "btc_bought"          : None,
        "btc_sold"            : round(btc_vendido, 8),
        "usdt_received"       : round(usdt_neto,   4),
        "ganancia_usdt"       : None,
        "usdt_balance"        : round(usdt_balance, 4),
        "btc_libre"           : round(btc_libre,   8),
        "btc_en_posiciones"   : 0.0,
        "positions_count"     : 0,
        "ignorado"            : False,
        "motivo_ignorado"     : motivo,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ESTRATEGIA PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ejecutar_estrategia(df: pd.DataFrame) -> dict:
    """
    Simula el Binance Spot Grid Bot vela a vela.

    Estado por nivel (dict):
      "buy_activa"  : bool   — hay una orden de compra pendiente en este nivel
      "sell_activa" : bool   — hay una orden de venta pendiente en este nivel+1
      "btc_cantidad": float  — BTC comprados (esperando venta)
      "usdt_origen" : float  — USDT usados en la compra (para calcular ganancia)
      "vela_compra" : int    — vela en la que se compró (filtro same-candle)
    """
    niveles       = generar_niveles()          # NUM_GRIDS + 1 precios
    n_intervalos  = NUM_GRIDS                  # = len(niveles) - 1

    capital_total   = float(SALDO_USDT_INICIAL) - USDT_RESERVA
    usdt_por_nivel  = capital_total / n_intervalos   # sizing uniforme
    comm_factor     = COMMISSION_PCT / 100.0

    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    dts    = df["datetime"].values

    # Estado inicial: todas las órdenes de compra activas
    # niveles[0] … niveles[n-1] son precios de compra
    # niveles[1] … niveles[n]   son los TP correspondientes
    estado = []
    for k in range(n_intervalos):
        estado.append({
            "precio_compra" : niveles[k],
            "precio_tp"     : niveles[k + 1],
            "buy_activa"    : True,
            "sell_activa"   : False,
            "btc_cantidad"  : 0.0,
            "usdt_origen"   : 0.0,
            "vela_compra"   : -1,
        })

    usdt_balance   = float(SALDO_USDT_INICIAL)
    btc_libre      = 0.0     # BTC acumulado de "sobrante" de ventas (no existe en Binance
                             # puro, pero lo registramos para comparabilidad con UPO)
    trade_history  = []
    bot_activo     = True    # False si stop_loss o take_profit se activan

    # Estadísticas
    precio_min_comprado = math.inf
    precio_max_vendido  = 0.0
    n_compras           = 0
    n_ventas            = 0
    n_stop_loss         = 0
    n_take_profit       = 0
    ganancia_ciclos     = 0.0

    n_velas = len(lows)

    for i in range(n_velas):
        if not bot_activo:
            break

        ts           = pd.Timestamp(dts[i])
        low_i        = float(lows[i])
        high_i       = float(highs[i])
        close_i      = float(closes[i])
        vela_alcista = closes[i] >= opens[i]

        # ── Comprobar Stop Loss / Take Profit sobre el cierre de la vela anterior
        # (Binance los evalúa continuamente; aproximamos con el precio de la vela)
        # ── Stop Loss ─────────────────────────────────────────────────────────
        if STOP_LOSS is not None and low_i <= STOP_LOSS:
            # Liquidar todas las posiciones abiertas al precio STOP_LOSS
            btc_total_en_pos = sum(e["btc_cantidad"] for e in estado if e["sell_activa"])
            if btc_total_en_pos > 0:
                usdt_bruto  = btc_total_en_pos * STOP_LOSS
                comision    = usdt_bruto * comm_factor
                usdt_neto   = usdt_bruto - comision
                usdt_balance += usdt_neto
                # Marcar todas las posiciones como cerradas
                for e in estado:
                    e["sell_activa"]  = False
                    e["buy_activa"]   = True
                    e["btc_cantidad"] = 0.0
                    e["usdt_origen"]  = 0.0
                trade_history.append(_registro_liquidacion(
                    ts, "STOP_LOSS", STOP_LOSS,
                    btc_total_en_pos, usdt_neto,
                    usdt_balance, btc_libre, "stop_loss_activado"
                ))
                n_stop_loss += 1
            bot_activo = False
            break

        # ── Take Profit ────────────────────────────────────────────────────────
        if TAKE_PROFIT is not None and high_i >= TAKE_PROFIT:
            btc_total_en_pos = sum(e["btc_cantidad"] for e in estado if e["sell_activa"])
            if btc_total_en_pos > 0:
                usdt_bruto  = btc_total_en_pos * TAKE_PROFIT
                comision    = usdt_bruto * comm_factor
                usdt_neto   = usdt_bruto - comision
                usdt_balance += usdt_neto
                for e in estado:
                    e["sell_activa"]  = False
                    e["buy_activa"]   = True
                    e["btc_cantidad"] = 0.0
                    e["usdt_origen"]  = 0.0
                trade_history.append(_registro_liquidacion(
                    ts, "TAKE_PROFIT", TAKE_PROFIT,
                    btc_total_en_pos, usdt_neto,
                    usdt_balance, btc_libre, "take_profit_activado"
                ))
                n_take_profit += 1
            bot_activo = False
            break

        # ── Helper: btc en posiciones activas ─────────────────────────────────
        def _btc_en_pos():
            return sum(e["btc_cantidad"] for e in estado if e["sell_activa"])

        # ── Procesamiento de compras ───────────────────────────────────────────
        def _procesar_compras():
            nonlocal usdt_balance, n_compras, precio_min_comprado
            for k, e in enumerate(estado):
                if not e["buy_activa"]:
                    continue
                if low_i > e["precio_compra"]:
                    continue
                # Precio alcanzado — ejecutar compra
                usdt_gasto  = usdt_por_nivel
                comision    = usdt_gasto * comm_factor
                btc_neto    = (usdt_gasto - comision) / e["precio_compra"]

                usdt_balance -= usdt_gasto
                e["buy_activa"]   = False
                e["sell_activa"]  = True
                e["btc_cantidad"] = btc_neto
                e["usdt_origen"]  = usdt_gasto
                e["vela_compra"]  = i

                if e["precio_compra"] < precio_min_comprado:
                    precio_min_comprado = e["precio_compra"]

                trade_history.append(_registro_compra(
                    ts, k, e["precio_compra"], e["precio_tp"],
                    usdt_gasto, comision, btc_neto,
                    usdt_balance, btc_libre, _btc_en_pos(), _n_pos()
                ))
                n_compras += 1

        # ── Procesamiento de ventas ────────────────────────────────────────────
        def _procesar_ventas():
            nonlocal usdt_balance, btc_libre, n_ventas, precio_max_vendido, ganancia_ciclos
            for k, e in enumerate(estado):
                if not e["sell_activa"]:
                    continue
                # FIX same-candle: no vender en la misma vela en que se compró
                if e["vela_compra"] == i:
                    continue
                if high_i < e["precio_tp"]:
                    continue
                # TP alcanzado — ejecutar venta
                # En Binance se vende EXACTAMENTE el BTC comprado en ese nivel
                btc_vendido = e["btc_cantidad"]
                usdt_bruto  = btc_vendido * e["precio_tp"]
                comision    = usdt_bruto * comm_factor
                usdt_neto   = usdt_bruto - comision
                ganancia    = usdt_neto - e["usdt_origen"]

                usdt_balance    += usdt_neto
                ganancia_ciclos += ganancia

                if e["precio_tp"] > precio_max_vendido:
                    precio_max_vendido = e["precio_tp"]

                trade_history.append(_registro_venta(
                    ts, k, e["precio_tp"], e["precio_compra"],
                    btc_vendido, comision, usdt_neto, ganancia,
                    usdt_balance, btc_libre, _btc_en_pos() - btc_vendido,
                    _n_pos() - 1
                ))
                n_ventas += 1

                # Reiniciar el nivel: orden de compra vuelve a estar activa
                e["sell_activa"]  = False
                e["buy_activa"]   = True
                e["btc_cantidad"] = 0.0
                e["usdt_origen"]  = 0.0
                e["vela_compra"]  = -1

        def _n_pos():
            return sum(1 for e in estado if e["sell_activa"])

        # ── Orden intracandle (idéntico a estrategia UPO) ──────────────────────
        if vela_alcista:
            _procesar_compras()
            _procesar_ventas()
        else:
            _procesar_ventas()
            _procesar_compras()

    # ── Resumen final ─────────────────────────────────────────────────────────
    precio_final     = float(closes[-1])
    btc_en_pos_final = sum(e["btc_cantidad"] for e in estado if e["sell_activa"])
    btc_total_final  = btc_libre + btc_en_pos_final
    portfolio_final  = usdt_balance + btc_total_final * precio_final
    pnl_pct          = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100

    # PP de posiciones abiertas
    usdt_en_pos       = sum(e["usdt_origen"] for e in estado if e["sell_activa"])
    precio_prom_final = (usdt_en_pos / btc_en_pos_final
                         if btc_en_pos_final > 0 else 0.0)

    # Niveles con posición abierta al cierre
    niveles_abiertos = [
        {
            "nivel_idx"     : k,
            "precio_compra" : round(e["precio_compra"], 4),
            "precio_tp"     : round(e["precio_tp"],     4),
            "btc_cantidad"  : round(e["btc_cantidad"],  8),
            "usdt_origen"   : round(e["usdt_origen"],   4),
        }
        for k, e in enumerate(estado) if e["sell_activa"]
    ]

    # Per-grid profit teórico
    if MODO_GRID == "geometrico":
        ratio = (PRECIO_SUPERIOR / PRECIO_INFERIOR) ** (1.0 / NUM_GRIDS)
        pgp   = (ratio - 1.0) * 100 - 2 * COMMISSION_PCT
    else:
        step = (PRECIO_SUPERIOR - PRECIO_INFERIOR) / NUM_GRIDS
        pgp  = None   # varía por nivel en modo aritmético

    summary = {
        "estrategia"              : "Binance Spot Grid Bot",
        "modo"                    : MODO_GRID,
        "fecha_inicio"            : str(df["datetime"].iloc[0]),
        "fecha_fin"               : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"      : SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(usdt_balance,           4),
        "btc_libre_final"         : round(btc_libre,              8),
        "btc_en_posiciones_final" : round(btc_en_pos_final,       8),
        "precio_promedio_final"   : round(precio_prom_final,      4),
        "portfolio_value_final"   : round(portfolio_final,        4),
        "pnl_pct"                 : round(pnl_pct,                4),
        "ganancia_ciclos_usdt"    : round(ganancia_ciclos,        4),
        "precio_min_comprado"     : round(precio_min_comprado
                                          if precio_min_comprado < math.inf
                                          else 0.0,               4),
        "precio_max_vendido"      : round(precio_max_vendido,     4),
        "total_compras"           : n_compras,
        "total_ventas"            : n_ventas,
        "total_trades"            : n_compras + n_ventas,
        "ciclos_completos"        : n_ventas,        # cada venta = 1 ciclo completo
        "niveles_abiertos_final"  : len(niveles_abiertos),
        "stop_loss_activado"      : n_stop_loss > 0,
        "take_profit_activado"    : n_take_profit > 0,
        "bot_detenido"            : not bot_activo,
        "per_grid_profit_neto"    : round(pgp, 4) if pgp is not None else "varía",
        "parametros": {
            "precio_superior"  : PRECIO_SUPERIOR,
            "precio_inferior"  : PRECIO_INFERIOR,
            "num_grids"        : NUM_GRIDS,
            "modo_grid"        : MODO_GRID,
            "usdt_por_nivel"   : round(usdt_por_nivel, 4),
            "stop_loss"        : STOP_LOSS,
            "take_profit"      : TAKE_PROFIT,
            "commission_pct"   : COMMISSION_PCT,
            "usdt_reserva_pct" : USDT_RESERVA_PCT,
        },
        "niveles_abiertos" : niveles_abiertos,
    }

    return {"summary": summary, "trade_history": trade_history}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  BINANCE SPOT GRID BOT — Simulación Local                    ║")
    print("║  BTC/USDT · Órdenes Límite · Grid Fijo                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    mostrar_configuracion()

    # Mostrar los primeros y últimos niveles del grid
    niveles = generar_niveles()
    print(f"  Niveles del grid generados ({len(niveles)} precios de referencia):")
    for k in range(min(4, NUM_GRIDS)):
        tp_pct = (niveles[k+1] / niveles[k] - 1) * 100
        print(f"    Grid {k+1:>3}: compra=${niveles[k]:>10,.2f}  →  venta=${niveles[k+1]:>10,.2f}"
              f"  (+{tp_pct:.2f}%)")
    if NUM_GRIDS > 8:
        print(f"    ... ({NUM_GRIDS - 8} niveles intermedios) ...")
    for k in range(max(4, NUM_GRIDS - 4), NUM_GRIDS):
        tp_pct = (niveles[k+1] / niveles[k] - 1) * 100
        print(f"    Grid {k+1:>3}: compra=${niveles[k]:>10,.2f}  →  venta=${niveles[k+1]:>10,.2f}"
              f"  (+{tp_pct:.2f}%)")
    print()

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config_binance_grid.py")
        return

    print("\nEjecutando estrategia...")
    results = ejecutar_estrategia(df)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Resultados guardados en: {RESULTS_JSON}")

    s = results["summary"]

    print("\n" + "═" * 66)
    print("  RESUMEN FINAL — BINANCE SPOT GRID")
    print("═" * 66)
    print(f"  Período              : {s['fecha_inicio'][:10]}  →  {s['fecha_fin'][:10]}")
    print(f"  Modo                 : {s['modo'].upper()}")
    print(f"  Capital inicial      : ${s['saldo_inicial_usdt']:>10,.2f}")
    print(f"  Portfolio final      : ${s['portfolio_value_final']:>10,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre           : ${s['usdt_balance_final']:>10,.4f}")
    print(f"  BTC en posiciones    :  {s['btc_en_posiciones_final']:>.8f} ₿"
          f"  ({s['niveles_abiertos_final']} niveles abiertos)")
    pp = s["precio_promedio_final"]
    if pp > 0:
        print(f"  Precio promedio pos. : ${pp:>10,.2f}")
    print(f"  Ganancia de ciclos   : ${s['ganancia_ciclos_usdt']:>10,.4f}"
          f"  (ciclos completos: {s['ciclos_completos']})")
    pmc = s.get("precio_min_comprado", 0)
    pmv = s.get("precio_max_vendido",  0)
    if pmc > 0: print(f"  Precio mín. comprado : ${pmc:>10,.2f}")
    if pmv > 0: print(f"  Precio máx. vendido  : ${pmv:>10,.2f}")
    pgp = s.get("per_grid_profit_neto")
    if pgp and pgp != "varía":
        print(f"  Per-grid profit neto : {pgp:>+10.4f}%  (por ciclo, neto de comisiones)")
    print(f"  Trades totales       :  {s['total_trades']}"
          f"  (compras: {s['total_compras']}  |  ventas: {s['total_ventas']})")
    if s["stop_loss_activado"]:
        print(f"  ⚠  STOP LOSS activado")
    if s["take_profit_activado"]:
        print(f"  ✓  TAKE PROFIT activado")
    print()
    if s["niveles_abiertos_final"] > 0:
        print(f"  Posiciones abiertas al cierre (precio final: ${float(0):,.2f}):")
        for pos in s["niveles_abiertos"][:5]:
            print(f"    Grid {pos['nivel_idx']+1:>3}: "
                  f"comprado a ${pos['precio_compra']:>10,.2f}  "
                  f"TP=${pos['precio_tp']:>10,.2f}  "
                  f"BTC={pos['btc_cantidad']:.6f}")
        if len(s["niveles_abiertos"]) > 5:
            print(f"    ... y {len(s['niveles_abiertos'])-5} más")
    print("═" * 66)


if __name__ == "__main__":
    main()
