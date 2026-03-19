"""
Estrategia Precio — Órdenes Límite · Gradiente Asintótico de Compra
══════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Backtesting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEÑALES (órdenes límite — congruente con producción)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPRA:
  Modo A — sin posiciones abiertas:
    limit_buy = ATH × (1 − PCT_CAIDA_ATH)
    Ejecución si low ≤ limit_buy

  Modo B — con posiciones abiertas:
    limit_buy = last_op_price × (1 − PCT_CAIDA)
    Ejecución si low ≤ limit_buy

  Precio de entrada = limit_buy  (no el low de la vela)
  last_op_price    = limit_buy   (se actualiza al precio de la orden)

VENTA (por posición individual):
  precio_tp = limit_buy_de_entrada × (1 + PCT_VENTA)
  Ejecución si high ≥ precio_tp

  Precio de venta  = precio_tp   (no el high de la vela)
  last_op_price    según LAST_OP_UPDATE_ON_SELL en config.

  BTC a vender: mínimo para recuperar usdt_invertido en esa posición
                incluye comisión de compra (ya pagada) y comisión de venta.
  BTC restante: se acumula como BTC libre.

REGLA last_op_price:
  Solo se actualiza cuando se ejecuta una COMPRA (= limit_buy de esa compra).
  Las ventas no lo modifican. De este modo, la cadena DCA siempre
  desciende desde el último precio de compra real, sin que las ventas
  "suban" el nivel de referencia.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIZING — GRADIENTE ASINTÓTICO DE COMPRA (sin cambios)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pct_capital_compra(limit_buy, ath) → % del capital disponible a usar.
  Ver config para parámetros de la curva.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORDEN INTRACANDLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Vela alcista (close ≥ open): evalúa compra → evalúa ventas
  Vela bajista (close  < open): evalúa ventas → evalúa compra

  Las posiciones creadas en la vela actual NO evalúan su TP
  en esa misma vela.

  Si múltiples TPs se alcanzan en la misma vela, todos se
  ejecutan en orden de más barato a más caro.
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
    PCT_CAIDA_ATH, PCT_CAIDA, PCT_VENTA,
    FLOOR_PCT,
    PENDIENTE_COMPRA,
    LAST_OP_UPDATE_ON_SELL,
    USDT_RESERVA_PCT,
    COMMISSION_PCT,
    mostrar_configuracion,
)

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROGRESIÓN LINEAL DE COMPRA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pct_capital_compra(limit_buy: float, ath: float) -> float:
    """
    Progresión lineal del % del capital disponible a usar.

      pos         = log(ATH / limit_buy) / log(100 / FLOOR_PCT)   ∈ [0, 1]
      pct_capital = min(pos × PENDIENTE_COMPRA, 100)

    pos = 0  (precio = ATH)      →  0%
    pos = 1  (precio = ATL_REF)  →  min(PENDIENTE_COMPRA, 100)%

    PENDIENTE_COMPRA = 100  →  línea recta 0%–100% entre ATH y piso.
    PENDIENTE_COMPRA > 100  →  satura al 100% antes de llegar al piso.
    PENDIENTE_COMPRA < 100  →  nunca llega al 100%.
    """
    if ath <= 0 or FLOOR_PCT <= 0 or limit_buy <= 0:
        return 0.0
    log_rango = math.log(100.0 / FLOOR_PCT)
    if log_rango <= 0:
        return 0.0
    pos = math.log(ath / limit_buy) / log_rango
    pos = max(0.0, min(1.0, pos))
    return min(pos * PENDIENTE_COMPRA, 100.0)


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

def _registro_compra(ts, limit_buy, pct, pos_grad,
                     usdt_a_usar, comision_compra, btc_adquirido,
                     precio_tp,
                     usdt_balance, btc_balance, btc_en_posiciones,
                     positions_count, ath, atl) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : "BUY",
        "price"                      : round(limit_buy,       4),
        "precio_tp"                  : round(precio_tp,       4),
        "ath"                        : round(ath,             4),
        "atl"                        : round(atl,             4),
        "pct_capital_usado"          : round(pct,             4),
        "pos_gradiente"              : round(pos_grad,        6),
        "usdt_spent"                 : round(usdt_a_usar,     4),
        "commission_compra_usdt"     : round(comision_compra, 4),
        "btc_bought"                 : round(btc_adquirido,   8),
        "btc_sold"                   : None,
        "commission_venta_usdt"      : None,
        "btc_accumulated"            : None,
        "usdt_received"              : None,
        "ganancia_usdt"              : None,
        "usdt_balance"               : round(usdt_balance,    4),
        "btc_balance"                : round(btc_balance,     8),
        "btc_en_posiciones"          : round(btc_en_posiciones, 8),
        "positions_count"            : positions_count,
        "ignorado"                   : False,
        "motivo_ignorado"            : None,
    }


def _registro_venta(ts, precio_tp, pos_grad,
                    btc_a_vender, comision_venta, usdt_neto,
                    btc_acumulado, ganancia,
                    usdt_balance, btc_balance, btc_en_posiciones,
                    positions_count, ath, atl) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : "SELL",
        "price"                      : round(precio_tp,       4),
        "precio_tp"                  : round(precio_tp,       4),
        "ath"                        : round(ath,             4),
        "atl"                        : round(atl,             4),
        "pct_capital_usado"          : None,
        "pos_gradiente"              : round(pos_grad,        6),
        "usdt_spent"                 : None,
        "commission_compra_usdt"     : None,
        "btc_sold"                   : round(btc_a_vender,    8),
        "commission_venta_usdt"      : round(comision_venta,  4),
        "btc_accumulated"            : round(btc_acumulado,   8),
        "usdt_received"              : round(usdt_neto,       4),
        "ganancia_usdt"              : round(ganancia,        4),
        "usdt_balance"               : round(usdt_balance,    4),
        "btc_balance"                : round(btc_balance,     8),
        "btc_en_posiciones"          : round(btc_en_posiciones, 8),
        "positions_count"            : positions_count,
        "ignorado"                   : False,
        "motivo_ignorado"            : None,
    }


def _registro_ignorado(ts, tipo, precio, motivo,
                        usdt_balance, btc_balance, btc_en_posiciones,
                        positions_count, ath, atl) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : tipo,
        "price"                      : round(precio,          4),
        "precio_tp"                  : None,
        "ath"                        : round(ath,             4),
        "atl"                        : round(atl,             4),
        "pct_capital_usado"          : None,
        "pos_gradiente"              : None,
        "usdt_spent"                 : None,
        "commission_compra_usdt"     : None,
        "btc_sold"                   : None,
        "commission_venta_usdt"      : None,
        "btc_accumulated"            : None,
        "usdt_received"              : None,
        "ganancia_usdt"              : None,
        "usdt_balance"               : round(usdt_balance,    4),
        "btc_balance"                : round(btc_balance,     8),
        "btc_en_posiciones"          : round(btc_en_posiciones, 8),
        "positions_count"            : positions_count,
        "ignorado"                   : True,
        "motivo_ignorado"            : motivo,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LÓGICA DE COMPRA Y VENTA  (funciones auxiliares para claridad)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _intentar_compra(i, ts, low, ath, atl,
                     posiciones, last_op_price,
                     usdt_balance, btc_balance,
                     trade_history):
    """
    Evalúa y ejecuta una orden límite de compra.

    Devuelve (posiciones, last_op_price, usdt_balance, compra_ejecutada).
    """
    # Calcular precio límite según modo
    if not posiciones:
        # Modo A: primera entrada desde ATH
        limit_buy = ath * (1.0 - PCT_CAIDA_ATH)
    else:
        # Modo B: DCA escalonado desde última operación
        limit_buy = last_op_price * (1.0 - PCT_CAIDA)

    # Verificar si la vela toca la orden
    if low > limit_buy:
        return posiciones, last_op_price, usdt_balance, False

    # Capital disponible (respetando reserva)
    usdt_disponible = usdt_balance - USDT_RESERVA
    if usdt_disponible <= 0:
        btc_en_pos = sum(p["btc_cantidad"] for p in posiciones)
        trade_history.append(_registro_ignorado(
            ts, "BUY", limit_buy, "sin_capital_sobre_reserva",
            usdt_balance, btc_balance, btc_en_pos, len(posiciones), ath, atl
        ))
        return posiciones, last_op_price, usdt_balance, False

    # Gradiente determina el % del capital disponible a usar
    pct         = pct_capital_compra(limit_buy, ath)
    usdt_a_usar = usdt_disponible * pct / 100.0

    if usdt_a_usar <= 0:
        btc_en_pos = sum(p["btc_cantidad"] for p in posiciones)
        trade_history.append(_registro_ignorado(
            ts, "BUY", limit_buy, "gradiente_cero",
            usdt_balance, btc_balance, btc_en_pos, len(posiciones), ath, atl
        ))
        return posiciones, last_op_price, usdt_balance, False

    # Ejecución al precio límite
    comision_compra = usdt_a_usar * (COMMISSION_PCT / 100.0)
    btc_adquirido   = (usdt_a_usar - comision_compra) / limit_buy
    precio_tp       = limit_buy * (1.0 + PCT_VENTA)

    # pos_gradiente para el log
    log_rango = math.log(100.0 / FLOOR_PCT)
    pos_grad  = round(math.log(ath / limit_buy) / log_rango, 6) if log_rango > 0 else 0.0

    # Actualizar estado
    usdt_balance -= usdt_a_usar
    last_op_price = limit_buy

    nueva_pos = {
        "precio_entrada"  : limit_buy,
        "usdt_invertido"  : usdt_a_usar,      # incluye la comisión de compra
        "btc_cantidad"    : btc_adquirido,     # BTC netos tras comisión de compra
        "precio_tp"       : precio_tp,
        "vela_creacion"   : i,                 # para no evaluar TP en la misma vela
    }
    posiciones.append(nueva_pos)

    btc_en_pos = sum(p["btc_cantidad"] for p in posiciones)
    trade_history.append(_registro_compra(
        ts, limit_buy, pct, pos_grad,
        usdt_a_usar, comision_compra, btc_adquirido,
        precio_tp,
        usdt_balance, btc_balance, btc_en_pos,
        len(posiciones), ath, atl
    ))

    return posiciones, last_op_price, usdt_balance, True


def _ejecutar_ventas(i, ts, high, ath, atl,
                     posiciones, last_op_price,
                     usdt_balance, btc_balance,
                     trade_history):
    """
    Evalúa todas las posiciones preexistentes (creadas en velas anteriores)
    y ejecuta las que alcanzan su TP.

    Devuelve (posiciones, last_op_price, usdt_balance, btc_balance).
    """
    # Solo evaluar posiciones creadas antes de esta vela
    posiciones_previas   = [p for p in posiciones if p["vela_creacion"] < i]
    posiciones_esta_vela = [p for p in posiciones if p["vela_creacion"] == i]

    posiciones_que_quedan = list(posiciones_esta_vela)  # las nuevas siempre sobreviven

    for pos in posiciones_previas:
        if high < pos["precio_tp"]:
            posiciones_que_quedan.append(pos)
            continue

        # TP alcanzado — ejecución al precio límite de venta
        precio_tp = pos["precio_tp"]

        # BTC a vender: recuperar usdt_invertido (que ya incluye la comisión de compra)
        # más cubrir la comisión de venta sobre el bruto.
        # Sea B = btc a vender, entonces:
        #   B × precio_tp × (1 − COMMISSION_PCT/100) = usdt_invertido
        #   B = usdt_invertido / (precio_tp × (1 − COMMISSION_PCT/100))
        factor_comision = 1.0 - COMMISSION_PCT / 100.0
        btc_a_vender    = pos["usdt_invertido"] / (precio_tp * factor_comision)

        # Salvaguarda: no vender más BTC de los que tiene la posición
        btc_a_vender  = min(btc_a_vender, pos["btc_cantidad"])
        usdt_bruto    = btc_a_vender * precio_tp
        comision_venta = usdt_bruto * (COMMISSION_PCT / 100.0)
        usdt_neto     = usdt_bruto - comision_venta

        btc_acumulado = pos["btc_cantidad"] - btc_a_vender
        ganancia      = usdt_neto - pos["usdt_invertido"]

        # pos_gradiente de la venta (posición relativa al rango de compra)
        log_rango = math.log(100.0 / FLOOR_PCT)
        pos_grad  = round(math.log(ath / precio_tp) / log_rango, 6) if log_rango > 0 else 0.0

        # Actualizar estado
        usdt_balance  += usdt_neto
        btc_balance   += btc_acumulado
        # LAST_OP_UPDATE_ON_SELL (config): si True, last_op_price sube al TP
        # de la venta. Si False (recomendado), last_op_price solo cambia en compras.
        if LAST_OP_UPDATE_ON_SELL:
            last_op_price = precio_tp

        btc_en_pos_restante = sum(p["btc_cantidad"] for p in posiciones_que_quedan)
        trade_history.append(_registro_venta(
            ts, precio_tp, pos_grad,
            btc_a_vender, comision_venta, usdt_neto,
            btc_acumulado, ganancia,
            usdt_balance, btc_balance, btc_en_pos_restante,
            len(posiciones_que_quedan), ath, atl
        ))

    return posiciones_que_quedan, last_op_price, usdt_balance, btc_balance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ESTRATEGIA PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ejecutar_estrategia(df: pd.DataFrame) -> dict:

    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    dts    = df["datetime"].values

    usdt_balance  = float(SALDO_USDT_INICIAL)
    btc_balance   = 0.0               # BTC libre (acumulado de ventas)
    posiciones    = []                 # posiciones abiertas individualmente
    last_op_price = None              # precio de la última operación ejecutada
    ath           = float(highs[0])
    atl           = float(lows[0])

    precio_min_comprado = math.inf
    precio_max_vendido  = 0.0

    trade_history = []
    n_velas = len(lows)

    for i in range(n_velas):
        ts = pd.Timestamp(dts[i])

        # ── Actualizar ATH / ATL ──────────────────────────────────────────────
        if highs[i] > ath:
            ath = float(highs[i])
        if lows[i] < atl:
            atl = float(lows[i])

        vela_alcista = closes[i] >= opens[i]

        if vela_alcista:
            # ── Vela alcista: low ocurrió antes → compra primero ─────────────
            posiciones, last_op_price, usdt_balance, compra_ok = _intentar_compra(
                i, ts, lows[i], ath, atl,
                posiciones, last_op_price,
                usdt_balance, btc_balance,
                trade_history
            )
            if compra_ok and precio_min_comprado > last_op_price:
                precio_min_comprado = last_op_price

            posiciones, last_op_price, usdt_balance, btc_balance = _ejecutar_ventas(
                i, ts, highs[i], ath, atl,
                posiciones, last_op_price,
                usdt_balance, btc_balance,
                trade_history
            )
            # Actualizar precio_max_vendido si hubo ventas
            ventas_esta_vela = [t for t in trade_history
                                 if t["type"] == "SELL"
                                 and t["datetime"] == ts.isoformat()
                                 and not t["ignorado"]]
            if ventas_esta_vela:
                precio_max_vendido = max(precio_max_vendido,
                                         max(v["price"] for v in ventas_esta_vela))

        else:
            # ── Vela bajista: high ocurrió antes → ventas primero ────────────
            posiciones, last_op_price, usdt_balance, btc_balance = _ejecutar_ventas(
                i, ts, highs[i], ath, atl,
                posiciones, last_op_price,
                usdt_balance, btc_balance,
                trade_history
            )
            ventas_esta_vela = [t for t in trade_history
                                 if t["type"] == "SELL"
                                 and t["datetime"] == ts.isoformat()
                                 and not t["ignorado"]]
            if ventas_esta_vela:
                precio_max_vendido = max(precio_max_vendido,
                                         max(v["price"] for v in ventas_esta_vela))

            posiciones, last_op_price, usdt_balance, compra_ok = _intentar_compra(
                i, ts, lows[i], ath, atl,
                posiciones, last_op_price,
                usdt_balance, btc_balance,
                trade_history
            )
            if compra_ok and precio_min_comprado > last_op_price:
                precio_min_comprado = last_op_price

    # ── Resumen final ─────────────────────────────────────────────────────────
    precio_final    = float(closes[-1])
    btc_en_pos_final = sum(p["btc_cantidad"] for p in posiciones)
    btc_total_final = btc_balance + btc_en_pos_final
    portfolio_final = usdt_balance + btc_total_final * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100

    # Precio promedio de posiciones abiertas (si las hay)
    usdt_invertido_abierto = sum(p["usdt_invertido"] for p in posiciones)
    precio_promedio_final  = (usdt_invertido_abierto / btc_en_pos_final
                               if btc_en_pos_final > 0 else 0.0)

    trades_activos = [t for t in trade_history if not t.get("ignorado", False)]
    compras        = [t for t in trades_activos if t["type"] == "BUY"]
    ventas         = [t for t in trades_activos if t["type"] == "SELL"]
    ignorados      = [t for t in trade_history  if t.get("ignorado",  False)]

    motivos = {}
    for t in ignorados:
        m = t.get("motivo_ignorado", "desconocido")
        motivos[m] = motivos.get(m, 0) + 1

    btc_acumulado_total = sum(
        t["btc_accumulated"] for t in ventas
        if t.get("btc_accumulated") is not None
    )

    summary = {
        "estrategia"              : "Precio — Órdenes Límite · Gradiente Asintótico de Compra",
        "fecha_inicio"            : str(df["datetime"].iloc[0]),
        "fecha_fin"               : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"      : SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(usdt_balance,           4),
        "btc_balance_final"       : round(btc_balance,            8),
        "btc_acumulado_total"     : round(btc_acumulado_total,    8),
        "btc_en_posiciones_final" : round(btc_en_pos_final,       8),
        "precio_promedio_final"   : round(precio_promedio_final,  4),
        "portfolio_value_final"   : round(portfolio_final,        4),
        "pnl_pct"                 : round(pnl_pct,                4),
        "precio_min_comprado"     : round(precio_min_comprado
                                          if precio_min_comprado < math.inf
                                          else 0.0,               4),
        "precio_max_vendido"      : round(precio_max_vendido,     4),
        "atl_final"               : round(atl,                    4),
        "total_trades_ejecutados" : len(trades_activos),
        "total_compras"           : len(compras),
        "total_ventas"            : len(ventas),
        "total_ignorados"         : len(ignorados),
        "ignorados_por_motivo"    : motivos,
        "positions_count_final"   : len(posiciones),
        "usdt_reserva_aplicada"   : round(USDT_RESERVA,           4),
        "parametros": {
            "pct_caida_ath"        : PCT_CAIDA_ATH,
            "pct_caida"            : PCT_CAIDA,
            "pct_venta"            : PCT_VENTA,
            "floor_pct"            : FLOOR_PCT,
            "pendiente_compra"       : PENDIENTE_COMPRA,
            "last_op_update_on_sell" : LAST_OP_UPDATE_ON_SELL,
            "usdt_reserva_pct"     : USDT_RESERVA_PCT,
            "commission_pct"       : COMMISSION_PCT,
        },
    }

    return {"summary": summary, "trade_history": trade_history}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PRECIO — ÓRDENES LÍMITE · GRADIENTE ASINTÓTICO DE COMPRA   ║")
    print("║  BTC/USDT · Backtesting                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    mostrar_configuracion()

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos. Revisar config_precio.py")
        return

    print("Ejecutando estrategia...")
    results = ejecutar_estrategia(df)

    output_path = RESULTS_JSON.replace(".json", "_precio.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Resultados guardados en: {output_path}")

    s = results["summary"]

    print("\n" + "═" * 62)
    print("  RESUMEN FINAL")
    print("═" * 62)
    print(f"  Período              : {s['fecha_inicio'][:10]}  →  {s['fecha_fin'][:10]}")
    print(f"  Capital inicial      : ${s['saldo_inicial_usdt']:>10,.2f}")
    print(f"  Portfolio final      : ${s['portfolio_value_final']:>10,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre           : ${s['usdt_balance_final']:>10,.4f}   (reserva: ${s['usdt_reserva_aplicada']:,.2f})")
    print(f"  BTC libre (acum.)    :  {s['btc_balance_final']:>.8f} ₿  ({s['btc_acumulado_total']:.8f} ₿ acumulado en ventas)")
    print(f"  BTC en posiciones    :  {s['btc_en_posiciones_final']:>.8f} ₿  ({s['positions_count_final']} posiciones abiertas)")
    pp = s["precio_promedio_final"]
    if pp > 0:
        print(f"  Precio promedio BTC  : ${pp:>10,.2f}  (posiciones abiertas)")
    print(f"  ATL registrado       : ${s['atl_final']:>10,.2f}")
    pmc = s.get("precio_min_comprado", 0)
    pmv = s.get("precio_max_vendido",  0)
    if pmc > 0: print(f"  Precio mín. comprado : ${pmc:>10,.2f}")
    if pmv > 0: print(f"  Precio máx. vendido  : ${pmv:>10,.2f}")
    print(f"  Trades ejecutados    :  {s['total_trades_ejecutados']}  "
          f"(compras: {s['total_compras']}  |  ventas: {s['total_ventas']})")
    print(f"  Señales ignoradas    :  {s['total_ignorados']}")
    for motivo, cnt in s.get("ignorados_por_motivo", {}).items():
        print(f"    · {motivo:<44}: {cnt}")
    print("═" * 62)


if __name__ == "__main__":
    main()