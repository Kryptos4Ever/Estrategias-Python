"""
Estrategia Grid ATH/ATL — Serie Geometrica con Flags
=====================================================
COMPRAS:
  · Niveles logaritmicos desde ATH hacia abajo cada PASO_PCT_COMPRA%
    hasta CAIDA_MAXIMA%.
  · Montos en serie geometrica: r = FACTOR_COMPRA^(1/(n-1))
    suma exacta = usdt_balance al momento del nuevo ATH.
  · Cada orden tiene flag LIBRE/USADO.
    - Al ejecutarse: flag → USADO
    - Se libera (USADO→LIBRE) cuando se ejecuta cualquier venta
      con precio_venta > nivel_compra
    - Nuevo ATH: tabla completa nueva, todos LIBRE
  · GUARDIA: no comprar si nivel >= precio_promedio_posiciones

VENTAS:
  · Niveles logaritmicos desde precio_promedio hacia arriba cada
    PASO_PCT_VENTA% hasta SUBIDA_MAXIMA%.
  · Mismo criterio geometrico con FACTOR_VENTA y btc_en_posiciones.
  · Cada orden tiene flag LIBRE/USADO.
    - Al ejecutarse: flag → USADO
    - Se libera cuando se ejecuta cualquier compra
      con precio_compra < nivel_venta
    - Nuevo ATL: tabla completa nueva, todos LIBRE
  · GUARDIA: no vender si nivel <= precio_promedio_posiciones

Salida: strategy_results.json
"""

import sqlite3
import json
import os
import math
import pandas as pd

from config import (
    DB_PATH, RESULTS_JSON,
    SALDO_USDT_INICIAL, FECHA_INICIO, FECHA_FIN,
    PASO_PCT_COMPRA, CAIDA_MAXIMA, FACTOR_COMPRA,
    PASO_PCT_VENTA,  SUBIDA_MAXIMA, FACTOR_VENTA,
    MIN_ACUMULAR_PCT, MAX_ACUMULAR_PCT,
    COMMISSION_PCT,
)

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _niveles_log(p_ini: float, p_fin: float,
                 paso_pct: float, direccion: str) -> list:
    precios = []
    factor  = (1.0 - paso_pct / 100.0) if direccion == "abajo" \
              else (1.0 + paso_pct / 100.0)
    p = p_ini * factor
    while True:
        if direccion == "abajo"  and p <= p_fin: break
        if direccion == "arriba" and p >= p_fin: break
        precios.append(p)
        p *= factor
    return precios


def _serie_geo(budget: float, factor: float, n: int) -> list:
    if n == 1:
        return [budget]
    r     = factor ** (1.0 / (n - 1))
    first = budget * (r - 1.0) / (r**n - 1.0)
    return [first * (r ** i) for i in range(n)]


# ─────────────────────────────────────────────────────────────
# CONSTRUCCION DE TABLAS
# ─────────────────────────────────────────────────────────────

def construir_tabla_compras(ath: float, usdt_balance: float) -> list:
    """
    Retorna lista ordenada de mayor a menor precio.
    Cada orden: { precio, monto_objetivo, distancia_pct, usado }
    """
    p_fin   = ath * (1.0 - CAIDA_MAXIMA / 100.0)
    precios = _niveles_log(ath, p_fin, PASO_PCT_COMPRA, "abajo")
    n       = len(precios)
    if n == 0:
        return []
    montos = _serie_geo(usdt_balance, FACTOR_COMPRA, n)
    return [
        {
            "precio"        : p,
            "monto_objetivo": m,
            "distancia_pct" : round((1.0 - p / ath) * 100, 4),
            "usado"         : False,
        }
        for p, m in zip(precios, montos)
    ]


def construir_tabla_ventas(precio_promedio: float,
                           btc_en_posiciones: float,
                           atl: float) -> list:
    """
    Retorna lista ordenada de menor a mayor precio.
    Ancla: precio_promedio (primer nivel = precio_prom * (1+PASO))
    Techo: atl * (1 + SUBIDA_MAXIMA/100)
           → SUBIDA_MAXIMA es % de subida desde el ATL historico,
             no desde el precio promedio.
    Cada orden: { precio, btc_objetivo, acumular_pct, distancia_pct, usado }
    """
    if precio_promedio <= 0 or btc_en_posiciones <= 0 or atl <= 0:
        return []
    p_fin   = atl * (1.0 + SUBIDA_MAXIMA / 100.0)
    if p_fin <= precio_promedio:
        return []  # el techo ya quedo por debajo del precio promedio
    precios = _niveles_log(precio_promedio, p_fin, PASO_PCT_VENTA, "arriba")
    n       = len(precios)
    if n == 0:
        return []
    montos  = _serie_geo(btc_en_posiciones, FACTOR_VENTA, n)
    ordenes = []
    for i, (p, m) in enumerate(zip(precios, montos)):
        t            = i / (n - 1) if n > 1 else 0.0
        acumular_pct = MAX_ACUMULAR_PCT - t * (MAX_ACUMULAR_PCT - MIN_ACUMULAR_PCT)
        ordenes.append({
            "precio"       : p,
            "btc_objetivo" : m,
            "acumular_pct" : round(acumular_pct, 4),
            "distancia_pct": round((p / precio_promedio - 1.0) * 100, 4),
            "usado"        : False,
        })
    return ordenes


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, open, high, low, close
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

    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    positions_count   = 0

    ath = None
    atl = None
    ordenes_compra = []   # mayor a menor precio
    ordenes_venta  = []   # menor a mayor precio

    trade_history = []
    n = len(df)
    print("Ejecutando estrategia...")

    for i in range(n):
        row        = df.iloc[i]
        ts         = row["datetime"]
        high_vela  = float(row["high"])
        low_vela   = float(row["low"])
        close_vela = float(row["close"])
        open_vela  = float(row["open"])

        precio_prom = usdt_invertido / btc_en_posiciones \
                      if btc_en_posiciones > 0 else 0.0

        # ── Nuevo ATH: tabla de compras nueva, todas LIBRE ───
        if ath is None or high_vela > ath:
            ath = high_vela
            ordenes_compra = construir_tabla_compras(ath, usdt_balance)

        # ── Nuevo ATL: tabla de ventas nueva, todas LIBRE ────
        if atl is None or low_vela < atl:
            atl = low_vela
            if btc_en_posiciones > 0 and precio_prom > 0:
                ordenes_venta = construir_tabla_ventas(precio_prom,
                                                       btc_en_posiciones,
                                                       atl)

        if i == 0:
            continue

        precio_prom = usdt_invertido / btc_en_posiciones \
                      if btc_en_posiciones > 0 else 0.0

        # ── EJECUTAR COMPRAS ─────────────────────────────────
        for orden in ordenes_compra:
            nivel = orden["precio"]

            # Saltar si ya usada
            if orden["usado"]:
                continue

            # Guardia: no comprar si nivel >= precio_promedio
            if precio_prom > 0 and nivel >= precio_prom:
                continue

            if open_vela > nivel >= low_vela:
                if usdt_balance <= 0:
                    continue

                usdt_a_usar   = min(orden["monto_objetivo"], usdt_balance)
                if usdt_a_usar <= 0:
                    continue

                comision      = usdt_a_usar * (COMMISSION_PCT / 100.0)
                usdt_neto     = usdt_a_usar - comision
                btc_adquirido = usdt_neto / nivel

                usdt_balance      -= usdt_a_usar
                btc_en_posiciones += btc_adquirido
                usdt_invertido    += usdt_a_usar
                positions_count   += 1

                # Marcar orden como USADA
                orden["usado"] = True

                # Liberar flags de ventas con nivel_venta < precio_compra
                for ov in ordenes_venta:
                    if ov["usado"] and ov["precio"] < nivel:
                        ov["usado"] = False

                precio_prom = usdt_invertido / btc_en_posiciones

                # Primera compra: crear tabla de ventas si no existe
                if not ordenes_venta and atl is not None:
                    ordenes_venta = construir_tabla_ventas(precio_prom,
                                                           btc_en_posiciones,
                                                           atl)

                trade_history.append({
                    "datetime"                   : ts.isoformat(),
                    "type"                       : "BUY",
                    "price"                      : round(nivel, 4),
                    "nivel_distancia_pct"        : round(orden["distancia_pct"], 4),
                    "ath"                        : round(ath, 4),
                    "atl"                        : round(atl, 4) if atl else None,
                    "usdt_spent"                 : round(usdt_a_usar, 4),
                    "btc_bought"                 : round(btc_adquirido, 8),
                    "commission_usdt"            : round(comision, 4),
                    "btc_sold"                   : None,
                    "btc_accumulated"            : None,
                    "usdt_received"              : None,
                    "ganancia_usdt"              : None,
                    "usdt_balance"               : round(usdt_balance, 4),
                    "btc_balance"                : round(btc_balance, 8),
                    "btc_en_posiciones"          : round(btc_en_posiciones, 8),
                    "positions_count"            : positions_count,
                    "precio_promedio_posiciones" : round(precio_prom, 4),
                })

        # ── EJECUTAR VENTAS ──────────────────────────────────
        for orden in ordenes_venta:
            nivel = orden["precio"]

            # Saltar si ya usada
            if orden["usado"]:
                continue

            # Guardia: no vender si nivel <= precio_promedio
            if precio_prom > 0 and nivel <= precio_prom:
                continue

            if open_vela < nivel <= high_vela and btc_en_posiciones > 0:

                btc_procesado  = min(orden["btc_objetivo"], btc_en_posiciones)
                if btc_procesado <= 0:
                    continue

                btc_a_acumular = btc_procesado * (orden["acumular_pct"] / 100.0)
                btc_a_vender   = btc_procesado - btc_a_acumular

                usdt_bruto = btc_a_vender * nivel
                comision   = usdt_bruto   * (COMMISSION_PCT / 100.0)
                usdt_neto  = usdt_bruto   - comision

                # Proporcion sobre btc_en_posiciones ANTES del decremento
                proporcion      = btc_procesado / btc_en_posiciones
                costo_procesado = usdt_invertido * proporcion
                ganancia        = usdt_neto - costo_procesado * \
                                  (1.0 - orden["acumular_pct"] / 100.0)

                btc_en_posiciones -= btc_procesado
                btc_balance       += btc_a_acumular
                usdt_balance      += usdt_neto
                usdt_invertido     = max(0.0, usdt_invertido - costo_procesado)
                positions_count   -= 1

                # Marcar orden como USADA
                orden["usado"] = True

                # Liberar flags de compras con nivel_compra < precio_venta
                for oc in ordenes_compra:
                    if oc["usado"] and oc["precio"] < nivel:
                        oc["usado"] = False

                precio_prom = usdt_invertido / btc_en_posiciones \
                              if btc_en_posiciones > 0 else 0.0

                trade_history.append({
                    "datetime"                   : ts.isoformat(),
                    "type"                       : "SELL",
                    "price"                      : round(nivel, 4),
                    "nivel_distancia_pct"        : round(orden["distancia_pct"], 4),
                    "ath"                        : round(ath, 4),
                    "atl"                        : round(atl, 4),
                    "acumular_pct_usado"         : round(orden["acumular_pct"], 4),
                    "usdt_spent"                 : None,
                    "btc_bought"                 : None,
                    "commission_usdt"            : round(comision, 4),
                    "btc_sold"                   : round(btc_a_vender, 8),
                    "btc_accumulated"            : round(btc_a_acumular, 8),
                    "usdt_received"              : round(usdt_neto, 4),
                    "ganancia_usdt"              : round(ganancia, 4),
                    "usdt_balance"               : round(usdt_balance, 4),
                    "btc_balance"                : round(btc_balance, 8),
                    "btc_en_posiciones"          : round(btc_en_posiciones, 8),
                    "positions_count"            : positions_count,
                    "precio_promedio_posiciones" : round(precio_prom, 4),
                })

        if i > 0 and i % 500_000 == 0:
            pct  = i / n * 100
            port = usdt_balance + (btc_balance + btc_en_posiciones) * close_vela
            libres_c = sum(1 for o in ordenes_compra if not o["usado"])
            libres_v = sum(1 for o in ordenes_venta  if not o["usado"])
            print(f"  {pct:.1f}%  vela {i:,}/{n:,}  "
                  f"trades={len(trade_history):,}  "
                  f"portfolio=${port:,.0f}  "
                  f"ATH=${ath:,.0f}  prom=${precio_prom:,.0f}  "
                  f"ordC_libres={libres_c}  ordV_libres={libres_v}")

    # ── Resumen final ─────────────────────────────────────────
    precio_final    = float(df["close"].iloc[-1])
    btc_total       = btc_balance + btc_en_posiciones
    portfolio_final = usdt_balance + btc_total * precio_final
    pnl_pct         = (portfolio_final - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100
    precio_prom = usdt_invertido / btc_en_posiciones

    compras = [t for t in trade_history if t["type"] == "BUY"]
    ventas  = [t for t in trade_history if t["type"] == "SELL"]
    btc_acumulado_total = sum(
        t["btc_accumulated"] for t in ventas if t["btc_accumulated"] is not None
    )

    summary = {
        "estrategia"              : "Grid ATH/ATL Serie Geometrica con Flags",
        "fecha_inicio"            : str(df["datetime"].iloc[0]),
        "fecha_fin"               : str(df["datetime"].iloc[-1]),
        "saldo_inicial_usdt"      : SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(usdt_balance, 4),
        "btc_balance_final"       : round(btc_balance, 8),
        "btc_acumulado_total"     : round(btc_acumulado_total, 8),
        "btc_en_posiciones_final" : round(btc_en_posiciones, 8),
        "portfolio_value_final"   : round(portfolio_final, 4),
        "precio_prom"             : round(precio_prom, 4),
        "pnl_pct"                 : round(pnl_pct, 4),
        "total_trades"            : len(trade_history),
        "total_compras"           : len(compras),
        "total_ventas"            : len(ventas),
        "positions_count_final"   : positions_count,
        "ath_final"               : round(ath, 4) if ath else None,
        "atl_final"               : round(atl, 6) if atl else None,
        "parametros": {
            "paso_pct_compra"  : PASO_PCT_COMPRA,
            "caida_maxima"     : CAIDA_MAXIMA,
            "factor_compra"    : FACTOR_COMPRA,
            "paso_pct_venta"   : PASO_PCT_VENTA,
            "subida_maxima"    : SUBIDA_MAXIMA,
            "factor_venta"     : FACTOR_VENTA,
            "min_acumular_pct" : MIN_ACUMULAR_PCT,
            "max_acumular_pct" : MAX_ACUMULAR_PCT,
            "commission_pct"   : COMMISSION_PCT,
        },
    }

    return {"summary": summary, "trade_history": trade_history}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ESTRATEGIA GRID ATH/ATL SERIE GEOMETRICA — BACKTESTING")
    print("=" * 64)
    print(f"\n  Compras : paso={PASO_PCT_COMPRA}% log  caida_max={CAIDA_MAXIMA}%  factor={FACTOR_COMPRA}")
    print(f"            ref: usdt_balance en cada nuevo ATH")
    print(f"            flag: cada nivel se usa 1 vez, se libera con venta > nivel")
    print(f"            guardia: no comprar >= precio_promedio_posiciones")
    print(f"  Ventas  : paso={PASO_PCT_VENTA}% log  subida_max={SUBIDA_MAXIMA}%  factor={FACTOR_VENTA}")
    print(f"            ancla: precio_prom en cada nuevo ATL")
    print(f"            flag: cada nivel se usa 1 vez, se libera con compra < nivel")
    print(f"            guardia: no vender <= precio_promedio_posiciones")

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos.")
        return

    results = ejecutar_estrategia(df)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    s = results["summary"]
    print(f"\nResultados guardados en: {RESULTS_JSON}")
    print("\n" + "=" * 64)
    print("  RESUMEN FINAL")
    print("=" * 64)
    print(f"  Periodo              : {s['fecha_inicio']}  ->  {s['fecha_fin']}")
    print(f"  Capital inicial      : ${s['saldo_inicial_usdt']:>12,.2f}")
    print(f"  Portfolio final      : ${s['portfolio_value_final']:>12,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre           : ${s['usdt_balance_final']:>12,.4f}")
    print(f"  BTC libre            :  {s['btc_balance_final']:>14.8f}  (acum: {s['btc_acumulado_total']:.8f})")
    print(f"  BTC en posiciones    :  {s['btc_en_posiciones_final']:>14.8f}")
    print(f"  Precio promedio BTC  : ${s['precio_prom']:>12,.4f}")
    print(f"  Compras              :  {s['total_compras']:,}")
    print(f"  Ventas               :  {s['total_ventas']:,}")
    print(f"  positions_count      :  {s['positions_count_final']:+d}")
    print(f"  ATH registrado       : ${s['ath_final']:,.2f}")
    print(f"  ATL registrado       : ${s['atl_final']:,.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()