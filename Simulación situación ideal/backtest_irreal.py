"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BACKTEST IRREAL — LOCAL BOTTOMS / LOCAL TOPS — BTC/USDT 1H          ║
║  Oráculo perfecto: compra en cada mínimo local, vende en cada máximo local  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Gestión de capital (slots fijos):
  · Al llegar a 0 posiciones abiertas se calcula:
        slot_usdt = usdt_balance / MAX_POSICIONES
    y queda fijo hasta el próximo vaciado de cartera.
  · Cada BUY consume exactamente 1 slot de USDT.
  · Cada SELL vende  btc_en_posiciones / positions_count  BTC,
    calculado después de cada compra y congelado entre ventas.

Valor de portafolio:
    usdt_balance + btc_en_posiciones × close_price_vela_actual

Uso:
    python backtest_irreal.py

Requiere:
    · btc_hourly.db   (generado por BTCUSDT_1H_Binance_data_downloader_optimized.py)
    · config.py       (todos los parámetros configurables)
"""

import sqlite3
import json
import math
from collections import deque

import config as C


# ═════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def cargar_datos(db_path: str, fecha_inicio: str, fecha_fin: str) -> list[dict]:
    """Carga las velas 1H del rango indicado desde la base de datos SQLite."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, datetime, open, high, low, close, volume
        FROM   btc_hourly
        WHERE  datetime >= ? AND datetime <= ?
        ORDER  BY timestamp ASC
        """,
        (fecha_inicio + " 00:00:00", fecha_fin + " 23:59:59"),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "ts":       row[0],
            "datetime": row[1],
            "open":     row[2],
            "high":     row[3],
            "low":      row[4],
            "close":    row[5],
            "volume":   row[6],
        }
        for row in rows
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 2. DETECCIÓN DE MÍNIMOS Y MÁXIMOS LOCALES (ORÁCULO PERFECTO)
# ═════════════════════════════════════════════════════════════════════════════

def detectar_señales(candles: list[dict], ventana: int) -> list[str | None]:
    """
    Retorna una lista paralela a `candles` con:
        "BUY"  → mínimo local (low ≤ todos los vecinos a distancia ≤ ventana)
        "SELL" → máximo local (high ≥ todos los vecinos a distancia ≤ ventana)
        None   → sin señal (o sin suficiente contexto a cada lado)

    Las primeras y últimas `ventana` velas reciben None por no tener contexto
    completo en ambas direcciones — requisito del oráculo perfecto.
    """
    n      = len(candles)
    result = [None] * n

    for i in range(ventana, n - ventana):
        vecinos = range(i - ventana, i + ventana + 1)

        low_i  = candles[i]["low"]
        high_i = candles[i]["high"]

        es_minimo = all(low_i  <= candles[j]["low"]  for j in vecinos if j != i)
        es_maximo = all(high_i >= candles[j]["high"] for j in vecinos if j != i)

        # Si una vela es simultáneamente mínimo y máximo (rango plano extremo)
        # se prioriza BUY (conservador).
        if es_minimo:
            result[i] = "BUY"
        elif es_maximo:
            result[i] = "SELL"

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3. PORTAFOLIO — GESTIÓN DE CAPITAL POR SLOTS FIJOS
# ═════════════════════════════════════════════════════════════════════════════

class Posicion:
    """Una compra puntual de BTC: precio de entrada y cantidad."""

    __slots__ = ("entry_price", "btc")

    def __init__(self, entry_price: float, btc: float):
        self.entry_price = entry_price
        self.btc         = btc


class Portfolio:
    """
    Gestión de capital mediante slots de tamaño fijo.

    Regla de slots
    ──────────────
    Cuando positions_count llega a 0 (cartera vaciada):
        slot_usdt = usdt_balance / MAX_POSICIONES

    El slot permanece INMUTABLE mientras haya posiciones abiertas.
    Sólo se recalcula al volver a 0 posiciones.

    Regla de venta
    ──────────────
    Después de cada BUY se actualiza:
        btc_por_venta = btc_en_posiciones / positions_count

    Este valor permanece INMUTABLE entre ventas y sólo cambia con la
    siguiente compra.
    """

    def __init__(self, usdt_inicial: float, max_posiciones: int):
        self.usdt           = usdt_inicial
        self.max_posiciones = max_posiciones
        self.posiciones: deque[Posicion] = deque()

        # Estado de slots — se inicializa en la primera operación
        self.slot_usdt      : float = usdt_inicial / max_posiciones
        self.btc_por_venta  : float = 0.0
        self.btc_acumulado_total: float = 0.0   # BTC totales que pasaron por ventas

    # ── Propiedades calculadas ─────────────────────────────────────────────

    @property
    def btc_en_posiciones(self) -> float:
        return sum(p.btc for p in self.posiciones)

    @property
    def positions_count(self) -> int:
        return len(self.posiciones)

    @property
    def precio_promedio_posiciones(self) -> float:
        total_btc = self.btc_en_posiciones
        if total_btc == 0:
            return 0.0
        return sum(p.entry_price * p.btc for p in self.posiciones) / total_btc

    def valor_portfolio(self, close_price: float) -> float:
        """USDT libre + valor de mercado del BTC en posiciones al close actual."""
        return self.usdt + self.btc_en_posiciones * close_price

    # ── Recálculos internos ────────────────────────────────────────────────

    def _recalcular_slot_si_vacio(self):
        """Si no hay posiciones, reinicia el slot_usdt con el USDT disponible."""
        if self.positions_count == 0:
            self.slot_usdt     = self.usdt / self.max_posiciones
            self.btc_por_venta = 0.0

    def _recalcular_btc_por_venta(self):
        """Actualiza btc_por_venta después de cada compra."""
        n = self.positions_count
        self.btc_por_venta = self.btc_en_posiciones / n if n > 0 else 0.0

    # ── Operaciones ────────────────────────────────────────────────────────

    def comprar(self, precio: float, commission_pct: float) -> dict | None:
        """
        Compra usando exactamente 1 slot de USDT.
        Actualiza btc_por_venta al finalizar.
        Retorna el detalle de la operación o None si el slot es insuficiente.
        """
        # Recalcular slot si la cartera estaba vacía
        self._recalcular_slot_si_vacio()

        usdt_a_gastar = self.slot_usdt
        if usdt_a_gastar < 1.0 or usdt_a_gastar > self.usdt + 1e-9:
            return None  # fondos insuficientes para cubrir el slot

        commission   = round(usdt_a_gastar * commission_pct / 100.0, 8)
        usdt_neto    = usdt_a_gastar - commission
        btc_comprado = round(usdt_neto / precio, 10)

        self.usdt -= usdt_a_gastar
        self.posiciones.append(Posicion(entry_price=precio, btc=btc_comprado))

        # Actualizar btc_por_venta DESPUÉS de registrar la nueva posición
        self._recalcular_btc_por_venta()

        return {
            "usdt_spent":      round(usdt_a_gastar, 8),
            "btc_bought":      btc_comprado,
            "commission_usdt": round(commission, 8),
            # % que representa el slot sobre el balance ANTES de comprar
            "pct_capital_usado": round(usdt_a_gastar / (self.usdt + usdt_a_gastar) * 100, 4),
        }

    def vender(self, precio: float, commission_pct: float) -> dict | None:
        """
        Vende exactamente btc_por_venta BTC (calculado en la última compra).
        Usa FIFO para reducir posiciones.
        Retorna el detalle o None si no hay BTC que vender.
        """
        if self.positions_count == 0 or self.btc_por_venta <= 0:
            return None

        btc_a_vender = round(self.btc_por_venta, 10)

        # Calcular costo promedio de lo que se va a vender (FIFO lectura)
        pendiente   = btc_a_vender
        costo_total = 0.0
        for pos in self.posiciones:
            if pendiente <= 0:
                break
            tomado       = min(pos.btc, pendiente)
            costo_total += pos.entry_price * tomado
            pendiente   -= tomado

        avg_entry = costo_total / btc_a_vender if btc_a_vender > 0 else 0.0

        # Reducir posiciones FIFO
        restante = btc_a_vender
        while restante > 1e-10 and self.posiciones:
            pos = self.posiciones[0]
            if pos.btc <= restante + 1e-10:
                restante -= pos.btc
                self.posiciones.popleft()
            else:
                pos.btc  -= restante
                restante  = 0.0

        usdt_bruto  = round(btc_a_vender * precio, 8)
        commission  = round(usdt_bruto * commission_pct / 100.0, 8)
        usdt_neto   = round(usdt_bruto - commission, 8)
        ganancia    = round(usdt_neto - avg_entry * btc_a_vender, 8)

        self.usdt               += usdt_neto
        self.btc_acumulado_total += btc_a_vender

        return {
            "btc_sold":         round(btc_a_vender, 10),
            "btc_accumulated":  0.0,      # el BTC sale de posiciones, no de balance libre
            "usdt_received":    usdt_neto,
            "commission_usdt":  commission,
            "ganancia_usdt":    ganancia,
            "pct_capital_usado": round(btc_a_vender / (self.btc_en_posiciones + btc_a_vender) * 100, 4),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 4. MOTOR DE BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def correr_backtest(candles: list[dict], señales: list[str | None], cfg) -> dict:
    """
    Itera sobre velas con señal, ejecuta compras/ventas y registra cada
    operación con el mismo esquema de campos que el JSON de referencia.
    """
    portfolio = Portfolio(cfg.SALDO_USDT_INICIAL, cfg.MAX_POSICIONES)

    trade_history: list[dict]     = []
    total_compras                 = 0
    total_ventas                  = 0
    total_ignorados               = 0
    ignorados_por_motivo: dict    = {}
    precio_min_comprado           = math.inf
    precio_max_vendido            = -math.inf

    for candle, señal in zip(candles, señales):
        if señal is None:
            continue

        precio_exec   = candle[cfg.PRECIO_COMPRA] if señal == "BUY" else candle[cfg.PRECIO_VENTA]
        close_price   = candle["close"]
        score_bot     = 100.0 if señal == "BUY"  else 0.0
        score_top     = 100.0 if señal == "SELL" else 0.0
        ignorado      = False
        motivo        = None
        op_detail     = {}

        # ── BUY ───────────────────────────────────────────────────────────
        if señal == "BUY":
            if portfolio.positions_count >= cfg.MAX_POSICIONES:
                ignorado = True
                motivo   = f"max_posiciones({cfg.MAX_POSICIONES})"
            else:
                result = portfolio.comprar(precio_exec, cfg.COMMISSION_PCT)
                if result is None:
                    ignorado = True
                    motivo   = "sin_usdt"
                else:
                    op_detail           = result
                    total_compras      += 1
                    precio_min_comprado = min(precio_min_comprado, precio_exec)

        # ── SELL ──────────────────────────────────────────────────────────
        elif señal == "SELL":
            if portfolio.positions_count == 0:
                ignorado = True
                motivo   = "sin_posiciones"
            else:
                result = portfolio.vender(precio_exec, cfg.COMMISSION_PCT)
                if result is None:
                    ignorado = True
                    motivo   = "sin_btc"
                else:
                    op_detail          = result
                    total_ventas      += 1
                    precio_max_vendido = max(precio_max_vendido, precio_exec)

        if ignorado:
            total_ignorados += 1
            ignorados_por_motivo[motivo] = ignorados_por_motivo.get(motivo, 0) + 1

        # ── Valor de portafolio en esta vela (close) ──────────────────────
        portfolio_value_vela = portfolio.valor_portfolio(close_price)

        # ── Registro en trade_history ─────────────────────────────────────
        trade_history.append({
            "datetime":                   candle["datetime"].replace(" ", "T"),
            "type":                       señal,
            "price":                      round(precio_exec, 8),
            "score_bot":                  score_bot,
            "score_top":                  score_top,
            "usdt_balance":               round(portfolio.usdt, 8),
            "btc_balance":                0.0,
            "btc_en_posiciones":          round(portfolio.btc_en_posiciones, 10),
            "positions_count":            portfolio.positions_count,
            "precio_promedio_posiciones": round(portfolio.precio_promedio_posiciones, 8),
            "portfolio_value":            round(portfolio_value_vela, 4),
            "pnl_pct_acumulado":          round(
                (portfolio_value_vela - cfg.SALDO_USDT_INICIAL) / cfg.SALDO_USDT_INICIAL * 100, 4
            ),
            "ignorado":                   ignorado,
            "motivo_ignorado":            motivo,
            # ── Compra (null si no aplica) ──
            "usdt_spent":                 op_detail.get("usdt_spent"),
            "btc_bought":                 op_detail.get("btc_bought"),
            # ── Común / venta (null si no aplica) ──
            "commission_usdt":            op_detail.get("commission_usdt"),
            "btc_sold":                   op_detail.get("btc_sold"),
            "btc_accumulated":            op_detail.get("btc_accumulated"),
            "usdt_received":              op_detail.get("usdt_received"),
            "ganancia_usdt":              op_detail.get("ganancia_usdt"),
            "pct_capital_usado":          op_detail.get("pct_capital_usado"),
        })

    # ── Métricas finales ──────────────────────────────────────────────────────
    precio_final     = candles[-1]["close"]
    precio_inicial   = candles[0]["close"]
    portfolio_value  = portfolio.valor_portfolio(precio_final)
    pnl_pct          = (portfolio_value - cfg.SALDO_USDT_INICIAL) / cfg.SALDO_USDT_INICIAL * 100
    buy_hold_pnl_pct = (precio_final - precio_inicial) / precio_inicial * 100

    atl = min(c["low"]  for c in candles)
    ath = max(c["high"] for c in candles)

    summary = {
        "estrategia":              "Backtest Irreal — Local Bottoms / Local Tops",
        "fecha_inicio":            cfg.FECHA_INICIO,
        "fecha_fin":               cfg.FECHA_FIN,
        "saldo_inicial_usdt":      cfg.SALDO_USDT_INICIAL,
        "usdt_balance_final":      round(portfolio.usdt, 8),
        "btc_balance_final":       0.0,
        "btc_acumulado_total":     round(portfolio.btc_acumulado_total, 10),
        "btc_en_posiciones_final": round(portfolio.btc_en_posiciones, 10),
        "precio_promedio_final":   round(portfolio.precio_promedio_posiciones, 8),
        "portfolio_value_final":   round(portfolio_value, 4),
        "pnl_pct":                 round(pnl_pct, 4),
        "buy_hold_pnl_pct":        round(buy_hold_pnl_pct, 4),
        "alpha_vs_bh":             round(pnl_pct - buy_hold_pnl_pct, 4),
        "precio_min_comprado":     round(precio_min_comprado, 4) if precio_min_comprado < math.inf  else None,
        "precio_max_vendido":      round(precio_max_vendido, 4)  if precio_max_vendido > -math.inf else None,
        "atl_final":               round(atl, 4),
        "ath_proyectado_final":    round(ath, 4),
        "total_trades_ejecutados": total_compras + total_ventas,
        "total_compras":           total_compras,
        "total_ventas":            total_ventas,
        "total_ignorados":         total_ignorados,
        "ordenes_canceladas":      0,
        "ignorados_por_motivo":    ignorados_por_motivo,
        "positions_count_final":   portfolio.positions_count,
        "parametros": {
            "ventana_local":    cfg.VENTANA_LOCAL,
            "precio_compra":    cfg.PRECIO_COMPRA,
            "precio_venta":     cfg.PRECIO_VENTA,
            "max_posiciones":   cfg.MAX_POSICIONES,
            "commission_pct":   cfg.COMMISSION_PCT,
            "slot_usdt_final":  round(portfolio.slot_usdt, 4),
        },
    }

    return {"summary": summary, "trade_history": trade_history}


# ═════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("   BACKTEST IRREAL — LOCAL BOTTOMS / LOCAL TOPS — BTC/USDT 1H")
    print("=" * 70)
    print(f"  Rango         : {C.FECHA_INICIO} → {C.FECHA_FIN}")
    print(f"  Capital init  : ${C.SALDO_USDT_INICIAL:,.2f} USDT")
    print(f"  Max posiciones: {C.MAX_POSICIONES}  "
          f"(slot inicial = ${C.SALDO_USDT_INICIAL / C.MAX_POSICIONES:,.2f})")
    print(f"  Ventana local : {C.VENTANA_LOCAL} velas a cada lado")
    print(f"  Precio compra : {C.PRECIO_COMPRA}   |  Precio venta: {C.PRECIO_VENTA}")
    print(f"  Comisión      : {C.COMMISSION_PCT}%")
    print("-" * 70)

    # 1. Cargar velas
    print("Cargando datos desde la base de datos...", end=" ", flush=True)
    candles = cargar_datos(C.DB_PATH, C.FECHA_INICIO, C.FECHA_FIN)
    if not candles:
        print("\n❌  No se encontraron velas en el rango indicado.")
        print("    Verificá DB_PATH y FECHA_INICIO/FIN en config.py")
        return
    print(f"OK  ({len(candles):,} velas)")

    # 2. Detectar señales
    print("Detectando mínimos y máximos locales...", end=" ", flush=True)
    señales = detectar_señales(candles, C.VENTANA_LOCAL)
    n_buy   = señales.count("BUY")
    n_sell  = señales.count("SELL")
    print(f"OK  (BUY={n_buy:,}  SELL={n_sell:,})")

    # 3. Ejecutar backtest
    print("Ejecutando backtest...", end=" ", flush=True)
    resultado = correr_backtest(candles, señales, C)
    print("OK")

    # 4. Guardar JSON
    with open(C.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"Resultado guardado en: {C.RESULTS_JSON}")

    # 5. Resumen en consola
    s = resultado["summary"]
    print("\n" + "=" * 70)
    print("   RESUMEN")
    print("=" * 70)
    print(f"  Portfolio final  : ${s['portfolio_value_final']:>12,.2f} USDT")
    print(f"  └─ USDT libre    : ${s['usdt_balance_final']:>12,.2f}")
    print(f"  └─ BTC en posic. :  {s['btc_en_posiciones_final']:.8f} BTC")
    print(f"  PnL              : {s['pnl_pct']:>+.2f}%")
    print(f"  Buy & Hold ref   : {s['buy_hold_pnl_pct']:>+.2f}%")
    print(f"  Alpha vs B&H     : {s['alpha_vs_bh']:>+.2f}%")
    print(f"  Compras          : {s['total_compras']:,}")
    print(f"  Ventas           : {s['total_ventas']:,}")
    print(f"  Ignorados        : {s['total_ignorados']:,}  → {s['ignorados_por_motivo']}")
    print(f"  Posiciones abier.: {s['positions_count_final']}")
    print("=" * 70)


if __name__ == "__main__":
    main()