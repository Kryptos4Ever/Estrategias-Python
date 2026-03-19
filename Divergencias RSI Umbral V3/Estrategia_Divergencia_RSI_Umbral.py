"""
Estrategia Divergencia RSI — Zona de Orden en Tiempo Real
══════════════════════════════════════════════════════════════════════
BTC/USDT · Velas Horarias · Backtesting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIFERENCIA FUNDAMENTAL CON LA VERSIÓN ANTERIOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Versión anterior (con lookahead):
  · Detecta divergencia en vela i (usando rsi_low[i] — requiere cierre).
  · Ejecuta la compra al low[i] de esa misma vela.
  · Problema: para saber rsi_low[i] hay que esperar el cierre de la vela,
    pero el precio de ejecución (low[i]) solo existe durante esa vela.
    En producción se ejecutaría en la vela siguiente, a un precio peor.

Esta versión (sin lookahead — ejecutable en producción):
  · Al CIERRE de la vela i, detecta si hay una divergencia potencial
    pendiente usando los datos disponibles hasta i.
  · Calcula la ZONA VÁLIDA de la próxima vela usando la fórmula analítica
    del x_umbral: el rango de precios donde, si la vela i+1 toca ese precio,
    la divergencia estará garantizada por construcción matemática.
  · Coloca una ORDEN LÍMITE al inicio de la zona (precio_ancla ± tick).
  · Si la vela i+1 alcanza el precio → la orden se llena → divergencia válida.
  · Si la vela i+1 cae por debajo de x_umbral → la orden se cancela.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MECÁNICA DE SEÑAL Y ORDEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPRA — al cierre de vela i se evalúa si la PRÓXIMA vela podría
  completar una divergencia alcista válida:

  Condición sobre datos ya cerrados (sin lookahead):
    1. window_lows = low[i-N+1 : i+1]  (ventana incluye la vela actual)
    2. idx_min     = índice del mínimo de precio en la ventana
    3. rsi_low[idx_min] ≤ RSI_BUY_TRIGGER        (âncla en sobreventa)
    4. idx_min < i   (el mínimo no es la vela actual — debe haber distancia)
    5. rsi_low[i] > rsi_low[idx_min]   (RSI de vela actual ya diverge)
       ↑ esto implica que si el precio sigue bajando con el mismo momentum,
         el RSI de la siguiente vela también divergirá.

  Zona válida para la orden (calculada analíticamente con avg_gain/loss[i]):
    α  = 1/RSI_LENGTH
    k  = (1-α)/α
    RS = rsi_ancla / (100 - rsi_ancla)

    x_umbral = low[i] + k×avg_loss[i] − k×avg_gain[i]/RS

    Si Ra ≤ 0 (colapso extremo): x_umbral = 0 (zona ilimitada)

  Orden para vela i+1:
    precio_orden  = precio_ancla × (1 − PROF_ZONA_PCT/100)
                    (0% = justo bajo el ancla; >0% = más profundo en la zona)
    precio_cancel = x_umbral
    Si low[i+1] ≤ precio_orden Y low[i+1] > x_umbral → FILL ✓
    Si low[i+1] ≤ x_umbral                           → CANCEL (divergencia rota)

  Precio de ejecución:
    = precio_orden  (orden límite — no el low exacto de la vela)

VENTA — lógica simétrica:
  Ventana: high[i-N+1 : i+1]
  idx_max: máximo de precio en ventana
  Filtro: rsi_high[idx_max] ≥ RSI_SELL_TRIGGER
  rsi_high[i] < rsi_high[idx_max]  (divergencia bajista presente)

  x_umbral_venta = high[i] + k×RS×avg_loss[i] − k×avg_gain[i]
  precio_orden   = precio_ancla × (1 + PROF_ZONA_PCT/100)
  FILL si high[i+1] ≥ precio_orden Y high[i+1] < x_umbral_venta

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENTES DE SIZING (sin cambios respecto a versión anterior)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Compra: pos = log(ATH/precio_orden) / log(100/FLOOR_PCT)
          pct = clamp(pos,0,1)^FACTOR_CAIDA × 100

  Venta:  ATH_PROY = ATL_actual × (1 + PCT_ATH_PROYECTADO/100)
          pos = log(precio_orden/PP) / log(ATH_PROY/PP)
          pct = clamp(pos,0,1)^FACTOR_SUBIDA × 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARÁMETRO NUEVO: PROF_ZONA_PCT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Controla a qué profundidad dentro de la zona válida se coloca la orden.
  0.0  → orden al precio del ancla (borde superior de zona, fill más probable)
  25.0 → orden al 25% de la amplitud (ancla→x_umbral), precio más agresivo
  Calibrar con Analisis_Zona_Divergencia.py antes de ajustar.
  Default conservador: 0.0 (= precio_ancla)
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
    FLOOR_PCT, FACTOR_CAIDA, FACTOR_SUBIDA, PCT_ATH_PROYECTADO,
    GUARDIA_COMPRA,
    GUARDIA_PRECIO_COMPRA,
    GUARDIA_PRECIO_VENTA,
    RSI_BUY_TRIGGER,
    RSI_SELL_TRIGGER,
    PROF_ZONA_PCT,
    USDT_RESERVA_PCT,
    BTC_PCT_TO_ACCUMULATE,
    COMMISSION_PCT,
    mostrar_configuracion,
)

DB_TABLE     = os.path.splitext(os.path.basename(DB_PATH))[0]
USDT_RESERVA = SALDO_USDT_INICIAL * USDT_RESERVA_PCT / 100



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS — RSI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_rsi_con_estado(series: pd.Series, length: int):
    """
    RSI de Wilder (EWM).
    Retorna (rsi_values, avg_gain_values, avg_loss_values) como arrays numpy.
    Los avg_gain/loss son necesarios para calcular x_umbral al cierre de cada vela.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    alpha    = 1.0 / length
    avg_gain = gain.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi.values, avg_gain.values, avg_loss.values


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS — ZONA DE ORDEN (fórmulas analíticas sin lookahead)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_zona_compra(avg_gain_i: float, avg_loss_i: float,
                         low_i: float, rsi_ancla: float,
                         precio_ancla: float) -> dict:
    """
    Calcula la zona válida de compra para la PRÓXIMA vela.

    Usando avg_gain[i], avg_loss[i] y low[i] (todos conocidos al cierre de i),
    encuentra el precio mínimo x_umbral tal que si la próxima vela toca un precio
    x con x_umbral < x < precio_ancla, el RSI de esa vela superará rsi_ancla.

    Fórmula (α = 1/RSI_LENGTH, k = (1-α)/α):
      x_umbral = low[i] + k×avg_loss[i] − k×avg_gain[i]×(100−Ra)/Ra

    Returns:
      zona_valida   bool    existe zona (x_umbral < precio_ancla)
      x_umbral      float   precio mínimo de la zona
      precio_orden  float   precio de la orden = ancla × (1 − PROF_ZONA_PCT/100)
      amplitud_abs  float   precio_ancla − x_umbral
      amplitud_pct  float   amplitud como % del precio_ancla
    """
    alpha = 1.0 / RSI_LENGTH
    k     = (1.0 - alpha) / alpha

    if rsi_ancla <= 0:
        # Zona ilimitada: Ra≈0 → cualquier precio bajo el ancla es válido
        x_umbral    = 0.0
        zona_valida = True
    elif rsi_ancla >= 100:
        return {"zona_valida": False}
    else:
        RS          = rsi_ancla / (100.0 - rsi_ancla)
        x_umbral    = low_i + k * avg_loss_i - k * avg_gain_i / RS
        zona_valida = (x_umbral < precio_ancla) and (x_umbral >= 0)

    if not zona_valida:
        return {"zona_valida": False}

    amplitud_abs = precio_ancla - x_umbral
    amplitud_pct = amplitud_abs / precio_ancla * 100.0 if precio_ancla > 0 else 0.0
    precio_orden = precio_ancla * (1.0 - PROF_ZONA_PCT / 100.0)
    # Garantizar que precio_orden esté dentro de la zona
    precio_orden = max(precio_orden, x_umbral + 1e-8)
    precio_orden = min(precio_orden, precio_ancla - 1e-8)

    return {
        "zona_valida"  : True,
        "x_umbral"     : x_umbral,
        "precio_orden" : precio_orden,
        "amplitud_abs" : amplitud_abs,
        "amplitud_pct" : amplitud_pct,
    }


def calcular_zona_venta(avg_gain_i: float, avg_loss_i: float,
                        high_i: float, rsi_ancla: float,
                        precio_ancla: float) -> dict:
    """
    Calcula la zona válida de venta para la PRÓXIMA vela.

    Fórmula simétrica (α = 1/RSI_LENGTH, k = (1-α)/α):
      x_umbral_venta = high[i] + k×RS×avg_loss[i] − k×avg_gain[i]
      donde RS = Ra / (100 - Ra)

    Si la próxima vela toca un precio x con precio_ancla < x < x_umbral_venta,
    el RSI de esa vela estará por debajo de rsi_ancla → divergencia bajista válida.
    """
    alpha = 1.0 / RSI_LENGTH
    k     = (1.0 - alpha) / alpha

    if rsi_ancla >= 100:
        x_umbral    = float('inf')
        zona_valida = True
    elif rsi_ancla <= 0:
        return {"zona_valida": False}
    else:
        RS          = rsi_ancla / (100.0 - rsi_ancla)
        x_umbral    = high_i + k * RS * avg_loss_i - k * avg_gain_i
        zona_valida = (x_umbral > precio_ancla)

    if not zona_valida:
        return {"zona_valida": False}

    amplitud_abs = (x_umbral - precio_ancla) if not math.isinf(x_umbral) else float('inf')
    amplitud_pct = (amplitud_abs / precio_ancla * 100.0
                    if precio_ancla > 0 and not math.isinf(amplitud_abs) else float('inf'))
    precio_orden = precio_ancla * (1.0 + PROF_ZONA_PCT / 100.0)
    if not math.isinf(x_umbral):
        precio_orden = min(precio_orden, x_umbral - 1e-8)
    precio_orden = max(precio_orden, precio_ancla + 1e-8)

    return {
        "zona_valida"  : True,
        "x_umbral"     : x_umbral,
        "precio_orden" : precio_orden,
        "amplitud_abs" : amplitud_abs,
        "amplitud_pct" : amplitud_pct,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS — GRADIENTES DE SIZING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pct_capital_compra(precio: float, ath: float) -> float:
    """
    Gradiente logarítmico de compra con ATL_REF FIJO.
      pos = log(ATH/precio) / log(100/FLOOR_PCT)
      pct = clamp(pos,0,1)^FACTOR_CAIDA × 100
    """
    if ath <= 0 or FLOOR_PCT <= 0 or precio <= 0:
        return 0.0
    log_rango = math.log(100.0 / FLOOR_PCT)
    if log_rango <= 0:
        return 0.0
    pos = math.log(ath / precio) / log_rango
    pos = max(0.0, min(1.0, pos))
    return (pos ** FACTOR_CAIDA) * 100.0


def pct_capital_venta(precio: float, atl: float, precio_promedio: float) -> float:
    """
    Gradiente logarítmico de venta anclado al PP con techo dinámico.

      ATH_PROY = ATL_actual × (1 + PCT_ATH_PROYECTADO / 100)
      pos      = log(precio / PP) / log(ATH_PROY / PP)   ∈ [0, 1]
      pct      = clamp(pos, 0, 1)^FACTOR_SUBIDA × 100

    ATH_PROY reemplaza al ATH histórico como techo del gradiente.
    Esto evita que la curva sea casi plana en bear markets donde el ATH
    real está muy lejos: el techo se calcula desde el ATL registrado,
    ajustándose dinámicamente a medida que el mercado encuentra nuevos mínimos.

    Retorna 0 si precio ≤ PP (guardia incorporada).
    """
    if atl <= 0 or precio_promedio <= 0 or precio <= precio_promedio:
        return 0.0
    ath_proy = atl * (1.0 + PCT_ATH_PROYECTADO / 100.0)
    if ath_proy <= precio_promedio:
        return 0.0
    log_amp = math.log(ath_proy / precio_promedio)
    if log_amp <= 0:
        return 0.0
    pos = math.log(precio / precio_promedio) / log_amp
    pos = max(0.0, min(1.0, pos))
    return (pos ** FACTOR_SUBIDA) * 100.0


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

def _registro_ignorado(ts, tipo, precio, rsi_l, rsi_h,
                       usdt_bal, btc_bal, btc_pos,
                       pos_count, ath, atl, motivo: str) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : tipo,
        "price"                      : round(precio, 4),
        "rsi_low"                    : round(rsi_l, 4) if not math.isnan(rsi_l) else None,
        "rsi_high"                   : round(rsi_h, 4) if not math.isnan(rsi_h) else None,
        "ath"                        : round(ath, 4),
        "atl"                        : round(atl, 4),
        "precio_orden"               : None,
        "precio_ancla"               : None,
        "x_umbral"                   : None,
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


def _registro_compra(ts, precio_orden, precio_ancla, x_umbral,
                     amplitud_pct, pct, pos_grad,
                     usdt_a_usar, comision, btc_adquirido,
                     rsi_l, rsi_h, ath, atl,
                     usdt_balance, btc_balance, btc_en_pos,
                     precio_promedio, positions_count) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : "BUY",
        "price"                      : round(precio_orden,  4),
        "rsi_low"                    : round(rsi_l,         4) if not math.isnan(rsi_l) else None,
        "rsi_high"                   : round(rsi_h,         4) if not math.isnan(rsi_h) else None,
        "ath"                        : round(ath,           4),
        "atl"                        : round(atl,           4),
        "precio_orden"               : round(precio_orden,  4),
        "precio_ancla"               : round(precio_ancla,  4),
        "x_umbral"                   : round(x_umbral,      4) if x_umbral else None,
        "amplitud_zona_pct"          : round(amplitud_pct,  4),
        "pct_capital_usado"          : round(pct,           4),
        "pos_gradiente"              : round(pos_grad,      6),
        "usdt_spent"                 : round(usdt_a_usar,   4),
        "btc_bought"                 : round(btc_adquirido, 8),
        "commission_usdt"            : round(comision,      4),
        "btc_sold"                   : None,
        "btc_accumulated"            : None,
        "usdt_received"              : None,
        "ganancia_usdt"              : None,
        "usdt_balance"               : round(usdt_balance,  4),
        "btc_balance"                : round(btc_balance,   8),
        "btc_en_posiciones"          : round(btc_en_pos,    8),
        "positions_count"            : positions_count,
        "precio_promedio_posiciones" : round(precio_promedio, 4),
        "ignorado"                   : False,
        "motivo_ignorado"            : None,
    }


def _registro_venta(ts, precio_orden, precio_ancla, x_umbral,
                    amplitud_pct, pct, pos_grad,
                    btc_a_vender, btc_a_acumular, comision,
                    usdt_neto, ganancia,
                    rsi_l, rsi_h, ath, atl,
                    usdt_balance, btc_balance, btc_en_pos,
                    precio_promedio, positions_count) -> dict:
    return {
        "datetime"                   : ts.isoformat(),
        "type"                       : "SELL",
        "price"                      : round(precio_orden,     4),
        "rsi_low"                    : round(rsi_l,            4) if not math.isnan(rsi_l) else None,
        "rsi_high"                   : round(rsi_h,            4) if not math.isnan(rsi_h) else None,
        "ath"                        : round(ath,              4),
        "atl"                        : round(atl,              4),
        "precio_orden"               : round(precio_orden,     4),
        "precio_ancla"               : round(precio_ancla,     4),
        "x_umbral"                   : round(x_umbral,         4) if x_umbral and not math.isinf(x_umbral) else None,
        "amplitud_zona_pct"          : round(amplitud_pct,     4) if not math.isinf(amplitud_pct) else None,
        "pct_capital_usado"          : round(pct,              4),
        "pos_gradiente"              : round(pos_grad,         6),
        "usdt_spent"                 : None,
        "btc_bought"                 : None,
        "commission_usdt"            : round(comision,         4),
        "btc_sold"                   : round(btc_a_vender,     8),
        "btc_accumulated"            : round(btc_a_acumular,   8),
        "usdt_received"              : round(usdt_neto,        4),
        "ganancia_usdt"              : round(ganancia,         4),
        "usdt_balance"               : round(usdt_balance,     4),
        "btc_balance"                : round(btc_balance,      8),
        "btc_en_posiciones"          : round(btc_en_pos,       8),
        "positions_count"            : positions_count,
        "precio_promedio_posiciones" : round(precio_promedio,  4),
        "ignorado"                   : False,
        "motivo_ignorado"            : None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ESTRATEGIA PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ejecutar_estrategia(df: pd.DataFrame) -> dict:

    # Calcular RSI y estados EWM sobre lows y highs
    rsi_low,  avg_g_low,  avg_l_low  = calcular_rsi_con_estado(df["low"],  RSI_LENGTH)
    rsi_high, avg_g_high, avg_l_high = calcular_rsi_con_estado(df["high"], RSI_LENGTH)

    lows   = df["low"].values.astype(float)
    highs  = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)
    dts    = df["datetime"].values
    n_velas = len(lows)

    # ── Estado financiero ─────────────────────────────────────────────────────
    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    positions_count   = 0

    precio_min_comprado = math.inf
    precio_max_vendido  = 0.0

    ath = float(np.max(highs[:N]))
    atl = float(np.min(lows[:N]))

    # ── Orden pendiente ───────────────────────────────────────────────────────
    # Al cierre de vela i puede quedar una orden activa para la vela i+1.
    # Se representa como un dict o None.
    orden_pendiente = None
    # {
    #   "tipo"         : "BUY" | "SELL"
    #   "precio_orden" : float   precio de la orden límite
    #   "x_umbral"     : float   precio de cancelación
    #   "precio_ancla" : float   precio del ancla de la divergencia
    #   "amplitud_pct" : float   amplitud de la zona (%)
    #   "rsi_ancla"    : float
    #   "rsi_señal"    : float   RSI de la vela que generó la señal
    # }

    # ── Contadores ────────────────────────────────────────────────────────────
    n_div_compra_sin_umbral = 0
    n_div_compra_con_umbral = 0
    n_div_venta_sin_umbral  = 0
    n_div_venta_con_umbral  = 0
    n_ordenes_canceladas    = 0   # órdenes que se cancelaron por x_umbral

    trade_history = []

    for i in range(N, n_velas):

        # ── Actualizar ATH/ATL con la vela ANTERIOR (sin lookahead) ──────────
        # Nota: se actualiza con highs/lows[i-1], no con la vela actual.
        # Así el ATH usado en el gradiente no incluye datos de la vela presente.
        if i > N:
            if highs[i-1] > ath: ath = float(highs[i-1])
            if lows[i-1]  < atl: atl = float(lows[i-1])

        ts = pd.Timestamp(dts[i])

        if np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]):
            orden_pendiente = None
            continue

        precio_promedio = (usdt_invertido / btc_en_posiciones
                           if btc_en_posiciones > 0 else 0.0)

        # ════════════════════════════════════════════════════════════════════
        # PASO 1 — Evaluar la orden pendiente de la vela anterior
        # ════════════════════════════════════════════════════════════════════
        if orden_pendiente is not None:
            op = orden_pendiente
            orden_pendiente = None   # se consume en esta vela (fill o cancel)

            if op["tipo"] == "BUY":
                # Fill si low[i] ≤ precio_orden Y low[i] > x_umbral
                # Cancel si low[i] ≤ x_umbral (divergencia rota)
                if lows[i] <= op["x_umbral"]:
                    n_ordenes_canceladas += 1
                    trade_history.append(_registro_ignorado(
                        ts, "BUY", op["precio_orden"],
                        float(rsi_low[i]), float(rsi_high[i]),
                        usdt_balance, btc_balance, btc_en_posiciones,
                        positions_count, ath, atl,
                        f"orden_cancelada_x_umbral"
                        f"(low={lows[i]:.0f}<=xu={op['x_umbral']:.0f})"
                    ))
                elif lows[i] <= op["precio_orden"]:
                    # ── FILL de compra ──────────────────────────────────────
                    precio_ej = op["precio_orden"]

                    usdt_disponible = usdt_balance - USDT_RESERVA
                    if usdt_disponible <= 0:
                        trade_history.append(_registro_ignorado(
                            ts, "BUY", precio_ej,
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl, "sin_capital_sobre_reserva"
                        ))
                    elif GUARDIA_COMPRA and btc_en_posiciones > 0 and precio_ej >= precio_promedio:
                        trade_history.append(_registro_ignorado(
                            ts, "BUY", precio_ej,
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl, "precio_sobre_promedio"
                        ))
                    elif GUARDIA_PRECIO_COMPRA and precio_min_comprado < math.inf and precio_ej >= precio_min_comprado:
                        trade_history.append(_registro_ignorado(
                            ts, "BUY", precio_ej,
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl, "precio_sobre_min_comprado"
                        ))
                    else:
                        pct         = pct_capital_compra(precio_ej, ath)
                        usdt_a_usar = usdt_disponible * pct / 100.0

                        if usdt_a_usar <= 0:
                            trade_history.append(_registro_ignorado(
                                ts, "BUY", precio_ej,
                                float(rsi_low[i]), float(rsi_high[i]),
                                usdt_balance, btc_balance, btc_en_posiciones,
                                positions_count, ath, atl, "gradiente_cero"
                            ))
                        else:
                            log_rango = math.log(100.0 / FLOOR_PCT)
                            pos_grad  = (round(math.log(ath / precio_ej) / log_rango, 6)
                                         if log_rango > 0 and ath > 0 else 0.0)

                            comision      = usdt_a_usar * (COMMISSION_PCT / 100.0)
                            btc_adquirido = (usdt_a_usar - comision) / precio_ej

                            usdt_balance      -= usdt_a_usar
                            btc_en_posiciones += btc_adquirido
                            usdt_invertido    += usdt_a_usar
                            positions_count   += 1
                            precio_promedio    = usdt_invertido / btc_en_posiciones
                            if precio_ej < precio_min_comprado:
                                precio_min_comprado = precio_ej

                            trade_history.append(_registro_compra(
                                ts, precio_ej, op["precio_ancla"], op["x_umbral"],
                                op["amplitud_pct"], pct, pos_grad,
                                usdt_a_usar, comision, btc_adquirido,
                                float(rsi_low[i]), float(rsi_high[i]), ath, atl,
                                usdt_balance, btc_balance, btc_en_posiciones,
                                precio_promedio, positions_count
                            ))
                # else: la vela no tocó el precio → orden no ejecutada (expiró)

            elif op["tipo"] == "SELL":
                # Fill si high[i] ≥ precio_orden Y high[i] < x_umbral_venta
                x_umb_v = op["x_umbral"]
                dentro  = (highs[i] >= op["precio_orden"] and
                           (math.isinf(x_umb_v) or highs[i] < x_umb_v))

                if not math.isinf(x_umb_v) and highs[i] >= x_umb_v:
                    n_ordenes_canceladas += 1
                    trade_history.append(_registro_ignorado(
                        ts, "SELL", op["precio_orden"],
                        float(rsi_low[i]), float(rsi_high[i]),
                        usdt_balance, btc_balance, btc_en_posiciones,
                        positions_count, ath, atl,
                        f"orden_cancelada_x_umbral_venta"
                        f"(high={highs[i]:.0f}>=xu={x_umb_v:.0f})"
                    ))
                elif dentro and btc_en_posiciones > 0:
                    precio_ej = op["precio_orden"]
                    precio_promedio = (usdt_invertido / btc_en_posiciones
                                       if btc_en_posiciones > 0 else 0.0)

                    pct      = pct_capital_venta(precio_ej, atl, precio_promedio)
                    btc_slot = btc_en_posiciones * pct / 100.0

                    if btc_slot <= 0:
                        motivo = ("precio_bajo_promedio"
                                  if precio_promedio > 0 and precio_ej <= precio_promedio
                                  else "gradiente_cero")
                        trade_history.append(_registro_ignorado(
                            ts, "SELL", precio_ej,
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl, motivo
                        ))
                    elif GUARDIA_PRECIO_VENTA and precio_max_vendido > 0 and precio_ej <= precio_max_vendido:
                        trade_history.append(_registro_ignorado(
                            ts, "SELL", precio_ej,
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl, "precio_bajo_max_vendido"
                        ))
                    else:
                        _ath_proy = atl * (1.0 + PCT_ATH_PROYECTADO / 100.0)
                        log_amp  = (math.log(_ath_proy / precio_promedio)
                                    if precio_promedio > 0 and _ath_proy > precio_promedio else 0)
                        pos_grad = (round(math.log(precio_ej / precio_promedio) / log_amp, 6)
                                    if log_amp > 0 and precio_ej > precio_promedio else 0.0)

                        btc_a_acumular     = btc_slot * (BTC_PCT_TO_ACCUMULATE / 100.0)
                        btc_a_vender       = btc_slot - btc_a_acumular
                        usdt_bruto         = btc_a_vender * precio_ej
                        comision           = usdt_bruto * (COMMISSION_PCT / 100.0)
                        usdt_neto          = usdt_bruto - comision
                        costo_proporcional = usdt_invertido * (btc_slot / btc_en_posiciones)
                        ganancia           = usdt_neto - costo_proporcional * (1.0 - BTC_PCT_TO_ACCUMULATE / 100.0)

                        btc_en_posiciones -= btc_slot
                        btc_balance       += btc_a_acumular
                        usdt_balance      += usdt_neto
                        usdt_invertido    -= costo_proporcional
                        usdt_invertido     = max(usdt_invertido, 0.0)
                        positions_count   -= 1
                        precio_promedio    = (usdt_invertido / btc_en_posiciones
                                              if btc_en_posiciones > 0 else 0.0)
                        if precio_ej > precio_max_vendido:
                            precio_max_vendido = precio_ej

                        amp = op.get("amplitud_pct", 0.0)
                        trade_history.append(_registro_venta(
                            ts, precio_ej, op["precio_ancla"], op["x_umbral"],
                            amp, pct, pos_grad,
                            btc_a_vender, btc_a_acumular, comision,
                            usdt_neto, ganancia,
                            float(rsi_low[i]), float(rsi_high[i]), ath, atl,
                            usdt_balance, btc_balance, btc_en_posiciones,
                            precio_promedio, positions_count
                        ))
                # else: la vela no tocó el precio → orden expiró

        # ════════════════════════════════════════════════════════════════════
        # PASO 2 — Detectar nueva señal en la vela actual (para vela i+1)
        # ════════════════════════════════════════════════════════════════════
        # La ventana incluye la vela actual: lows[i-N+1 : i+1]
        window_lows  = lows[i-N+1  : i+1]
        window_highs = highs[i-N+1 : i+1]

        nueva_orden = None

        # ── COMPRA: divergencia alcista ───────────────────────────────────
        # El mínimo de precio en la ventana debe ser alguna vela ANTERIOR
        # (idx_rel < N-1), y el RSI actual ya debe estar por encima del RSI del ancla.
        idx_min_rel = int(window_lows.argmin())
        if idx_min_rel < (len(window_lows) - 1):   # el mínimo no es la vela actual
            idx_ancla  = i - N + 1 + idx_min_rel
            Ra_c       = float(rsi_low[idx_ancla])
            rsi_i_c    = float(rsi_low[i])

            if not math.isnan(Ra_c) and rsi_i_c > Ra_c:
                n_div_compra_sin_umbral += 1
                if Ra_c <= RSI_BUY_TRIGGER:
                    n_div_compra_con_umbral += 1
                    precio_ancla_c = float(lows[idx_ancla])

                    zona = calcular_zona_compra(
                        float(avg_g_low[i]),
                        float(avg_l_low[i]),
                        float(lows[i]),
                        Ra_c,
                        precio_ancla_c,
                    )
                    if zona["zona_valida"]:
                        nueva_orden = {
                            "tipo"         : "BUY",
                            "precio_orden" : zona["precio_orden"],
                            "x_umbral"     : zona["x_umbral"],
                            "precio_ancla" : precio_ancla_c,
                            "amplitud_pct" : zona["amplitud_pct"],
                            "rsi_ancla"    : Ra_c,
                            "rsi_señal"    : rsi_i_c,
                        }
                    else:
                        n_div_compra_con_umbral -= 1   # zona inválida, no contar
                        trade_history.append(_registro_ignorado(
                            ts, "BUY", float(lows[i]),
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl,
                            "zona_compra_invalida"
                        ))
                else:
                    trade_history.append(_registro_ignorado(
                        ts, "BUY", float(lows[i]),
                        float(rsi_low[i]), float(rsi_high[i]),
                        usdt_balance, btc_balance, btc_en_posiciones,
                        positions_count, ath, atl,
                        f"rsi_ancla_compra_fuera_umbral"
                        f"(rsi_ancla={Ra_c:.1f}>{RSI_BUY_TRIGGER})"
                    ))

        # ── VENTA: divergencia bajista (solo si no hay señal de compra) ────
        if nueva_orden is None:
            idx_max_rel = int(window_highs.argmax())
            if idx_max_rel < (len(window_highs) - 1):
                idx_ancla  = i - N + 1 + idx_max_rel
                Ra_v       = float(rsi_high[idx_ancla])
                rsi_i_v    = float(rsi_high[i])

                if not math.isnan(Ra_v) and rsi_i_v < Ra_v:
                    n_div_venta_sin_umbral += 1
                    if Ra_v >= RSI_SELL_TRIGGER:
                        n_div_venta_con_umbral += 1
                        precio_ancla_v = float(highs[idx_ancla])

                        zona = calcular_zona_venta(
                            float(avg_g_high[i]),
                            float(avg_l_high[i]),
                            float(highs[i]),
                            Ra_v,
                            precio_ancla_v,
                        )
                        if zona["zona_valida"]:
                            nueva_orden = {
                                "tipo"         : "SELL",
                                "precio_orden" : zona["precio_orden"],
                                "x_umbral"     : zona["x_umbral"],
                                "precio_ancla" : precio_ancla_v,
                                "amplitud_pct" : zona.get("amplitud_pct", float("inf")),
                                "rsi_ancla"    : Ra_v,
                                "rsi_señal"    : rsi_i_v,
                            }
                        else:
                            n_div_venta_con_umbral -= 1
                            trade_history.append(_registro_ignorado(
                                ts, "SELL", float(highs[i]),
                                float(rsi_low[i]), float(rsi_high[i]),
                                usdt_balance, btc_balance, btc_en_posiciones,
                                positions_count, ath, atl,
                                "zona_venta_invalida"
                            ))
                    else:
                        trade_history.append(_registro_ignorado(
                            ts, "SELL", float(highs[i]),
                            float(rsi_low[i]), float(rsi_high[i]),
                            usdt_balance, btc_balance, btc_en_posiciones,
                            positions_count, ath, atl,
                            f"rsi_ancla_venta_fuera_umbral"
                            f"(rsi_ancla={Ra_v:.1f}<{RSI_SELL_TRIGGER})"
                        ))

        # La nueva orden queda pendiente para la próxima vela
        # (una señal anterior no ejecutada se reemplaza por la nueva)
        if nueva_orden is not None:
            orden_pendiente = nueva_orden

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
        clave = m.split("(")[0] if "(" in m else m
        motivos[clave] = motivos.get(clave, 0) + 1

    btc_acumulado_total = sum(
        t["btc_accumulated"] for t in ventas
        if t.get("btc_accumulated") is not None
    )

    tasa_filtro_compra = (
        round((1 - n_div_compra_con_umbral / n_div_compra_sin_umbral) * 100, 1)
        if n_div_compra_sin_umbral > 0 else 0.0
    )
    tasa_filtro_venta = (
        round((1 - n_div_venta_con_umbral / n_div_venta_sin_umbral) * 100, 1)
        if n_div_venta_sin_umbral > 0 else 0.0
    )

    summary = {
        "estrategia"              : "Divergencia RSI — Zona de Orden en Tiempo Real (v2)",
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
        "precio_min_comprado"     : round(precio_min_comprado
                                          if precio_min_comprado < math.inf else 0.0, 4),
        "precio_max_vendido"      : round(precio_max_vendido,   4),
        "atl_final"               : round(atl,                  4),
        "ath_proyectado_final"     : round(atl * (1.0 + PCT_ATH_PROYECTADO / 100.0), 4),
        "total_trades_ejecutados" : len(trades_activos),
        "total_compras"           : len(compras),
        "total_ventas"            : len(ventas),
        "total_ignorados"         : len(ignorados),
        "ordenes_canceladas"      : n_ordenes_canceladas,
        "ignorados_por_motivo"    : motivos,
        "positions_count_final"   : positions_count,
        "usdt_reserva_aplicada"   : round(USDT_RESERVA, 4),
        "umbral_filtro": {
            "rsi_buy_trigger"               : RSI_BUY_TRIGGER,
            "rsi_sell_trigger"              : RSI_SELL_TRIGGER,
            "prof_zona_pct"                 : PROF_ZONA_PCT,
            "divergencias_compra_detectadas": n_div_compra_sin_umbral,
            "divergencias_compra_aprobadas" : n_div_compra_con_umbral,
            "tasa_rechazo_compra_pct"       : tasa_filtro_compra,
            "divergencias_venta_detectadas" : n_div_venta_sin_umbral,
            "divergencias_venta_aprobadas"  : n_div_venta_con_umbral,
            "tasa_rechazo_venta_pct"        : tasa_filtro_venta,
        },
        "parametros": {
            "rsi_length"            : RSI_LENGTH,
            "N"                     : N,
            "floor_pct"             : FLOOR_PCT,
            "factor_caida"          : FACTOR_CAIDA,
            "factor_subida"         : FACTOR_SUBIDA,
            "pct_ath_proyectado"    : PCT_ATH_PROYECTADO,
            "guardia_compra"        : GUARDIA_COMPRA,
            "guardia_precio_compra" : GUARDIA_PRECIO_COMPRA,
            "guardia_precio_venta"  : GUARDIA_PRECIO_VENTA,
            "rsi_buy_trigger"       : RSI_BUY_TRIGGER,
            "rsi_sell_trigger"      : RSI_SELL_TRIGGER,
            "prof_zona_pct"         : PROF_ZONA_PCT,
            "usdt_reserva_pct"      : USDT_RESERVA_PCT,
            "btc_pct_to_accumulate" : BTC_PCT_TO_ACCUMULATE,
            "commission_pct"        : COMMISSION_PCT,
        },
    }

    return {"summary": summary, "trade_history": trade_history}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DIVERGENCIA RSI — ZONA DE ORDEN EN TIEMPO REAL  (v2)        ║")
    print("║  BTC/USDT · Sin lookahead · Órdenes límite reales            ║")
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
    u = s["umbral_filtro"]

    print("\n" + "═" * 64)
    print("  RESUMEN FINAL")
    print("═" * 64)
    print(f"  Período              : {s['fecha_inicio'][:10]}  →  {s['fecha_fin'][:10]}")
    print(f"  Capital inicial      : ${s['saldo_inicial_usdt']:>10,.2f}")
    print(f"  Portfolio final      : ${s['portfolio_value_final']:>10,.2f}   ({s['pnl_pct']:+.2f}%)")
    print(f"  USDT libre           : ${s['usdt_balance_final']:>10,.4f}   (reserva: ${s['usdt_reserva_aplicada']:,.2f})")
    print(f"  BTC libre (acum.)    :  {s['btc_balance_final']:>.8f} ₿  ({s['btc_acumulado_total']:.8f} ₿ acumulado)")
    print(f"  BTC en posiciones    :  {s['btc_en_posiciones_final']:>.8f} ₿")
    pp = s["precio_promedio_final"]
    if pp > 0:
        print(f"  Precio promedio BTC  : ${pp:>10,.2f}")
    print(f"  ATL registrado       : ${s['atl_final']:>10,.2f}")
    print(f"  ATH proyectado final : ${s['ath_proyectado_final']:>10,.2f}"  f"  (ATL × {1 + PCT_ATH_PROYECTADO/100:.1f})")
    pmc = s.get("precio_min_comprado", 0)
    pmv = s.get("precio_max_vendido",  0)
    if pmc > 0: print(f"  Precio mín. comprado : ${pmc:>10,.2f}")
    if pmv > 0: print(f"  Precio máx. vendido  : ${pmv:>10,.2f}")
    print(f"  Trades ejecutados    :  {s['total_trades_ejecutados']}"
          f"  (compras: {s['total_compras']}  |  ventas: {s['total_ventas']})")
    print(f"  Órdenes canceladas   :  {s['ordenes_canceladas']}"
          f"  (x_umbral alcanzado antes del fill)")
    print(f"  Señales ignoradas    :  {s['total_ignorados']}")
    for motivo, cnt in s.get("ignorados_por_motivo", {}).items():
        print(f"    · {motivo:<48}: {cnt}")

    print(f"\n  {'─'*62}")
    print(f"  FILTRO UMBRAL RSI + ZONA")
    print(f"  {'─'*62}")
    print(f"  RSI_BUY_TRIGGER      : ≤ {u['rsi_buy_trigger']}")
    print(f"  RSI_SELL_TRIGGER     : ≥ {u['rsi_sell_trigger']}")
    print(f"  PROF_ZONA_PCT        : {u['prof_zona_pct']}%")
    print(f"  Divergencias compra  : {u['divergencias_compra_detectadas']} detectadas → "
          f"{u['divergencias_compra_aprobadas']} con zona válida  "
          f"({u['tasa_rechazo_compra_pct']:.1f}% rechazadas)")
    print(f"  Divergencias venta   : {u['divergencias_venta_detectadas']} detectadas → "
          f"{u['divergencias_venta_aprobadas']} con zona válida  "
          f"({u['tasa_rechazo_venta_pct']:.1f}% rechazadas)")
    print(f"  positions_count      :  {s['positions_count_final']:+d}  "
          f"{'(más compras)' if s['positions_count_final'] > 0 else '(más ventas)' if s['positions_count_final'] < 0 else '(equilibrado)'}")
    print("═" * 64)


if __name__ == "__main__":
    main()