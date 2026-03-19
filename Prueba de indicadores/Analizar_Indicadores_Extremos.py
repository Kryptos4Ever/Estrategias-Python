"""
Analizador de Correlación de Indicadores con Extremos Locales — BTC/USDT
─────────────────────────────────────────────────────────────────────────
Detecta mínimos y máximos locales usando dos métodos:
  A) Ventana fija: extremo de las últimas/siguientes 1440 velas (1 día)
  B) Cambio de tendencia: precio se mueve 5% desde el extremo

Para cada extremo detectado, registra el valor de 8 indicadores:
  1. Distancia % a EMA200
  2. Bollinger Bands %B
  3. Stochastic RSI
  4. OBV divergencia
  5. CCI
  6. Williams %R
  7. Patrones de vela (Hammer / Engulfing)
  8. Pivot Points (distancia a S1/R1)

Calcula la correlación estadística de cada indicador con los extremos
y produce un reporte + gráficos para orientar el diseño de señales.

Salida:
  · reporte en consola
  · indicadores_extremos.png
  · indicadores_extremos.json
"""

import sqlite3
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from config import DB_PATH, FECHA_INICIO, FECHA_FIN

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

# ─────────────────────────────────────────────────────────────
# PARÁMETROS DEL ANÁLISIS
# ─────────────────────────────────────────────────────────────
VENTANA_FIJA     = 1440      # velas a cada lado para extremo por ventana fija
PCT_TENDENCIA    = 0.05      # 5% de movimiento para confirmar extremo por tendencia
MUESTRA_MAX      = None   # limitar velas para acelerar (None = todas)

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    print(f"Conectando a {DB_PATH}...")
    conn  = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM   {DB_TABLE}
        ORDER  BY timestamp ASC
    """
    if MUESTRA_MAX:
        query += f" LIMIT {MUESTRA_MAX}"

    df = pd.read_sql(query, conn)
    conn.close()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    if FECHA_INICIO:
        df = df[df["datetime"] >= pd.to_datetime(FECHA_INICIO)]
    if FECHA_FIN:
        df = df[df["datetime"] <= pd.to_datetime(FECHA_FIN)]

    df = df.reset_index(drop=True)
    print(f"Velas cargadas : {len(df):,}  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")
    return df


# ─────────────────────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(close: pd.Series, rsi_len=14, stoch_len=14, k=3, d=3) -> pd.DataFrame:
    rsi       = calc_rsi(close, rsi_len)
    rsi_min   = rsi.rolling(stoch_len).min()
    rsi_max   = rsi.rolling(stoch_len).max()
    stoch_k   = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    stoch_k_s = stoch_k.rolling(k).mean()
    stoch_d   = stoch_k_s.rolling(d).mean()
    return pd.DataFrame({"stoch_k": stoch_k_s, "stoch_d": stoch_d})

def calc_bollinger(close: pd.Series, length=20, std_mult=2.0) -> pd.DataFrame:
    mid   = close.rolling(length).mean()
    std   = close.rolling(length).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    width = (upper - lower) / mid
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                          "bb_pct_b": pct_b, "bb_width": width})

def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, length=20) -> pd.Series:
    tp      = (high + low + close) / 3
    tp_mean = tp.rolling(length).mean()
    tp_mad  = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - tp_mean) / (0.015 * tp_mad + 1e-10)

def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length=14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def calc_obv_divergence(close: pd.Series, obv: pd.Series, length=20) -> pd.Series:
    """
    Divergencia OBV: diferencia normalizada entre la tendencia del precio
    y la tendencia del OBV en una ventana deslizante.
    Positivo = divergencia alcista (precio baja, OBV sube)
    Negativo = divergencia bajista (precio sube, OBV baja)
    """
    price_slope = close.diff(length) / close.shift(length)
    obv_slope   = obv.diff(length)   / (obv.shift(length).abs() + 1e-10)
    return obv_slope - price_slope

def calc_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Pivot points clásicos basados en la vela anterior."""
    pp = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    r1 = 2 * pp - low.shift(1)
    s1 = 2 * pp - high.shift(1)
    r2 = pp + (high.shift(1) - low.shift(1))
    s2 = pp - (high.shift(1) - low.shift(1))
    dist_r1 = (r1 - close) / close * 100   # % hasta R1 (negativo si precio > R1)
    dist_s1 = (close - s1) / close * 100   # % sobre S1 (negativo si precio < S1)
    return pd.DataFrame({"pp": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2,
                          "dist_r1": dist_r1, "dist_s1": dist_s1})

def detectar_patrones_vela(open_: pd.Series, high: pd.Series,
                            low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Hammer:         sombra inferior >= 2x cuerpo, sombra superior pequeña, cuerpo en tercio superior
    Inverted Hammer: sombra superior >= 2x cuerpo, sombra inferior pequeña
    Bullish Engulfing: vela alcista que engulle la vela bajista anterior
    Bearish Engulfing: vela bajista que engulle la vela alcista anterior
    """
    body      = (close - open_).abs()
    upper_sh  = high - close.clip(lower=open_)
    lower_sh  = open_.clip(upper=close) - low

    hammer    = (lower_sh >= 2 * body) & (upper_sh <= 0.3 * body) & (body > 0)
    inv_hammer = (upper_sh >= 2 * body) & (lower_sh <= 0.3 * body) & (body > 0)

    bull_eng  = ((close > open_) &
                 (close.shift(1) < open_.shift(1)) &
                 (close > open_.shift(1)) &
                 (open_ < close.shift(1)))

    bear_eng  = ((close < open_) &
                 (close.shift(1) > open_.shift(1)) &
                 (close < open_.shift(1)) &
                 (open_ > close.shift(1)))

    return pd.DataFrame({
        "hammer"    : hammer.astype(int),
        "inv_hammer": inv_hammer.astype(int),
        "bull_eng"  : bull_eng.astype(int),
        "bear_eng"  : bear_eng.astype(int),
    })


# ─────────────────────────────────────────────────────────────
# DETECCIÓN DE EXTREMOS LOCALES
# ─────────────────────────────────────────────────────────────

def extremos_ventana_fija(close: pd.Series, ventana: int) -> pd.DataFrame:
    """
    Mínimo local: close es el mínimo de [i-ventana, i+ventana]
    Máximo local: close es el máximo de [i-ventana, i+ventana]
    """
    roll_min = close.rolling(window=2*ventana+1, center=True).min()
    roll_max = close.rolling(window=2*ventana+1, center=True).max()
    es_min   = (close == roll_min).astype(int)
    es_max   = (close == roll_max).astype(int)
    return pd.DataFrame({"local_min_fixed": es_min, "local_max_fixed": es_max})

def extremos_por_tendencia(close: pd.Series, pct: float) -> pd.DataFrame:
    """
    Detecta mínimos y máximos confirmados:
    - Mínimo confirmado: desde ese punto el precio subió al menos pct% antes de bajar pct%
    - Máximo confirmado: desde ese punto el precio bajó al menos pct% antes de subir pct%
    Algoritmo de zigzag eficiente.
    """
    arr    = close.values
    n      = len(arr)
    mins   = np.zeros(n, dtype=int)
    maxs   = np.zeros(n, dtype=int)

    # Estado: buscando mínimo (0) o máximo (1)
    estado     = 0
    idx_ref    = 0
    precio_ref = arr[0]

    for i in range(1, n):
        if estado == 0:  # buscando mínimo
            if arr[i] < precio_ref:
                precio_ref = arr[i]
                idx_ref    = i
            elif arr[i] >= precio_ref * (1 + pct):
                mins[idx_ref] = 1
                estado     = 1
                precio_ref = arr[i]
                idx_ref    = i
        else:  # buscando máximo
            if arr[i] > precio_ref:
                precio_ref = arr[i]
                idx_ref    = i
            elif arr[i] <= precio_ref * (1 - pct):
                maxs[idx_ref] = 1
                estado     = 0
                precio_ref = arr[i]
                idx_ref    = i

    return pd.DataFrame({"local_min_trend": mins, "local_max_trend": maxs})


# ─────────────────────────────────────────────────────────────
# ANÁLISIS DE CORRELACIÓN
# ─────────────────────────────────────────────────────────────

def stats_en_extremos(indicador: pd.Series, extremos: pd.Series,
                       nombre: str, tipo: str) -> dict:
    """
    Para cada extremo detectado, toma el valor del indicador
    y calcula estadísticas descriptivas + correlación punto-biserial.
    """
    from scipy import stats as sp_stats
    mask    = extremos == 1
    vals    = indicador[mask].dropna()
    no_vals = indicador[~mask].dropna()

    if len(vals) < 10:
        return {}

    # Correlación punto-biserial (indicador numérico vs extremo 0/1)
    combined = pd.concat([indicador, extremos], axis=1).dropna()
    if len(combined) > 100:
        corr, pval = sp_stats.pointbiserialr(
            combined.iloc[:, 1].values,
            combined.iloc[:, 0].values
        )
    else:
        corr, pval = 0.0, 1.0

    percentiles = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])

    return {
        "indicador"        : nombre,
        "tipo_extremo"     : tipo,
        "n_extremos"       : int(mask.sum()),
        "media_en_extremo" : round(float(vals.mean()), 4),
        "media_general"    : round(float(no_vals.mean()), 4),
        "diferencia"       : round(float(vals.mean() - no_vals.mean()), 4),
        "mediana"          : round(float(np.median(vals)), 4),
        "std"              : round(float(vals.std()), 4),
        "p05"              : round(float(percentiles[0]), 4),
        "p10"              : round(float(percentiles[1]), 4),
        "p25"              : round(float(percentiles[2]), 4),
        "p50"              : round(float(percentiles[3]), 4),
        "p75"              : round(float(percentiles[4]), 4),
        "p90"              : round(float(percentiles[5]), 4),
        "p95"              : round(float(percentiles[6]), 4),
        "correlacion"      : round(float(corr), 4),
        "p_valor"          : round(float(pval), 6),
        "significativo"    : pval < 0.05,
    }


# ─────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generar_graficos(df: pd.DataFrame, indicadores: dict,
                     extremos: dict, resultados: list):

    fig = plt.figure(figsize=(22, 26))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)
    bg  = "#16213e"
    fig.patch.set_facecolor("#1a1a2e")

    # Colores por tipo de extremo
    c_min_f = "#00BCD4"   # mínimo ventana fija
    c_max_f = "#FF5722"   # máximo ventana fija
    c_min_t = "#4CAF50"   # mínimo tendencia
    c_max_t = "#E91E63"   # máximo tendencia

    # ── Panel 1: Precio + extremos detectados ────────────────
    ax0 = fig.add_subplot(gs[0, :])
    muestra_idx = df.index[::10]   # cada 10 velas para no saturar
    ax0.plot(df.loc[muestra_idx, "datetime"], df.loc[muestra_idx, "close"],
             color="white", linewidth=0.5, alpha=0.6, label="Precio BTC")

    for col, color, label in [
        ("local_min_fixed", c_min_f, f"Mín ventana ({VENTANA_FIJA})"),
        ("local_max_fixed", c_max_f, f"Máx ventana ({VENTANA_FIJA})"),
        ("local_min_trend", c_min_t, f"Mín tendencia ({PCT_TENDENCIA*100:.0f}%)"),
        ("local_max_trend", c_max_t, f"Máx tendencia ({PCT_TENDENCIA*100:.0f}%)"),
    ]:
        if col in df.columns:
            mask = df[col] == 1
            ax0.scatter(df.loc[mask, "datetime"], df.loc[mask, "close"],
                        color=color, s=8, alpha=0.8, label=f"{label} ({mask.sum():,})", zorder=5)

    ax0.set_yscale("log")
    ax0.set_title("Precio BTC + Extremos Locales Detectados", fontweight="bold")
    ax0.set_ylabel("Precio (log)")
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(True, alpha=0.2)

    # ── Paneles 2-9: Distribución de cada indicador en extremos ─
    indicador_cols = [
        ("ema200_dist",  "Distancia % EMA200",    "Mín=negativo (precio bajo EMA)"),
        ("bb_pct_b",     "Bollinger %B",           "Mín=cerca de 0 (banda inf)"),
        ("stoch_k",      "Stochastic RSI (K)",     "Mín=<20  Máx=>80"),
        ("obv_div",      "OBV Divergencia",        "Mín=positivo (divergencia alcista)"),
        ("cci",          "CCI",                    "Mín=<-100  Máx=>+100"),
        ("williams_r",   "Williams %R",            "Mín=<-80  Máx=>-20"),
        ("dist_s1",      "Distancia a S1 (%)",     "Mín=cerca o bajo S1"),
        ("hammer",       "Hammer (patrón vela)",   "Binario: 1=presente"),
    ]

    posiciones = [(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)]
    pares_extremo = [
        ("local_min_fixed", c_min_f, "Mín-Fijo"),
        ("local_min_trend", c_min_t, "Mín-Tend"),
        ("local_max_fixed", c_max_f, "Máx-Fijo"),
        ("local_max_trend", c_max_t, "Máx-Tend"),
    ]

    for idx, ((col, titulo, subtitulo), pos) in enumerate(zip(indicador_cols[:6], posiciones)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.set_facecolor(bg)

        if col not in df.columns:
            ax.set_title(f"{titulo}\n(no disponible)")
            continue

        serie = df[col].dropna()

        # Distribución general (fondo gris)
        q1, q99 = serie.quantile(0.01), serie.quantile(0.99)
        bins = np.linspace(q1, q99, 60)
        ax.hist(serie.clip(q1, q99), bins=bins, color="gray", alpha=0.3,
                label="General", density=True)

        # Distribución en cada tipo de extremo
        for ext_col, color, label in pares_extremo:
            if ext_col not in df.columns:
                continue
            mask = df[ext_col] == 1
            vals = df.loc[mask, col].dropna().clip(q1, q99)
            if len(vals) > 5:
                ax.hist(vals, bins=bins, color=color, alpha=0.5,
                        label=f"{label} (n={len(vals):,})", density=True)
                ax.axvline(vals.median(), color=color, linewidth=1.5,
                           linestyle="--", alpha=0.9)

        ax.set_title(f"{titulo}\n{subtitulo}", fontweight="bold", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Estilo ejes superiores
    for ax in [ax0]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle("Correlación de Indicadores con Extremos Locales — BTC/USDT",
                 fontsize=15, fontweight="bold", color="white", y=1.005)

    nombre = "indicadores_extremos.png"
    plt.savefig(nombre, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Gráfico guardado: {nombre}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────

def imprimir_reporte(resultados: list):
    sep = "=" * 72

    # Agrupar por tipo de extremo
    tipos = ["local_min_fixed", "local_max_fixed", "local_min_trend", "local_max_trend"]
    labels = {
        "local_min_fixed" : f"MÍNIMOS — Ventana fija ({VENTANA_FIJA} velas)",
        "local_max_fixed" : f"MÁXIMOS — Ventana fija ({VENTANA_FIJA} velas)",
        "local_min_trend" : f"MÍNIMOS — Tendencia ({PCT_TENDENCIA*100:.0f}%)",
        "local_max_trend" : f"MÁXIMOS — Tendencia ({PCT_TENDENCIA*100:.0f}%)",
    }

    for tipo in tipos:
        grupo = [r for r in resultados if r.get("tipo_extremo") == tipo and r]
        if not grupo:
            continue

        print(f"\n{sep}")
        print(f"  {labels[tipo]}")
        print(f"  Extremos detectados: {grupo[0]['n_extremos']:,}")
        print(sep)
        print(f"  {'Indicador':<25} {'Media extrem':>12} {'Media gral':>12} "
              f"{'Diferencia':>12} {'Correlac.':>10} {'Signif.':>8}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

        # Ordenar por correlación absoluta
        grupo_ord = sorted(grupo, key=lambda x: abs(x.get("correlacion", 0)), reverse=True)
        for r in grupo_ord:
            sig = "✓" if r.get("significativo") else " "
            print(f"  {r['indicador']:<25} {r['media_en_extremo']:>12.4f} "
                  f"{r['media_general']:>12.4f} {r['diferencia']:>12.4f} "
                  f"{r['correlacion']:>10.4f} {sig:>8}")

    # Ranking global de indicadores
    print(f"\n{sep}")
    print("  RANKING GLOBAL — Indicadores más correlacionados con extremos")
    print(sep)

    ranking = {}
    for r in resultados:
        if not r:
            continue
        ind = r["indicador"]
        if ind not in ranking:
            ranking[ind] = []
        ranking[ind].append(abs(r.get("correlacion", 0)))

    ranking_avg = {k: np.mean(v) for k, v in ranking.items()}
    for i, (ind, avg) in enumerate(sorted(ranking_avg.items(),
                                           key=lambda x: x[1], reverse=True), 1):
        barra = "█" * int(avg * 200)
        print(f"  {i}. {ind:<25}  corr.media={avg:.4f}  {barra}")

    print(f"\n{sep}")
    print("  SUGERENCIAS DE USO EN ESTRATEGIA")
    print(sep)

    top3 = sorted(ranking_avg.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  Los 3 indicadores con mayor correlación son:")
    for ind, avg in top3:
        # Buscar los valores medianos en mínimos
        r_min = next((r for r in resultados
                      if r.get("indicador") == ind and "min" in r.get("tipo_extremo", "")), {})
        r_max = next((r for r in resultados
                      if r.get("indicador") == ind and "max" in r.get("tipo_extremo", "")), {})
        print(f"\n  → {ind}  (correlación media: {avg:.4f})")
        if r_min:
            print(f"     En mínimos: mediana={r_min.get('mediana', 'N/A'):.2f}  "
                  f"P10={r_min.get('p10', 'N/A'):.2f}  P90={r_min.get('p90', 'N/A'):.2f}")
        if r_max:
            print(f"     En máximos: mediana={r_max.get('mediana', 'N/A'):.2f}  "
                  f"P10={r_max.get('p10', 'N/A'):.2f}  P90={r_max.get('p90', 'N/A'):.2f}")
    print(sep)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  ANALIZADOR DE INDICADORES VS EXTREMOS LOCALES — BTC/USDT")
    print("=" * 72)

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Calcular todos los indicadores
    print("\nCalculando indicadores...")

    ema200           = calc_ema(df["close"], 200)
    df["ema200_dist"] = (df["close"] - ema200) / ema200 * 100

    bb               = calc_bollinger(df["close"])
    df["bb_pct_b"]   = bb["bb_pct_b"]

    stoch            = calc_stoch_rsi(df["close"])
    df["stoch_k"]    = stoch["stoch_k"]
    df["stoch_d"]    = stoch["stoch_d"]

    obv              = calc_obv(df["close"], df["volume"])
    df["obv_div"]    = calc_obv_divergence(df["close"], obv)

    df["cci"]        = calc_cci(df["high"], df["low"], df["close"])
    df["williams_r"] = calc_williams_r(df["high"], df["low"], df["close"])

    pivots           = calc_pivot_points(df["high"], df["low"], df["close"])
    df["dist_s1"]    = pivots["dist_s1"]
    df["dist_r1"]    = pivots["dist_r1"]

    patrones         = detectar_patrones_vela(df["open"], df["high"],
                                               df["low"], df["close"])
    df["hammer"]     = patrones["hammer"]
    df["bull_eng"]   = patrones["bull_eng"]
    df["bear_eng"]   = patrones["bear_eng"]

    print("  Indicadores calculados ✓")

    # 3. Detectar extremos locales
    print("\nDetectando extremos locales...")

    ext_fija   = extremos_ventana_fija(df["close"], VENTANA_FIJA)
    df["local_min_fixed"] = ext_fija["local_min_fixed"]
    df["local_max_fixed"] = ext_fija["local_max_fixed"]

    ext_tend   = extremos_por_tendencia(df["close"], PCT_TENDENCIA)
    df["local_min_trend"] = ext_tend["local_min_trend"]
    df["local_max_trend"] = ext_tend["local_max_trend"]

    for col in ["local_min_fixed", "local_max_fixed", "local_min_trend", "local_max_trend"]:
        print(f"  {col:<22}: {df[col].sum():>6,} extremos")

    # 4. Calcular correlaciones
    print("\nCalculando correlaciones...")

    indicadores_num = [
        ("ema200_dist", "Distancia %EMA200"),
        ("bb_pct_b",    "Bollinger %B"),
        ("stoch_k",     "Stoch RSI (K)"),
        ("obv_div",     "OBV Divergencia"),
        ("cci",         "CCI"),
        ("williams_r",  "Williams %R"),
        ("dist_s1",     "Distancia S1%"),
        ("hammer",      "Hammer"),
        ("bull_eng",    "Bullish Engulf."),
        ("bear_eng",    "Bearish Engulf."),
    ]

    extremos_cols = [
        ("local_min_fixed", "local_min_fixed"),
        ("local_max_fixed", "local_max_fixed"),
        ("local_min_trend", "local_min_trend"),
        ("local_max_trend", "local_max_trend"),
    ]

    resultados = []
    for ind_col, ind_nombre in indicadores_num:
        for ext_col, ext_tipo in extremos_cols:
            r = stats_en_extremos(df[ind_col], df[ext_col], ind_nombre, ext_tipo)
            resultados.append(r)

    # 5. Reporte
    imprimir_reporte(resultados)

    # 6. Guardar JSON
    with open("indicadores_extremos.json", "w") as f:
        json.dump({
            "config": {
                "ventana_fija"    : VENTANA_FIJA,
                "pct_tendencia"   : PCT_TENDENCIA,
                "muestra_max"     : MUESTRA_MAX,
            },
            "resultados": [r for r in resultados if r],
        }, f, indent=2, default=str)
    print("\nDatos guardados: indicadores_extremos.json")

    # 7. Gráficos
    print("Generando gráficos...")
    generar_graficos(df, indicadores_num, extremos_cols, resultados)


if __name__ == "__main__":
    try:
        from scipy import stats as _
    except ImportError:
        print("Instalá scipy: pip install scipy")
        exit(1)
    main()
