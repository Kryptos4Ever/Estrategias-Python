"""
Análisis de Distribución de Frecuencias de Rachas
══════════════════════════════════════════════════
Estrategia: Divergencia RSI · BTC/USDT · Velas Horarias
Parámetros a analizar: RSI_LENGTH=12, N=5  (posición #8 del ranking)

Genera:
  · Histogramas de frecuencia de rachas de compra y venta
  · Tabla de distribución acumulada
  · Análisis estadístico completo (media, mediana, percentiles, etc.)
  · Gráfico de rachas a lo largo del tiempo
  · Guardado como: analisis_rachas_RSI12_N5.png
                   analisis_rachas_RSI12_N5.csv
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ─── Parámetros fijos del análisis ───────────────────────────────────────────
RSI_LENGTH = 12
N          = 5

# ─── Importar config ─────────────────────────────────────────────────────────
try:
    from config import (
        DB_PATH, FECHA_INICIO, FECHA_FIN,
        SALDO_USDT_INICIAL,
        USDT_PCT_TO_USE, BTC_PCT_TO_SELL, BTC_PCT_TO_ACCUMULATE,
        COMMISSION_PCT,
    )
    print("✓ config.py cargado correctamente")
except ImportError:
    print("⚠ config.py no encontrado — usando valores por defecto")
    DB_PATH               = r"btc_hourly.db"
    FECHA_INICIO          = '2021-11-10'
    FECHA_FIN             = '2025-10-06'
    SALDO_USDT_INICIAL    = 1000
    USDT_PCT_TO_USE       = 7
    BTC_PCT_TO_SELL       = 7
    BTC_PCT_TO_ACCUMULATE = 1
    COMMISSION_PCT        = 0.1

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]
OUT_PNG  = f"analisis_rachas_RSI{RSI_LENGTH}_N{N}.png"
OUT_CSV  = f"analisis_rachas_RSI{RSI_LENGTH}_N{N}.csv"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(series: pd.Series, length: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST CON REGISTRO DETALLADO DE RACHAS
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_backtest_detallado(df: pd.DataFrame) -> dict:
    """
    Ejecuta el backtest y devuelve:
      · trade_log     : lista de dicts con cada trade (datetime, type, price, racha_actual)
      · rachas_buy    : lista con la longitud de cada racha de compras completada
      · rachas_sell   : lista con la longitud de cada racha de ventas completada
      · rachas_timeline: lista de dicts {datetime_inicio, datetime_fin, tipo, longitud}
    """
    lows    = df["low"].values.astype(float)
    highs   = df["high"].values.astype(float)
    closes  = df["close"].values.astype(float)
    dts     = df["datetime"].values
    rsi_l   = calcular_rsi(df["low"],  RSI_LENGTH).values.astype(float)
    rsi_h   = calcular_rsi(df["high"], RSI_LENGTH).values.astype(float)

    usdt_balance      = float(SALDO_USDT_INICIAL)
    btc_balance       = 0.0
    btc_en_posiciones = 0.0
    usdt_invertido    = 0.0
    positions_count   = 0

    trade_log         = []   # cada trade individual
    rachas_buy        = []   # longitud de cada racha de compras completada
    rachas_sell       = []   # longitud de cada racha de ventas completada
    rachas_timeline   = []   # historial de rachas con fechas

    # Estado de racha actual
    racha_tipo_actual  = None
    racha_len_actual   = 0
    racha_inicio_dt    = None

    def cerrar_racha_actual(nueva_dt):
        nonlocal racha_tipo_actual, racha_len_actual, racha_inicio_dt
        if racha_tipo_actual is not None and racha_len_actual > 0:
            rachas_timeline.append({
                "tipo"            : racha_tipo_actual,
                "longitud"        : racha_len_actual,
                "datetime_inicio" : str(racha_inicio_dt),
                "datetime_fin"    : str(nueva_dt),
            })
            if racha_tipo_actual == "BUY":
                rachas_buy.append(racha_len_actual)
            else:
                rachas_sell.append(racha_len_actual)

    n = len(lows)
    for i in range(N, n):
        window_lows  = lows[i - N : i]
        window_highs = highs[i - N : i]
        traded       = False

        # ── COMPRA ───────────────────────────────────────────────────────────
        if usdt_balance > 0:
            if lows[i] < window_lows.min():
                idx_min = i - N + int(window_lows.argmin())
                if rsi_l[i] > rsi_l[idx_min]:
                    price         = lows[i]
                    usdt_a_usar   = usdt_balance * (USDT_PCT_TO_USE / 100)
                    comision      = usdt_a_usar * (COMMISSION_PCT / 100)
                    usdt_neto     = usdt_a_usar - comision
                    btc_adquirido = usdt_neto / price

                    usdt_balance      -= usdt_a_usar
                    btc_en_posiciones += btc_adquirido
                    usdt_invertido    += usdt_a_usar
                    positions_count   += 1

                    # Gestión de racha
                    if racha_tipo_actual != "BUY":
                        cerrar_racha_actual(dts[i])
                        racha_tipo_actual = "BUY"
                        racha_len_actual  = 1
                        racha_inicio_dt   = dts[i]
                    else:
                        racha_len_actual += 1

                    trade_log.append({
                        "datetime"     : str(pd.Timestamp(dts[i])),
                        "type"         : "BUY",
                        "price"        : round(price, 2),
                        "rsi_low"      : round(rsi_l[i], 4),
                        "rsi_high"     : round(rsi_h[i], 4),
                        "racha_actual" : racha_len_actual,
                        "usdt_balance" : round(usdt_balance, 2),
                        "btc_posiciones": round(btc_en_posiciones, 8),
                    })
                    traded = True

        # ── VENTA ─────────────────────────────────────────────────────────────
        if not traded and btc_en_posiciones > 0:
            if highs[i] > window_highs.max():
                idx_max = i - N + int(window_highs.argmax())
                if rsi_h[i] < rsi_h[idx_max]:
                    price          = highs[i]
                    btc_procesado  = btc_en_posiciones * (BTC_PCT_TO_SELL / 100)
                    btc_a_acumular = btc_procesado * (BTC_PCT_TO_ACCUMULATE / 100)
                    btc_a_vender   = btc_procesado - btc_a_acumular
                    usdt_bruto     = btc_a_vender * price
                    comision       = usdt_bruto * (COMMISSION_PCT / 100)
                    usdt_neto      = usdt_bruto - comision

                    proporcion         = btc_procesado / (btc_en_posiciones + btc_procesado)
                    usdt_invertido    -= usdt_invertido * proporcion
                    btc_en_posiciones -= btc_procesado
                    btc_balance       += btc_a_acumular
                    usdt_balance      += usdt_neto
                    positions_count   -= 1

                    # Gestión de racha
                    if racha_tipo_actual != "SELL":
                        cerrar_racha_actual(dts[i])
                        racha_tipo_actual = "SELL"
                        racha_len_actual  = 1
                        racha_inicio_dt   = dts[i]
                    else:
                        racha_len_actual += 1

                    trade_log.append({
                        "datetime"      : str(pd.Timestamp(dts[i])),
                        "type"          : "SELL",
                        "price"         : round(price, 2),
                        "rsi_low"       : round(rsi_l[i], 4),
                        "rsi_high"      : round(rsi_h[i], 4),
                        "racha_actual"  : racha_len_actual,
                        "usdt_balance"  : round(usdt_balance, 2),
                        "btc_posiciones": round(btc_en_posiciones, 8),
                    })

    # Cerrar última racha abierta
    cerrar_racha_actual(dts[-1])

    return {
        "trade_log"      : trade_log,
        "rachas_buy"     : rachas_buy,
        "rachas_sell"    : rachas_sell,
        "rachas_timeline": rachas_timeline,
        "precio_final"   : float(closes[-1]),
        "usdt_balance"   : usdt_balance,
        "btc_balance"    : btc_balance,
        "btc_posiciones" : btc_en_posiciones,
        "positions_count": positions_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS DE RACHAS
# ══════════════════════════════════════════════════════════════════════════════

def estadisticas_rachas(rachas: list, nombre: str) -> pd.DataFrame:
    """Tabla de distribución de frecuencias para una lista de rachas."""
    if not rachas:
        return pd.DataFrame()

    s = pd.Series(rachas)
    freq        = s.value_counts().sort_index()
    freq_rel    = (freq / freq.sum() * 100).round(2)
    freq_acum   = freq_rel.cumsum().round(2)

    df_dist = pd.DataFrame({
        "longitud_racha"    : freq.index,
        "frecuencia_abs"    : freq.values,
        "frecuencia_pct"    : freq_rel.values,
        "frecuencia_acum_pct": freq_acum.values,
    })

    print(f"\n{'─'*55}")
    print(f"  Distribución de rachas de {nombre}")
    print(f"{'─'*55}")
    print(f"  Total rachas    : {len(rachas)}")
    print(f"  Media           : {s.mean():.2f}")
    print(f"  Mediana         : {s.median():.1f}")
    print(f"  Moda            : {s.mode().iloc[0]}")
    print(f"  Máximo          : {s.max()}")
    print(f"  Percentil 75    : {s.quantile(0.75):.1f}")
    print(f"  Percentil 90    : {s.quantile(0.90):.1f}")
    print(f"  Percentil 95    : {s.quantile(0.95):.1f}")
    print(f"  % rachas = 1    : {freq_rel.get(1, 0):.1f}%")
    print(f"  % rachas ≤ 3    : {freq_acum.get(min(3, s.max()), freq_rel.sum()):.1f}%")
    print(f"  % rachas ≤ 5    : {freq_acum.get(min(5, s.max()), freq_rel.sum()):.1f}%")
    print(f"\n  Tabla de frecuencias:")
    print(f"  {'Long':>5}  {'Frec':>6}  {'%':>7}  {'Acum%':>7}")
    print(f"  {'─'*32}")
    for _, row in df_dist.iterrows():
        print(f"  {int(row['longitud_racha']):>5}  "
              f"{int(row['frecuencia_abs']):>6}  "
              f"{row['frecuencia_pct']:>6.1f}%  "
              f"{row['frecuencia_acum_pct']:>6.1f}%")

    return df_dist


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def graficar(resultado: dict, df_datos: pd.DataFrame):
    rachas_buy  = resultado["rachas_buy"]
    rachas_sell = resultado["rachas_sell"]
    timeline    = pd.DataFrame(resultado["rachas_timeline"])

    BUY_COLOR  = "#2196F3"   # azul
    SELL_COLOR = "#FF5722"   # naranja-rojo
    BG_COLOR   = "#0D1117"
    PANEL_COLOR= "#161B22"
    TEXT_COLOR = "#E6EDF3"
    GRID_COLOR = "#21262D"

    fig = plt.figure(figsize=(18, 14), facecolor=BG_COLOR)
    fig.suptitle(
        f"Análisis de Distribución de Rachas · Divergencia RSI  |  RSI={RSI_LENGTH}  N={N}  |  "
        f"{FECHA_INICIO} → {FECHA_FIN}",
        fontsize=13, color=TEXT_COLOR, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_hist_b  = fig.add_subplot(gs[0, 0])   # Histograma compras
    ax_hist_s  = fig.add_subplot(gs[0, 1])   # Histograma ventas
    ax_cdf     = fig.add_subplot(gs[0, 2])   # CDF comparativa
    ax_box     = fig.add_subplot(gs[1, 0])   # Boxplot comparativo
    ax_tl      = fig.add_subplot(gs[1, 1:])  # Timeline de rachas
    ax_heat_b  = fig.add_subplot(gs[2, 0])   # Heatmap / barras acumuladas compras
    ax_heat_s  = fig.add_subplot(gs[2, 1])   # Heatmap / barras acumuladas ventas
    ax_stats   = fig.add_subplot(gs[2, 2])   # Tabla de stats

    for ax in [ax_hist_b, ax_hist_s, ax_cdf, ax_box, ax_tl, ax_heat_b, ax_heat_s, ax_stats]:
        ax.set_facecolor(PANEL_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    s_buy  = pd.Series(rachas_buy)
    s_sell = pd.Series(rachas_sell)
    max_racha = max(s_buy.max(), s_sell.max())
    bins = np.arange(0.5, max_racha + 1.5, 1)

    # ── 1. Histograma compras ─────────────────────────────────────────────────
    freq_b = s_buy.value_counts().sort_index()
    ax_hist_b.bar(freq_b.index, freq_b.values, color=BUY_COLOR, alpha=0.85, width=0.7, edgecolor=BG_COLOR)
    ax_hist_b.set_title("Distribución · Rachas COMPRA", fontsize=9, pad=6)
    ax_hist_b.set_xlabel("Longitud de racha")
    ax_hist_b.set_ylabel("Frecuencia absoluta")
    ax_hist_b.xaxis.set_major_locator(MaxNLocator(integer=True))
    for x, y in zip(freq_b.index, freq_b.values):
        ax_hist_b.text(x, y + 0.3, str(y), ha="center", va="bottom",
                       fontsize=7, color=TEXT_COLOR)

    # ── 2. Histograma ventas ──────────────────────────────────────────────────
    freq_s = s_sell.value_counts().sort_index()
    ax_hist_s.bar(freq_s.index, freq_s.values, color=SELL_COLOR, alpha=0.85, width=0.7, edgecolor=BG_COLOR)
    ax_hist_s.set_title("Distribución · Rachas VENTA", fontsize=9, pad=6)
    ax_hist_s.set_xlabel("Longitud de racha")
    ax_hist_s.set_ylabel("Frecuencia absoluta")
    ax_hist_s.xaxis.set_major_locator(MaxNLocator(integer=True))
    for x, y in zip(freq_s.index, freq_s.values):
        ax_hist_s.text(x, y + 0.3, str(y), ha="center", va="bottom",
                       fontsize=7, color=TEXT_COLOR)

    # ── 3. CDF comparativa ───────────────────────────────────────────────────
    for s, color, label in [(s_buy, BUY_COLOR, "Compra"), (s_sell, SELL_COLOR, "Venta")]:
        vals = np.sort(s.values)
        cdf  = np.arange(1, len(vals)+1) / len(vals) * 100
        ax_cdf.step(vals, cdf, color=color, linewidth=2, label=label, where="post")
    ax_cdf.axhline(75, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_cdf.axhline(90, color="gray", linestyle=":",  linewidth=0.8, alpha=0.6)
    ax_cdf.text(max_racha * 0.98, 76, "P75", color="gray", fontsize=7, ha="right")
    ax_cdf.text(max_racha * 0.98, 91, "P90", color="gray", fontsize=7, ha="right")
    ax_cdf.set_title("Frecuencia Acumulada (CDF)", fontsize=9, pad=6)
    ax_cdf.set_xlabel("Longitud de racha")
    ax_cdf.set_ylabel("% rachas ≤ longitud")
    ax_cdf.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    ax_cdf.set_ylim(0, 105)

    # ── 4. Boxplot comparativo ────────────────────────────────────────────────
    bp = ax_box.boxplot(
        [rachas_buy, rachas_sell],
        labels=["Compra", "Venta"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=TEXT_COLOR),
        capprops=dict(color=TEXT_COLOR),
        flierprops=dict(markerfacecolor=TEXT_COLOR, marker="o", markersize=4, alpha=0.5),
    )
    bp["boxes"][0].set_facecolor(BUY_COLOR)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(SELL_COLOR)
    bp["boxes"][1].set_alpha(0.7)
    ax_box.set_title("Boxplot Comparativo", fontsize=9, pad=6)
    ax_box.set_ylabel("Longitud de racha")

    # ── 5. Timeline de rachas ─────────────────────────────────────────────────
    if not timeline.empty:
        timeline["datetime_inicio"] = pd.to_datetime(timeline["datetime_inicio"])
        timeline["datetime_fin"]    = pd.to_datetime(timeline["datetime_fin"])

        for _, row in timeline.iterrows():
            color = BUY_COLOR if row["tipo"] == "BUY" else SELL_COLOR
            alpha = min(0.3 + row["longitud"] * 0.08, 0.95)
            ax_tl.barh(
                y     = row["longitud"],
                width = (row["datetime_fin"] - row["datetime_inicio"]).total_seconds() / 3600,
                left  = row["datetime_inicio"].timestamp() / 3600,
                height= 0.6,
                color = color,
                alpha = alpha,
            )
        # Formatear eje x con fechas aproximadas
        tick_ts = [pd.Timestamp(f"{y}-01-01") for y in range(2022, 2026)]
        ax_tl.set_xticks([t.timestamp() / 3600 for t in tick_ts])
        ax_tl.set_xticklabels([t.strftime("%Y") for t in tick_ts], fontsize=8)
        ax_tl.set_title("Timeline de Rachas a lo largo del período", fontsize=9, pad=6)
        ax_tl.set_xlabel("Año")
        ax_tl.set_ylabel("Longitud de racha")
        ax_tl.yaxis.set_major_locator(MaxNLocator(integer=True))

        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=BUY_COLOR, label="Compra"), Patch(facecolor=SELL_COLOR, label="Venta")]
        ax_tl.legend(handles=legend_els, fontsize=8, facecolor=PANEL_COLOR,
                     labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)

    # ── 6. Barras apiladas % compras ──────────────────────────────────────────
    pct_b = (freq_b / freq_b.sum() * 100).round(2)
    acum_b = pct_b.cumsum()
    ax_heat_b.barh(freq_b.index, pct_b.values, color=BUY_COLOR, alpha=0.85, height=0.6, edgecolor=BG_COLOR)
    for x, y in zip(pct_b.values, freq_b.index):
        ax_heat_b.text(x + 0.3, y, f"{x:.1f}%", va="center", fontsize=7, color=TEXT_COLOR)
    ax_heat_b.set_title("% por longitud · Compra", fontsize=9, pad=6)
    ax_heat_b.set_xlabel("Frecuencia relativa (%)")
    ax_heat_b.set_ylabel("Longitud de racha")
    ax_heat_b.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 7. Barras apiladas % ventas ───────────────────────────────────────────
    pct_s = (freq_s / freq_s.sum() * 100).round(2)
    ax_heat_s.barh(freq_s.index, pct_s.values, color=SELL_COLOR, alpha=0.85, height=0.6, edgecolor=BG_COLOR)
    for x, y in zip(pct_s.values, freq_s.index):
        ax_heat_s.text(x + 0.3, y, f"{x:.1f}%", va="center", fontsize=7, color=TEXT_COLOR)
    ax_heat_s.set_title("% por longitud · Venta", fontsize=9, pad=6)
    ax_heat_s.set_xlabel("Frecuencia relativa (%)")
    ax_heat_s.set_ylabel("Longitud de racha")
    ax_heat_s.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 8. Tabla de estadísticas ──────────────────────────────────────────────
    ax_stats.axis("off")
    stats_data = [
        ["Métrica",           "COMPRA",                        "VENTA"],
        ["Total rachas",      str(len(rachas_buy)),             str(len(rachas_sell))],
        ["Media",             f"{s_buy.mean():.2f}",           f"{s_sell.mean():.2f}"],
        ["Mediana",           f"{s_buy.median():.1f}",         f"{s_sell.median():.1f}"],
        ["Moda",              str(s_buy.mode().iloc[0]),        str(s_sell.mode().iloc[0])],
        ["Máximo",            str(int(s_buy.max())),            str(int(s_sell.max()))],
        ["P75",               f"{s_buy.quantile(0.75):.1f}",   f"{s_sell.quantile(0.75):.1f}"],
        ["P90",               f"{s_buy.quantile(0.90):.1f}",   f"{s_sell.quantile(0.90):.1f}"],
        ["P95",               f"{s_buy.quantile(0.95):.1f}",   f"{s_sell.quantile(0.95):.1f}"],
        ["% racha = 1",       f"{(s_buy==1).mean()*100:.1f}%", f"{(s_sell==1).mean()*100:.1f}%"],
        ["% racha ≤ 3",
            f"{(s_buy<=3).mean()*100:.1f}%",
            f"{(s_sell<=3).mean()*100:.1f}%"],
        ["% racha ≤ 5",
            f"{(s_buy<=5).mean()*100:.1f}%",
            f"{(s_sell<=5).mean()*100:.1f}%"],
    ]

    tbl = ax_stats.table(
        cellText  = stats_data[1:],
        colLabels = stats_data[0],
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_COLOR if r % 2 == 0 else "#1C2128")
        cell.set_edgecolor(GRID_COLOR)
        cell.set_text_props(color=TEXT_COLOR)
        if r == 0:
            cell.set_facecolor("#2D333B")
            cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
        if c == 1 and r > 0:
            cell.set_text_props(color=BUY_COLOR)
        if c == 2 and r > 0:
            cell.set_text_props(color=SELL_COLOR)

    ax_stats.set_title("Estadísticas Comparativas", fontsize=9, pad=6, color=TEXT_COLOR)

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"\n✓ Gráfico guardado: {OUT_PNG}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ANÁLISIS DE RACHAS · RSI={RSI_LENGTH} · N={N} · Velas Horarias{' '*14}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    df = cargar_datos()
    if df.empty:
        print("ERROR: No hay datos.")
        return

    print(f"\nEjecutando backtest con RSI={RSI_LENGTH}, N={N}...")
    resultado = ejecutar_backtest_detallado(df)

    # Estadísticas en consola
    df_stats_b = estadisticas_rachas(resultado["rachas_buy"],  "COMPRA")
    df_stats_s = estadisticas_rachas(resultado["rachas_sell"], "VENTA")

    # Guardar CSV con todas las rachas
    df_rachas = pd.DataFrame(resultado["rachas_timeline"])
    df_rachas.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Rachas guardadas: {OUT_CSV}")

    # Portfolio final
    pf = resultado["usdt_balance"] + (resultado["btc_balance"] + resultado["btc_posiciones"]) * resultado["precio_final"]
    pnl = (pf - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL * 100
    print(f"\n  Portfolio final : ${pf:,.2f}  ({pnl:+.2f}%)")
    print(f"  Total trades    : {len(resultado['trade_log'])}")

    print("\nGenerando gráficos...")
    graficar(resultado, df)


if __name__ == "__main__":
    main()
