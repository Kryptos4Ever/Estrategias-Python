"""
Graficador v2 — Estrategias BTC/USDT
══════════════════════════════════════════════════════════════════
Mejoras respecto a v1:
  • Autónomo — no requiere config.py (parámetros por argumento o constantes)
  • Panel 6: Score Compuesto Bot/Top con umbrales y alertas
  • Panel 7: Drawdown continuo del portfolio
  • Comparación Buy & Hold en panel de portfolio
  • Métricas adicionales: Sharpe ratio, Win rate, Profit Factor, Calmar
  • Tema oscuro opcional (DARK_MODE = True)
  • Anotaciones de trades significativos en precio
  • Resumen enriquecido en consola

Ejecutar:
    python Graficador_v2.py
    python Graficador_v2.py --json resultados.json --db datos.db --dark
"""

import json
import argparse
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ─── Importar config ─────────────────────────────────────────────────────────
try:
    from config import (
        DB_PATH, RESULTS_JSON, FECHA_INICIO, FECHA_FIN, SALDO_USDT_INICIAL,
        COMMISSION_PCT, 
        DARK_MODE, OUTPUT_PNG, DPI,
    )
    # Adaptar comisión: en config es porcentaje (0.1), internamente usamos fracción
    COMMISSION_PCT_F = COMMISSION_PCT / 100.0
except ImportError as e:
    print(f"✗ No se encontró config.py: {e}")
    print("  Crea config.py en el mismo directorio.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIGHT = {
    "bg":         "#f8f9fa",
    "panel_bg":   "#fdfdfd",
    "text":       "#111111",
    "grid":       "#dddddd",
    "price_line": "#222222",
    "buy_ok":     "#16a34a",
    "sell_ok":    "#dc2626",
    "buy_ign":    "#9ca3af",
    "sell_ign":   "#fca5a5",
    "usdt_line":  "#2563eb",
    "btc_line":   "#ea580c",
    "port_line":  "#16a34a",
    "score_bot":  "#16a34a",
    "score_top":  "#dc2626",
    "dd_line":    "#7c3aed",
    "bh_line":    "#0891b2",
    "pp_line":    "#2563eb",
}
DARK = {
    "bg":         "#0a0e1a",
    "panel_bg":   "#111827",
    "text":       "#e2e8f0",
    "grid":       "#1e293b",
    "price_line": "#cbd5e1",
    "buy_ok":     "#00ff88",
    "sell_ok":    "#ff3366",
    "buy_ign":    "#475569",
    "sell_ign":   "#7f1d1d",
    "usdt_line":  "#60a5fa",
    "btc_line":   "#fb923c",
    "port_line":  "#4ade80",
    "score_bot":  "#00ff88",
    "score_top":  "#ff3366",
    "dd_line":    "#a78bfa",
    "bh_line":    "#22d3ee",
    "pp_line":    "#818cf8",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_drawdown(series: np.ndarray) -> np.ndarray:
    pico = np.maximum.accumulate(series)
    return (series - pico) / np.where(pico == 0, 1, pico) * 100

def calcular_sharpe(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

def calcular_calmar(portfolio: np.ndarray, periods_per_year: float = 8760) -> float:
    if len(portfolio) < 2:
        return 0.0
    ret_anual = (portfolio[-1] / portfolio[0]) ** (periods_per_year / len(portfolio)) - 1
    max_dd    = abs(calcular_drawdown(portfolio).min()) / 100
    return float(ret_anual / max_dd) if max_dd > 0 else 0.0

def interpolar_serie(price_index, trade_df, col, fill_value=0.0):
    if col not in trade_df.columns:
        return pd.Series(fill_value, index=price_index)
    s = trade_df[col].copy()
    s = s[~s.index.duplicated(keep="last")]
    s = s.reindex(price_index, method="ffill")
    return s.fillna(fill_value)

def separar_trades(df):
    if "ignorado" in df.columns:
        ejecutados = df[~df["ignorado"].fillna(False).astype(bool)]
        ignorados  = df[ df["ignorado"].fillna(False).astype(bool)]
    else:
        ejecutados, ignorados = df, df.iloc[0:0]
    return (ejecutados[ejecutados["type"]=="BUY"],
            ejecutados[ejecutados["type"]=="SELL"],
            ignorados[ignorados["type"]=="BUY"],
            ignorados[ignorados["type"]=="SELL"])

def usd_fmt(x, _): return f"${x:,.0f}"
def pct_fmt(x, _): return f"{x:.0f}%"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASE PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GraficadorV2:

    def __init__(self, dark_mode=DARK_MODE):
        self.price_data   = None
        self.trade_data   = None
        self.results_data = None
        self.C            = DARK if dark_mode else LIGHT
        self.dark         = dark_mode

    # ── Carga ────────────────────────────────────────────────────

    def cargar_precios(self, db_path=DB_PATH, table=None) -> bool:
        if table is None:
            table = os.path.splitext(os.path.basename(db_path))[0]
        try:
            conn = sqlite3.connect(db_path)
            df   = pd.read_sql(f"SELECT * FROM {table} ORDER BY timestamp ASC", conn)
            conn.close()
            df["datetime"]  = pd.to_datetime(df["timestamp"], unit="ms")
            self.price_data = df
            print(f"✓ Precios: {len(df):,} velas  [{df.datetime.min():%Y-%m-%d} → {df.datetime.max():%Y-%m-%d}]")
            return True
        except Exception as e:
            print(f"✗ Precios: {e}"); return False

    def cargar_resultados(self, json_path=RESULTS_JSON) -> bool:
        try:
            with open(json_path) as f:
                self.results_data = json.load(f)
            trades = self.results_data.get("trade_history", [])
            self.trade_data = pd.DataFrame(trades)
            self.trade_data["datetime"] = pd.to_datetime(self.trade_data["datetime"])
            print(f"✓ Resultados: {len(self.trade_data):,} registros desde '{json_path}'")
            return True
        except Exception as e:
            print(f"✗ Resultados: {e}"); return False

    # ── Preparación ──────────────────────────────────────────────

    def _rango(self):
        fi = pd.to_datetime(FECHA_INICIO) if FECHA_INICIO else self.price_data["datetime"].min()
        ff = pd.to_datetime(FECHA_FIN)    if FECHA_FIN    else self.price_data["datetime"].max()
        return fi, ff

    def _price_cont(self):
        fi, ff = self._rango()
        price  = self.price_data[
            (self.price_data["datetime"] >= fi) &
            (self.price_data["datetime"] <= ff)
        ].copy().set_index("datetime")

        if "ignorado" in self.trade_data.columns:
            td = self.trade_data[~self.trade_data["ignorado"].fillna(False).astype(bool)].copy()
        else:
            td = self.trade_data.copy()
        td = td.set_index("datetime")

        for col, fill in [("usdt_balance", SALDO_USDT_INICIAL),
                          ("btc_balance", 0.0),
                          ("btc_en_posiciones", 0.0),
                          ("positions_count", 0.0)]:
            price[col] = interpolar_serie(price.index, td, col, fill)

        if "precio_promedio_posiciones" in td.columns:
            pp = td["precio_promedio_posiciones"].replace(0, np.nan)
            pp = pp[~pp.index.duplicated(keep="last")]
            price["precio_promedio"] = pp.reindex(price.index, method="ffill")
        else:
            price["precio_promedio"] = np.nan

        # Scores si existen
        for sc in ["score_bot", "score_top"]:
            if sc in td.columns:
                price[sc] = interpolar_serie(price.index, td, sc, 50.0)

        price["btc_total"]       = price["btc_balance"] + price["btc_en_posiciones"]
        price["btc_value"]       = price["btc_total"] * price["close"]
        price["portfolio_value"] = price["usdt_balance"] + price["btc_value"]
        price["drawdown"]        = calcular_drawdown(price["portfolio_value"].values)

        # Buy & Hold line
        price_bh_start            = price["close"].iloc[0]
        bh_btc                    = SALDO_USDT_INICIAL / price_bh_start
        price["bh_value"]         = bh_btc * price["close"]

        # Hourly returns para Sharpe
        price["port_ret"] = price["portfolio_value"].pct_change().fillna(0)
        return price

    def _filtrar_trades(self):
        fi, ff = self._rango()
        m = (self.trade_data["datetime"] >= fi) & (self.trade_data["datetime"] <= ff)
        return self.trade_data[m].copy()

    # ── Análisis en consola ──────────────────────────────────────

    def analisis_consola(self):
        if self.trade_data is None: return
        s   = self.results_data.get("summary", {}) if self.results_data else {}
        par = s.get("parametros", {})
        pc  = self._price_cont()
        trades = self._filtrar_trades()
        c_ok, v_ok, c_ign, v_ign = separar_trades(trades)

        sep = "═" * 62
        print(f"\n{sep}")
        print("  PARÁMETROS DE LA ESTRATEGIA")
        print(sep)
        print(f"  Estrategia           : {s.get('estrategia','?')}")
        print(f"  Umbral Bot/Top       : {par.get('thr_bot','?')} / {par.get('thr_top','?')}")
        print(f"  Cooldown (velas)     : {par.get('cooldown_velas','?')}h")
        print(f"  % USDT por señal     : {par.get('pct_usdt_por_senal','?')}%")
        print(f"  % BTC por señal      : {par.get('pct_btc_por_senal','?')}%")
        print(f"  Máx posiciones       : {par.get('max_posiciones','?')}")
        print(f"  Comisión             : {par.get('commission_pct','?')}%")

        print(f"\n{sep}")
        print("  RESUMEN PORTFOLIO")
        print(sep)
        pnl     = s.get("pnl_pct", 0)
        pfin    = s.get("portfolio_value_final", 0)
        bh_pnl  = s.get("buy_hold_pnl_pct", 0)
        alpha   = s.get("alpha_vs_bh", pnl - bh_pnl)
        sharpe  = calcular_sharpe(pc["port_ret"].values)
        calmar  = calcular_calmar(pc["portfolio_value"].values)
        dd_max  = pc["drawdown"].min()
        sign    = "+" if pnl >= 0 else ""
        print(f"  Capital inicial      : ${SALDO_USDT_INICIAL:>10,.2f}")
        print(f"  Portfolio final      : ${pfin:>10,.2f}   ({sign}{pnl:.2f}%)")
        print(f"  Buy & Hold           :             ({bh_pnl:+.2f}%)")
        print(f"  Alpha vs B&H         :             ({alpha:+.2f}%)")
        print(f"  Max Drawdown         : {dd_max:.2f}%")
        print(f"  Sharpe Ratio (hvol)  : {sharpe:.3f}")
        print(f"  Calmar Ratio         : {calmar:.3f}")

        # Win rate & profit factor
        ganancias = [t for t in (v_ok["ganancia_usdt"].dropna().tolist()
                                  if "ganancia_usdt" in v_ok.columns else []) if t is not None]
        if ganancias:
            wins  = [g for g in ganancias if g > 0]
            loses = [g for g in ganancias if g < 0]
            wr    = len(wins)/len(ganancias)*100
            pf    = sum(wins)/abs(sum(loses)) if loses else float('inf')
            print(f"  Win Rate             : {wr:.1f}%  ({len(wins)}W / {len(loses)}L)")
            print(f"  Profit Factor        : {pf:.2f}")
            print(f"  Ganancia total       : ${sum(ganancias):>10,.2f}")
            print(f"  Avg ganancia/pérdida : ${np.mean(wins):,.2f} / ${np.mean(loses) if loses else 0:,.2f}")

        print(f"\n{sep}")
        print("  TRADES")
        print(sep)
        fi_dt = pd.to_datetime(s.get("fecha_inicio", ""))
        ff_dt = pd.to_datetime(s.get("fecha_fin", ""))
        dur   = (ff_dt - fi_dt).days if fi_dt and ff_dt else 0
        tot   = len(c_ok) + len(v_ok)
        print(f"  Período              : {str(fi_dt)[:10]}  →  {str(ff_dt)[:10]}  ({dur}d)")
        print(f"  Ejecutados           : {tot}  (compras: {len(c_ok)}  ventas: {len(v_ok)})")
        print(f"  Ignorados            : {len(c_ign)+len(v_ign)}")
        if dur > 0 and tot > 0:
            print(f"  Frecuencia           : {tot/(dur/30):.1f} trades/mes  |  {dur/tot:.1f}d entre trades")
        print(sep)

    # ── Gráfico principal ────────────────────────────────────────

    def crear_grafico(self, output=OUTPUT_PNG):
        if self.price_data is None or self.trade_data is None:
            print("Faltan datos."); return

        C  = self.C
        pc = self._price_cont()
        tr = self._filtrar_trades()
        c_ok, v_ok, c_ign, v_ign = separar_trades(tr)
        s  = self.results_data.get("summary", {}) if self.results_data else {}
        par = s.get("parametros", {})

        has_scores = "score_bot" in pc.columns and pc["score_bot"].notna().any()

        # ── Layout ────────────────────────────────────────────────
        n_panels     = 7 if has_scores else 6
        height_ratios = [3, 1.1, 1.0, 0.8, 1.2, 0.85]
        if has_scores:
            height_ratios = [3, 1.2, 1.1, 1.0, 0.8, 1.2, 0.85]

        fig = plt.figure(figsize=(22, 26 if has_scores else 23), facecolor=C["bg"])
        gs  = gridspec.GridSpec(n_panels, 1, figure=fig,
                                height_ratios=height_ratios,
                                hspace=0.06, top=0.965, bottom=0.03,
                                left=0.07, right=0.97)
        axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

        # Estilo común
        for ax in axes:
            ax.set_facecolor(C["panel_bg"])
            ax.tick_params(colors=C["text"], labelsize=8)
            ax.grid(True, color=C["grid"], lw=0.5, alpha=0.7)
            for sp in ax.spines.values():
                sp.set_color(C["grid"]); sp.set_linewidth(0.7)

        pnl    = s.get("pnl_pct", 0)
        pfin   = s.get("portfolio_value_final", pc["portfolio_value"].iloc[-1])
        fi_s   = str(s.get("fecha_inicio", ""))[:10]
        ff_s   = str(s.get("fecha_fin",    ""))[:10]
        bh_pnl = s.get("buy_hold_pnl_pct", 0)
        sharpe = calcular_sharpe(pc["port_ret"].values)
        dd_max = pc["drawdown"].min()
        fig.suptitle(
            f"Análisis de Estrategia BTC/USDT  ·  {fi_s} → {ff_s}\n"
            f"PnL: {pnl:+.2f}%   B&H: {bh_pnl:+.2f}%   "
            f"Portfolio: ${pfin:,.0f}   "
            f"Compras: {s.get('total_compras','-')}   Ventas: {s.get('total_ventas','-')}   "
            f"Sharpe: {sharpe:.2f}   MaxDD: {dd_max:.1f}%",
            fontsize=11, fontweight="bold", color=C["text"], y=0.996
        )

        ax_idx = 0

        # ══ Panel 1: Precio + Trades ═══════════════════════════════
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(pc.index, pc["close"], color=C["price_line"],
                lw=0.8, alpha=0.85, label="Precio BTC", zorder=2)

        # Ignorados (detrás)
        if len(c_ign):
            ax.scatter(c_ign["datetime"], c_ign["price"], marker="x",
                       c=C["buy_ign"], alpha=0.4, s=20, lw=0.9, zorder=3,
                       label=f"Compras ign. ({len(c_ign)})")
        if len(v_ign):
            ax.scatter(v_ign["datetime"], v_ign["price"], marker="x",
                       c=C["sell_ign"], alpha=0.4, s=20, lw=0.9, zorder=3,
                       label=f"Ventas ign. ({len(v_ign)})")

        # Ejecutados
        if len(c_ok):
            ax.scatter(c_ok["datetime"], c_ok["price"], c=C["buy_ok"],
                       s=20, alpha=0.8, zorder=5, label=f"Compras ({len(c_ok)})")
        if len(v_ok):
            ax.scatter(v_ok["datetime"], v_ok["price"], c=C["sell_ok"],
                       s=20, alpha=0.8, zorder=5, label=f"Ventas ({len(v_ok)})")

        # Precio promedio posiciones
        pp_s = pc["precio_promedio"].replace(0, np.nan)
        if pp_s.notna().any():
            ax.plot(pc.index, pp_s, color=C["pp_line"], ls="--", lw=1.2,
                    alpha=0.8, label="PP Posiciones", zorder=4)

        pp_fin = s.get("precio_promedio_final", 0)
        if pp_fin and pp_fin > 0:
            ax.axhline(pp_fin, color=C["pp_line"], ls=":", alpha=0.4, lw=0.8)
            ax.annotate(f"PP: ${pp_fin:,.0f}", xy=(pc.index[-1], pp_fin),
                        xytext=(-90, 5), textcoords="offset points",
                        fontsize=7.5, color=C["pp_line"])

        # Stats box
        ath = s.get("ath_proyectado_final", pc["close"].max())
        atl = s.get("atl_final", pc["close"].min())
        box_txt = f"ATH: ${ath:,.0f}\nATL: ${atl:,.0f}\nPP: ${pp_fin:,.0f}"
        ax.text(0.01, 0.05, box_txt, transform=ax.transAxes, fontsize=7.5,
                va="bottom", color=C["text"],
                bbox=dict(boxstyle="round,pad=0.4", fc=C["panel_bg"], ec=C["grid"], alpha=0.85))

        ax.set_ylabel("Precio BTC (USD)", color=C["text"], fontsize=9)
        ax.set_title("Precio BTC + Operaciones", color=C["text"], fontsize=10, pad=4)
        ax.legend(loc="upper left", fontsize=7.5, ncol=2,
                  facecolor=C["panel_bg"], labelcolor=C["text"], framealpha=0.9)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
        ax.xaxis.set_visible(False)

        # ══ Panel 2: Score Compuesto (si existe) ════════════════════
        if has_scores:
            ax = axes[ax_idx]; ax_idx += 1
            sb = pc["score_bot"].rolling(4, min_periods=1).mean()
            st = pc["score_top"].rolling(4, min_periods=1).mean()

            ax.fill_between(pc.index, sb, 0, color=C["score_bot"], alpha=0.3,
                            label="Score BOTTOM")
            ax.fill_between(pc.index, -st, 0, color=C["score_top"], alpha=0.3,
                            label="Score TOP (inv.)")
            ax.plot(pc.index,  sb, color=C["score_bot"], lw=0.9)
            ax.plot(pc.index, -st, color=C["score_top"], lw=0.9)
            ax.axhline(0,   color=C["grid"], lw=0.8)
            ax.axhline( 75, color=C["score_bot"], lw=0.7, ls="--", alpha=0.6,
                        label="Umbral BUY 75")
            ax.axhline(-75, color=C["score_top"], lw=0.7, ls="--", alpha=0.6,
                        label="Umbral SELL 75")

            # Señales ejecutadas como líneas verticales
            for _, row in c_ok.iterrows():
                ax.axvline(row["datetime"], color=C["score_bot"], alpha=0.2, lw=0.5)
            for _, row in v_ok.iterrows():
                ax.axvline(row["datetime"], color=C["score_top"], alpha=0.2, lw=0.5)

            ax.set_ylim(-110, 110)
            ax.set_yticks([-100,-75,0,75,100])
            ax.set_yticklabels(["Top 100","Top 75","0","Bot 75","Bot 100"],
                               fontsize=7, color=C["text"])
            ax.set_ylabel("Score 0–100", color=C["text"], fontsize=9)
            ax.set_title("Score Compuesto (DNA + Lyapunov + PE + Delta)",
                         color=C["text"], fontsize=10, pad=4)
            ax.legend(loc="upper left", fontsize=7, ncol=2,
                      facecolor=C["panel_bg"], labelcolor=C["text"], framealpha=0.85)
            ax.xaxis.set_visible(False)

        # ══ Panel 3: Balance USDT ════════════════════════════════════
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(pc.index, pc["usdt_balance"], color=C["usdt_line"], lw=1.4,
                label="Balance USDT")
        ax.fill_between(pc.index, 0, pc["usdt_balance"],
                        color=C["usdt_line"], alpha=0.2)
        ax.axhline(0, color=C["sell_ok"], ls="--", alpha=0.6, lw=0.8, label="$0")
        reserva = s.get("usdt_reserva_aplicada", 0)
        if reserva > 0:
            ax.axhline(reserva, color="#f59e0b", ls=":", alpha=0.7, lw=1,
                       label=f"Reserva (${reserva:,.0f})")
        usdt_fin = s.get("usdt_balance_final", 0)
        ax.annotate(f"${usdt_fin:,.2f}",
                    xy=(pc.index[-1], usdt_fin),
                    xytext=(-80, 8), textcoords="offset points",
                    fontsize=8, color=C["usdt_line"],
                    arrowprops=dict(arrowstyle="-", color=C["usdt_line"], alpha=0.5))
        ax.set_ylabel("Balance USDT", color=C["text"], fontsize=9)
        ax.set_title("Evolución del Balance USDT", color=C["text"], fontsize=10, pad=4)
        ax.legend(loc="upper right", fontsize=7.5,
                  facecolor=C["panel_bg"], labelcolor=C["text"])
        ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
        ax.xaxis.set_visible(False)

        # ══ Panel 4: Balance BTC ═════════════════════════════════════
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(pc.index, pc["btc_en_posiciones"], color="#60a5fa", lw=1.1,
                label="BTC en posiciones")
        ax.plot(pc.index, pc["btc_balance"],       color="#c084fc", lw=1.1,
                label="BTC libre (acum.)")
        ax.plot(pc.index, pc["btc_total"],         color=C["btc_line"], lw=1.8,
                label="BTC total")
        ax.fill_between(pc.index, 0, pc["btc_total"],
                        color=C["btc_line"], alpha=0.15)
        btc_fin = s.get("btc_en_posiciones_final", 0) + s.get("btc_balance_final", 0)
        ax.text(0.01, 0.96,
                f"Pos: {s.get('btc_en_posiciones_final',0):.6f} ₿\n"
                f"Libre: {s.get('btc_balance_final',0):.6f} ₿\n"
                f"Total: {btc_fin:.6f} ₿",
                transform=ax.transAxes, fontsize=7.5, va="top", color=C["text"],
                bbox=dict(boxstyle="round,pad=0.35", fc=C["panel_bg"], ec=C["grid"], alpha=0.85))
        ax.set_ylabel("Balance BTC (₿)", color=C["text"], fontsize=9)
        ax.set_title("Evolución del Balance BTC", color=C["text"], fontsize=10, pad=4)
        ax.legend(loc="upper center", fontsize=7.5,
                  facecolor=C["panel_bg"], labelcolor=C["text"])
        ax.xaxis.set_visible(False)

        # ══ Panel 5: Posiciones abiertas ════════════════════════════
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(pc.index, pc["positions_count"], color="#a78bfa", lw=1.4,
                label="Posiciones abiertas")
        ax.fill_between(pc.index, 0, pc["positions_count"],
                        color="#a78bfa", alpha=0.2)
        ax.axhline(0, color=C["grid"], ls="--", alpha=0.5, lw=0.7)
        pos_max = int(pc["positions_count"].max())
        pos_fin = s.get("positions_count_final", 0)
        ax.text(0.01, 0.96, f"Máx: {pos_max}   Final: {pos_fin:+d}",
                transform=ax.transAxes, fontsize=8, va="top", color=C["text"],
                bbox=dict(boxstyle="round,pad=0.35", fc=C["panel_bg"], ec=C["grid"], alpha=0.85))
        ax.set_ylabel("N° Posiciones", color=C["text"], fontsize=9)
        ax.set_title("Posiciones Abiertas Netas", color=C["text"], fontsize=10, pad=4)
        ax.xaxis.set_visible(False)

        # ══ Panel 6: Portfolio total + Buy & Hold ═══════════════════
        ax = axes[ax_idx]; ax_idx += 1

        # Buy & Hold
        ax.plot(pc.index, pc["bh_value"], color=C["bh_line"], lw=1.2,
                ls="--", alpha=0.75, label=f"Buy & Hold ({bh_pnl:+.1f}%)")

        ax.plot(pc.index, pc["usdt_balance"], color=C["usdt_line"],
                lw=1, alpha=0.7, label="Valor USDT")
        ax.plot(pc.index, pc["btc_value"],    color=C["btc_line"],
                lw=1, alpha=0.7, label="Valor BTC")
        ax.plot(pc.index, pc["portfolio_value"], color=C["port_line"],
                lw=2, label=f"Portfolio ({pnl:+.1f}%)")

        ax.fill_between(pc.index, SALDO_USDT_INICIAL, pc["portfolio_value"],
                        where=pc["portfolio_value"] >= SALDO_USDT_INICIAL,
                        color=C["port_line"], alpha=0.15, label="Ganancia")
        ax.fill_between(pc.index, SALDO_USDT_INICIAL, pc["portfolio_value"],
                        where=pc["portfolio_value"] < SALDO_USDT_INICIAL,
                        color=C["sell_ok"], alpha=0.15, label="Pérdida")
        ax.axhline(SALDO_USDT_INICIAL, color=C["grid"], ls="--", lw=0.8, alpha=0.7,
                   label=f"Capital inicial (${SALDO_USDT_INICIAL:,})")

        pnl_c = C["port_line"] if pnl >= 0 else C["sell_ok"]
        ax.annotate(f"${pfin:,.0f}  ({pnl:+.2f}%)",
                    xy=(pc.index[-1], pfin),
                    xytext=(-140, 14 if pnl >= 0 else -18),
                    textcoords="offset points",
                    fontsize=9, fontweight="bold", color=pnl_c,
                    arrowprops=dict(arrowstyle="-", color=pnl_c, alpha=0.5))

        # Métricas box
        alpha_v = s.get("alpha_vs_bh", pnl - bh_pnl)
        wrate   = ""
        if "ganancia_usdt" in v_ok.columns:
            g = v_ok["ganancia_usdt"].dropna()
            if len(g) > 0:
                wrate = f"\nWin rate: {(g>0).mean()*100:.1f}%"
        ax.text(0.01, 0.06,
                f"Max DD: {dd_max:.1f}%   Sharpe: {sharpe:.2f}   "
                f"Alpha: {alpha_v:+.1f}%{wrate}",
                transform=ax.transAxes, fontsize=8, va="bottom", color=C["text"],
                bbox=dict(boxstyle="round,pad=0.4", fc=C["panel_bg"], ec=C["grid"], alpha=0.85))

        ax.set_ylabel("Valor Portfolio (USD)", color=C["text"], fontsize=9)
        ax.set_title("Valor Total del Portfolio  vs  Buy & Hold",
                     color=C["text"], fontsize=10, pad=4)
        ax.legend(loc="upper left", fontsize=7.5, ncol=2,
                  facecolor=C["panel_bg"], labelcolor=C["text"])
        ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
        ax.xaxis.set_visible(False)

        # ══ Panel 7: Drawdown ════════════════════════════════════════
        ax = axes[ax_idx]; ax_idx += 1
        dd = pc["drawdown"].values
        bh_dd = calcular_drawdown(pc["bh_value"].values)
        ax.fill_between(pc.index, dd, 0, color=C["dd_line"], alpha=0.5,
                        label=f"DD Estrategia  (min={dd.min():.1f}%)")
        ax.plot(pc.index, dd, color=C["dd_line"], lw=0.8)
        ax.fill_between(pc.index, bh_dd, 0, color=C["bh_line"], alpha=0.25,
                        label=f"DD Buy & Hold  (min={bh_dd.min():.1f}%)")
        ax.axhline(0, color=C["grid"], lw=0.7)
        ax.axhline(dd.min(), color=C["dd_line"], lw=0.8, ls=":", alpha=0.7)
        ax.set_ylabel("Drawdown (%)", color=C["text"], fontsize=9)
        ax.set_title("Drawdown Continuo — Estrategia vs Buy & Hold",
                     color=C["text"], fontsize=10, pad=4)
        ax.legend(loc="lower right", fontsize=7.5,
                  facecolor=C["panel_bg"], labelcolor=C["text"])
        ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))

        # ── Eje X compartido ──────────────────────────────────────
        locator   = mdates.AutoDateLocator(minticks=10, maxticks=26)
        formatter = mdates.ConciseDateFormatter(locator)
        for i, ax in enumerate(axes):
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(),
                     color=C["text"], fontsize=8,
                     visible=(i == len(axes)-1))

        plt.savefig(output, dpi=DPI, bbox_inches="tight",
                    facecolor=C["bg"], edgecolor="none")
        print(f"\n✓ Gráfico guardado: {output}")

        # ── Maximizar ventana antes de mostrar (multi-backend) ────────────
        try:
            manager = plt.get_current_fig_manager()
            backend = matplotlib.get_backend().lower()
            if "qt" in backend:
                manager.window.showMaximized()
            elif "tk" in backend:
                try:
                    manager.window.state("zoomed")       # Windows / macOS
                except Exception:
                    pass
                try:
                    manager.window.attributes("-zoomed", True)  # Linux/TkAgg
                except Exception:
                    pass
            elif "wx" in backend:
                manager.frame.Maximize(True)
            elif "gtk" in backend:
                manager.window.maximize()
            else:
                try:
                    manager.full_screen_toggle()         # fallback genérico
                except Exception:
                    pass
        except Exception:
            pass   # si falla, se abre con el tamaño por defecto

        plt.show()
        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Graficador v2 — Estrategias BTC")
    parser.add_argument("--json",  default=RESULTS_JSON,  help="Archivo JSON de resultados")
    parser.add_argument("--db",    default=DB_PATH,        help="Base de datos SQLite")
    parser.add_argument("--dark",  action="store_true",    default=DARK_MODE, help="Tema oscuro")
    parser.add_argument("--light", action="store_true",    help="Tema claro")
    parser.add_argument("--out",   default=OUTPUT_PNG,    help="Archivo PNG de salida")
    args = parser.parse_args()

    dark = args.dark and not args.light

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        GRAFICADOR v2 — ESTRATEGIAS BTC/USDT             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    g = GraficadorV2(dark_mode=dark)

    if not g.cargar_precios(args.db):
        return
    if not g.cargar_resultados(args.json):
        return

    g.analisis_consola()
    print("\nGenerando gráfico...")
    g.crear_grafico(args.out)


if __name__ == "__main__":
    try:
        import matplotlib, pandas, numpy
    except ImportError as e:
        print(f"Dependencia faltante: {e}\npip install matplotlib pandas numpy")
        exit(1)
    main()