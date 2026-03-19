"""
Graficador y Analizador de Resultados de Estrategias BTC
═══════════════════════════════════════════════════════════
Ejecutar: python Graficador.py

Compatibilidad: soporta JSONs antiguos (sin campo 'ignorado') y nuevos.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
import os
import sqlite3

from config_binance_grid import DB_PATH, FECHA_INICIO, FECHA_FIN, SALDO_USDT_INICIAL

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_drawdown_maximo(series: np.ndarray) -> float:
    """Máximo drawdown porcentual respecto al pico anterior."""
    pico = np.maximum.accumulate(series)
    dd   = (series - pico) / np.where(pico == 0, 1, pico)
    return float(dd.min() * 100)


def _es_ignorado(row) -> bool:
    """
    Retorna True si el trade fue ignorado.
    Compatible con JSONs antiguos que no tienen el campo 'ignorado'.
    """
    val = row.get("ignorado", False)
    if val is None:
        return False
    return bool(val)


def _separar_trades(df: pd.DataFrame):
    """
    Separa el DataFrame en cuatro grupos:
      compras_ok, ventas_ok  → trades ejecutados (ploteados con color)
      compras_ign, ventas_ign → trades ignorados  (ploteados con X gris)

    Backward-compatible: si no existe columna 'ignorado' todos son ejecutados.
    """
    if "ignorado" in df.columns:
        ejecutados  = df[df["ignorado"].fillna(False) == False]
        ignorados   = df[df["ignorado"].fillna(False) == True]
    else:
        ejecutados  = df
        ignorados   = df.iloc[0:0]   # vacío

    compras_ok  = ejecutados[ejecutados["type"] == "BUY"]
    ventas_ok   = ejecutados[ejecutados["type"] == "SELL"]
    compras_ign = ignorados[ignorados["type"] == "BUY"]
    ventas_ign  = ignorados[ignorados["type"] == "SELL"]

    return compras_ok, ventas_ok, compras_ign, ventas_ign


def _interpolar_serie(price_index: pd.DatetimeIndex,
                      trade_df: pd.DataFrame,
                      col: str,
                      fill_value=0.0) -> pd.Series:
    """
    Interpola una columna del trade_df sobre el índice de precios (ffill).
    Rellena NaN iniciales con fill_value.
    """
    if col not in trade_df.columns:
        return pd.Series(fill_value, index=price_index)
    s = trade_df[col].copy()
    s = s[~s.index.duplicated(keep="last")]
    s = s.reindex(price_index, method="ffill")
    s = s.fillna(fill_value)
    return s


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASE PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Graficador:

    def __init__(self):
        self.price_data  = None   # DataFrame crudo de velas
        self.trade_data  = None   # DataFrame de TODOS los trades (incl. ignorados)
        self.results_data = None  # dict completo del JSON

    # ── Carga ─────────────────────────────────────────────────────────────────

    def cargar_datos_precios(self) -> bool:
        try:
            conn  = sqlite3.connect(DB_PATH)
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM   {DB_TABLE}
                ORDER  BY timestamp ASC
            """
            df = pd.read_sql(query, conn)
            conn.close()

            df["datetime"]  = pd.to_datetime(df["timestamp"], unit="ms")
            self.price_data = df
            print(f"✓ Precios cargados : {len(df):,} velas")
            return True
        except Exception as e:
            print(f"✗ Error cargando precios: {e}")
            return False

    def cargar_resultados_estrategia(self, archivo=None) -> bool:
        if archivo is None:
            try:
                from config_binance_grid import RESULTS_JSON
                if os.path.exists(RESULTS_JSON):
                    archivo = RESULTS_JSON
            except ImportError:
                pass

        if archivo is None:
            candidatos = [f for f in os.listdir(".")
                          if f.endswith(".json") and "result" in f.lower()]
            if not candidatos:
                print("✗ No se encontró archivo de resultados")
                return False
            archivo = candidatos[0]

        try:
            with open(archivo, "r") as f:
                self.results_data = json.load(f)

            trades = self.results_data.get("trade_history", [])
            self.trade_data = pd.DataFrame(trades)
            self.trade_data["datetime"] = pd.to_datetime(self.trade_data["datetime"])

            print(f"✓ Resultados cargados : {len(self.trade_data)} registros desde '{archivo}'")
            return True
        except Exception as e:
            print(f"✗ Error cargando resultados: {e}")
            return False

    # ── Preparación de datos ──────────────────────────────────────────────────

    def _preparar_series_continuas(self) -> pd.DataFrame:
        """
        Interpola los balances del trade_data sobre el índice de velas
        para obtener series continuas (sin gaps) usadas en el portfolio.
        Solo usa trades ejecutados para los balances.
        """
        # Filtrar por rango de fechas
        fi = pd.to_datetime(FECHA_INICIO) if FECHA_INICIO else self.price_data["datetime"].min()
        ff = pd.to_datetime(FECHA_FIN)    if FECHA_FIN    else self.price_data["datetime"].max()

        price = self.price_data[
            (self.price_data["datetime"] >= fi) &
            (self.price_data["datetime"] <= ff)
        ].copy().set_index("datetime")

        # Solo trades ejecutados para los balances continuos
        if "ignorado" in self.trade_data.columns:
            td_ejec = self.trade_data[self.trade_data["ignorado"].fillna(False) == False].copy()
        else:
            td_ejec = self.trade_data.copy()

        td_ejec = td_ejec.set_index("datetime")

        price["usdt_balance"]      = _interpolar_serie(price.index, td_ejec, "usdt_balance",      SALDO_USDT_INICIAL)
        price["btc_balance"]       = _interpolar_serie(price.index, td_ejec, "btc_balance",        0.0)
        price["btc_en_posiciones"] = _interpolar_serie(price.index, td_ejec, "btc_en_posiciones",  0.0)
        price["positions_count"]   = _interpolar_serie(price.index, td_ejec, "positions_count",    0.0)

        # PP solo de trades ejecutados (excluyendo ignorados)
        if "precio_promedio_posiciones" in td_ejec.columns:
            pp = td_ejec["precio_promedio_posiciones"].copy()
            pp = pp[~pp.index.duplicated(keep="last")]
            pp = pp.replace(0, np.nan)
            price["precio_promedio"] = pp.reindex(price.index, method="ffill")
        else:
            price["precio_promedio"] = np.nan

        price["btc_total"]      = price["btc_balance"] + price["btc_en_posiciones"]
        price["btc_value"]      = price["btc_total"] * price["close"]
        price["portfolio_value"] = price["usdt_balance"] + price["btc_value"]

        return price

    def _filtrar_trades_por_rango(self) -> pd.DataFrame:
        """Trade_data filtrado por rango de fechas del config_binance_grid."""
        fi = pd.to_datetime(FECHA_INICIO) if FECHA_INICIO else self.trade_data["datetime"].min()
        ff = pd.to_datetime(FECHA_FIN)    if FECHA_FIN    else self.trade_data["datetime"].max()
        mask = (self.trade_data["datetime"] >= fi) & (self.trade_data["datetime"] <= ff)
        return self.trade_data[mask].copy()

    # ── Gráfico principal ─────────────────────────────────────────────────────

    def crear_grafico_principal(self):
        if self.price_data is None or self.trade_data is None:
            print("Faltan datos. Cargá precios y resultados primero.")
            return

        price_cont = self._preparar_series_continuas()
        trades     = self._filtrar_trades_por_rango()

        compras_ok, ventas_ok, compras_ign, ventas_ign = _separar_trades(trades)

        s   = self.results_data.get("summary", {}) if self.results_data else {}
        par = s.get("parametros", {})

        fig, axes = plt.subplots(
            5, 1, figsize=(20, 22), sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1, 0.8, 1.2]}
        )
        fig.patch.set_facecolor("#f8f9fa")

        pnl   = s.get("pnl_pct", 0)
        pfin  = s.get("portfolio_value_final", 0)
        fi_s  = str(s.get("fecha_inicio", FECHA_INICIO or ""))[:10]
        ff_s  = str(s.get("fecha_fin",    FECHA_FIN    or ""))[:10]
        fig.suptitle(
            f"Análisis de Estrategia BTC/USDT  ·  {fi_s} → {ff_s}\n"
            f"PnL: {pnl:+.2f}%   Portfolio final: ${pfin:,.2f}   "
            f"Compras: {s.get('total_compras','-')}   Ventas: {s.get('total_ventas','-')}   "
            f"Ignorados: {s.get('total_ignorados', '-')}",
            fontsize=12, fontweight="bold", y=0.995
        )

        # ── Panel 1: Precio BTC + Trades ─────────────────────────────────────
        ax1 = axes[0]
        ax1.plot(price_cont.index, price_cont["close"],
                 color="black", alpha=0.65, linewidth=0.8, label="Precio BTC", zorder=2)

        # Ignorados (X gris semitransparente — primero para que queden detrás)
        if len(compras_ign) > 0:
            ax1.scatter(compras_ign["datetime"], compras_ign["price"],
                        marker="x", color="gray", alpha=0.45, s=22, linewidths=1,
                        label=f"Compras ignoradas ({len(compras_ign)})", zorder=3)
        if len(ventas_ign) > 0:
            ax1.scatter(ventas_ign["datetime"], ventas_ign["price"],
                        marker="x", color="salmon", alpha=0.45, s=22, linewidths=1,
                        label=f"Ventas ignoradas ({len(ventas_ign)})", zorder=3)

        # Ejecutados
        if len(compras_ok) > 0:
            ax1.scatter(compras_ok["datetime"], compras_ok["price"],
                        color="green", alpha=0.75, s=18,
                        label=f"Compras ({len(compras_ok)})", zorder=5)
        if len(ventas_ok) > 0:
            ax1.scatter(ventas_ok["datetime"], ventas_ok["price"],
                        color="red", alpha=0.75, s=18,
                        label=f"Ventas ({len(ventas_ok)})", zorder=5)

        # Precio promedio (solo de ejecutados)
        if "precio_promedio" in price_cont.columns:
            pp_serie = price_cont["precio_promedio"].replace(0, np.nan)
            ax1.plot(price_cont.index, pp_serie,
                     color="royalblue", linestyle="--", linewidth=1.4,
                     label="Precio Promedio Posiciones", zorder=4)

        # Anotación precio promedio final
        pp_fin = s.get("precio_promedio_final", 0)
        if pp_fin and pp_fin > 0:
            ax1.axhline(pp_fin, color="royalblue", linestyle=":", alpha=0.5, linewidth=1)
            ax1.annotate(f"PP final: ${pp_fin:,.0f}",
                         xy=(price_cont.index[-1], pp_fin),
                         xytext=(-120, 6), textcoords="offset points",
                         fontsize=8, color="royalblue")

        # Stats box panel 1
        ath_f = s.get("ath_final", 0)
        atl_f = s.get("atl_final", 0)
        stats_txt = (
            f"ATH: ${ath_f:,.0f}\n"
            f"ATL: ${atl_f:,.0f}\n"
            f"PP pos: ${pp_fin:,.0f}"
        )
        ax1.text(0.01, 0.04, stats_txt, transform=ax1.transAxes,
                 fontsize=8, verticalalignment="bottom",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

        ax1.set_ylabel("Precio BTC (USD)")
        ax1.set_title("Precio BTC + Operaciones de Trading")
        ax1.legend(loc="lower left", fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.25)
        ax1.set_yscale("log")

        # ── Panel 2: Balance USDT continuo ────────────────────────────────────
        ax2 = axes[1]
        ax2.plot(price_cont.index, price_cont["usdt_balance"],
                 color="royalblue", linewidth=1.5, label="Balance USDT")
        ax2.fill_between(price_cont.index, 0, price_cont["usdt_balance"],
                         alpha=0.25, color="royalblue")
        ax2.axhline(0, color="red", linestyle="--", alpha=0.6, linewidth=0.9, label="$0")

        usdt_fin = s.get("usdt_balance_final", 0)
        reserva  = s.get("usdt_reserva_aplicada", 0)
        if reserva > 0:
            ax2.axhline(reserva, color="orange", linestyle=":", alpha=0.7,
                        linewidth=1, label=f"Reserva (${reserva:,.0f})")
        ax2.annotate(f"Final: ${usdt_fin:,.2f}",
                     xy=(price_cont.index[-1], usdt_fin),
                     xytext=(-100, 8), textcoords="offset points",
                     fontsize=8, color="royalblue",
                     arrowprops=dict(arrowstyle="-", color="royalblue", alpha=0.5))

        ax2.set_ylabel("Balance USDT")
        ax2.set_title("Evolución del Balance USDT")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.25)

        # ── Panel 3: Balance BTC continuo ─────────────────────────────────────
        ax3 = axes[2]
        ax3.plot(price_cont.index, price_cont["btc_en_posiciones"],
                 color="steelblue", linewidth=1.3, label="BTC en posiciones")
        ax3.plot(price_cont.index, price_cont["btc_balance"],
                 color="purple", linewidth=1.3, label="BTC libre (acum.)")
        ax3.plot(price_cont.index, price_cont["btc_total"],
                 color="darkorange", linewidth=2, label="BTC total")
        ax3.fill_between(price_cont.index, 0, price_cont["btc_total"],
                         alpha=0.18, color="darkorange")

        btc_pos = s.get("btc_en_posiciones_final", 0)
        btc_lib = s.get("btc_balance_final", 0)
        btc_tot = btc_pos + btc_lib
        stats_btc = (f"Posiciones: {btc_pos:.6f} ₿\n"
                     f"Libre acum: {btc_lib:.6f} ₿\n"
                     f"Total:      {btc_tot:.6f} ₿")
        ax3.text(0.01, 0.97, stats_btc, transform=ax3.transAxes,
                 fontsize=8, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

        ax3.set_ylabel("Balance BTC (₿)")
        ax3.set_title("Evolución del Balance BTC")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, alpha=0.25)

        # ── Panel 4: Posiciones abiertas ──────────────────────────────────────
        ax4 = axes[3]
        ax4.plot(price_cont.index, price_cont["positions_count"],
                 color="darkorchid", linewidth=1.5, label="Posiciones abiertas")
        ax4.fill_between(price_cont.index, 0, price_cont["positions_count"],
                         alpha=0.2, color="darkorchid")
        ax4.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        pos_max = int(price_cont["positions_count"].max())
        pos_fin = s.get("positions_count_final", 0)
        ax4.text(0.01, 0.97, f"Máx: {pos_max}   Final: {pos_fin:+d}",
                 transform=ax4.transAxes, fontsize=8, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

        ax4.set_ylabel("N° Posiciones")
        ax4.set_title("Posiciones Abiertas Netas")
        ax4.legend(loc="upper left", fontsize=8)
        ax4.grid(True, alpha=0.25)

        # ── Panel 5: Valor total del portfolio ────────────────────────────────
        ax5 = axes[4]
        ax5.plot(price_cont.index, price_cont["usdt_balance"],
                 color="dodgerblue", linewidth=1.2, label="Valor USDT", alpha=0.8)
        ax5.plot(price_cont.index, price_cont["btc_value"],
                 color="darkorange", linewidth=1.2, label="Valor BTC", alpha=0.8)
        ax5.plot(price_cont.index, price_cont["portfolio_value"],
                 color="green", linewidth=2, label="Portfolio total")

        ax5.fill_between(price_cont.index, SALDO_USDT_INICIAL, price_cont["portfolio_value"],
                         where=(price_cont["portfolio_value"] >= SALDO_USDT_INICIAL),
                         alpha=0.18, color="green", label="Ganancia")
        ax5.fill_between(price_cont.index, SALDO_USDT_INICIAL, price_cont["portfolio_value"],
                         where=(price_cont["portfolio_value"] < SALDO_USDT_INICIAL),
                         alpha=0.18, color="red", label="Pérdida")
        ax5.axhline(SALDO_USDT_INICIAL, color="gray", linestyle="--", alpha=0.6, linewidth=1,
                    label=f"Capital inicial (${SALDO_USDT_INICIAL:,})")

        # Anotación valor final y PnL
        port_fin = s.get("portfolio_value_final", price_cont["portfolio_value"].iloc[-1])
        pnl_c    = "green" if pnl >= 0 else "red"
        ax5.annotate(f"${port_fin:,.2f}  ({pnl:+.2f}%)",
                     xy=(price_cont.index[-1], port_fin),
                     xytext=(-130, 12 if pnl >= 0 else -18), textcoords="offset points",
                     fontsize=9, fontweight="bold", color=pnl_c,
                     arrowprops=dict(arrowstyle="-", color=pnl_c, alpha=0.5))

        # Drawdown
        dd = calcular_drawdown_maximo(price_cont["portfolio_value"].values)
        ax5.text(0.01, 0.04,
                 f"Max Drawdown: {dd:.1f}%",
                 transform=ax5.transAxes, fontsize=8, verticalalignment="bottom",
                 color="darkred",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

        ax5.set_ylabel("Valor Portfolio (USD)")
        ax5.set_xlabel("Fecha")
        ax5.set_title("Valor Total del Portfolio")
        ax5.legend(loc="upper left", fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.25)

        # ── Formato eje X compartido ──────────────────────────────────────────
        locator   = mdates.AutoDateLocator(minticks=10, maxticks=28)
        formatter = mdates.DateFormatter("%d/%m/%y")
        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
            ax.set_facecolor("#fdfdfd")

        plt.tight_layout(rect=[0, 0, 1, 0.997])

        # Maximizar ventana
        try:
            plt.get_current_fig_manager().window.state("zoomed")
        except Exception:
            try:
                plt.get_current_fig_manager().window.showMaximized()
            except Exception:
                pass

        nombre = "analisis_estrategia_completo.png"
        plt.savefig(nombre, dpi=150, bbox_inches="tight")
        print(f"\n✓ Gráfico guardado: {nombre}")
        plt.show()

    # ── Análisis en consola ───────────────────────────────────────────────────

    def crear_analisis_detallado(self):
        if self.trade_data is None:
            return

        s   = self.results_data.get("summary", {}) if self.results_data else {}
        par = s.get("parametros", {})

        trades    = self._filtrar_trades_por_rango()
        c_ok, v_ok, c_ign, v_ign = _separar_trades(trades)

        sep = "═" * 62

        # ── config_binance_griduración ─────────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  PARÁMETROS DE LA ESTRATEGIA")
        print(sep)
        if par:
            print(f"  RSI_LENGTH / N       : {par.get('rsi_length','?')} / {par.get('N','?')}")
            print(f"  ATH_CAIDA_MAXIMA     : {par.get('ath_caida_maxima','?')}%   FACTOR_CAIDA  : {par.get('factor_caida','?')}")
            print(f"  ATL_SUBIDA_MAXIMA    : {par.get('atl_subida_maxima','?')}%  FACTOR_SUBIDA : {par.get('factor_subida','?')}")
            gc = par.get('guardia_compra')
            gv = par.get('guardia_venta')
            if gc is not None:
                print(f"  Guardia compra       : {'✓ activa' if gc else '✗ desactivada'}")
                print(f"  Guardia venta        : {'✓ activa' if gv else '✗ desactivada'}")
            print(f"  USDT reserva         : {par.get('usdt_reserva_pct','?')}%")
            print(f"  Comisión             : {par.get('commission_pct','?')}%")

        # ── Resumen portfolio ─────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  RESUMEN PORTFOLIO")
        print(sep)
        pnl  = s.get("pnl_pct", 0)
        pfin = s.get("portfolio_value_final", 0)
        sign = "+" if pnl >= 0 else ""
        print(f"  Capital inicial      : ${s.get('saldo_inicial_usdt', SALDO_USDT_INICIAL):>10,.2f}")
        print(f"  Portfolio final      : ${pfin:>10,.2f}   ({sign}{pnl:.2f}%)")
        print(f"  USDT libre           : ${s.get('usdt_balance_final',0):>10,.4f}")
        print(f"  BTC en posiciones    :  {s.get('btc_en_posiciones_final',0):>.8f} ₿")
        print(f"  BTC libre (acum.)    :  {s.get('btc_balance_final',0):>.8f} ₿")
        pp = s.get("precio_promedio_final", 0)
        if pp and pp > 0:
            print(f"  Precio prom. pos.    : ${pp:>10,.2f}")
        print(f"  ATH registrado       : ${s.get('ath_final',0):>10,.2f}")
        print(f"  ATL registrado       : ${s.get('atl_final',0):>10,.2f}")
        print(f"  Posiciones finales   : {s.get('positions_count_final',0):+d}")

        # Drawdown
        try:
            trades_ejec = trades[trades.get("ignorado", pd.Series(False, index=trades.index)).fillna(False) == False] \
                          if "ignorado" in trades.columns else trades
        except Exception:
            trades_ejec = trades
        if self.price_data is not None and len(trades) > 0:
            pc = self._preparar_series_continuas()
            dd = calcular_drawdown_maximo(pc["portfolio_value"].values)
            pv_max = pc["portfolio_value"].max()
            print(f"  Valor máximo portf.  : ${pv_max:>10,.2f}")
            print(f"  Max Drawdown         : {dd:.2f}%")

        # ── Trades ejecutados ─────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  TRADES EJECUTADOS")
        print(sep)
        print(f"  Total ejecutados     : {len(c_ok) + len(v_ok)}  "
              f"(compras: {len(c_ok)}  |  ventas: {len(v_ok)})")

        if len(c_ok) > 0:
            print(f"\n  Compras:")
            print(f"    Precio promedio    : ${c_ok['price'].mean():,.2f}")
            print(f"    Precio mínimo      : ${c_ok['price'].min():,.2f}")
            print(f"    Precio máximo      : ${c_ok['price'].max():,.2f}")
            if "usdt_spent" in c_ok.columns:
                usdt_tot = c_ok["usdt_spent"].dropna().sum()
                usdt_avg = c_ok["usdt_spent"].dropna().mean()
                print(f"    USDT total gastado : ${usdt_tot:,.4f}")
                print(f"    USDT promedio/trade: ${usdt_avg:,.4f}")
            if "pct_capital_usado" in c_ok.columns:
                pct_avg = c_ok["pct_capital_usado"].dropna().mean()
                pct_max = c_ok["pct_capital_usado"].dropna().max()
                print(f"    % capital prom.    : {pct_avg:.3f}%")
                print(f"    % capital máx.     : {pct_max:.3f}%")

        if len(v_ok) > 0:
            print(f"\n  Ventas:")
            print(f"    Precio promedio    : ${v_ok['price'].mean():,.2f}")
            print(f"    Precio mínimo      : ${v_ok['price'].min():,.2f}")
            print(f"    Precio máximo      : ${v_ok['price'].max():,.2f}")
            if "ganancia_usdt" in v_ok.columns:
                gan = v_ok["ganancia_usdt"].dropna()
                print(f"    Ganancia total     : ${gan.sum():,.4f}")
                print(f"    Ganancia promedio  : ${gan.mean():,.4f}")
                neg = (gan < 0).sum()
                if neg > 0:
                    print(f"    ⚠ Ventas a pérdida : {neg}")
            if "pct_capital_usado" in v_ok.columns:
                pct_avg = v_ok["pct_capital_usado"].dropna().mean()
                print(f"    % BTC prom. vendido: {pct_avg:.3f}%")

        # ── Ignorados ─────────────────────────────────────────────────────────
        total_ign = len(c_ign) + len(v_ign)
        if total_ign > 0:
            print(f"\n{sep}")
            print("  SEÑALES IGNORADAS")
            print(sep)
            print(f"  Total ignoradas      : {total_ign}  "
                  f"(compras: {len(c_ign)}  |  ventas: {len(v_ign)})")
            motivos = s.get("ignorados_por_motivo", {})
            if motivos:
                print(f"  Por motivo:")
                for motivo, cnt in sorted(motivos.items(), key=lambda x: -x[1]):
                    print(f"    · {motivo:<35}: {cnt}")

        # ── Temporal ──────────────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  ANÁLISIS TEMPORAL")
        print(sep)
        fi_dt = pd.to_datetime(s.get("fecha_inicio", ""))
        ff_dt = pd.to_datetime(s.get("fecha_fin", ""))
        duracion = (ff_dt - fi_dt).days if fi_dt and ff_dt else 0
        tot_trades = len(c_ok) + len(v_ok)
        print(f"  Período              : {str(fi_dt)[:10]}  →  {str(ff_dt)[:10]}")
        print(f"  Duración             : {duracion} días")
        if duracion > 0 and tot_trades > 0:
            print(f"  Trades/mes prom.     : {tot_trades / (duracion / 30):.1f}")
            print(f"  Días entre trades    : {duracion / tot_trades:.1f}")
        print(sep)

    def analizar_periodos_criticos(self):
        if self.trade_data is None:
            return

        trades = self._filtrar_trades_por_rango()
        if "ignorado" in trades.columns:
            ejec = trades[trades["ignorado"].fillna(False) == False]
        else:
            ejec = trades

        criticos = ejec[ejec["usdt_balance"] <= 0]
        if len(criticos) == 0:
            print("\n✓ Sin períodos críticos — USDT siempre por encima de $0")
            return

        print(f"\n{'═'*62}")
        print(f"  ⚠ PERÍODOS CRÍTICOS (USDT ≤ $0): {len(criticos)} trades")
        print(f"{'═'*62}")
        for _, row in criticos.head(10).iterrows():
            print(f"  {row['datetime'].strftime('%Y-%m-%d %H:%M')}  "
                  f"USDT: ${row['usdt_balance']:.2f}  "
                  f"Posiciones: {row['positions_count']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         GRAFICADOR — ESTRATEGIAS BTC/USDT                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    g = Graficador()

    if not g.cargar_datos_precios():
        print("No se pudieron cargar los datos de precios.")
        return

    if not g.cargar_resultados_estrategia():
        print("No se pudieron cargar los resultados de la estrategia.")
        return

    g.crear_analisis_detallado()
    g.analizar_periodos_criticos()

    print("\nGenerando gráfico...")
    g.crear_grafico_principal()


if __name__ == "__main__":
    try:
        import matplotlib
        import pandas
        import numpy
    except ImportError as e:
        print(f"Dependencia faltante: {e}")
        print("Ejecutar: pip install matplotlib pandas numpy")
        exit(1)

    main()