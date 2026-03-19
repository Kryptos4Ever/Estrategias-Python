"""
Graficador y Analizador de Resultados de Estrategias BTC
Ejecutar: python Graficador.py
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import os
from data_manager import DataManager
from config import DB_PATH, FECHA_INICIO, FECHA_FIN

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

def calcular_drawdown_maximo(series):
    max_anteriores = np.maximum.accumulate(series)
    drawdowns = (series - max_anteriores) / max_anteriores
    return drawdowns.min() * 100

class Graficador:
    def __init__(self):
        self.data_manager = DataManager()
        self.price_data = None
        self.trade_data = None
        self.results_data = None

    def cargar_datos_precios(self):
        try:
            import sqlite3
            import pandas as pd
            from config import DB_PATH

            conn = sqlite3.connect(DB_PATH)
            query = f"""
            SELECT timestamp, open, high, low, close, volume 
            FROM {DB_TABLE} 
            ORDER BY timestamp ASC
            """
            df = pd.read_sql(query, conn)
            conn.close()

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            self.price_data = df
            print(f"Datos de precios cargados: {len(df)} registros")

            # --- Cargar resultados de la estrategia (trades) ---
            self.trade_data = self.data_manager.cargar_resultados_df()  # Asegúrate de tener este método

            # --- Interpolar balances de los trades sobre todas las velas ---
            price_df = self.price_data.copy()
            trade_df = self.trade_data.copy()

            price_df = price_df.set_index('datetime')
            trade_df = trade_df.set_index('datetime')
            trade_df = trade_df[~trade_df.index.duplicated(keep='last')]

            for col in ['usdt_balance', 'btc_balance', 'btc_en_posiciones']:
                if col in trade_df.columns:
                    price_df[col] = trade_df[col].reindex(price_df.index, method='ffill')
                else:
                    price_df[col] = 0.0

            # --- AQUÍ VA EL BLOQUE PARA RELLENAR NaN INICIALES ---
            from config import SALDO_USDT_INICIAL
            price_df['usdt_balance'] = price_df['usdt_balance'].fillna(SALDO_USDT_INICIAL)
            price_df['btc_balance'] = price_df['btc_balance'].fillna(0.0)
            price_df['btc_en_posiciones'] = price_df['btc_en_posiciones'].fillna(0.0)

            price_df['btc_total'] = price_df['btc_balance'] + price_df['btc_en_posiciones']
            price_df['btc_value'] = price_df['btc_total'] * price_df['close']
            price_df['portfolio_value'] = price_df['usdt_balance'] + price_df['btc_value']

            self.price_df = price_df

            return True
        except Exception as e:
            print(f"Error al cargar datos de precios: {e}")
            return False

    def cargar_resultados_estrategia(self, archivo=None):
        if archivo is None:
            from config import RESULTS_JSON
            if os.path.exists(RESULTS_JSON):
                archivo = RESULTS_JSON
                print(f"Cargando resultados: {archivo}")
            else:
                archivos_disponibles = [f for f in os.listdir('.') if f.endswith('_results.json')]
                if not archivos_disponibles:
                    print("No se encontraron archivos de resultados")
                    print("Ejecuta primero una estrategia: python Estrategia_1.py")
                    return False
                archivo = archivos_disponibles[0]
                print(f"Cargando automáticamente: {archivo}")

        try:
            with open(archivo, 'r') as f:
                self.results_data = json.load(f)

            self.trade_data = pd.DataFrame(self.results_data['trade_history'])
            self.trade_data['datetime'] = pd.to_datetime(self.trade_data['datetime'])

            print(f"Resultados cargados: {len(self.trade_data)} trades desde {archivo}")
            return True

        except Exception as e:
            print(f"Error cargando resultados: {e}")
            return False

    def _filtrar_por_rango_fechas(self):
        # Determinar fechas de inicio y fin reales
        price_df = self.price_data
        trade_df = self.trade_data

        # Si FECHA_INICIO/FIN es None, usar extremos de los datos
        fecha_inicio = pd.to_datetime(FECHA_INICIO) if FECHA_INICIO else price_df['datetime'].min()
        fecha_fin = pd.to_datetime(FECHA_FIN) if FECHA_FIN else price_df['datetime'].max()

        # Filtrar ambos dataframes
        self.price_data = price_df[(price_df['datetime'] >= fecha_inicio) & (price_df['datetime'] <= fecha_fin)].copy()
        self.trade_data = trade_df[(trade_df['datetime'] >= fecha_inicio) & (trade_df['datetime'] <= fecha_fin)].copy()

    def crear_grafico_principal(self):
        if self.price_data is None or self.trade_data is None:
            print("Faltan datos para generar gráficos")
            return

        # Filtrar por rango de fechas de config
        self._filtrar_por_rango_fechas()

        # Crear figura con subplots alineados y compartir eje X
        fig, axes = plt.subplots(
            5, 1, figsize=(18, 20), sharex=True,
            gridspec_kw={'height_ratios': [1.8, 1, 1, 1, 1]}
        )

        # Separar compras y ventas
        compras = self.trade_data[self.trade_data['type'] == 'BUY']
        ventas = self.trade_data[self.trade_data['type'] == 'SELL']

        # 1. PRECIO BTC + OPERACIONES + Precio promedio de compra de posiciones (línea continua)
        ax1 = axes[0]
        ax1.plot(self.price_data['datetime'], self.price_data['close'], 
                color='black', alpha=0.7, linewidth=1, label='Precio BTC')
        if len(compras) > 0:
            ax1.scatter(compras['datetime'], compras['price'], 
                       color='green', alpha=0.6, s=15, label=f'Compras ({len(compras)})', zorder=5)
        if len(ventas) > 0:
            ax1.scatter(ventas['datetime'], ventas['price'], 
                       color='red', alpha=0.6, s=15, label=f'Ventas ({len(ventas)})', zorder=5)
        # --- Línea continua del precio promedio de compra de posiciones ---
        if 'precio_promedio_posiciones' in self.trade_data.columns:
            # Elimina duplicados en los índices antes de reindexar
            price_datetimes = self.price_data['datetime'].drop_duplicates()
            trade_promedio = self.trade_data.set_index('datetime')['precio_promedio_posiciones']
            trade_promedio = trade_promedio[~trade_promedio.index.duplicated(keep='last')]
            serie_promedio = trade_promedio.reindex(price_datetimes, method='ffill')
            serie_promedio = serie_promedio.replace(0, np.nan)
            ax1.plot(price_datetimes, serie_promedio,
                     color='blue', linestyle='--', linewidth=1.5,
                     label='Precio Promedio Compra Posiciones')
        ax1.set_ylabel('Precio BTC (USD)')
        ax1.set_title('Precio BTC + Operaciones de Trading')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. BALANCE USDT
        ax2 = axes[1]
        ax2.plot(self.trade_data['datetime'], self.trade_data['usdt_balance'], 
                color='blue', linewidth=2, label='Balance USDT')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Saldo $0')
        ax2.fill_between(self.trade_data['datetime'], 0, self.trade_data['usdt_balance'], 
                        alpha=0.3, color='blue')
        ax2.set_ylabel('Balance USDT')
        ax2.set_title('Evolución del Balance USDT')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. BALANCE BTC (3 líneas: en posiciones, acumulado/libre, total)
        if 'btc_en_posiciones' in self.trade_data.columns and 'btc_balance' in self.trade_data.columns:
            self.trade_data['btc_total'] = self.trade_data['btc_en_posiciones'] + self.trade_data['btc_balance']
        else:
            self.trade_data['btc_total'] = self.trade_data['btc_balance']

        ax3 = axes[2]
        if 'btc_en_posiciones' in self.trade_data.columns:
            ax3.plot(self.trade_data['datetime'], self.trade_data['btc_en_posiciones'],
                     color='blue', linewidth=1.5, label='BTC en Posiciones')
        if 'btc_balance' in self.trade_data.columns:
            ax3.plot(self.trade_data['datetime'], self.trade_data['btc_balance'],
                     color='purple', linewidth=1.5, label='BTC Acumulado (libre)')
        ax3.plot(self.trade_data['datetime'], self.trade_data['btc_total'],
                 color='orange', linewidth=2, label='BTC Total')
        ax3.fill_between(self.trade_data['datetime'], 0, self.trade_data['btc_total'],
                         alpha=0.2, color='orange')
        ax3.set_ylabel('Balance BTC')
        ax3.set_title('Evolución del Balance BTC')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 4. POSICIONES ABIERTAS
        ax4 = axes[3]
        ax4.plot(self.trade_data['datetime'], self.trade_data['positions_count'], 
                color='purple', linewidth=2, label='Posiciones Abiertas')
        ax4.fill_between(self.trade_data['datetime'], 0, self.trade_data['positions_count'], 
                        alpha=0.3, color='purple')
        ax4.set_ylabel('Número de Posiciones')
        ax4.set_title('Número de Posiciones Abiertas')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

        # Interpolar balances de los trades sobre todas las velas
        price_df = self.price_data.copy()
        trade_df = self.trade_data.copy()

        # Asegúrate de que los índices sean datetime y únicos
        price_df = price_df.set_index('datetime')
        trade_df = trade_df.set_index('datetime')
        trade_df = trade_df[~trade_df.index.duplicated(keep='last')]

        # Interpola los balances y cantidades de BTC/USDT
        for col in ['usdt_balance', 'btc_balance', 'btc_en_posiciones']:
            if col in trade_df.columns:
                price_df[col] = trade_df[col].reindex(price_df.index, method='ffill')
            else:
                price_df[col] = 0.0

        # Calcula BTC total
        price_df['btc_total'] = price_df['btc_balance'] + price_df['btc_en_posiciones']

        # Calcula el valor de BTC y el valor total del portfolio en cada vela
        price_df['btc_value'] = price_df['btc_total'] * price_df['close']
        price_df['portfolio_value'] = price_df['usdt_balance'] + price_df['btc_value']

        # Filtrar self.price_df por el rango de fechas de config
        fecha_inicio = pd.to_datetime(FECHA_INICIO) if FECHA_INICIO else self.price_df.index.min()
        fecha_fin = pd.to_datetime(FECHA_FIN) if FECHA_FIN else self.price_df.index.max()
        mask = (self.price_df.index >= fecha_inicio) & (self.price_df.index <= fecha_fin)
        price_df_filtrado = self.price_df.loc[mask]

        # 5. VALOR TOTAL DEL PORTFOLIO (3 líneas: USDT, valor BTC, total)
        from config import SALDO_USDT_INICIAL
        ax5 = axes[4]
        ax5.plot(price_df_filtrado.index, price_df_filtrado['usdt_balance'],
                color='dodgerblue', linewidth=1.5, label='Valor USDT')
        ax5.plot(price_df_filtrado.index, price_df_filtrado['btc_value'],
                color='orange', linewidth=1.5, label='Valor BTC Total')
        ax5.plot(price_df_filtrado.index, price_df_filtrado['portfolio_value'],
                color='green', linewidth=2, label='Valor Total Portfolio')
        ax5.fill_between(price_df_filtrado.index, SALDO_USDT_INICIAL, price_df_filtrado['portfolio_value'],
                        where=(price_df_filtrado['portfolio_value'] >= SALDO_USDT_INICIAL),
                        alpha=0.2, color='green', label='Ganancia')
        ax5.fill_between(price_df_filtrado.index, SALDO_USDT_INICIAL, price_df_filtrado['portfolio_value'],
                        where=(price_df_filtrado['portfolio_value'] < SALDO_USDT_INICIAL),
                        alpha=0.2, color='red', label='Pérdida')

        ax5.axhline(y=SALDO_USDT_INICIAL, color='gray', linestyle='--', alpha=0.7,
                    label=f'Capital Inicial (${SALDO_USDT_INICIAL:,})')
        ax5.set_ylabel('Valor Portfolio (USD)')
        ax5.set_xlabel('Fecha')
        ax5.set_title('Valor Total del Portfolio')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)

        # Mejorar fechas en eje X: ticks y formato DD/MM/YY en todos los ejes
        locator = mdates.AutoDateLocator(minticks=10, maxticks=30)
        formatter = mdates.DateFormatter('%d/%m/%y')
        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

        plt.tight_layout(rect=[0, 0, 1, 1])

        # Maximizar ventana antes de mostrar
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')  # Windows
        except AttributeError:
            try:
                mng.window.showMaximized()  # Linux
            except AttributeError:
                pass

        nombre_archivo = 'analisis_estrategia_completo.png'
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {nombre_archivo}")

        plt.show()

    def crear_analisis_detallado(self):
        fecha_inicio = pd.to_datetime(FECHA_INICIO) if FECHA_INICIO else self.price_df.index.min()
        fecha_fin = pd.to_datetime(FECHA_FIN) if FECHA_FIN else self.price_df.index.max()
        mask = (self.price_df.index >= fecha_inicio) & (self.price_df.index <= fecha_fin)
        price_df_filtrado = self.price_df.loc[mask]

        """Crear análisis estadístico detallado"""
        if self.trade_data is None:
            return

        print("\n" + "="*60)
        print("ANÁLISIS ESTADÍSTICO DETALLADO")
        print("="*60)

        compras = self.trade_data[self.trade_data['type'] == 'BUY']
        ventas = self.trade_data[self.trade_data['type'] == 'SELL']

        # Estadísticas de compras
        if len(compras) > 0:
            print(f"\nESTADÍSTICAS DE COMPRAS:")
            print(f"   Total compras: {len(compras)}")
            print(f"   Precio promedio: ${compras['price'].mean():,.2f}")
            print(f"   Precio mínimo: ${compras['price'].min():,.2f}")
            print(f"   Precio máximo: ${compras['price'].max():,.2f}")
            if 'usdt_spent' in compras.columns:
                print(f"   Monto promedio: ${compras['usdt_spent'].mean():,.2f}")
                print(f"   Total invertido: ${compras['usdt_spent'].sum():,.2f}")
            if 'caida_ath_pct' in compras.columns:
                print(f"   Caída máxima desde ATH: {compras['caida_ath_pct'].max():.2f}%")

        # Estadísticas de ventas
        if len(ventas) > 0:
            print(f"\nESTADÍSTICAS DE VENTAS:")
            print(f"   Total ventas: {len(ventas)}")
            print(f"   Precio promedio: ${ventas['price'].mean():,.2f}")
            if 'ganancia_usdt' in ventas.columns:
                ganancias = ventas['ganancia_usdt'].dropna()
                if len(ganancias) > 0:
                    print(f"   Ganancia promedio: ${ganancias.mean():,.2f}")
                    print(f"   Ganancia total: ${ganancias.sum():,.2f}")

        # Análisis temporal
        print(f"\nANÁLISIS TEMPORAL:")
        fecha_inicio = self.trade_data['datetime'].min()
        fecha_fin = self.trade_data['datetime'].max()
        duracion = (fecha_fin - fecha_inicio).days
        print(f"   Período: {fecha_inicio.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')}")
        print(f"   Duración: {duracion} días")
        if duracion > 0:
            print(f"   Trades por mes: {len(self.trade_data) / (duracion / 30):.1f}")
        else:
            print("   Trades por mes: N/A (duración menor a 1 día)")

        # Análisis de portfolio
        if 'portfolio_value' in price_df_filtrado.columns and len(price_df_filtrado) > 0:
            valor_inicial = price_df_filtrado['portfolio_value'].iloc[0]
            valor_final = price_df_filtrado['portfolio_value'].iloc[-1]
            valor_maximo = price_df_filtrado['portfolio_value'].max()
            if len(price_df_filtrado['portfolio_value']) > 20:
                valor_minimo_20 = price_df_filtrado['portfolio_value'].iloc[20:].min()
            else:
                valor_minimo_20 = price_df_filtrado['portfolio_value'].min()
            drawdown_maximo = calcular_drawdown_maximo(price_df_filtrado['portfolio_value'].values)

            print("\nANÁLISIS DE PORTFOLIO:")
            print(f"   Valor inicial: ${valor_inicial:,.2f}")
            print(f"   Valor final:   ${valor_final:,.2f}")
            print(f"   Valor máximo:  ${valor_maximo:,.2f}")
            print(f"   Valor mínimo (desde vela 20): ${valor_minimo_20:,.2f}")
            print(f"   Drawdown máximo: {drawdown_maximo:.2f}%")
        else:
            print("\nANÁLISIS DE PORTFOLIO:")
            print("   No hay datos suficientes para el análisis de portfolio.")
                    
    def analizar_periodos_criticos(self):
        """Analizar períodos donde el saldo USDT llegó a 0"""
        if self.trade_data is None:
            return

        # Encontrar períodos críticos (saldo USDT <= $0)
        periodos_criticos = self.trade_data[self.trade_data['usdt_balance'] <= 0]

        if len(periodos_criticos) == 0:
            print("\nNo se encontraron períodos críticos (saldo USDT > $0)")
            return

        print(f"\nANÁLISIS DE PERÍODOS CRÍTICOS")
        print("="*50)
        print(f"Períodos con saldo ≤ $0: {len(periodos_criticos)}")

        # Mostrar algunos ejemplos
        print(f"\nEjemplos de períodos críticos:")
        for i, (_, periodo) in enumerate(periodos_criticos.head(10).iterrows()):
            print(f"   {i+1}. {periodo['datetime'].strftime('%Y-%m-%d %H:%M')} - "
                  f"USDT: ${periodo['usdt_balance']:.2f} - "
                  f"Posiciones: {periodo['positions_count']}")

def main():
    print("GRAFICADOR DE ESTRATEGIAS BTC")
    print("="*50)

    graficador = Graficador()

    print("Cargando datos...")

    if not graficador.cargar_datos_precios():
        print("No se pudieron cargar los datos de precios")
        return

    if not graficador.cargar_resultados_estrategia():
        print("No se pudieron cargar los resultados de la estrategia")
        return

    # Primero mostrar el análisis en terminal
    graficador.crear_analisis_detallado()
    graficador.analizar_periodos_criticos()

    print("\nGenerando gráfico completo...")
    graficador.crear_grafico_principal()

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Error: Falta instalar dependencias")
        print(f"Ejecuta: pip install matplotlib pandas numpy")
        print(f"Error específico: {e}")
        exit(1)

    main()