"""
Estrategia 1: BTC Accumulation con progresión aritmética
Ejecutar: python Estrategia_1.py
"""
from datetime import datetime, timezone
from config import *
from utils import calcular_monto_aritmetico
from data_manager import DataManager
from datetime import datetime

class Estrategia:
    def __init__(self):
        self.saldo_usdt = SALDO_USDT_INICIAL
        self.saldo_inicial = SALDO_USDT_INICIAL
        self.saldo_btc = 0.0
        self.ath = 0.0
        self.posiciones = []
        self.trade_history = []
        self.last_operation_price = None
        self.ultimo_cierre = 0.0
        self.ultimo_timestamp = None

    def _calcular_precio_promedio_posiciones(self, excluir_indice=None):
        """Calcula el precio promedio de compra de las posiciones abiertas.
        Si excluir_indice se indica, excluye esa posición (útil para ventas)."""
        if excluir_indice is not None:
            posiciones = [p for i, p in enumerate(self.posiciones) if i != excluir_indice]
        else:
            posiciones = self.posiciones
        btc_en_posiciones = sum([p['cantidad'] for p in posiciones])
        if btc_en_posiciones > 0:
            costo_total = sum([p['cantidad'] * p['precio'] for p in posiciones])
            return costo_total / btc_en_posiciones
        else:
            return 0

    def procesar_vela(self, vela):
        """Procesa una vela y ejecuta la lógica de trading"""
        timestamp, open_, high, low, close = vela

        # Guardar el último timestamp para el trade final
        self.ultimo_timestamp = timestamp

        # Crear ATH si es None
        if self.ath is None:
            self.ath = high

        # Actualizar ATH
        if high > self.ath:
            self.ath = high
        # Guardar el último precio de cierre
        self.ultimo_cierre = close

        # Calcular caída desde ATH
        pct_caida_ath = (self.ath - low) / self.ath if self.ath > 0 else 0

        # Procesar compras y ventas
        self._procesar_ventas(timestamp, high)
        self._procesar_compras(timestamp, low, pct_caida_ath)

    def _procesar_ventas(self, timestamp, precio):
        """Lógica de ventas"""
        posiciones_a_remover = []

        for i, pos in enumerate(self.posiciones):
            if precio >= pos['objetivo_venta']:
                # Calcular venta total
                usdt_invertido = pos['precio'] * pos['cantidad']
                ganancia_potencial = (pos['objetivo_venta'] - pos['precio']) * pos['cantidad']
                usdt_a_recuperar = usdt_invertido + 0.5 * ganancia_potencial
                cantidad_a_vender = usdt_a_recuperar / pos['objetivo_venta']

                # Ejecutar venta (siempre total)
                self.saldo_usdt += usdt_a_recuperar
                # El resto del BTC (no vendido) es la mitad de la ganancia potencial en BTC
                btc_remanente = pos['cantidad'] - cantidad_a_vender
                self.saldo_btc += btc_remanente

                self.last_operation_price = precio
                posiciones_a_remover.append(i)

                # Calcular BTC en posiciones abiertas después de la venta (sin la posición actual)
                btc_en_posiciones = sum([p['cantidad'] for j, p in enumerate(self.posiciones) if j != i])
                precio_promedio_posiciones = self._calcular_precio_promedio_posiciones(excluir_indice=i)

                # Registrar trade
                venta_info = {
                    "datetime": datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "type": "SELL",
                    "price": pos['objetivo_venta'],
                    "amount": cantidad_a_vender,
                    "usdt_balance": self.saldo_usdt,
                    "btc_balance": self.saldo_btc,
                    "positions_count": len(self.posiciones) - 1,
                    "ath": self.ath,
                    "ganancia_usdt": cantidad_a_vender,
                    "btc_en_posiciones": btc_en_posiciones,
                    "precio_promedio_posiciones": precio_promedio_posiciones
                }
                self.trade_history.append(venta_info)

        # Remover posiciones vendidas completamente
        for i in reversed(posiciones_a_remover):
            self.posiciones.pop(i)

    def _procesar_compras(self, timestamp, precio, pct_caida_ath):
        """Lógica de compras: solo un criterio por vela"""
        if self.posiciones and precio <= self.last_operation_price * (1 - PCT_CAIDA):
            self._ejecutar_compra(timestamp, precio, pct_caida_ath, "COMPRA_ADICIONAL")
        elif len(self.posiciones) < MAX_POSICIONES_ATH and precio <= self.ath * (1 - PCT_CAIDA_ATH):
            self._ejecutar_compra(timestamp, precio, pct_caida_ath, "COMPRA_POR_ATH")

    def _ejecutar_compra(self, timestamp, precio, pct_caida_ath, tipo):
        """Ejecuta una compra"""
        monto_usdt = calcular_monto_aritmetico(pct_caida_ath, self.saldo_inicial)

        if self.saldo_usdt >= monto_usdt and monto_usdt > 0:
            cantidad_btc = monto_usdt / precio
            self.saldo_usdt -= monto_usdt
            self.last_operation_price = precio

            # Agregar posición
            self.posiciones.append({
                'precio': precio,
                'cantidad': cantidad_btc,
                'objetivo_venta': precio * (1 + PCT_TAKE_PROFIT_1)
            })
            # Calcular BTC en posiciones abiertas (incluyendo la recién creada)
            btc_en_posiciones = sum([p['cantidad'] for p in self.posiciones])
            precio_promedio_posiciones = self._calcular_precio_promedio_posiciones()

            # Registrar trade
            compra_info = {
                "datetime": datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                "type": "BUY",
                "subtype": tipo,
                "price": precio,
                "amount": cantidad_btc,
                "usdt_spent": monto_usdt,
                "usdt_balance": self.saldo_usdt,
                "btc_balance": self.saldo_btc,
                "positions_count": len(self.posiciones),
                "ath": self.ath,
                "caida_ath_pct": pct_caida_ath * 100,
                "btc_en_posiciones": btc_en_posiciones,
                "precio_promedio_posiciones": precio_promedio_posiciones
            }

            self.trade_history.append(compra_info)

    def registrar_trade_final(self):
        """Registra un trade virtual final para que el análisis y el graficador sean consistentes"""
        btc_en_posiciones = sum([p['cantidad'] for p in self.posiciones])
        btc_total = self.saldo_btc + btc_en_posiciones
        trade_final = {
            "datetime": datetime.fromtimestamp(self.ultimo_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "FINAL",
            "price": self.ultimo_cierre,
            "usdt_balance": self.saldo_usdt,
            "btc_balance": self.saldo_btc,
            "btc_en_posiciones": btc_en_posiciones,
            "positions_count": len(self.posiciones),
            "portfolio_value": self.saldo_usdt + btc_total * self.ultimo_cierre
        }
        self.trade_history.append(trade_final)

    def mostrar_resumen_final(self):
        """Muestra solo resumen básico al final"""
        btc_en_posiciones = sum([p['cantidad'] for p in self.posiciones])
        btc_total = self.saldo_btc + btc_en_posiciones
        valor_btc_usd = btc_total * self.ultimo_cierre if self.ultimo_cierre > 0 else 0
        valor_total_portfolio = self.saldo_usdt + valor_btc_usd
        costo_total = sum([p['cantidad'] * p['precio'] for p in self.posiciones])
        precio_promedio = (costo_total / btc_en_posiciones) if btc_en_posiciones > 0 else 0

        compras = len([t for t in self.trade_history if t['type'] == 'BUY'])
        ventas = len([t for t in self.trade_history if t['type'] == 'SELL'])

        print("\n" + "="*50)
        print("RESUMEN ESTRATEGIA 1")
        print("="*50)
        print(f"USDT final: ${self.saldo_usdt:,.2f}")
        print(f"BTC solo en posiciones abiertas: {btc_en_posiciones:.8f}")
        print(f"BTC solo en acumulado: {self.saldo_btc:.8f}")
        print(f"BTC total: {btc_total:.8f}")
        print(f"Valor portfolio: ${valor_total_portfolio:,.2f}")
        print(f"Ganancia: ${valor_total_portfolio - SALDO_USDT_INICIAL:,.2f}")
        print(f"ROI: {((valor_total_portfolio - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL) * 100:.2f}%")
        print(f"Compras: {compras} | Ventas: {ventas}")
        print(f"Posiciones abiertas: {len(self.posiciones)}")
        print(f"Precio promedio de compra BTC en posiciones: ${precio_promedio:,.2f}")
        print("="*50)

def ejecutar_backtest():
    """Función principal para ejecutar el backtest"""
    print("ESTRATEGIA 1: BTC ACCUMULATION")
    print("="*50)

    # Mostrar configuración
    mostrar_configuracion()

    try:
        # Cargar datos
        data_manager = DataManager()
        velas = data_manager.obtener_velas()

        if not velas:
            print("No hay datos para procesar")
            return None

        # Ejecutar estrategia
        print(f"\nProcesando {len(velas):,} velas...")
        estrategia = Estrategia()

        # Procesar velas
        for vela in velas:
            estrategia.procesar_vela(vela)

        # Registrar trade final para consistencia con el graficador
        estrategia.registrar_trade_final()

        # Mostrar resumen
        estrategia.mostrar_resumen_final()

        # Guardar resultados
        from config import RESULTS_JSON
        data_manager.guardar_resultados(estrategia.trade_history, RESULTS_JSON)
        print(f"\nResultados guardados en: {RESULTS_JSON}")
        print(f"Para análisis detallado ejecuta: python Graficador.py")

        return estrategia

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    ejecutar_backtest()