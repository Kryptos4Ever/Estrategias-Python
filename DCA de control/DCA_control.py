"""
DCA Control: Estrategia de acumulación de BTC con Dollar Cost Averaging
Ejecutar: python DCA_control.py
"""
from datetime import datetime, timezone
from DCA_config import *
from utils import calcular_monto_aritmetico
from data_manager import DataManager
from datetime import datetime
import math

class Estrategia:
    def __init__(self):
        self.saldo_usdt = SALDO_USDT_INICIAL
        self.saldo_inicial = SALDO_USDT_INICIAL
        self.saldo_btc = 0.0
        self.ath = None
        self.posiciones = []  # Mantener esta lista aunque esté vacía para compatibilidad
        self.trade_history = []
        self.last_operation_price = None
        self.ultimo_cierre = 0.0
        self.ultimo_timestamp = None
        self.nivel_compra_actual = 0  # Nivel discreto de compra actual (0 = sin compras)
        self.incremento_nivel = PCT_CAIDA_ATH  # Incremento para el primer nivel

    def _calcular_precio_promedio_posiciones(self):
        """Calcula el precio promedio de todas las compras realizadas"""
        if not self.trade_history:
            return 0
            
        compras = [t for t in self.trade_history if t['type'] == 'BUY']
        if not compras:
            return 0
            
        total_btc = sum([t['amount'] for t in compras])
        total_usdt = sum([t['usdt_spent'] for t in compras])
        
        if total_btc > 0:
            return total_usdt / total_btc
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
            # Resetear nivel de compra si hay nuevo ATH
            self.nivel_compra_actual = 0
            
        # Guardar el último precio de cierre
        self.ultimo_cierre = close

        # Calcular caída desde ATH
        pct_caida_ath = (self.ath - low) / self.ath if self.ath > 0 else 0

        # Procesar solo compras (sin ventas)
        self._procesar_compras(timestamp, low, pct_caida_ath)

    def _procesar_compras(self, timestamp, precio, pct_caida_ath):
        """Lógica de compras basada en niveles discretos de caída desde ATH"""
        
        # Calcular la caída actual desde ATH
        caida_actual = (self.ath - precio) / self.ath if self.ath > 0 else 0
        
        # Calcular el nivel discreto correspondiente a la caída actual
        # Nivel 1: PCT_CAIDA_ATH (ej: 10%)
        # Nivel 2: PCT_CAIDA_ATH + PCT_CAIDA_LAST_ATH_BUY (ej: 20%)
        # Nivel 3: PCT_CAIDA_ATH + 2*PCT_CAIDA_LAST_ATH_BUY (ej: 30%)
        # etc.
        
        # Calcular cuántos incrementos después del nivel base
        incrementos_adicionales = 0
        if caida_actual > PCT_CAIDA_ATH:
            incrementos_adicionales = math.floor((caida_actual - PCT_CAIDA_ATH) / PCT_CAIDA_LAST_ATH_BUY)
        
        # Nivel discreto actual (1, 2, 3, etc.)
        nivel_actual = 1 + incrementos_adicionales if caida_actual >= PCT_CAIDA_ATH else 0
        
        # Ejecutar compra solo si alcanzamos un nuevo nivel y es mayor al nivel actual
        if nivel_actual > self.nivel_compra_actual and nivel_actual > 0:
            # Calcular la caída exacta del nivel para el registro
            caida_nivel = PCT_CAIDA_ATH + (nivel_actual - 1) * PCT_CAIDA_LAST_ATH_BUY
            
            # Tipo de compra
            tipo = "PRIMERA_COMPRA_ATH" if nivel_actual == 1 else f"COMPRA_NIVEL_{nivel_actual}"
            
            # Ejecutar la compra
            self._ejecutar_compra(timestamp, precio, caida_nivel, tipo)
            
            # Actualizar el nivel de compra actual
            self.nivel_compra_actual = nivel_actual

    def _ejecutar_compra(self, timestamp, precio, pct_caida_ath, tipo):
        """Ejecuta una compra"""
        from DCA_config import MIN_COMPRA_PCT, MAX_COMPRA_PCT, CAIDA_MAXIMA
        
        monto_usdt = calcular_monto_aritmetico(
            pct_caida_ath, 
            self.saldo_inicial,
            min_compra_pct=MIN_COMPRA_PCT,
            max_compra_pct=MAX_COMPRA_PCT,
            caida_maxima=CAIDA_MAXIMA
        )

        if self.saldo_usdt >= monto_usdt and monto_usdt > 0:
            cantidad_btc = monto_usdt / precio
            self.saldo_usdt -= monto_usdt
            self.last_operation_price = precio
            self.saldo_btc += cantidad_btc  # En DCA, acumulamos directamente en saldo_btc

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
                "positions_count": 0,  # En DCA no hay posiciones individuales
                "btc_en_posiciones": 0.0,  # En DCA no hay BTC en posiciones
                "ath": self.ath,
                "caida_ath_pct": pct_caida_ath * 100,
                "nivel_compra": self.nivel_compra_actual,
                "precio_promedio_posiciones": self._calcular_precio_promedio_posiciones()
            }

            self.trade_history.append(compra_info)

    def registrar_trade_final(self):
        """Registra un trade virtual final para que el análisis y el graficador sean consistentes"""
        btc_total = self.saldo_btc
        trade_final = {
            "datetime": datetime.fromtimestamp(self.ultimo_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "FINAL",
            "price": self.ultimo_cierre,
            "usdt_balance": self.saldo_usdt,
            "btc_balance": self.saldo_btc,
            "btc_en_posiciones": 0.0,  # En DCA no hay BTC en posiciones
            "positions_count": 0,  # En DCA no hay posiciones individuales
            "portfolio_value": self.saldo_usdt + btc_total * self.ultimo_cierre
        }
        self.trade_history.append(trade_final)

    def mostrar_resumen_final(self):
        """Muestra solo resumen básico al final"""
        btc_total = self.saldo_btc
        valor_btc_usd = btc_total * self.ultimo_cierre if self.ultimo_cierre > 0 else 0
        valor_total_portfolio = self.saldo_usdt + valor_btc_usd
        
        # Calcular precio promedio de compra
        compras = [t for t in self.trade_history if t['type'] == 'BUY']
        total_btc = sum([t['amount'] for t in compras]) if compras else 0
        total_usdt = sum([t['usdt_spent'] for t in compras]) if compras else 0
        precio_promedio = total_usdt / total_btc if total_btc > 0 else 0

        print("\n" + "="*50)
        print("RESUMEN ESTRATEGIA DCA")
        print("="*50)
        print(f"USDT final: ${self.saldo_usdt:,.2f}")
        print(f"BTC acumulado: {self.saldo_btc:.8f}")
        print(f"Valor portfolio: ${valor_total_portfolio:,.2f}")
        print(f"Ganancia: ${valor_total_portfolio - SALDO_USDT_INICIAL:,.2f}")
        print(f"ROI: {((valor_total_portfolio - SALDO_USDT_INICIAL) / SALDO_USDT_INICIAL) * 100:.2f}%")
        print(f"Compras: {len(compras)}")
        print(f"Precio promedio de compra BTC: ${precio_promedio:,.2f}")
        print("="*50)

def ejecutar_backtest():
    """Función principal para ejecutar el backtest"""
    print("ESTRATEGIA DCA: ACUMULACIÓN BTC")
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
        from DCA_config import RESULTS_JSON
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
