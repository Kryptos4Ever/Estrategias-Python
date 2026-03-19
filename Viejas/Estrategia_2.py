"""
Estrategia 2: BTC Accumulation con progresión aritmética
Ejecutar: python Estrategia_2.py
"""
from datetime import datetime, timezone
from config import *
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
        self.posiciones = []
        self.trade_history = []
        self.last_operation_price = None
        self.ultimo_cierre = 0.0
        self.ultimo_timestamp = None
        self.last_ath_buy_price = None
        self.nivel_compra_actual = 0

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
            # Resetear nivel de compra si hay nuevo ATH
            self.nivel_compra_actual = 0
        # Guardar el último precio de cierre
        self.ultimo_cierre = close

        # Calcular caída desde ATH
        pct_caida_ath = (self.ath - low) / self.ath if self.ath > 0 else 0

        # Procesar compras y ventas
        self._procesar_ventas(timestamp, high)
        self._procesar_compras(timestamp, low, pct_caida_ath)

    def _procesar_ventas(self, timestamp, precio):
        posiciones_a_remover = []

        # --- 1. Procesar TP2 primero (venta total) ---
        # --- 1. Procesar TP2 primero (venta total) ---
        for i in range(len(self.posiciones) - 1, -1, -1):  # Iteramos en reversa
            pos = self.posiciones[i]
            if precio >= pos['objetivo_tp_2']:
                cantidad_a_vender = pos['cantidad']
                btc_a_guardar = cantidad_a_vender * BTC_KEEP_PCT
                btc_a_vender = cantidad_a_vender * (1 - BTC_KEEP_PCT)
                usdt_obtenido = btc_a_vender * pos['objetivo_tp_2']

                self.saldo_btc += btc_a_guardar
                self.saldo_usdt += usdt_obtenido
                self.last_operation_price = pos['objetivo_tp_2']
                
                # Eliminar la posición inmediatamente
                self.posiciones.pop(i)
                
                # Calcular btc_en_posiciones DESPUÉS de eliminar
                btc_en_posiciones = sum([p['cantidad'] for p in self.posiciones])
                
                # Registrar trade TP2
                venta_info = {
                    "datetime": datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "type": "SELL",
                    "price": pos['objetivo_tp_2'],
                    "amount": cantidad_a_vender,
                    "btc_to_keep": btc_a_guardar,
                    "btc_to_sell": btc_a_vender,
                    "usdt_obtained": usdt_obtenido,
                    "usdt_balance": self.saldo_usdt,
                    "btc_balance": self.saldo_btc,
                    "positions_count": len(self.posiciones),
                    "btc_en_posiciones": btc_en_posiciones,
                }
                self.trade_history.append(venta_info)


        # --- 2. Procesar TP1 (venta parcial) ---
        for pos in self.posiciones:
            if not pos['tp1_ejecutado'] and precio >= pos['objetivo_tp_1']:
                cantidad_a_vender = pos['cantidad'] * TP1_SELL_PCT
                btc_a_guardar = cantidad_a_vender * BTC_KEEP_PCT
                btc_a_vender = cantidad_a_vender * (1 - BTC_KEEP_PCT)
                usdt_obtenido = btc_a_vender * pos['objetivo_tp_1']

                self.saldo_btc += btc_a_guardar
                self.saldo_usdt += usdt_obtenido

                self.last_operation_price = pos['objetivo_tp_1']

                pos['cantidad'] -= cantidad_a_vender
                pos['tp1_ejecutado'] = True

                # Registrar trade TP1
                venta_info = {
                    "datetime": datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "type": "SELL",
                    "price": pos['objetivo_tp_1'],
                    "amount": cantidad_a_vender,
                    "btc_to_keep": btc_a_guardar,
                    "btc_to_sell": btc_a_vender,
                    "usdt_obtained": usdt_obtenido,
                    "usdt_balance": self.saldo_usdt,
                    "btc_balance": self.saldo_btc,
                    "positions_count": len(self.posiciones),
                    "btc_en_posiciones": sum([p['cantidad'] for p in self.posiciones]),
                }
                self.trade_history.append(venta_info)

        

    def _procesar_compras(self, timestamp, precio, pct_caida_ath):
        """Lógica de compras: evaluar criterios independientemente"""
        compra_ejecutada = False
        # 1. Compras por caída desde ATH (usando niveles discretos)
        caida_actual = (self.ath - precio) / self.ath if self.ath > 0 else 0        
        # Calcular el nivel discreto correspondiente a la caída actual
        incrementos_adicionales = 0
        if caida_actual > PCT_CAIDA_ATH:
            incrementos_adicionales = math.floor((caida_actual - PCT_CAIDA_ATH) / PCT_CAIDA_LAST_ATH_BUY)        
        # Nivel discreto actual (1, 2, 3, etc.)
        nivel_actual = 1 + incrementos_adicionales if caida_actual >= PCT_CAIDA_ATH else 0
        # Ejecutar compra solo si alcanzamos un nuevo nivel y es mayor al nivel actual
        if nivel_actual > 0 and (self.last_ath_buy_price is None or nivel_actual > self.nivel_compra_actual):
            # Calcular la caída exacta del nivel para el registro
            caida_nivel = PCT_CAIDA_ATH + (nivel_actual - 1) * PCT_CAIDA_LAST_ATH_BUY
            # Actualizar el nivel de compra actual
            self.nivel_compra_actual = nivel_actual
            self.last_ath_buy_price = precio
            # Ejecutar la compra
            self._ejecutar_compra(timestamp, precio, caida_nivel, "COMPRA_POR_ATH")
            compra_ejecutada = True
        
        # 2. Compra adicional por caída desde última operación (si no se ejecutó compra ATH)
        if not compra_ejecutada and self.last_operation_price is not None:
            if precio <= self.last_operation_price * (1 - PCT_CAIDA):
                self._ejecutar_compra(timestamp, precio, pct_caida_ath, "COMPRA_ADICIONAL")

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
                'objetivo_tp_1': precio * (1 + PCT_TAKE_PROFIT_1),
                'objetivo_tp_2': precio * (1 + PCT_TAKE_PROFIT_2),
                'tp1_ejecutado': False
})

            # Si es compra por ATH, guardar el precio
            if tipo == "COMPRA_POR_ATH":
                self.last_ath_buy_price = precio

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
                "precio_promedio_posiciones": precio_promedio_posiciones,
                "nivel_compra": self.nivel_compra_actual if tipo == "COMPRA_POR_ATH" else None,
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
        print("RESUMEN ESTRATEGIA")
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
    print("ESTRATEGIA 2: ACUMULACIÓN BTC CON 2 TP")
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