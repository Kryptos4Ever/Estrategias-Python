"""
Gestión de datos de la base de datos y archivos JSON
"""
import sqlite3
import json
from datetime import datetime, timezone
from config import DB_PATH, FECHA_INICIO, FECHA_FIN, RESULTS_JSON
import os
import pandas as pd

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]

class DataManager:
    def __init__(self):
        self.db_path = DB_PATH
    
    def obtener_velas(self):
        """Obtener velas con filtro de fechas opcional"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Construir query base
        base_query = f"SELECT timestamp, open, high, low, close FROM {DB_TABLE}"
        conditions = []
        params = []
        
        # Agregar condiciones de fecha si están definidas
        if FECHA_INICIO:
            timestamp_inicio = self._convertir_fecha_a_timestamp(FECHA_INICIO)
            if timestamp_inicio:
                conditions.append("timestamp >= ?")
                params.append(timestamp_inicio)
                print(f"Filtro desde: {self._timestamp_a_fecha(timestamp_inicio)}")
        
        if FECHA_FIN:
            timestamp_fin = self._convertir_fecha_a_timestamp(FECHA_FIN, es_fin=True)
            if timestamp_fin:
                conditions.append("timestamp <= ?")
                params.append(timestamp_fin)
                print(f"Filtro hasta: {self._timestamp_a_fecha(timestamp_fin)}")
        
        # Construir query final
        if conditions:
            query = f"{base_query} WHERE {' AND '.join(conditions)} ORDER BY timestamp ASC"
        else:
            query = f"{base_query} ORDER BY timestamp ASC"
        
        cur.execute(query, params)
        velas = cur.fetchall()
        
        # Mostrar información del rango obtenido
        if velas:
            primera_vela = datetime.fromtimestamp(velas[0][0] / 1000, tz=timezone.utc)
            ultima_vela = datetime.fromtimestamp(velas[-1][0] / 1000, tz=timezone.utc)
            print(f"Datos obtenidos: {len(velas)} velas")
            print(f"Rango real: {primera_vela.strftime('%Y-%m-%d %H:%M')} a {ultima_vela.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("No se encontraron datos en el rango especificado")
        
        conn.close()
        return velas
    
    def _convertir_fecha_a_timestamp(self, fecha_str, es_fin=False):
        """Convierte string de fecha a timestamp"""
        try:
            if len(fecha_str) == 10:  # Solo fecha
                fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d")
                if es_fin:
                    # Si es fecha fin, incluir todo el día
                    fecha_dt = fecha_dt.replace(hour=23, minute=59, second=59)
            else:  # Fecha y hora
                fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
            
            return int(fecha_dt.timestamp() * 1000)
        except ValueError as e:
            print(f"Error en fecha '{fecha_str}': {e}")
            print("Formato esperado: 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM:SS'")
            return None
    
    def _timestamp_a_fecha(self, timestamp):
        """Convierte timestamp a string legible"""
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    def guardar_resultados(self, trade_history, filename=None):
        """Guardar historial de trades en JSON"""
        if filename is None:
            from config import RESULTS_JSON
            filename = RESULTS_JSON
        
        results = {
            "trade_history": trade_history,
            "config": {
                "fecha_inicio": FECHA_INICIO,
                "fecha_fin": FECHA_FIN,
                "generado": datetime.now().isoformat()
            }
        }
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Historial guardado en {filename}")
    
    def cargar_resultados(self, filename=None):
        """Cargar resultados desde JSON"""
        if filename is None:
            from config import RESULTS_JSON
            filename = RESULTS_JSON
        
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"No se encontró {filename}")
            return None
        
    def cargar_resultados_df(self, archivo=None):
        """Carga el historial de trades desde el archivo JSON y lo devuelve como DataFrame"""
        if archivo is None:
            archivo = RESULTS_JSON
        with open(archivo, 'r') as f:
            data = json.load(f)
        trade_history = data['trade_history'] if 'trade_history' in data else data
        df = pd.DataFrame(trade_history)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df