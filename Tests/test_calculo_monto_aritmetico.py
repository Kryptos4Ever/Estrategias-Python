"""
Test unitario para calcular_monto_aritmetico y simulación de compras consecutivas.
Completa los parámetros de prueba aquí abajo:
"""

# ==== CONFIGURACIÓN DE PRUEBA ====
MONTO_INICIAL = 1000          # Saldo USDT inicial para la simulación
NIVELES = 20                 # Número de caídas consecutivas a simular
CAIDA_POR_VELA = 0.05        # Caída porcentual incremental por nivel
MIN_COMPRA_PCT = 0.01       # % del saldo para la primera compra
MAX_COMPRA_PCT = 0.05        # % máximo del saldo para la compra con máxima caída
CAIDA_MAXIMA = 0.99           # Caída máxima histórica esperada
# ==== FIN DE CONFIGURACIÓN ====

import sys
import os
import pandas as pd

# Ajusta el path para importar utils.py si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar la función directamente sin importar las constantes de config.py
from utils import calcular_monto_aritmetico

def test_monto_no_supera_saldo():
    saldo = 1000
    monto = calcular_monto_aritmetico(
        0.5, saldo,
        min_compra_pct=MIN_COMPRA_PCT,
        max_compra_pct=MAX_COMPRA_PCT,
        caida_maxima=CAIDA_MAXIMA
    )
    assert monto <= saldo
    print(f"Test monto_no_supera_saldo: OK - Monto calculado: {monto:.2f}")

def test_monto_minimo():
    saldo = 1000
    esperado = saldo * MIN_COMPRA_PCT
    monto = calcular_monto_aritmetico(
        0, saldo,
        min_compra_pct=MIN_COMPRA_PCT,
        max_compra_pct=MAX_COMPRA_PCT,
        caida_maxima=CAIDA_MAXIMA
    )
    assert abs(monto - esperado) < 1e-2
    print(f"Test monto_minimo: OK - Esperado: {esperado:.2f}, Obtenido: {monto:.2f}")

def test_monto_aumenta_con_caida():
    saldo = 1000
    m1 = calcular_monto_aritmetico(
        0.1, saldo,
        min_compra_pct=MIN_COMPRA_PCT,
        max_compra_pct=MAX_COMPRA_PCT,
        caida_maxima=CAIDA_MAXIMA
    )
    m2 = calcular_monto_aritmetico(
        0.5, saldo,
        min_compra_pct=MIN_COMPRA_PCT,
        max_compra_pct=MAX_COMPRA_PCT,
        caida_maxima=CAIDA_MAXIMA
    )
    m3 = calcular_monto_aritmetico(
        0.9, saldo,
        min_compra_pct=MIN_COMPRA_PCT,
        max_compra_pct=MAX_COMPRA_PCT,
        caida_maxima=CAIDA_MAXIMA
    )
    assert m1 <= m2 <= m3
    print(f"Test monto_aumenta_con_caida: OK - Montos: {m1:.2f} <= {m2:.2f} <= {m3:.2f}")

def test_simulacion_caidas_consecutivas():
    saldo = MONTO_INICIAL
    saldo_inicial = MONTO_INICIAL
    resultados = []
    precio_inicial = 20000  # ATH constante
    
    for i in range(NIVELES):
        # Calcular caída incremental respecto al ATH (no compuesta)
        nivel_caida = (i + 1) * CAIDA_POR_VELA  # 5%, 10%, 15%, etc.
        caida_desde_ath = min(nivel_caida, 0.99)  # Limitar a 99% máximo
        
        # Calcular precio con la caída incremental
        precio = precio_inicial * (1 - caida_desde_ath)
        
        monto = calcular_monto_aritmetico(
            caida_desde_ath, saldo_inicial,
            min_compra_pct=MIN_COMPRA_PCT,
            max_compra_pct=MAX_COMPRA_PCT,
            caida_maxima=CAIDA_MAXIMA
        )
        monto = min(monto, saldo)
        saldo_restante = saldo - monto
        resultados.append({
            "Nivel": i + 1,
            "Caída desde ATH (%)": round(caida_desde_ath * 100, 2),
            "Precio compra": round(precio, 2),
            "Monto compra": round(monto, 2),
            "Saldo restante": round(saldo_restante, 2)
        })
        saldo = saldo_restante
        if saldo <= 0:
            break
    df = pd.DataFrame(resultados)
    print("\nTabla de compras consecutivas con caídas incrementales del 5% respecto al ATH:")
    print(df.to_string(index=False))
    assert all(df["Saldo restante"] >= 0)
    print(f"Test simulacion_caidas_consecutivas: OK - {len(resultados)} niveles simulados")

if __name__ == "__main__":
    print("Ejecutando tests para calcular_monto_aritmetico...\n")
    print(f"Configuración de prueba:")
    print(f"- MIN_COMPRA_PCT: {MIN_COMPRA_PCT:.2%}")
    print(f"- MAX_COMPRA_PCT: {MAX_COMPRA_PCT:.2%}")
    print(f"- CAIDA_MAXIMA: {CAIDA_MAXIMA:.2%}")
    print(f"- NIVELES: {NIVELES}")
    print(f"- CAIDA_POR_VELA: {CAIDA_POR_VELA:.2%}")
    print()
    
    test_monto_no_supera_saldo()
    test_monto_minimo()
    test_monto_aumenta_con_caida()
    test_simulacion_caidas_consecutivas()
    print("\nTodos los tests pasaron exitosamente.")
