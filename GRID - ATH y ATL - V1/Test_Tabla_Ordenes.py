"""
Test de Tabla de Ordenes — Verifica gradualidad y suma exacta
=============================================================
Muestra la tabla completa de compras y ventas para valores de prueba.
"""

from config import (
    SALDO_USDT_INICIAL,
    PASO_PCT_COMPRA, CAIDA_MAXIMA, FACTOR_COMPRA,
    PASO_PCT_VENTA,  SUBIDA_MAXIMA, FACTOR_VENTA,
)

# ── Parametros del test ──────────────────────────────────────
ATH_TEST         = 69000.0
ATL_TEST         = 15476.0   # techo de ventas = ATL * (1 + SUBIDA_MAXIMA/100)
USDT_TEST        = SALDO_USDT_INICIAL
PRECIO_PROM_TEST = 35000.0
BTC_TEST         = 0.0285


# ── Helpers ──────────────────────────────────────────────────

def _niveles_log(p_ini, p_fin, paso, direccion):
    precios = []
    factor  = (1.0 - paso / 100.0) if direccion == "abajo" \
              else (1.0 + paso / 100.0)
    p = p_ini * factor
    while True:
        if direccion == "abajo"  and p <= p_fin: break
        if direccion == "arriba" and p >= p_fin: break
        precios.append(p)
        p *= factor
    return precios


def _serie_geo(budget, factor, n):
    if n == 1:
        return [budget]
    r     = factor ** (1.0 / (n - 1))
    first = budget * (r - 1.0) / (r**n - 1.0)
    return [first * (r ** i) for i in range(n)]


# ── Construccion ─────────────────────────────────────────────

def tabla_compras(ath, usdt_balance):
    p_fin   = ath * (1.0 - CAIDA_MAXIMA / 100.0)
    precios = _niveles_log(ath, p_fin, PASO_PCT_COMPRA, "abajo")
    montos  = _serie_geo(usdt_balance, FACTOR_COMPRA, len(precios))
    return [
        {"precio": p, "caida_pct": (1.0 - p/ath)*100, "monto": m}
        for p, m in zip(precios, montos)
    ]


def tabla_ventas(precio_prom, btc, atl):
    p_fin   = atl * (1.0 + SUBIDA_MAXIMA / 100.0)
    precios = _niveles_log(precio_prom, p_fin, PASO_PCT_VENTA, "arriba")
    montos  = _serie_geo(btc, FACTOR_VENTA, len(precios))
    return [
        {"precio": p, "subida_pct": (p/precio_prom - 1.0)*100, "monto_btc": m}
        for p, m in zip(precios, montos)
    ]


# ── Impresion ────────────────────────────────────────────────

def imprimir(filas, titulo, es_btc):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    n = len(filas)
    if n == 0:
        print("  (sin niveles)")
        return

    montos = [f["monto_btc"] if es_btc else f["monto"] for f in filas]
    r      = (montos[-1] / montos[0]) if montos[0] > 0 else 0
    budget = sum(montos)

    if es_btc:
        print(f"  n={n}  r={r**(1/(n-1)) if n>1 else 1:.5f}  "
              f"first={montos[0]:.7f}  last={montos[-1]:.7f}  "
              f"factor={r:.2f}x  sum={budget:.7f}")
        print(sep)
        print(f"  {'#':>4}  {'Precio':>12}  {'Subida%':>8}  {'BTC':>13}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*8}  {'-'*13}")
        for i, f in enumerate(filas):
            print(f"  {i+1:>4}  ${f['precio']:>11,.0f}  "
                  f"{f['subida_pct']:>7.1f}%  {f['monto_btc']:>13.7f}")
    else:
        print(f"  n={n}  r={r**(1/(n-1)) if n>1 else 1:.5f}  "
              f"first=${montos[0]:.4f}  last=${montos[-1]:.4f}  "
              f"factor={r:.2f}x  sum=${budget:.4f}")
        print(sep)
        print(f"  {'#':>4}  {'Precio':>12}  {'Caida%':>7}  {'USDT':>12}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*7}  {'-'*12}")
        for i, f in enumerate(filas):
            print(f"  {i+1:>4}  ${f['precio']:>11,.0f}  "
                  f"{f['caida_pct']:>6.1f}%  ${f['monto']:>11.4f}")

    print(sep)
    ref      = BTC_TEST if es_btc else USDT_TEST
    ok_sum   = abs(budget - ref) < (1e-7 if es_btc else 0.01)
    ok_cre   = montos[-1] >= montos[0]
    fmt_sum  = f"{budget:.7f} BTC" if es_btc else f"${budget:.4f} USDT"
    print(f"  Suma exacta : {'✓ OK' if ok_sum else '✗ ERROR'}  ({fmt_sum})")
    print(f"  Creciente   : {'✓ SI' if ok_cre else '✗ NO'}")
    print(sep)


def main():
    sep = "=" * 72
    print(sep)
    print("  TEST DE TABLA DE ORDENES")
    print(sep)
    print(f"  Compras: paso={PASO_PCT_COMPRA}%  caida_max={CAIDA_MAXIMA}%  factor={FACTOR_COMPRA}")
    print(f"  Ventas:  paso={PASO_PCT_VENTA}%  subida_max={SUBIDA_MAXIMA}%  factor={FACTOR_VENTA}")

    filas_c = tabla_compras(ATH_TEST, USDT_TEST)
    imprimir(filas_c,
             f"COMPRAS  ATH=${ATH_TEST:,.0f}  USDT=${USDT_TEST:,.2f}",
             es_btc=False)

    filas_v = tabla_ventas(PRECIO_PROM_TEST, BTC_TEST, ATL_TEST)
    p_max   = ATL_TEST * (1.0 + SUBIDA_MAXIMA / 100.0)
    imprimir(filas_v,
             f"VENTAS  PrecioProm=${PRECIO_PROM_TEST:,.0f}  ATL=${ATL_TEST:,.0f}  "
             f"Techo=${p_max:,.0f}  BTC={BTC_TEST:.4f}",
             es_btc=True)


if __name__ == "__main__":
    main()