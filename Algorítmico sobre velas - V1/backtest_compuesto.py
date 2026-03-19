"""
backtest_compuesto.py — Backtester de Señal Compuesta BTC/USDT
══════════════════════════════════════════════════════════════════════════════
Pipeline completo autónomo:
  1. Carga datos desde SQLite
  2. Feature Engineering (DNA de velas)
  3. Método ③: Espacio de Fase — Lyapunov local + Dimensión Fractal (Higuchi)
  4. Método ④: Entropía de Permutación Multivariada + Transfer Entropy
  5. Labeling de Local Tops/Bottoms (multi-escala, sin lookahead)
  6. Random Forest con TimeSeriesSplit → probabilidades de reversal
  7. Regresión Logística → pesos óptimos del score compuesto 0–100
  8. Backtesting con gestión de capital adaptativa
  9. Exporta JSON compatible con Graficador.py

Uso:
    python backtest_compuesto.py            # usa config.py
    python backtest_compuesto.py --fast     # salta recalculo si hay cache .npy
    python backtest_compuesto.py --nograf   # solo JSON, sin gráfico

Dependencias:
    pip install numpy pandas scipy scikit-learn matplotlib
"""

import os
import sys
import json
import time
import argparse
import sqlite3
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ─── Importar config ─────────────────────────────────────────────────────────
try:
    from config import (
        DB_PATH, RESULTS_JSON, FECHA_INICIO, FECHA_FIN, SALDO_USDT_INICIAL,
        COMMISSION_PCT, THR_BOT, THR_TOP, COOLDOWN_VELAS, VENTANA_SCORE,
        SUAVIZADO_SCORE, PCT_USDT_POR_SEÑAL, PCT_BTC_POR_SEÑAL, MAX_POSICIONES,
        USDT_RESERVA_PCT, SIZING_ADAPTATIVO, ORDEN_LIQUIDACION,
        VENTANA_DNA, CLIP_RANGE_REL, CLIP_TRADE_DENS,
        TAU_EMBEDDING, DIM_EMBEDDING, W_LYAPUNOV, K_VECINOS,
        WIN_HFD, KMAX_HFD, WIN_LYAP_NORM,
        PE_ORDER, PE_DELAY, PE_VENTANA,
        PE_PESO_CLOSE, PE_PESO_DELTA, PE_PESO_LWK, PE_PESO_TRADE, WIN_PE_NORM,
        RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES, RF_N_SPLITS_CV,
        LABEL_ORDERS, LABEL_MIN_SWING, NEUTROS_RATIO,
        LR_C, LR_N_SPLITS_CV,
        DARK_MODE, OUTPUT_PNG, DPI,
    )
    # Adaptar comisión: en config es porcentaje (0.1), internamente usamos fracción
    COMMISSION_PCT_F = COMMISSION_PCT / 100.0
except ImportError as e:
    print(f"✗ No se encontró config.py: {e}")
    print("  Crea config.py en el mismo directorio.")
    sys.exit(1)

DB_TABLE = os.path.splitext(os.path.basename(DB_PATH))[0]
CACHE_DIR = ".cache_compuesto"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILIDADES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def banner(titulo):
    sep = "═" * 62
    print(f"\n{sep}\n  {titulo}\n{sep}")

def timer(t0, label=""):
    elapsed = time.time() - t0
    print(f"  ✓ {label} — {elapsed:.1f}s")
    return time.time()

def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)

def save_cache(name, arr):
    np.save(cache_path(name), arr)

def load_cache(name):
    p = cache_path(name) + ".npy"
    return np.load(p) if os.path.exists(p) else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 1 — CARGA DE DATOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cargar_datos():
    banner("FASE 1 — CARGA DE DATOS")
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {DB_TABLE} ORDER BY timestamp ASC", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.reset_index(drop=True)
    timer(t0, f"{len(df):,} velas  [{df.datetime.min():%Y-%m-%d} → {df.datetime.max():%Y-%m-%d}]")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 2 — DNA DE VELAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_dna(df, use_cache=False):
    banner("FASE 2 — DNA DE VELAS")
    t0 = time.time()
    if use_cache:
        c = load_cache("dna")
        if c is not None:
            timer(t0, f"DNA cargado desde cache — shape {c.shape}")
            return c
    tr            = (df.high - df.low).replace(0, np.nan).ffill()
    body_ratio    = ((df.close - df.open) / tr).clip(-1, 1)
    upper_wick    = ((df.high - df[["open","close"]].max(axis=1)) / tr).clip(0, 1)
    lower_wick    = ((df[["open","close"]].min(axis=1) - df.low) / tr).clip(0, 1)
    delta_ratio   = (df.taker_buy_base_volume / (df.volume + 1e-9)).clip(0, 1)
    roll_tr       = tr.rolling(VENTANA_DNA, min_periods=1).mean()
    range_rel     = (tr / (roll_tr + 1e-9)).clip(0, CLIP_RANGE_REL)
    roll_trades   = df.trades_count.rolling(VENTANA_DNA, min_periods=1).mean()
    trade_density = (df.trades_count / (roll_trades + 1e-9)).clip(0, CLIP_TRADE_DENS)
    dna = pd.DataFrame({
        "body_ratio": body_ratio, "upper_wick": upper_wick,
        "lower_wick": lower_wick, "delta_ratio": delta_ratio,
        "range_rel":  range_rel,  "trade_density": trade_density,
    }).fillna(0).values
    save_cache("dna", dna)
    timer(t0, f"DNA shape {dna.shape}")
    return dna


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 3 — ESPACIO DE FASE: LYAPUNOV + HIGUCHI FD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_lyapunov_hfd(df, use_cache=False):
    banner("FASE 3 — ESPACIO DE FASE (Lyapunov + HFD)")
    t0   = time.time()
    N    = len(df)

    if use_cache:
        ly = load_cache("lyap")
        hf = load_cache("hfd")
        if ly is not None and hf is not None:
            timer(t0, "Lyapunov+HFD cargados desde cache")
            return ly, hf

    close  = df.close.values
    roll_m = pd.Series(close).rolling(200, min_periods=1).mean().values
    roll_s = pd.Series(close).rolling(200, min_periods=1).std().fillna(1).values
    cn     = (close - roll_m) / (roll_s + 1e-9)

    # ── Delay embedding ────────────────────────────────────────────────
    lag = TAU_EMBEDDING * (DIM_EMBEDDING - 1)
    X   = np.column_stack([cn[i:N - lag + i] for i in range(0, lag + 1, TAU_EMBEDDING)])
    M   = X.shape[0]
    print(f"  Embedding: {X.shape}  (tau={TAU_EMBEDDING}, dim={DIM_EMBEDDING})")

    # ── Lyapunov local ─────────────────────────────────────────────────
    from sklearn.neighbors import BallTree
    print(f"  Construyendo BallTree ({M:,} puntos)...")
    tree  = BallTree(X)
    lyap  = np.full(M, np.nan)
    BLOCK = 2000
    for start in range(0, M - W_LYAPUNOV, BLOCK):
        end    = min(start + BLOCK, M - W_LYAPUNOV)
        batch  = X[start:end]
        _, idxs = tree.query(batch, k=K_VECINOS + 1)
        for bi, i in enumerate(range(start, end)):
            nb    = idxs[bi, 1:]
            valid = nb[nb + W_LYAPUNOV < M]
            if len(valid) == 0:
                continue
            d0 = np.linalg.norm(X[i]          - X[valid],          axis=1) + 1e-12
            dW = np.linalg.norm(X[i+W_LYAPUNOV] - X[valid+W_LYAPUNOV], axis=1) + 1e-12
            lyap[i] = np.mean(np.log(dW / d0)) / W_LYAPUNOV
    lyap_full = np.full(N, np.nan)
    lyap_full[lag:lag + M] = lyap
    lyap_full = pd.Series(lyap_full).ffill().bfill().values

    # ── Higuchi FD ─────────────────────────────────────────────────────
    print(f"  Calculando HFD (ventana={WIN_HFD})...")
    def higuchi_fd(series):
        n  = len(series)
        L  = []
        ks = range(1, KMAX_HFD + 1)
        for k in ks:
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, n, k)
                if len(idx) < 2:
                    continue
                Lmk = (np.sum(np.abs(np.diff(series[idx]))) *
                       (n - 1) / ((len(idx) - 1) * k * k))
                Lk.append(Lmk)
            L.append(np.mean(Lk) if Lk else np.nan)
        L    = np.array(L)
        valid = ~np.isnan(L) & (L > 0)
        if valid.sum() < 2:
            return np.nan
        return np.polyfit(np.log(np.array(list(ks))[valid]),
                          np.log(L[valid]), 1)[0]

    hfd_arr = np.full(N, np.nan)
    for i in range(WIN_HFD, N):
        hfd_arr[i] = higuchi_fd(cn[i - WIN_HFD:i])
    hfd_arr = pd.Series(hfd_arr).ffill().bfill().values

    save_cache("lyap", lyap_full)
    save_cache("hfd",  hfd_arr)
    timer(t0, f"Lyap mean={np.nanmean(lyap_full):.4f}  HFD mean={np.nanmean(hfd_arr):.4f}")
    return lyap_full, hfd_arr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 4 — ENTROPÍA DE PERMUTACIÓN MULTIVARIADA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_pe_te(df, dna, use_cache=False):
    banner("FASE 4 — ENTROPÍA DE PERMUTACIÓN + TRANSFER ENTROPY")
    t0 = time.time()

    if use_cache:
        pm = load_cache("pe_matrix")
        te = load_cache("te_dc")
        if pm is not None and te is not None:
            timer(t0, "PE+TE cargados desde cache")
            return pm, te

    from itertools import permutations
    from math import log2

    N = len(df)
    perms   = list(permutations(range(PE_ORDER)))
    p2i     = {p: i for i, p in enumerate(perms)}
    nfact   = len(perms)
    step    = PE_ORDER * PE_DELAY

    def perm_indices(series):
        """Asigna a cada vela el índice de su patrón de permutación."""
        px = np.full(N, np.nan)
        for i in range(step, N):
            w   = series[i - step:i:PE_DELAY]
            px[i] = p2i[tuple(np.argsort(w).tolist())]
        return px

    def pe_series(series):
        """Entropía de permutación con ventana deslizante."""
        px  = perm_indices(series)
        pe  = np.full(N, np.nan)
        for i in range(step + PE_VENTANA, N):
            w     = px[i - PE_VENTANA:i]
            valid = w[~np.isnan(w)].astype(int)
            if len(valid) < PE_VENTANA // 2:
                continue
            cnt  = np.bincount(valid, minlength=nfact).astype(float)
            cnt /= cnt.sum()
            p    = cnt[cnt > 0]
            pe[i] = -np.sum(p * np.log2(p + 1e-12)) / log2(nfact)
        return pe

    channels = {
        "close":         df.close.values,
        "delta_ratio":   dna[:, 3],
        "lower_wick":    dna[:, 2],
        "trade_density": dna[:, 5],
    }
    pe_dict = {}
    for name, series in channels.items():
        print(f"  PE({name})...")
        pe_dict[name] = pe_series(series)

    # ── Transfer Entropy (delta → close) ─────────────────────────────
    print("  Transfer Entropy (delta → close)...")
    px_delta = perm_indices(channels["delta_ratio"])
    px_close = perm_indices(channels["close"])
    te_arr   = np.full(N, np.nan)
    WIN_TE   = 48
    for i in range(step + WIN_TE + 1, N):
        pxd = px_delta[i - WIN_TE:i]
        pxc = px_close[i - WIN_TE:i]
        pxcf = px_close[i - WIN_TE + 1:i + 1]
        m    = min(len(pxd), len(pxc), len(pxcf))
        valid = (~np.isnan(pxd[:m])) & (~np.isnan(pxc[:m])) & (~np.isnan(pxcf[:m]))
        if valid.sum() < WIN_TE // 2:
            continue
        a, b, c = pxd[:m][valid].astype(int), pxc[:m][valid].astype(int), pxcf[:m][valid].astype(int)
        j3 = np.zeros((nfact, nfact, nfact))
        j2 = np.zeros((nfact, nfact))
        for x, y, z in zip(a, b, c):
            j3[x, y, z] += 1; j2[y, z] += 1
        j3 /= (j3.sum() + 1e-12); j2 /= (j2.sum() + 1e-12)
        tv = 0.0
        for x in range(nfact):
            for y in range(nfact):
                for z in range(nfact):
                    p3 = j3[x, y, z]
                    if p3 < 1e-12: continue
                    p2  = j2[y, z]
                    pb  = j2[y, :].sum()
                    if p2 < 1e-12 or pb < 1e-12: continue
                    tv += p3 * log2((p3 * pb) / (p2 ** 2 + 1e-12) + 1e-12)
        te_arr[i] = abs(tv)

    pe_matrix = np.column_stack(list(pe_dict.values()))
    te_arr    = pd.Series(te_arr).ffill().bfill().values

    save_cache("pe_matrix", pe_matrix)
    save_cache("te_dc",     te_arr)
    timer(t0, f"PE matrix {pe_matrix.shape}  TE mean={np.nanmean(te_arr):.4f}")
    return pe_matrix, te_arr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 5 — LABELING + RANDOM FOREST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_rf_proba(df, dna, lyap, hfd, pe_matrix, te_arr, use_cache=False):
    banner("FASE 5 — LABELING + RANDOM FOREST")
    t0 = time.time()

    if use_cache:
        pb = load_cache("prob_bot_raw")
        pt = load_cache("prob_top_raw")
        lb = load_cache("labels")
        if pb is not None and pt is not None and lb is not None:
            timer(t0, "RF cargado desde cache")
            return pb, pt, lb

    from scipy.signal import argrelextrema
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    close  = df.close.values
    N      = len(close)

    # ── Labeling multi-escala ─────────────────────────────────────────
    labels = np.zeros(N, dtype=int)
    for order in LABEL_ORDERS:
        for bi in argrelextrema(close, np.less_equal, order=order)[0]:
            lo, hi = max(0, bi - order), min(N - 1, bi + order)
            if (close[lo:hi].max() - close[bi]) / close[bi] > LABEL_MIN_SWING:
                labels[bi] = 1
        for ti in argrelextrema(close, np.greater_equal, order=order)[0]:
            lo, hi = max(0, ti - order), min(N - 1, ti + order)
            if (close[ti] - close[lo:hi].min()) / close[ti] > LABEL_MIN_SWING:
                labels[ti] = -1
    y3 = np.where(labels == 1, 1, np.where(labels == -1, 2, 0))
    print(f"  Labels → neutro:{(y3==0).sum()} bottom:{(y3==1).sum()} top:{(y3==2).sum()}")

    # ── Feature matrix ────────────────────────────────────────────────
    dna_df  = pd.DataFrame(dna, columns=["body","uwk","lwk","delta","range","trade"])
    feats   = [dna]
    for w in [12, 24, 48]:
        feats.append(dna_df.rolling(w, min_periods=1).mean().values)
        feats.append(dna_df.rolling(w, min_periods=1).std().fillna(0).values)
    for arr in [lyap, hfd]:
        arr_c = pd.Series(arr).ffill().bfill().values
        feats.append(arr_c.reshape(-1, 1))
        for w in [12, 24]:
            feats.append(pd.Series(arr_c).rolling(w, min_periods=1).mean().values.reshape(-1, 1))
            feats.append(pd.Series(arr_c).rolling(w, min_periods=1).std().fillna(0).values.reshape(-1, 1))
    pe_c = pd.DataFrame(pe_matrix).ffill().bfill().values
    te_c = pd.Series(te_arr).ffill().bfill().values
    feats.append(pe_c)
    feats.append(te_c.reshape(-1, 1))
    X = np.column_stack([f if f.ndim == 2 else f.reshape(-1, 1) for f in feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Feature matrix: {X.shape}")

    # ── Balanceo y entrenamiento ──────────────────────────────────────
    START   = 200
    X_v, y_v = X[START:], y3[START:]
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_v)
    rng     = np.random.RandomState(42)
    bot_v   = np.where(y_v == 1)[0]
    top_v   = np.where(y_v == 2)[0]
    neu_v   = np.where(y_v == 0)[0]
    n_sig   = len(bot_v) + len(top_v)
    neu_s   = rng.choice(neu_v, min(n_sig * NEUTROS_RATIO, len(neu_v)), replace=False)
    idx_bal = np.sort(np.concatenate([bot_v, top_v, neu_s]))
    X_bal, y_bal = X_sc[idx_bal], y_v[idx_bal]

    tscv    = TimeSeriesSplit(n_splits=RF_N_SPLITS_CV)
    proba_full = np.zeros((len(X_v), 3))
    fold_counts = np.zeros(len(X_v))

    print(f"  Entrenando RF ({RF_N_ESTIMATORS} árboles, {RF_N_SPLITS_CV} folds)...")
    for fold, (tr, te) in enumerate(tscv.split(X_bal)):
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        rf.fit(X_bal[tr], y_bal[tr])
        # Predecir sobre todo el dataset de validación
        p = rf.predict_proba(X_sc)
        proba_full += p
        fold_counts += 1
        print(f"    Fold {fold+1}/{RF_N_SPLITS_CV} ✓")

    proba_avg  = proba_full / RF_N_SPLITS_CV
    prob_bot_r = np.concatenate([np.full(START, 0.1), proba_avg[:, 1]])
    prob_top_r = np.concatenate([np.full(START, 0.1), proba_avg[:, 2]])
    labels_full = np.concatenate([np.zeros(START, dtype=int), y3[START:]])

    save_cache("prob_bot_raw", prob_bot_r)
    save_cache("prob_top_raw", prob_top_r)
    save_cache("labels",       labels_full)
    timer(t0, "RF completo")
    return prob_bot_r, prob_top_r, labels_full


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 6 — SCORE COMPUESTO ADAPTATIVO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_score_compuesto(df, dna, lyap, pe_matrix, te_arr,
                              prob_bot_r, prob_top_r, labels, use_cache=False):
    banner("FASE 6 — SCORE COMPUESTO ADAPTATIVO")
    t0 = time.time()

    if use_cache:
        sb = load_cache("score_bot")
        st = load_cache("score_top")
        if sb is not None and st is not None:
            timer(t0, "Scores cargados desde cache")
            return sb, st

    from scipy.stats import percentileofscore
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from scipy.ndimage import uniform_filter1d

    N = len(df)

    # ── Componentes ──────────────────────────────────────────────────
    prob_bot_sm = pd.Series(prob_bot_r).rolling(SUAVIZADO_SCORE, min_periods=1).mean().values
    prob_top_sm = pd.Series(prob_top_r).rolling(SUAVIZADO_SCORE, min_periods=1).mean().values

    # Lyapunov adaptativo
    lyap_c    = pd.Series(lyap).ffill().bfill().values
    lyap_pct  = np.full(N, 0.5)
    for i in range(WIN_LYAP_NORM, N):
        lyap_pct[i] = percentileofscore(lyap_c[i - WIN_LYAP_NORM:i], lyap_c[i]) / 100.0
    lyap_pct  = pd.Series(lyap_pct).ffill().bfill().values
    lyap_rev  = np.abs(lyap_pct - 0.5) * 2
    lyap_bot  = lyap_rev * (1 - lyap_pct)
    lyap_top  = lyap_rev * lyap_pct

    # PE tensión
    pe_cl = pd.Series(pe_matrix[:, 0]).ffill().bfill().rolling(12, min_periods=1).mean().values
    pe_dl = pd.Series(pe_matrix[:, 1]).ffill().bfill().rolling(12, min_periods=1).mean().values
    pe_lw = pd.Series(pe_matrix[:, 2]).ffill().bfill().rolling(12, min_periods=1).mean().values
    pe_td = pd.Series(pe_matrix[:, 3]).ffill().bfill().rolling(12, min_periods=1).mean().values
    pe_comp = (PE_PESO_CLOSE * pe_cl + PE_PESO_DELTA * pe_dl +
               PE_PESO_LWK   * pe_lw + PE_PESO_TRADE * pe_td)
    pe_ten  = np.full(N, 0.5)
    for i in range(WIN_PE_NORM, N):
        pe_ten[i] = 1.0 - percentileofscore(pe_comp[i - WIN_PE_NORM:i], pe_comp[i]) / 100.0
    pe_ten = pd.Series(pe_ten).ffill().bfill().values

    # Delta divergencia
    delta  = pd.Series(dna[:, 3]).rolling(12, min_periods=1).mean().values
    p_chg  = pd.Series(df.close.values).pct_change(12).fillna(0).values
    d_chg  = pd.Series(delta).diff(12).fillna(0).values
    def rzsc(arr, w=200):
        s = pd.Series(arr)
        m = s.rolling(w, min_periods=1).mean()
        st_ = s.rolling(w, min_periods=1).std().fillna(1).replace(0, 1)
        return ((s - m) / st_).values
    pz = rzsc(p_chg); dz = rzsc(d_chg)
    div_bot_raw = np.clip(-pz * dz, 0, None)
    div_top_raw = np.clip( pz * dz, 0, None)
    def rp(arr, w=500):
        out = np.full(len(arr), 0.5)
        for i in range(w, len(arr)):
            out[i] = percentileofscore(arr[i-w:i], arr[i]) / 100.0
        return out
    div_bot = rp(div_bot_raw); div_top = rp(div_top_raw)

    # Morfología
    body   = dna[:, 0]
    b6  = pd.Series(body).rolling(6,  min_periods=1).mean().values
    b24 = pd.Series(body).rolling(24, min_periods=1).mean().values
    morph_bot = np.where(b24 < -0.05, np.clip(b6 - b24, 0, None), 0.0)
    morph_top = np.where(b24 >  0.05, np.clip(b24 - b6, 0, None), 0.0)
    morph_bot = rp(morph_bot); morph_top = rp(morph_top)

    # ── Regresión Logística para pesos ───────────────────────────────
    inter_bot = prob_bot_sm * pe_ten
    inter_top = prob_top_sm * pe_ten
    Xb = np.column_stack([prob_bot_sm, lyap_bot, pe_ten, div_bot, morph_bot,
                          inter_bot, lyap_bot * pe_ten,
                          prob_bot_sm * div_bot, pe_ten * div_bot])
    Xt = np.column_stack([prob_top_sm, lyap_top, pe_ten, div_top, morph_top,
                          inter_top, lyap_top * pe_ten,
                          prob_top_sm * div_top, pe_ten * div_top])
    y3 = np.where(labels == 1, 1, np.where(labels == -1, 2, 0))
    yb = (y3 == 1).astype(int); yt = (y3 == 2).astype(int)

    START = 600
    sc_b = StandardScaler().fit(Xb[START:]); sc_t = StandardScaler().fit(Xt[START:])
    Xb_sc = sc_b.transform(Xb[START:]); Xt_sc = sc_t.transform(Xt[START:])
    tscv  = TimeSeriesSplit(n_splits=LR_N_SPLITS_CV)

    def opt_weights(Xsc, yv):
        ws = []
        n_features = Xsc.shape[1]
        for fold_idx, (tr, te) in enumerate(tscv.split(Xsc)):
            # Verificar que el fold de entrenamiento tiene al menos 2 clases
            clases_presentes = np.unique(yv[tr])
            if len(clases_presentes) < 2:
                print(f"    ⚠ Fold {fold_idx+1} omitido — solo clase {clases_presentes} en train")
                continue
            # Verificar mínimo de muestras positivas
            n_positivos = (yv[tr] == 1).sum()
            if n_positivos < 3:
                print(f"    ⚠ Fold {fold_idx+1} omitido — solo {n_positivos} muestras positivas")
                continue
            try:
                lr = LogisticRegression(C=LR_C, class_weight="balanced",
                                        max_iter=1000, random_state=42)
                lr.fit(Xsc[tr], yv[tr])
                ws.append(lr.coef_[0])
            except Exception as e:
                print(f"    ⚠ Fold {fold_idx+1} error: {e}")
                continue
        # Si ningún fold fue válido, devolver pesos uniformes
        if not ws:
            print("    ⚠ Ningún fold válido — usando pesos uniformes")
            return np.ones(n_features) / n_features
        w = np.mean(ws, axis=0)
        w = np.clip(w, 0, None)
        s = w.sum()
        return w / s if s > 0 else np.ones(n_features) / n_features

    print("  Optimizando pesos BOTTOM..."); wb = opt_weights(Xb_sc, yb[START:])
    print("  Optimizando pesos TOP...");    wt = opt_weights(Xt_sc, yt[START:])

    # ── Score adaptativo ─────────────────────────────────────────────
    raw_bot = uniform_filter1d(Xb @ wb, size=SUAVIZADO_SCORE)
    raw_top = uniform_filter1d(Xt @ wt, size=SUAVIZADO_SCORE)

    def adaptive_score(raw, w=VENTANA_SCORE):
        out = np.full(len(raw), 50.0)
        for i in range(w, len(raw)):
            p10, p90 = np.percentile(raw[i-w:i], 10), np.percentile(raw[i-w:i], 90)
            out[i] = 50.0 if p90 == p10 else float(np.clip((raw[i]-p10)/(p90-p10)*100, 0, 100))
        return out

    score_bot = adaptive_score(raw_bot)
    score_top = adaptive_score(raw_top)
    save_cache("score_bot", score_bot)
    save_cache("score_top", score_top)
    timer(t0, f"Scores  bot mean={score_bot.mean():.1f}  top mean={score_top.mean():.1f}")
    return score_bot, score_top


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 7 — DEDUPLICACIÓN DE SEÑALES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def dedup_eventos(score, threshold, cooldown=12):
    """Agrupa alertas continuas en eventos y retorna el índice del pico de cada uno."""
    N, in_alert, below, start, events = len(score), False, 0, 0, []
    for i in range(N):
        if score[i] >= threshold:
            if not in_alert:
                in_alert = True; start = i; below = 0
            else:
                below = 0
        else:
            if in_alert:
                below += 1
                if below >= cooldown:
                    seg = score[start:i - cooldown + 1]
                    events.append(start + int(np.argmax(seg)))
                    in_alert = False; below = 0
    if in_alert:
        events.append(start + int(np.argmax(score[start:])))
    return np.array(events)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE 8 — BACKTESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ejecutar_backtest(df, score_bot, score_top):
    banner("FASE 7 — BACKTESTING")
    t0 = time.time()

    close = df.close.values
    dt    = pd.to_datetime(df.datetime)
    N     = len(close)

    # Aplicar rango de fechas
    fi = pd.Timestamp(FECHA_INICIO) if FECHA_INICIO else dt.iloc[0]
    ff = pd.Timestamp(FECHA_FIN)    if FECHA_FIN    else dt.iloc[-1]
    start_idx = int(np.searchsorted(dt.values, fi.to_datetime64()))
    end_idx   = int(np.searchsorted(dt.values, ff.to_datetime64(), side="right")) - 1
    end_idx   = min(end_idx, N - 1)
    print(f"  Rango: {dt.iloc[start_idx]:%Y-%m-%d} → {dt.iloc[end_idx]:%Y-%m-%d}")
    print(f"  Precio inicio: ${close[start_idx]:,.0f}  →  fin: ${close[end_idx]:,.0f}")

    # Suavizar scores
    sb_sm = pd.Series(score_bot).rolling(SUAVIZADO_SCORE, min_periods=1).mean().values
    st_sm = pd.Series(score_top).rolling(SUAVIZADO_SCORE, min_periods=1).mean().values

    # Eventos dedupados en el rango
    sb_range = sb_sm[start_idx:end_idx + 1]
    st_range = st_sm[start_idx:end_idx + 1]
    ev_b = dedup_eventos(sb_range, THR_BOT, COOLDOWN_VELAS) + start_idx
    ev_t = dedup_eventos(st_range, THR_TOP, COOLDOWN_VELAS) + start_idx
    ev_b_set = set(ev_b.tolist()); ev_t_set = set(ev_t.tolist())

    eventos = sorted([(i, "BUY") for i in ev_b_set] + [(i, "SELL") for i in ev_t_set])
    print(f"  Señales BUY: {len(ev_b)}  SELL: {len(ev_t)}")

    # ── Estado ───────────────────────────────────────────────────────
    usdt       = float(SALDO_USDT_INICIAL)
    posiciones = []   # [{precio: float, btc: float}]
    trade_hist = []
    n_comp = n_venta = n_ign = 0
    ign_mot = {}
    last_bot = start_idx - COOLDOWN_VELAS - 1
    last_top = start_idx - COOLDOWN_VELAS - 1

    def btc_pos_total():  return sum(p["btc"] for p in posiciones)
    def pp():
        t = btc_pos_total()
        return sum(p["btc"] * p["precio"] for p in posiciones) / t if t > 0 else 0.0

    def add(i, tipo, precio, ign=False, motivo=None, **kw):
        t = {
            "datetime":                   dt.iloc[i].isoformat(),
            "type":                       tipo,
            "price":                      float(precio),
            "score_bot":                  float(sb_sm[i]),
            "score_top":                  float(st_sm[i]),
            "usdt_balance":               float(usdt),
            "btc_balance":                0.0,
            "btc_en_posiciones":          float(btc_pos_total()),
            "positions_count":            len(posiciones),
            "precio_promedio_posiciones": float(pp()),
            "ignorado":                   ign,
            "motivo_ignorado":            motivo,
            "usdt_spent": None, "btc_bought": None, "commission_usdt": None,
            "btc_sold": None, "btc_accumulated": None, "usdt_received": None,
            "ganancia_usdt": None, "pct_capital_usado": None,
        }
        t.update(kw)
        trade_hist.append(t)

    def ordenar_posiciones_venta(pos, precio_actual):
        if ORDEN_LIQUIDACION == "fifo":
            return list(pos)
        elif ORDEN_LIQUIDACION == "lifo":
            return list(reversed(pos))
        elif ORDEN_LIQUIDACION == "mejor_pnl":
            return sorted(pos, key=lambda x: precio_actual - x["precio"], reverse=True)
        elif ORDEN_LIQUIDACION == "peor_pnl":
            return sorted(pos, key=lambda x: precio_actual - x["precio"])
        return list(pos)

    # ── Loop ──────────────────────────────────────────────────────────
    for (i, tipo) in eventos:
        precio = close[i]

        if tipo == "BUY":
            if (i - last_bot) < COOLDOWN_VELAS:
                m = f"cooldown({i-last_bot}h<{COOLDOWN_VELAS}h)"
                add(i,"BUY",precio,ign=True,motivo=m); ign_mot[m[:40]] = ign_mot.get(m[:40],0)+1; n_ign+=1; continue
            if len(posiciones) >= MAX_POSICIONES:
                m = f"max_posiciones({MAX_POSICIONES})"
                add(i,"BUY",precio,ign=True,motivo=m); ign_mot[m] = ign_mot.get(m,0)+1; n_ign+=1; continue

            if SIZING_ADAPTATIVO:
                intens = min((sb_sm[i] - THR_BOT) / max(100 - THR_BOT, 1), 1.0)
                pct    = PCT_USDT_POR_SEÑAL * (0.75 + 0.25 * intens)
            else:
                pct = PCT_USDT_POR_SEÑAL

            reserva   = SALDO_USDT_INICIAL * USDT_RESERVA_PCT
            disp      = max(0.0, usdt - reserva)
            gasto     = disp * pct
            if gasto < 5.0:
                m = f"usdt_bajo(${usdt:.1f})"
                add(i,"BUY",precio,ign=True,motivo=m); ign_mot["usdt_insuficiente"] = ign_mot.get("usdt_insuficiente",0)+1; n_ign+=1; continue

            comision    = gasto * COMMISSION_PCT_F
            btc_comprado = (gasto - comision) / precio
            usdt        -= gasto
            posiciones.append({"precio": precio, "btc": btc_comprado})
            last_bot     = i; n_comp += 1
            add(i, "BUY", precio, usdt_spent=gasto, btc_bought=btc_comprado,
                commission_usdt=comision, pct_capital_usado=pct*100)

        else:  # SELL
            if (i - last_top) < COOLDOWN_VELAS:
                m = f"cooldown({i-last_top}h<{COOLDOWN_VELAS}h)"
                add(i,"SELL",precio,ign=True,motivo=m); ign_mot[m[:40]] = ign_mot.get(m[:40],0)+1; n_ign+=1; continue
            if not posiciones:
                m = "sin_posiciones"
                add(i,"SELL",precio,ign=True,motivo=m); ign_mot[m] = ign_mot.get(m,0)+1; n_ign+=1; continue

            if SIZING_ADAPTATIVO:
                intens = min((st_sm[i] - THR_TOP) / max(100 - THR_TOP, 1), 1.0)
                pct    = PCT_BTC_POR_SEÑAL * (0.75 + 0.25 * intens)
            else:
                pct = PCT_BTC_POR_SEÑAL

            total_btc   = btc_pos_total()
            btc_vender  = total_btc * pct
            sorted_pos  = ordenar_posiciones_venta(posiciones, precio)
            restante     = btc_vender
            nuevas_pos   = []
            btc_vend     = 0.0
            costo_base   = 0.0

            for pos in sorted_pos:
                if restante <= 0:
                    nuevas_pos.append(pos); continue
                if pos["btc"] <= restante:
                    btc_vend   += pos["btc"]; costo_base += pos["btc"] * pos["precio"]; restante -= pos["btc"]
                else:
                    btc_vend   += restante;   costo_base += restante * pos["precio"]
                    nuevas_pos.append({"precio": pos["precio"], "btc": pos["btc"] - restante}); restante = 0

            posiciones  = nuevas_pos
            usdt_bruto  = btc_vend * precio
            comision    = usdt_bruto * COMMISSION_PCT_F
            usdt_recib  = usdt_bruto - comision
            ganancia    = usdt_recib - costo_base
            usdt       += usdt_recib
            last_top    = i; n_venta += 1
            add(i, "SELL", precio, btc_sold=btc_vend, usdt_received=usdt_recib,
                ganancia_usdt=ganancia, commission_usdt=comision,
                btc_accumulated=0.0, pct_capital_usado=pct*100)

    # ── Métricas finales ──────────────────────────────────────────────
    precio_fin  = close[end_idx]
    btc_pos_fin = btc_pos_total()
    port_fin    = usdt + btc_pos_fin * precio_fin
    pnl_pct     = (port_fin / SALDO_USDT_INICIAL - 1) * 100

    bh_btc  = SALDO_USDT_INICIAL * (1 - COMMISSION_PCT_F) / close[start_idx]
    bh_fin  = bh_btc * precio_fin
    bh_pnl  = (bh_fin / SALDO_USDT_INICIAL - 1) * 100

    ganancias = [t["ganancia_usdt"] for t in trade_hist
                 if t["type"]=="SELL" and not t["ignorado"] and t["ganancia_usdt"] is not None]
    wins  = [g for g in ganancias if g > 0]
    loses = [g for g in ganancias if g < 0]
    wr    = len(wins) / len(ganancias) * 100 if ganancias else 0
    pf    = sum(wins) / abs(sum(loses)) if loses else float("inf")

    # ── Consola ────────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Capital inicial      : ${SALDO_USDT_INICIAL:>10,.2f}")
    print(f"  Portfolio final      : ${port_fin:>10,.2f}  ({pnl_pct:+.2f}%)")
    print(f"  Buy & Hold           : ${bh_fin:>10,.2f}  ({bh_pnl:+.2f}%)")
    print(f"  Alpha vs B&H         :             ({pnl_pct - bh_pnl:+.2f}%)")
    print(f"  USDT final libre     : ${usdt:>10,.2f}")
    print(f"  BTC en posiciones    :  {btc_pos_fin:.6f} ₿  (${btc_pos_fin*precio_fin:,.0f})")
    print(f"\n  Compras              : {n_comp}  |  Ventas: {n_venta}  |  Ignorados: {n_ign}")
    if ganancias:
        print(f"  Win Rate             : {wr:.1f}%  ({len(wins)}W/{len(loses)}L)")
        print(f"  Profit Factor        : {pf:.2f}")
        print(f"  Ganancia total ventas: ${sum(ganancias):,.2f}")
    print(f"{sep}")

    ejec_buy  = [t for t in trade_hist if t["type"]=="BUY"  and not t["ignorado"]]
    ejec_sell = [t for t in trade_hist if t["type"]=="SELL" and not t["ignorado"]]

    summary = {
        "estrategia":               "Señal Compuesta DNA+Lyapunov+PE+Delta",
        "fecha_inicio":             dt.iloc[start_idx].isoformat()[:10],
        "fecha_fin":                dt.iloc[end_idx].isoformat()[:10],
        "saldo_inicial_usdt":       SALDO_USDT_INICIAL,
        "usdt_balance_final":       float(usdt),
        "btc_balance_final":        0.0,
        "btc_acumulado_total":      0.0,
        "btc_en_posiciones_final":  float(btc_pos_fin),
        "precio_promedio_final":    float(pp()),
        "portfolio_value_final":    float(port_fin),
        "pnl_pct":                  float(pnl_pct),
        "buy_hold_pnl_pct":         float(bh_pnl),
        "alpha_vs_bh":              float(pnl_pct - bh_pnl),
        "precio_min_comprado":      float(min((t["price"] for t in ejec_buy),  default=0)),
        "precio_max_vendido":       float(max((t["price"] for t in ejec_sell), default=0)),
        "atl_final":                float(close[start_idx:end_idx+1].min()),
        "ath_proyectado_final":     float(close[start_idx:end_idx+1].max()),
        "total_trades_ejecutados":  n_comp + n_venta,
        "total_compras":            n_comp,
        "total_ventas":             n_venta,
        "total_ignorados":          n_ign,
        "ordenes_canceladas":       0,
        "ignorados_por_motivo":     ign_mot,
        "positions_count_final":    len(posiciones),
        "usdt_reserva_aplicada":    SALDO_USDT_INICIAL * USDT_RESERVA_PCT,
        "umbral_filtro":            float(THR_BOT),
        "win_rate_pct":             float(wr),
        "profit_factor":            float(pf) if pf != float("inf") else 999.0,
        "ganancia_total_ventas":    float(sum(ganancias)) if ganancias else 0.0,
        "parametros": {
            "thr_bot":              THR_BOT,    "thr_top":       THR_TOP,
            "cooldown_velas":       COOLDOWN_VELAS,
            "pct_usdt_por_senal":   PCT_USDT_POR_SEÑAL * 100,
            "pct_btc_por_senal":    PCT_BTC_POR_SEÑAL  * 100,
            "max_posiciones":       MAX_POSICIONES,
            "usdt_reserva_pct":     USDT_RESERVA_PCT   * 100,
            "commission_pct":       COMMISSION_PCT,
            "sizing_adaptativo":    SIZING_ADAPTATIVO,
            "orden_liquidacion":    ORDEN_LIQUIDACION,
            "ventana_score":        VENTANA_SCORE,
            "N":                    "adaptativo_score_compuesto",
            "rsi_length":           "N/A",
            "ath_caida_maxima":     "N/A",
            "atl_subida_maxima":    "N/A",
            "factor_caida":         "N/A",
            "factor_subida":        "N/A",
            "guardia_compra":       True,
            "guardia_venta":        True,
        },
    }

    timer(t0, "Backtesting completo")
    return {"summary": summary, "trade_history": trade_hist}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Backtester Señal Compuesta BTC/USDT")
    parser.add_argument("--fast",    action="store_true",
                        help="Usa cache .npy si existe (evita recalcular fases pesadas)")
    parser.add_argument("--nograf",  action="store_true",
                        help="Solo genera JSON, sin gráfico")
    parser.add_argument("--nocache", action="store_true",
                        help="Ignora y borra el cache existente")
    args = parser.parse_args()

    if args.nocache and os.path.exists(CACHE_DIR):
        import shutil; shutil.rmtree(CACHE_DIR)
        print(f"✓ Cache eliminado")

    use_cache = args.fast

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   BACKTESTER — SEÑAL COMPUESTA BTC/USDT                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  DB        : {DB_PATH}")
    print(f"  Rango     : {FECHA_INICIO or 'inicio'} → {FECHA_FIN or 'fin'}")
    print(f"  Capital   : ${SALDO_USDT_INICIAL:,}  |  Comisión: {COMMISSION_PCT}%")
    print(f"  Umbral    : bot={THR_BOT}  top={THR_TOP}  cooldown={COOLDOWN_VELAS}h")
    print(f"  Cache     : {'activado (--fast)' if use_cache else 'desactivado'}")

    t_total = time.time()

    df          = cargar_datos()
    dna         = calcular_dna(df, use_cache)
    lyap, hfd   = calcular_lyapunov_hfd(df, use_cache)
    pe_mat, te  = calcular_pe_te(df, dna, use_cache)
    pb_r, pt_r, labels = calcular_rf_proba(df, dna, lyap, hfd, pe_mat, te, use_cache)
    score_bot, score_top = calcular_score_compuesto(
        df, dna, lyap, pe_mat, te, pb_r, pt_r, labels, use_cache
    )
    results = ejecutar_backtest(df, score_bot, score_top)

    # ── Guardar JSON ─────────────────────────────────────────────────
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ JSON guardado: {RESULTS_JSON}  ({len(results['trade_history'])} registros)")

    elapsed = time.time() - t_total
    print(f"✓ Pipeline completo en {elapsed/60:.1f} min")

    # ── Gráfico ───────────────────────────────────────────────────────
    if not args.nograf:
        try:
            import Graficador as G
            import matplotlib; matplotlib.use("Agg" if not sys.stdout.isatty() else "TkAgg")
            G.DB_PATH            = DB_PATH
            G.RESULTS_JSON       = RESULTS_JSON
            G.FECHA_INICIO       = FECHA_INICIO
            G.FECHA_FIN          = FECHA_FIN
            G.SALDO_USDT_INICIAL = SALDO_USDT_INICIAL
            G.OUTPUT_PNG        = OUTPUT_PNG
            G.DPI                = DPI
            g = G.Graficador(dark_mode=DARK_MODE)
            g.cargar_precios(DB_PATH, DB_TABLE)
            g.cargar_resultados(RESULTS_JSON)
            g.analisis_consola()
            print(f"\nGenerando gráfico → {OUTPUT_PNG}...")
            g.crear_grafico(OUTPUT_PNG)
        except ImportError:
            print("\n⚠  Graficador.py no encontrado — solo se generó el JSON.")
        except Exception as e:
            print(f"\n⚠  Error en graficador: {e}")


if __name__ == "__main__":
    main()