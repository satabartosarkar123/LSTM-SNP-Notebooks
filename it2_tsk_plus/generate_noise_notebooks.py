#!/usr/bin/env python3
"""Generate IT2 TSK+ notebooks WITH Gaussian noise (0.5%, 5%, 10%).
Mirrors: Fuzzy_LSTM_SNP_With_Gaussian_Noises structure."""
import json, os

DS = {
    "dataset_A": {"name": "Dow Jones Industrial Index",
                  "csv": "../../content/monthly-closings-of-the-dowjones.csv"},
    "dataset_B": {"name": "Lake Erie Levels",
                  "csv": "../../content/monthly-lake-erie-levels-1921-19.csv"},
    "dataset_C": {"name": "Milk Production",
                  "csv": "../../content/monthly-milk-production-pounds-p.csv"},
    "dataset_D": {"name": "S&P 500",
                  "csv": "../../content/sp500.csv"},
}

def md(s):
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in s.split("\n")]}
def code(s):
    L=s.split("\n")
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":[l+"\n" for l in L[:-1]]+[L[-1]]}

C_PID = code("""\
# ============================================================
# PROCESS IDENTIFICATION
# ============================================================
import os
print(f"Process ID (PID): {os.getpid()}")""")

C_TIMER_START = code("""\
# ============================================================
# NOTEBOOK TIMER — START
# ============================================================
import time as _timer_module
_NOTEBOOK_START_TIME = _timer_module.time()
print(f"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}")""")

C_IMPORTS = code("""\
# ============================================================
# ALL IMPORTS  (pure NumPy — no sklearn / pandas-fuzzy / statsmodels)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import csv, math

np.set_printoptions(precision=6, suppress=True)
print(f"NumPy version: {np.__version__}")

def calc_rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(p))**2)))
def calc_mse(a, p):
    return float(np.mean((np.asarray(a)-np.asarray(p))**2))
def calc_nmse(a, p):
    a,p=np.asarray(a,float),np.asarray(p,float)
    mse=np.mean((a-p)**2); d=np.linalg.norm(p-np.mean(a),2)**2
    return float(mse/d) if d!=0 else float('inf')
def calc_smape(a, p):
    a,p=np.asarray(a,float),np.asarray(p,float)
    return float(100.0/len(a)*np.sum(2*np.abs(p-a)/(np.abs(a)+np.abs(p)+1e-8)))""")

def cell_load(d):
    return code(f"""\
# ============================================================
# 1. Load Time Series Data — {d['name']}
# ============================================================
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")),
                        r"{d['csv']}")
dates, values = [], []
with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) < 2: continue
        dates.append(row[0].strip().strip('"'))
        values.append(float(row[1].strip().strip('"')))
data_original = np.array(values, dtype=np.float64)
print(f"Dataset : {d['name']}")
print(f"Shape   : {{data_original.shape}}")
print(f"First 5 : {{list(zip(dates[:5], data_original[:5]))}}")""")

C_CORE = code("""\
# ============================================================
# 2. Core IT2 TSK+ Functions  (Li & Yang 2018, Eq. 4-15)
# ============================================================

def similarity_t1(A, Ap, f=8):
    '''Eq. 4-5: similarity between two weighted trapezoidal type-1
    fuzzy sets A=(a1,a2,a3,a4,w) and Ap=(a1p,a2p,a3p,a4p,wp).
    All values in normalized [0,1] domain.
    s(A,Ap) = (1 - sum|ai-aip|/4) * d * min(w,wp)/max(w,wp).'''
    a  = np.array(A[:4],  dtype=np.float64)
    ap = np.array(Ap[:4], dtype=np.float64)
    wa, wap = float(A[4]), float(Ap[4])
    shape = max(0.0, 1.0 - np.sum(np.abs(a - ap)) / 4.0)
    crisp_a  = (a[0]==a[1] and a[1]==a[2] and a[2]==a[3])
    crisp_ap = (ap[0]==ap[1] and ap[1]==ap[2] and ap[2]==ap[3])
    if crisp_a and crisp_ap:
        d = 1.0
    else:
        dist = abs(float(np.mean(a)) - float(np.mean(ap)))
        d = 1.0 - 1.0 / (1.0 + np.exp(-f * dist + 5.0))
    wr = min(wa, wap) / max(wa, wap) if max(wa, wap) > 0 else 1.0
    return float(np.clip(shape * d * wr, 0.0, 1.0))

def it2_matching(it2_set, obs_trap, f=8):
    '''Eq. 10: IT2 matching degree. Returns (s_lo, s_hi).'''
    s_lmf = similarity_t1(it2_set['lmf'], obs_trap, f)
    s_umf = similarity_t1(it2_set['umf'], obs_trap, f)
    return (min(s_lmf, s_umf), max(s_lmf, s_umf))

def firing_strength(antecedents, observations, f=8):
    '''Eq. 11: firing strength interval via min t-norm.'''
    lo_list, hi_list = [], []
    for ant, obs in zip(antecedents, observations):
        s_lo, s_hi = it2_matching(ant, obs, f)
        lo_list.append(s_lo); hi_list.append(s_hi)
    return (min(lo_list), min(hi_list))

def km_type_reduce(alphas, cons):
    '''Eq. 14-15: Karnik-Mendel type-reduction.
    Returns (c_lower, c_upper, c_defuzz).'''
    n = len(alphas)
    if n == 0: return (0., 0., 0.)
    alo = np.array([a[0] for a in alphas])
    ahi = np.array([a[1] for a in alphas])
    clo = np.array([c[0] for c in cons])
    chi = np.array([c[1] for c in cons])
    if alo.sum() + ahi.sum() < 1e-15:
        fb = float(np.mean((clo+chi)/2.0))
        return (fb, fb, fb)
    # c_lower
    ix = np.argsort(clo)
    cls,als,ahs = clo[ix],alo[ix],ahi[ix]
    L = n // 2
    for _ in range(n+2):
        num = ahs[:L+1].dot(cls[:L+1]) + als[L+1:].dot(cls[L+1:])
        den = ahs[:L+1].sum() + als[L+1:].sum()
        c_lower = float(num/den) if den>1e-15 else float(cls.mean())
        Ln = max(0, min(int(np.searchsorted(cls,c_lower,side='right'))-1, n-1))
        if Ln==L: break
        L=Ln
    # c_upper
    ix = np.argsort(chi)
    chs,als2,ahs2 = chi[ix],alo[ix],ahi[ix]
    R = n // 2
    for _ in range(n+2):
        num = als2[:R+1].dot(chs[:R+1]) + ahs2[R+1:].dot(chs[R+1:])
        den = als2[:R+1].sum() + ahs2[R+1:].sum()
        c_upper = float(num/den) if den>1e-15 else float(chs.mean())
        Rn = max(0, min(int(np.searchsorted(chs,c_upper,side='right'))-1, n-1))
        if Rn==R: break
        R=Rn
    return (c_lower, c_upper, (c_lower+c_upper)/2.0)

def build_windows(series, n):
    X, Y = [], []
    for t in range(n, len(series)):
        X.append(series[t-n:t]); Y.append(series[t])
    return np.array(X), np.array(Y)

def window_summary(w):
    return np.array([np.min(w), np.percentile(w,25),
                     np.percentile(w,75), np.max(w)])

def kmeans_lloyd(X, k, seed=42, max_iter=300, tol=1e-6):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), k, replace=False)
    C = X[idx].copy(); lab = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        D = np.array([np.linalg.norm(X-C[j],axis=1) if X.ndim>1
                       else np.abs(X-C[j]) for j in range(k)]).T
        new_lab = np.argmin(D, axis=1)
        newC = np.empty_like(C)
        for j in range(k):
            m=X[new_lab==j]
            newC[j] = m.mean(axis=0) if len(m)>0 else C[j]
        if np.sum((newC-C)**2)<tol:
            lab=new_lab; C=newC; break
        C=newC; lab=new_lab
    return lab, C

print("IT2 TSK+ core functions defined (Eq. 4-15).")""")

C_NOISE_LOOP = code("""\
# ============================================================
# 3. Gaussian Noise Experiment Loop  (0.5%, 5%, 10%)
# ============================================================
s_x = np.std(data_original)
noise_levels = [0.005, 0.05, 0.10]
N_RULES = 2; SEED = 42

all_noise_results = {}

for lam in noise_levels:
    sigma = lam * s_x
    print("\\n" + "=" * 80)
    print(f"EVALUATING NOISE LEVEL: {lam*100:.1f}% (lambda={lam}, sigma={sigma:.6f})")
    print("=" * 80 + "\\n")

    np.random.seed(42)
    noise = np.random.normal(0, sigma, size=data_original.shape)
    data = data_original + noise

    # Train / Test Split
    TRAIN_RATIO = 0.83
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    test_data  = data[split_idx:]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Normalization
    NORM_LO = float(train_data.min())
    NORM_HI = float(train_data.max())
    NORM_RNG = NORM_HI - NORM_LO if NORM_HI != NORM_LO else 1.0
    def norm_fn(x): return (np.asarray(x) - NORM_LO) / NORM_RNG

    # Sliding windows
    WINDOW_SIZE = min(100, len(train_data) // 2)
    X_train, Y_train = build_windows(train_data, WINDOW_SIZE)
    print(f"Window size: {WINDOW_SIZE}, X_train: {X_train.shape}")

    # Window summaries & clustering
    X_sum = np.array([window_summary(X_train[i]) for i in range(len(X_train))])
    labels, centroids = kmeans_lloyd(X_sum, N_RULES, seed=SEED)
    print(f"Cluster sizes: {[int(np.sum(labels==j)) for j in range(N_RULES)]}")

    # Build rules
    rules = []
    for r in range(N_RULES):
        members = X_sum[labels == r]
        targets = Y_train[labels == r]
        vals = members.flatten()
        umf = tuple(float(norm_fn(np.percentile(vals, p))) for p in [5,25,75,95])
        lmf = tuple(float(norm_fn(np.percentile(vals, p))) for p in [10,30,70,90])
        ant = {'umf': (*umf, 1.0), 'lmf': (*lmf, 0.9)}
        mu, sig = float(np.mean(targets)), float(np.std(targets))
        rules.append({'antecedents': [ant], 'consequence': (mu-sig, mu+sig)})

    for i, rl in enumerate(rules):
        c = rl['consequence']
        print(f"  Rule {i+1}: consequence=[{c[0]:.4f}, {c[1]:.4f}]")

    # Forecasting
    actuals, preds, iv_lo, iv_hi = [], [], [], []
    for t in range(split_idx, len(data)):
        if t - WINDOW_SIZE < 0: continue
        w = data[t - WINDOW_SIZE : t]
        s = window_summary(w)
        obs_trap = (*tuple(float(x) for x in norm_fn(s)), 1.0)
        a_list, c_list = [], []
        for rl in rules:
            alo, ahi = firing_strength(rl['antecedents'], [obs_trap])
            a_list.append((alo, ahi)); c_list.append(rl['consequence'])
        cL, cU, cF = km_type_reduce(a_list, c_list)
        actuals.append(data[t]); preds.append(cF)
        iv_lo.append(cL); iv_hi.append(cU)

    actuals = np.array(actuals); preds = np.array(preds)
    iv_lo = np.array(iv_lo); iv_hi = np.array(iv_hi)

    # Metrics
    r = calc_rmse(actuals, preds); m = calc_mse(actuals, preds)
    n = calc_nmse(actuals, preds); s = calc_smape(actuals, preds)
    iw = float(np.mean(iv_hi - iv_lo))

    print(f"\\n  RMSE:  {r:.6f}")
    print(f"  MSE:   {m:.6f}")
    print(f"  NMSE:  {n:.10f}")
    print(f"  sMAPE: {s:.6f}")
    print(f"  Mean interval width: {iw:.6f}")

    all_noise_results[lam] = {
        'rmse': r, 'mse': m, 'nmse': n, 'smape': s,
        'interval_width': iw,
        'actuals': actuals, 'preds': preds,
        'iv_lo': iv_lo, 'iv_hi': iv_hi,
    }

print("\\n\\nForecasting with Gaussian noise complete.")""")

C_SUMMARY = code("""\
# ============================================================
# 4. Summary Table — All Noise Levels
# ============================================================
print("=" * 82)
print(f"{'Noise %':<10s} {'RMSE':>12s} {'MSE':>14s} {'NMSE':>14s} {'sMAPE':>10s} {'IntWidth':>10s}")
print("-" * 82)
for lam in sorted(all_noise_results):
    res = all_noise_results[lam]
    print(f"{lam*100:>6.1f}%    {res['rmse']:12.6f} {res['mse']:14.6f} "
          f"{res['nmse']:14.10f} {res['smape']:10.6f} {res['interval_width']:10.6f}")
print("=" * 82)""")

C_PLOT = code("""\
# ============================================================
# 5. Plots — Predictions vs Actual per Noise Level
# ============================================================
fig, axes = plt.subplots(len(all_noise_results), 1, figsize=(14, 4*len(all_noise_results)),
                          sharex=False)
if len(all_noise_results) == 1: axes = [axes]

for ax, lam in zip(axes, sorted(all_noise_results)):
    res = all_noise_results[lam]
    t_ax = np.arange(len(res['actuals']))
    ax.plot(t_ax, res['actuals'], lw=2.0, label="Actual", color="#2c3e50")
    ax.plot(t_ax, res['preds'], lw=1.4, label="IT2 TSK+", color="#e74c3c", ls="--", alpha=0.85)
    ax.fill_between(t_ax, res['iv_lo'], res['iv_hi'],
                    alpha=0.12, color="#3498db", label="KM interval")
    ax.set_title(f"Noise {lam*100:.1f}% — RMSE={res['rmse']:.4f}, sMAPE={res['smape']:.4f}")
    ax.set_xlabel("Test step"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()""")

C_TIMER_END = code("""\
# ============================================================
# NOTEBOOK TIMER — END
# ============================================================
import time as _timer_module
_NOTEBOOK_END_TIME = _timer_module.time()
_NOTEBOOK_ELAPSED = _NOTEBOOK_END_TIME - _NOTEBOOK_START_TIME
_hours, _rem = divmod(_NOTEBOOK_ELAPSED, 3600)
_minutes, _seconds = divmod(_rem, 60)
print(f"\\nTotal notebook execution time: {int(_hours)}h {int(_minutes)}m {_seconds:.2f}s")
print(f"Total seconds: {_NOTEBOOK_ELAPSED:.2f}")""")

def make_nb(d):
    return {
        "nbformat":4,"nbformat_minor":5,
        "metadata":{
            "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python","version":"3.10.0"},
        },
        "cells":[
            md(f"# IT2 TSK+ with Gaussian Noise — {d['name']}\n"
               "### Li & Yang et al. (2018), IEEE FUZZ-IEEE 2018\n"
               "Noise levels: 0.5%, 5%, 10%. Metrics: RMSE, MSE, NMSE, sMAPE."),
            C_PID, C_TIMER_START, C_IMPORTS, cell_load(d),
            C_CORE, C_NOISE_LOOP, C_SUMMARY, C_PLOT, C_TIMER_END,
        ],
    }

sd = os.path.dirname(os.path.abspath(__file__))
for folder, d in DS.items():
    nb = make_nb(d)
    out = os.path.join(sd, folder, "it2_tsk_plus_noise.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(nb, f, indent=1)
    print(f"Created {out}")
print("\nDone.")
