#!/usr/bin/env python3
"""Generate 4 IT2 TSK+ notebooks — Li & Yang (2018) IEEE FUZZ-IEEE."""
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

C0 = code("""\
# ============================================================
# PROCESS IDENTIFICATION
# ============================================================
import os
print(f"Process ID (PID): {os.getpid()}")""")

C1 = code("""\
# ============================================================
# NOTEBOOK TIMER — START
# ============================================================
import time as _timer_module
_NOTEBOOK_START_TIME = _timer_module.time()
print(f"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}")""")

C2 = code("""\
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
data = np.array(values, dtype=np.float64)
print(f"Dataset : {d['name']}")
print(f"Shape   : {{data.shape}}")
print(f"First 5 : {{list(zip(dates[:5], data[:5]))}}")""")

C3 = code("""\
# ============================================================
# 2. Train / Test Split  (83 % / 17 %)
# ============================================================
TRAIN_RATIO = 0.83
split_idx = int(len(data) * TRAIN_RATIO)
train_data = data[:split_idx]
test_data  = data[split_idx:]
print(f"Total : {len(data)}")
print(f"Train : {len(train_data)}  (idx 0..{split_idx-1})")
print(f"Test  : {len(test_data)}  (idx {split_idx}..{len(data)-1})")""")

C4 = code("""\
# ============================================================
# 3. Sliding Window Construction
# ============================================================
WINDOW_SIZE = min(100, len(train_data) // 2)
print(f"Window size (n): {WINDOW_SIZE}")

def build_windows(series, n):
    X, Y = [], []
    for t in range(n, len(series)):
        X.append(series[t-n:t]); Y.append(series[t])
    return np.array(X), np.array(Y)

X_train, Y_train = build_windows(train_data, WINDOW_SIZE)
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")""")

# ── Cell 5: IT2 Fuzzy Set Utilities ──────────────────────────
C5 = code("""\
# ============================================================
# 4. IT2 Fuzzy Set Utilities
#    Implements Eq. 4, 5, 10, 11 of Li & Yang (2018).
# ============================================================

# Normalization to [0,1] domain — paper assumes normalized domain
NORM_LO = float(train_data.min())
NORM_HI = float(train_data.max())
NORM_RNG = NORM_HI - NORM_LO if NORM_HI != NORM_LO else 1.0
def norm(x): return (np.asarray(x) - NORM_LO) / NORM_RNG

def similarity_t1(A, Ap, f=8):
    '''Eq. 4-5: similarity between two weighted trapezoidal type-1
    fuzzy sets A=(a1,a2,a3,a4,w) and Ap=(a1p,a2p,a3p,a4p,wp).
    All values must be in the normalized [0,1] domain.

    s(A,Ap) = (1 - sum|ai-aip|/4) * d * min(w,wp)/max(w,wp)

    Distance factor d (Eq. 5):
      d = 1                                if both A and Ap are crisp
      d = 1 - 1/(1 + exp(-f*||A,Ap||+5))   otherwise
    ||A,Ap|| = Euclidean distance between representative values (centroids).
    Sensitivity factor f=8 per paper.'''
    a  = np.array(A[:4],  dtype=np.float64)
    ap = np.array(Ap[:4], dtype=np.float64)
    wa, wap = float(A[4]), float(Ap[4])

    # Shape similarity: Eq. 4 numerator part
    shape = max(0.0, 1.0 - np.sum(np.abs(a - ap)) / 4.0)

    # Distance factor: Eq. 5
    crisp_a  = (a[0]==a[1] and a[1]==a[2] and a[2]==a[3])
    crisp_ap = (ap[0]==ap[1] and ap[1]==ap[2] and ap[2]==ap[3])
    if crisp_a and crisp_ap:
        d = 1.0
    else:
        rep_a  = float(np.mean(a))   # centroid as representative value
        rep_ap = float(np.mean(ap))
        dist = abs(rep_a - rep_ap)
        d = 1.0 - 1.0 / (1.0 + np.exp(-f * dist + 5.0))

    # Weight ratio
    wr = min(wa, wap) / max(wa, wap) if max(wa, wap) > 0 else 1.0
    return float(np.clip(shape * d * wr, 0.0, 1.0))

def it2_matching(it2_set, obs_trap, f=8):
    '''Eq. 10: IT2 matching degree between an IT2 antecedent
    and a type-1 observation (trapezoidal or crisp singleton).

    s_tilde(A_tilde, obs) = [s(LMF, obs), s(UMF, obs)]

    obs_trap = (o1,o2,o3,o4,w) — for crisp x use (x,x,x,x,1).
    Returns (s_lo, s_hi) interval.'''
    s_lmf = similarity_t1(it2_set['lmf'], obs_trap, f)
    s_umf = similarity_t1(it2_set['umf'], obs_trap, f)
    return (min(s_lmf, s_umf), max(s_lmf, s_umf))

def firing_strength(antecedents, observations, f=8):
    '''Eq. 11: firing strength interval for rule with k antecedents.
    Uses min t-norm (meet operator) over matching degree intervals.

    antecedents  : list of IT2 set dicts (one per input variable)
    observations : list of type-1 trapezoid tuples (one per input variable)

    alpha_tilde = [min(s_lo_1,...,s_lo_k), min(s_hi_1,...,s_hi_k)]'''
    lo_list, hi_list = [], []
    for ant, obs in zip(antecedents, observations):
        s_lo, s_hi = it2_matching(ant, obs, f)
        lo_list.append(s_lo); hi_list.append(s_hi)
    return (min(lo_list), min(hi_list))

print("IT2 utilities defined:")
print("  similarity_t1  (Eq. 4-5)")
print("  it2_matching   (Eq. 10)")
print("  firing_strength (Eq. 11)")
print(f"Normalization: [{NORM_LO:.4f}, {NORM_HI:.4f}]")""")

# ── Cell 6: Rule Base Construction ───────────────────────────
C6 = code("""\
# ============================================================
# 5. Rule Base Construction
#    Lloyd's k-means (pure NumPy), IT2 antecedent fitting,
#    0-order TSK consequence intervals.
# ============================================================
N_RULES = 2
SEED = 42

def kmeans_lloyd(X, k, seed=42, max_iter=300, tol=1e-6):
    '''Pure NumPy Lloyd k-means. Returns (labels, centroids).'''
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.choice(n, k, replace=False)
    C = X[idx].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        D = np.array([np.linalg.norm(X-C[j], axis=1) if X.ndim>1
                       else np.abs(X-C[j]) for j in range(k)]).T
        new_lab = np.argmin(D, axis=1)
        newC = np.empty_like(C)
        for j in range(k):
            m = X[new_lab==j]
            newC[j] = m.mean(axis=0) if len(m)>0 else C[j]
        if np.sum((newC-C)**2) < tol:
            lab=new_lab; C=newC; break
        C=newC; lab=new_lab
    return lab, C

def window_summary(w):
    '''Compress lag window to 4-point trapezoidal summary:
    [min, p25, p75, max].'''
    return np.array([np.min(w), np.percentile(w,25),
                     np.percentile(w,75), np.max(w)])

# Summarise all training windows
X_sum = np.array([window_summary(X_train[i]) for i in range(len(X_train))])
print(f"Window summaries shape: {X_sum.shape}")

# Cluster into N_RULES groups
labels, centroids = kmeans_lloyd(X_sum, N_RULES, seed=SEED)
print(f"Cluster sizes: {[int(np.sum(labels==j)) for j in range(N_RULES)]}")

# Build rules
rules = []
for r in range(N_RULES):
    members = X_sum[labels == r]
    targets = Y_train[labels == r]
    vals = members.flatten()

    # UMF trapezoid in normalized domain: [p5, p25, p75, p95], w=1.0
    umf = tuple(float(norm(np.percentile(vals, p))) for p in [5,25,75,95])
    # LMF contracted: [p10, p30, p70, p90], w=0.9
    lmf = tuple(float(norm(np.percentile(vals, p))) for p in [10,30,70,90])

    ant = {'umf': (*umf, 1.0), 'lmf': (*lmf, 0.9)}

    # 0-order consequence: crisp interval [mean-std, mean+std]  (Eq. 12)
    mu, sigma = float(np.mean(targets)), float(np.std(targets))
    con = (mu - sigma, mu + sigma)

    rules.append({'antecedents': [ant], 'consequence': con})

print(f"\\n{'='*60}")
print(f"Rule Base Summary ({N_RULES} rules)")
print(f"{'='*60}")
for i, rl in enumerate(rules):
    a = rl['antecedents'][0]
    c = rl['consequence']
    print(f"\\nRule {i+1}:")
    print(f"  UMF (norm): {a['umf']}")
    print(f"  LMF (norm): {a['lmf']}")
    print(f"  Consequence: [{c[0]:.4f}, {c[1]:.4f}]")""")

# ── Cell 7: KM Algorithm ────────────────────────────────────
C7 = code("""\
# ============================================================
# 6. Karnik-Mendel Type-Reduction  (Eq. 14-15)
# ============================================================
def km_type_reduce(alphas, cons):
    '''KM algorithm for IT2 type-reduction.
    Eq. 14: compute c_lower and c_upper via iterative switch-point search.
    Eq. 15: c = (c_lower + c_upper) / 2.

    alphas : list of (alpha_lo, alpha_hi) per rule
    cons   : list of (c_lo, c_hi) per rule

    For c_lower (minimise):
      Sort by c_lo ascending.
      Rules i<=L use alpha_hi (upper), rules i>L use alpha_lo (lower).

    For c_upper (maximise):
      Sort by c_hi ascending.
      Rules i<=R use alpha_lo (lower), rules i>R use alpha_hi (upper).

    Returns (c_lower, c_upper, c_defuzz).'''
    n = len(alphas)
    if n == 0: return (0., 0., 0.)
    alo = np.array([a[0] for a in alphas])
    ahi = np.array([a[1] for a in alphas])
    clo = np.array([c[0] for c in cons])
    chi = np.array([c[1] for c in cons])

    # Guard: all zero firing
    if alo.sum() + ahi.sum() < 1e-15:
        fb = float(np.mean((clo+chi)/2.0))
        return (fb, fb, fb)

    # ── c_lower via KM (Eq. 14 top) ──
    ix = np.argsort(clo)
    clo_s = clo[ix]; alo_s = alo[ix]; ahi_s = ahi[ix]
    L = n // 2
    for _ in range(n + 2):
        # i<=L use ahi, i>L use alo
        num = ahi_s[:L+1].dot(clo_s[:L+1]) + alo_s[L+1:].dot(clo_s[L+1:])
        den = ahi_s[:L+1].sum() + alo_s[L+1:].sum()
        c_lower = float(num/den) if den > 1e-15 else float(clo_s.mean())
        Ln = int(np.searchsorted(clo_s, c_lower, side='right')) - 1
        Ln = max(0, min(Ln, n-1))
        if Ln == L: break
        L = Ln

    # ── c_upper via KM (Eq. 14 bottom) ──
    ix = np.argsort(chi)
    chi_s = chi[ix]; alo_s2 = alo[ix]; ahi_s2 = ahi[ix]
    R = n // 2
    for _ in range(n + 2):
        # i<=R use alo, i>R use ahi
        num = alo_s2[:R+1].dot(chi_s[:R+1]) + ahi_s2[R+1:].dot(chi_s[R+1:])
        den = alo_s2[:R+1].sum() + ahi_s2[R+1:].sum()
        c_upper = float(num/den) if den > 1e-15 else float(chi_s.mean())
        Rn = int(np.searchsorted(chi_s, c_upper, side='right')) - 1
        Rn = max(0, min(Rn, n-1))
        if Rn == R: break
        R = Rn

    return (c_lower, c_upper, (c_lower + c_upper) / 2.0)

# ── Worked example on first test window ──
if split_idx >= WINDOW_SIZE:
    ew = data[split_idx - WINDOW_SIZE : split_idx]
    es = window_summary(ew)
    obs_trap = (*tuple(float(x) for x in norm(es)), 1.0)
    print("Worked Example — first test window")
    print(f"  Window summary (raw): {es}")
    print(f"  Observation trapezoid (norm): {obs_trap}")
    a_list, c_list = [], []
    for i, rl in enumerate(rules):
        alo, ahi = firing_strength(rl['antecedents'], [obs_trap])
        a_list.append((alo, ahi)); c_list.append(rl['consequence'])
        print(f"  Rule {i+1}: alpha=[{alo:.6f}, {ahi:.6f}]  "
              f"c=[{rl['consequence'][0]:.4f}, {rl['consequence'][1]:.4f}]")
    cL, cU, cF = km_type_reduce(a_list, c_list)
    print(f"  KM result: c_lower={cL:.4f}, c_upper={cU:.4f}, c_final={cF:.4f}")
    print(f"  Actual: {data[split_idx]:.4f}")""")

# ── Cell 8: Forecasting Loop ────────────────────────────────
C8 = code("""\
# ============================================================
# 7. Forecasting Loop
# ============================================================
actuals, preds, iv_lo, iv_hi = [], [], [], []

for t in range(split_idx, len(data)):
    if t - WINDOW_SIZE < 0: continue
    w = data[t - WINDOW_SIZE : t]
    s = window_summary(w)
    obs_trap = (*tuple(float(x) for x in norm(s)), 1.0)

    a_list, c_list = [], []
    for rl in rules:
        alo, ahi = firing_strength(rl['antecedents'], [obs_trap])
        a_list.append((alo, ahi)); c_list.append(rl['consequence'])

    cL, cU, cF = km_type_reduce(a_list, c_list)
    actuals.append(data[t]); preds.append(cF)
    iv_lo.append(cL); iv_hi.append(cU)

actuals = np.array(actuals); preds = np.array(preds)
iv_lo = np.array(iv_lo); iv_hi = np.array(iv_hi)
print(f"Forecasting done: {len(actuals)} test steps")""")

# ── Cell 9: Results ──────────────────────────────────────────
C9 = code("""\
# ============================================================
# 8. Results — RMSE, MSE, NMSE, sMAPE
# ============================================================
variants = {"IT2 TSK+ (KM defuzz)": preds}
all_rmse, all_mse, all_nmse, all_smape = {}, {}, {}, {}
print("=" * 72)
print(f"{'Model':<22s} {'RMSE':>12s} {'MSE':>14s} {'NMSE':>14s} {'sMAPE':>10s}")
print("-" * 72)
for name, p in variants.items():
    r=calc_rmse(actuals,p); m=calc_mse(actuals,p)
    n=calc_nmse(actuals,p); s=calc_smape(actuals,p)
    all_rmse[name]=r; all_mse[name]=m; all_nmse[name]=n; all_smape[name]=s
    print(f"{name:<22s} {r:12.6f} {m:14.6f} {n:14.10f} {s:10.6f}")
print("=" * 72)
print(f"\\nMean interval width (uncertainty): {float(np.mean(iv_hi-iv_lo)):.6f}")
best = min(variants, key=lambda k: all_rmse[k])
print(f"\\nBest variant: {best}")
print(f"  RMSE:  {all_rmse[best]:.6f}")
print(f"  MSE:   {all_mse[best]:.6f}")
print(f"  NMSE:  {all_nmse[best]:.10f}")
print(f"  sMAPE: {all_smape[best]:.6f}")""")

# ── Cell 10: Plot ────────────────────────────────────────────
C10 = code("""\
# ============================================================
# 9. Predictions vs Actual with Uncertainty Interval
# ============================================================
plt.figure(figsize=(12, 5))
plt.plot(actuals, label='Actual', color='blue', linewidth=1.5)
plt.plot(preds, label=f'Predicted ({best})', color='red', linewidth=1.5, linestyle='--')
plt.title('IT2 TSK+ Fuzzy Inference — Li & Yang (2018)\\nPredictions vs Actual')
plt.xlabel('Time Step'); plt.ylabel('Value')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
t_ax = np.arange(len(actuals))
ax.plot(t_ax, actuals, lw=2.0, label="Actual", color="#2c3e50")
ax.plot(t_ax, preds, lw=1.4, label="IT2 TSK+", color="#e74c3c", ls="--", alpha=0.85)
ax.fill_between(t_ax, iv_lo, iv_hi, alpha=0.12, color="#3498db", label="KM interval [c_,c̄]")
ax.set_xlabel("Test time step"); ax.set_ylabel("Value")
ax.set_title("IT2 TSK+ — Prediction with Uncertainty Interval")
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()""")

C11 = code("""\
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
            md(f"# IT2 TSK+ Fuzzy Inference System — {d['name']}\n"
               "### Li & Yang et al. (2018), IEEE FUZZ-IEEE 2018\n"
               "Pure NumPy implementation. Metrics: RMSE, MSE, NMSE, sMAPE."),
            C0, C1, C2, cell_load(d), C3, C4, C5, C6, C7, C8, C9, C10, C11,
        ],
    }

sd = os.path.dirname(os.path.abspath(__file__))
for folder, d in DS.items():
    nb = make_nb(d)
    out = os.path.join(sd, folder, "it2_tsk_plus.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(nb, f, indent=1)
    print(f"Created {out}")
print("\nDone.")
