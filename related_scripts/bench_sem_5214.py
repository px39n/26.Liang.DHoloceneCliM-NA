"""Direct benchmark at n=5214 — single threshold, single eval.
Measures actual slogdet + matmul time to validate extrapolation.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time as _t
import numpy as np
from scipy.spatial.distance import pdist, squareform

n = 5214
T = 60
nan_frac = 0.15

print(f"n={n}, T={T}, NaN fraction={nan_frac}", flush=True)

np.random.seed(42)
coords = np.random.randn(n, 2) * 100000
R = np.random.randn(n, T)
mask = np.random.rand(n, T) < nan_frac
R[mask] = np.nan

print("Building distance matrix...", flush=True)
t0 = _t.time()
D = squareform(pdist(coords))
dt_pdist = _t.time() - t0
print(f"  pdist: {dt_pdist:.3f}s", flush=True)

print("Building W matrix...", flush=True)
t0 = _t.time()
W = np.exp(-(D ** 2) / (2 * 50000 ** 2))
np.fill_diagonal(W, 0)
rs = W.sum(axis=1, keepdims=True)
rs[rs == 0] = 1.0
W /= rs
dt_wbuild = _t.time() - t0
print(f"  W build: {dt_wbuild:.3f}s", flush=True)

A = np.eye(n) - 0.5 * W

print("slogdet (LU)...", flush=True)
t0 = _t.time()
s, ld = np.linalg.slogdet(A)
dt_slogdet = _t.time() - t0
print(f"  slogdet: {dt_slogdet:.3f}s (sign={s:.0f}, logdet={ld:.2f})", flush=True)

print("cholesky (optimized)...", flush=True)
from scipy.linalg import cholesky as _chol
t0 = _t.time()
try:
    L = _chol(A, lower=True)
    ld_chol = 2.0 * np.sum(np.log(np.diag(L)))
    dt_chol = _t.time() - t0
    print(f"  chol: {dt_chol:.3f}s (logdet={ld_chol:.2f})", flush=True)
except Exception as e:
    dt_chol = _t.time() - t0
    print(f"  chol FAILED: {e} ({dt_chol:.3f}s)", flush=True)

print("matmul (fill-0 vectorized)...", flush=True)
t0 = _t.time()
nan_mask = np.isnan(R)
R_filled = np.where(nan_mask, 0.0, R)
eps = A @ R_filled
eps[nan_mask] = np.nan
dt_matmul = _t.time() - t0
print(f"  matmul: {dt_matmul:.3f}s", flush=True)

dt_1eval = dt_slogdet + dt_matmul
print(f"\n--- Results ---", flush=True)
print(f"1 eval (slogdet + matmul): {dt_1eval:.3f}s", flush=True)

n_eval = 15
n_thresh = 4
n_months = 12
dt_1thresh = dt_pdist + dt_wbuild + n_eval * dt_1eval
dt_1month = n_thresh * dt_1thresh
dt_12months = n_months * dt_1month

print(f"\nProjection (assumes 15 evals/thresh, 4 thresholds):", flush=True)
print(f"  1 threshold: {dt_1thresh:.0f}s = {dt_1thresh/60:.1f} min", flush=True)
print(f"  1 month (4 thresh): {dt_1month:.0f}s = {dt_1month/60:.1f} min", flush=True)
print(f"  12 months: {dt_12months:.0f}s = {dt_12months/3600:.2f} hours", flush=True)

print("\nDONE", flush=True)
