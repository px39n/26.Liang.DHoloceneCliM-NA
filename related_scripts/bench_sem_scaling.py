"""Multi-scale SEM timing benchmark.

Tests slogdet + matmul performance at various station counts
to predict real production time for n=5214.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time as _t
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Check BLAS backend
print("Checking BLAS backend...", flush=True)
try:
    import scipy
    print(f"  scipy: {scipy.__version__}", flush=True)
    blas_info = np.__config__.blas_opt_info if hasattr(np.__config__, 'blas_opt_info') else {}
    print(f"  numpy config blas: {blas_info}", flush=True)
except Exception as e:
    print(f"  config check failed: {e}", flush=True)

# Quick MKL test
t0 = _t.time()
a = np.random.randn(1000, 1000)
_ = np.linalg.slogdet(a)
dt_1k = _t.time() - t0
print(f"  slogdet(1000x1000) = {dt_1k:.4f}s", flush=True)

sizes = [50, 100, 200, 500, 1000, 1500, 2000, 3000]
T = 60
nan_frac = 0.15

results = []
print(f"\n{'n':>6} | {'pdist':>7} | {'W_bld':>7} | {'slogdet':>7} | {'matmul':>7} | "
      f"{'1eval':>7} | {'4th_tot':>8} | {'12mo':>8}", flush=True)
print("-" * 85, flush=True)

for n in sizes:
    np.random.seed(42)
    coords = np.random.randn(n, 2) * 100000
    R = np.random.randn(n, T)
    mask = np.random.rand(n, T) < nan_frac
    R[mask] = np.nan

    t0 = _t.time()
    D = squareform(pdist(coords))
    dt_pdist = _t.time() - t0

    t0 = _t.time()
    W = np.exp(-(D ** 2) / (2 * 50000 ** 2))
    np.fill_diagonal(W, 0)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W /= rs
    dt_wbuild = _t.time() - t0

    A = np.eye(n) - 0.5 * W

    t0 = _t.time()
    s, ld = np.linalg.slogdet(A)
    dt_slogdet = _t.time() - t0

    t0 = _t.time()
    eps = np.full_like(R, np.nan)
    for ti in range(T):
        res_t = R[:, ti]
        valid = ~np.isnan(res_t)
        nv = valid.sum()
        if nv < 2:
            continue
        eps[valid, ti] = A[np.ix_(valid, valid)] @ res_t[valid]
    dt_matmul = _t.time() - t0

    dt_1eval = dt_slogdet + dt_matmul
    n_eval_per_th = 15
    n_th = 4
    dt_4th = n_th * (dt_pdist + dt_wbuild + n_eval_per_th * dt_1eval)
    dt_12mo = 12 * dt_4th

    print(f"{n:>6} | {dt_pdist:>6.3f}s | {dt_wbuild:>6.3f}s | {dt_slogdet:>6.3f}s | "
          f"{dt_matmul:>6.3f}s | {dt_1eval:>6.3f}s | {dt_4th:>7.1f}s | {dt_12mo:>7.0f}s",
          flush=True)

    results.append({
        "n": n, "pdist": dt_pdist, "W_build": dt_wbuild,
        "slogdet": dt_slogdet, "matmul_nan": dt_matmul,
        "one_eval": dt_1eval, "four_thresh": dt_4th, "twelve_months": dt_12mo,
    })

# Extrapolate to 5214 using cubic fit on slogdet
print("\n--- Extrapolation to n=5214 ---", flush=True)
ns = np.array([r["n"] for r in results])
sds = np.array([r["slogdet"] for r in results])
mms = np.array([r["matmul_nan"] for r in results])

# Fit slogdet ~ a * n^3
a_sd = np.median(sds / ns ** 3)
# Fit matmul ~ b * n^2
a_mm = np.median(mms / ns ** 2)

n_target = 5214
sd_pred = a_sd * n_target ** 3
mm_pred = a_mm * n_target ** 2
eval_pred = sd_pred + mm_pred
th_pred = 4 * (15 * eval_pred + 2.0)
mo_pred = 12 * th_pred

print(f"  slogdet({n_target}) predicted: {sd_pred:.2f}s", flush=True)
print(f"  matmul({n_target}) predicted:  {mm_pred:.2f}s", flush=True)
print(f"  1 eval predicted:    {eval_pred:.2f}s", flush=True)
print(f"  4 thresholds:        {th_pred:.0f}s = {th_pred/60:.1f} min", flush=True)
print(f"  12 months:           {mo_pred:.0f}s = {mo_pred/3600:.1f} hours", flush=True)

with open(r"D:\Dataset\DPastCliM-NA\interim\pcr_station\bench_sem_scaling.json", "w") as f:
    json.dump({"results": results, "extrapolation": {
        "n": n_target, "slogdet_s": sd_pred, "matmul_s": mm_pred,
        "per_eval_s": eval_pred, "4_thresholds_s": th_pred,
        "12_months_s": mo_pred, "12_months_h": mo_pred / 3600,
    }}, f, indent=2)

print("\nDONE", flush=True)
