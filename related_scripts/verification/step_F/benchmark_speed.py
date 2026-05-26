"""Benchmark: Python SEM + ARMA speed on 500 stations, 125 years."""
import sys, struct, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

ml_out = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\matlab')

# Load data
with open(ml_out / 'residuals_sub.bin', 'rb') as f:
    n_st, T = struct.unpack('ii', f.read(8))
    residuals = np.frombuffer(f.read(), dtype=np.float32).reshape(n_st, T)

with open(ml_out / 'coords_sub.bin', 'rb') as f:
    raw = np.frombuffer(f.read(), dtype=np.float64)
    half = len(raw) // 2
    coords = np.column_stack([raw[:half], raw[half:]])

valid_mask = np.sum(~np.isnan(residuals), axis=1) >= 30
print(f"Data: {n_st} stations, {T} years, {valid_mask.sum()} valid")

# --- Benchmark SEM fitting ---
from caz.sem import _gaussian_weight_matrix, _profile_neg_loglik

t0 = time.perf_counter()

from scipy.optimize import minimize_scalar
thresholds = np.linspace(25_000, 100_000, 4)
best_nLL = np.inf
for th in thresholds:
    W = _gaussian_weight_matrix(coords, th)
    result = minimize_scalar(
        lambda lam: _profile_neg_loglik(lam, residuals, W, valid_mask, 0.1),
        bounds=(0.0, 0.999), method='bounded',
        options={'xatol': 1e-6, 'maxiter': 200}
    )
    if result.fun < best_nLL:
        best_nLL = result.fun
        lambda_hat = result.x
        W_best = W
        threshold_best = th

t_sem = time.perf_counter() - t0
print(f"\nSEM fitting: {t_sem:.2f}s (lambda={lambda_hat:.4f})")

# --- Benchmark eps_mat computation ---
t0 = time.perf_counter()
A_best = np.eye(n_st) - lambda_hat * W_best
eps_mat = np.full_like(residuals, np.nan)
for t in range(T):
    res_t = residuals[:, t]
    valid = ~np.isnan(res_t)
    if valid.sum() < 2:
        continue
    eps_mat[valid, t] = A_best[np.ix_(valid, valid)] @ res_t[valid]
t_eps = time.perf_counter() - t0
print(f"eps_mat computation: {t_eps:.2f}s")

# --- Benchmark ARMA fitting ---
from caz.sem import fit_arma_per_station
t0 = time.perf_counter()
models = fit_arma_per_station(eps_mat, valid_mask, p=1, q=1)
t_arma = time.perf_counter() - t0
print(f"ARMA(1,1) CSS fitting ({valid_mask.sum()} stations): {t_arma:.2f}s")

# --- Benchmark ARMA simulation ---
from caz.sem import simulate_arma_noise, inverse_sem
n_sim = 22000  # 22ka production
t0 = time.perf_counter()
rng = np.random.default_rng(42)
eps_arma = simulate_arma_noise(models, valid_mask, n_sim, p=1, q=1, rng=rng)
t_sim = time.perf_counter() - t0
print(f"ARMA simulation ({n_sim} timesteps): {t_sim:.2f}s")

# --- Benchmark inverse SEM ---
t0 = time.perf_counter()
u_mat = inverse_sem(eps_arma, W_best, lambda_hat)
t_inv = time.perf_counter() - t0
print(f"Inverse SEM ({n_sim} timesteps): {t_inv:.2f}s")

# --- Total ---
total = t_sem + t_eps + t_arma + t_sim + t_inv
print(f"\n=== TOTAL Python: {total:.2f}s ===")
print(f"  SEM fitting:     {t_sem:.2f}s")
print(f"  eps_mat:          {t_eps:.2f}s")
print(f"  ARMA CSS fitting: {t_arma:.2f}s")
print(f"  ARMA simulation:  {t_sim:.2f}s")
print(f"  Inverse SEM:      {t_inv:.2f}s")
