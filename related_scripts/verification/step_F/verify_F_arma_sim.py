"""Step F2+F3 verification: ARMA fitting + noise simulation.

Since MATLAB lacks Econometrics Toolbox, we:
1. Fit ARMA(1,1) in Python on MATLAB's eps_mat
2. Save coefficients for MATLAB to use
3. Both simulate ARMA noise with same seed/coefficients
4. Both apply inverse SEM
5. Compare final spatially correlated noise
"""
import sys, struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from caz.sem import fit_arma_per_station, simulate_arma_noise, inverse_sem

ml_out = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\matlab')
py_out = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\python')
py_out.mkdir(parents=True, exist_ok=True)

# Load MATLAB's eps_mat and SEM params
with open(ml_out / 'eps_mat.bin', 'rb') as f:
    n_st, T = struct.unpack('ii', f.read(8))
    eps_mat = np.frombuffer(f.read(), dtype=np.float32).reshape(n_st, T)

with open(ml_out / 'sem_params.bin', 'rb') as f:
    lambda_hat = struct.unpack('d', f.read(8))[0]
    threshold_best = struct.unpack('d', f.read(8))[0]
    n_st2, T2 = struct.unpack('ii', f.read(8))
    sigma2_ml = np.frombuffer(f.read(), dtype=np.float64)

with open(ml_out / 'coords_sub.bin', 'rb') as f:
    raw = np.frombuffer(f.read(), dtype=np.float64)
    half = len(raw) // 2
    coords = np.column_stack([raw[:half], raw[half:]])

valid_mask = np.sum(~np.isnan(eps_mat), axis=1) >= 30
print(f"Loaded eps_mat: {eps_mat.shape}, lambda={lambda_hat:.6f}, {valid_mask.sum()} valid stations")

# F2: Fit ARMA(1,1) on each station's eps_mat
p, q = 1, 1
models = fit_arma_per_station(eps_mat, valid_mask, p=p, q=q)

# Save coefficients for MATLAB
n_valid = valid_mask.sum()
valid_idx = np.where(valid_mask)[0]
ar_coeffs = np.zeros(n_st)
ma_coeffs = np.zeros(n_st)
variances = np.zeros(n_st)
for i in valid_idx:
    if models[i] is not None:
        ar_coeffs[i] = models[i]['ar'][0] if len(models[i]['ar']) > 0 else 0.0
        ma_coeffs[i] = models[i]['ma'][0] if len(models[i]['ma']) > 0 else 0.0
        variances[i] = models[i]['variance']

print(f"\nARMA(1,1) fit summary ({n_valid} stations):")
print(f"  AR coeff: mean={ar_coeffs[valid_mask].mean():.4f}, std={ar_coeffs[valid_mask].std():.4f}")
print(f"  MA coeff: mean={ma_coeffs[valid_mask].mean():.4f}, std={ma_coeffs[valid_mask].std():.4f}")
print(f"  Variance: mean={variances[valid_mask].mean():.4f}, std={variances[valid_mask].std():.4f}")

# Save for MATLAB
with open(py_out / 'arma_coeffs.bin', 'wb') as f:
    f.write(struct.pack('i', n_st))
    f.write(ar_coeffs.astype(np.float64).tobytes())
    f.write(ma_coeffs.astype(np.float64).tobytes())
    f.write(variances.astype(np.float64).tobytes())
    f.write(valid_mask.astype(np.int32).tobytes())

# F3: Simulate ARMA noise (fixed seed for reproducibility)
n_sim_time = 200  # simulate 200 timesteps
rng = np.random.default_rng(seed=12345)
eps_arma_py = simulate_arma_noise(models, valid_mask, n_sim_time, start_time=0, p=p, q=q, rng=rng)

# Also save the random draws for MATLAB to use identically
rng2 = np.random.default_rng(seed=12345)
eta_all = np.zeros((n_st, n_sim_time))
for i in range(n_st):
    if valid_mask[i] and models[i] is not None:
        sigma = np.sqrt(models[i]['variance'])
        eta_all[i, :] = sigma * rng2.standard_normal(n_sim_time)

with open(py_out / 'eta_noise.bin', 'wb') as f:
    f.write(struct.pack('ii', n_st, n_sim_time))
    eta_all.astype(np.float64).tofile(f)

with open(py_out / 'eps_arma.bin', 'wb') as f:
    f.write(struct.pack('ii', n_st, n_sim_time))
    eps_arma_py.astype(np.float64).tofile(f)

# Apply inverse SEM
from scipy.spatial.distance import pdist, squareform
D = squareform(pdist(coords))
h = threshold_best
W = np.exp(-(D**2) / (2 * h**2))
np.fill_diagonal(W, 0.0)
row_sums = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
W /= row_sums
eigvals = np.linalg.eigvals(W)
rho = np.max(np.abs(eigvals))
if rho >= 1.0:
    W /= (rho + 1e-2)

u_mat_py = inverse_sem(eps_arma_py, W, lambda_hat)

with open(py_out / 'u_mat.bin', 'wb') as f:
    f.write(struct.pack('ii', n_st, n_sim_time))
    u_mat_py.astype(np.float64).tofile(f)

print(f"\nF3 outputs: eps_arma shape={eps_arma_py.shape}, u_mat shape={u_mat_py.shape}")
print(f"  eps_arma: mean={eps_arma_py[valid_mask].mean():.4f}, std={eps_arma_py[valid_mask].std():.4f}")
print(f"  u_mat: mean={u_mat_py[valid_mask].mean():.4f}, std={u_mat_py[valid_mask].std():.4f}")
print(f"\nSaved: arma_coeffs.bin, eta_noise.bin, eps_arma.bin, u_mat.bin to {py_out}")
