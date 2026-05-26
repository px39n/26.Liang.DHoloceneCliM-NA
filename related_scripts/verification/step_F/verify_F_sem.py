"""Step F verification: SEM fitting — compare Python vs MATLAB outputs."""
import sys, struct
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from caz.sem import _gaussian_weight_matrix, _profile_neg_loglik, fit_sem

# ---------- replicate MATLAB's data loading ----------
ml_dir_D = Path(r'D:\Dataset\DPastCliM-NA\verification\step_D\matlab')
py_dir_A = Path(r'D:\Dataset\DPastCliM-NA\verification\step_A\python')
ml_out   = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\matlab')
py_out   = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\python')
py_out.mkdir(parents=True, exist_ok=True)

# yhat from Step D MATLAB
with open(ml_dir_D / 'yhat_full.bin', 'rb') as f:
    T_ml, n_st = struct.unpack('ii', f.read(8))
    yhat = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, n_st)

with open(ml_dir_D / 'station_ids.txt') as f:
    station_ids = [l.strip() for l in f if l.strip()]

# obs
import pandas as pd
obs = pd.read_parquet(py_dir_A / 'ghcn_tas_obs.parquet')
obs_jan = obs[(obs['month'] == 1) & (obs['year'] >= 1875) & (obs['year'] <= 1999)]
unique_ids = sorted(obs_jan['ID'].unique())
unique_years = sorted(obs_jan['year'].unique())
id2col = {sid: i for i, sid in enumerate(unique_ids)}
yr2row = {yr: i for i, yr in enumerate(unique_years)}
Y_mat = np.full((len(unique_years), len(unique_ids)), np.nan, dtype=np.float32)
for _, row in obs_jan.iterrows():
    Y_mat[yr2row[row['year']], id2col[row['ID']]] = row['value']

# ESM time alignment
import xarray as xr
ds = xr.open_dataset(r'D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc')
time_ka = ds['time'].values.astype(float)
cal_year_frac = 1950.0 + time_ka * 1000.0
months_since_0 = np.round(cal_year_frac * 12).astype(int)
year_arr = months_since_0 // 12
month_arr = months_since_0 % 12 + 1
cal_mask = (year_arr >= 1875) & (year_arr <= 1999) & (month_arr == 1)
years_jan_cal = year_arr[cal_mask]
ds.close()

common_years = sorted(set(unique_years) & set(years_jan_cal))
ia = [unique_years.index(y) for y in common_years]
Y_aligned = Y_mat[ia, :]
valid_count = np.sum(~np.isnan(Y_aligned), axis=0)
keep = valid_count >= 30
keep_ids = [unique_ids[i] for i in range(len(unique_ids)) if keep[i]]

# Match MATLAB intersect behavior: sorted intersection based on station_ids order
sid_set = set(station_ids) & set(keep_ids)
# MATLAB [~,ia2,ib2] = intersect(string(station_ids), unique_ids(keep))
# intersect returns sorted common elements; ia2 indexes into first, ib2 into second
common_sorted = sorted(sid_set)
ia2 = [station_ids.index(s) for s in common_sorted]
ib2 = [unique_ids.index(s) for s in common_sorted]

Y_obs = Y_aligned[:, ib2].T   # (S, T)
Yhat  = yhat[:, ia2].T         # (S, T)
n_stations = Y_obs.shape[0]
T = Y_obs.shape[1]
print(f"Full dataset: {n_stations} stations, {T} years")

residuals = Y_obs - Yhat

# station metadata + Albers projection
meta = pd.read_parquet(py_dir_A / 'ghcn_tas_meta.parquet')
matched_ids = [station_ids[i] for i in ia2]
meta_sub = meta[meta['ID'].isin(matched_ids)].set_index('ID')
st_lat = np.array([meta_sub.loc[sid, 'lat'] for sid in matched_ids])
st_lon = np.array([meta_sub.loc[sid, 'lon'] for sid in matched_ids])

def albers_forward(lon_deg, lat_deg):
    a = 6378137.0; f = 1/298.257222101; e2 = 2*f - f**2; e = np.sqrt(e2)
    phi1, phi2 = np.radians(29.5), np.radians(45.5)
    phi0, lam0 = np.radians(23.0), np.radians(-96.0)
    m1 = np.cos(phi1)/np.sqrt(1-e2*np.sin(phi1)**2)
    m2 = np.cos(phi2)/np.sqrt(1-e2*np.sin(phi2)**2)
    def qfun(p): return (1-e2)*(np.sin(p)/(1-e2*np.sin(p)**2) - np.log((1-e*np.sin(p))/(1+e*np.sin(p)))/(2*e))
    q0, q1, q2 = qfun(phi0), qfun(phi1), qfun(phi2)
    n = (m1**2-m2**2)/(q2-q1); C = m1**2+n*q1; rho0 = a*np.sqrt(C-n*q0)/n
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    q = qfun(phi); rho = a*np.sqrt(C-n*q)/n; theta = n*(lam-lam0)
    return rho*np.sin(theta), rho0-rho*np.cos(theta)

x_proj, y_proj = albers_forward(st_lon, st_lat)

# Load MATLAB's sub_idx to use the exact same stations
sub_idx_file = ml_out / 'sub_idx.bin'
if sub_idx_file.exists() and n_stations > 500:
    sub_idx = np.fromfile(sub_idx_file, dtype=np.int32) - 1  # MATLAB 1-based
    residuals = residuals[sub_idx, :]
    x_proj = x_proj[sub_idx]
    y_proj = y_proj[sub_idx]
    n_stations = len(sub_idx)
    print(f"Subsampled to {n_stations} stations (using MATLAB indices)")

coords = np.column_stack([x_proj, y_proj])
valid_mask = np.sum(~np.isnan(residuals), axis=1) >= 30

# Compare residuals with MATLAB (no longer overriding — Python runs independently)
ml_res_file = ml_out / 'residuals_sub.bin'
if ml_res_file.exists():
    with open(ml_res_file, 'rb') as f:
        hdr = struct.unpack('ii', f.read(8))
        ml_res = np.frombuffer(f.read(), dtype=np.float32).reshape(hdr[0], hdr[1])
    both_valid = ~np.isnan(residuals) & ~np.isnan(ml_res)
    if both_valid.any():
        res_diff = np.abs(residuals[both_valid] - ml_res[both_valid])
        print(f"Residuals vs MATLAB: max|diff|={res_diff.max():.4e}, mean={res_diff.mean():.4e}, N={both_valid.sum()}")
    print(f"Python NaN frac: {np.isnan(residuals).mean():.4f}, MATLAB: {np.isnan(ml_res).mean():.4f}")

# Recompute valid_mask from actual residuals (may be MATLAB's)
valid_mask = np.sum(~np.isnan(residuals), axis=1) >= 30
print(f"valid_mask sum (final): {valid_mask.sum()}")

# ---------- SEM fitting (same logic as MATLAB) ----------
thresholds = np.linspace(25_000, 100_000, 4)
penalty_weight = 0.1
best_nLL = np.inf
lambda_hat = 0.5
W_best = np.eye(n_stations)
threshold_best = 50_000.0

from scipy.optimize import minimize_scalar

for th in thresholds:
    W = _gaussian_weight_matrix(coords, th)
    result = minimize_scalar(
        lambda lam: _profile_neg_loglik(lam, residuals, W, valid_mask, penalty_weight),
        bounds=(0.0, 0.999), method='bounded',
        options={'xatol': 1e-6, 'maxiter': 200}
    )
    print(f"  threshold={th:.0f}, lambda={result.x:.4f}, nLL={result.fun:.2f}")
    if result.fun < best_nLL:
        best_nLL = result.fun
        lambda_hat = result.x
        W_best = W
        threshold_best = th

print(f"Best: threshold={threshold_best:.0f}, lambda={lambda_hat:.6f}")

# compute eps_mat
A_best = np.eye(n_stations) - lambda_hat * W_best
eps_mat = np.full_like(residuals, np.nan)
for t in range(T):
    res_t = residuals[:, t]
    valid = ~np.isnan(res_t)
    if valid.sum() < 2:
        continue
    eps_mat[valid, t] = A_best[np.ix_(valid, valid)] @ res_t[valid]

sigma2_hat = np.nanmean(eps_mat ** 2, axis=1)

# save Python outputs
with open(py_out / 'sem_params.bin', 'wb') as f:
    f.write(struct.pack('d', lambda_hat))
    f.write(struct.pack('d', threshold_best))
    f.write(struct.pack('ii', n_stations, T))
    f.write(sigma2_hat.astype(np.float64).tobytes())

with open(py_out / 'eps_mat.bin', 'wb') as f:
    f.write(struct.pack('ii', n_stations, T))
    eps_mat.astype(np.float32).tofile(f)

print(f"\nPython outputs saved to: {py_out}")

# ---------- compare with MATLAB ----------
print("\n=== Comparison with MATLAB ===")
with open(ml_out / 'sem_params.bin', 'rb') as f:
    ml_lambda = struct.unpack('d', f.read(8))[0]
    ml_threshold = struct.unpack('d', f.read(8))[0]
    ml_nst, ml_T = struct.unpack('ii', f.read(8))
    ml_sigma2 = np.frombuffer(f.read(), dtype=np.float64)

print(f"MATLAB: lambda={ml_lambda:.6f}, threshold={ml_threshold:.0f}")
print(f"Python: lambda={lambda_hat:.6f}, threshold={threshold_best:.0f}")
print(f"lambda diff: {abs(ml_lambda - lambda_hat):.2e}")

with open(ml_out / 'eps_mat.bin', 'rb') as f:
    hdr = struct.unpack('ii', f.read(8))
    ml_eps = np.frombuffer(f.read(), dtype=np.float32).reshape(hdr[0], hdr[1])

# only compare if same threshold was chosen
if ml_threshold == threshold_best:
    both_valid = ~np.isnan(eps_mat) & ~np.isnan(ml_eps)
    diff = np.abs(eps_mat[both_valid] - ml_eps[both_valid])
    print(f"eps_mat: max|diff|={diff.max():.4e}, mean|diff|={diff.mean():.4e}")
    both_s2_valid = ~np.isnan(sigma2_hat[:ml_sigma2.shape[0]]) & ~np.isnan(ml_sigma2)
    s2_diff = np.abs(sigma2_hat[:ml_sigma2.shape[0]][both_s2_valid] - ml_sigma2[both_s2_valid])
    print(f"sigma2: max|diff|={s2_diff.max():.4e}, mean|diff|={s2_diff.mean():.4e} ({both_s2_valid.sum()} valid stations)")
else:
    print("Different thresholds chosen — comparing raw eps_mat not meaningful")
    print("Check if subsample ordering differs between MATLAB and Python")
