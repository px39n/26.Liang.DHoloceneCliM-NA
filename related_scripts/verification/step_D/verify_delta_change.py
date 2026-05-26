"""Verify delta-change (30-year moving-mean) computation matches MATLAB."""
import sys, h5py
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
sys.path.insert(0, r'D:\OneDrive\Code\25.Liang.DPastCliM-NA\src')
from caz.pcr import _movmean_2d, _nearest_interp_ts

mat_path = r'D:\Dataset\DPastCliM-NA\verification\step_D\dc_matlab.mat'

with h5py.File(mat_path, 'r') as f:
    lat = np.array(f['lat']).flatten()
    lon = np.array(f['lon']).flatten()
    sta_lat = np.array(f['sta_lat']).flatten()
    sta_lon = np.array(f['sta_lon']).flatten()
    T_cal  = int(np.array(f['T_cal']).item())
    T_full = int(np.array(f['T_full']).item())
    n_mov  = int(np.array(f['n_mov']).item())

    # MATLAB (ny,nx,T) -> HDF5 C-order: reversed dims -> (T,nx,ny)
    ff_raw = np.array(f['field_full'])   # (T, nx, ny)
    fp_raw = np.array(f['field_pr'])

    # MATLAB mu_adj: (ny*nx, T) col-major -> HDF5: (T, ny*nx) but pixel order is col-major
    ml_mu_adj_tas_flat = np.array(f['mu_adj_tas'])  # (T, ny*nx) with MATLAB col-major pixel order
    ml_mu_adj_pr_flat  = np.array(f['mu_adj_pr'])

    ml_at_sta_tas = np.array(f['mu_adj_at_sta_tas'])  # (T, n_sta)
    ml_at_sta_pr  = np.array(f['mu_adj_at_sta_pr'])

ny = len(lat)
nx = len(lon)
print(f'ny={ny}, nx={nx}, T_full={T_full}, T_cal={T_cal}, n_mov={n_mov}')
print(f'HDF5 field shape: {ff_raw.shape}')

# Reconstruct MATLAB's 3D field: HDF5 (T, nx, ny) -> need (T, ny, nx)
field_tas = ff_raw.transpose(0, 2, 1)  # (T, ny, nx)
field_pr  = fp_raw.transpose(0, 2, 1)

# MATLAB flattens (ny,nx) in column-major: pixel_idx = iy + ny*ix
# Reshape ml_mu_adj from MATLAB's col-major flat to 3D
# ml_mu_adj_tas_flat shape is (T, ny*nx) with col-major pixel indexing
# In MATLAB: reshape(mu_adj, ny, nx, T) would give correct 3D
# In HDF5/Python: the flat (T, ny*nx) uses MATLAB's col-major pixel order
# MATLAB col-major: first index (iy) varies fastest -> pixel = iy + ny * ix
# So reshape in Fortran order to get (ny, nx) correctly
ml_mu_adj_tas_3d = ml_mu_adj_tas_flat.reshape(T_full, ny, nx, order='F')
ml_mu_adj_pr_3d  = ml_mu_adj_pr_flat.reshape(T_full, ny, nx, order='F')

# === Python TAS delta-change ===
M_full = field_tas.reshape(T_full, ny * nx)
i_cal = slice(T_full - T_cal, T_full)
M_cal = M_full[i_cal]

mu_cal = np.mean(M_cal, axis=0)
mu_mov = _movmean_2d(M_full, n_mov)
mu_adj_tas = mu_mov - mu_cal[None, :]
mu_adj_tas_3d = mu_adj_tas.reshape(T_full, ny, nx)

diff_tas = np.abs(mu_adj_tas_3d - ml_mu_adj_tas_3d)
print(f'\n=== TAS delta-change ===')
print(f'Max diff: {diff_tas.max():.2e}')
print(f'Mean diff: {diff_tas.mean():.2e}')
print(f'Py range: [{mu_adj_tas.min():.6f}, {mu_adj_tas.max():.6f}]')

# === Python PR delta-change ===
M_full_pr = field_pr.reshape(T_full, ny * nx)
M_cal_pr = M_full_pr[i_cal]

M_t = 1.0 + np.min(M_cal_pr, axis=0, keepdims=True)
mu_cal_pr = np.mean(np.log(M_cal_pr + M_t), axis=0)
mu_mov_pr = _movmean_2d(np.log(M_full_pr + M_t), n_mov)
mu_adj_pr = mu_mov_pr - mu_cal_pr[None, :]
mu_adj_pr_3d = mu_adj_pr.reshape(T_full, ny, nx)

diff_pr = np.abs(mu_adj_pr_3d - ml_mu_adj_pr_3d)
print(f'\n=== PR delta-change ===')
print(f'Max diff: {diff_pr.max():.2e}')
print(f'Mean diff: {diff_pr.mean():.2e}')

# === Nearest-neighbor interpolation ===
py_at_sta_tas = _nearest_interp_ts(
    mu_adj_tas_3d.astype(np.float32), lat.astype(np.float32), lon.astype(np.float32),
    sta_lat.astype(np.float32), sta_lon.astype(np.float32)
)
py_at_sta_pr = _nearest_interp_ts(
    mu_adj_pr_3d.astype(np.float32), lat.astype(np.float32), lon.astype(np.float32),
    sta_lat.astype(np.float32), sta_lon.astype(np.float32)
)

ml_sta_tas = ml_at_sta_tas if ml_at_sta_tas.shape == py_at_sta_tas.shape else ml_at_sta_tas.T
ml_sta_pr  = ml_at_sta_pr if ml_at_sta_pr.shape == py_at_sta_pr.shape else ml_at_sta_pr.T

diff_interp_tas = np.abs(py_at_sta_tas - ml_sta_tas)
diff_interp_pr  = np.abs(py_at_sta_pr - ml_sta_pr)

print(f'\n=== Nearest-neighbor interp (TAS) ===')
print(f'Max diff: {diff_interp_tas.max():.2e}')
print(f'Mean diff: {diff_interp_tas.mean():.2e}')

print(f'\n=== Nearest-neighbor interp (PR) ===')
print(f'Max diff: {diff_interp_pr.max():.2e}')
print(f'Mean diff: {diff_interp_pr.mean():.2e}')

# Summary
tol_dc = 1e-10
tol_interp = 1e-4
p1 = diff_tas.max() < tol_dc
p2 = diff_pr.max() < tol_dc
p3 = diff_interp_tas.max() < tol_interp
p4 = diff_interp_pr.max() < tol_interp

print(f'\n=== SUMMARY ===')
print(f'Delta-change (tas): {"PASS" if p1 else "FAIL"} (max={diff_tas.max():.2e})')
print(f'Delta-change (pr):  {"PASS" if p2 else "FAIL"} (max={diff_pr.max():.2e})')
print(f'Interp (tas):       {"PASS" if p3 else "FAIL"} (max={diff_interp_tas.max():.2e})')
print(f'Interp (pr):        {"PASS" if p4 else "FAIL"} (max={diff_interp_pr.max():.2e})')
print(f'ALL PASS: {p1 and p2 and p3 and p4}')
