"""Step D PI verification: regression MSE → PI (January, tas).

Computes sigma2_hat and PI = Yhat ± 1.96*sigma (analytical, exact for Gaussian).
Compares with MATLAB output from verify_D_pi.m.
"""
import struct
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

TRACE_TAS = Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc")
OUT_DIR = Path(r"D:\Dataset\DPastCliM-NA\verification\step_D\python")
ML_DIR = Path(r"D:\Dataset\DPastCliM-NA\verification\step_D\matlab")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1875, 1999
N_PC = 5
TRAIN_FRAC = 0.7
TARGET_MONTH = 1

# ---- Load ESM field (same as verify_D_prediction.py) ----
print("Loading ESM field...")
ds = xr.open_dataset(TRACE_TAS)
trefht = ds["TREFHT"].values
lat = ds["lat"].values.astype(np.float64)
lon = ds["lon"].values.astype(np.float64)
time_ka = ds["time"].values.astype(np.float64)
ds.close()

cal_year_frac = 1950 + time_ka * 1000
months_since_0 = np.round(cal_year_frac * 12).astype(np.int64)
year_arr = (months_since_0 // 12).astype(np.int32)
month_arr = (months_since_0 % 12 + 1).astype(np.int32)

tas_map = trefht - 273.15
lon180 = ((lon + 180) % 360) - 180
lon_order = np.argsort(lon180)
lon_sorted = lon180[lon_order]
tas_map = tas_map[:, :, lon_order]
lat_order = np.argsort(lat)
lat_sorted = lat[lat_order]
tas_map = tas_map[:, lat_order, :]

lon_mask = (lon_sorted >= -185) & (lon_sorted <= -45)
lat_mask = (lat_sorted >= -90) & (lat_sorted <= 90)
tas_na = tas_map[:, lat_mask, :][:, :, lon_mask]

cal_mask = (year_arr >= YEAR_MIN) & (year_arr <= YEAR_MAX) & (month_arr == TARGET_MONTH)
esm_jan_cal = tas_na[cal_mask, :, :]
years_jan_cal = year_arr[cal_mask]
T_cal = esm_jan_cal.shape[0]
field_2d = esm_jan_cal.transpose(0, 2, 1).reshape(T_cal, -1)

# ---- PCA ----
mu = field_2d.mean(axis=0)
Xc = field_2d - mu
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
scores_full = U * s
pcs = scores_full[:, :N_PC]

# ---- Load station obs ----
obs = pd.read_parquet(r"D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_obs.parquet")
obs_jan = obs[(obs["month"] == TARGET_MONTH) &
              (obs["year"] >= YEAR_MIN) & (obs["year"] <= YEAR_MAX)].copy()

unique_ids = sorted(obs_jan["ID"].unique())
unique_years = sorted(obs_jan["year"].unique())
id_map = {v: i for i, v in enumerate(unique_ids)}
yr_map = {v: i for i, v in enumerate(unique_years)}
Y_mat = np.full((len(unique_years), len(unique_ids)), np.nan, dtype=np.float32)
for _, row in obs_jan.iterrows():
    Y_mat[yr_map[row["year"]], id_map[row["ID"]]] = row["value"]

common_years = np.intersect1d(unique_years, years_jan_cal)
ia = np.array([unique_years.index(y) for y in common_years])
ib = np.array([list(years_jan_cal).index(y) for y in common_years])
Y_aligned = Y_mat[ia, :]
pcs_aligned = pcs[ib, :]
valid_count = np.sum(~np.isnan(Y_aligned), axis=0)
keep = valid_count >= 30
Y_aligned = Y_aligned[:, keep]
station_ids = [unique_ids[i] for i in range(len(unique_ids)) if keep[i]]
n_stations = Y_aligned.shape[1]

T_common = Y_aligned.shape[0]
n_train = max(round(T_common * TRAIN_FRAC), 30)

X_full = np.column_stack([np.ones(T_common), pcs_aligned])
X_tr = X_full[:n_train, :]
Y_tr = Y_aligned[:n_train, :]

# ---- Per-station OLS + sigma2_hat + PI ----
print(f"Computing prediction + PI for {n_stations} stations...")
Yhat_full = np.full((T_common, n_stations), np.nan, dtype=np.float32)
sigma2_hat = np.full(n_stations, np.nan, dtype=np.float32)
PI_lo = np.full((T_common, n_stations), np.nan, dtype=np.float32)
PI_hi = np.full((T_common, n_stations), np.nan, dtype=np.float32)

for s in range(n_stations):
    ys = Y_tr[:, s]
    m = ~np.isnan(ys)
    if m.sum() < 20:
        continue
    b, _, _, _ = np.linalg.lstsq(X_tr[m, :], ys[m], rcond=None)
    yhat = (X_full @ b).astype(np.float32)
    Yhat_full[:, s] = yhat

    res = ys[m] - (X_tr[m, :] @ b).astype(np.float32)
    sigma2_hat[s] = float(np.mean(res ** 2))
    sigma = np.sqrt(sigma2_hat[s])
    PI_lo[:, s] = yhat - 1.96 * sigma
    PI_hi[:, s] = yhat + 1.96 * sigma

print(f"  sigma2_hat range: [{np.nanmin(sigma2_hat):.6f}, {np.nanmax(sigma2_hat):.6f}]")

np.savez(OUT_DIR / "pi_verification.npz",
         sigma2_hat=sigma2_hat, pi_lo=PI_lo, pi_hi=PI_hi,
         yhat=Yhat_full, station_ids=np.array(station_ids))

# ---- Compare with MATLAB ----
ml_sig_file = ML_DIR / "sigma2_hat.bin"
ml_pilo_file = ML_DIR / "pi_lo.bin"
ml_pihi_file = ML_DIR / "pi_hi.bin"
ml_ids_file = ML_DIR / "station_ids.txt"

if not ml_sig_file.exists():
    print(f"\nMATLAB PI output not found. Run verify_D_pi.m first.")
else:
    print("\n--- Step D PI comparison ---")

    # load MATLAB sigma2_hat
    with open(ml_sig_file, "rb") as f:
        n_ml = struct.unpack("i", f.read(4))[0]
        sig2_ml = np.frombuffer(f.read(), dtype=np.float32)

    # load MATLAB PI
    with open(ml_pilo_file, "rb") as f:
        T_ml, n_ml2 = struct.unpack("ii", f.read(8))
        pilo_ml = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, n_ml2)
    with open(ml_pihi_file, "rb") as f:
        _, _ = struct.unpack("ii", f.read(8))
        pihi_ml = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, n_ml2)

    ml_ids = [l.strip() for l in open(ml_ids_file)]

    common = sorted(set(station_ids) & set(ml_ids))
    print(f"  Common stations: {len(common)}")

    py_idx = {v: i for i, v in enumerate(station_ids)}
    ml_idx = {v: i for i, v in enumerate(ml_ids)}
    py_sel = np.array([py_idx[s] for s in common])
    ml_sel = np.array([ml_idx[s] for s in common])

    # sigma2_hat comparison
    sig2_py = sigma2_hat[py_sel]
    sig2_ml_sel = sig2_ml[ml_sel]
    both_valid = ~np.isnan(sig2_py) & ~np.isnan(sig2_ml_sel)
    sig2_diff = np.abs(sig2_py - sig2_ml_sel)
    print(f"\n  sigma2_hat:")
    print(f"    Valid: {both_valid.sum()}")
    print(f"    Max |diff|: {sig2_diff[both_valid].max():.6f}")
    print(f"    Mean |diff|: {sig2_diff[both_valid].mean():.6f}")

    # PI comparison
    pilo_py = PI_lo[:, py_sel]
    pilo_ml_c = pilo_ml[:, ml_sel]
    pihi_py = PI_hi[:, py_sel]
    pihi_ml_c = pihi_ml[:, ml_sel]

    valid_pi = ~np.isnan(pilo_py) & ~np.isnan(pilo_ml_c)
    diff_lo = np.abs(pilo_py - pilo_ml_c)
    diff_hi = np.abs(pihi_py - pihi_ml_c)
    print(f"\n  PI_lo:")
    print(f"    Valid: {valid_pi.sum()}")
    print(f"    Max |diff|: {diff_lo[valid_pi].max():.6f}")
    print(f"    Mean |diff|: {diff_lo[valid_pi].mean():.6f}")
    print(f"\n  PI_hi:")
    print(f"    Max |diff|: {diff_hi[valid_pi].max():.6f}")
    print(f"    Mean |diff|: {diff_hi[valid_pi].mean():.6f}")

    # PI width comparison
    width_py = pihi_py - pilo_py
    width_ml = pihi_ml_c - pilo_ml_c
    width_diff = np.abs(width_py - width_ml)
    print(f"\n  PI width (hi - lo):")
    print(f"    Max |diff|: {width_diff[valid_pi].max():.6f}")
    print(f"    Mean |diff|: {width_diff[valid_pi].mean():.6f}")

print("\nDone.")
