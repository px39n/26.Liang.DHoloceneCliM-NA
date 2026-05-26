"""Step D verification: PCR prediction on calibration period (January, tas).

Reproduces identical PCA + OLS from Step C, predicts on full cal period,
then compares with MATLAB output.
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

# ---- Load ESM field (same as Step C) ----
print("Loading ESM field...")
ds = xr.open_dataset(TRACE_TAS)
trefht = ds["TREFHT"].values  # (time, lat, lon)
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
tas_na = tas_map[:, :, :][:, lat_mask, :][:, :, lon_mask]
esm_lon = lon_sorted[lon_mask]
esm_lat = lat_sorted[lat_mask]

cal_mask = (year_arr >= YEAR_MIN) & (year_arr <= YEAR_MAX) & (month_arr == TARGET_MONTH)
esm_jan_cal = tas_na[cal_mask, :, :]
years_jan_cal = year_arr[cal_mask]
T_cal = esm_jan_cal.shape[0]

# Flatten: match MATLAB (lat first = transpose(0,2,1))
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

# deterministic split
T_common = Y_aligned.shape[0]
n_train = max(round(T_common * TRAIN_FRAC), 30)

X_full = np.column_stack([np.ones(T_common), pcs_aligned])
X_tr = X_full[:n_train, :]
Y_tr = Y_aligned[:n_train, :]

# ---- Per-station OLS prediction ----
print(f"Predicting on full cal period: {n_stations} stations x {T_common} years")
Yhat_full = np.full((T_common, n_stations), np.nan, dtype=np.float32)

for s in range(n_stations):
    ys = Y_tr[:, s]
    m = ~np.isnan(ys)
    if m.sum() < 20:
        continue
    b, _, _, _ = np.linalg.lstsq(X_tr[m, :], ys[m], rcond=None)
    Yhat_full[:, s] = (X_full @ b).astype(np.float32)

# save
np.savez(OUT_DIR / "yhat_full.npz", yhat=Yhat_full,
         station_ids=np.array(station_ids))

# ---- Compare with MATLAB ----
ml_bin = ML_DIR / "yhat_full.bin"
if not ml_bin.exists():
    print(f"\nMATLAB output not found at {ml_bin}. Run verify_D_prediction.m first.")
else:
    print("\n--- Step D comparison ---")
    with open(ml_bin, "rb") as f:
        T_ml, n_ml = struct.unpack("ii", f.read(8))
        yhat_ml = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, n_ml)

    ml_ids_file = ML_DIR / "station_ids.txt"
    ml_ids = [l.strip() for l in open(ml_ids_file)]

    common_ids = sorted(set(station_ids) & set(ml_ids))
    print(f"  Python stations: {n_stations}, MATLAB stations: {n_ml}")
    print(f"  Common stations: {len(common_ids)}")

    py_idx = {v: i for i, v in enumerate(station_ids)}
    ml_idx = {v: i for i, v in enumerate(ml_ids)}
    py_sel = np.array([py_idx[s] for s in common_ids])
    ml_sel = np.array([ml_idx[s] for s in common_ids])

    yh_py = Yhat_full[:, py_sel]
    yh_ml = yhat_ml[:, ml_sel]

    both_valid = ~np.isnan(yh_py) & ~np.isnan(yh_ml)
    diff = np.abs(yh_py - yh_ml)
    print(f"  Valid comparisons: {both_valid.sum()}")
    print(f"  Max |diff|: {diff[both_valid].max():.6f}")
    print(f"  Mean |diff|: {diff[both_valid].mean():.6f}")
    print(f"  Median |diff|: {np.median(diff[both_valid]):.6f}")

    corrs = []
    for s in range(len(common_ids)):
        m = both_valid[:, s]
        if m.sum() > 10:
            corrs.append(np.corrcoef(yh_py[m, s], yh_ml[m, s])[0, 1])
    print(f"  Mean station correlation: {np.mean(corrs):.6f}")
    print(f"  Min station correlation:  {np.min(corrs):.6f}")

print("\nDone.")
