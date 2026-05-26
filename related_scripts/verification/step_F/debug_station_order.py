"""Debug: compare station ordering between MATLAB and Python for Step F input."""
import sys, struct
import numpy as np
import pandas as pd
from pathlib import Path

py_dir_A = Path(r'D:\Dataset\DPastCliM-NA\verification\step_A\python')
ml_dir_D = Path(r'D:\Dataset\DPastCliM-NA\verification\step_D\matlab')

# Load yhat station_ids from Step D
with open(ml_dir_D / 'station_ids.txt') as f:
    station_ids = [l.strip() for l in f if l.strip()]

# Load obs
obs = pd.read_parquet(py_dir_A / 'ghcn_tas_obs.parquet')
obs_jan = obs[(obs['month'] == 1) & (obs['year'] >= 1875) & (obs['year'] <= 1999)]
unique_ids = sorted(obs_jan['ID'].unique())
unique_years = sorted(obs_jan['year'].unique())

# Build Y_mat
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

# Python intersection
sid_set = set(station_ids) & set(keep_ids)
common_sorted = sorted(sid_set)
ia2_py = [station_ids.index(s) for s in common_sorted]
ib2_py = [unique_ids.index(s) for s in common_sorted]

print(f"station_ids (from Step D): {len(station_ids)}")
print(f"unique_ids (from obs): {len(unique_ids)}")
print(f"keep_ids (>=30 valid): {len(keep_ids)}")
print(f"common_sorted: {len(common_sorted)}")

# Now check: do MATLAB and Python get same intersection?
# Save Python's common_sorted for MATLAB comparison
with open(ml_dir_D / 'py_common_ids.txt', 'w') as f:
    for s in common_sorted:
        f.write(s + '\n')

# Load MATLAB's residuals and check the first 5 station IDs
print(f"\nFirst 10 common IDs (Python): {common_sorted[:10]}")
print(f"ia2_py[:10] (indices into station_ids): {ia2_py[:10]}")
print(f"station_ids at ia2[:10]: {[station_ids[i] for i in ia2_py[:10]]}")

# Check Y_obs construction
Y_obs = Y_aligned[:, ib2_py].T
print(f"\nY_obs shape: {Y_obs.shape}")
print(f"NaN fraction: {np.isnan(Y_obs).mean():.4f}")
print(f"valid_mask (>=30 per station): {(np.sum(~np.isnan(Y_obs), axis=1) >= 30).sum()}")

# Load yhat
with open(ml_dir_D / 'yhat_full.bin', 'rb') as f:
    T_ml, n_st = struct.unpack('ii', f.read(8))
    yhat = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, n_st)

Yhat = yhat[:, ia2_py].T
residuals = Y_obs - Yhat
print(f"residuals NaN fraction: {np.isnan(residuals).mean():.4f}")

# Apply MATLAB subsample
ml_out = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\matlab')
sub_idx = np.fromfile(ml_out / 'sub_idx.bin', dtype=np.int32) - 1
res_sub = residuals[sub_idx, :]
print(f"\nAfter MATLAB sub_idx (500 stations):")
print(f"  NaN fraction: {np.isnan(res_sub).mean():.4f}")
print(f"  valid_mask: {(np.sum(~np.isnan(res_sub), axis=1) >= 30).sum()}")

# Load MATLAB's saved residuals and compare
with open(ml_out / 'residuals_sub.bin', 'rb') as f:
    hdr = struct.unpack('ii', f.read(8))
    ml_res = np.frombuffer(f.read(), dtype=np.float32).reshape(hdr[0], hdr[1])

print(f"  MATLAB NaN fraction: {np.isnan(ml_res).mean():.4f}")
print(f"  MATLAB valid_mask: {(np.sum(~np.isnan(ml_res), axis=1) >= 30).sum()}")

# Direct comparison
both = ~np.isnan(res_sub) & ~np.isnan(ml_res)
if both.any():
    diff = np.abs(res_sub[both] - ml_res[both])
    print(f"  Where both valid: max|diff|={diff.max():.4e}, mean|diff|={diff.mean():.4e}")
    print(f"  N valid pairs: {both.sum()}")
else:
    print("  No overlapping valid entries!")

# Check which stations are at the same position
print(f"\n  Python sub station[0] ID: {common_sorted[sub_idx[0]]}")
# Load MATLAB station order
ml_station_file = ml_out / 'station_ids_sub.txt'
if ml_station_file.exists():
    with open(ml_station_file) as f:
        ml_sub_ids = [l.strip() for l in f if l.strip()]
    print(f"  MATLAB sub station[0] ID: {ml_sub_ids[0]}")
    match_count = sum(1 for a, b in zip(common_sorted, [common_sorted[i] for i in sub_idx]) if a == b)
    print(f"  Stations match at same positions: {match_count}/{len(sub_idx)}")
