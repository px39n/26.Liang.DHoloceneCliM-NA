"""Debug: compare actual Y_obs and Yhat values at specific stations."""
import sys, struct
import numpy as np
import pandas as pd
from pathlib import Path

py_dir_A = Path(r'D:\Dataset\DPastCliM-NA\verification\step_A\python')
ml_dir_D = Path(r'D:\Dataset\DPastCliM-NA\verification\step_D\matlab')
ml_out = Path(r'D:\Dataset\DPastCliM-NA\verification\step_F\matlab')

# Load yhat
with open(ml_dir_D / 'yhat_full.bin', 'rb') as f:
    T_ml, n_st = struct.unpack('ii', f.read(8))
    yhat = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, n_st)

with open(ml_dir_D / 'station_ids.txt') as f:
    station_ids = [l.strip() for l in f if l.strip()]

# Load obs
obs = pd.read_parquet(py_dir_A / 'ghcn_tas_obs.parquet')
obs_jan = obs[(obs['month'] == 1) & (obs['year'] >= 1875) & (obs['year'] <= 1999)]
unique_ids = sorted(obs_jan['ID'].unique())
unique_years = sorted(obs_jan['year'].unique())

# Build observation matrix
id2col = {sid: i for i, sid in enumerate(unique_ids)}
yr2row = {yr: i for i, yr in enumerate(unique_years)}
Y_mat = np.full((len(unique_years), len(unique_ids)), np.nan, dtype=np.float32)
for _, row in obs_jan.iterrows():
    Y_mat[yr2row[row['year']], id2col[row['ID']]] = row['value']

# Year alignment
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

print(f"T_ml (yhat time dim): {T_ml}")
print(f"common_years: {len(common_years)}, range [{min(common_years)}, {max(common_years)}]")
print(f"Y_aligned shape: {Y_aligned.shape}")
print(f"yhat shape: {yhat.shape}")

# Check if yhat T matches Y_aligned T
if T_ml != len(common_years):
    print(f"\n*** MISMATCH: yhat has {T_ml} timesteps but Y_aligned has {len(common_years)} ***")
    print(f"This means the residuals are computed over different time ranges!")

# check a specific station
test_sid = station_ids[0]
test_idx_in_unique = unique_ids.index(test_sid)
print(f"\nStation {test_sid}:")
print(f"  Y_aligned[:5]: {Y_aligned[:5, test_idx_in_unique]}")
print(f"  Yhat[:5]: {yhat[:5, 0]}")
print(f"  Residual[:5]: {Y_aligned[:5, test_idx_in_unique] - yhat[:5, 0]}")
print(f"  Y_obs NaN count: {np.isnan(Y_aligned[:, test_idx_in_unique]).sum()}/{Y_aligned.shape[0]}")

# sub_idx check
sub_idx = np.fromfile(ml_out / 'sub_idx.bin', dtype=np.int32) - 1
print(f"\nsub_idx[0]={sub_idx[0]}, station={station_ids[sub_idx[0]]}")
sid50 = station_ids[sub_idx[0]]
uid50 = unique_ids.index(sid50)
print(f"  Y_aligned[:5]: {Y_aligned[:5, uid50]}")
print(f"  Yhat[:5, sub_idx[0]]: {yhat[:5, sub_idx[0]]}")
