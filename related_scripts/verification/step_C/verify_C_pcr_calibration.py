"""Step C verification (Python): PCR calibration for January, tas.

Runs identical logic to the MATLAB verify_C script:
  - PCA on NA-windowed ESM grid field for cal years (Jan only)
  - Per-station OLS with intercept
  - Fixed n_pc=5, 70/30 split, rng seed=2026

Then compares eigenvalues, PC scores, regression coefficients, RMSE
with MATLAB outputs from step_C/matlab/.
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caz.io.trace import load_trace_var, trace_time_to_year_month

OUT_DIR = Path(r"D:\Dataset\DPastCliM-NA\verification\step_C\python")
ML_DIR  = Path(r"D:\Dataset\DPastCliM-NA\verification\step_C\matlab")
TRACE_TAS = Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc")
OBS_PQ = Path(r"D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_obs.parquet")
META_PQ = Path(r"D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet")

YEAR_MIN, YEAR_MAX = 1875, 1999
N_PC = 5
TRAIN_FRAC = 0.7
RNG_SEED = 2026
TARGET_MONTH = 1


def read_bin_vec(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        (n,) = struct.unpack("i", f.read(4))
        return np.frombuffer(f.read(), dtype=np.float32)[:n]


def read_bin_dvec(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        (n,) = struct.unpack("i", f.read(4))
        return np.frombuffer(f.read(), dtype=np.float64)[:n]


def read_bin_mat(p: Path, dtype=np.float32) -> np.ndarray:
    with open(p, "rb") as f:
        r, c = struct.unpack("ii", f.read(8))
        return np.frombuffer(f.read(), dtype=dtype).reshape(r, c)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load ESM grid field ----
    print("Loading TraCE tas for PCA...")
    da = load_trace_var(TRACE_TAS, "tas")
    lon0 = da["lon"].values
    if lon0.max() > 180:
        new_lon = ((lon0 + 180) % 360) - 180
        da = da.assign_coords(lon=new_lon).sortby("lon")
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.sortby("lat")
    year, month = trace_time_to_year_month(da["time"].values)
    da = da.assign_coords(year=("time", year), month_of_year=("time", month))

    # apply NA window (matching MATLAB: lon [-185,-45], lat all)
    from caz.io.trace import select_na_window
    da = select_na_window(da, lon_min=-180, lon_max=-50, lat_min=-90, lat_max=90, pad=5.0)

    cal_mask = (year >= YEAR_MIN) & (year <= YEAR_MAX) & (month == TARGET_MONTH)
    # recalculate year/month after spatial subset (time dim unchanged)
    da_cal = da.isel(time=cal_mask).compute()
    years_jan_cal = year[cal_mask]
    print(f"  Jan cal steps: {cal_mask.sum()}")

    arr = da_cal.values.astype(np.float32)  # (T, ny, nx)
    T_cal, ny, nx = arr.shape
    # flatten with lat-first order to match MATLAB (column-major: lat varies fastest)
    field_2d = arr.transpose(0, 2, 1).reshape(T_cal, nx * ny)  # (T, nx*ny) lat-first
    print(f"  field_2d: {field_2d.shape}")

    # ---- PCA ----
    mu = field_2d.mean(axis=0)
    Xc = field_2d - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores_full = U * s
    var_frac = (s**2) / max(np.sum(s**2), 1e-30)

    eofs = Vt[:N_PC]
    pcs = scores_full[:, :N_PC]
    ev_top = var_frac[:N_PC]
    print(f"PCA: top-{N_PC} EV: {ev_top * 100}")

    # save PCA outputs
    with open(OUT_DIR / "eigenvalues.bin", "wb") as f:
        f.write(struct.pack("i", len(var_frac)))
        f.write(var_frac.astype(np.float64).tobytes())
    with open(OUT_DIR / "pc_scores.bin", "wb") as f:
        f.write(struct.pack("ii", T_cal, N_PC))
        f.write(pcs.astype(np.float32).tobytes())
    with open(OUT_DIR / "field_mean.bin", "wb") as f:
        f.write(struct.pack("i", len(mu)))
        f.write(mu.astype(np.float32).tobytes())

    # ---- station obs ----
    print("Preparing station observations...")
    obs = pd.read_parquet(OBS_PQ)
    obs_jan = obs[(obs["month"] == TARGET_MONTH) &
                  (obs["year"] >= YEAR_MIN) & (obs["year"] <= YEAR_MAX)]
    station_years = obs_jan.pivot_table(index="year", columns="ID", values="value", aggfunc="first")
    common_years = station_years.index.intersection(pd.Index(years_jan_cal))
    station_years = station_years.loc[common_years]

    # align PCs
    pc_idx = pd.Index(years_jan_cal).get_indexer(common_years)
    pcs_aligned = pcs[pc_idx]

    # filter >= 30 valid years
    valid_count = station_years.notna().sum()
    keep = valid_count >= 30
    station_years = station_years.loc[:, keep]
    station_ids = station_years.columns.values
    n_stations = len(station_ids)
    Y = station_years.values.astype(np.float32)
    print(f"  stations >= 30 years: {n_stations}")

    # ---- train/test split (deterministic: first 70% years for train) ----
    T_common = Y.shape[0]
    n_train = max(int(T_common * TRAIN_FRAC), 30)
    idx_train = np.arange(n_train)
    idx_test = np.arange(n_train, T_common)
    perm = np.arange(T_common)

    X_tr = np.column_stack([np.ones(n_train, dtype=np.float32), pcs_aligned[idx_train]])
    X_te = np.column_stack([np.ones(len(idx_test), dtype=np.float32), pcs_aligned[idx_test]])
    Y_tr = Y[idx_train]
    Y_te = Y[idx_test]

    # ---- per-station OLS ----
    print("Running per-station OLS (with intercept)...")
    beta = np.full((N_PC + 1, n_stations), np.nan, dtype=np.float32)
    rmse_train = np.full(n_stations, np.nan, dtype=np.float32)
    rmse_test = np.full(n_stations, np.nan, dtype=np.float32)
    r2_train = np.full(n_stations, np.nan, dtype=np.float32)

    for s in range(n_stations):
        ys = Y_tr[:, s]
        m = np.isfinite(ys)
        if m.sum() < 20:
            continue
        b, _, _, _ = np.linalg.lstsq(X_tr[m], ys[m], rcond=None)
        beta[:, s] = b
        yhat = X_tr[m] @ b
        resid = ys[m] - yhat
        rmse_train[s] = float(np.sqrt(np.mean(resid**2)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((ys[m] - ys[m].mean()) ** 2))
        r2_train[s] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        ys_te = Y_te[:, s]
        mt = np.isfinite(ys_te)
        if mt.any():
            pred_te = X_te[mt] @ b
            rmse_test[s] = float(np.sqrt(np.mean((pred_te - ys_te[mt]) ** 2)))

    print(f"  median RMSE train: {np.nanmedian(rmse_train):.4f}")
    print(f"  median RMSE test:  {np.nanmedian(rmse_test):.4f}")
    print(f"  median R2 train:   {np.nanmedian(r2_train):.4f}")

    # save
    with open(OUT_DIR / "beta.bin", "wb") as f:
        f.write(struct.pack("ii", N_PC + 1, n_stations))
        f.write(beta.tobytes())
    with open(OUT_DIR / "rmse_train.bin", "wb") as f:
        f.write(struct.pack("i", n_stations))
        f.write(rmse_train.tobytes())
    with open(OUT_DIR / "rmse_test.bin", "wb") as f:
        f.write(struct.pack("i", n_stations))
        f.write(rmse_test.tobytes())
    with open(OUT_DIR / "split_perm.bin", "wb") as f:
        f.write(struct.pack("i", len(perm)))
        f.write(perm.astype(np.int32).tobytes())
    with open(OUT_DIR / "station_ids.txt", "w") as f:
        for sid in station_ids:
            f.write(f"{sid}\n")

    print(f"\nPython outputs saved to {OUT_DIR}")

    # ---- compare with MATLAB ----
    print("\n" + "=" * 50)
    print("Comparison: Python vs MATLAB")
    print("=" * 50)

    if not (ML_DIR / "eigenvalues.bin").exists():
        print("MATLAB outputs not found — run verify_C_pcr_calibration.m first")
        return

    # eigenvalues
    ml_ev = read_bin_dvec(ML_DIR / "eigenvalues.bin")
    py_ev = var_frac
    n_comp = min(len(ml_ev), len(py_ev), 20)
    print(f"\nEigenvalues (top {n_comp}):")
    print(f"  max |diff|:  {np.max(np.abs(py_ev[:n_comp] - ml_ev[:n_comp])):.2e}")
    print(f"  mean |diff|: {np.mean(np.abs(py_ev[:n_comp] - ml_ev[:n_comp])):.2e}")

    # PC scores (note: sign may be flipped)
    ml_pcs = read_bin_mat(ML_DIR / "pc_scores.bin")
    py_pcs = pcs.astype(np.float32)
    print(f"\nPC scores shape: Python {py_pcs.shape}, MATLAB {ml_pcs.shape}")
    if py_pcs.shape == ml_pcs.shape:
        for k in range(N_PC):
            corr = np.corrcoef(py_pcs[:, k], ml_pcs[:, k])[0, 1]
            diff = np.abs(py_pcs[:, k] - ml_pcs[:, k])
            # check if sign-flipped
            corr_flip = np.corrcoef(py_pcs[:, k], -ml_pcs[:, k])[0, 1]
            sign_note = " (sign-flipped)" if abs(corr_flip) > abs(corr) else ""
            print(f"  PC{k+1}: corr={max(abs(corr),abs(corr_flip)):.6f}, "
                  f"max|diff|={diff.max():.4f}{sign_note}")

    # field mean
    ml_mu = read_bin_vec(ML_DIR / "field_mean.bin")
    py_mu = mu.astype(np.float32)
    if len(ml_mu) == len(py_mu):
        dm = np.abs(py_mu - ml_mu)
        print(f"\nField mean: max|diff|={dm.max():.6f}, mean|diff|={dm.mean():.6f}")

    # station IDs overlap
    ml_ids = [l.strip() for l in open(ML_DIR / "station_ids.txt").readlines()]
    py_ids = list(station_ids)
    common_ids = set(py_ids) & set(ml_ids)
    print(f"\nStations: Python {len(py_ids)}, MATLAB {len(ml_ids)}, "
          f"overlap {len(common_ids)}")

    # RMSE comparison for common stations
    if common_ids:
        ml_rmse_tr = read_bin_vec(ML_DIR / "rmse_train.bin")
        py_order = [py_ids.index(sid) for sid in ml_ids if sid in common_ids]
        ml_order = [ml_ids.index(sid) for sid in ml_ids if sid in common_ids]
        py_rmse_sub = rmse_train[py_order[:len(ml_order)]]
        ml_rmse_sub = ml_rmse_tr[ml_order[:len(py_order)]]
        d = np.abs(py_rmse_sub - ml_rmse_sub)
        valid = np.isfinite(d)
        if valid.any():
            print(f"\nRMSE train (common stations):")
            print(f"  max|diff|:  {d[valid].max():.6f}")
            print(f"  mean|diff|: {d[valid].mean():.6f}")


if __name__ == "__main__":
    main()
