"""Station-level PCR demo: TraCE-21k II → GHCN tas reconstruction.

Pipeline:
  1. Load GHCN tas Parquet (already region-filtered to NA).
  2. Load TraCE TREFHT, NA window, regrid not needed (use coarse grid as-is).
  3. Build per-month annual time series:
       - obs[year,month,station] from GHCN
       - esm[year,month,lat,lon] from TraCE in the calibration period (1875-1999)
  4. Calibrate PCR per month.
  5. Project full 22 ka through trained EOFs+regression -> (time, station).
  6. Save:
       - models pickle  (interim/pcr_models_tas.pkl)
       - station-level reconstruction NetCDF (interim/recon_station_tas.nc)
       - calibration diagnostics CSV (interim/pcr_diag_tas.csv)

Run:
  C:\\Users\\isxzl\\miniconda3\\envs\\caz\\python.exe related_scripts\\run_pcr_station.py
"""
from __future__ import annotations

import argparse
import pickle
import time as _t
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caz.io.ghcn import filter_min_record, filter_time, subset_region_df
from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month
from caz.pcr import calibrate_month, predict_month


# ---------------------------------------------------------------- inputs/outputs
GHCN_OBS_TAS = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_tas_obs.parquet")
GHCN_OBS_PR  = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_pr_obs.parquet")
TRACE_T  = Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc")
TRACE_P  = Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.PRECT.nc")
OUT_DIR  = Path(r"D:\Dataset\DPastCliM-NA\interim\trace21k\models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_CAL_MIN = 1875
YEAR_CAL_MAX = 1999     # TraCE-21k II ends 1999
N_YEAR_VAL = 50
N_YEAR_TEST = 15


def build_esm_year_month_da(da_full: xr.DataArray) -> xr.DataArray:
    """Add 'year' and 'month' coords to TraCE time axis."""
    yr, mo = trace_time_to_year_month(da_full["time"].values)
    return da_full.assign_coords(year=("time", yr), month=("time", mo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["tas", "pr"], default="tas")
    args = ap.parse_args()
    var = args.var
    obs_path = GHCN_OBS_TAS if var == "tas" else GHCN_OBS_PR
    trace_path = TRACE_T if var == "tas" else TRACE_P

    print("=" * 60)
    print(f"Station-level PCR: TraCE-21k II → GHCN {var}")
    print("=" * 60)
    t0 = _t.time()

    # ---------------------------------------------------------------- 1. obs
    print(f"\n[1] loading GHCN obs from {obs_path}")
    obs = pd.read_parquet(obs_path)
    print(f"    raw rows: {len(obs):,} | stations: {obs['ID'].nunique():,}")
    obs = filter_time(obs, YEAR_CAL_MIN, YEAR_CAL_MAX)
    obs = filter_min_record(obs, min_years=20)
    print(f"    after time {YEAR_CAL_MIN}-{YEAR_CAL_MAX} + 20yr filter: "
          f"{len(obs):,} rows / {obs['ID'].nunique():,} stations")

    # pr transform is now handled inside calibrate_month (per-station O_t)

    # ---------------------------------------------------------------- 2. ESM
    print(f"\n[2] loading TraCE {trace_path.name}")
    tr = load_trace_var(trace_path, var)
    tr = select_na_window(tr)
    tr = build_esm_year_month_da(tr)
    print(f"    NA-window shape: {tr.shape}  | year range {int(tr.year.min())} - {int(tr.year.max())}")

    # eager load for speed (NA subset is small)
    tr_load = tr.load()

    # ---------------------------------------------------------------- 3. calib slice
    cal_mask = (tr_load.year >= YEAR_CAL_MIN) & (tr_load.year <= YEAR_CAL_MAX)
    tr_cal = tr_load.isel(time=cal_mask.values)
    print(f"    calib slice: {tr_cal.shape}  ({YEAR_CAL_MIN}-{YEAR_CAL_MAX})")

    # ---------------------------------------------------------------- 4. per-month PCR
    print("\n[3] per-month PCR calibration")
    rng = np.random.default_rng(2026)
    models = {}
    diag_rows = []

    for month in range(1, 13):
        m_mask = tr_cal.month.values == month
        esm_m_cal = tr_cal.isel(time=m_mask)
        esm_m_full = tr_load.isel(time=tr_load.month.values == month)
        if esm_m_cal.sizes["time"] < 30:
            print(f"  month {month}: not enough calib samples, skipped")
            continue

        try:
            mdl = calibrate_month(
                month=month,
                esm_da=esm_m_cal,
                obs_long=obs,
                var_name=var,
                n_year_val=N_YEAR_VAL,
                n_year_test=N_YEAR_TEST,
                rng=rng,
            )
        except ValueError as e:
            print(f"  month {month}: {e}")
            continue

        models[month] = mdl
        rmse_med = float(np.nanmedian(mdl.rmse_train))
        r2_med = float(np.nanmedian(mdl.r2_train))
        rmse_te_med = float(np.nanmedian(mdl.rmse_test)) if mdl.rmse_test is not None else np.nan
        ev_top = float(mdl.pc_var_frac[mdl.pc_indices].sum() * 100)
        print(
            f"  month {month:2d}: stations={len(mdl.station_id):>5,d}  "
            f"PCs={mdl.n_pc} ({ev_top:5.1f}% EV)  "
            f"RMSE train={rmse_med:5.2f}  test={rmse_te_med:5.2f}  R2={r2_med:5.3f}"
        )
        diag_rows.append({
            "month": month,
            "n_stations": int(len(mdl.station_id)),
            "rmse_train_median": rmse_med,
            "rmse_test_median": rmse_te_med,
            "r2_train_median": r2_med,
            "ev_topK_pct": ev_top,
        })

    if not models:
        raise RuntimeError("no months calibrated")

    diag_df = pd.DataFrame(diag_rows)
    diag_csv = OUT_DIR / f"pcr_diag_{var}.csv"
    diag_df.to_csv(diag_csv, index=False)
    print(f"\n    diagnostics -> {diag_csv}")
    print(diag_df.to_string(index=False))

    # ---------------------------------------------------------------- 5. project full transient
    print("\n[4] projecting full 22 ka transient (per month)")
    pred_per_month = []
    for month, mdl in models.items():
        esm_m = tr_load.isel(time=tr_load.month.values == month)
        m_mask_cal = tr_cal.month.values == month
        esm_m_cal = tr_cal.isel(time=m_mask_cal)
        pred = predict_month(mdl, esm_m, esm_da_cal=esm_m_cal)
        pred_per_month.append(pred)
        print(f"  month {month:2d}: predicted {pred.shape}")

    print("    concatenating along time...")
    recon = xr.concat(pred_per_month, dim="time").sortby("time")
    print(f"    final recon shape: {recon.shape}")

    # pr inverse transform is now handled inside predict_month
    if var == "pr":
        recon = recon.clip(min=0.0)
        recon.attrs["units"] = "mm/day"

    # ---------------------------------------------------------------- 6. save
    print("\n[5] saving outputs")
    var_name = f"{var}_recon"
    recon_nc = OUT_DIR / f"recon_station_{var}.nc"
    # tas: 0.01 degC resolution; pr: 0.01 mm/day resolution (max ~327 mm/day fits in int16)
    enc = {
        var_name: {
            "dtype": "int16",
            "scale_factor": 0.01,
            "add_offset": 0.0,
            "_FillValue": -32768,
            "zlib": True,
            "complevel": 5,
            "chunksizes": (1024, 256),
        }
    }
    # chunk to dask so xarray writes in streaming fashion (avoid full int16 materialization)
    ds = recon.to_dataset(name=var_name).chunk({"time": 12000, "station": 4096})
    ds.to_netcdf(recon_nc, mode="w", encoding=enc, compute=True)
    print(f"    {recon_nc}  ({recon_nc.stat().st_size / 1e6:.1f} MB)")

    models_pkl = OUT_DIR / f"pcr_models_{var}.pkl"
    with models_pkl.open("wb") as f:
        pickle.dump(models, f)
    print(f"    {models_pkl}  ({models_pkl.stat().st_size / 1e6:.1f} MB)")

    print(f"\nDone in {_t.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
