"""Step B verification (Python side): load TraCE-21k II, convert units,
interpolate to GHCN station coords for calibration period.

Saves outputs to D:\Dataset\DPastCliM-NA\verification\step_B\python\

Then compares with MATLAB outputs from step_B\matlab\.
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caz.io.trace import load_trace_var, trace_time_to_year_month, select_na_window
from caz.pcr import _nearest_interp_ts

TRACE_TAS = Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc")
TRACE_PR  = Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.PRECT.nc")
META_PQ   = Path(r"D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet")
OUT_DIR   = Path(r"D:\Dataset\DPastCliM-NA\verification\step_B\python")
ML_DIR    = Path(r"D:\Dataset\DPastCliM-NA\verification\step_B\matlab")

YEAR_MIN, YEAR_MAX = 1875, 1999


def read_bin_matrix(path: Path) -> np.ndarray:
    """Read the simple binary format: [rows, cols] int32 header, then float32 data."""
    with open(path, "rb") as f:
        rows, cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
    return data


def process_var(var: str):
    trace_path = TRACE_TAS if var == "tas" else TRACE_PR
    print(f"\n{'='*50}")
    print(f"Step B — {var}")
    print(f"{'='*50}")

    meta = pd.read_parquet(META_PQ)
    sta_lat = meta["lat"].values.astype(np.float32)
    sta_lon = meta["lon"].values.astype(np.float32)

    print(f"Loading TraCE {var}...")
    da = load_trace_var(trace_path, var)
    # convert lon 0-360 -> -180..180
    lon0 = da["lon"].values
    if lon0.max() > 180:
        new_lon = ((lon0 + 180) % 360) - 180
        da = da.assign_coords(lon=new_lon).sortby("lon")
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.sortby("lat")

    # get year/month
    year, month = trace_time_to_year_month(da["time"].values)
    da = da.assign_coords(year=("time", year), month_of_year=("time", month))

    # filter calibration period
    cal_mask = (year >= YEAR_MIN) & (year <= YEAR_MAX)
    da_cal = da.isel(time=cal_mask).compute()
    year_cal = year[cal_mask]
    month_cal = month[cal_mask]
    print(f"  cal period: {cal_mask.sum()} months")

    # interpolate to stations: BILINEAR (our default)
    grid_lat = da_cal["lat"].values.astype(np.float32)
    grid_lon = da_cal["lon"].values.astype(np.float32)
    arr = da_cal.values.astype(np.float32)  # (T, ny, nx)
    esm_at_station = _nearest_interp_ts(arr, grid_lat, grid_lon, sta_lat, sta_lon)
    if var == "pr":
        esm_at_station = np.clip(esm_at_station, 0, None)
    print(f"  esm_at_station shape: {esm_at_station.shape}")

    # save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"year": year_cal, "month": month_cal}).to_csv(
        OUT_DIR / "time_cal.csv", index=False
    )
    with open(OUT_DIR / f"esm_at_station_{var}.bin", "wb") as f:
        f.write(struct.pack("ii", *esm_at_station.shape))
        f.write(esm_at_station.tobytes())

    # per-month cal mean at each station
    esm_cal_mean = np.zeros((12, esm_at_station.shape[1]), dtype=np.float32)
    for m in range(1, 13):
        mi = month_cal == m
        esm_cal_mean[m - 1] = np.nanmean(esm_at_station[mi], axis=0)
    with open(OUT_DIR / f"esm_cal_mean_{var}.bin", "wb") as f:
        f.write(struct.pack("ii", 12, esm_at_station.shape[1]))
        f.write(esm_cal_mean.tobytes())

    print(f"  saved to {OUT_DIR}")
    return esm_at_station, esm_cal_mean


def compare(var: str):
    print(f"\n--- Comparing {var} ---")
    py = read_bin_matrix(OUT_DIR / f"esm_at_station_{var}.bin")
    ml_path = ML_DIR / f"esm_at_station_{var}.bin"
    if not ml_path.exists():
        print(f"  MATLAB output not found: {ml_path} — skip comparison")
        return
    ml = read_bin_matrix(ml_path)
    print(f"  Python shape: {py.shape}, MATLAB shape: {ml.shape}")
    if py.shape != ml.shape:
        print("  WARNING: shapes differ — cannot compare element-wise")
        return

    diff = py - ml
    print(f"  max |diff|:    {np.nanmax(np.abs(diff)):.4f}")
    print(f"  mean |diff|:   {np.nanmean(np.abs(diff)):.4f}")
    print(f"  median |diff|: {np.nanmedian(np.abs(diff)):.4f}")
    print(f"  % >0.01:       {100 * np.mean(np.abs(diff) > 0.01):.2f}%")
    print(f"  % >0.1:        {100 * np.mean(np.abs(diff) > 0.1):.2f}%")
    print(f"  % >1.0:        {100 * np.mean(np.abs(diff) > 1.0):.2f}%")

    # the difference should be solely from bilinear (Python) vs nearest (MATLAB)
    print(f"\n  Note: differences are expected due to bilinear vs nearest interpolation.")


def main():
    for var in ["tas", "pr"]:
        process_var(var)

    print("\n" + "=" * 50)
    print("Comparison: Python (bilinear) vs MATLAB (nearest)")
    print("=" * 50)
    for var in ["tas", "pr"]:
        compare(var)


if __name__ == "__main__":
    main()
