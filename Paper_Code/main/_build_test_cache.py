"""Build cached test-set merged data for all validation tables.

Merges recon_cal predictions (test split) with GHCN obs and ESM values,
saves as parquet for fast reuse by T2, T4-5 scripts.

Usage: python _build_test_cache.py [--gcm {trace21k,mpi-esm-cr}]
Output: {station_cal_dir}/test_cache_{var}.parquet
"""
import sys, argparse, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import numpy as np
import pandas as pd
from _common import GHCN_DIR, DATA_DIR

GCM_CONFIG = {
    "trace21k":   {"year_cal_max": 1999},
    "mpi-esm-cr": {"year_cal_max": 1949},
}


def _load_esm_at_stations(gcm_name, var, sids, lats, lons, years, months,
                           year_cal_max):
    from caz.io.trace import select_na_window

    if gcm_name == "trace21k":
        from caz.io.trace import load_trace_var, trace_time_to_year_month
        esm_paths = {
            "tas": DATA_DIR / "TraCE21k" / "TraCE-21K-II.monthly.TREFHT.nc",
            "pr":  DATA_DIR / "TraCE21k" / "TraCE-21K-II.monthly.PRECT.nc",
        }
        tr = load_trace_var(esm_paths[var], var)
        tr = select_na_window(tr)
        yr, mo = trace_time_to_year_month(tr["time"].values)
    else:
        from caz.io.mpi_esm import load_mpi_esm_var, mpi_esm_time_to_year_month
        esm_dir = DATA_DIR / "MPI-ESM-CR"
        tr = load_mpi_esm_var(esm_dir / var, var,
                              year_min_ce=1875, year_max_ce=year_cal_max)
        tr = select_na_window(tr)
        yr, mo = mpi_esm_time_to_year_month(tr["time"].values)

    tr = tr.assign_coords(year=("time", yr), month=("time", mo))
    cal_mask = (tr.year >= 1875) & (tr.year <= year_cal_max)
    tr_cal = tr.isel(time=cal_mask.values).load()

    yr_arr, mo_arr = tr_cal.year.values, tr_cal.month.values
    ym_to_tidx = {}
    for t_i in range(len(yr_arr)):
        ym_to_tidx[(int(yr_arr[t_i]), int(mo_arr[t_i]))] = t_i

    unique_sids = np.unique(sids)
    sid_to_lat = dict(zip(sids, lats))
    sid_to_lon = dict(zip(sids, lons))

    esm_vals = np.full(len(sids), np.nan, dtype=np.float32)
    sid_indices = {}
    for i, s in enumerate(sids):
        sid_indices.setdefault(s, []).append(i)

    for sid in unique_sids:
        esm_point = tr_cal.sel(lat=sid_to_lat[sid], lon=sid_to_lon[sid],
                               method="nearest").values
        for idx in sid_indices[sid]:
            y, m = int(years[idx]), int(months[idx])
            t_i = ym_to_tidx.get((y, m))
            if t_i is not None:
                esm_vals[idx] = esm_point[t_i]

    return esm_vals


def build_cache(var: str, gcm_name: str):
    cfg = GCM_CONFIG[gcm_name]
    station_cal = DATA_DIR / "interim" / gcm_name / "station_cal"
    model_dir = DATA_DIR / "interim" / gcm_name / "models"
    cache_path = station_cal / f"test_cache_{var}.parquet"

    print(f"\n=== Building test cache: {var} ({gcm_name}) ===", flush=True)

    with open(model_dir / "split_calibration.pkl", "rb") as f:
        split_info = pickle.load(f)[var]
    test_years = set(int(y) for y in split_info["test_years"])

    print("  Loading recon (test years) ...", flush=True)
    recon = pd.read_csv(station_cal / f"recon_cal_{var}.csv",
                        usecols=["station_id", "lon", "lat", "year", "month",
                                 "value", "pi_lo", "pi_hi", "value_real"])
    recon_test = recon[recon["year"].isin(test_years)].copy()
    print(f"  {len(recon_test):,} test rows", flush=True)

    print("  Loading obs ...", flush=True)
    obs = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_obs.parquet")
    obs_sub = obs[["ID", "year", "month", "value"]].rename(
        columns={"ID": "station_id", "value": "obs"})
    merged = recon_test.merge(obs_sub, on=["station_id", "year", "month"],
                              how="inner")
    print(f"  {len(merged):,} matched rows", flush=True)

    print("  Loading ESM ...", flush=True)
    merged["esm"] = _load_esm_at_stations(
        gcm_name, var,
        merged["station_id"].values,
        merged["lat"].values.astype(float),
        merged["lon"].values.astype(float),
        merged["year"].values.astype(int),
        merged["month"].values.astype(int),
        cfg["year_cal_max"],
    )

    merged.to_parquet(cache_path, index=False)
    mb = cache_path.stat().st_size / 1e6
    print(f"  Saved: {cache_path} ({mb:.0f} MB, {len(merged):,} rows)")
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcm", default="trace21k", choices=list(GCM_CONFIG))
    args = ap.parse_args()

    for var in ["tas", "pr"]:
        build_cache(var, args.gcm)
    print("\nDone.")
