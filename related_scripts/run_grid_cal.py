"""Step 4-cal: Grid cal-period station fields to 0.20° target grid.

Grids three products (no variance correction — that is Step 5-cal):
  1. grid_obs_test_{var}.nc  — GHCN non-removed stations (test years only)
  2. grid_pcr_test_{var}.nc  — PCR predictions (test years only), raw
  3. grid_esm_test_{var}.nc  — TraCE-21k II regridded (test years only)

Matches Guaita's PCR_downscaling_gridding_v1.m + _obs.m:
  - Albers EPSG:5070 projection
  - Sibson natural-neighbor + nearest extrapolation
  - Land mask applied

Optimization: groups timesteps by station set, builds weight matrix once per
group, then applies via sparse matrix multiply for all timesteps in that group.

Run:
  python related_scripts/run_grid_cal.py --var tas
  python related_scripts/run_grid_cal.py --var pr
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time as _t
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm

from caz.gridding import GridSpec, _albers_forward
from caz.natneighbor import sibson_weight_matrix

DATA_ROOT = Path(r"D:\Dataset\DPastCliM-NA")
GHCN_DIR = DATA_ROOT / "GHCN" / "interim"
MASK_PATH = DATA_ROOT / "static" / "landmask_NA_020.nc"

GCM_CONFIG = {
    "trace21k": {
        "year_cal_max": 1999,
        "esm_paths": {
            "tas": DATA_ROOT / "TraCE21k" / "TraCE-21K-II.monthly.TREFHT.nc",
            "pr":  DATA_ROOT / "TraCE21k" / "TraCE-21K-II.monthly.PRECT.nc",
        },
    },
    "mpi-esm-cr": {
        "year_cal_max": 1949,
        "esm_dir": DATA_ROOT / "MPI-ESM-CR",
    },
}

GRID_SPEC = GridSpec(
    lat_min=15.0, lat_max=75.0,
    lon_min=-170.0, lon_max=-50.0,
    res_deg=0.20,
)

YEAR_CAL_MIN = 1875
MIN_STATION_FRAC = 0.05


def _load_split(model_dir: Path, var: str):
    with open(model_dir / "split_calibration.pkl", "rb") as f:
        sp = pickle.load(f)[var]
    return set(sp["test_years"]), sp["station_flags"]


def _load_landmask(glat, glon):
    if not MASK_PATH.exists():
        return None
    mds = xr.open_dataset(MASK_PATH)
    mask = mds["mask"].astype(float).interp(lat=glat, lon=glon, method="nearest").values > 0.5
    mds.close()
    return mask


def _batched_grid(timestep_list, spec, mask, desc="Grid"):
    """Grid multiple timesteps, caching weight matrices by station set.

    Parameters
    ----------
    timestep_list : list of (ti, values_1d, lat_1d, lon_1d)
    spec : GridSpec
    mask : (ny, nx) bool or None

    Returns
    -------
    results : dict of ti -> (ny, nx) float32
    """
    glat, glon = spec.lat_lon()
    ny, nx = len(glat), len(glon)

    in_box_lat = (spec.lat_min - 5, spec.lat_max + 5)
    in_box_lon = (spec.lon_min - 5, spec.lon_max + 5)

    longrid, latgrid = np.meshgrid(glon, glat)
    x_grd, y_grd = _albers_forward(longrid.ravel(), latgrid.ravel())
    qpts = np.column_stack([x_grd, y_grd])

    groups = defaultdict(list)
    for ti, vals, lat, lon in timestep_list:
        box = (
            (lat >= in_box_lat[0]) & (lat <= in_box_lat[1])
            & (lon >= in_box_lon[0]) & (lon <= in_box_lon[1])
        )
        lat_b, lon_b, vals_b = lat[box], lon[box], vals[box]
        key = hash(lat_b.astype(np.float32).tobytes() + lon_b.astype(np.float32).tobytes())
        groups[key].append((ti, vals_b, lat_b, lon_b))

    print(f"  {len(groups)} unique station sets for {len(timestep_list)} timesteps", flush=True)

    results = {}
    pbar = tqdm(total=len(timestep_list), desc=desc)

    for group_items in groups.values():
        lat0 = group_items[0][2].astype(np.float32)
        lon0 = group_items[0][3].astype(np.float32)
        x_st, y_st = _albers_forward(lon0, lat0)
        st_pts = np.column_stack([x_st, y_st])

        pt_map = defaultdict(list)
        for i in range(len(st_pts)):
            k = (round(st_pts[i, 0], 6), round(st_pts[i, 1], 6))
            pt_map[k].append(i)

        dedup_pts = []
        dedup_to_orig = []
        for k, indices in pt_map.items():
            dedup_pts.append(st_pts[indices[0]])
            dedup_to_orig.append(indices)
        dedup_pts_arr = np.array(dedup_pts)

        W = sibson_weight_matrix(dedup_pts_arr, qpts)

        nnz_per_row = np.asarray(W.getnnz(axis=1)).ravel()
        hull_nans_mask = (nnz_per_row == 0)
        near_idx = None
        if hull_nans_mask.any():
            near = NearestNDInterpolator(st_pts, np.arange(len(x_st)))
            near_idx = near(qpts).astype(int)

        n_dedup = len(dedup_pts)

        for ti, vals_b, _, _ in group_items:
            vals_dedup = np.empty(n_dedup, dtype=np.float64)
            for di, orig_indices in enumerate(dedup_to_orig):
                vals_dedup[di] = np.mean(vals_b[orig_indices])

            row = (W @ vals_dedup).astype(np.float32)

            first_valid = row.copy()
            nans = np.isnan(first_valid) | hull_nans_mask
            if nans.any() and near_idx is not None:
                first_valid[nans] = vals_b[near_idx[nans]].astype(np.float32)

            g = first_valid.reshape(ny, nx)
            if mask is not None:
                g = np.where(mask, g, np.nan)
            results[ti] = g
            pbar.update(1)

    pbar.close()
    return results


def main():
    from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month
    from caz.io.mpi_esm import load_mpi_esm_var, mpi_esm_time_to_year_month

    ap = argparse.ArgumentParser(description="Step 4-cal: Grid cal-period fields")
    ap.add_argument("--var", choices=["tas", "pr"], required=True)
    ap.add_argument("--gcm", choices=list(GCM_CONFIG.keys()), default="trace21k")
    args = ap.parse_args()
    var = args.var
    gcm = args.gcm
    cfg = GCM_CONFIG[gcm]

    OUT_DIR     = DATA_ROOT / "interim" / gcm / "grid_cal"
    STATION_DIR = DATA_ROOT / "interim" / gcm / "station_cal"
    MODEL_DIR   = DATA_ROOT / "interim" / gcm / "models"

    timings = {}
    print("=" * 60)
    print(f"Step 4-cal: Grid Cal-Period Fields -- {var} [{gcm}]")
    print("=" * 60)
    t_total = _t.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    glat, glon = GRID_SPEC.lat_lon()
    ny, nx = len(glat), len(glon)
    mask = _load_landmask(glat, glon)
    print(f"  Grid: {ny}x{nx}, mask: {mask is not None}", flush=True)

    test_years, station_flags = _load_split(MODEL_DIR, var)
    print(f"  Test years: {sorted(test_years)}", flush=True)

    # ── 1. Grid test observations ──────────────────────────────
    print(f"\n[1/3] Gridding test observations ...", flush=True)
    t0 = _t.time()

    obs = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_obs.parquet")
    meta = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_meta.parquet")
    meta_dict = meta.set_index("ID")[["lat", "lon"]].to_dict("index")

    valid_sids = {sid for sid, flag in station_flags.items() if flag != "removed"}
    obs_test = obs[obs["ID"].isin(valid_sids) & obs["year"].isin(test_years)].copy()
    obs_test["lat"] = obs_test["ID"].map(lambda s: meta_dict.get(s, {}).get("lat", np.nan))
    obs_test["lon"] = obs_test["ID"].map(lambda s: meta_dict.get(s, {}).get("lon", np.nan))
    obs_test = obs_test.dropna(subset=["lat", "lon", "value"])

    n_total_stations = len(valid_sids)
    min_stations = int(n_total_stations * MIN_STATION_FRAC)
    print(f"  Obs: {len(obs_test):,} rows, {obs_test['ID'].nunique()} stations", flush=True)
    print(f"  Min stations per timestep: {min_stations}", flush=True)

    test_ym = obs_test.groupby(["year", "month"]).size().reset_index()[["year", "month"]].values
    test_ym = test_ym[np.lexsort((test_ym[:, 1], test_ym[:, 0]))]
    n_test_times = len(test_ym)

    ts_data = []
    for ti, (yr, mo) in enumerate(test_ym):
        sub = obs_test[(obs_test["year"] == yr) & (obs_test["month"] == mo)]
        if len(sub) < min_stations:
            continue
        ts_data.append((
            ti,
            sub["value"].values.astype(np.float32),
            sub["lat"].values.astype(np.float32),
            sub["lon"].values.astype(np.float32),
        ))

    grid_obs_test = np.full((n_test_times, ny, nx), np.nan, dtype=np.float32)
    obs_results = _batched_grid(ts_data, GRID_SPEC, mask, desc="Grid obs_test")
    for ti, g in obs_results.items():
        grid_obs_test[ti] = g

    timings["grid_obs_test"] = _t.time() - t0
    print(f"  Done: {n_test_times} timesteps ({timings['grid_obs_test']:.1f}s)", flush=True)

    obs_test_nc = OUT_DIR / f"grid_obs_test_{var}.nc"
    ds_obs = xr.Dataset(
        {var: (("time", "lat", "lon"), grid_obs_test)},
        coords={
            "time": np.arange(n_test_times),
            "year": ("time", test_ym[:, 0].astype(np.int32)),
            "month": ("time", test_ym[:, 1].astype(np.int32)),
            "lat": glat, "lon": glon,
        },
    )
    enc = {var: {"dtype": "float32", "zlib": True, "complevel": 4}}
    ds_obs.to_netcdf(obs_test_nc, encoding=enc)
    obs_mb = obs_test_nc.stat().st_size / 1e6
    print(f"  Saved: {obs_test_nc} ({obs_mb:.0f} MB)", flush=True)
    del grid_obs_test, ds_obs, obs_test

    # ── 2. Grid PCR predictions (test years only) ─────────────
    print(f"\n[2/3] Gridding PCR predictions (test years only) ...", flush=True)
    t0 = _t.time()

    recon = pd.read_csv(STATION_DIR / f"recon_cal_{var}.csv",
                        usecols=["station_id", "lon", "lat", "year", "month", "value"])
    recon_test = recon[recon["year"].isin(test_years)].copy()
    del recon

    ts_data_pcr = []
    for ti, (yr, mo) in enumerate(test_ym):
        sub = recon_test[(recon_test["year"] == yr) & (recon_test["month"] == mo)]
        sub = sub.dropna(subset=["value"])
        if len(sub) < min_stations:
            continue
        ts_data_pcr.append((
            ti,
            sub["value"].values.astype(np.float32),
            sub["lat"].values.astype(np.float32),
            sub["lon"].values.astype(np.float32),
        ))

    grid_pcr = np.full((n_test_times, ny, nx), np.nan, dtype=np.float32)
    pcr_results = _batched_grid(ts_data_pcr, GRID_SPEC, mask, desc="Grid PCR")
    for ti, g in pcr_results.items():
        grid_pcr[ti] = g

    timings["grid_pcr"] = _t.time() - t0
    print(f"  Done: {n_test_times} timesteps ({timings['grid_pcr']:.1f}s)", flush=True)
    del recon_test

    # ── 3. Grid ESM (test years only) ───────────────────────
    print(f"\n[3/3] Gridding ESM (test years only) ...", flush=True)
    t0 = _t.time()

    if gcm == "trace21k":
        tr = load_trace_var(cfg["esm_paths"][var], var)
        tr = select_na_window(tr)
        yr_tr, mo_tr = trace_time_to_year_month(tr["time"].values)
    else:
        tr = load_mpi_esm_var(cfg["esm_dir"] / var, var,
                              year_min_ce=YEAR_CAL_MIN, year_max_ce=cfg["year_cal_max"])
        tr = select_na_window(tr)
        yr_tr, mo_tr = mpi_esm_time_to_year_month(tr["time"].values)

    from scipy.interpolate import RegularGridInterpolator
    glon_mesh, glat_mesh = np.meshgrid(glon, glat)
    pts = np.column_stack([glat_mesh.ravel(), glon_mesh.ravel()])

    grid_esm = np.full((n_test_times, ny, nx), np.nan, dtype=np.float32)
    for ti, (yr, mo) in enumerate(tqdm(test_ym, desc="Regrid ESM")):
        idx = np.where((yr_tr == yr) & (mo_tr == mo))[0]
        if len(idx) == 0:
            continue
        esm_slice = tr.isel(time=int(idx[0])).values
        interp_fn = RegularGridInterpolator(
            (tr.lat.values, tr.lon.values), esm_slice, method="nearest",
            bounds_error=False, fill_value=np.nan)
        g = interp_fn(pts).reshape(ny, nx).astype(np.float32)
        if mask is not None:
            g = np.where(mask, g, np.nan)
        grid_esm[ti] = g

    timings["grid_esm"] = _t.time() - t0
    print(f"  ESM regridded ({timings['grid_esm']:.1f}s)", flush=True)

    # Save
    pcr_nc = OUT_DIR / f"grid_pcr_test_{var}.nc"
    esm_nc = OUT_DIR / f"grid_esm_test_{var}.nc"

    coords = {
        "time": np.arange(n_test_times),
        "year": ("time", test_ym[:, 0].astype(np.int32)),
        "month": ("time", test_ym[:, 1].astype(np.int32)),
        "lat": glat, "lon": glon,
    }
    enc_f32 = {var: {"dtype": "float32", "zlib": True, "complevel": 4}}

    ds_pcr = xr.Dataset({var: (("time", "lat", "lon"), grid_pcr)}, coords=coords)
    ds_pcr.to_netcdf(pcr_nc, encoding=enc_f32)
    pcr_mb = pcr_nc.stat().st_size / 1e6

    ds_esm = xr.Dataset({var: (("time", "lat", "lon"), grid_esm)}, coords=coords)
    ds_esm.to_netcdf(esm_nc, encoding=enc_f32)
    esm_mb = esm_nc.stat().st_size / 1e6

    timings["save"] = _t.time() - t0
    timings["total"] = _t.time() - t_total

    print("\n" + "=" * 60)
    print(f"DONE: {var} test-period gridding (Step 4)")
    print(f"  Obs test:  {obs_test_nc}  ({obs_mb:.0f} MB)")
    print(f"  PCR test:  {pcr_nc}  ({pcr_mb:.0f} MB)")
    print(f"  ESM test:  {esm_nc}  ({esm_mb:.0f} MB)")
    print(f"  Time: {timings['total']:.1f}s total")
    print("=" * 60, flush=True)

    timing_path = OUT_DIR / f"timing_grid_cal_{var}.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)


if __name__ == "__main__":
    main()
