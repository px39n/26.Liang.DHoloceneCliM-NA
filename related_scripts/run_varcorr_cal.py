"""Step 5-cal: Grid full cal period, apply variance correction, save products.

Grids PCR + ESM for the full calibration period (1500 months),
applies 30-yr moving-window variance correction, saves:
  - grid_pcr_cal_{var}.nc  (variance-corrected, full cal period)
  - grid_esm_cal_{var}.nc  (regridded ESM, full cal period)
  - grid_pcr_vc_test_{var}.nc  (variance-corrected, test years only, for F5/F6)

Uses batching optimization: groups timesteps by station set,
computes Sibson weight matrix once per group.

Run:
  python related_scripts/run_varcorr_cal.py --var tas
  python related_scripts/run_varcorr_cal.py --var pr
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
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from tqdm import tqdm

from caz.gridding import GridSpec, _albers_forward, variance_correction
from caz.natneighbor import sibson_weight_matrix
from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month

OUT_DIR     = Path(r"D:\Dataset\DPastCliM-NA\interim\trace21k\grid_cal")
STATION_DIR = Path(r"D:\Dataset\DPastCliM-NA\interim\trace21k\station_cal")
MASK_PATH = Path(r"D:\Dataset\DPastCliM-NA\static\landmask_NA_020.nc")
TRACE = {
    "tas": Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc"),
    "pr":  Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.PRECT.nc"),
}

GRID_SPEC = GridSpec(
    lat_min=15.0, lat_max=75.0,
    lon_min=-170.0, lon_max=-50.0,
    res_deg=0.20,
)

YEAR_CAL_MIN = 1875
YEAR_CAL_MAX = 1999
VAR_CORR_WINDOW = 30 * 12


def _load_split(var: str):
    with open(STATION_DIR / "split_calibration.pkl", "rb") as f:
        sp = pickle.load(f)[var]
    return set(sp["test_years"]), sp["station_flags"]


def _load_landmask(glat, glon):
    if not MASK_PATH.exists():
        return None
    mds = xr.open_dataset(MASK_PATH)
    mask = mds["mask"].astype(float).interp(lat=glat, lon=glon, method="nearest").values > 0.5
    mds.close()
    return mask


def _batched_grid(timestep_list, spec, mask, qpts, x_grd, ny, nx, desc="Grid"):
    """Grid timesteps, caching weight matrices by station set.

    Uses batched sparse matrix-matrix multiply (W @ V_matrix) instead of
    repeated matrix-vector products for ~10x speedup on the matmul phase.
    """
    in_box_lat = (spec.lat_min - 5, spec.lat_max + 5)
    in_box_lon = (spec.lon_min - 5, spec.lon_max + 5)

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
    n_done = 0

    for gi, group_items in enumerate(groups.values()):
        t_grp = _t.time()
        n_ts = len(group_items)
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
        n_dedup = len(dedup_pts)

        # Pre-compute fast indexing for single-station dedup groups
        single_mask = np.array([len(g) == 1 for g in dedup_to_orig])
        single_idx = np.array([g[0] for g in dedup_to_orig if len(g) == 1])
        multi_groups = [(i, g) for i, g in enumerate(dedup_to_orig) if len(g) > 1]

        W = sibson_weight_matrix(dedup_pts_arr, qpts)

        nnz_per_row = np.asarray(W.getnnz(axis=1)).ravel()
        hull_nans_mask = (nnz_per_row == 0)
        near_idx = None
        if hull_nans_mask.any():
            near = NearestNDInterpolator(st_pts, np.arange(len(x_st)))
            near_idx = near(qpts).astype(int)

        # Build value matrix (n_dedup, n_ts) — vectorized for single-station groups
        V = np.empty((n_dedup, n_ts), dtype=np.float64)
        ti_list = []
        raw_vals_list = []
        for col, (ti, vals_b, _, _) in enumerate(group_items):
            ti_list.append(ti)
            raw_vals_list.append(vals_b)
            V[single_mask, col] = vals_b[single_idx]
            for di, g in multi_groups:
                V[di, col] = np.mean(vals_b[g])

        # Batched sparse matmul: (M, n_ts) = (M, N) @ (N, n_ts)
        G = (W @ V).astype(np.float32)  # (M, n_ts)

        # Post-process: hull extrapolation + land mask
        for col in range(n_ts):
            row = G[:, col]
            nans = np.isnan(row) | hull_nans_mask
            if nans.any() and near_idx is not None:
                row[nans] = raw_vals_list[col][near_idx[nans]].astype(np.float32)

            g = row.reshape(ny, nx)
            if mask is not None:
                g = np.where(mask, g, np.nan)
            results[ti_list[col]] = g

        n_done += n_ts
        dt = _t.time() - t_grp
        print(f"  {desc} group {gi+1}/{len(groups)}: {n_ts} timesteps, "
              f"{n_dedup} stations, {dt:.1f}s ({n_done}/{len(timestep_list)} done)",
              flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["tas", "pr"], required=True)
    args = ap.parse_args()
    var = args.var

    timings = {}
    print("=" * 60)
    print(f"Step 5-cal: Full-period gridding + variance correction -- {var}")
    print("=" * 60)
    t_total = _t.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    glat, glon = GRID_SPEC.lat_lon()
    ny, nx = len(glat), len(glon)
    mask = _load_landmask(glat, glon)

    longrid, latgrid = np.meshgrid(glon, glat)
    x_grd, y_grd = _albers_forward(longrid.ravel(), latgrid.ravel())
    qpts = np.column_stack([x_grd, y_grd])

    test_years, station_flags = _load_split(var)

    # ── 1. Grid PCR for full cal period ──────────────────────
    print(f"\n[1/3] Gridding PCR (full cal period) ...", flush=True)
    t0 = _t.time()

    recon = pd.read_csv(STATION_DIR / f"recon_cal_{var}.csv",
                        usecols=["station_id", "lon", "lat", "year", "month", "value"])

    # Build timestep list via groupby (65x faster than per-timestep boolean filter)
    grouped = recon.groupby(["year", "month"], sort=False)
    sorted_groups = sorted(grouped, key=lambda x: (x[0][0], x[0][1]))
    all_ym = np.array([(yr, mo) for (yr, mo), _ in sorted_groups], dtype=np.int32)
    n_times = len(all_ym)
    print(f"  {n_times} timesteps", flush=True)

    ts_data = []
    for ti, ((yr, mo), sub) in enumerate(sorted_groups):
        if len(sub) < 10:
            continue
        ts_data.append((
            ti,
            sub["value"].values.astype(np.float32),
            sub["lat"].values.astype(np.float32),
            sub["lon"].values.astype(np.float32),
        ))

    grid_pcr = np.full((n_times, ny, nx), np.nan, dtype=np.float32)
    pcr_results = _batched_grid(ts_data, GRID_SPEC, mask, qpts, x_grd, ny, nx, desc="Grid PCR")
    for ti, g in pcr_results.items():
        grid_pcr[ti] = g
    del recon, ts_data, pcr_results

    timings["grid_pcr"] = _t.time() - t0
    print(f"  PCR gridded in {timings['grid_pcr']:.1f}s", flush=True)

    # ── 2. Grid ESM for full cal period ──────────────────────
    print(f"\n[2/3] Gridding ESM (full cal period, optimized) ...", flush=True)
    t0 = _t.time()

    tr = load_trace_var(TRACE[var], var)
    tr = select_na_window(tr)
    yr_tr, mo_tr = trace_time_to_year_month(tr["time"].values)
    cal_mask_tr = (yr_tr >= YEAR_CAL_MIN) & (yr_tr <= YEAR_CAL_MAX)
    tr_cal = tr.isel(time=cal_mask_tr)

    esm_lat = tr_cal.lat.values
    esm_lon = tr_cal.lon.values
    glon_mesh, glat_mesh = np.meshgrid(glon, glat)
    pts_esm = np.column_stack([glat_mesh.ravel(), glon_mesh.ravel()])

    esm_lat_f64 = esm_lat.astype(np.float64)
    esm_lon_f64 = esm_lon.astype(np.float64)
    pts_esm_f64 = pts_esm.astype(np.float64)

    dummy_slice = np.zeros((len(esm_lat), len(esm_lon)), dtype=np.float64)
    interp_fn = RegularGridInterpolator(
        (esm_lat_f64, esm_lon_f64), dummy_slice, method="nearest",
        bounds_error=False, fill_value=np.nan)

    raw_idx = interp_fn._find_indices(pts_esm_f64.T)
    idx_lat = np.clip(raw_idx[0][0].astype(int), 0, len(esm_lat) - 1)
    idx_lon = np.clip(raw_idx[1][0].astype(int), 0, len(esm_lon) - 1)

    n_esm = tr_cal.sizes["time"]
    grid_esm = np.full((n_esm, ny, nx), np.nan, dtype=np.float32)

    tr_cal_loaded = tr_cal.values
    for ti in tqdm(range(n_esm), desc="Regrid ESM"):
        flat = tr_cal_loaded[ti][idx_lat, idx_lon].astype(np.float32)
        g = flat.reshape(ny, nx)
        if mask is not None:
            g = np.where(mask, g, np.nan)
        grid_esm[ti] = g

    timings["grid_esm"] = _t.time() - t0
    print(f"  ESM gridded in {timings['grid_esm']:.1f}s", flush=True)
    del tr_cal, tr_cal_loaded

    # ── 2b. Save PCR raw grid (before VC) ──────────────────
    pcr_raw_nc = OUT_DIR / f"grid_pcr_raw_{var}.nc"
    coords_save = {
        "time": np.arange(n_times),
        "year": ("time", all_ym[:, 0].astype(np.int32)),
        "month": ("time", all_ym[:, 1].astype(np.int32)),
        "lat": glat, "lon": glon,
    }
    enc_raw = {var: {"dtype": "float32", "zlib": True, "complevel": 4}}
    ds_raw = xr.Dataset({var: (("time", "lat", "lon"), grid_pcr)}, coords=coords_save)
    ds_raw.to_netcdf(pcr_raw_nc, encoding=enc_raw)
    raw_mb = pcr_raw_nc.stat().st_size / 1e6
    print(f"  Saved PCR raw: {pcr_raw_nc} ({raw_mb:.0f} MB)", flush=True)
    del ds_raw

    # ── 3. Variance correction (chunked for memory safety) ─
    print(f"\n[3/3] Applying variance correction (window={VAR_CORR_WINDOW} months) ...", flush=True)
    t0 = _t.time()

    CHUNK_ROWS = 75
    grid_pcr_vc = np.empty_like(grid_pcr)
    n_chunks = (ny + CHUNK_ROWS - 1) // CHUNK_ROWS
    for ci, r0 in enumerate(range(0, ny, CHUNK_ROWS)):
        r1 = min(r0 + CHUNK_ROWS, ny)
        t_chunk = _t.time()
        grid_pcr_vc[:, r0:r1, :] = variance_correction(
            grid_pcr[:, r0:r1, :], grid_esm[:, r0:r1, :],
            window=VAR_CORR_WINDOW)
        dt = _t.time() - t_chunk
        print(f"  chunk {ci+1}/{n_chunks} (rows {r0}-{r1}): {dt:.1f}s", flush=True)

    timings["varcorr"] = _t.time() - t0
    print(f"  Variance correction done in {timings['varcorr']:.1f}s", flush=True)

    # ── 4. Save products ─────────────────────────────────────
    print(f"\n[4/4] Saving products ...", flush=True)
    t0_save = _t.time()
    coords = {
        "time": np.arange(n_times),
        "year": ("time", all_ym[:, 0].astype(np.int32)),
        "month": ("time", all_ym[:, 1].astype(np.int32)),
        "lat": glat, "lon": glon,
    }
    enc = {var: {"dtype": "float32", "zlib": True, "complevel": 4}}

    pcr_vc_nc = OUT_DIR / f"grid_pcr_cal_{var}.nc"
    ds_pcr = xr.Dataset({var: (("time", "lat", "lon"), grid_pcr_vc)}, coords=coords)
    ds_pcr.to_netcdf(pcr_vc_nc, encoding=enc)
    pcr_mb = pcr_vc_nc.stat().st_size / 1e6
    print(f"  Saved PCR (vc): {pcr_vc_nc} ({pcr_mb:.0f} MB)", flush=True)

    esm_nc = OUT_DIR / f"grid_esm_cal_{var}.nc"
    ds_esm = xr.Dataset({var: (("time", "lat", "lon"), grid_esm)}, coords=coords)
    ds_esm.to_netcdf(esm_nc, encoding=enc)
    esm_mb = esm_nc.stat().st_size / 1e6
    print(f"  Saved ESM: {esm_nc} ({esm_mb:.0f} MB)", flush=True)

    # Extract test years and save for F5/F6
    test_mask = np.array([yr in test_years for yr in all_ym[:, 0]])
    grid_pcr_vc_test = grid_pcr_vc[test_mask]
    grid_esm_test = grid_esm[test_mask]
    test_ym = all_ym[test_mask]
    n_test = test_mask.sum()

    test_coords = {
        "time": np.arange(n_test),
        "year": ("time", test_ym[:, 0].astype(np.int32)),
        "month": ("time", test_ym[:, 1].astype(np.int32)),
        "lat": glat, "lon": glon,
    }

    pcr_test_nc = OUT_DIR / f"grid_pcr_vc_test_{var}.nc"
    ds_t = xr.Dataset({var: (("time", "lat", "lon"), grid_pcr_vc_test)}, coords=test_coords)
    ds_t.to_netcdf(pcr_test_nc, encoding=enc)
    print(f"  Saved PCR vc test: {pcr_test_nc} ({n_test} timesteps)", flush=True)

    esm_test_nc = OUT_DIR / f"grid_esm_vc_test_{var}.nc"
    ds_et = xr.Dataset({var: (("time", "lat", "lon"), grid_esm_test)}, coords=test_coords)
    ds_et.to_netcdf(esm_test_nc, encoding=enc)
    print(f"  Saved ESM test: {esm_test_nc} ({n_test} timesteps)", flush=True)

    timings["save"] = _t.time() - t0_save
    print(f"  Save done in {timings['save']:.1f}s", flush=True)

    timings["total"] = _t.time() - t_total

    # ── Timing summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE: {var} variance correction")
    print(f"{'─'*60}")
    for k, v in timings.items():
        print(f"  {k:20s}: {v:8.1f}s")
    print(f"{'─'*60}")
    print(f"  {'TOTAL':20s}: {timings['total']:8.1f}s")
    print(f"{'='*60}", flush=True)

    with open(OUT_DIR / f"timing_varcorr_{var}.json", "w") as f:
        json.dump(timings, f, indent=2)


if __name__ == "__main__":
    main()
