"""Full North America station -> 0.20 deg grid via IDW, decadal-mean output.

NA window: lat 15..75 N, lon -170..-50 W -> 300 x 600 = 180,000 cells.
Time: decadal mean over 22 ka -> ~2,205 steps.

Usage:
    python related_scripts/run_grid_full_na.py --var tas
    python related_scripts/run_grid_full_na.py --var pr
"""
from __future__ import annotations

import argparse
import time as _t
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caz.gridding import GridSpec, idw_gridding


REC_DIR  = Path(r"D:\Dataset\DPastCliM-NA\interim\trace21k\station_cal")
META_TAS = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_tas_meta.parquet")
META_PR  = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_pr_meta.parquet")
OUT_DIR  = Path(r"D:\Dataset\DPastCliM-NA\output\trace21k\full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["tas", "pr"], default="tas")
    ap.add_argument("--res", type=float, default=0.20)
    args = ap.parse_args()
    var = args.var

    t0 = _t.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_nc = REC_DIR / f"recon_station_{var}.nc"
    meta_pq = META_TAS if var == "tas" else META_PR
    out_nc = OUT_DIR / f"grid_{var}_NA_decadal.nc"

    print(f"loading recon {rec_nc}")
    ds = xr.open_dataset(rec_nc, chunks={"time": 12 * 100, "station": -1})
    print("  recon dims:", dict(ds.sizes))

    meta = pd.read_parquet(meta_pq).rename(columns={"ID": "station"})
    print(f"  meta stations: {len(meta):,}")

    # NA window (with small pad for boundary IDW)
    pad = 3.0
    win_lat = (15.0 - pad, 75.0 + pad)
    win_lon = (-170.0 - pad, -50.0 + pad)
    in_win = meta["lat"].between(*win_lat) & meta["lon"].between(*win_lon)
    meta_w = meta[in_win].set_index("station")
    print(f"  stations near NA window: {len(meta_w):,}")

    sids = pd.Index(ds["station"].values.astype(str), name="station")
    sids_keep = sids.intersection(meta_w.index)
    print(f"  recon stations in NA window: {len(sids_keep):,}")
    ds = ds.sel(station=sids_keep.values)
    meta_w = meta_w.reindex(sids_keep)
    pts_lat = meta_w["lat"].values.astype(np.float32)
    pts_lon = meta_w["lon"].values.astype(np.float32)

    # decadal mean
    yr = ds["year"].values.astype(np.int32)
    dec = (yr // 10 * 10).astype(np.int32)
    print(f"  reducing monthly -> decadal on lazy array...")
    ds = ds.assign_coords(decade=("time", dec))
    var_name = f"{var}_recon"
    decadal = ds[var_name].groupby("decade").mean(dim="time", skipna=True).compute()
    dec_yrs = decadal["decade"].values.astype(np.int32)
    dec_vals = decadal.values.astype(np.float32)
    print(f"  decadal aggregated: {dec_vals.shape}")

    spec = GridSpec(lat_min=15.0, lat_max=75.0, lon_min=-170.0, lon_max=-50.0, res_deg=args.res)
    print(f"  grid: {spec}")
    print("  IDW gridding...")
    t1 = _t.time()
    grid, lat, lon = idw_gridding(dec_vals, pts_lat, pts_lon, spec, k=8, power=2.0, radius_km=1500.0)
    print(f"  gridded {grid.shape} in {_t.time() - t1:.1f}s  (~{grid.nbytes/1e9:.2f} GB float32)")

    # encoding: tas -> 0.01 degC; pr -> 0.01 mm/day
    enc = {var: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768,
                 "zlib": True, "complevel": 5}}

    out = xr.Dataset(
        {var: (("time", "lat", "lon"), grid)},
        coords={"time": dec_yrs, "lat": lat, "lon": lon},
        attrs={
            "title": f"DPastCliM-NA {var} reconstruction: TraCE-21k II -> PCR (per-month, n_pc=5) -> IDW",
            "source": str(rec_nc),
            "method": "PCR (per-month) with delta-change ESM trend correction, IDW k=8 p=2 r=1500km",
            "time_units": "decadal mean, year (CE) of decade start",
            "variable": var,
        },
    )
    out.to_netcdf(out_nc, mode="w", encoding=enc)
    print(f"  wrote {out_nc}  ({out_nc.stat().st_size / 1e6:.1f} MB)")

    print("\nsanity (regional-mean over NA):")
    for q_yr in (-21000, -11000, -8000, -6000, -3000, 0, 1900, 1990):
        idx = int(np.argmin(np.abs(dec_yrs - q_yr)))
        m_yr = dec_yrs[idx]
        mn = np.nanmean(grid[idx])
        unit = "degC" if var == "tas" else "mm/day"
        print(f"  decade {m_yr:>6d}: NA-mean = {mn:8.3f} {unit}")

    print(f"\ndone in {_t.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
