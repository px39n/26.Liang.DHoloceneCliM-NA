"""Apply station -> grid IDW to a small NE-US sub-region as a smoke test.

Reads the station-level reconstruction produced by `run_pcr_station.py`
(`recon_station_tas.nc`) and outputs a small gridded NetCDF:

  - NE-US window: 35-45 N x 80-65 W  (10 deg lat x 15 deg lon @ 0.20 deg
    = 51 lat x 76 lon = 3,876 cells)
  - Time: decadal means (~2,205 time steps from 22050 yearly steps after
    monthly aggregation) to keep file size manageable.

Output: D:\\Dataset\\DPastCliM-NA\\interim\\grid_test\\grid_tas_NE_US_decadal.nc
"""
from __future__ import annotations

import time as _t
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caz.gridding import GridSpec, idw_gridding


REC_NC = Path(r"D:\Dataset\DPastCliM-NA\interim\pcr_station\recon_station_tas.nc")
GHCN_META = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_tas_meta.parquet")
OUT_NC = Path(r"D:\Dataset\DPastCliM-NA\interim\grid_test\grid_tas_NE_US_decadal.nc")


def main():
    t0 = _t.time()
    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    print(f"loading recon {REC_NC}")
    ds = xr.open_dataset(REC_NC, chunks={"time": 12 * 100, "station": -1})
    print("  recon dims:", dict(ds.sizes))

    meta = pd.read_parquet(GHCN_META).rename(columns={"ID": "station"})
    print(f"  meta stations: {len(meta):,}")

    # ----- first restrict stations to those near the NE-US window (huge speedup) -----
    pad = 5.0
    win_lat = (35.0 - pad, 45.0 + pad)
    win_lon = (-80.0 - pad, -65.0 + pad)
    in_win = (
        meta["lat"].between(*win_lat)
        & meta["lon"].between(*win_lon)
    )
    meta_w = meta[in_win].set_index("station")
    print(f"  stations near NE-US window: {len(meta_w):,}")

    sids = pd.Index(ds["station"].values.astype(str), name="station")
    sids_keep = sids.intersection(meta_w.index)
    print(f"  recon stations in window: {len(sids_keep):,}")
    ds = ds.sel(station=sids_keep.values)
    meta_w = meta_w.reindex(sids_keep)
    pts_lat = meta_w["lat"].values.astype(np.float32)
    pts_lon = meta_w["lon"].values.astype(np.float32)

    # ----- decadal mean: aggregate inside xarray (lazy + chunked) -----
    yr = ds["year"].values.astype(np.int32)
    dec = (yr // 10 * 10).astype(np.int32)
    print(f"  reducing monthly -> decadal on the lazy array...")
    ds = ds.assign_coords(decade=("time", dec))
    decadal = ds["tas_recon"].groupby("decade").mean(dim="time", skipna=True).compute()
    dec_yrs = decadal["decade"].values.astype(np.int32)
    dec_vals = decadal.values.astype(np.float32)   # (n_dec, n_stations_in_window)
    print(f"  decadal aggregated: {dec_vals.shape}")

    # ----- gridding -----
    spec = GridSpec(lat_min=35.0, lat_max=45.0, lon_min=-80.0, lon_max=-65.0, res_deg=0.20)
    print(f"  grid: {spec}")

    print("  IDW gridding...")
    t1 = _t.time()
    grid, lat, lon = idw_gridding(dec_vals, pts_lat, pts_lon, spec, k=8, power=2.0, radius_km=600.0)
    print(f"  gridded {grid.shape} in {_t.time() - t1:.1f}s")

    # ----- save -----
    out = xr.Dataset(
        {"tas": (("time", "lat", "lon"), grid)},
        coords={"time": dec_yrs, "lat": lat, "lon": lon},
        attrs={
            "title": "DPastCliM-NA grid test (decadal): TraCE-21k II -> PCR -> IDW on NE-US",
            "source": str(REC_NC),
            "method": "PCR per month + IDW gridding (k=8, p=2)",
            "time_units": "decadal mean, year (CE) of decade start",
        },
    )
    enc = {"tas": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768,
                   "zlib": True, "complevel": 5}}
    out.to_netcdf(OUT_NC, mode="w", encoding=enc)
    print(f"  wrote {OUT_NC}  ({OUT_NC.stat().st_size / 1e6:.2f} MB)")

    # ----- quick sanity prints -----
    print("\nsanity checks:")
    for q_yr in (-21000, -11000, -6000, 0, 1900, 1990):
        idx = int(np.argmin(np.abs(dec_yrs - q_yr)))
        m_yr = dec_yrs[idx]
        mn = np.nanmean(grid[idx])
        print(f"  decade {m_yr:>6d}: regional-mean annual tas = {mn:6.2f} degC")

    print(f"\ndone in {_t.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
