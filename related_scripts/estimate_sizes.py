"""Estimate on-disk size for each Caz product.

Strategy:
- Pure analytic estimate for full datasets (no big files written).
- Tiny empirical write (<= ~10 MB) only for compression-ratio sanity check.

Run:
    conda run -n caz python related_scripts/estimate_sizes.py
"""
from pathlib import Path
import numpy as np
import xarray as xr
import tempfile

OUT = Path(tempfile.mkdtemp(prefix="caz_size_"))

def fmt(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024: return f"{n:7.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"

# ---------- shape constants ----------
GCM_LAT, GCM_LON = 48, 96         # T31 ~ TraCE-21k
HIST_MONTHS      = 165 * 12       # 1850-2014
HOLO_MONTHS      = 22000 * 12     # ~22 ka monthly (TraCE-21k)
N_STATION_T      = 7000
N_STATION_P      = 12000
NA_LAT, NA_LON   = 350, 600       # 0.20 deg, 15-85N x 170-50W

def analytic(name, n_elem, n_vars=1, dtype_bytes=4):
    sz = n_elem * n_vars * dtype_bytes
    print(f"  {name:55s} {fmt(sz)}")
    return sz

print("="*72)
print("ANALYTIC SIZES (float32, uncompressed)")
print("="*72)

print("\n[1] ESM historical (per GCM, tas+pr):")
analytic("native 48x96, 1980 mo, 2 vars",
         HIST_MONTHS*GCM_LAT*GCM_LON, 2)

print("\n[2] ESM Holocene transient (per GCM, tas+pr):")
analytic("monthly  264000 step x 48x96, 2 vars",
         HOLO_MONTHS*GCM_LAT*GCM_LON, 2)
analytic("decadal  26400  step x 48x96, 2 vars",
         HOLO_MONTHS//120*GCM_LAT*GCM_LON, 2)

print("\n[3] GHCN-m stations:")
analytic("tas 7000 sta  x 1980 mo", N_STATION_T*HIST_MONTHS, 1)
analytic("pr  12000 sta x 1980 mo", N_STATION_P*HIST_MONTHS, 1)

print("\n[4] Land mask 0.20 NA (350x600 bool):")
analytic("mask", NA_LAT*NA_LON, 1, 1)

print("\n[5] Per-GCM downscaled (NA 0.20, 6 vars: tas,pr + 4 PI):")
analytic("monthly  264000 step",
         HOLO_MONTHS*NA_LAT*NA_LON, 6)
analytic("decadal  26400 step",
         HOLO_MONTHS//120*NA_LAT*NA_LON, 6)
analytic("century  264 step",
         HOLO_MONTHS//1200*NA_LAT*NA_LON, 6)

print("\n[6] Ensemble combined (NA 0.20, 8 vars):")
analytic("monthly", HOLO_MONTHS*NA_LAT*NA_LON, 8)
analytic("decadal", HOLO_MONTHS//120*NA_LAT*NA_LON, 8)
analytic("century", HOLO_MONTHS//1200*NA_LAT*NA_LON, 8)

print("\n" + "="*72)
print("EMPIRICAL COMPRESSION RATIO (tiny write, 100 mo only)")
print("="*72)
def write_nc(name, dims_shape, var_names):
    coords = {d: np.arange(s) for d, s in dims_shape.items()}
    shape = tuple(dims_shape.values())
    dims  = tuple(dims_shape.keys())
    rng = np.random.default_rng(0)
    dv = {v: (dims, rng.standard_normal(shape).astype("float32")) for v in var_names}
    ds = xr.Dataset(dv, coords=coords)
    enc = {v: {"zlib": True, "complevel": 4} for v in var_names}
    p = OUT / f"{name}.nc"
    ds.to_netcdf(p, encoding=enc)
    return p.stat().st_size

T = 100
sz_raw = T*GCM_LAT*GCM_LON*4*2
sz_nc  = write_nc("gcm_100mo", {"time": T, "lat": GCM_LAT, "lon": GCM_LON}, ["tas","pr"])
ratio_gcm = sz_nc / sz_raw
print(f"  GCM-grid 100mo tas+pr  : raw {fmt(sz_raw)} -> nc {fmt(sz_nc)}  (x{ratio_gcm:.2f})")

sz_raw2 = T*NA_LAT*NA_LON*4*6
sz_nc2  = write_nc("ds_100mo", {"time": T, "lat": NA_LAT, "lon": NA_LON},
                   ["tas","pr","tas_lo","tas_hi","pr_lo","pr_hi"])
ratio_ds = sz_nc2 / sz_raw2
print(f"  NA0.20  100mo 6vars    : raw {fmt(sz_raw2)} -> nc {fmt(sz_nc2)}  (x{ratio_ds:.2f})")

print("\nNote: random data compresses poorly; real climate fields typically x0.3-0.6.")
print(f"\ntmp files: {OUT}")
