"""Step E verification: station-to-grid interpolation (January, tas).

Matches Guaita's approach: Albers projection (EPSG:5070) + Sibson
natural-neighbor interpolation with nearest extrapolation.
"""
import struct
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from caz.natneighbor import sibson_interp, sibson_weight_matrix


def albers_forward(lon_deg, lat_deg):
    """Albers Equal-Area Conic, EPSG:5070 (NAD83 / Conus Albers).
    GRS80 ellipsoid, std parallels 29.5/45.5 N, origin 23N 96W."""
    a = 6378137.0
    f = 1 / 298.257222101
    e2 = 2 * f - f ** 2
    e = np.sqrt(e2)

    phi1 = np.radians(29.5)
    phi2 = np.radians(45.5)
    phi0 = np.radians(23.0)
    lam0 = np.radians(-96.0)

    def _m(phi):
        return np.cos(phi) / np.sqrt(1 - e2 * np.sin(phi) ** 2)

    def _q(phi):
        sp = np.sin(phi)
        return (1 - e2) * (sp / (1 - e2 * sp ** 2) -
                           np.log((1 - e * sp) / (1 + e * sp)) / (2 * e))

    m1, m2 = _m(phi1), _m(phi2)
    q0, q1, q2 = _q(phi0), _q(phi1), _q(phi2)
    n = (m1 ** 2 - m2 ** 2) / (q2 - q1)
    C = m1 ** 2 + n * q1
    rho0 = a * np.sqrt(C - n * q0) / n

    phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lam = np.radians(np.asarray(lon_deg, dtype=np.float64))
    q = _q(phi)
    rho = a * np.sqrt(C - n * q) / n
    theta = n * (lam - lam0)

    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y

OUT_DIR = Path(r"D:\Dataset\DPastCliM-NA\verification\step_E\python")
ML_DIR = Path(r"D:\Dataset\DPastCliM-NA\verification\step_E\matlab")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load Step D predictions ----
d4 = np.load(r"D:\Dataset\DPastCliM-NA\verification\step_D\python\yhat_full.npz",
             allow_pickle=True)
yhat = d4["yhat"]
station_ids = list(d4["station_ids"])
T, n_st = yhat.shape

# ---- Load station metadata ----
meta = pd.read_parquet(
    r"D:\Dataset\DPastCliM-NA\verification\step_A\python\ghcn_tas_meta.parquet")
meta_dict = {row["ID"]: (row["lat"], row["lon"]) for _, row in meta.iterrows()}

matched = []
for i, sid in enumerate(station_ids):
    if sid in meta_dict:
        matched.append((i, meta_dict[sid][0], meta_dict[sid][1]))
idx_matched = np.array([m[0] for m in matched])
st_lat = np.array([m[1] for m in matched])
st_lon = np.array([m[2] for m in matched])
yhat_matched = yhat[:, idx_matched]
print(f"Matched {len(matched)} / {n_st} stations with metadata")

# ---- Target grid (NE-US, 0.5 deg) ----
lat_min, lat_max = 35.0, 50.0
lon_min, lon_max = -90.0, -65.0
res = 0.5
grid_lat = np.arange(lat_min, lat_max + 1e-9, res)
grid_lon = np.arange(lon_min, lon_max + 1e-9, res)
glon, glat = np.meshgrid(grid_lon, grid_lat)
ny, nx = glat.shape

# ---- Filter stations in region (+5 deg buffer) ----
in_region = ((st_lat >= lat_min - 5) & (st_lat <= lat_max + 5) &
             (st_lon >= lon_min - 5) & (st_lon <= lon_max + 5))
print(f"Stations in buffered region: {in_region.sum()}")
st_lat_r = st_lat[in_region]
st_lon_r = st_lon[in_region]
yhat_r = yhat_matched[:, in_region]

# ---- Project to Albers (NAD83 / Conus Albers, EPSG:5070) ----
x_proj, y_proj = albers_forward(st_lon_r, st_lat_r)
xgrid_proj, ygrid_proj = albers_forward(glon.ravel(), glat.ravel())

# Verify projection matches MATLAB
ml_proj_file = ML_DIR / "proj_coords.bin"
if ml_proj_file.exists():
    with open(ml_proj_file, "rb") as f:
        n_pts = struct.unpack("i", f.read(4))[0]
        ml_x = np.frombuffer(f.read(n_pts * 8), dtype=np.float64)
        ml_y = np.frombuffer(f.read(n_pts * 8), dtype=np.float64)
        n_grid = struct.unpack("i", f.read(4))[0]
        ml_xg = np.frombuffer(f.read(n_grid * 8), dtype=np.float64)
        ml_yg = np.frombuffer(f.read(n_grid * 8), dtype=np.float64)
    print(f"\n--- Projection comparison ---")
    print(f"  Station proj max |diff_x|: {np.abs(x_proj - ml_x).max():.4f} m")
    print(f"  Station proj max |diff_y|: {np.abs(y_proj - ml_y).max():.4f} m")
    print(f"  Grid proj max |diff_x|: {np.abs(xgrid_proj - ml_xg).max():.4f} m")
    print(f"  Grid proj max |diff_y|: {np.abs(ygrid_proj - ml_yg).max():.4f} m")

# ---- Gridding: Sibson natural-neighbor in projected coords ----
import time as _time
qpts = np.column_stack([xgrid_proj, ygrid_proj])
gridded = np.full((T, ny, nx), np.nan, dtype=np.float32)

_t0 = _time.time()
for t in range(T):
    z = yhat_r[t, :]
    valid = ~np.isnan(z)
    if valid.sum() < 4:
        continue
    pts = np.column_stack([x_proj[valid], y_proj[valid]])
    vals = sibson_interp(pts, z[valid].astype(np.float64), qpts)
    nans = np.isnan(vals)
    if nans.any():
        near = NearestNDInterpolator(pts, z[valid])
        vals[nans] = near(qpts[nans])
    gridded[t] = vals.astype(np.float32).reshape(ny, nx)
print(f"Gridded {T} timesteps in {_time.time() - _t0:.1f}s")

print(f"\nGridded {T} timesteps to {ny}x{nx} grid")

np.savez(OUT_DIR / "gridded.npz", gridded=gridded,
         grid_lat=grid_lat, grid_lon=grid_lon)

# ---- Compare with MATLAB ----
ml_bin = ML_DIR / "gridded.bin"
if not ml_bin.exists():
    print(f"\nMATLAB output not found at {ml_bin}. Run verify_E_gridding.m first.")
else:
    print("\n--- Step E comparison (MATLAB natural vs Python Sibson natural) ---")
    with open(ml_bin, "rb") as f:
        T_ml, ny_ml, nx_ml = struct.unpack("iii", f.read(12))
        g_ml = np.frombuffer(f.read(), dtype=np.float32).reshape(T_ml, ny_ml, nx_ml)

    print(f"  MATLAB shape: ({T_ml}, {ny_ml}, {nx_ml})")
    print(f"  Python shape: {gridded.shape}")

    both_valid = ~np.isnan(gridded) & ~np.isnan(g_ml)
    diff = np.abs(gridded - g_ml)
    print(f"  Valid grid cells: {both_valid.sum()}")
    print(f"  Max |diff|: {diff[both_valid].max():.6f}")
    print(f"  Mean |diff|: {diff[both_valid].mean():.6f}")
    print(f"  Median |diff|: {np.median(diff[both_valid]):.6f}")
    p95 = np.percentile(diff[both_valid], 95)
    p99 = np.percentile(diff[both_valid], 99)
    print(f"  P95 |diff|: {p95:.6f}")
    print(f"  P99 |diff|: {p99:.6f}")

    within_01 = (diff[both_valid] < 0.1).mean() * 100
    within_05 = (diff[both_valid] < 0.5).mean() * 100
    print(f"  % within 0.1: {within_01:.1f}%")
    print(f"  % within 0.5: {within_05:.1f}%")

print("\nDone.")
