"""Benchmark: Python Sibson gridding speed at production scale.

Tests:
1. Weight matrix computation (sibson_weight_matrix)
2. Batched sparse matmul (W @ V)
3. Full gridding pipeline

Pair with benchmark_gridding.m for MATLAB comparison.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from caz.gridding import GridSpec, _albers_forward
from caz.natneighbor import sibson_weight_matrix

SPEC = GridSpec(lat_min=15.0, lat_max=75.0, lon_min=-170.0, lon_max=-50.0, res_deg=0.20)


def setup_grid():
    glat, glon = SPEC.lat_lon()
    ny, nx = len(glat), len(glon)
    longrid, latgrid = np.meshgrid(glon, glat)
    x_grd, y_grd = _albers_forward(longrid.ravel(), latgrid.ravel())
    qpts = np.column_stack([x_grd, y_grd])
    return qpts, ny, nx


def make_stations(n_st, seed=42):
    np.random.seed(seed)
    lat = np.random.uniform(20, 70, n_st).astype(np.float32)
    lon = np.random.uniform(-165, -55, n_st).astype(np.float32)
    x, y = _albers_forward(lon, lat)
    return np.column_stack([x, y])


def main():
    print("=== Gridding Benchmark (Python, Numba-parallel) ===\n")
    qpts, ny, nx = setup_grid()
    M = len(qpts)
    print(f"Grid: {ny}x{nx} = {M} query points\n")

    # Warmup
    small = make_stations(20)
    _ = sibson_weight_matrix(small, qpts[:100])

    results = {}
    for label, n_st in [("Small (800)", 800), ("Production (7000)", 7000)]:
        print(f"--- {label} stations ---")
        st = make_stations(n_st)

        t0 = time.perf_counter()
        W = sibson_weight_matrix(st, qpts)
        t_wm = time.perf_counter() - t0
        print(f"  Weight matrix: {t_wm:.3f}s (nnz={W.nnz})")

        V = np.random.randn(n_st, 125)
        t0 = time.perf_counter()
        G = W @ V
        t_mm = time.perf_counter() - t0
        print(f"  Batched matmul (125 cols): {t_mm:.3f}s")

        vals = np.random.randn(n_st)
        t0 = time.perf_counter()
        for _ in range(125):
            r = W @ vals
        t_mv = time.perf_counter() - t0
        print(f"  125x single matmul: {t_mv:.3f}s")

        results[label] = {"weight_matrix": t_wm, "batch_mm": t_mm, "single_mv": t_mv}
        print()

    # Summary
    print("=== Summary ===")
    for label, r in results.items():
        print(f"  {label}:")
        for k, v in r.items():
            print(f"    {k}: {v:.3f}s")

    # Production timing from run_varcorr_cal.py
    import json
    for var in ["tas", "pr"]:
        p = Path(r"D:\Dataset\DPastCliM-NA\interim\grid_cal") / f"timing_varcorr_{var}.json"
        if p.exists():
            with open(p) as f:
                t = json.load(f)
            print(f"\n  Production {var} (run_varcorr_cal.py):")
            for k, v in t.items():
                print(f"    {k}: {v:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
