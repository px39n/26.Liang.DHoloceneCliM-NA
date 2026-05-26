"""F5: Mean bias + percentile difference maps for temperature.

Matches Guaita's Figure 6: 3 rows × 2 columns
  Row 1: Mean bias (PCR−obs, ESM−obs)
  Row 2: 10th percentile difference
  Row 3: 90th percentile difference

Uses gridded products from Step 4 (per-timestep gridding, then statistics).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))
from _common import sync_output, DATA_DIR, GRID_CAL, RESULTS_MAIN

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

VAR = "tas"
UNIT = "°C"
BIAS_LIMIT = 5.0
LAT_MIN, LAT_MAX = 15.0, 75.0
LON_MIN, LON_MAX = -170.0, -50.0
RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_MAIN / "F5_bias_map_tas.png"
GRID_DIR = GRID_CAL

_TOL_BURD = np.array([
    [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
    [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77],
    [178, 24, 43],
]) / 255.0
CMAP_BIAS = LinearSegmentedColormap.from_list("tol_burd", [tuple(c) for c in _TOL_BURD], N=10)
PROJ = ccrs.EckertIV(central_longitude=-110)


def _boundary(ax):
    corners = np.array([
        [LON_MIN, LAT_MIN], [LON_MAX, LAT_MIN],
        [LON_MAX, LAT_MAX], [LON_MIN, LAT_MAX]])
    pts = []
    for i in range(4):
        j = (i + 1) % 4
        lo = np.linspace(corners[i, 0], corners[j, 0], 80)
        la = np.linspace(corners[i, 1], corners[j, 1], 80)
        pts.extend(zip(lo, la))
    pts = np.array(pts)
    proj_pts = PROJ.transform_points(ccrs.PlateCarree(), pts[:, 0], pts[:, 1])[:, :2]
    ax.set_boundary(mpath.Path(proj_pts), transform=PROJ)
    x0, x1 = proj_pts[:, 0].min(), proj_pts[:, 0].max()
    y0, y1 = proj_pts[:, 1].min(), proj_pts[:, 1].max()
    dx, dy = (x1 - x0) * 0.02, (y1 - y0) * 0.02
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


def _load_grids():
    obs = xr.open_dataset(GRID_DIR / f"grid_obs_test_{VAR}.nc")[VAR].values
    pcr = xr.open_dataset(GRID_DIR / f"grid_pcr_test_{VAR}.nc")[VAR].values
    esm = xr.open_dataset(GRID_DIR / f"grid_esm_test_{VAR}.nc")[VAR].values
    lat = xr.open_dataset(GRID_DIR / f"grid_obs_test_{VAR}.nc")["lat"].values
    lon = xr.open_dataset(GRID_DIR / f"grid_obs_test_{VAR}.nc")["lon"].values
    return obs, pcr, esm, lat, lon


def main():
    print(f"F5: Loading gridded test-period data for {VAR} ...", flush=True)
    obs, pcr, esm, glat, glon = _load_grids()
    nt = obs.shape[0]
    print(f"  Shape: {obs.shape} ({nt} timesteps)", flush=True)

    mean_obs = np.nanmean(obs, axis=0)
    mean_pcr = np.nanmean(pcr, axis=0)
    mean_esm = np.nanmean(esm, axis=0)
    p10_obs = np.nanpercentile(obs, 10, axis=0)
    p10_pcr = np.nanpercentile(pcr, 10, axis=0)
    p10_esm = np.nanpercentile(esm, 10, axis=0)
    p90_obs = np.nanpercentile(obs, 90, axis=0)
    p90_pcr = np.nanpercentile(pcr, 90, axis=0)
    p90_esm = np.nanpercentile(esm, 90, axis=0)

    panels = [
        (mean_pcr - mean_obs, mean_esm - mean_obs, "Mean Bias"),
        (p10_pcr - p10_obs,   p10_esm - p10_obs,   "Δ P10"),
        (p90_pcr - p90_obs,   p90_esm - p90_obs,   "Δ P90"),
    ]

    for row_title, (left, right, _) in zip(
        ["Mean", "P10", "P90"], panels
    ):
        for side, data in [("PCR−Obs", left), ("ESM−Obs", right)]:
            sm = np.nanmean(data)
            ss = np.nanstd(data)
            print(f"  {row_title} {side}: {sm:+.3f} ± {ss:.3f} {UNIT}", flush=True)

    print("  Plotting 3×2 panels ...", flush=True)
    bounds = np.linspace(-BIAS_LIMIT, BIAS_LIMIT, 11)
    norm = BoundaryNorm(bounds, CMAP_BIAS.N)

    fig = plt.figure(figsize=(14, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.12)
    axes = np.empty((3, 2), dtype=object)
    for r in range(3):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=PROJ)

    for row_i, (left, right, title) in enumerate(panels):
        for col_i, (data, label) in enumerate(
            [(left, "PCR − Obs"), (right, "ESM − Obs")]
        ):
            ax = axes[row_i, col_i]
            _boundary(ax)
            ax.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="0.2")
            ax.add_feature(cfeature.BORDERS, linewidth=0.9, color="black")
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            im = ax.pcolormesh(
                glon, glat, data,
                transform=ccrs.PlateCarree(),
                cmap=CMAP_BIAS, norm=norm, shading="auto", zorder=1,
            )

            spatial_mean = np.nanmean(data)
            spatial_std = np.nanstd(data)
            ax.set_title(f"({chr(97 + row_i * 2 + col_i)}) {title}: {label}",
                         fontsize=13, fontweight="bold", pad=6)
            ax.text(0.03, 0.07, f"{spatial_mean:+.2f} ± {spatial_std:.2f}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=11, fontweight="bold", color="k")

            cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8, aspect=25)
            cb.set_label(UNIT, fontsize=11, fontweight="bold")
            cb.ax.tick_params(labelsize=10)
            for t in cb.ax.get_yticklabels():
                t.set_fontweight("bold")

    fig.suptitle(f"Temperature Validation — Test Period ({UNIT})",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT}", flush=True)
    sync_output(OUT)


if __name__ == "__main__":
    main()
