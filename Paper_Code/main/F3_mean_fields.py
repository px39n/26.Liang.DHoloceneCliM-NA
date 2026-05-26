"""F3: Mean fields, PI range, and anomaly maps from predict windows.

Matches Guaita's Figure 4 layout: 4 rows × 4 cols = 16 panels (a–p).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))
from _common import sync_output, DATA_DIR, RESULTS_MAIN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import NearestNDInterpolator

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)

GCM = "trace21k"
PREDICT_DIR = DATA_DIR / "output" / GCM / "predict"

WINDOWS = ["lgm", "midhol", "recent"]
WINDOW_KA = ["20–19 ka", "5–4 ka", "1.0–0 ka"]

GRID_RES = 0.20
LAT_MIN, LAT_MAX = 15.0, 75.0
LON_MIN, LON_MAX = -170.0, -50.0
MASK_PATH = DATA_DIR / "static" / "landmask_NA_020.nc"

_TOL_YLORB = np.array([
    [255, 255, 229], [255, 247, 188], [254, 227, 145], [254, 196, 79],
    [251, 154, 41], [236, 112, 20], [204, 76, 2], [153, 52, 4], [102, 37, 6],
]) / 255.0
_TOL_IRIDESCENT = np.array([
    [254, 251, 233], [234, 240, 181], [194, 227, 210], [155, 210, 225],
    [123, 188, 231], [147, 152, 210], [154, 112, 158], [104, 73, 87],
]) / 255.0
_TOL_BURD = np.array([
    [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
    [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77],
    [178, 24, 43],
]) / 255.0
_TOL_SMOOTHRAINBOW = np.array([
    [227, 230, 249], [221, 216, 239], [209, 193, 225], [195, 168, 209],
    [181, 143, 194], [167, 120, 180], [155, 98, 167], [140, 78, 153],
    [111, 76, 155], [96, 89, 169], [85, 104, 184], [78, 121, 197],
    [77, 138, 198], [78, 150, 188], [84, 158, 179], [89, 165, 169],
    [96, 171, 158], [105, 177, 144], [119, 183, 125], [140, 188, 104],
    [166, 190, 84], [190, 188, 72], [209, 181, 65], [221, 170, 60],
    [228, 156, 57], [231, 140, 53], [230, 121, 50], [228, 99, 45],
    [223, 72, 40], [218, 34, 34], [184, 34, 30], [149, 33, 27],
    [114, 30, 23], [82, 26, 19],
]) / 255.0


def _cmap(rgb, name, n=256):
    return LinearSegmentedColormap.from_list(name, [tuple(c) for c in rgb], N=n)


CMAP_TAS_ABS = _cmap(_TOL_YLORB, "tol_ylorb", 12)
CMAP_PR_ABS = _cmap(_TOL_IRIDESCENT, "tol_iridescent", 14)
CMAP_ANOM = _cmap(_TOL_BURD, "tol_burd", 10)
CMAP_PI = _cmap(_TOL_SMOOTHRAINBOW, "tol_smoothrainbow", 10)

PROJ = ccrs.EckertIV(central_longitude=-110)


def _load_landmask(glat, glon):
    import xarray as xr
    if not MASK_PATH.exists():
        return None
    mds = xr.open_dataset(MASK_PATH)
    mask = mds["mask"].astype(float).interp(lat=glat, lon=glon, method="nearest").values > 0.5
    mds.close()
    return mask


def _grid(lats, lons, vals, glat, glon, mask):
    finite = np.isfinite(vals)
    lats, lons, vals = lats[finite], lons[finite], vals[finite]
    if len(vals) == 0:
        return np.full((len(glat), len(glon)), np.nan, dtype=np.float32)
    glon_m, glat_m = np.meshgrid(glon, glat)
    pts = np.column_stack([glat_m.ravel(), glon_m.ravel()])
    g = NearestNDInterpolator(np.column_stack([lats, lons]), vals)(pts)
    g = g.reshape(len(glat), len(glon)).astype(np.float32)
    if mask is not None:
        g = np.where(mask, g, np.nan)
    return g


def _load_stats(var, wname):
    df = pd.read_csv(PREDICT_DIR / f"recon_station_{var}_{wname}.csv",
                     usecols=["station_id", "lat", "lon", "value", "pi_lo", "pi_hi"])
    df = df.dropna(subset=["value"])
    stats = df.groupby("station_id").agg(
        lat=("lat", "first"), lon=("lon", "first"),
        value=("value", "mean"),
        pi_width=("pi_lo", lambda x: (df.loc[x.index, "pi_hi"] - df.loc[x.index, "pi_lo"]).mean()),
    )
    return stats


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


def _panel(ax, glon, glat, data, cmap, vmin, vmax, title, n_discrete):
    _boundary(ax)
    ax.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="0.2")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, color="black")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    bounds = np.linspace(vmin, vmax, n_discrete + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.pcolormesh(glon, glat, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, shading="auto", zorder=1)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

    m, s = np.nanmean(data), np.nanstd(data)
    ax.text(0.03, 0.07, f"{m:.2f} ± {s:.2f}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold", color="k")
    return im


def main():
    print("F3: Mean fields + PI + anomalies (Guaita Fig.4)", flush=True)

    glat = np.arange(LAT_MIN, LAT_MAX + GRID_RES / 2, GRID_RES)
    glon = np.arange(LON_MIN, LON_MAX + GRID_RES / 2, GRID_RES)
    mask = _load_landmask(glat, glon)

    fig = plt.figure(figsize=(16, 15))
    outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.38)

    cfgs = {
        "tas": dict(
            unit="°C",
            abs_cmap=CMAP_TAS_ABS, abs_lim=(-25, 35), abs_n=12, abs_step=10,
            anom_cmap=CMAP_ANOM, anom_lim=(-2.5, 2.5), anom_n=10,
            pi_cmap=CMAP_PI, pi_lim=(0, 15), pi_n=10, pi_step=3,
            row0=0, row1=1, lbl=0),
        "pr": dict(
            unit="mm day$^{-1}$",
            abs_cmap=CMAP_PR_ABS, abs_lim=(0, 7), abs_n=14, abs_step=1,
            anom_cmap=CMAP_ANOM, anom_lim=(-1.25, 1.25), anom_n=10,
            pi_cmap=CMAP_PI, pi_lim=(0, 15), pi_n=10, pi_step=3,
            row0=2, row1=3, lbl=8),
    }

    axes = np.empty((4, 4), dtype=object)
    for r in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[r], wspace=0.12)
        for c in range(4):
            axes[r, c] = fig.add_subplot(inner[c], projection=PROJ)

    ims = {}

    for var, C in cfgs.items():
        print(f"\n  {var}", flush=True)
        r0, r1, L = C["row0"], C["row1"], C["lbl"]
        gm, gp = {}, {}
        for w in WINDOWS:
            st = _load_stats(var, w)
            la, lo = st["lat"].values, st["lon"].values
            gm[w] = _grid(la, lo, st["value"].values, glat, glon, mask)
            gp[w] = _grid(la, lo, st["pi_width"].values, glat, glon, mask)

        fm = np.nanmean(np.stack(list(gm.values())), axis=0)
        fp = np.nanmean(np.stack(list(gp.values())), axis=0)

        # Col 0: Mean field
        im = _panel(axes[r0, 0], glon, glat, fm,
                     C["abs_cmap"], *C["abs_lim"],
                     f"({chr(97+L)}) Mean {var}\n3-wind. avg.", C["abs_n"])
        ims[(r0, "L")] = (im, C["abs_lim"], C["abs_step"], C["unit"])

        im = _panel(axes[r1, 0], glon, glat, fp,
                     C["pi_cmap"], *C["pi_lim"],
                     f"({chr(97+L+4)}) 95% PI {var}\n3-wind. avg.", C["pi_n"])
        ims[(r1, "L")] = (im, C["pi_lim"], C["pi_step"], C["unit"])

        # Cols 1-3: anomalies
        for ci, (w, wk) in enumerate(zip(WINDOWS, WINDOW_KA)):
            anom = gm[w] - fm
            pi_a = gp[w] - fp
            im = _panel(axes[r0, 1+ci], glon, glat, anom,
                         C["anom_cmap"], *C["anom_lim"],
                         f"({chr(97+L+1+ci)}) Anomaly\n{wk}", C["anom_n"])
            if ci == 0:
                ims[(r0, "R")] = (im, C["anom_lim"], None, C["unit"])

            im = _panel(axes[r1, 1+ci], glon, glat, pi_a,
                         C["anom_cmap"], *C["anom_lim"],
                         f"({chr(97+L+5+ci)}) PI anomaly\n{wk}", C["anom_n"])
            if ci == 0:
                ims[(r1, "R")] = (im, C["anom_lim"], None, C["unit"])

    # Colorbars
    for r in range(4):
        if (r, "L") in ims:
            im, lim, step, unit = ims[(r, "L")]
            cb = fig.colorbar(im, ax=axes[r, 0], orientation="horizontal",
                              pad=0.08, shrink=0.75, aspect=15)
            cb.set_label(unit, fontsize=11, fontweight="bold")
            cb.ax.tick_params(labelsize=10)
            for t in cb.ax.get_xticklabels():
                t.set_fontweight("bold")
            if step:
                cb.set_ticks(np.arange(np.ceil(lim[0]/step)*step, lim[1]+step/2, step))

        if (r, "R") in ims:
            im, lim, _, unit = ims[(r, "R")]
            cb = fig.colorbar(im, ax=[axes[r, c] for c in range(1, 4)],
                              orientation="horizontal", pad=0.08,
                              shrink=0.65, aspect=28)
            cb.set_label(unit, fontsize=11, fontweight="bold")
            cb.ax.tick_params(labelsize=10)
            for t in cb.ax.get_xticklabels():
                t.set_fontweight("bold")

    fig.suptitle("TraCE-21k II — Downscaled Mean Fields, 95% PI, and Anomalies",
                 fontsize=15, fontweight="bold", y=1.01)

    out = RESULTS_MAIN / "F3_mean_fields.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}", flush=True)
    sync_output(out)


if __name__ == "__main__":
    main()
