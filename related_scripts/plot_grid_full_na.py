"""Plot full-NA gridded tas / pr at key Holocene epochs + regional time series.

Usage:
    python related_scripts/plot_grid_full_na.py --var tas
    python related_scripts/plot_grid_full_na.py --var pr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr


GRID_DIR = Path(r"D:\Dataset\DPastCliM-NA\interim\grid_full_na")
OUT_DIR = Path(r"D:\OneDrive\Code\25.Liang.DPastCliM-NA\figures")


EPOCHS = [
    (-20050, "LGM"),
    (-14000, "Late glacial"),
    (-9000, "Early Holocene"),
    (-6000, "Mid-Holocene"),
    (-3000, "Late Holocene"),
    (0,     "0 CE"),
    (1000,  "MCA"),
    (1700,  "LIA"),
    (1900,  "1900 CE"),
    (1990,  "1990 CE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["tas", "pr"], default="tas")
    args = ap.parse_args()
    var = args.var
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_nc = GRID_DIR / f"grid_{var}_NA_decadal.nc"

    ds = xr.open_dataset(grid_nc)
    print(ds)
    arr = ds[var]
    unit = "°C" if var == "tas" else "mm/day"

    if var == "tas":
        vmin, vmax, cmap = -30, 28, "RdBu_r"
    else:
        vmin, vmax, cmap = 0.0, 8.0, "YlGnBu"

    # spatial maps
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    for ax, (y, lab) in zip(axes.flat, EPOCHS):
        idx = int(np.argmin(np.abs(ds["time"].values - y)))
        z = arr.isel(time=idx).values
        im = ax.pcolormesh(ds["lon"], ds["lat"], z, vmin=vmin, vmax=vmax,
                           cmap=cmap, shading="auto")
        ax.set_title(f"{lab}\n(dec={int(ds['time'].values[idx])} CE, "
                     f"mean={np.nanmean(z):.2f}{unit})", fontsize=9)
        ax.set_xlim(-170, -50); ax.set_ylim(15, 75)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cb.set_label(f"{var} ({unit})")
    fig.suptitle(f"DPastCliM-NA full-NA {var}: TraCE-21k II → PCR + delta-change → IDW (0.20°)",
                 fontsize=13)
    out_maps = OUT_DIR / f"NA_{var}_epochs.png"
    fig.savefig(out_maps, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_maps}")

    # time series
    series = arr.mean(dim=("lat", "lon"), skipna=True).values
    yr_bp = 1950 - ds["time"].values
    fig, ax = plt.subplots(figsize=(12, 4))
    color = "firebrick" if var == "tas" else "steelblue"
    ax.plot(yr_bp, series, color=color, lw=0.8)
    ax.set_xlabel("years BP")
    ax.set_ylabel(f"NA-mean {var} ({unit})")
    ax.invert_xaxis()
    ax.set_xscale("symlog", linthresh=200)
    ax.set_title(f"NA regional-mean {var}, 22 ka BP - present (decadal mean)")
    ax.grid(alpha=0.3)
    for y, lab in EPOCHS:
        bp = 1950 - y
        ax.axvline(bp, color="grey", lw=0.4, alpha=0.5)
        ax.annotate(lab.split()[0], (bp, ax.get_ylim()[1]), fontsize=7,
                    rotation=90, ha="center", va="top", alpha=0.6)
    out_ts = OUT_DIR / f"NA_{var}_timeseries.png"
    fig.savefig(out_ts, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_ts}")


if __name__ == "__main__":
    main()
