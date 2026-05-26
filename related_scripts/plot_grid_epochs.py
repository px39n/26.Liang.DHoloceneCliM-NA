"""Plot NE-US gridded tas at key epochs and a time series of regional mean."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

GRID_NC = Path(r"D:\Dataset\DPastCliM-NA\interim\grid_test\grid_tas_NE_US_decadal.nc")
OUT_DIR = Path(r"D:\OneDrive\Code\25.Liang.DPastCliM-NA\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ds = xr.open_dataset(GRID_NC)
    print(ds)

    epochs = [
        (-20050, "LGM ~22 ka BP"),
        (-14000, "Late glacial ~16 ka BP"),
        (-9000, "Early Holocene ~11 ka BP"),
        (-6000, "Mid-Holocene ~8 ka BP"),
        (-3000, "Late Holocene ~5 ka BP"),
        (0,     "0 CE"),
        (1000,  "1000 CE (MCA)"),
        (1700,  "1700 CE (LIA)"),
        (1900,  "1900 CE"),
        (1990,  "1990 CE"),
    ]

    # spatial maps
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    vmin, vmax = -25, 22
    for ax, (y, lab) in zip(axes.flat, epochs):
        idx = int(np.argmin(np.abs(ds["time"].values - y)))
        z = ds["tas"].isel(time=idx).values
        im = ax.pcolormesh(ds["lon"], ds["lat"], z, vmin=vmin, vmax=vmax, cmap="RdBu_r", shading="auto")
        ax.set_title(f"{lab}\n(decade={int(ds['time'].values[idx])} CE, mean={np.nanmean(z):.1f}°C)")
        ax.set_xlim(-80, -65); ax.set_ylim(35, 45)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cb.set_label("annual tas (°C)")
    fig.suptitle("DPastCliM-NA grid test: NE-US (0.20°), TraCE-21k II → PCR + delta-change", fontsize=14)
    out_maps = OUT_DIR / "ne_us_tas_epochs.png"
    fig.savefig(out_maps, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_maps}")

    # time series of regional mean
    series = ds["tas"].mean(dim=("lat", "lon"), skipna=True).values
    yr_bp = 1950 - ds["time"].values
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(yr_bp, series, color="firebrick", lw=0.8)
    ax.set_xlabel("years BP")
    ax.set_ylabel("NE-US annual tas (°C)")
    ax.invert_xaxis()
    ax.set_xscale("symlog", linthresh=200)
    ax.set_title("NE-US regional-mean annual tas, 22 ka BP - present (decadal)")
    ax.grid(alpha=0.3)
    for y, lab in epochs:
        bp = 1950 - y
        ax.axvline(bp, color="grey", lw=0.4, alpha=0.5)
        ax.annotate(lab.split()[0], (bp, ax.get_ylim()[1] - 1), fontsize=7,
                    rotation=90, ha="center", va="top", alpha=0.6)
    out_ts = OUT_DIR / "ne_us_tas_timeseries.png"
    fig.savefig(out_ts, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_ts}")


if __name__ == "__main__":
    main()
