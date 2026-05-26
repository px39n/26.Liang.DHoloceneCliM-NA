"""F1: GHCN station coverage map + latitude KDE (our Fig 2 = Guaita Fig 2).

Input:  split_calibration.pkl (single source of truth)
Output: F1_station_coverage.png

Layout: 2×2 (top=tas, bottom=pr; left=map, right=KDE)
Colours: orange=calibration, green=test-only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from _common import GHCN_DIR, MODELS_DIR, RESULTS_MAIN, sync_output

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_MAIN / "F1_station_coverage.png"

COLOR_CAL  = np.array([230, 159, 0]) / 255
COLOR_TEST = np.array([0, 158, 115]) / 255

LIM_LAT = (5, 72)
LIM_LON = (-175, -50)


def _load_station_meta(var: str, split_info: dict) -> pd.DataFrame:
    """Load station metadata + cal/test_only flags from split file."""
    meta = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_meta.parquet")
    station_flags = split_info["station_flags"]

    meta["flag"] = meta["ID"].map(station_flags).fillna("removed")
    meta = meta[meta["flag"].isin(("cal", "test_only"))].copy()
    meta["flag_cal"] = meta["flag"] == "cal"
    meta["flag_test"] = meta["flag"] == "test_only"
    return meta


def _plot_panel(ax, meta, var_label, panel_label):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent([LIM_LON[0], LIM_LON[1], LIM_LAT[0], LIM_LAT[1]],
                  crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="k")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#888888")

    t = ccrs.PlateCarree()
    idx_cal = meta["flag_cal"].values
    idx_test = meta["flag_test"].values

    ax.plot(meta.loc[idx_cal, "lon"].values, meta.loc[idx_cal, "lat"].values,
            "o", ms=1, mfc=COLOR_CAL, mec=COLOR_CAL, transform=t)
    if idx_test.sum() > 0:
        ax.plot(meta.loc[idx_test, "lon"].values, meta.loc[idx_test, "lat"].values,
                "o", ms=1, mfc=COLOR_TEST, mec=COLOR_TEST, transform=t)

    n_cal = idx_cal.sum()
    n_test = idx_test.sum()
    ax.text(LIM_LON[0] + 0.03 * (LIM_LON[1] - LIM_LON[0]),
            LIM_LAT[0] + 0.04 * (LIM_LAT[1] - LIM_LAT[0]),
            f"cal: {n_cal}  test-only: {n_test}  total: {len(meta)}",
            fontsize=9, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="k", linewidth=0.5),
            transform=t)

    ax.text(0.02, 0.95, f"({panel_label})", fontsize=12, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes)
    ax.set_title(f"{var_label} GHCN-m stations", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)


def _plot_kde(ax, meta):
    idx_cal = meta["flag_cal"].values
    idx_test = meta["flag_test"].values
    lat_smooth = np.linspace(LIM_LAT[0], LIM_LAT[1], 200)
    bw = 2.0

    f_cal = gaussian_kde(
        meta.loc[idx_cal, "lat"],
        bw_method=bw / meta.loc[idx_cal, "lat"].std()
    )(lat_smooth) if idx_cal.sum() > 1 else np.zeros_like(lat_smooth)

    f_test = gaussian_kde(
        meta.loc[idx_test, "lat"],
        bw_method=bw / meta.loc[idx_test, "lat"].std()
    )(lat_smooth) if idx_test.sum() > 1 else np.zeros_like(lat_smooth)

    ax.plot(f_cal, lat_smooth, color=COLOR_CAL, linewidth=2, label="Calibration")
    ax.plot(f_test, lat_smooth, color=COLOR_TEST, linewidth=2, label="Test-only")

    ymax = max(f_cal.max(), f_test.max() if f_test.max() > 0 else 0)
    ax.set_xlim(0, ymax * 1.1 + 1e-6)
    ax.set_ylim(LIM_LAT)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title("KDE", fontsize=11)
    ax.set_xlabel("Density", fontsize=10)
    ax.tick_params(labelsize=9)


def main():
    import cartopy.crs as ccrs

    with open(MODELS_DIR / "split_calibration.pkl", "rb") as f:
        splits = pickle.load(f)

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], hspace=0.25, wspace=0.05)

    for i, var in enumerate(["tas", "pr"]):
        meta = _load_station_meta(var, splits[var])
        print(f"{var}: {len(meta)} stations "
              f"(cal={meta['flag_cal'].sum()}, test-only={meta['flag_test'].sum()})")

        ax_map = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        _plot_panel(ax_map, meta, var, "ab"[i])

        ax_kde = fig.add_subplot(gs[i, 1])
        _plot_kde(ax_kde, meta)

    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)
    sync_output(OUT)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
