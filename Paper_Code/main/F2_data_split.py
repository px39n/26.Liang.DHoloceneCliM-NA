"""F2: Train/Val/Test split heatmap (our Fig 3 = Guaita Fig 3).

Input:  split_calibration.pkl (single source of truth) + GHCN obs
Output: F2_data_split.png

Layout: 2×1 (top=tas first 100 stations, bottom=pr first 100 stations)
Axes: x=Station index, y=Year
Colours: grey=nodata, orange=calibration, blue=model selection, green=testing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from _common import GHCN_DIR, MODELS_DIR, RESULTS_MAIN, sync_output

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT     = RESULTS_MAIN / "F2_data_split.png"
OUT_CSV = RESULTS_MAIN / "F2_data_split.csv"

N_STATIONS_SHOW = 100

COLOR_NODATA = "#d9d9d9"
COLOR_CAL    = np.array([230, 159, 0]) / 255
COLOR_MS     = np.array([0, 114, 178]) / 255
COLOR_TEST   = np.array([0, 158, 115]) / 255


def _build_split_matrix(var: str, split_info: dict):
    """Build station × year split matrix from split_calibration.pkl.

    Returns (n_sta, n_yr) int8, station_ids, years.
    Values: 0=nodata, 1=cal, 2=model_selection/val, 3=test
    """
    obs = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_obs.parquet")
    common_years = split_info["common_years"]
    station_flags = split_info["station_flags"]

    cal_years_set = set(split_info["cal_years"])
    val_years_set = set(split_info["val_years"])
    test_years_set = set(split_info["test_years"])

    # Only keep stations not removed
    kept_ids = sorted(sid for sid, flag in station_flags.items()
                     if flag in ("cal", "test_only"))
    sid2idx = {s: i for i, s in enumerate(kept_ids)}

    n_sta = len(kept_ids)
    n_yr = len(common_years)

    # Vectorized: build (station, year) → has_data boolean via pivot
    obs_filt = obs[obs["ID"].isin(sid2idx) & obs["year"].isin(common_years)]
    has_data = (
        obs_filt.groupby(["ID", "year"]).size().reset_index(name="cnt")
    )

    # Build year → split label mapping
    year_to_label = {}
    for y in cal_years_set:
        year_to_label[y] = 1
    for y in val_years_set:
        year_to_label[y] = 2
    for y in test_years_set:
        year_to_label[y] = 3

    # Vectorized assignment
    split_mat = np.zeros((n_sta, n_yr), dtype=np.int8)
    yr2idx_map = {int(y): i for i, y in enumerate(common_years)}

    si_arr = has_data["ID"].map(sid2idx).values
    yi_arr = has_data["year"].map(yr2idx_map).values
    flag_arr = has_data["ID"].map(station_flags).values
    yr_arr = has_data["year"].values

    label_arr = np.array([year_to_label.get(int(y), 0) for y in yr_arr], dtype=np.int8)

    # test_only stations: all cells become test (3)
    is_test_only = flag_arr == "test_only"
    label_arr[is_test_only] = 3

    valid = ~(np.isnan(si_arr) | np.isnan(yi_arr))
    si_valid = si_arr[valid].astype(int)
    yi_valid = yi_arr[valid].astype(int)
    label_valid = label_arr[valid]

    split_mat[si_valid, yi_valid] = label_valid

    return split_mat, kept_ids, common_years


def _plot_split_panel(ax, split_mat, years, title, panel_label):
    cmap = mcolors.ListedColormap([COLOR_NODATA, COLOR_CAL, COLOR_MS, COLOR_TEST])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(split_mat[:N_STATIONS_SHOW].T, aspect="auto", cmap=cmap, norm=norm,
              interpolation="nearest", origin="lower")

    ax.set_xlabel("Station", fontsize=11)
    ax.set_ylabel("Year", fontsize=11)

    n_yr = len(years)
    ytick_step = max(1, n_yr // 6)
    ytick_idx = np.arange(0, n_yr, ytick_step)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(years[ytick_idx], fontsize=9)

    xtick_step = 10
    xtick_idx = np.arange(0, min(N_STATIONS_SHOW, split_mat.shape[0]), xtick_step)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(xtick_idx, fontsize=9)

    ax.text(0.02, 0.96, f"({panel_label})", fontsize=12, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)


def main():
    with open(MODELS_DIR / "split_calibration.pkl", "rb") as f:
        splits = pickle.load(f)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), facecolor="white")
    csv_rows = []

    for i, var in enumerate(["tas", "pr"]):
        split_info = splits[var]
        split_mat, station_ids, years = _build_split_matrix(var, split_info)

        n_cal = (split_mat == 1).sum()
        n_ms  = (split_mat == 2).sum()
        n_test = (split_mat == 3).sum()
        n_total = n_cal + n_ms + n_test

        n_cal_sta = sum(1 for sid in station_ids
                       if split_info["station_flags"][sid] == "cal")
        n_test_sta = sum(1 for sid in station_ids
                        if split_info["station_flags"][sid] == "test_only")

        print(f"{var}: {len(station_ids)} stations ({n_cal_sta} cal + {n_test_sta} test-only) × {len(years)} years")
        print(f"  Cal cells: {n_cal:,} ({100*n_cal/n_total:.1f}%), "
              f"MS: {n_ms:,} ({100*n_ms/n_total:.1f}%), "
              f"Test: {n_test:,} ({100*n_test/n_total:.1f}%)")

        csv_rows.append({
            "var": var, "n_stations": len(station_ids),
            "n_cal_stations": n_cal_sta, "n_test_only_stations": n_test_sta,
            "n_years": len(years),
            "n_cal": n_cal, "n_ms": n_ms, "n_test": n_test,
            "pct_cal": round(100*n_cal/n_total, 1),
            "pct_ms": round(100*n_ms/n_total, 1),
            "pct_test": round(100*n_test/n_total, 1),
        })

        _plot_split_panel(axes[i], split_mat, years,
                         f"Split for {var} dataset (first {N_STATIONS_SHOW} stations)",
                         "ab"[i])

    # Compute overall percentages for legend (average of both vars)
    avg_cal = np.mean([r["pct_cal"] for r in csv_rows])
    avg_ms = np.mean([r["pct_ms"] for r in csv_rows])
    avg_test = np.mean([r["pct_test"] for r in csv_rows])
    legend_patches = [
        mpatches.Patch(color=COLOR_CAL, label=f"Calibration ({avg_cal:.1f}%)"),
        mpatches.Patch(color=COLOR_MS, label=f"Model selection ({avg_ms:.1f}%)"),
        mpatches.Patch(color=COLOR_TEST, label=f"Testing ({avg_test:.1f}%)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=11, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    sync_output(OUT, OUT_CSV)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
