"""F4: Station-level timeseries + PI (our Fig 4 = Guaita Fig 5).

Layout: 4 rows (tas-1, pr-1, tas-2, pr-2) × 3 columns (time windows).
Matches Guaita's plot_downscaled_timeline_point.m style.

Colors (colorblind-friendly, matching MATLAB):
  Dark gray = GHCN-m obs
  Vermilion = raw ESM
  Sky blue  = PCR downscaled
  Light gray shading = 95% PI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from _common import GHCN_DIR, MODELS_DIR, STATION_CAL, TRACE_DIR, RESULTS_MAIN, sync_output

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_MAIN / "F4_station_timeseries.png"
OUT_CSV = RESULTS_MAIN / "F4_station_timeseries.csv"

WINDOWS = [(1900, 1905), (1950, 1955), (1995, 1999)]
WINDOW_LABELS = ["1900–1905", "1950–1955", "1995–1999"]

# Guaita's colorblind-friendly palette
C_OBS = (0.2, 0.2, 0.2)       # Dark gray
C_ESM = (0.90, 0.45, 0.30)    # Vermilion
C_PCR = (0.35, 0.70, 0.90)    # Sky blue
C_PI  = (0.7, 0.7, 0.7)       # Light gray


def _load_esm_cal(var: str):
    from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month
    trace_files = {
        "tas": TRACE_DIR / "TraCE-21K-II.monthly.TREFHT.nc",
        "pr":  TRACE_DIR / "TraCE-21K-II.monthly.PRECT.nc",
    }
    tr = load_trace_var(trace_files[var], var)
    tr = select_na_window(tr)
    yr, mo = trace_time_to_year_month(tr["time"].values)
    tr = tr.assign_coords(year=("time", yr), month=("time", mo))
    cal_mask = (tr.year >= 1875) & (tr.year <= 1999)
    return tr.isel(time=cal_mask.values).load()


def _plot_panel(ax, sid, recon, obs_df, tr_cal, var, win,
               is_top_row=False, is_left_col=False):
    y0, y1 = win
    r = recon[(recon["station_id"] == sid) &
              (recon["year"] >= y0) & (recon["year"] <= y1)].copy()
    r = r.sort_values(["year", "month"])
    if len(r) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return

    # X-axis as fractional years
    x_r = r["year"].values + (r["month"].values - 1) / 12.0

    # PI shading
    ax.fill_between(x_r, r["pi_lo"].values, r["pi_hi"].values,
                    color=C_PI, alpha=0.2, edgecolor="none")
    # ESM
    lat_val, lon_val = r["lat"].iloc[0], r["lon"].iloc[0]
    esm_ts = tr_cal.sel(lat=lat_val, lon=lon_val, method="nearest")
    esm_mask = (esm_ts.year >= y0) & (esm_ts.year <= y1)
    esm_win = esm_ts.isel(time=esm_mask.values)
    x_esm = esm_win.year.values + (esm_win.month.values - 1) / 12.0
    ax.plot(x_esm, esm_win.values, linewidth=0.8, color=C_ESM, label="ESM", zorder=1)

    # Obs
    o = obs_df[(obs_df["ID"] == sid) &
               (obs_df["year"] >= y0) & (obs_df["year"] <= y1)].copy()
    if len(o) > 0:
        o = o.sort_values(["year", "month"])
        x_obs = o["year"].values + (o["month"].values - 1) / 12.0
        ax.plot(x_obs, o["value"].values, linewidth=0.8, color=C_OBS,
                label="GHCN-m", zorder=2)

    # PCR
    ax.plot(x_r, r["value"].values, linewidth=1.0, color=C_PCR,
            label="PCR", zorder=3)

    # Mean dashed line
    mean_val = r["value"].mean()
    ax.axhline(mean_val, color=C_PCR, linestyle="--", linewidth=0.8)

    # X-axis: only start and end
    ax.set_xlim(y0 - 0.25, y1 + 0.25)
    ax.set_xticks([y0, y1])
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    # Y limits
    if var == "pr":
        ax.set_ylim(bottom=0)


def main():
    with open(MODELS_DIR / "split_calibration.pkl", "rb") as f:
        splits = pickle.load(f)

    # 2 locations, each with co-located tas + pr
    tas_meta = pd.read_parquet(GHCN_DIR / "ghcn_tas_meta.parquet")
    pr_meta = pd.read_parquet(GHCN_DIR / "ghcn_pr_meta.parquet")
    tas_cal = set(s for s, f in splits["tas"]["station_flags"].items() if f == "cal")
    pr_cal = set(s for s, f in splits["pr"]["station_flags"].items() if f == "cal")
    tas_meta = tas_meta[tas_meta["ID"].isin(tas_cal)]
    pr_meta = pr_meta[pr_meta["ID"].isin(pr_cal)]

    loc_lats = [30.0, 55.0]
    station_list = []  # [(var, sid, label), ...]
    for i, target_lat in enumerate(loc_lats):
        tas_meta_tmp = tas_meta.copy()
        tas_meta_tmp["dist"] = np.abs(tas_meta_tmp["lat"] - target_lat)
        tas_sid = tas_meta_tmp.nsmallest(1, "dist")["ID"].iloc[0]
        tas_row = tas_meta_tmp[tas_meta_tmp["ID"] == tas_sid].iloc[0]

        pr_meta_tmp = pr_meta.copy()
        pr_meta_tmp["dist"] = np.sqrt((pr_meta_tmp["lat"] - tas_row["lat"])**2 +
                                       (pr_meta_tmp["lon"] - tas_row["lon"])**2)
        pr_sid = pr_meta_tmp.nsmallest(1, "dist")["ID"].iloc[0]

        station_list.append(("tas", tas_sid, f"tas-{i+1}"))
        station_list.append(("pr", pr_sid, f"pr-{i+1}"))

    print(f"Stations: {[(v, s, l) for v, s, l in station_list]}")

    n_rows = len(station_list)  # 4
    n_cols = len(WINDOWS)       # 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), facecolor="white",
                             squeeze=False)

    csv_rows = []
    loaded_esm = {}
    loaded_recon = {}
    loaded_obs = {}

    for row, (var, sid, label) in enumerate(station_list):
        if var not in loaded_esm:
            print(f"Loading {var} data ...", flush=True)
            loaded_esm[var] = _load_esm_cal(var)
            loaded_obs[var] = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_obs.parquet")
            sid_set = set(s for v, s, _ in station_list if v == var)
            chunks = []
            for chunk in pd.read_csv(STATION_CAL / f"recon_cal_{var}.csv", chunksize=500_000):
                chunks.append(chunk[chunk["station_id"].isin(sid_set)])
            loaded_recon[var] = pd.concat(chunks, ignore_index=True)

        recon = loaded_recon[var]
        obs = loaded_obs[var]
        tr_cal = loaded_esm[var]

        r_sample = recon[recon["station_id"] == sid]
        if len(r_sample) == 0:
            print(f"  WARNING: {sid} not found")
            continue
        lat_val = float(r_sample["lat"].iloc[0])
        lon_val = float(r_sample["lon"].iloc[0])
        unit = "°C" if var == "tas" else "mm day⁻¹"

        for col, (win, wlabel) in enumerate(zip(WINDOWS, WINDOW_LABELS)):
            ax = axes[row, col]
            _plot_panel(ax, sid, recon, obs, tr_cal, var, win)

            if row == 0:
                ax.set_title(wlabel, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(unit, fontsize=10)
            if col == n_cols - 1:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(f"{label}\n({lat_val:.1f}°N, {lon_val:.1f}°E)",
                              fontsize=9, rotation=0, labelpad=50, va="center")

        # Legend on first panel only
        if row == 0:
            axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.8)

        csv_rows.append({"label": label, "var": var, "station_id": sid,
                        "lat": lat_val, "lon": lon_val})

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    sync_output(OUT, OUT_CSV)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
