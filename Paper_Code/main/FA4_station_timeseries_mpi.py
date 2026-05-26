"""FA4: Station-level timeseries + PI for MPI-ESM (appendix version of F4).

Layout: 4 rows (tas-1, pr-1, tas-2, pr-2) x 3 columns (time windows).
MPI-ESM calibration period: 1875-1949.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import GHCN_DIR, DATA_DIR, RESULTS_MAIN, sync_output

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_MAIN / "FA4_station_timeseries_mpi.png"
OUT_CSV = RESULTS_MAIN / "FA4_station_timeseries_mpi.csv"

GCM = "mpi-esm-cr"
MODELS_DIR = DATA_DIR / "interim" / GCM / "models"
STATION_CAL = DATA_DIR / "interim" / GCM / "station_cal"

WINDOWS = [(1880, 1885), (1910, 1915), (1945, 1949)]
WINDOW_LABELS = ["1880–1885", "1910–1915", "1945–1949"]

C_OBS = (0.2, 0.2, 0.2)
C_ESM = (0.90, 0.45, 0.30)
C_PCR = (0.35, 0.70, 0.90)
C_PI  = (0.7, 0.7, 0.7)


def _load_esm_cal(var: str):
    from caz.io.mpi_esm import load_mpi_esm_var, mpi_esm_time_to_year_month
    from caz.io.trace import select_na_window
    esm_dir = DATA_DIR / "MPI-ESM-CR"
    tr = load_mpi_esm_var(esm_dir / var, var, year_min_ce=1875, year_max_ce=1949)
    tr = select_na_window(tr)
    yr, mo = mpi_esm_time_to_year_month(tr["time"].values)
    tr = tr.assign_coords(year=("time", yr), month=("time", mo))
    return tr.load()


def _plot_panel(ax, sid, recon, obs_df, tr_cal, var, win):
    y0, y1 = win
    r = recon[(recon["station_id"] == sid) &
              (recon["year"] >= y0) & (recon["year"] <= y1)].copy()
    r = r.sort_values(["year", "month"])
    if len(r) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return

    x_r = r["year"].values + (r["month"].values - 1) / 12.0

    ax.fill_between(x_r, r["pi_lo"].values, r["pi_hi"].values,
                    color=C_PI, alpha=0.2, edgecolor="none")

    lat_val, lon_val = r["lat"].iloc[0], r["lon"].iloc[0]
    esm_ts = tr_cal.sel(lat=lat_val, lon=lon_val, method="nearest")
    esm_mask = (esm_ts.year >= y0) & (esm_ts.year <= y1)
    esm_win = esm_ts.isel(time=esm_mask.values)
    x_esm = esm_win.year.values + (esm_win.month.values - 1) / 12.0
    ax.plot(x_esm, esm_win.values, linewidth=0.8, color=C_ESM, label="ESM", zorder=1)

    o = obs_df[(obs_df["ID"] == sid) &
               (obs_df["year"] >= y0) & (obs_df["year"] <= y1)].copy()
    if len(o) > 0:
        o = o.sort_values(["year", "month"])
        x_obs = o["year"].values + (o["month"].values - 1) / 12.0
        ax.plot(x_obs, o["value"].values, linewidth=0.8, color=C_OBS,
                label="GHCN-m", zorder=2)

    ax.plot(x_r, r["value"].values, linewidth=1.0, color=C_PCR,
            label="PCR", zorder=3)

    mean_val = r["value"].mean()
    ax.axhline(mean_val, color=C_PCR, linestyle="--", linewidth=0.8)

    ax.set_xlim(y0 - 0.25, y1 + 0.25)
    ax.set_xticks([y0, y1])
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    if var == "pr":
        ax.set_ylim(bottom=0)


def main():
    with open(MODELS_DIR / "split_calibration.pkl", "rb") as f:
        splits = pickle.load(f)

    tas_meta = pd.read_parquet(GHCN_DIR / "ghcn_tas_meta.parquet")
    pr_meta = pd.read_parquet(GHCN_DIR / "ghcn_pr_meta.parquet")
    tas_cal = set(s for s, f in splits["tas"]["station_flags"].items() if f == "cal")
    pr_cal = set(s for s, f in splits["pr"]["station_flags"].items() if f == "cal")
    tas_meta = tas_meta[tas_meta["ID"].isin(tas_cal)]
    pr_meta = pr_meta[pr_meta["ID"].isin(pr_cal)]

    loc_lats = [30.0, 55.0]
    station_list = []
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

    n_rows = len(station_list)
    n_cols = len(WINDOWS)
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

        if row == 0:
            axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.8)

        csv_rows.append({"label": label, "var": var, "station_id": sid,
                        "lat": lat_val, "lon": lon_val})

    fig.suptitle("MPI-ESM 1.2 CR — Station Timeseries + 95% PI",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    sync_output(OUT, OUT_CSV, is_supplementary=True)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
