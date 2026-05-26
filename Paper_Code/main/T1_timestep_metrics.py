"""T1: Timestep-level performance metrics (our Table 1 = Guaita Table 1).

Computes MB, MAE, KGE on test-set observations at each timestep (station-month).
Annual + seasonal (MAM, JJA, SON, DJF) for both PCR and ESM.
Supports multiple GCMs — each GCM adds PCR/ESM columns.

Input:  recon_cal_{var}.csv, GHCN obs, ESM data (per GCM)
Output: T1_timestep_metrics.csv
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import numpy as np
import pandas as pd
from _common import GHCN_DIR, DATA_DIR, RESULTS_MAIN, sync_output

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_MAIN / "T1_timestep_metrics.csv"

GCMS = [
    {"name": "trace21k", "label": "TraCE", "year_cal_max": 1999},
    {"name": "mpi-esm-cr", "label": "MPI", "year_cal_max": 1949},
]

SEASONS = {
    "Annual": list(range(1, 13)),
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
}


def kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 10:
        return np.nan
    o, s = obs[mask], sim[mask]
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def _load_esm_at_stations(gcm_name: str, var: str, station_ids, lats, lons,
                          years, months, year_cal_max: int):
    """Get raw ESM values at station locations."""
    from caz.io.trace import select_na_window

    if gcm_name == "trace21k":
        from caz.io.trace import load_trace_var, trace_time_to_year_month
        esm_paths = {
            "tas": DATA_DIR / "TraCE21k" / "TraCE-21K-II.monthly.TREFHT.nc",
            "pr":  DATA_DIR / "TraCE21k" / "TraCE-21K-II.monthly.PRECT.nc",
        }
        tr = load_trace_var(esm_paths[var], var)
        tr = select_na_window(tr)
        yr, mo = trace_time_to_year_month(tr["time"].values)
    else:
        from caz.io.mpi_esm import load_mpi_esm_var, mpi_esm_time_to_year_month
        esm_dir = DATA_DIR / "MPI-ESM-CR"
        tr = load_mpi_esm_var(esm_dir / var, var,
                              year_min_ce=1875, year_max_ce=year_cal_max)
        tr = select_na_window(tr)
        yr, mo = mpi_esm_time_to_year_month(tr["time"].values)

    tr = tr.assign_coords(year=("time", yr), month=("time", mo))
    cal_mask = (tr.year >= 1875) & (tr.year <= year_cal_max)
    tr_cal = tr.isel(time=cal_mask.values).load()

    yr_arr = tr_cal.year.values
    mo_arr = tr_cal.month.values
    ym_to_tidx = {}
    for t_i in range(len(yr_arr)):
        ym_to_tidx[(int(yr_arr[t_i]), int(mo_arr[t_i]))] = t_i

    unique_sids = np.unique(station_ids)
    sid_to_lat = dict(zip(station_ids, lats))
    sid_to_lon = dict(zip(station_ids, lons))

    esm_vals = np.full(len(station_ids), np.nan, dtype=np.float32)
    for sid in unique_sids:
        lat_s = sid_to_lat[sid]
        lon_s = sid_to_lon[sid]
        esm_point = tr_cal.sel(lat=lat_s, lon=lon_s, method="nearest").values
        mask = station_ids == sid
        for idx in np.where(mask)[0]:
            y, m = int(years[idx]), int(months[idx])
            t_i = ym_to_tidx.get((y, m))
            if t_i is not None:
                esm_vals[idx] = esm_point[t_i]

    return esm_vals


def _load_test_recon(gcm_name: str, var: str):
    """Load recon CSV and filter to test rows using split pkl."""
    station_cal = DATA_DIR / "interim" / gcm_name / "station_cal"
    model_dir = DATA_DIR / "interim" / gcm_name / "models"

    with open(model_dir / "split_calibration.pkl", "rb") as f:
        split_info = pickle.load(f)[var]
    test_years = set(int(y) for y in split_info["test_years"])

    recon = pd.read_csv(station_cal / f"recon_cal_{var}.csv",
                        usecols=["station_id", "lon", "lat", "year", "month", "value"])
    recon_test = recon[recon["year"].isin(test_years)].copy()
    return recon_test


def main():
    rows = []

    for var in ["tas", "pr"]:
        print(f"\n{'='*60}")
        print(f"  {var}")
        print(f"{'='*60}")

        obs = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_obs.parquet")
        obs_sub = obs[["ID", "year", "month", "value"]].rename(
            columns={"ID": "station_id", "value": "obs"})

        for gcm in GCMS:
            gname = gcm["name"]
            glabel = gcm["label"]
            print(f"\n  --- {glabel} ({gname}) ---", flush=True)

            recon_test = _load_test_recon(gname, var)
            print(f"  {len(recon_test):,} test rows", flush=True)

            merged = recon_test.merge(obs_sub, on=["station_id", "year", "month"],
                                      how="inner")
            print(f"  {len(merged):,} matched rows", flush=True)

            print("  Loading ESM ...", flush=True)
            merged["esm"] = _load_esm_at_stations(
                gname, var,
                merged["station_id"].values,
                merged["lat"].values.astype(float),
                merged["lon"].values.astype(float),
                merged["year"].values.astype(int),
                merged["month"].values.astype(int),
                gcm["year_cal_max"],
            )

            pcr_vals = merged["value"].values.astype(float)
            obs_vals = merged["obs"].values.astype(float)
            esm_vals = merged["esm"].values.astype(float)
            month_vals = merged["month"].values.astype(int)

            for season_name, season_months in SEASONS.items():
                s_mask = np.isin(month_vals, season_months)
                pcr_s, obs_s, esm_s = pcr_vals[s_mask], obs_vals[s_mask], esm_vals[s_mask]

                valid_pcr = np.isfinite(pcr_s) & np.isfinite(obs_s)
                valid_esm = np.isfinite(esm_s) & np.isfinite(obs_s)

                mb_pcr = np.nanmean(pcr_s[valid_pcr] - obs_s[valid_pcr])
                mae_pcr = np.nanmean(np.abs(pcr_s[valid_pcr] - obs_s[valid_pcr]))
                kge_pcr = kge(obs_s[valid_pcr], pcr_s[valid_pcr])

                mb_esm = np.nanmean(esm_s[valid_esm] - obs_s[valid_esm])
                mae_esm = np.nanmean(np.abs(esm_s[valid_esm] - obs_s[valid_esm]))
                kge_esm = kge(obs_s[valid_esm], esm_s[valid_esm])

                rows.append({
                    "var": var, "gcm": glabel, "season": season_name,
                    "metric": "MB",
                    "PCR": round(mb_pcr, 3), "ESM": round(mb_esm, 3),
                    "n": int(valid_pcr.sum()),
                })
                rows.append({
                    "var": var, "gcm": glabel, "season": season_name,
                    "metric": "MAE",
                    "PCR": round(mae_pcr, 3), "ESM": round(mae_esm, 3),
                    "n": int(valid_pcr.sum()),
                })
                rows.append({
                    "var": var, "gcm": glabel, "season": season_name,
                    "metric": "KGE",
                    "PCR": round(kge_pcr, 3), "ESM": round(kge_esm, 3),
                    "n": int(valid_pcr.sum()),
                })

                print(f"  {season_name:8s}  MB: PCR={mb_pcr:+.3f} ESM={mb_esm:+.3f}  "
                      f"MAE: {mae_pcr:.3f}/{mae_esm:.3f}  KGE: {kge_pcr:.3f}/{kge_esm:.3f}  "
                      f"(n={int(valid_pcr.sum()):,})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
