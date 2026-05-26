"""T2: Station-level performance metrics (our Table 2 = Guaita Table 2).

For each station: time-average predictions and obs over test set,
then compute r, MB, MAE across stations.
Annual + seasonal (MAM, JJA, SON, DJF) for both PCR and ESM.
Supports multiple GCMs.

Input:  test_cache_{var}.parquet per GCM (from _build_test_cache.py)
Output: T2_station_metrics.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import numpy as np
import pandas as pd
from _common import DATA_DIR, RESULTS_MAIN

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_MAIN / "T2_station_metrics.csv"

GCMS = [
    {"name": "trace21k", "label": "TraCE"},
    {"name": "mpi-esm-cr", "label": "MPI"},
]

SEASONS = {
    "Annual": list(range(1, 13)),
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
}


def main():
    rows = []

    for var in ["tas", "pr"]:
        print(f"\n{'='*60}")
        print(f"  {var}")
        print(f"{'='*60}")

        for gcm in GCMS:
            gname, glabel = gcm["name"], gcm["label"]
            cache_path = DATA_DIR / "interim" / gname / "station_cal" / f"test_cache_{var}.parquet"
            print(f"\n  --- {glabel} ({gname}) ---", flush=True)
            df = pd.read_parquet(cache_path)
            print(f"  {len(df):,} rows from cache", flush=True)

            for season_name, season_months in SEASONS.items():
                s = df[df["month"].isin(season_months)].copy()

                sta = s.groupby("station_id").agg(
                    pcr_mean=("value", "mean"),
                    obs_mean=("obs", "mean"),
                    esm_mean=("esm", "mean"),
                    n_obs=("obs", "count"),
                ).dropna()

                n_sta = len(sta)
                if n_sta < 3:
                    print(f"  {season_name}: too few stations ({n_sta})")
                    continue

                pcr_m = sta["pcr_mean"].values
                obs_m = sta["obs_mean"].values
                esm_m = sta["esm_mean"].values

                valid_pcr = np.isfinite(pcr_m) & np.isfinite(obs_m)
                valid_esm = np.isfinite(esm_m) & np.isfinite(obs_m)
                r_pcr = np.corrcoef(pcr_m[valid_pcr], obs_m[valid_pcr])[0, 1]
                r_esm = np.corrcoef(esm_m[valid_esm], obs_m[valid_esm])[0, 1]

                mb_pcr = np.mean(pcr_m[valid_pcr] - obs_m[valid_pcr])
                mb_esm = np.mean(esm_m[valid_esm] - obs_m[valid_esm])

                mae_pcr = np.mean(np.abs(pcr_m[valid_pcr] - obs_m[valid_pcr]))
                mae_esm = np.mean(np.abs(esm_m[valid_esm] - obs_m[valid_esm]))

                print(f"  {season_name:8s}  r: {r_pcr:.3f}/{r_esm:.3f}  "
                      f"MB: {mb_pcr:+.3f}/{mb_esm:+.3f}  "
                      f"MAE: {mae_pcr:.3f}/{mae_esm:.3f}  (n={n_sta})")

                for metric, pcr_val, esm_val in [
                    ("r", r_pcr, r_esm),
                    ("MB", mb_pcr, mb_esm),
                    ("MAE", mae_pcr, mae_esm),
                ]:
                    rows.append({
                        "var": var, "gcm": glabel, "season": season_name,
                        "metric": metric,
                        "PCR": round(pcr_val, 3), "ESM": round(esm_val, 3),
                        "n_stations": n_sta,
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
