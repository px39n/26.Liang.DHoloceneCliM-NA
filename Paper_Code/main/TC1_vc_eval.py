"""TC1: Variance correction evaluation table.

Compares gridded PCR bias metrics (mean, P10, P90) before and after
applying Guaita's 30-month moving-window variance correction, evaluated
on the test set against GHCN-m observations.

Input:  grid_pcr_raw_{var}.nc, grid_obs_test_{var}.nc, grid_esm_cal_{var}.nc
Output: TC1_vc_eval.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))
from _common import DATA_DIR, GRID_CAL, RESULTS_MAIN

import numpy as np
import pandas as pd
import xarray as xr

GRID_DIR = GRID_CAL
RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_MAIN / "TC1_vc_eval.csv"
VC_WINDOW = 30  # months (Guaita's original setting)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from caz.gridding import variance_correction


def _load(var: str):
    obs = xr.open_dataset(GRID_DIR / f"grid_obs_test_{var}.nc")[var].values
    pcr_raw_full = xr.open_dataset(GRID_DIR / f"grid_pcr_raw_{var}.nc")
    esm_cal_full = xr.open_dataset(GRID_DIR / f"grid_esm_cal_{var}.nc")

    year_test = xr.open_dataset(GRID_DIR / f"grid_obs_test_{var}.nc")["year"].values
    year_cal = pcr_raw_full["year"].values
    test_years = np.unique(year_test)
    test_mask = np.isin(year_cal, test_years)

    pcr_raw = pcr_raw_full[var].values
    esm_cal = esm_cal_full[var].values

    # Apply VC on full cal period, then extract test
    print(f"  Applying VC (window={VC_WINDOW}) on full cal grid...", flush=True)
    pcr_vc_full = variance_correction(pcr_raw, esm_cal, window=VC_WINDOW)

    pcr_raw_test = pcr_raw[test_mask]
    pcr_vc_test = pcr_vc_full[test_mask]

    pcr_raw_full.close()
    esm_cal_full.close()
    return obs, pcr_raw_test, pcr_vc_test


def _bias_stats(src, obs):
    """Compute mean bias, P10 diff, P90 diff statistics."""
    s_mean = np.nanmean(src, axis=0)
    o_mean = np.nanmean(obs, axis=0)
    s_p10 = np.nanpercentile(src, 10, axis=0)
    o_p10 = np.nanpercentile(obs, 10, axis=0)
    s_p90 = np.nanpercentile(src, 90, axis=0)
    o_p90 = np.nanpercentile(obs, 90, axis=0)

    b_mean = s_mean - o_mean
    b_p10 = s_p10 - o_p10
    b_p90 = s_p90 - o_p90

    rows = []
    for label, b in [("Mean bias", b_mean), ("P10 diff", b_p10), ("P90 diff", b_p90)]:
        v = np.isfinite(b)
        rows.append({
            "metric": label,
            "median": float(np.nanmedian(b[v])),
            "mae": float(np.nanmean(np.abs(b[v]))),
            "rmse": float(np.sqrt(np.nanmean(b[v] ** 2))),
        })
    return rows


def main():
    print("TC1: Variance Correction Evaluation", flush=True)
    all_rows = []

    for var in ["tas", "pr"]:
        print(f"\n  Loading {var}...", flush=True)
        obs, pcr_raw, pcr_vc = _load(var)
        print(f"  Test shape: {obs.shape}", flush=True)

        for src_name, src_data in [("Raw", pcr_raw), ("VC", pcr_vc)]:
            stats = _bias_stats(src_data, obs)
            for s in stats:
                s["var"] = var
                s["source"] = src_name
            all_rows.extend(stats)

    df = pd.DataFrame(all_rows)
    df = df[["var", "source", "metric", "median", "mae", "rmse"]]
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\nSaved: {OUT_CSV}", flush=True)


    # Print summary
    print("\n" + df.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
