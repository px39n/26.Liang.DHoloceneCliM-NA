"""Step 0: Generate global train/val/test split for Calibration Production.

Follows Guaita's PCR_calibration_v5.m:
- ONE global randperm → same cal/val/test years for ALL 12 months
- Station screening:
  - < 20 cal years (across all 12 months) → removed
  - 20-30 cal years → demoted to test_only
  - >= 30 cal years → included in calibration

Output: split_calibration.pkl → single source of truth for all downstream steps.

Per-GCM splits: each GCM may have a different YEAR_END based on ESM temporal
coverage. Same SEED ensures comparable random assignment; station_flags are
recomputed per GCM because fewer cal years may change station eligibility.

Run:
  python generate_split.py                    # default: trace21k (1875-1999)
  python generate_split.py --gcm mpi-esm-cr   # MPI-ESM (1875-1949)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

GHCN_DIR = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim")
DATA_ROOT = Path(r"D:\Dataset\DPastCliM-NA")

SEED = 2026

YEAR_START = 1875

# TraCE is the reference GCM. Thresholds for other GCMs are scaled
# proportionally to their n_cal / trace_n_cal.
TRACE_MIN_REMOVE = 20
TRACE_MIN_CAL    = 30

GCM_CONFIG = {
    "trace21k": {
        "year_end": 1999,
        "n_year_val": 38,
        "n_year_test": 11,
    },
    "mpi-esm-cr": {
        "year_end": 1949,
        "n_year_val": 23,
        "n_year_test": 7,
    },
}


def _generate_trace_base_split() -> dict[str, set]:
    """Generate the base year split from TraCE-21k (1875-1999).

    All other GCMs derive their split by subsetting these year sets.
    Returns {role: set_of_years} for cal, val, test.
    """
    cfg = GCM_CONFIG["trace21k"]
    all_years = np.arange(YEAR_START, cfg["year_end"] + 1)
    T = len(all_years)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(T)
    n_val = cfg["n_year_val"]
    n_test = cfg["n_year_test"]

    val_years  = set(all_years[np.sort(perm[:n_val])])
    test_years = set(all_years[np.sort(perm[n_val:n_val + n_test])])
    cal_years  = set(all_years[np.sort(perm[n_val + n_test:])])
    return {"cal": cal_years, "val": val_years, "test": test_years}


def generate_split(var: str, year_end: int, base_split: dict[str, set]) -> dict:
    """Generate split for one variable, deriving from TraCE base split.

    For TraCE itself this is identity; for other GCMs it subsets years
    to [YEAR_START, year_end] and recomputes station flags.
    """
    obs = pd.read_parquet(GHCN_DIR / f"ghcn_{var}_obs.parquet")

    common_years = np.arange(YEAR_START, year_end + 1)
    year_set = set(common_years)

    cal_years_set  = base_split["cal"]  & year_set
    val_years_set  = base_split["val"]  & year_set
    test_years_set = base_split["test"] & year_set

    cal_arr  = np.sort(np.array(list(cal_years_set)))
    val_arr  = np.sort(np.array(list(val_years_set)))
    test_arr = np.sort(np.array(list(test_years_set)))

    idx_cal  = np.searchsorted(common_years, cal_arr)
    idx_val  = np.searchsorted(common_years, val_arr)
    idx_test = np.searchsorted(common_years, test_arr)

    # Scale station thresholds proportionally to TraCE's n_cal (76).
    trace_cfg = GCM_CONFIG["trace21k"]
    trace_total = trace_cfg["year_end"] - YEAR_START + 1
    trace_n_cal = trace_total - trace_cfg["n_year_val"] - trace_cfg["n_year_test"]
    n_cal = len(cal_arr)
    scale = n_cal / trace_n_cal
    min_remove = max(1, int(round(TRACE_MIN_REMOVE * scale)))
    min_cal_th = max(min_remove + 1, int(round(TRACE_MIN_CAL * scale)))

    all_station_ids = sorted(obs["ID"].unique())
    obs_cal = obs[obs["year"].isin(cal_years_set)]
    cal_months_per_station = obs_cal.groupby("ID").size()

    station_flags = {}
    for sid in all_station_ids:
        n_cal_months = cal_months_per_station.get(sid, 0)
        n_cal_years = n_cal_months // 12
        if n_cal_years < min_remove:
            station_flags[sid] = "removed"
        elif n_cal_years < min_cal_th:
            station_flags[sid] = "test_only"
        else:
            station_flags[sid] = "cal"

    n_cal_sta = sum(1 for v in station_flags.values() if v == "cal")
    n_test_sta = sum(1 for v in station_flags.values() if v == "test_only")
    n_removed = sum(1 for v in station_flags.values() if v == "removed")

    T = len(common_years)
    print(f"  {var}: {len(all_station_ids)} total stations")
    print(f"    thresholds: remove<{min_remove} yr, test_only {min_remove}-{min_cal_th} yr, cal>={min_cal_th} yr")
    print(f"    cal: {n_cal_sta}, test_only: {n_test_sta}, removed: {n_removed}")
    print(f"    years: {T} ({YEAR_START}-{year_end})")
    print(f"    split: cal={len(cal_arr)}, val={len(val_arr)}, test={len(test_arr)}")

    return {
        "var": var,
        "common_years": common_years,
        "idx_cal": idx_cal,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "cal_years": cal_arr,
        "val_years": val_arr,
        "test_years": test_arr,
        "station_flags": station_flags,
        "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser(description="Step 0: Generate cal/val/test split")
    ap.add_argument("--gcm", default="trace21k", choices=list(GCM_CONFIG.keys()),
                    help="GCM identifier (determines year range)")
    args = ap.parse_args()

    gcm = args.gcm
    cfg = GCM_CONFIG[gcm]
    out_dir = DATA_ROOT / "interim" / gcm / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GCM: {gcm}  |  years: {YEAR_START}-{cfg['year_end']}")

    base_split = _generate_trace_base_split()
    print(f"Base split (TraCE): cal={len(base_split['cal'])}, "
          f"val={len(base_split['val'])}, test={len(base_split['test'])}")

    splits = {}
    for var in ["tas", "pr"]:
        print(f"\n=== {var} ===")
        splits[var] = generate_split(var, cfg["year_end"], base_split)

    out_path = out_dir / "split_calibration.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
