"""Step 1: PCR calibration only.

Trains 12 monthly PCR models using calibration-period ESM + GHCN observations.
Does NOT load full transient data. Does NOT project.

Output:
  pcr_models_{var}.pkl  — 12 MonthPCRModel objects
  pcr_diag_{var}.csv    — per-month diagnostics (RMSE, R², etc.)

Run:
  python run_calibrate.py --var tas                  # default: trace21k
  python run_calibrate.py --var tas --gcm mpi-esm-cr # MPI-ESM
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time as _t
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from tqdm import tqdm

from caz.io.ghcn import filter_min_record, filter_time
from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month
from caz.io.mpi_esm import load_mpi_esm_var, mpi_esm_time_to_year_month
from caz.pcr import calibrate_month

DATA_ROOT = Path(r"D:\Dataset\DPastCliM-NA")

GHCN = {
    "tas": DATA_ROOT / "GHCN" / "interim" / "ghcn_tas_obs.parquet",
    "pr":  DATA_ROOT / "GHCN" / "interim" / "ghcn_pr_obs.parquet",
}

GCM_CONFIG = {
    "trace21k": {
        "year_cal_max": 1999,
        "n_year_val": 38,
        "n_year_test": 11,
        "esm_paths": {
            "tas": DATA_ROOT / "TraCE21k" / "TraCE-21K-II.monthly.TREFHT.nc",
            "pr":  DATA_ROOT / "TraCE21k" / "TraCE-21K-II.monthly.PRECT.nc",
        },
    },
    "mpi-esm-cr": {
        "year_cal_max": 1949,
        "n_year_val": 23,
        "n_year_test": 7,
        "esm_dir": DATA_ROOT / "MPI-ESM-CR",
    },
}

YEAR_CAL_MIN = 1875


def _load_esm(gcm: str, var: str, year_min: int, year_max: int):
    """Load ESM data for the cal period, return (DataArray, year, month)."""
    if gcm == "trace21k":
        cfg = GCM_CONFIG[gcm]
        tr = load_trace_var(cfg["esm_paths"][var], var)
        tr = select_na_window(tr)
        yr, mo = trace_time_to_year_month(tr["time"].values)
        tr = tr.assign_coords(year=("time", yr), month=("time", mo))
        mask = (tr.year >= year_min) & (tr.year <= year_max)
        return tr.isel(time=mask.values).load()
    elif gcm == "mpi-esm-cr":
        cfg = GCM_CONFIG[gcm]
        da = load_mpi_esm_var(cfg["esm_dir"] / var, var,
                              year_min_ce=year_min, year_max_ce=year_max)
        da = select_na_window(da)
        yr, mo = mpi_esm_time_to_year_month(da["time"].values)
        da = da.assign_coords(year=("time", yr), month=("time", mo))
        mask = (da.year >= year_min) & (da.year <= year_max)
        return da.isel(time=mask.values).load()
    else:
        raise ValueError(f"Unknown GCM: {gcm}")


def main():
    ap = argparse.ArgumentParser(description="Step 1: PCR calibration")
    ap.add_argument("--var", choices=["tas", "pr"], required=True)
    ap.add_argument("--gcm", default="trace21k", choices=list(GCM_CONFIG.keys()))
    args = ap.parse_args()
    var = args.var
    gcm = args.gcm

    cfg = GCM_CONFIG[gcm]
    YEAR_CAL_MAX = cfg["year_cal_max"]
    N_YEAR_VAL = cfg["n_year_val"]
    N_YEAR_TEST = cfg["n_year_test"]
    OUT_DIR = DATA_ROOT / "interim" / gcm / "models"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timings = {}
    print("=" * 60)
    print(f"Step 1: PCR Calibration — {var} [{gcm}]")
    print(f"  Cal period: {YEAR_CAL_MIN}-{YEAR_CAL_MAX}")
    print("=" * 60)
    t_total = _t.time()

    # ── Load obs ──────────────────────────────────────────────
    t0 = _t.time()
    print(f"\n[1/3] Loading GHCN obs ...", flush=True)
    obs = pd.read_parquet(GHCN[var])
    obs = filter_time(obs, YEAR_CAL_MIN, YEAR_CAL_MAX)
    obs = filter_min_record(obs, min_years=20)
    timings["load_obs"] = _t.time() - t0
    print(f"       {len(obs):,} rows / {obs['ID'].nunique():,} stations  "
          f"({timings['load_obs']:.1f}s)", flush=True)

    # ── Load ESM cal-period only ──────────────────────────────
    t0 = _t.time()
    print(f"\n[2/3] Loading {gcm} cal-period ({YEAR_CAL_MIN}-{YEAR_CAL_MAX}) ...", flush=True)
    tr_cal = _load_esm(gcm, var, YEAR_CAL_MIN, YEAR_CAL_MAX)
    timings["load_esm"] = _t.time() - t0
    print(f"       shape: {tr_cal.shape}  ({timings['load_esm']:.1f}s)", flush=True)
    mem_mb = tr_cal.values.nbytes / 1e6
    print(f"       RAM: {mem_mb:.1f} MB", flush=True)

    # ── Load global split ─────────────────────────────────────
    split_path = OUT_DIR / "split_calibration.pkl"
    if split_path.exists():
        with open(split_path, "rb") as f:
            split_data = pickle.load(f)
        sp = split_data[var]
        global_split = (sp["idx_cal"], sp["idx_val"], sp["idx_test"])
        print(f"       Split loaded: cal={len(sp['idx_cal'])}, "
              f"val={len(sp['idx_val'])}, test={len(sp['idx_test'])}", flush=True)
    else:
        global_split = None
        print("       WARNING: split_calibration.pkl not found, using legacy per-month split", flush=True)

    # ── Calibrate 12 months ───────────────────────────────────
    print(f"\n[3/3] Calibrating 12 months ...", flush=True)
    rng = np.random.default_rng(2026)
    models = {}
    diag_rows = []
    t_cal_total = _t.time()

    for month in tqdm(range(1, 13), desc="Calibrating", unit="month"):
        t0 = _t.time()
        m_mask = tr_cal.month.values == month
        esm_m = tr_cal.isel(time=m_mask)
        if esm_m.sizes["time"] < 30:
            print(f"  month {month:2d}: skipped (< 30 samples)", flush=True)
            continue

        try:
            mdl = calibrate_month(
                month=month, esm_da=esm_m, obs_long=obs,
                var_name=var, n_year_val=N_YEAR_VAL, n_year_test=N_YEAR_TEST,
                rng=rng,
                split_indices=global_split,
            )
        except ValueError as e:
            print(f"  month {month:2d}: FAILED — {e}", flush=True)
            continue

        dt = _t.time() - t0
        models[month] = mdl
        rmse_med = float(np.nanmedian(mdl.rmse_train))
        r2_med = float(np.nanmedian(mdl.r2_train))
        rmse_te = float(np.nanmedian(mdl.rmse_test)) if mdl.rmse_test is not None else np.nan
        ev_top = float(mdl.pc_var_frac[mdl.pc_indices].sum() * 100)

        print(f"  month {month:2d}: {len(mdl.station_id):>5,d} sta  "
              f"PCs={mdl.n_pc} ({ev_top:4.1f}% EV)  "
              f"RMSE tr={rmse_med:.2f} te={rmse_te:.2f}  "
              f"R2={r2_med:.3f}  ({dt:.1f}s)", flush=True)

        diag_rows.append({
            "month": month, "n_stations": len(mdl.station_id),
            "n_pc": mdl.n_pc, "ev_pct": ev_top,
            "rmse_train": rmse_med, "rmse_test": rmse_te,
            "r2_train": r2_med, "time_s": dt,
        })
        timings[f"cal_month_{month:02d}"] = dt

    timings["cal_total"] = _t.time() - t_cal_total

    if not models:
        raise RuntimeError("No months calibrated")

    # ── Save ──────────────────────────────────────────────────
    pkl_path = OUT_DIR / f"pcr_models_{var}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(models, f)
    pkl_mb = pkl_path.stat().st_size / 1e6

    diag_df = pd.DataFrame(diag_rows)
    diag_csv = OUT_DIR / f"pcr_diag_{var}.csv"
    diag_df.to_csv(diag_csv, index=False)

    timings["total"] = _t.time() - t_total

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"DONE: {var} calibration")
    print(f"  Models: {pkl_path}  ({pkl_mb:.1f} MB)")
    print(f"  Diag:   {diag_csv}")
    print(f"  Time:   {timings['total']:.1f}s total "
          f"({timings['cal_total']:.1f}s calibration)")
    print("=" * 60, flush=True)

    # Save timing log
    timing_path = OUT_DIR / f"timing_calibrate_{var}.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"  Timing log: {timing_path}", flush=True)


if __name__ == "__main__":
    main()
