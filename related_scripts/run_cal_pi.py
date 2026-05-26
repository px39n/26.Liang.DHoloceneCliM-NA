"""Step 2: Fit SEM + ARMA noise models on calibration residuals.

Reads pcr_models_{var}.pkl, recomputes calibration residuals,
fits SEM (spatial) and ARMA (temporal) noise models per month.

Output:
  sem_model_{var}_m{01-12}.pkl   — per-month: lambda, sigma2, W, threshold, valid_mask
  arma_model_{var}_m{01-12}.pkl  — per-month: list of {ar, ma, variance} per station

Run:
  python related_scripts/run_cal_pi.py --var tas                  # default: trace21k
  python related_scripts/run_cal_pi.py --var tas --gcm mpi-esm-cr # MPI-ESM
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
from caz.sem import fit_sem, fit_arma_per_station
from caz.gridding import _albers_forward

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
    """Load ESM data for the cal period, return loaded DataArray with year/month coords."""
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


def _recompute_residuals(mdl, esm_cal_field, obs_long, month, var_name,
                         year_cal_min, year_cal_max, n_year_val, n_year_test):
    """Recompute calibration residuals from a trained model.

    Returns (residual_mat, valid_mask, coords_albers).
    residual_mat: (S, T_cal) with NaN for missing.
    """
    T, ny, nx = esm_cal_field.shape
    field = esm_cal_field.reshape(T, ny * nx).astype(np.float32)

    mu_field = mdl.field_mean

    obs_m = obs_long[obs_long["month"] == month][["ID", "year", "value"]].copy()
    station_years = obs_m.pivot_table(
        index="year", columns="ID", values="value", aggfunc="first"
    )

    esm_years = np.arange(year_cal_min, year_cal_max + 1)
    common_years = station_years.index.intersection(pd.Index(esm_years))
    station_years = station_years.loc[common_years]
    Y_raw = station_years.values.astype(np.float32)

    model_sids = set(mdl.station_id)
    keep = np.array([sid in model_sids for sid in station_years.columns])
    Y_raw = Y_raw[:, keep]
    sid_arr = station_years.columns.values[keep]

    sid_to_model_idx = {s: i for i, s in enumerate(mdl.station_id)}
    model_idx = np.array([sid_to_model_idx[s] for s in sid_arr])

    S = len(sid_arr)

    if var_name == "pr" and mdl.O_t is not None:
        O_t = mdl.O_t[model_idx]
        mu_gO = mdl.mu_gO[model_idx]
        Y = np.full_like(Y_raw, np.nan)
        valid = np.isfinite(Y_raw)
        for s in range(S):
            v = valid[:, s]
            Y[v, s] = np.log(Y_raw[v, s] + O_t[s]) - mu_gO[s]
    else:
        mu_gO_local = np.nanmean(Y_raw, axis=0)
        Y = Y_raw - mu_gO_local[None, :]

    T_common = Y.shape[0]
    rng = np.random.default_rng(2026)
    n_val = min(n_year_val, T_common // 2)
    n_test = min(n_year_test, (T_common - n_val) // 3)
    perm = rng.permutation(T_common)
    idx_cal = np.sort(perm[n_val + n_test:])

    # Use model's stored selected EOFs
    if hasattr(mdl, 'pc_indices') and mdl.pc_indices is not None:
        eofs_for_proj = mdl.eofs[mdl.pc_indices]
    else:
        eofs_for_proj = mdl.eofs

    pc_idx_in_esm = np.searchsorted(esm_years, common_years.values)
    field_centered = field[pc_idx_in_esm] - mu_field[None, :]
    pcs_all = (field_centered @ eofs_for_proj.T).astype(np.float32)

    pcs_cal = pcs_all[idx_cal]
    Y_cal = Y[idx_cal]

    X_cal = np.column_stack([np.ones(pcs_cal.shape[0], dtype=np.float32), pcs_cal])
    beta_sub = mdl.beta[:, model_idx]
    Y_hat = X_cal @ beta_sub

    residual_mat = (Y_cal - Y_hat).T  # (S, T_cal)

    valid_mask = np.array([
        np.isfinite(residual_mat[s]).sum() >= 20 for s in range(S)
    ])

    x_alb, y_alb = _albers_forward(
        mdl.station_lon[model_idx], mdl.station_lat[model_idx]
    )
    coords = np.column_stack([x_alb, y_alb])

    return residual_mat, valid_mask, coords


def main():
    ap = argparse.ArgumentParser(description="Step 2: SEM + ARMA noise model fitting")
    ap.add_argument("--var", choices=["tas", "pr"], required=True)
    ap.add_argument("--gcm", choices=list(GCM_CONFIG.keys()), default="trace21k")
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
    print(f"Step 2: SEM + ARMA Fitting — {var} [{gcm}]")
    print(f"  Cal period: {YEAR_CAL_MIN}-{YEAR_CAL_MAX}")
    print("=" * 60)
    t_total = _t.time()

    # Load models
    t0 = _t.time()
    print(f"\n[1/3] Loading PCR models ...", flush=True)
    with open(OUT_DIR / f"pcr_models_{var}.pkl", "rb") as f:
        models = pickle.load(f)
    print(f"       {len(models)} months loaded", flush=True)
    timings["load_models"] = _t.time() - t0

    # Load obs
    t0 = _t.time()
    print(f"\n[2/3] Loading GHCN obs ...", flush=True)
    obs = pd.read_parquet(GHCN[var])
    obs = filter_time(obs, YEAR_CAL_MIN, YEAR_CAL_MAX)
    obs = filter_min_record(obs, min_years=20)
    print(f"       {len(obs):,} rows / {obs['ID'].nunique():,} stations", flush=True)
    timings["load_obs"] = _t.time() - t0

    # Load ESM cal-period
    t0 = _t.time()
    print(f"       Loading {gcm} cal-period ...", flush=True)
    tr_cal = _load_esm(gcm, var, YEAR_CAL_MIN, YEAR_CAL_MAX)
    print(f"       shape: {tr_cal.shape}", flush=True)
    timings["load_esm"] = _t.time() - t0

    # Fit SEM + ARMA per month
    print(f"\n[3/3] Fitting SEM + ARMA ...", flush=True)
    sem_results = {}
    arma_results = {}

    pbar = tqdm(sorted(models.keys()), desc="SEM+ARMA", unit="month")
    for month in pbar:
        t0 = _t.time()
        mdl = models[month]
        n_sta = len(mdl.station_id)
        pbar.set_postfix_str(f"M{month:02d} ({n_sta} sta) residuals...")

        m_mask = tr_cal.month.values == month
        esm_m_cal = tr_cal.isel(time=m_mask).values

        residual_mat, valid_mask, coords = _recompute_residuals(
            mdl, esm_m_cal, obs, month, var,
            YEAR_CAL_MIN, YEAR_CAL_MAX, N_YEAR_VAL, N_YEAR_TEST
        )

        pbar.set_postfix_str(f"M{month:02d} SEM fitting...")
        t_sem = _t.time()
        lam, sigma2, eps_mat, W, threshold = fit_sem(
            residual_mat, coords, valid_mask
        )
        dt_sem = _t.time() - t_sem

        pbar.set_postfix_str(f"M{month:02d} ARMA fitting...")
        t_arma = _t.time()
        arma_models_month = fit_arma_per_station(eps_mat, valid_mask)
        n_fitted = sum(1 for m in arma_models_month if m is not None)
        dt_arma = _t.time() - t_arma

        sem_month_data = {
            "lambda": lam, "sigma2": sigma2, "W": W,
            "threshold": threshold, "valid_mask": valid_mask,
        }

        # Save per-month (avoids accumulating all months in memory)
        sem_m_path = OUT_DIR / f"sem_model_{var}_m{month:02d}.pkl"
        with open(sem_m_path, "wb") as f:
            pickle.dump(sem_month_data, f)
        arma_m_path = OUT_DIR / f"arma_model_{var}_m{month:02d}.pkl"
        with open(arma_m_path, "wb") as f:
            pickle.dump(arma_models_month, f)

        sem_mb = sem_m_path.stat().st_size / 1e6
        dt_month = _t.time() - t0
        timings[f"month_{month:02d}"] = {"sem": dt_sem, "arma": dt_arma, "total": dt_month}
        pbar.set_postfix_str(
            f"M{month:02d} done | lam={lam:.3f} th={threshold/1000:.0f}km "
            f"SEM={dt_sem:.0f}s ARMA={dt_arma:.0f}s  pkl={sem_mb:.0f}MB"
        )

        del sem_month_data, arma_models_month, residual_mat, eps_mat, W

    sem_mb = sum((OUT_DIR / f"sem_model_{var}_m{m:02d}.pkl").stat().st_size
                 for m in sorted(models.keys())) / 1e6
    arma_mb = sum((OUT_DIR / f"arma_model_{var}_m{m:02d}.pkl").stat().st_size
                  for m in sorted(models.keys())) / 1e6

    timings["total"] = _t.time() - t_total

    # Summary
    print("\n" + "=" * 60)
    print(f"DONE: {var} SEM + ARMA fitting (per-month files)")
    print(f"  SEM total:  {sem_mb:.1f} MB  (12 files)")
    print(f"  ARMA total: {arma_mb:.1f} MB  (12 files)")
    print(f"  Time: {timings['total']:.1f}s total")
    print("=" * 60, flush=True)

    timing_path = OUT_DIR / f"timing_sem_arma_{var}.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"  Timing log: {timing_path}", flush=True)


if __name__ == "__main__":
    main()
