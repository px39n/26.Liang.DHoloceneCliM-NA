"""Step 3: Station-level projection with prediction intervals.

Modes:
  --cal-only  Cal-period prediction (monthly, det+PI+realization)
  --predict   3 showcase windows (yearly, det+PI only, no realization)
  (default)   Full 22ka (monthly, det+PI+realization)

Run:
  python related_scripts/run_project.py --var tas --cal-only                # cal period (trace21k)
  python related_scripts/run_project.py --var tas --cal-only --gcm mpi-esm-cr
  python related_scripts/run_project.py --var tas --predict
  python related_scripts/run_project.py --var tas              # full 22ka
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
import xarray as xr
from tqdm import tqdm

from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month
from caz.io.mpi_esm import load_mpi_esm_var, mpi_esm_time_to_year_month
from caz.pcr import predict_month
from caz.sem import simulate_arma_noise, inverse_sem

DATA_ROOT = Path(r"D:\Dataset\DPastCliM-NA")

YEAR_CAL_MIN = 1875

GCM_CONFIG = {
    "trace21k": {
        "year_cal_max": 1999,
        "esm_paths": {
            "tas": DATA_ROOT / "TraCE21k" / "TraCE-21K-II.monthly.TREFHT.nc",
            "pr":  DATA_ROOT / "TraCE21k" / "TraCE-21K-II.monthly.PRECT.nc",
        },
    },
    "mpi-esm-cr": {
        "year_cal_max": 1949,
        "esm_dir": DATA_ROOT / "MPI-ESM-CR",
    },
}


def _load_esm_full(gcm: str, var: str):
    """Load full ESM transient, return (DataArray, year_arr, month_arr)."""
    if gcm == "trace21k":
        cfg = GCM_CONFIG[gcm]
        tr = load_trace_var(cfg["esm_paths"][var], var)
        tr = select_na_window(tr)
        yr, mo = trace_time_to_year_month(tr["time"].values)
        tr = tr.assign_coords(year=("time", yr), month=("time", mo))
        return tr.load(), yr, mo
    elif gcm == "mpi-esm-cr":
        cfg = GCM_CONFIG[gcm]
        da = load_mpi_esm_var(cfg["esm_dir"] / var, var)
        da = select_na_window(da)
        yr, mo = mpi_esm_time_to_year_month(da["time"].values)
        da = da.assign_coords(year=("time", yr), month=("time", mo))
        return da.load(), yr, mo
    else:
        raise ValueError(f"Unknown GCM: {gcm}")


def _load_esm_cal(gcm: str, var: str):
    """Load only cal-period ESM, return (DataArray, year_arr, month_arr)."""
    cfg = GCM_CONFIG[gcm]
    year_max = cfg["year_cal_max"]
    if gcm == "trace21k":
        tr = load_trace_var(cfg["esm_paths"][var], var)
        tr = select_na_window(tr)
        yr, mo = trace_time_to_year_month(tr["time"].values)
        tr = tr.assign_coords(year=("time", yr), month=("time", mo))
        mask = (tr.year >= YEAR_CAL_MIN) & (tr.year <= year_max)
        tr = tr.isel(time=mask.values).load()
        yr2, mo2 = trace_time_to_year_month(tr["time"].values)
        return tr, yr2, mo2
    elif gcm == "mpi-esm-cr":
        da = load_mpi_esm_var(GCM_CONFIG[gcm]["esm_dir"] / var, var,
                              year_min_ce=YEAR_CAL_MIN, year_max_ce=year_max)
        da = select_na_window(da)
        yr, mo = mpi_esm_time_to_year_month(da["time"].values)
        da = da.assign_coords(year=("time", yr), month=("time", mo))
        mask = (da.year >= YEAR_CAL_MIN) & (da.year <= year_max)
        return da.isel(time=mask.values).load(), yr[mask.values], mo[mask.values]
    else:
        raise ValueError(f"Unknown GCM: {gcm}")

PREDICT_WINDOWS = {
    "lgm":    (-20000, -19000),
    "midhol": (-5000,  -4000),
    "recent": (990,     1990),
}


def run_window(pcr_models, tr_load, yr_full, mo_full, tr_cal,
               var, has_pi, has_real, sem_per_month, sem_results, arma_results,
               year_min, year_max, yearly, out_path, model_dir=None):
    """Project a single time window, optionally aggregate to yearly, save CSV."""
    mask = (yr_full >= year_min) & (yr_full <= year_max)
    yr_w = yr_full[mask]
    mo_w = mo_full[mask]
    tr_w = tr_load.isel(time=mask)

    all_sids = set()
    for mdl in pcr_models.values():
        all_sids.update(mdl.station_id)
    station_ids = np.array(sorted(all_sids))
    n_sta = len(station_ids)
    sid_to_idx = {s: i for i, s in enumerate(station_ids)}

    meta_lon = np.full(n_sta, np.nan, dtype=np.float32)
    meta_lat = np.full(n_sta, np.nan, dtype=np.float32)
    for mdl in pcr_models.values():
        for j, sid in enumerate(mdl.station_id):
            idx = sid_to_idx.get(sid, -1)
            if idx >= 0:
                meta_lon[idx] = mdl.station_lon[j]
                meta_lat[idx] = mdl.station_lat[j]

    T_w = len(yr_w)
    val_full = np.full((T_w, n_sta), np.nan, dtype=np.float32)
    pi_lo_full = np.full((T_w, n_sta), np.nan, dtype=np.float32) if has_pi else None
    pi_hi_full = np.full((T_w, n_sta), np.nan, dtype=np.float32) if has_pi else None
    real_full = np.full((T_w, n_sta), np.nan, dtype=np.float32) if has_real else None

    for month in tqdm(sorted(pcr_models.keys()), desc=f"  [{year_min},{year_max}]", unit="m"):
        mdl = pcr_models[month]
        esm_m = tr_w.isel(time=mo_w == month)
        esm_m_cal = tr_cal.isel(time=tr_cal.month.values == month)

        pred = predict_month(mdl, esm_m, esm_da_cal=esm_m_cal)
        pred_vals = pred.values
        if var == "pr":
            pred_vals = np.maximum(pred_vals, 0.0)
        pred_sids = pred.station.values
        col_map = np.array([sid_to_idx[s] for s in pred_sids])
        t_idx = np.where(mo_w == month)[0]

        sigma = np.sqrt(mdl.sigma2_hat).astype(np.float32)
        if has_pi:
            if var == "tas":
                pi_lo_m = pred_vals - 1.96 * sigma[None, :]
                pi_hi_m = pred_vals + 1.96 * sigma[None, :]
            else:
                O_t = mdl.O_t.astype(np.float32)
                sig2 = mdl.sigma2_hat.astype(np.float32)
                base = np.maximum(pred_vals + O_t[None, :], 1e-10)
                pi_lo_m = np.maximum(
                    base * np.exp(-sig2[None, :] / 2 - 1.96 * sigma[None, :]) - O_t[None, :], 0.0).astype(np.float32)
                pi_hi_m = np.maximum(
                    base * np.exp(-sig2[None, :] / 2 + 1.96 * sigma[None, :]) - O_t[None, :], 0.0).astype(np.float32)

        if has_real:
            if sem_per_month:
                with open(model_dir / f"sem_model_{var}_m{month:02d}.pkl", "rb") as _f:
                    sem_r = pickle.load(_f)
                with open(model_dir / f"arma_model_{var}_m{month:02d}.pkl", "rb") as _f:
                    arma_m = pickle.load(_f)
            else:
                sem_r = sem_results[month]
                arma_m = arma_results[month]
            T_m = pred_vals.shape[0]
            rng = np.random.default_rng(812 + month)
            eps = simulate_arma_noise(arma_m, sem_r["valid_mask"], T_m, rng=rng)
            u = inverse_sem(eps, sem_r["W"], sem_r["lambda"])
            u_m = np.zeros_like(pred_vals)
            valid_sids = mdl.station_id[sem_r["valid_mask"]]
            month_sid_to_j = {s: j for j, s in enumerate(pred_sids)}
            for k, sid in enumerate(valid_sids):
                jj = month_sid_to_j.get(sid, -1)
                if jj >= 0:
                    u_m[:, jj] = u[k, :].astype(np.float32)
            if var == "pr":
                O_t = mdl.O_t.astype(np.float32)
                u_var = np.var(u_m, axis=0, keepdims=True)
                real_m = (pred_vals + O_t[None, :]) * np.exp(u_m - u_var / 2) - O_t[None, :]
                real_m = np.maximum(real_m, 0.0)
            else:
                real_m = pred_vals + u_m

        val_full[np.ix_(t_idx, col_map)] = pred_vals.astype(np.float32)
        if has_pi:
            pi_lo_full[np.ix_(t_idx, col_map)] = pi_lo_m
            pi_hi_full[np.ix_(t_idx, col_map)] = pi_hi_m
        if has_real:
            real_full[np.ix_(t_idx, col_map)] = real_m.astype(np.float32)

    # Aggregate to yearly if requested
    if yearly:
        unique_years = np.unique(yr_w)
        n_yr = len(unique_years)
        val_yr = np.full((n_yr, n_sta), np.nan, dtype=np.float32)
        pi_lo_yr = np.full((n_yr, n_sta), np.nan, dtype=np.float32) if has_pi else None
        pi_hi_yr = np.full((n_yr, n_sta), np.nan, dtype=np.float32) if has_pi else None
        real_yr = np.full((n_yr, n_sta), np.nan, dtype=np.float32) if has_real else None
        for i, y in enumerate(unique_years):
            ymask = yr_w == y
            val_yr[i] = np.nanmean(val_full[ymask], axis=0)
            if has_pi:
                pi_lo_yr[i] = np.nanmean(pi_lo_full[ymask], axis=0)
                pi_hi_yr[i] = np.nanmean(pi_hi_full[ymask], axis=0)
            if has_real:
                real_yr[i] = np.nanmean(real_full[ymask], axis=0)
        time_out = unique_years
        val_out, pi_lo_out, pi_hi_out, real_out = val_yr, pi_lo_yr, pi_hi_yr, real_yr
    else:
        time_out = yr_w
        month_out = mo_w
        val_out = val_full
        pi_lo_out, pi_hi_out, real_out = pi_lo_full, pi_hi_full, real_full

    # Build and save CSV
    n_time = len(time_out)
    df_dict = {
        "station_id": np.tile(station_ids, n_time),
        "lon": np.tile(meta_lon, n_time),
        "lat": np.tile(meta_lat, n_time),
        "year": np.repeat(time_out, n_sta),
    }
    if not yearly:
        df_dict["month"] = np.repeat(month_out, n_sta)
    df_dict["value"] = val_out.ravel()
    if has_real and real_out is not None:
        df_dict["value_real"] = real_out.ravel()
    if has_pi:
        df_dict["pi_lo"] = pi_lo_out.ravel()
        df_dict["pi_hi"] = pi_hi_out.ravel()

    df = pd.DataFrame(df_dict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path}  ({mb:.0f} MB, {n_time * n_sta:,} rows)")
    return df


def main():
    ap = argparse.ArgumentParser(description="Step 3: Station-level projection")
    ap.add_argument("--var", choices=["tas", "pr"], required=True)
    ap.add_argument("--gcm", choices=list(GCM_CONFIG.keys()), default="trace21k")
    ap.add_argument("--cal-only", action="store_true",
                    help="Cal-period prediction only (monthly, det+PI+real)")
    ap.add_argument("--predict", action="store_true",
                    help="Predict mode: 3 showcase windows, yearly, det+PI only")
    ap.add_argument("--det-only", action="store_true",
                    help="Skip PI and realization, only deterministic predictions")
    ap.add_argument("--no-real", action="store_true",
                    help="Skip realization (SEM+ARMA noise) but still compute PI")
    ap.add_argument("--monthly", action="store_true", default=True,
                    help="Output monthly resolution (default: True)")
    args = ap.parse_args()
    var = args.var
    gcm = args.gcm
    cfg = GCM_CONFIG[gcm]

    if args.predict:
        args.no_real = True

    MODEL_DIR = DATA_ROOT / "interim" / gcm / "models"
    STATION_CAL_DIR = DATA_ROOT / "interim" / gcm / "station_cal"
    OUT_FULL = DATA_ROOT / "output" / gcm / "full"
    OUT_PRED = DATA_ROOT / "output" / gcm / "predict"

    if args.cal_only:
        mode = "cal"
        OUT_DIR = STATION_CAL_DIR
    elif args.predict:
        mode = "predict"
        OUT_DIR = OUT_PRED
    else:
        mode = "full"
        OUT_DIR = OUT_FULL
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timings = {}
    print("=" * 60)
    print(f"Step 3: {mode} projection — {var} [{gcm}]")
    if args.cal_only:
        print(f"  Cal period: {YEAR_CAL_MIN}-{cfg['year_cal_max']}")
    print("=" * 60)
    t_total = _t.time()

    # ── Load models ───────────────────────────────────────────
    t0 = _t.time()
    print(f"\n[1] Loading models ...", flush=True)
    with open(MODEL_DIR / f"pcr_models_{var}.pkl", "rb") as f:
        pcr_models = pickle.load(f)
    print(f"    PCR: {len(pcr_models)} months", flush=True)

    sem_results = arma_results = None
    want_real = not args.det_only and not args.no_real
    sem_per_month = False
    if want_real:
        sem_m1 = MODEL_DIR / f"sem_model_{var}_m01.pkl"
        if sem_m1.exists():
            sem_per_month = True
            print(f"    SEM + ARMA: per-month files detected (lazy load)", flush=True)
        else:
            sem_path = MODEL_DIR / f"sem_models_{var}.pkl"
            arma_path = MODEL_DIR / f"arma_models_{var}.pkl"
            if sem_path.exists() and arma_path.exists():
                with open(sem_path, "rb") as f:
                    sem_results = pickle.load(f)
                with open(arma_path, "rb") as f:
                    arma_results = pickle.load(f)
                print(f"    SEM + ARMA: loaded (for realization)", flush=True)
            else:
                print(f"    WARNING: SEM/ARMA not found, skipping realization", flush=True)

    timings["load_models"] = _t.time() - t0

    # ── Load ESM ──────────────────────────────────────────────
    t0 = _t.time()
    if args.cal_only:
        print(f"\n[2] Loading {gcm} cal-period ESM ...", flush=True)
        tr_load, yr_full, mo_full = _load_esm_cal(gcm, var)
        tr_cal = tr_load
    else:
        print(f"\n[2] Loading {gcm} full transient (NA window) ...", flush=True)
        tr_load, yr_full, mo_full = _load_esm_full(gcm, var)
        cal_mask = (tr_load.year >= YEAR_CAL_MIN) & (tr_load.year <= cfg["year_cal_max"])
        tr_cal = tr_load.isel(time=cal_mask.values)
    timings["load_esm"] = _t.time() - t0
    print(f"    shape: {tr_load.shape}  ({timings['load_esm']:.1f}s)", flush=True)

    has_pi = not args.det_only
    has_real = (not args.det_only and not args.no_real
                and (sem_results is not None or sem_per_month))

    # ── Dispatch by mode ──────────────────────────────────────
    if args.cal_only:
        print(f"\n[3] Cal-only mode: {YEAR_CAL_MIN}-{cfg['year_cal_max']} monthly", flush=True)
        out_path = OUT_DIR / f"recon_cal_{var}.csv"
        run_window(pcr_models, tr_load, yr_full, mo_full, tr_cal,
                   var, has_pi, has_real, sem_per_month, sem_results, arma_results,
                   YEAR_CAL_MIN, cfg["year_cal_max"], yearly=False, out_path=out_path,
                   model_dir=MODEL_DIR)
    elif args.predict:
        print(f"\n[3] Predict mode: {len(PREDICT_WINDOWS)} windows (yearly, det+PI)", flush=True)
        for wname, (y0, y1) in PREDICT_WINDOWS.items():
            t_w = _t.time()
            out_path = OUT_DIR / f"recon_station_{var}_{wname}.csv"
            print(f"\n--- Window '{wname}': {y0} to {y1} ---")
            run_window(pcr_models, tr_load, yr_full, mo_full, tr_cal,
                       var, has_pi, has_real, sem_per_month, sem_results, arma_results,
                       y0, y1, yearly=True, out_path=out_path, model_dir=MODEL_DIR)
            timings[f"window_{wname}"] = _t.time() - t_w
    else:
        print(f"\n[3] Full mode: monthly projection", flush=True)
        out_path = OUT_DIR / f"recon_station_{var}.csv"
        run_window(pcr_models, tr_load, yr_full, mo_full, tr_cal,
                   var, has_pi, has_real, sem_per_month, sem_results, arma_results,
                   yr_full.min(), yr_full.max(), yearly=False, out_path=out_path,
                   model_dir=MODEL_DIR)

    timings["total"] = _t.time() - t_total
    print("\n" + "=" * 60)
    print(f"DONE: {var} {mode} projection [{gcm}]  ({timings['total']:.1f}s total)")
    print("=" * 60, flush=True)

    timing_path = OUT_DIR / f"timing_project_{var}.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)


if __name__ == "__main__":
    main()
