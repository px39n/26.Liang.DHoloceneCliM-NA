"""Step 3-cal: Calibration-period monthly prediction (det + PI + realization).

Predicts on cal-period ESM (1875-1999) at monthly resolution.
For validation against GHCN observations (Table 1-2, Fig 4, etc.).

With --real flag: loads per-month SEM+ARMA models and generates a proper
spatially-correlated noise realization (matching run_project.py logic).
Without --real: value_real = det (placeholder).

Input:
  pcr_models_{var}.pkl  — from run_calibrate.py
  split_calibration.pkl — from generate_split.py
  sem_model_{var}_m{01-12}.pkl + arma_model_{var}_m{01-12}.pkl  (if --real)

Output:
  recon_cal_{var}.csv   — CSV long-format, monthly
      columns: station_id, lon, lat, year, month, value, pi_lo, pi_hi,
               value_real, split

Run:
  python related_scripts/run_cal_predict.py --var tas
  python related_scripts/run_cal_predict.py --var tas --real
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

from caz.io.trace import load_trace_var, select_na_window, trace_time_to_year_month
from caz.pcr import predict_month
from caz.sem import simulate_arma_noise, inverse_sem

MODEL_DIR = Path(r"D:\Dataset\DPastCliM-NA\interim\trace21k\models")
OUT_DIR   = Path(r"D:\Dataset\DPastCliM-NA\interim\trace21k\station_cal")
TRACE = {
    "tas": Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.TREFHT.nc"),
    "pr":  Path(r"D:\Dataset\DPastCliM-NA\TraCE21k\TraCE-21K-II.monthly.PRECT.nc"),
}

YEAR_CAL_MIN = 1875
YEAR_CAL_MAX = 1999


def main():
    ap = argparse.ArgumentParser(description="Step 3-cal: Cal-period monthly prediction")
    ap.add_argument("--var", choices=["tas", "pr"], required=True)
    ap.add_argument("--real", action="store_true",
                    help="Generate realization with SEM+ARMA noise (requires per-month models)")
    args = ap.parse_args()
    var = args.var
    want_real = args.real

    timings = {}
    print("=" * 60)
    print(f"Step 3-cal: Cal-Period Monthly Prediction — {var}")
    print("=" * 60)
    t_total = _t.time()

    # ── Load models + split ────────────────────────────────────
    t0 = _t.time()
    print(f"\n[1/4] Loading models + split ...", flush=True)
    with open(MODEL_DIR / f"pcr_models_{var}.pkl", "rb") as f:
        pcr_models = pickle.load(f)
    with open(MODEL_DIR / "split_calibration.pkl", "rb") as f:
        split_data = pickle.load(f)[var]
    timings["load_models"] = _t.time() - t0
    print(f"       PCR: {len(pcr_models)} months  ({timings['load_models']:.1f}s)", flush=True)

    cal_years_set = set(split_data["cal_years"])
    val_years_set = set(split_data["val_years"])
    test_years_set = set(split_data["test_years"])
    station_flags = split_data["station_flags"]

    # ── Load cal-period ESM only ───────────────────────────────
    t0 = _t.time()
    print(f"\n[2/4] Loading TraCE cal-period ({YEAR_CAL_MIN}-{YEAR_CAL_MAX}) ...", flush=True)
    tr = load_trace_var(TRACE[var], var)
    tr = select_na_window(tr)
    yr, mo = trace_time_to_year_month(tr["time"].values)
    tr = tr.assign_coords(year=("time", yr), month=("time", mo))
    cal_mask = (tr.year >= YEAR_CAL_MIN) & (tr.year <= YEAR_CAL_MAX)
    tr_cal = tr.isel(time=cal_mask.values).load()
    yr_cal = tr_cal.year.values
    mo_cal = tr_cal.month.values
    timings["load_esm"] = _t.time() - t0
    mem_mb = tr_cal.values.nbytes / 1e6
    print(f"       shape: {tr_cal.shape}  RAM: {mem_mb:.0f} MB  "
          f"({timings['load_esm']:.1f}s)", flush=True)

    # ── Determine station union ────────────────────────────────
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

    T_cal = len(yr_cal)
    print(f"       {n_sta} stations × {T_cal} timesteps", flush=True)

    # ── Allocate ───────────────────────────────────────────────
    val_full = np.full((T_cal, n_sta), np.nan, dtype=np.float32)
    pi_lo_full = np.full((T_cal, n_sta), np.nan, dtype=np.float32)
    pi_hi_full = np.full((T_cal, n_sta), np.nan, dtype=np.float32)
    real_full = np.full((T_cal, n_sta), np.nan, dtype=np.float32) if want_real else None

    if want_real:
        sem_m1 = MODEL_DIR / f"sem_model_{var}_m01.pkl"
        if not sem_m1.exists():
            print(f"  ERROR: SEM per-month files not found, falling back to det", flush=True)
            want_real = False
        else:
            print(f"       SEM+ARMA per-month files detected", flush=True)

    # ── Predict per month ──────────────────────────────────────
    print(f"\n[3/4] Predicting 12 months ...", flush=True)
    t_proj = _t.time()

    for month in tqdm(sorted(pcr_models.keys()), desc="Months", unit="month"):
        t0 = _t.time()
        mdl = pcr_models[month]
        m_mask = mo_cal == month
        esm_m = tr_cal.isel(time=m_mask)
        esm_m_cal = esm_m

        pred = predict_month(mdl, esm_m, esm_da_cal=esm_m_cal)
        pred_vals = pred.values
        if var == "pr":
            pred_vals = np.maximum(pred_vals, 0.0)
        pred_sids = pred.station.values

        col_map = np.array([sid_to_idx[s] for s in pred_sids])
        t_idx = np.where(m_mask)[0]

        sigma = np.sqrt(mdl.sigma2_hat).astype(np.float32)
        if var == "tas":
            pi_lo_m = pred_vals - 1.96 * sigma[None, :]
            pi_hi_m = pred_vals + 1.96 * sigma[None, :]
        else:
            O_t = mdl.O_t.astype(np.float32)
            sig2 = mdl.sigma2_hat.astype(np.float32)
            base = pred_vals + O_t[None, :]
            base = np.maximum(base, 1e-10)
            pi_lo_m = np.maximum(
                base * np.exp(-sig2[None, :] / 2 - 1.96 * sigma[None, :]) - O_t[None, :],
                0.0).astype(np.float32)
            pi_hi_m = np.maximum(
                base * np.exp(-sig2[None, :] / 2 + 1.96 * sigma[None, :]) - O_t[None, :],
                0.0).astype(np.float32)

        val_full[np.ix_(t_idx, col_map)] = pred_vals.astype(np.float32)
        pi_lo_full[np.ix_(t_idx, col_map)] = pi_lo_m
        pi_hi_full[np.ix_(t_idx, col_map)] = pi_hi_m

        # Realization: det + SEM+ARMA noise
        if want_real:
            with open(MODEL_DIR / f"sem_model_{var}_m{month:02d}.pkl", "rb") as _f:
                sem_r = pickle.load(_f)
            with open(MODEL_DIR / f"arma_model_{var}_m{month:02d}.pkl", "rb") as _f:
                arma_m = pickle.load(_f)
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
                O_t_r = mdl.O_t.astype(np.float32)
                u_var = np.var(u_m, axis=0, keepdims=True)
                real_m = (pred_vals + O_t_r[None, :]) * np.exp(u_m - u_var / 2) - O_t_r[None, :]
                real_m = np.maximum(real_m, 0.0)
            else:
                real_m = pred_vals + u_m
            real_full[np.ix_(t_idx, col_map)] = real_m.astype(np.float32)
            del sem_r, arma_m, eps, u, u_m, real_m

        dt = _t.time() - t0
        real_tag = " +real" if want_real else ""
        tqdm.write(f"  month {month:2d}: ({pred_vals.shape[0]}, {pred_vals.shape[1]})  "
                   f"sigma [{sigma.min():.3f}, {sigma.max():.3f}]{real_tag}  ({dt:.1f}s)")

    timings["proj_total"] = _t.time() - t_proj
    print(f"       Prediction done ({timings['proj_total']:.1f}s)", flush=True)

    # ── Build CSV ──────────────────────────────────────────────
    print(f"\n[4/4] Building CSV ...", flush=True)
    t0 = _t.time()

    time_rep_yr = np.repeat(yr_cal, n_sta)
    time_rep_mo = np.repeat(mo_cal, n_sta)
    sta_rep = np.tile(station_ids, T_cal)
    lon_rep = np.tile(meta_lon, T_cal)
    lat_rep = np.tile(meta_lat, T_cal)

    # Assign split label per (station, year)
    def _split_label(sid, yr_val):
        flag = station_flags.get(sid, "removed")
        if flag == "removed":
            return "removed"
        if flag == "test_only":
            return "test"
        yr_val = int(yr_val)
        if yr_val in cal_years_set:
            return "cal"
        if yr_val in val_years_set:
            return "val"
        if yr_val in test_years_set:
            return "test"
        return "cal"

    split_arr = np.array([_split_label(s, y) for s, y in zip(sta_rep, time_rep_yr)])

    df = pd.DataFrame({
        "station_id": sta_rep,
        "lon": lon_rep,
        "lat": lat_rep,
        "year": time_rep_yr,
        "month": time_rep_mo,
        "value": val_full.ravel(),
        "pi_lo": pi_lo_full.ravel(),
        "pi_hi": pi_hi_full.ravel(),
        "value_real": real_full.ravel() if want_real else val_full.ravel(),
        "split": split_arr,
    })

    # Drop rows where value is NaN (station not in this month's model)
    n_before = len(df)
    df = df.dropna(subset=["value"])
    n_after = len(df)

    csv_path = OUT_DIR / f"recon_cal_{var}.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    csv_mb = csv_path.stat().st_size / 1e6
    timings["save"] = _t.time() - t0
    timings["total"] = _t.time() - t_total

    print("\n" + "=" * 60)
    print(f"DONE: {var} cal-period monthly prediction")
    print(f"  Output: {csv_path}  ({csv_mb:.0f} MB, {n_after:,} rows)")
    print(f"  Dropped {n_before - n_after:,} NaN rows")
    print(f"  Time:   {timings['total']:.1f}s total")
    print(f"          Models: {timings['load_models']:.1f}s  "
          f"ESM: {timings['load_esm']:.1f}s  "
          f"Proj: {timings['proj_total']:.1f}s  "
          f"Save: {timings['save']:.1f}s")
    print("=" * 60, flush=True)

    timing_path = OUT_DIR / f"timing_cal_predict_{var}.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)


if __name__ == "__main__":
    main()
