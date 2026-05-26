"""Principal-Component Regression (PCR) for paleoclimate downscaling.

Follows Guaita et al. (2024) ``PCR_calibration_v5.m`` exactly:

    1. 3-way random split: train / validation / test (by year counts).
    2. PCA on calibration ESM field.
    3. Forward-stepwise PC selection using validation RMSE (>1% improvement).
    4. Per-station OLS with per-station pr transform ``log(x + O_t)``.
    5. Prediction with time-varying delta-change (30-step moving mean).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nearest_interp_ts(
    field: np.ndarray,           # (T, ny, nx)
    lat_grid: np.ndarray,        # (ny,) ascending
    lon_grid: np.ndarray,        # (nx,) ascending
    pts_lat: np.ndarray,         # (S,)
    pts_lon: np.ndarray,         # (S,)
) -> np.ndarray:
    """Nearest-neighbor interp of a (T,ny,nx) field at S station points -> (T,S).

    Matches Guaita's griddedInterpolant(..., 'nearest') behaviour.
    """
    iy = np.array([np.argmin(np.abs(lat_grid - la)) for la in pts_lat])
    ix = np.array([np.argmin(np.abs(lon_grid - lo)) for lo in pts_lon])
    return field[:, iy, ix].astype(np.float32)


def _pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based PCA on samples-by-features matrix X.

    Returns
    -------
    scores      (n_samples, n_pc)  PC scores after centring
    components  (n_pc, n_features) EOF rows (right singular vectors)
    var_frac    (n_pc,)            fraction of variance explained
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    scores = U * s
    var = (s ** 2) / max(n - 1, 1)
    var_frac = var / var.sum() if var.sum() > 0 else var
    return scores.astype(np.float32), Vt.astype(np.float32), var_frac.astype(np.float32), mu.astype(np.float32)


def _ols(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-station OLS fit.  Y (T,S), X (T,k+1) with leading intercept col.

    Returns
    -------
    beta (k+1, S)  coefficients
    yhat (T, S)
    resid (T, S)
    rmse (S,)
    """
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    yhat = X @ beta
    resid = Y - yhat
    rmse = np.sqrt(np.nanmean(resid ** 2, axis=0))
    return beta.astype(np.float32), yhat.astype(np.float32), resid.astype(np.float32), rmse.astype(np.float32)


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------

@dataclass
class MonthPCRModel:
    month: int
    pc_indices: np.ndarray        # selected PC indices (0-based), e.g. [0,1,2,4]
    eofs: np.ndarray              # (n_all_pc, ny*nx) all EOFs from PCA
    pc_var_frac: np.ndarray       # (n_all_pc,) all PC variance fractions
    field_mean: np.ndarray        # (ny*nx,) mu_M global field-mean used for PCA centering
    station_id: np.ndarray        # (S,)
    station_lat: np.ndarray       # (S,)
    station_lon: np.ndarray       # (S,)
    beta: np.ndarray              # (n_selected+1, S) intercept first
    rmse_train: np.ndarray        # (S,)
    rmse_val: np.ndarray | None = None
    rmse_test: np.ndarray | None = None
    r2_train: np.ndarray | None = None
    grid_lat: np.ndarray | None = None
    grid_lon: np.ndarray | None = None
    esm_cal_mean_at_station: np.ndarray | None = None
    var_name: str = "tas"
    O_t: np.ndarray | None = None       # (S,) per-station pr offset
    mu_gO: np.ndarray | None = None     # (S,) per-station mean of log-transformed obs
    sigma2_hat: np.ndarray | None = None # (S,) residual variance per station
    idx_cal: np.ndarray | None = None    # indices into common time axis: training years
    idx_val: np.ndarray | None = None    # indices into common time axis: validation years
    idx_test: np.ndarray | None = None   # indices into common time axis: test years
    common_years: np.ndarray | None = None  # (T_common,) actual years for index mapping

    @property
    def n_pc(self) -> int:
        return len(self.pc_indices)


@dataclass
class PCRPipeline:
    """Bundle of 12 monthly models keyed by month-of-year."""
    models: dict[int, MonthPCRModel] = field(default_factory=dict)
    grid_shape: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# core API
# ---------------------------------------------------------------------------

def project_pcs(
    field: np.ndarray,        # (T, ny*nx)
    eofs: np.ndarray,         # (n_pc, ny*nx) or (n_all, ny*nx)
    field_mean: np.ndarray,   # (ny*nx,)
    pc_indices: np.ndarray | None = None,  # if given, select these from eofs
) -> np.ndarray:
    """Project a centred field onto trained EOFs."""
    if pc_indices is not None:
        eofs = eofs[pc_indices]
    return ((field - field_mean) @ eofs.T).astype(np.float32)


def _forward_stepwise_select(
    pcs_cal: np.ndarray,       # (T_cal, n_all)
    pcs_val: np.ndarray,       # (T_val, n_all)
    Y_cal: np.ndarray,         # (T_cal, S)
    Y_val: np.ndarray,         # (T_val, S)
    var_frac: np.ndarray,      # (n_all,)
    frac_threshold: float = 0.001,
    improvement: float = 0.99,
    max_outer: int = 16,
    max_candidate: int = 20,
) -> np.ndarray:
    """Forward-stepwise PC selection matching Guaita's PCR_calibration_v5.m.

    Returns array of selected PC indices (0-based).
    """
    n_pc_min = 0
    for k in range(len(var_frac)):
        if var_frac[k] * 100 > 10:
            n_pc_min = k + 1
    if n_pc_min == 0:
        n_pc_min = 1

    selected = list(range(n_pc_min))

    def _val_rmse(pc_set):
        X_c = np.column_stack([np.ones(pcs_cal.shape[0], dtype=np.float32),
                               pcs_cal[:, pc_set]])
        X_v = np.column_stack([np.ones(pcs_val.shape[0], dtype=np.float32),
                               pcs_val[:, pc_set]])
        rmses = np.full(Y_cal.shape[1], np.nan, dtype=np.float32)
        for s in range(Y_cal.shape[1]):
            m_c = np.isfinite(Y_cal[:, s])
            m_v = np.isfinite(Y_val[:, s])
            if m_c.sum() < 20 or m_v.sum() < 5:
                continue
            b, *_ = np.linalg.lstsq(X_c[m_c], Y_cal[m_c, s], rcond=None)
            pred = X_v[m_v] @ b
            rmses[s] = float(np.sqrt(np.nanmean((pred - Y_val[m_v, s]) ** 2)))
        return float(np.nanmean(rmses))

    best_rmse = _val_rmse(selected)
    pool = list(range(n_pc_min, min(n_pc_min + max_candidate, len(var_frac))))

    for _ in range(min(max_outer, len(pool))):
        improved = False
        best_candidate = None
        for pc in [c for c in pool if c not in selected]:
            if var_frac[pc] * 100 < frac_threshold:
                continue
            trial = selected + [pc]
            rmse_trial = _val_rmse(trial)
            if rmse_trial < improvement * best_rmse:
                best_rmse = rmse_trial
                best_candidate = pc
                improved = True
        if improved and best_candidate is not None:
            selected.append(best_candidate)
        else:
            break

    return np.array(selected, dtype=np.int32)


def calibrate_month(
    month: int,
    esm_da: xr.DataArray,            # (T, ny, nx) ESM field for this month, calib period
    obs_long: "pd.DataFrame",        # cols: ID, year, value, lat, lon
    meta_df: "pd.DataFrame" | None = None,
    var_name: str = "tas",
    n_year_val: int = 50,
    n_year_test: int = 15,
    rng: np.random.Generator | None = None,
    split_indices: tuple | None = None,  # (idx_cal, idx_val, idx_test) from generate_split
) -> MonthPCRModel:
    """Calibrate one month's PCR model following Guaita's methodology.

    3-way split (train/val/test by year), forward-stepwise PC selection,
    per-station pr transform.

    If split_indices is provided, uses the pre-computed global split.
    Otherwise falls back to internal rng.permutation (legacy behavior).
    """
    import pandas as pd

    if rng is None:
        rng = np.random.default_rng(2026)

    # ------------------------------------------------------------------ ESM PCA
    arr = esm_da.values                  # (T, ny, nx)
    T, ny, nx = arr.shape
    field = arr.reshape(T, ny * nx).astype(np.float32)
    scores_all, components_all, var_frac_all, mu = _pca(field)

    # ------------------------------------------------------------------ stations
    obs_m = obs_long[obs_long["month"] == month][["ID", "year", "value"]].copy()
    if obs_m.empty:
        raise ValueError(f"no observations for month {month}")

    station_years = obs_m.pivot_table(
        index="year", columns="ID", values="value", aggfunc="first"
    )
    esm_years = pd.Index(esm_da["year"].values, name="year")
    common_years = station_years.index.intersection(esm_years)
    if len(common_years) < 30:
        raise ValueError(
            f"month {month}: only {len(common_years)} overlap years"
        )

    station_years = station_years.loc[common_years]
    pc_idx = esm_years.get_indexer(common_years)

    Y_raw = station_years.values.astype(np.float32)  # (T_common, S)
    valid_per_station = np.isfinite(Y_raw).sum(axis=0)
    min_valid = max(10, int(len(common_years) * 0.15))
    keep = valid_per_station >= min_valid
    Y_raw = Y_raw[:, keep]
    station_id = station_years.columns.values[keep]
    S = Y_raw.shape[1]
    if S == 0:
        raise ValueError(f"month {month}: no station with >={min_valid} valid years")

    # ------------------------------------------------------------------ pr transform
    O_t = None
    mu_gO = None
    if var_name == "pr":
        O_t = np.zeros(S, dtype=np.float32)
        mu_gO = np.zeros(S, dtype=np.float32)
        Y = np.full_like(Y_raw, np.nan)
        for s in range(S):
            vals = Y_raw[:, s]
            valid = np.isfinite(vals)
            O_t[s] = 1.0 + float(np.nanmin(vals))
            Y[:, s] = np.where(valid, np.log(vals + O_t[s]), np.nan)
            mu_gO[s] = float(np.nanmean(Y[valid, s]))
        Y = Y - mu_gO[None, :]
    else:
        Y = Y_raw.copy()
        mu_gO_dummy = np.nanmean(Y, axis=0)
        Y = Y - mu_gO_dummy[None, :]
        mu_gO = mu_gO_dummy

    # ------------------------------------------------------------------ 3-way split
    T_common = Y.shape[0]
    if split_indices is not None:
        idx_cal, idx_val, idx_test = split_indices
    else:
        n_val = min(n_year_val, T_common // 2)
        n_test = min(n_year_test, (T_common - n_val) // 3)
        perm = rng.permutation(T_common)
        idx_val = np.sort(perm[:n_val])
        idx_test = np.sort(perm[n_val:n_val + n_test])
        idx_cal = np.sort(perm[n_val + n_test:])

    pcs_all = scores_all[pc_idx]   # (T_common, n_all)
    pcs_cal = pcs_all[idx_cal]
    pcs_val = pcs_all[idx_val]
    pcs_test = pcs_all[idx_test] if len(idx_test) > 0 else None

    Y_cal = Y[idx_cal]
    Y_val = Y[idx_val]
    Y_test = Y[idx_test] if len(idx_test) > 0 else None

    # ------------------------------------------------------------------ stepwise PC select
    max_pcs = min(scores_all.shape[1], T_common)
    selected = _forward_stepwise_select(
        pcs_cal[:, :max_pcs], pcs_val[:, :max_pcs],
        Y_cal, Y_val, var_frac_all[:max_pcs],
    )

    # ------------------------------------------------------------------ final OLS
    X_cal = np.column_stack([np.ones(pcs_cal.shape[0], dtype=np.float32),
                             pcs_cal[:, selected]])
    n_sel = len(selected)
    beta = np.full((n_sel + 1, S), np.nan, dtype=np.float32)
    rmse_tr = np.full(S, np.nan, dtype=np.float32)
    rmse_val = np.full(S, np.nan, dtype=np.float32)
    rmse_te = np.full(S, np.nan, dtype=np.float32)
    r2_tr = np.full(S, np.nan, dtype=np.float32)
    sigma2_hat = np.full(S, np.nan, dtype=np.float32)

    X_val_sel = np.column_stack([np.ones(pcs_val.shape[0], dtype=np.float32),
                                 pcs_val[:, selected]])

    for s in range(S):
        ys = Y_cal[:, s]
        m = np.isfinite(ys)
        if m.sum() < 20:
            continue
        b, yh, res, _ = _ols(ys[m, None], X_cal[m])
        beta[:, s] = b[:, 0]
        rmse_tr[s] = float(np.sqrt(np.mean(res ** 2)))
        sigma2_hat[s] = float(np.mean(res ** 2))
        ss_res = float(np.sum(res ** 2))
        ss_tot = float(np.sum((ys[m] - ys[m].mean()) ** 2))
        r2_tr[s] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Validation RMSE
        ys_v = Y_val[:, s]
        mv = np.isfinite(ys_v)
        if mv.any():
            pred_v = X_val_sel[mv] @ b
            rmse_val[s] = float(np.sqrt(np.mean((pred_v[:, 0] - ys_v[mv]) ** 2)))

        # Test RMSE
        if pcs_test is not None:
            X_te_sel = np.column_stack([np.ones(pcs_test.shape[0], dtype=np.float32),
                                        pcs_test[:, selected]])
            ys_t = Y_test[:, s]
            mt = np.isfinite(ys_t)
            if mt.any():
                pred_t = X_te_sel[mt] @ b
                rmse_te[s] = float(np.sqrt(np.mean((pred_t[:, 0] - ys_t[mt]) ** 2)))

    # ------------------------------------------------------------------ station coords
    obs_meta = (
        obs_long[obs_long["month"] == month]
        .groupby("ID")[["lat", "lon"]].first()
        .reindex(station_id)
    )
    pts_lat = obs_meta["lat"].to_numpy(dtype=np.float32)
    pts_lon = obs_meta["lon"].to_numpy(dtype=np.float32)

    grid_lat = esm_da["lat"].values.astype(np.float32)
    grid_lon = esm_da["lon"].values.astype(np.float32)
    esm_at_st = _nearest_interp_ts(arr.astype(np.float32), grid_lat, grid_lon,
                                    pts_lat, pts_lon)
    esm_cal_mean = np.nanmean(esm_at_st, axis=0).astype(np.float32)

    return MonthPCRModel(
        month=month,
        pc_indices=selected,
        eofs=components_all,
        pc_var_frac=var_frac_all,
        field_mean=mu,
        station_id=np.asarray(station_id, dtype=object),
        station_lat=pts_lat,
        station_lon=pts_lon,
        beta=beta,
        rmse_train=rmse_tr,
        rmse_val=rmse_val,
        rmse_test=rmse_te,
        r2_train=r2_tr,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        esm_cal_mean_at_station=esm_cal_mean,
        var_name=var_name,
        O_t=O_t,
        mu_gO=mu_gO,
        sigma2_hat=sigma2_hat,
        idx_cal=idx_cal,
        idx_val=idx_val,
        idx_test=idx_test if len(idx_test) > 0 else None,
        common_years=np.asarray(common_years, dtype=int),
    )


def predict_month(
    model: MonthPCRModel,
    esm_da: xr.DataArray,         # (T, ny, nx) full transient for this month
    esm_da_cal: xr.DataArray | None = None,  # cal-period ESM for moving-mean trend
    delta_change: bool = True,
    n_mov: int = 30,
) -> xr.DataArray:
    """Project full transient ESM through the trained model -> (time, station).

    Follows Guaita's ds_ESM_mat_v1.m:
    - For tas: pred = regression_anomaly + mu_gO + moving_mean_trend
    - For pr:  pred = exp(regression_anomaly + mu_gO_trend + sigma2/2) - O_t
    """
    arr = esm_da.values
    T, ny, nx = arr.shape
    field = arr.reshape(T, ny * nx).astype(np.float32)
    selected = model.pc_indices
    eofs_sel = model.eofs[selected]

    if delta_change and esm_da_cal is not None:
        cal_field = esm_da_cal.values.reshape(-1, ny * nx).astype(np.float32)
        if model.var_name == "pr":
            M_t = 1.0 + np.min(cal_field, axis=0, keepdims=True)
            mu_cal = np.mean(np.log(cal_field + M_t), axis=0)
            full_log = np.log(field + M_t)
            mu_mov = _movmean_2d(full_log, n_mov)
            mu_adj = mu_mov - mu_cal[None, :]
        else:
            mu_cal = np.mean(cal_field, axis=0)
            mu_mov = _movmean_2d(field, n_mov)
            mu_adj = mu_mov - mu_cal[None, :]

        pcs = ((field - mu_mov) @ eofs_sel.T).astype(np.float32)
    else:
        pcs = ((field - model.field_mean) @ eofs_sel.T).astype(np.float32)
        mu_adj = None

    X = np.column_stack([np.ones(T, dtype=np.float32), pcs])
    SgO_hat = X @ model.beta  # (T, S) regression anomaly

    if model.var_name == "pr" and model.O_t is not None:
        mu_gO_t = model.mu_gO[None, :]
        if mu_adj is not None:
            mu_gO_local = mu_gO_t + _nearest_interp_ts(
                mu_adj.reshape(T, ny, nx).astype(np.float32),
                model.grid_lat, model.grid_lon,
                model.station_lat, model.station_lon,
            )
        else:
            mu_gO_local = np.broadcast_to(mu_gO_t, SgO_hat.shape)
        sig2 = model.sigma2_hat if model.sigma2_hat is not None else np.zeros(SgO_hat.shape[1])
        Yhat = np.exp(SgO_hat + mu_gO_local + sig2[None, :] / 2) - model.O_t[None, :]
    else:
        mu_gO_t = model.mu_gO[None, :]
        if mu_adj is not None:
            mu_gO_local = mu_gO_t + _nearest_interp_ts(
                mu_adj.reshape(T, ny, nx).astype(np.float32),
                model.grid_lat, model.grid_lon,
                model.station_lat, model.station_lon,
            )
        else:
            mu_gO_local = np.broadcast_to(mu_gO_t, SgO_hat.shape)
        Yhat = SgO_hat + mu_gO_local

    return xr.DataArray(
        Yhat.astype(np.float32),
        dims=("time", "station"),
        coords={
            "time": esm_da["time"].values,
            "year": ("time", esm_da["year"].values),
            "station": model.station_id,
        },
        name="value",
        attrs={"month": model.month, "n_pc": model.n_pc,
               "var_name": model.var_name, "delta_change": int(delta_change)},
    )


def _movmean_2d(X: np.ndarray, window: int) -> np.ndarray:
    """Moving average along axis=0, matching MATLAB movmean behavior.

    MATLAB movmean(x, k) for even k uses asymmetric window:
      kb = k//2 elements before, kf = (k-1)//2 elements after.
    """
    T = X.shape[0]
    out = np.empty_like(X)
    kb = window // 2           # elements before current
    kf = (window - 1) // 2    # elements after current
    for t in range(T):
        lo = max(0, t - kb)
        hi = min(T, t + kf + 1)
        out[t] = np.nanmean(X[lo:hi], axis=0)
    return out
