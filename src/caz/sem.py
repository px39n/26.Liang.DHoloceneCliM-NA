"""Spatial Error Model (SEM) for residual noise modeling.

Implements Guaita's approach:
1. Fit SEM via profiled MLE: residuals = (I - lambda*W)^{-1} * eps
2. W = row-normalized Gaussian decay spatial weight matrix
3. ARMA(p,q) fitted to the whitened residuals eps per station
4. Simulation: generate ARMA noise → inverse SEM → spatially correlated noise
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import spsolve
from scipy import sparse


def _spectral_radius_power(W, tol=1e-10, max_iter=1000):
    """Estimate spectral radius of W via power iteration — O(n²) per iter."""
    n = W.shape[0]
    v = np.random.RandomState(0).randn(n)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(max_iter):
        Wv = W @ v
        rho_new = np.linalg.norm(Wv)
        if rho_new < 1e-15:
            return 0.0
        v = Wv / rho_new
        if abs(rho_new - rho) < tol * max(rho, 1e-10):
            return rho_new
        rho = rho_new
    return rho


def _gaussian_weight_matrix(coords, threshold):
    """Gaussian decay spatial weight matrix (row-normalized).

    Parameters
    ----------
    coords : (S, 2) — projected station coordinates (metres)
    threshold : float — decay scale h (metres)

    Returns
    -------
    W : (S, S) array — row-normalized weight matrix (zero diagonal)
    """
    D = squareform(pdist(coords))
    h = threshold
    W = np.exp(-(D ** 2) / (2 * h ** 2))
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    # Row-stochastic matrices have ρ=1 (Perron-Frobenius).
    # Gaussian kernel with h >= 25km on NA stations → no isolated stations.
    # Always rescale by (1 + 1e-2), matching MATLAB's eig(W) → rho=1.0 path.
    row_totals = W.sum(axis=1)
    if np.all(row_totals > 1e-10):
        W /= (1.0 + 1e-2)
    else:
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        if rho >= 1.0:
            W /= (rho + 1e-2)

    return W


def _profile_neg_loglik(lam, residual_mat, W, valid_mask, penalty_weight=0.1,
                        *, _cache=None):
    """Profiled negative log-likelihood for SEM.

    When _cache is provided with 'eigs_W_sub', uses the eigenvalue method
    (LeSage & Pace): logdet(I-λW) = Σlog|1-λ·eig_i|, O(n) per eval.
    Otherwise falls back to chol + slogdet.
    """
    from scipy.linalg import lapack

    T = residual_mat.shape[1]

    if _cache is not None:
        eps_buf = _cache['eps_buf']
        WvR_buf = _cache['WvR_buf']
        W_v_full = _cache['W_v_full']
        R_v = _cache['R_v']
        R_filled = _cache['R_filled']
        nan_mask_v = _cache['nan_mask_v']
        n_valid_v = _cache['n_valid_v']

        np.dot(W_v_full, R_filled, out=WvR_buf)
        np.multiply(WvR_buf, -lam, out=eps_buf)
        np.add(eps_buf, R_v, out=eps_buf)
        eps_buf[nan_mask_v] = np.nan

        eps_sq = np.square(eps_buf, out=WvR_buf)
        eps_sq[nan_mask_v] = 0.0
        sigma2_v = np.sum(eps_sq, axis=1)
        sigma2_v /= np.maximum(n_valid_v, 1)
        sigma2_v[n_valid_v == 0] = np.nan
        eps_v = eps_buf
    else:
        S = W.shape[0]
        A = np.eye(S, dtype=np.float64) - lam * W
        nan_mask = np.isnan(residual_mat)
        R_filled = np.where(nan_mask, 0.0, residual_mat)
        eps_mat = A @ R_filled
        eps_mat[nan_mask] = np.nan

        n_valid_per_st = np.sum(~np.isnan(eps_mat), axis=1)
        sigma2 = np.where(n_valid_per_st > 0,
                          np.nansum(eps_mat ** 2, axis=1) / np.maximum(n_valid_per_st, 1),
                          np.nan)
        vm_idx = np.where(valid_mask)[0]
        sigma2_v = sigma2[vm_idx]
        eps_v = eps_mat[vm_idx]

    if np.any((sigma2_v <= 0) | np.isnan(sigma2_v)):
        return 1e30

    # --- log-determinant ---
    if _cache is not None and 'eigs_W_sub' in _cache:
        eigs = _cache['eigs_W_sub']
        vals = 1.0 - lam * eigs
        logdetA = np.sum(np.log(np.abs(vals))).real
        real_mask = np.abs(eigs.imag) < 1e-10
        real_vals = 1.0 - lam * eigs[real_mask].real
        if np.sum(real_vals < 0) % 2 != 0:
            return 1e30
    else:
        if _cache is not None:
            A_buf = _cache['A_buf']
            neg_W_sub_F = _cache['neg_W_sub_F']
            I_v_F = _cache['I_v_F']
            np.multiply(neg_W_sub_F, lam, out=A_buf)
            np.add(A_buf, I_v_F, out=A_buf)
            A_sub = A_buf
        else:
            A_sub = A[np.ix_(np.where(valid_mask)[0], np.where(valid_mask)[0])]

        try:
            if _cache is not None:
                chol_buf = _cache['chol_buf']
                np.copyto(chol_buf, A_buf)
                U, info = lapack.dpotrf(chol_buf, lower=0, overwrite_a=1)
            else:
                A_f = np.asfortranarray(A_sub)
                U, info = lapack.dpotrf(A_f, lower=0, overwrite_a=1)
            if info != 0:
                raise np.linalg.LinAlgError(f"dpotrf failed with info={info}")
            logdetA = 2.0 * np.sum(np.log(np.diag(U)))
        except np.linalg.LinAlgError:
            sign, logdetA = np.linalg.slogdet(A_sub)
            if sign <= 0:
                return 1e30

    term1 = -(T / 2.0) * np.sum(np.log(sigma2_v))
    term2 = T * logdetA

    eps_safe = np.where(np.isnan(eps_v), 0.0, eps_v)
    quad = np.sum(eps_safe ** 2 / sigma2_v[:, None])
    term3 = -0.5 * quad

    logLik = term1 + term2 + term3
    penalty = -penalty_weight * np.log(max(1.0 - lam, 1e-10))
    return -(logLik - penalty)


def fit_sem(residual_mat, coords, valid_mask):
    """Fit Spatial Error Model via profiled MLE.

    Matches MATLAB's fit_SEM_MLE_fmincon: W is built from ALL station coords,
    eps computed for all stations with per-timestep NaN masking, likelihood
    terms use only valid_mask stations.

    Parameters
    ----------
    residual_mat : (S, T) — OLS regression residuals (NaN for missing)
    coords : (S, 2) — Albers-projected station coordinates (metres)
    valid_mask : (S,) bool — which stations are calibrated

    Returns
    -------
    lambda_hat : float — spatial autocorrelation parameter
    sigma2_hat : (S,) — per-station noise variance
    eps_mat : (S, T) — SEM-whitened residuals
    W_best : (S, S) — best spatial weight matrix (ALL stations)
    threshold_best : float — best distance threshold
    """
    S = residual_mat.shape[0]
    vm_idx = np.where(valid_mask)[0]
    n_v = len(vm_idx)

    nan_mask = np.isnan(residual_mat)
    R_filled = np.where(nan_mask, 0.0, residual_mat)
    nan_mask_v = nan_mask[vm_idx]

    thresholds = np.linspace(25_000, 100_000, 4)
    best_nLL = np.inf
    lambda_hat = 0.5
    W_best = np.eye(S, dtype=np.float32)
    threshold_best = 50_000.0

    I_v = np.eye(n_v, dtype=np.float64)

    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    import time as _time

    th_iter = _tqdm(thresholds, desc="    SEM thresholds", unit="th", leave=False) if _has_tqdm else thresholds
    for th in th_iter:
        _t0_th = _time.perf_counter()
        if _has_tqdm:
            _tqdm.write(f"      [{th/1000:.0f}km] building W...")
        W = _gaussian_weight_matrix(coords, th)
        _dt_W = _time.perf_counter() - _t0_th

        W_sub = W[np.ix_(vm_idx, vm_idx)].astype(np.float64)

        _t0_eig = _time.perf_counter()
        if _has_tqdm:
            _tqdm.write(f"      [{th/1000:.0f}km] W done ({_dt_W:.1f}s), computing eigvals ({n_v}x{n_v})...")
        eigs_W_sub = np.linalg.eigvals(W_sub)
        _dt_eig = _time.perf_counter() - _t0_eig
        if _has_tqdm:
            _tqdm.write(f"      [{th/1000:.0f}km] eigvals done ({_dt_eig:.1f}s), starting optimizer...")

        cache = {
            'neg_W_sub_F': np.asfortranarray(-W_sub),
            'I_v_F': np.eye(n_v, dtype=np.float64, order='F'),
            'A_buf': np.empty((n_v, n_v), dtype=np.float64, order='F'),
            'chol_buf': np.empty((n_v, n_v), dtype=np.float64, order='F'),
            'eps_buf': np.empty((n_v, residual_mat.shape[1]), dtype=np.float64),
            'WvR_buf': np.empty((n_v, residual_mat.shape[1]), dtype=np.float64),
            'W_v_full': W[vm_idx, :].astype(np.float64),
            'R_v': R_filled[vm_idx].copy(),
            'R_filled': R_filled,
            'nan_mask_v': nan_mask_v,
            'n_valid_v': np.sum(~nan_mask_v, axis=1).astype(np.float64),
            'eigs_W_sub': eigs_W_sub,
        }

        eval_pbar = _tqdm(desc=f"      evals (h={th/1000:.0f}km)", unit="ev",
                          leave=False) if _has_tqdm else None

        def obj(lam, _W=W, _c=cache, _pb=eval_pbar):
            nll = _profile_neg_loglik(lam, residual_mat, _W, valid_mask, _cache=_c)
            if _pb is not None:
                _pb.update(1)
            return nll

        result = minimize_scalar(obj, bounds=(0.0, 0.9), method='bounded',
                                 options={'xatol': 1e-6, 'maxiter': 200})

        if eval_pbar is not None:
            eval_pbar.close()

        if result.fun < best_nLL:
            best_nLL = result.fun
            lambda_hat = result.x
            W_best = W
            threshold_best = th

    A_best = np.eye(S) - lambda_hat * W_best
    T = residual_mat.shape[1]
    eps_mat = np.full_like(residual_mat, np.nan)
    for t in range(T):
        res_t = residual_mat[:, t]
        valid = ~np.isnan(res_t)
        if valid.sum() < 2:
            continue
        eps_mat[valid, t] = A_best[np.ix_(valid, valid)] @ res_t[valid]

    sigma2_hat = np.nanmean(eps_mat ** 2, axis=1)

    return lambda_hat, sigma2_hat, eps_mat, W_best, threshold_best


try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):
        def wrapper(f): return f
        return wrapper if not args or not callable(args[0]) else args[0]


@_njit(cache=True)
def _arma11_css_core(y, phi_vals_coarse, theta_vals_coarse,
                     phi_vals_fine, theta_vals_fine):
    """Numba-accelerated ARMA(1,1) CSS grid search."""
    n = len(y)
    best_css = 1e30
    best_phi = 0.0
    best_theta = 0.0

    for i in range(len(phi_vals_coarse)):
        phi = phi_vals_coarse[i]
        for j in range(len(theta_vals_coarse)):
            theta = theta_vals_coarse[j]
            css = 0.0
            e_prev = 0.0
            for t in range(1, n):
                e_t = y[t] - phi * y[t - 1] - theta * e_prev
                css += e_t * e_t
                e_prev = e_t
            if css < best_css:
                best_css = css
                best_phi = phi
                best_theta = theta

    lo_phi = max(best_phi - 0.1, -0.99)
    hi_phi = min(best_phi + 0.1, 0.99)
    lo_theta = max(best_theta - 0.1, -0.99)
    hi_theta = min(best_theta + 0.1, 0.99)

    for i in range(len(phi_vals_fine)):
        phi = lo_phi + (hi_phi - lo_phi) * i / max(len(phi_vals_fine) - 1, 1)
        for j in range(len(theta_vals_fine)):
            theta = lo_theta + (hi_theta - lo_theta) * j / max(len(theta_vals_fine) - 1, 1)
            css = 0.0
            e_prev = 0.0
            for t in range(1, n):
                e_t = y[t] - phi * y[t - 1] - theta * e_prev
                css += e_t * e_t
                e_prev = e_t
            if css < best_css:
                best_css = css
                best_phi = phi
                best_theta = theta

    return best_phi, best_theta, best_css / n


def _fit_arma11_css(y):
    """Fit ARMA(1,1) via Conditional Sum of Squares (grid search + refinement).

    Matches Guaita's MATLAB approach. More robust than MLE for short series.
    """
    n = len(y)
    if n < 12:
        return 0.0, 0.0, float(np.var(y))
    phi_c = np.linspace(-0.95, 0.95, 39)
    theta_c = np.linspace(-0.95, 0.95, 39)
    phi_f = np.linspace(0, 1, 21)  # placeholder, actual range computed in core
    theta_f = np.linspace(0, 1, 21)
    return _arma11_css_core(y, phi_c, theta_c, phi_f, theta_f)


def fit_arma_per_station(eps_mat, valid_mask, p=1, q=1, progress=True):
    """Fit ARMA(p,q) to SEM-whitened residuals per station.

    Uses CSS (Conditional Sum of Squares) grid search — same as Guaita's MATLAB.
    Returns list of dicts with keys: 'ar', 'ma', 'variance'.
    """
    S = eps_mat.shape[0]
    models = [None] * S

    iterator = range(S)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="    ARMA stations", unit="sta", leave=False)
        except ImportError:
            pass

    for i in iterator:
        if not valid_mask[i]:
            continue
        y = eps_mat[i, :]
        y = y[~np.isnan(y)]
        phi, theta, var = _fit_arma11_css(y)
        models[i] = {'ar': np.array([phi]), 'ma': np.array([theta]),
                     'variance': var}

    return models


def simulate_arma_noise(models, valid_mask, n_time, start_time=0,
                        p=1, q=1, rng=None):
    """Simulate ARMA noise for each station.

    Parameters
    ----------
    models : list of dicts from fit_arma_per_station
    valid_mask : (S,) bool
    n_time : int — number of timesteps
    start_time : int — start generating from this index
    p, q : ARMA orders
    rng : numpy random generator

    Returns
    -------
    eps_arma : (S, n_time) — simulated ARMA noise
    """
    if rng is None:
        rng = np.random.default_rng()

    S = len(models)
    eps_arma = np.zeros((S, n_time))

    # Pre-extract parameters for vectorized/Numba path
    phi_arr = np.zeros(S)
    theta_arr = np.zeros(S)
    sigma_arr = np.zeros(S)
    active = np.zeros(S, dtype=np.bool_)
    for i in range(S):
        if valid_mask[i] and models[i] is not None:
            active[i] = True
            phi_arr[i] = models[i]['ar'][0] if len(models[i]['ar']) > 0 else 0.0
            theta_arr[i] = models[i]['ma'][0] if len(models[i]['ma']) > 0 else 0.0
            sigma_arr[i] = np.sqrt(models[i]['variance'])

    # Generate all noise at once
    eta_all = np.zeros((S, n_time))
    for i in np.where(active)[0]:
        eta_all[i, :] = sigma_arr[i] * rng.standard_normal(n_time)

    eps_arma = _simulate_arma_batch(phi_arr, theta_arr, eta_all, active,
                                     n_time, start_time)
    return eps_arma


@_njit(cache=True)
def _simulate_arma_batch(phi_arr, theta_arr, eta_all, active, n_time, start_time):
    S = len(phi_arr)
    eps_arma = np.zeros((S, n_time))
    for i in range(S):
        if not active[i]:
            continue
        phi = phi_arr[i]
        theta = theta_arr[i]
        for t in range(max(1, start_time), n_time):
            eps_arma[i, t] = phi * eps_arma[i, t-1] + theta * eta_all[i, t-1] + eta_all[i, t]
    return eps_arma


def inverse_sem(eps_arma, W, lam):
    """Convert ARMA noise back through SEM: u = (I - lambda*W)^{-1} * eps.

    Parameters
    ----------
    eps_arma : (S, T) — ARMA noise
    W : (S, S) — spatial weight matrix
    lam : float — spatial autocorrelation parameter

    Returns
    -------
    u_mat : (S, T) — spatially correlated noise
    """
    from scipy.linalg import lu_factor, lu_solve
    S = W.shape[0]
    A = np.eye(S) - lam * W
    lu, piv = lu_factor(A)
    return lu_solve((lu, piv), eps_arma)
