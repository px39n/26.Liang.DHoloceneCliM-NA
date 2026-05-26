"""Station → grid interpolation using Albers-projected natural-neighbor (Sibson).

Follows Guaita's approach: project station/grid coordinates to NAD83 Conus
Albers (EPSG:5070), then perform Sibson natural-neighbor interpolation with
nearest-neighbor extrapolation outside the convex hull.

This matches MATLAB's scatteredInterpolant(..., 'natural', 'nearest').
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.interpolate import NearestNDInterpolator

from caz.natneighbor import sibson_interp, sibson_weight_matrix


@dataclass
class GridSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    res_deg: float = 0.20

    def lat_lon(self) -> tuple[np.ndarray, np.ndarray]:
        lat = np.arange(self.lat_min, self.lat_max + 1e-9, self.res_deg, dtype=np.float32)
        lon = np.arange(self.lon_min, self.lon_max + 1e-9, self.res_deg, dtype=np.float32)
        return lat, lon


def _albers_forward(lon_deg: np.ndarray, lat_deg: np.ndarray):
    """Project lon/lat (degrees) to EPSG:5070 Albers x/y (metres)."""
    a = 6378137.0
    f = 1 / 298.257222101
    e2 = 2 * f - f ** 2
    e = np.sqrt(e2)

    phi1 = np.radians(29.5)
    phi2 = np.radians(45.5)
    phi0 = np.radians(23.0)
    lam0 = np.radians(-96.0)

    def _m(phi):
        return np.cos(phi) / np.sqrt(1 - e2 * np.sin(phi) ** 2)

    def _q(phi):
        sp = np.sin(phi)
        return (1 - e2) * (sp / (1 - e2 * sp ** 2) -
                           np.log((1 - e * sp) / (1 + e * sp)) / (2 * e))

    m1, m2 = _m(phi1), _m(phi2)
    q0, q1, q2 = _q(phi0), _q(phi1), _q(phi2)
    n = (m1 ** 2 - m2 ** 2) / (q2 - q1)
    C = m1 ** 2 + n * q1
    rho0 = a * np.sqrt(C - n * q0) / n

    phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lam = np.radians(np.asarray(lon_deg, dtype=np.float64))
    q = _q(phi)
    rho = a * np.sqrt(C - n * q) / n
    theta = n * (lam - lam0)

    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y


def delaunay_gridding(
    values: np.ndarray,            # (T, S) station predictions
    pts_lat: np.ndarray,           # (S,)
    pts_lon: np.ndarray,           # (S,)
    spec: GridSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sibson natural-neighbor interpolation in Albers-projected coordinates.

    Returns
    -------
    grid : (T, ny, nx) float32
    lat  : (ny,)
    lon  : (nx,)
    """
    lat, lon = spec.lat_lon()
    ny, nx = lat.size, lon.size
    longrid, latgrid = np.meshgrid(lon, lat)

    in_box = (
        (pts_lat >= spec.lat_min - 5) & (pts_lat <= spec.lat_max + 5)
        & (pts_lon >= spec.lon_min - 5) & (pts_lon <= spec.lon_max + 5)
    )
    pts_lat = pts_lat[in_box]
    pts_lon = pts_lon[in_box]
    values = values[:, in_box]
    if pts_lat.size == 0:
        raise ValueError("no stations in (padded) grid region")

    x_st, y_st = _albers_forward(pts_lon, pts_lat)
    x_grd, y_grd = _albers_forward(longrid.ravel(), latgrid.ravel())
    qpts = np.column_stack([x_grd, y_grd])

    st_pts = np.column_stack([x_st, y_st])
    T = values.shape[0]
    has_nan = np.isnan(values).any()

    if has_nan:
        # per-timestep interpolation (handles varying NaN patterns)
        grid_flat = np.full((T, x_grd.size), np.nan, dtype=np.float32)
        for t in range(T):
            z = values[t, :]
            valid = ~np.isnan(z)
            if valid.sum() < 4:
                continue
            pts = np.column_stack([x_st[valid], y_st[valid]])
            nn_vals = sibson_interp(pts, z[valid].astype(np.float64), qpts)
            nans = np.isnan(nn_vals)
            if nans.any():
                near = NearestNDInterpolator(pts, z[valid])
                nn_vals[nans] = near(qpts[nans])
            grid_flat[t] = nn_vals.astype(np.float32)
    else:
        # precomputed weight matrix (much faster for NaN-free data)
        from collections import defaultdict
        pt_map = defaultdict(list)
        for i in range(len(st_pts)):
            key = (round(st_pts[i, 0], 6), round(st_pts[i, 1], 6))
            pt_map[key].append(i)

        dedup_pts = []
        dedup_to_orig = []
        for key, indices in pt_map.items():
            dedup_pts.append(st_pts[indices[0]])
            dedup_to_orig.append(indices)
        dedup_pts_arr = np.array(dedup_pts)

        W = sibson_weight_matrix(dedup_pts_arr, qpts)

        # average duplicate station values
        n_dedup = len(dedup_pts)
        vals_dedup = np.empty((T, n_dedup), dtype=np.float64)
        for di, orig_indices in enumerate(dedup_to_orig):
            vals_dedup[:, di] = values[:, orig_indices].mean(axis=1)

        grid_flat = (W @ vals_dedup.T).T.astype(np.float32)

        # fill outside hull with nearest
        nnz_per_row = np.asarray(W.getnnz(axis=1)).ravel()
        hull_nans = np.isnan(grid_flat[0]) | (nnz_per_row == 0)
        if hull_nans.any():
            near = NearestNDInterpolator(st_pts, np.arange(len(x_st)))
            near_idx = near(qpts).astype(int)
            for t in range(T):
                grid_flat[t, hull_nans] = values[t, near_idx[hull_nans]].astype(np.float32)

    grid = grid_flat.reshape(T, ny, nx)
    return grid, lat, lon


@njit(parallel=True, cache=True)
def _varcorr_fused_kernel(pcr, esm, out, kb, ka):
    """Fused variance correction: movmean + movstd + ratio in one pass per row.

    Avoids creating large intermediate arrays (~16 GB for full grid).
    Each row is independent → parallelized with prange.
    """
    n_rows, n_cols = pcr.shape
    for i in prange(n_rows):
        # ---- Pass 1: compute movmean for pcr and esm ----
        pcr_mm = np.empty(n_cols)
        esm_mm = np.empty(n_cols)

        sp, se = 0.0, 0.0
        cp, ce = 0, 0
        lo0 = max(0, -kb)
        hi0 = min(n_cols, ka + 1)
        for k in range(lo0, hi0):
            vp, ve = pcr[i, k], esm[i, k]
            if vp == vp:
                sp += vp; cp += 1
            if ve == ve:
                se += ve; ce += 1
        pcr_mm[0] = sp / cp if cp > 0 else np.nan
        esm_mm[0] = se / ce if ce > 0 else np.nan

        for j in range(1, n_cols):
            old_idx = j - 1 - kb
            if 0 <= old_idx < n_cols:
                vp = pcr[i, old_idx]
                if vp == vp: sp -= vp; cp -= 1
                ve = esm[i, old_idx]
                if ve == ve: se -= ve; ce -= 1
            new_idx = j + ka
            if 0 <= new_idx < n_cols:
                vp = pcr[i, new_idx]
                if vp == vp: sp += vp; cp += 1
                ve = esm[i, new_idx]
                if ve == ve: se += ve; ce += 1
            pcr_mm[j] = sp / cp if cp > 0 else np.nan
            esm_mm[j] = se / ce if ce > 0 else np.nan

        # ---- Pass 2: compute movstd of anomalies ----
        # anomalies: pcr_anom[j] = pcr[i,j] - pcr_mm[j]
        sp2, sp22 = 0.0, 0.0
        se2, se22 = 0.0, 0.0
        cp2, ce2 = 0, 0

        for k in range(lo0, hi0):
            ap = pcr[i, k] - pcr_mm[k]
            ae = esm[i, k] - esm_mm[k]
            if ap == ap:
                sp2 += ap; sp22 += ap * ap; cp2 += 1
            if ae == ae:
                se2 += ae; se22 += ae * ae; ce2 += 1

        if cp2 > 1:
            pvar = (sp22 - sp2 * sp2 / cp2) / (cp2 - 1)
            pstd = np.sqrt(max(pvar, 0.0))
        elif cp2 == 1:
            pstd = 0.0
        else:
            pstd = np.nan

        if ce2 > 1:
            evar = (se22 - se2 * se2 / ce2) / (ce2 - 1)
            estd = np.sqrt(max(evar, 0.0))
        elif ce2 == 1:
            estd = 0.0
        else:
            estd = np.nan

        ratio = estd / pstd if pstd > 0.0 and pstd == pstd and estd == estd else 1.0
        anom = pcr[i, 0] - pcr_mm[0]
        out[i, 0] = pcr_mm[0] + anom * ratio if anom == anom else np.nan

        for j in range(1, n_cols):
            old_idx = j - 1 - kb
            if 0 <= old_idx < n_cols:
                ap = pcr[i, old_idx] - pcr_mm[old_idx]
                ae = esm[i, old_idx] - esm_mm[old_idx]
                if ap == ap:
                    sp2 -= ap; sp22 -= ap * ap; cp2 -= 1
                if ae == ae:
                    se2 -= ae; se22 -= ae * ae; ce2 -= 1
            new_idx = j + ka
            if 0 <= new_idx < n_cols:
                ap = pcr[i, new_idx] - pcr_mm[new_idx]
                ae = esm[i, new_idx] - esm_mm[new_idx]
                if ap == ap:
                    sp2 += ap; sp22 += ap * ap; cp2 += 1
                if ae == ae:
                    se2 += ae; se22 += ae * ae; ce2 += 1

            if cp2 > 1:
                pvar = (sp22 - sp2 * sp2 / cp2) / (cp2 - 1)
                pstd = np.sqrt(max(pvar, 0.0))
            elif cp2 == 1:
                pstd = 0.0
            else:
                pstd = np.nan

            if ce2 > 1:
                evar = (se22 - se2 * se2 / ce2) / (ce2 - 1)
                estd = np.sqrt(max(evar, 0.0))
            elif ce2 == 1:
                estd = 0.0
            else:
                estd = np.nan

            ratio = estd / pstd if pstd > 0.0 and pstd == pstd and estd == estd else 1.0
            anom = pcr[i, j] - pcr_mm[j]
            out[i, j] = pcr_mm[j] + anom * ratio if anom == anom else np.nan


def variance_correction(
    pcr_grid: np.ndarray,
    esm_grid: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Adjust PCR gridded output to match ESM variability (Guaita's movstd scaling).

    For each grid cell, rescales PCR anomalies so that their local standard
    deviation matches that of the ESM field, preserving the PCR moving mean.

    Uses a fused Numba kernel that computes movmean + movstd + ratio in a
    single pass per row, avoiding large intermediate arrays.

    Parameters
    ----------
    pcr_grid : (T, ny, nx) or (n_cells, T)
        PCR-downscaled gridded field.
    esm_grid : same shape as pcr_grid
        ESM field on the same grid, same time indices.
    window : int
        Moving-window size (in time steps) for movmean/movstd.

    Returns
    -------
    adjusted : same shape as pcr_grid
        Variance-corrected PCR field.
    """
    orig_shape = pcr_grid.shape
    if pcr_grid.ndim == 3:
        T, ny, nx = pcr_grid.shape
        pcr = np.ascontiguousarray(pcr_grid.reshape(T, ny * nx).T, dtype=np.float64)
        esm = np.ascontiguousarray(esm_grid.reshape(T, ny * nx).T, dtype=np.float64)
    else:
        pcr = np.ascontiguousarray(pcr_grid, dtype=np.float64)
        esm = np.ascontiguousarray(esm_grid, dtype=np.float64)

    kb, ka = _window_params(window)
    out = np.empty_like(pcr)
    _varcorr_fused_kernel(pcr, esm, out, kb, ka)

    if pcr_grid.ndim == 3:
        return out.T.reshape(orig_shape).astype(np.float32)
    return out.reshape(orig_shape).astype(np.float32)


def _window_params(window: int):
    """Return (kb, ka) for MATLAB-compatible centered window."""
    if window % 2 == 0:
        return window // 2, window // 2 - 1
    return window // 2, window // 2


@njit(parallel=True, cache=True)
def _movmean_2d_kernel(arr, out, kb, ka):
    """Numba-parallel row-wise moving mean with NaN skipping."""
    n_rows, n_cols = arr.shape
    for i in prange(n_rows):
        s = 0.0
        c = 0
        # Initialize window for j=0: indices [max(0,-kb), min(n_cols, ka+1))
        lo0 = max(0, -kb)
        hi0 = min(n_cols, ka + 1)
        for k in range(lo0, hi0):
            v = arr[i, k]
            if v == v:  # not NaN
                s += v
                c += 1
        out[i, 0] = s / c if c > 0 else np.nan
        # Slide window for j=1..n_cols-1
        for j in range(1, n_cols):
            # Remove element leaving the window (index j-1-kb)
            old_idx = j - 1 - kb
            if 0 <= old_idx < n_cols:
                v = arr[i, old_idx]
                if v == v:
                    s -= v
                    c -= 1
            # Add element entering the window (index j+ka)
            new_idx = j + ka
            if 0 <= new_idx < n_cols:
                v = arr[i, new_idx]
                if v == v:
                    s += v
                    c += 1
            out[i, j] = s / c if c > 0 else np.nan


def _movmean_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Row-wise moving mean (axis=1) matching MATLAB movmean(..., w, 2, 'omitnan').

    Uses Numba JIT with parallel prange for near-C performance.
    """
    kb, ka = _window_params(window)
    out = np.empty_like(arr, dtype=np.float64)
    _movmean_2d_kernel(arr.astype(np.float64), out, kb, ka)
    return out


@njit(parallel=True, cache=True)
def _movstd_2d_kernel(arr, out, kb, ka):
    """Numba-parallel row-wise moving std (ddof=1) with NaN skipping."""
    n_rows, n_cols = arr.shape
    for i in prange(n_rows):
        s = 0.0
        s2 = 0.0
        c = 0
        lo0 = max(0, -kb)
        hi0 = min(n_cols, ka + 1)
        for k in range(lo0, hi0):
            v = arr[i, k]
            if v == v:
                s += v
                s2 += v * v
                c += 1
        if c > 1:
            var = (s2 - s * s / c) / (c - 1)
            out[i, 0] = np.sqrt(max(var, 0.0))
        elif c == 1:
            out[i, 0] = 0.0
        else:
            out[i, 0] = np.nan
        for j in range(1, n_cols):
            old_idx = j - 1 - kb
            if 0 <= old_idx < n_cols:
                v = arr[i, old_idx]
                if v == v:
                    s -= v
                    s2 -= v * v
                    c -= 1
            new_idx = j + ka
            if 0 <= new_idx < n_cols:
                v = arr[i, new_idx]
                if v == v:
                    s += v
                    s2 += v * v
                    c += 1
            if c > 1:
                var = (s2 - s * s / c) / (c - 1)
                out[i, j] = np.sqrt(max(var, 0.0))
            elif c == 1:
                out[i, j] = 0.0
            else:
                out[i, j] = np.nan


def _movstd_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Row-wise moving std (axis=1) matching MATLAB movstd(..., w, 0, 2, 'omitnan').

    MATLAB movstd flag=0 → ddof=1 (sample std).
    Uses Numba JIT with parallel prange for near-C performance.
    """
    kb, ka = _window_params(window)
    out = np.empty_like(arr, dtype=np.float64)
    _movstd_2d_kernel(arr.astype(np.float64), out, kb, ka)
    return out


# backward-compatible alias
idw_gridding = delaunay_gridding
