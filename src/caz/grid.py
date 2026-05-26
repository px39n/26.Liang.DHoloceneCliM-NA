"""Methods §5: station -> 0.20° grid via natural-neighbor; variance rescale; land mask."""
from __future__ import annotations
import numpy as np
import xarray as xr


def natural_neighbor_interp(
    values: np.ndarray, lon: np.ndarray, lat: np.ndarray,
    target_lon: np.ndarray, target_lat: np.ndarray,
) -> np.ndarray:
    """Natural-neighbor interpolation (scipy / naturalneighbor / verde)."""
    raise NotImplementedError("natural_neighbor_interp: pick backend")


def variance_rescale(
    grid: xr.DataArray,
    M_grid: xr.DataArray,
    window: int = 30 * 12,
) -> xr.DataArray:
    """Methods §5.2: anomaly *= sigma_mov(M) / sigma_mov(grid)  (in-place)."""
    grid_anom = grid - grid.rolling(time=window, min_periods=1).mean()
    s_M = M_grid.rolling(time=window, min_periods=1).std()
    s_g = grid.rolling(time=window, min_periods=1).std()
    return grid - grid_anom + grid_anom * (s_M / s_g)


def apply_landmask(grid: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return grid.where(mask)
