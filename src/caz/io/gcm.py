"""Load ESM (GCM) NetCDF: historical (calibration) and Holocene transient.

Schema product 1 / 2.
"""
from __future__ import annotations
from pathlib import Path
import xarray as xr


def load_gcm(path: str | Path, vars: tuple[str, ...] = ("tas", "pr")) -> xr.Dataset:
    """Open a single ESM NetCDF, return Dataset with requested vars.

    Expected dims: (time, lat, lon).  No regridding, no unit conversion here.
    """
    ds = xr.open_dataset(path, chunks={"time": 240})
    missing = [v for v in vars if v not in ds.data_vars]
    if missing:
        raise KeyError(f"missing variables {missing} in {path}")
    return ds[list(vars)]


def to_anomaly(ds: xr.Dataset) -> xr.Dataset:
    """Per-grid temporal-mean removed (M' = M - mean_t M).  Methods §1.2."""
    return ds - ds.mean("time")
