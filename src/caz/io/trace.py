"""TraCE-21k II loader and unit conversion.

The two files we use:
  - TraCE-21K-II.monthly.TREFHT.nc   (Reference height temperature, K)
  - TraCE-21K-II.monthly.PRECT.nc    (Total precipitation rate, m/s)

Conventions:
  - dims: (time, lat, lon=0..360)
  - time: float64, units 'ka BP' (negative for past, ~0 for 1990 CE)
    264600 monthly steps spanning approx -22.0 to ~0.05 ka BP
  - lat: 48 cells, -87.16..87.16 (Gaussian T31)
  - lon: 96 cells, 0..356.25 step 3.75
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


TRACE_VAR = {"tas": "TREFHT", "pr": "PRECT"}
PR_M_S_TO_MM_DAY = 86400.0 * 1000.0  # m/s -> mm/day  (1 m/s * 86400 s/day * 1000 mm/m)


def load_trace_var(path: str | Path, var: str) -> xr.DataArray:
    """Open one TraCE NetCDF and return the variable in standard units.

    var = 'tas'  -> TREFHT in degC
    var = 'pr'   -> PRECT in mm/day
    """
    src_name = TRACE_VAR.get(var, var)
    ds = xr.open_dataset(path, chunks={"time": 12000})
    if src_name not in ds:
        raise KeyError(f"variable {src_name!r} not in {path}")
    da = ds[src_name]

    if var == "tas":
        da = da - 273.15
        da.attrs["units"] = "degC"
    elif var == "pr":
        da = da * PR_M_S_TO_MM_DAY
        da.attrs["units"] = "mm/day"

    da.name = var
    da = da.assign_attrs(source=str(path), original_var=src_name)
    return da


def trace_time_to_calendar_year(time: np.ndarray) -> np.ndarray:
    """Convert TraCE 'ka BP' time to fractional Gregorian year (CE).

    BP convention: t_BP = 0 corresponds to 1950 CE.
    TraCE-21k II appears to use ka BP (kiloyears before 1950).
    Verify with the file: max time should be slightly > 0 (around 1990 CE).
    """
    return 1950.0 + np.asarray(time) * 1000.0


def trace_time_to_year_month(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate (year, month) from monthly TraCE float time array.

    The monthly time stamps are at month centres so we round to the nearest month
    after converting to CE.
    """
    cal_year = trace_time_to_calendar_year(time)
    months_since_0 = np.round(cal_year * 12.0).astype(np.int64)
    year = (months_since_0 // 12).astype(np.int32)
    month = ((months_since_0 % 12) + 1).astype(np.int8)
    return year, month


def select_na_window(
    da: xr.DataArray,
    lon_min: float = -180.0,
    lon_max: float = -50.0,
    lat_min: float = 7.0,
    lat_max: float = 75.0,
    pad: float = 5.0,
) -> xr.DataArray:
    """Subset to a North American window with `pad` deg margin.

    Handles 0-360 longitude convention by rolling and renaming.
    """
    if "lon" not in da.coords:
        raise KeyError("expected 'lon' coordinate")

    lon0 = da["lon"]
    if float(lon0.max()) > 180:
        new_lon = ((lon0 + 180.0) % 360.0) - 180.0
        da = da.assign_coords(lon=new_lon).sortby("lon")

    lat = da["lat"]
    if float(lat[0]) > float(lat[-1]):
        da = da.sortby("lat")

    da = da.sel(
        lat=slice(lat_min - pad, lat_max + pad),
        lon=slice(lon_min - pad, lon_max + pad),
    )
    return da
