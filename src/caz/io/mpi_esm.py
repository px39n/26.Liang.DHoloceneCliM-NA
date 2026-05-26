"""MPI-ESM 1.2 CR (PalMod2) loader and unit conversion.

250 NetCDF files per variable, each covering 100 model-years (1200 monthly steps).
Model year 1 = 25000 BP = -23050 CE.  Model year 25000 = 1 BP = 1949 CE.

Conventions (same T31 grid as TraCE-21k II):
  - dims: (time, lat, lon=0..360)
  - lat: 48 cells, -87.16..87.16 (Gaussian T31)
  - lon: 96 cells, 0..356.25 step 3.75
  - tas: K  ->  degC
  - pr:  kg m-2 s-1  ->  mm/day
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import xarray as xr


BP_OFFSET = 23050  # model year 1 corresponds to CE year (-23050 + 1) = -23049

PR_KG_M2_S_TO_MM_DAY = 86400.0  # kg/m²/s → mm/day  (1 kg/m² = 1 mm)


def _model_year_to_ce(model_year: int) -> int:
    """Convert MPI-ESM model year to CE year."""
    return model_year - BP_OFFSET - 1


def _ce_to_model_year(ce_year: int) -> int:
    """Convert CE year to MPI-ESM model year."""
    return ce_year + BP_OFFSET + 1


def load_mpi_esm_var(
    data_dir: str | Path,
    var: str,
    year_min_ce: int | None = None,
    year_max_ce: int | None = None,
) -> xr.DataArray:
    """Load MPI-ESM multi-file dataset for one variable, convert to standard units.

    Parameters
    ----------
    data_dir : path to directory containing NC files for one variable
    var : 'tas' or 'pr'
    year_min_ce, year_max_ce : optional CE year bounds to restrict file loading
    """
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files in {data_dir}")

    if year_min_ce is not None or year_max_ce is not None:
        files = _filter_files_by_year(files, year_min_ce, year_max_ce)

    ds = xr.open_mfdataset(files, combine="by_coords",
                           chunks={"time": 1200}, use_cftime=True)
    da = ds[var]

    if var == "tas":
        da = da - 273.15
        da.attrs["units"] = "degC"
    elif var == "pr":
        da = da * PR_KG_M2_S_TO_MM_DAY
        da.attrs["units"] = "mm/day"

    da.name = var
    da = da.assign_attrs(source=str(data_dir), gcm="mpi-esm-cr")
    return da


def _filter_files_by_year(files: list[str], year_min_ce: int | None,
                          year_max_ce: int | None) -> list[str]:
    """Keep only files whose model-year range overlaps [year_min_ce, year_max_ce]."""
    kept = []
    for f in files:
        stem = Path(f).stem
        parts = stem.split("_")[-1]  # e.g. "0000101-0010012"
        yr_start = int(parts[:7]) // 100  # model year start (floor)
        yr_end = int(parts[8:15]) // 100    # model year end (floor)
        ce_start = _model_year_to_ce(yr_start)
        ce_end = _model_year_to_ce(yr_end)
        if year_max_ce is not None and ce_start > year_max_ce:
            continue
        if year_min_ce is not None and ce_end < year_min_ce:
            continue
        kept.append(f)
    return kept


def mpi_esm_time_to_year_month(
    time_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (CE year, month) from cftime time coordinate."""
    years = np.array([t.year for t in time_values], dtype=np.int32)
    months = np.array([t.month for t in time_values], dtype=np.int8)
    ce_years = years - BP_OFFSET - 1
    return ce_years, months
