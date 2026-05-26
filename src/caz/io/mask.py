"""Land mask loader on the 0.20° NA grid.  Schema product 4."""
from __future__ import annotations
from pathlib import Path
import xarray as xr


def load_landmask(path: str | Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if "mask" not in ds:
        raise KeyError(f"'mask' var missing in {path}")
    return ds["mask"].astype(bool)
