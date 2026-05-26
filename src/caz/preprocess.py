"""Methods §1: regrid GCM to target 0.20°, compute anomalies, build splits."""
from __future__ import annotations
import xarray as xr

from .config import Config, Grid, Region
from .io.ghcn import subset_region, time_split
from .io.gcm import to_anomaly


def regrid_akima(ds: xr.Dataset, grid: Grid, region: Region) -> xr.Dataset:
    """Akima 3rd-order cubic Hermite regridding to target lat/lon (Methods §1.1)."""
    raise NotImplementedError("regrid_akima: implement with akima/xesmf")


def preprocess(gcm_hist: xr.Dataset, ghcn: xr.Dataset, cfg: Config) -> dict:
    """Pipeline §1.

    Returns dict:
        - M_prime  (Nt, Ng) anomaly matrix
        - stations subset
        - splits {cal, msel, test}
    """
    M = regrid_akima(gcm_hist, cfg.grid, cfg.region)
    M_prime = to_anomaly(M)

    sta = subset_region(ghcn, cfg.region, min_years=cfg.calibration.min_record_years)
    splits = time_split(
        sta,
        cal=cfg.calibration.cal_years,
        msel=cfg.calibration.msel_years,
        test=cfg.calibration.test_years,
    )

    return {"M_prime": M_prime, "stations": sta, "splits": splits}
