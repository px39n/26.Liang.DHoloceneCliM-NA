"""GHCN-m v4 station loader.

Mirrors logic of `25.Guaita.DPastCliM-NA/preprocessing/GHCNm.m`:

  - tas: parse fixed-width `.inv` (metadata) + `.dat` (12 monthly TAVG values
         per station-year, units 0.01 degC, missing = -9999).
  - pr:  loop per-station CSVs, compute mm/day from monthly mm * 10 divided
         by days-in-month.

Output is a pandas DataFrame in long form with columns
    [ID, year, month, value, lat, lon, elev]
plus a metadata DataFrame [ID, lat, lon, elev, name].

Save to Parquet via `dump_parquet()` for reuse.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Region

__all__ = [
    "load_ghcn_tas",
    "load_ghcn_pr",
    "subset_region_df",
    "filter_min_record",
    "to_xarray_station",
    "dump_parquet",
    "load_ghcn",
    "subset_region",
    "time_split",
]

MISSING_TAS = -9999
TAS_SCALE = 100.0  # 0.01 degC -> degC
PR_SCALE = 10.0    # 0.1 mm/month -> mm/month


def _parse_inv_tas(path: Path) -> pd.DataFrame:
    """Parse `ghcnm.tavg.v4.*.inv`: 11-char ID, lat, lon, elev, location."""
    cols = [(0, 11), (12, 20), (21, 30), (31, 38), (38, 100)]
    names = ["ID", "lat", "lon", "elev", "name"]
    df = pd.read_fwf(
        path,
        colspecs=cols,
        names=names,
        dtype={"ID": str, "lat": float, "lon": float, "elev": float, "name": str},
    )
    df["name"] = df["name"].str.strip()
    return df


def _parse_dat_tas(path: Path) -> pd.DataFrame:
    """Parse `.dat`: each row = ID(11) + Year(4) + ELEM(4) + 12 * (Value(5) + 3-char flags).

    Returns long DataFrame [ID, year, month, value_C].
    """
    n_rows = sum(1 for _ in path.open("r"))
    ids = np.empty(n_rows, dtype=object)
    years = np.empty(n_rows, dtype=np.int32)
    vals = np.empty((n_rows, 12), dtype=np.float32)

    with path.open("r") as fh:
        for i, line in enumerate(fh):
            ids[i] = line[0:11]
            years[i] = int(line[11:15])
            for m in range(12):
                start = 19 + m * 8
                v = line[start : start + 5].strip()
                vals[i, m] = MISSING_TAS if not v else int(v)

    long = pd.DataFrame(
        {
            "ID": np.repeat(ids, 12),
            "year": np.repeat(years, 12),
            "month": np.tile(np.arange(1, 13, dtype=np.int8), n_rows),
            "value": vals.reshape(-1) / TAS_SCALE,
        }
    )
    long.loc[long["value"] <= MISSING_TAS / TAS_SCALE + 0.0001, "value"] = np.nan
    long.loc[long["value"] == -99.99, "value"] = np.nan
    return long


def load_ghcn_tas(inv_path: str | Path, dat_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GHCN-m v4 monthly mean temperature.

    Returns
    -------
    obs : DataFrame [ID, year, month, value (degC), lat, lon, elev]
    meta : DataFrame [ID, lat, lon, elev, name]
    """
    meta = _parse_inv_tas(Path(inv_path))
    long = _parse_dat_tas(Path(dat_path))
    obs = long.merge(meta[["ID", "lat", "lon", "elev"]], on="ID", how="left")
    obs = obs.dropna(subset=["value"]).reset_index(drop=True)
    return obs, meta


def load_ghcn_pr(
    inv_path: str | Path | None,
    csv_dir: str | Path,
    region: Region | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GHCN-m v4 monthly precipitation from per-station CSV directory.

    CSV format (no header): ID, "name", lat, lon, elev, YYYYMM, value*10[mm], 4 flags.
    Value in source is monthly total in 0.1 mm; output is **mm/day**.

    If `region` is given, station files outside the bounding box are skipped
    (large speedup over loading 126k stations globally).
    """
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("*.csv"))

    cols = ["ID", "name", "lat", "lon", "elev", "yyyymm", "value", "mflag", "qflag", "sflag", "src"]
    dtypes = {
        "ID": str, "name": str, "lat": float, "lon": float, "elev": float,
        "yyyymm": int, "value": float,
        "mflag": str, "qflag": str, "sflag": str, "src": str,
    }

    meta_rows: list[dict] = []
    obs_chunks: list[pd.DataFrame] = []

    iterator = files
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(files, desc="GHCN pr CSVs", unit="file")
        except ImportError:
            pass

    for fp in iterator:
        try:
            df = pd.read_csv(
                fp,
                header=None,
                names=cols,
                dtype=dtypes,
                quotechar='"',
                skipinitialspace=True,
                na_values=["", " "],
            )
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if df.empty:
            continue

        st_lat, st_lon = float(df["lat"].iloc[0]), float(df["lon"].iloc[0])
        if region is not None and not (
            region.lat_min <= st_lat <= region.lat_max
            and region.lon_min <= st_lon <= region.lon_max
        ):
            continue

        df["year"] = (df["yyyymm"] // 100).astype(np.int32)
        df["month"] = (df["yyyymm"] % 100).astype(np.int8)
        df["value"] = df["value"] / PR_SCALE
        days = np.array([calendar.monthrange(y, m)[1] for y, m in zip(df["year"], df["month"])])
        df["value"] = df["value"] / days

        df.loc[df["value"] < 0, "value"] = np.nan

        obs_chunks.append(
            df[["ID", "year", "month", "value", "lat", "lon", "elev"]].reset_index(drop=True)
        )
        meta_rows.append(
            {"ID": str(df["ID"].iloc[0]), "name": str(df["name"].iloc[0]).strip().strip('"'),
             "lat": st_lat, "lon": st_lon, "elev": float(df["elev"].iloc[0])}
        )

    if not obs_chunks:
        empty_obs = pd.DataFrame(columns=["ID", "year", "month", "value", "lat", "lon", "elev"])
        empty_meta = pd.DataFrame(columns=["ID", "name", "lat", "lon", "elev"])
        return empty_obs, empty_meta

    obs = pd.concat(obs_chunks, ignore_index=True).dropna(subset=["value"])
    meta = pd.DataFrame(meta_rows).drop_duplicates("ID").reset_index(drop=True)
    return obs, meta


def subset_region_df(obs: pd.DataFrame, region: Region) -> pd.DataFrame:
    """Bounding-box filter on a long-form station DataFrame."""
    m = (
        (obs["lat"] >= region.lat_min) & (obs["lat"] <= region.lat_max)
        & (obs["lon"] >= region.lon_min) & (obs["lon"] <= region.lon_max)
    )
    return obs.loc[m].reset_index(drop=True)


def filter_min_record(obs: pd.DataFrame, min_years: int = 20) -> pd.DataFrame:
    """Drop stations with fewer than `min_years * 12` non-missing months."""
    counts = obs.groupby("ID")["value"].size()
    keep = counts.index[counts >= min_years * 12]
    return obs.loc[obs["ID"].isin(keep)].reset_index(drop=True)


def filter_time(obs: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    m = (obs["year"] >= year_min) & (obs["year"] <= year_max)
    return obs.loc[m].reset_index(drop=True)


def to_xarray_station(obs: pd.DataFrame) -> xr.Dataset:
    """Convert long-form DataFrame to (station, time) xarray.

    Time is encoded as numpy datetime64 of the month's first day.
    """
    obs = obs.copy()
    obs["time"] = pd.to_datetime(
        dict(year=obs["year"], month=obs["month"], day=1)
    )
    wide = obs.pivot_table(index="ID", columns="time", values="value", aggfunc="first")
    meta = obs.groupby("ID")[["lat", "lon", "elev"]].first().reindex(wide.index)
    ds = xr.Dataset(
        data_vars={"value": (("station", "time"), wide.values.astype(np.float32))},
        coords={
            "station": wide.index.values.astype(str),
            "time": wide.columns.values,
            "lat": ("station", meta["lat"].values.astype(np.float32)),
            "lon": ("station", meta["lon"].values.astype(np.float32)),
            "elev": ("station", meta["elev"].values.astype(np.float32)),
        },
    )
    return ds


def dump_parquet(obs: pd.DataFrame, meta: pd.DataFrame, out_dir: str | Path, tag: str) -> dict[str, Path]:
    """Write `obs` and `meta` to `<out_dir>/ghcn_<tag>_obs.parquet` etc."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs_path = out / f"ghcn_{tag}_obs.parquet"
    meta_path = out / f"ghcn_{tag}_meta.parquet"
    obs.to_parquet(obs_path, index=False)
    meta.to_parquet(meta_path, index=False)
    return {"obs": obs_path, "meta": meta_path}


# ----- Backward-compatible thin wrappers expected elsewhere -------------------

def load_ghcn(path: str | Path, var: str = "tas") -> xr.Dataset:
    """Load a previously-prepared GHCN store (Parquet → xarray)."""
    p = Path(path)
    if p.suffix == ".parquet":
        obs = pd.read_parquet(p)
        return to_xarray_station(obs)
    if p.suffix in {".nc", ".nc4"}:
        return xr.open_dataset(p)
    if p.suffix == ".zarr" or p.is_dir():
        return xr.open_zarr(p)
    raise ValueError(f"unsupported GHCN store: {p}")


def subset_region(ds: xr.Dataset, region: Region, min_years: int = 20) -> xr.Dataset:
    """Methods §1.3: bbox + minimum record length on station xarray."""
    in_box = (
        (ds.lon >= region.lon_min) & (ds.lon <= region.lon_max)
        & (ds.lat >= region.lat_min) & (ds.lat <= region.lat_max)
    )
    ds = ds.where(in_box, drop=True)
    n_obs = ds["value"].notnull().sum("time")
    return ds.where(n_obs >= min_years * 12, drop=True)


def time_split(ds: xr.Dataset, cal: int, msel: int, test: int) -> dict[str, xr.Dataset]:
    """Methods §1.4: split each station's record into cal / msel / test windows."""
    raise NotImplementedError("time_split: implement in §1.4")
