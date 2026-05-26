"""Runtime config: paths, region, grid, calibration windows."""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field


class Region(BaseModel):
    """Geographic bounding box (lon in [-180, 180], lat in [-90, 90])."""
    lon_min: float = -170.0
    lon_max: float = -50.0
    lat_min: float = 15.0
    lat_max: float = 85.0


class Grid(BaseModel):
    """Target downscaling grid."""
    resolution_deg: float = 0.20


class Calibration(BaseModel):
    """Time splits for fitting / model selection / testing (years)."""
    cal_years: int = 75
    msel_years: int = 50
    test_years: int = 15
    min_record_years: int = 20


class Modes(BaseModel):
    """PCA mode selection thresholds."""
    init_ev_pct: float = 10.0       # initially keep modes >= 10% explained var
    candidate_ev_pct: float = 0.001  # try modes with >= 0.001%
    rmse_drop_pct: float = 1.0       # accept if avg RMSE drops by >= 1%
    n_realizations: int = 50
    max_modes: int = 16


class Compress(BaseModel):
    """Output compression policy per product tier."""
    standard_codec: str = "zstd"      # lossless: blosc/zstd
    standard_level: int = 5
    fast_codec: str = "sz3"           # lossy
    fast_rel_err: float = 0.01        # ~1% relative error -> CR ~100x
    int16_scale_tas: float = 0.01     # 0.01 degC
    int16_scale_pr: float = 0.01      # 0.01 mm/day


class Paths(BaseModel):
    """All on-disk locations."""
    project_root: Path = Path(__file__).resolve().parents[2]
    raw: Path = project_root / "data" / "raw"
    interim: Path = project_root / "data" / "interim"
    processed: Path = project_root / "data" / "processed"
    output: Path = project_root / "data" / "output"


class Config(BaseModel):
    region: Region = Field(default_factory=Region)
    grid: Grid = Field(default_factory=Grid)
    calibration: Calibration = Field(default_factory=Calibration)
    modes: Modes = Field(default_factory=Modes)
    compress: Compress = Field(default_factory=Compress)
    paths: Paths = Field(default_factory=Paths)


DEFAULT = Config()
