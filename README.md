# DHoloceneCliM-NA

**Downscaled Holocene Climate for North America**

A Python reimplementation and extension of the [DPastCliM-NA](https://github.com/Env-an-Stat-group/25.Guaita.DPastCliM-NA) statistical downscaling pipeline, producing monthly-resolution, station- and grid-level temperature and precipitation reconstructions spanning the last 22,000 years over North America.

## Method

The pipeline applies **Principal Component Regression (PCR)** to downscale coarse Earth System Model (ESM) transient simulations to the GHCN station network, with uncertainty quantification via:

1. **PCR calibration** on GHCN observations (1875-1999) with automatic PCA mode selection
2. **Spatial Error Model (SEM)** + **ARMA(1,1)** noise modeling for stochastic realizations
3. **Sibson natural-neighbor interpolation** to a 0.20 degree Albers Equal-Area grid
4. **Variance correction** (30-yr moving-window std matching) to align PCR variability with ESM

## Supported GCMs

| Model | Time span | Resolution | Reference |
|-------|-----------|------------|-----------|
| CCSM3 TraCE-21k II | 22 ka | 3.75 deg | He & Clark 2022 |
| MPI-ESM 1.2 CR | 26 ka | T31 (~3.75 deg) | Kapsch 2022 |

## Repository Structure

```
src/caz/                  Core Python package (PCR, SEM, gridding, IO)
Paper_Code/
  commons/_common.py      Shared paths and utilities
  main/                   All figure and table scripts
related_scripts/          Production pipeline drivers + MATLAB verification
tests/                    Smoke tests + MATLAB comparison
25.Guaita.DPastCliM-NA/   Reference MATLAB code (git submodule)
```

## Reproducing Figures and Tables

This section describes how to reproduce all paper figures and tables from pre-computed pipeline outputs. The production pipeline itself (Steps 0-5) is documented separately in `related_scripts/`.

### 1. Setup

```bash
git clone --recurse-submodules https://github.com/px39n/26.Liang.DHoloceneCliM-NA.git
cd 26.Liang.DHoloceneCliM-NA

conda create -n caz python=3.11
conda activate caz
pip install -e ".[dev]"
```

### 2. Data

Obtain the pipeline output data and place it under a single root directory. Edit the data path in `Paper_Code/commons/_common.py`:

```python
DATA_DIR = Path(r"/path/to/your/DPastCliM-NA")
```

Expected directory layout:

```
DPastCliM-NA/
├── GHCN/interim/                  GHCN station observations (Parquet)
├── TraCE21k/                      ESM input NetCDF files
├── static/landmask_NA_020.nc      North America land mask
├── interim/{gcm}/
│   ├── models/                    Trained PCR/SEM/ARMA models (.pkl)
│   ├── station_cal/               Station-level predictions (.csv)
│   └── grid_cal/                  Gridded fields (.nc)
│       ├── grid_obs_test_{var}.nc
│       ├── grid_pcr_raw_{var}.nc
│       ├── grid_pcr_test_{var}.nc
│       ├── grid_esm_cal_{var}.nc
│       └── grid_pcr_cal_{var}.nc
└── output/{gcm}/predict/          Predict-window outputs (.nc/.csv)
```

where `{gcm}` is `trace21k` or `mpi-esm-cr`, and `{var}` is `tas` or `pr`.

### 3. Generate Figures and Tables

Each script in `Paper_Code/main/` is self-contained and reads directly from the data directory above.

```bash
cd Paper_Code/main
python F1_station_coverage.py
python T1_timestep_metrics.py
# ... etc.
```

#### Main Paper

| Script | Output | Description |
|--------|--------|-------------|
| `F1_station_coverage.py` | Fig. 1 | GHCN station spatial coverage map |
| `F2_data_split.py` | Fig. 2 | Cal/val/test temporal split + PCR mode count |
| `F3_mean_fields.py` | Fig. 3 | Predict-window mean fields, PI width, and anomalies (TraCE) |
| `F4_station_timeseries.py` | Fig. 4 | Representative station time series with 95% PI |
| `F5_bias_map_tas.py` | Fig. 5 | Temperature bias maps: mean, P10, P90 (PCR-Obs vs ESM-Obs) |
| `F6_bias_map_pr.py` | Fig. 6 | Precipitation bias maps: mean, P10, P90 |
| `T1_timestep_metrics.py` | Table 1 | Per-timestep validation (RMSE, R2, bias) |
| `T2_station_metrics.py` | Table 2 | Per-station validation metrics |
| `T45_mss.py` | Table 4-5 | Model skill score (MSS) summary |
| `TC1_vc_eval.py` | Table C1 | Variance correction evaluation |

#### Appendix (MPI-ESM)

| Script | Output | Description |
|--------|--------|-------------|
| `FA1_station_coverage_mpi.py` | Fig. A1 | Station coverage (MPI-ESM) |
| `FA2_data_split_mpi.py` | Fig. A2 | Data split (MPI-ESM) |
| `FA3_mean_fields_mpi.py` | Fig. A3 | Predict-window fields (MPI-ESM) |
| `FA4_station_timeseries_mpi.py` | Fig. A4 | Station time series (MPI-ESM) |
| `FA5_bias_map_tas_mpi.py` | Fig. A5 | Temperature bias maps (MPI-ESM) |
| `FA6_bias_map_pr_mpi.py` | Fig. A6 | Precipitation bias maps (MPI-ESM) |

All figures are saved to `Results/main/` (or `Results/supplementary/` for appendix) and automatically copied to `latex_paper/Figures/`.

## Production Pipeline

The full production pipeline (training models, running projections, gridding) is driven by scripts in `related_scripts/`. See the docstrings in each script for usage:

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `prepare_ghcn.py`, `make_static.py`, `generate_split.py` | Data preparation |
| 1 | `run_calibrate.py --var {tas,pr}` | PCR model training |
| 2 | `run_cal_pi.py --var {tas,pr}` | SEM + ARMA noise model fitting |
| 3 | `run_project.py --var {tas,pr}` | Station-level prediction (cal or full 22 ka) |
| 4 | `run_grid_cal.py --var {tas,pr}` | Gridding (Sibson natural-neighbor) |
| 5 | `run_varcorr_cal.py --var {tas,pr}` | Variance correction |

## MATLAB Verification

Paired Python/MATLAB scripts in `related_scripts/verification/` verify numerical consistency at each pipeline step (A-F: data loading, PCR, prediction, gridding, variance correction, noise simulation).

## Related Work

This project builds on Guaita et al.'s MATLAB implementation ([25.Guaita.DPastCliM-NA](https://github.com/Env-an-Stat-group/25.Guaita.DPastCliM-NA)), extending it with multi-GCM support, Numba-accelerated gridding, and Python-native uncertainty quantification.

## License

See repository for license details.
