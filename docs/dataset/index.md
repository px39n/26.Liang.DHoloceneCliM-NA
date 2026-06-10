# Internal Data

This dataset is shared internally among collaborators. It contains all inputs needed to re-run the full PCR downscaling pipeline from scratch, as well as pre-computed calibration products for reproducing paper figures without re-running the pipeline.

## Download

1. **Download the shared dataset** from [OneDrive](https://1drv.ms/u/c/a388ead2000977f1/IQDeWnrv0C7VT5tWfo-QEitvAUQZiAsAlZg-v-CVl334qfU?e=ddtVlf) — file: `DPastCliM-NA_share.zip` (1.4 GB)

2. **Unzip** to a local directory (e.g. `D:\Dataset\DPastCliM-NA\`). The zip preserves the internal directory structure — do not rename or rearrange.

3. **Download TraCE-21k II** (required only for re-running the pipeline, not for figure reproduction):
    - Go to [NCAR Climate Data Gateway — TraCE-21ka](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm3.trace.html) (free registration required)
    - Download **TREFHT** (monthly) → rename to `TraCE-21K-II.monthly.TREFHT.nc`
    - Download **PRECT** (monthly) → rename to `TraCE-21K-II.monthly.PRECT.nc`
    - Place both files in `DATA_DIR/TraCE21k/`

4. **Set the environment variable `DPASTCLIM_DATA_DIR`** to the directory you unzipped to (e.g. `D:\Dataset\DPastCliM-NA`). All scripts — both `Paper_Code/` and `related_scripts/` — read from this single variable. If unset, the default `D:\Dataset\DPastCliM-NA` is used.

    ```powershell
    # PowerShell (current session)
    $env:DPASTCLIM_DATA_DIR = "D:\Dataset\DPastCliM-NA"

    # Or set permanently via System Environment Variables
    [System.Environment]::SetEnvironmentVariable("DPASTCLIM_DATA_DIR", "D:\Dataset\DPastCliM-NA", "User")
    ```

After these steps your directory should look like:

```
DATA_DIR/
├── GHCN/interim/          (from zip)
├── TraCE21k/              (from step 3, only needed for pipeline)
│   ├── TraCE-21K-II.monthly.TREFHT.nc
│   └── TraCE-21K-II.monthly.PRECT.nc
├── static/                (from zip)
└── interim/trace21k/      (from zip)
```

## Directory

```
DATA_DIR/
├── GHCN/interim/                    # GHCN-m v4 preprocessed observations
├── TraCE21k/                        # CCSM3 TraCE-21k II transient simulation
├── static/                          # Grid definitions and masks
├── interim/trace21k/
│   ├── models/                      # Trained statistical models
│   ├── station_cal/                 # Station-level calibration predictions
│   └── grid_cal/                    # Gridded calibration products
```

---

## File reference

### `GHCN/interim/`

Preprocessed GHCN-m v4 monthly climate observations for North America. Converted from raw fixed-width text files into Parquet format by `prepare_ghcn.py`. These are the **ground-truth observations** used for model training and validation.

| File | Description | Size | Producer | Consumers |
|------|-------------|------|----------|-----------|
| `ghcn_tas_obs.parquet` | Monthly temperature observations (station × time, sparse) | 15 MB | `prepare_ghcn.py` | `run_calibrate.py`, `run_cal_pi.py`, `run_grid_cal.py`, `F2`, `F4` |
| `ghcn_tas_meta.parquet` | Station metadata for tas (ID, lon, lat, elev, record length) | 1 MB | `prepare_ghcn.py` | `run_calibrate.py`, `F1` |
| `ghcn_pr_obs.parquet` | Monthly precipitation observations | 25 MB | `prepare_ghcn.py` | `run_calibrate.py`, `run_cal_pi.py`, `run_grid_cal.py`, `F2`, `F4` |
| `ghcn_pr_meta.parquet` | Station metadata for pr | 1 MB | `prepare_ghcn.py` | `run_calibrate.py`, `F1` |

---

### `TraCE21k/`

CCSM3 TraCE-21k II transient climate simulation — the **ESM forcing input** that drives the downscaling. Covers the last 22,000 years at monthly resolution on a ~3.75° global grid. **Not included in the zip** due to size (~10 GB total); download separately following the instructions below.

| File | Description | Size | Producer | Consumers |
|------|-------------|------|----------|-----------|
| `TraCE-21K-II.monthly.TREFHT.nc` | Surface temperature (K), 22 ka transient | 5 GB | External (NCAR) | `run_calibrate.py`, `run_cal_pi.py`, `run_project.py`, `run_grid_cal.py` |
| `TraCE-21K-II.monthly.PRECT.nc` | Total precipitation rate (m/s), 22 ka transient | 5 GB | External (NCAR) | `run_calibrate.py`, `run_cal_pi.py`, `run_project.py`, `run_grid_cal.py` |

#### TraCE-21k download

1. Go to the [NCAR Climate Data Gateway — TraCE-21ka](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm3.trace.html) (requires free registration)
2. Navigate to **TraCE-21k II** → **Monthly** output
3. Download the following two variables:
   - `TREFHT` (Reference Height Temperature) → save as `TraCE-21K-II.monthly.TREFHT.nc`
   - `PRECT` (Total Precipitation) → save as `TraCE-21K-II.monthly.PRECT.nc`
4. Place both files in `DATA_DIR/TraCE21k/`:

```
DATA_DIR/
└── TraCE21k/
    ├── TraCE-21K-II.monthly.TREFHT.nc   ← must be this exact filename
    └── TraCE-21K-II.monthly.PRECT.nc    ← must be this exact filename
```

These files are only needed if you want to re-run the production pipeline (Steps 1–5). For reproducing paper figures from pre-computed products, TraCE-21k is **not required**.

---

### `static/`

Static grid definitions and masks used during the gridding step.

| File | Description | Size | Producer | Consumers |
|------|-------------|------|----------|-----------|
| `landmask_NA_020.nc` | Boolean land mask on 0.20° Albers Equal-Area grid (301×601) | 200 KB | `make_static.py` | `run_grid_cal.py`, `F3`, `F5`, `F6` |

---

### `interim/trace21k/models/`

Trained statistical models from the calibration pipeline. PCR models and the split file are **shared** (small, essential). SEM and ARMA models are **not shared** because they are large and regeneratable.

| File | Description | Size | Shared? | Producer | Consumers |
|------|-------------|------|---------|----------|-----------|
| `split_calibration.pkl` | Cal/val/test year assignment (seed=2026, 76/38/11 yr) | 0.5 MB | Yes | `generate_split.py` | `run_calibrate.py`, `run_grid_cal.py`, `F2` |
| `pcr_models_tas.pkl` | 12 monthly PCR models for tas (EOFs, betas, mu_gO, sigma2) | 10 MB | Yes | `run_calibrate.py` | `run_cal_pi.py`, `run_project.py` |
| `pcr_models_pr.pkl` | 12 monthly PCR models for pr | 14 MB | Yes | `run_calibrate.py` | `run_cal_pi.py`, `run_project.py` |
| `sem_model_{var}_m{01-12}.pkl` | Per-month SEM noise model (lambda, W, sigma2). Regenerate with `run_cal_pi.py` (~2 hrs) | 5–14 GB | **No** | `run_cal_pi.py` | `run_project.py` |
| `arma_model_{var}_m{01-12}.pkl` | Per-month ARMA(1,1) coefficients per station | 16 MB | **No** | `run_cal_pi.py` | `run_project.py` |

---

### `interim/trace21k/station_cal/`

Station-level predictions for the calibration period (1875–1999). These are the **main outputs consumed by Paper_Code** for validation figures and tables.

| File | Description | Size | Producer | Consumers |
|------|-------------|------|----------|-----------|
| `recon_cal_tas.csv` | Deterministic prediction + 95% PI + realization, monthly, all cal stations | 770 MB | `run_project.py --cal-only` | `run_grid_cal.py`, `F4` |
| `recon_cal_pr.csv` | Same for precipitation | 1.25 GB | `run_project.py --cal-only` | `run_grid_cal.py`, `F4` |
| `test_cache_tas.parquet` | Cached test-set observations + predictions for fast metric computation | 50 MB | `_build_test_cache.py` | `T1`, `T2`, `T45` |
| `test_cache_pr.parquet` | Same for precipitation | 80 MB | `_build_test_cache.py` | `T1`, `T2`, `T45` |

---

### `interim/trace21k/grid_cal/`

Gridded fields on the 0.20° Albers Equal-Area North America grid (301 lat × 601 lon). These are the inputs for spatial validation figures (bias maps, mean fields).

| File | Description | Size | Producer | Consumers |
|------|-------------|------|----------|-----------|
| `grid_pcr_raw_tas.nc` | PCR predictions gridded via Sibson interpolation, **no** variance correction | 308 MB | `run_grid_cal.py` | `F3`, `F5`, `TC1` |
| `grid_pcr_raw_pr.nc` | Same for precipitation | 308 MB | `run_grid_cal.py` | `F3`, `F6`, `TC1` |
| `grid_esm_cal_tas.nc` | TraCE-21k regridded to 0.20° target (ESM baseline for comparison) | 30 MB | `run_grid_cal.py` | `F5`, `TC1` |
| `grid_esm_cal_pr.nc` | Same for precipitation | 33 MB | `run_grid_cal.py` | `F6`, `TC1` |
| `grid_obs_test_tas.nc` | GHCN test-only stations gridded (test years × 12 months) | 22 MB | `run_grid_cal.py` | `F5`, `TC1` |
| `grid_obs_test_pr.nc` | Same for precipitation | 23 MB | `run_grid_cal.py` | `F6`, `TC1` |
| `grid_pcr_cal_tas.nc` | PCR with 30-yr variance correction. **Evaluation only** — final figures use `grid_pcr_raw_*` instead | 200 MB | `run_varcorr_cal.py` | `TC1` only |
| `grid_pcr_cal_pr.nc` | Same for precipitation. **Evaluation only** | 199 MB | `run_varcorr_cal.py` | `TC1` only |

