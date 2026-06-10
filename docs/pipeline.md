# Production pipeline

This page describes the calibration production workflow — training PCR downscaling models on 1875–1999 observations and generating station-level and gridded products for paper figures.

All scripts live in `related_scripts/` and accept `--var tas` or `--var pr`. Run from the repo root or from inside `related_scripts/`.

Prerequisites: the `caz` environment activated, and `DPASTCLIM_DATA_DIR` set (see [Internal Data](dataset/index.md)).

---

## Overview

```
GHCN obs + TraCE-21k cal-period
        ↓
Step 0: Prepare inputs (GHCN parquet, land mask, split)
        ↓
Step 1: Train PCR models (12 monthly regressions)
        ↓
Step 2: Fit SEM + ARMA noise models on residuals
        ↓
Step 3: Predict cal-period (det + PI + realization)
        ↓
Step 4: Grid station predictions → 0.20° Albers grid
        ↓
Step 5: Variance correction (evaluation only)
        ↓
Step 7: Paper figures (Paper_Code/main/F*.py, T*.py)
```

---

## Step 0 — Prepare inputs

Three one-time scripts to prepare observations, grid mask, and train/val/test split.

| Sub-step | Script | Output | Description |
|----------|--------|--------|-------------|
| 0a | `prepare_ghcn.py` | `GHCN/interim/*.parquet` | Convert raw GHCN-m v4 fixed-width text into Parquet, with NA region filter |
| 0b | `make_static.py` | `static/landmask_NA_020.nc` | Build 0.20° Albers Equal-Area grid + land mask from Natural Earth shapefile |
| 0c | `generate_split.py` | `models/split_calibration.pkl` | Global random year assignment: 76 cal / 38 val / 11 test years (seed=2026) |

!!! note
    Step 0a requires raw GHCN text files (not included in shared data — the output parquets are already provided). Steps 0b and 0c only need to be re-run if you change grid resolution or split seed.

---

## Step 1 — PCR Calibration

```bash
python run_calibrate.py --var tas
python run_calibrate.py --var pr
```

Train 12 monthly Principal Component Regression models using calibration-period ESM (TraCE-21k 1875–1999) and GHCN observations.

- **Input**: `GHCN/interim/ghcn_{var}_obs.parquet` + TraCE-21k cal-period + `split_calibration.pkl`
- **Output**: `models/pcr_models_{var}.pkl` (12 `MonthPCRModel` objects with EOFs, betas, statistics)
- **Time**: ~2 min (tas), ~2 min (pr)
- **Method**: PCA on the ESM field → regress PC scores against station observations. Automatic mode selection (retain 95% variance).

---

## Step 2 — SEM + ARMA noise models

```bash
python run_cal_pi.py --var tas
python run_cal_pi.py --var pr
```

Fit spatial error model (SEM) and temporal ARMA(1,1) on calibration residuals, enabling stochastic realizations with spatially-correlated noise.

- **Input**: `pcr_models_{var}.pkl` + GHCN obs + TraCE cal-period
- **Output**: `sem_model_{var}_m{01-12}.pkl` (per-month SEM: λ, σ², W, threshold) + `arma_model_{var}_m{01-12}.pkl`
- **Time**: ~24 min (tas), ~93 min (pr)
- **Storage**: SEM models are large (5–15 GB) and **not included in shared data** — regenerate with this step

!!! warning
    This is the most time-consuming step. The SEM fitting involves iterative spatial weight matrix construction for ~7,000–13,000 stations per month.

---

## Step 3 — Cal-period prediction

```bash
python run_cal_predict.py --var tas
python run_cal_predict.py --var pr --real
```

Predict on cal-period ESM (1875–1999) at monthly resolution. Produces deterministic reconstruction, 95% prediction intervals (analytical), and optionally a noise realization (requires Step 2 models).

- **Input**: `pcr_models_{var}.pkl` + TraCE cal-period ESM + (optionally) SEM/ARMA models
- **Output**: `station_cal/recon_cal_{var}.csv` (long-format: station_id, lon, lat, year, month, value, pi_lo, pi_hi, value_real, split)
- **Time**: ~5 min (tas), ~8 min (pr)
- **PI method**: tas uses analytical ±1.96√MSE; pr uses lognormal quantiles

These CSVs are the **main inputs for figure reproduction** (Fig 4, Table 1–2).

---

## Step 4 — Gridding

```bash
python run_grid_cal.py --var tas
python run_grid_cal.py --var pr
```

Interpolate station-level predictions to the 0.20° Albers Equal-Area North America grid (301×601 cells) using Sibson natural-neighbor interpolation.

- **Input**: `recon_cal_{var}.csv` + `ghcn_{var}_obs.parquet` + `split_calibration.pkl` + TraCE-21k + `landmask_NA_020.nc`
- **Output**:
    - `grid_pcr_raw_{var}.nc` — PCR predictions gridded (1500 months, full cal period)
    - `grid_esm_cal_{var}.nc` — TraCE-21k regridded to target (for comparison)
    - `grid_obs_test_{var}.nc` — GHCN test-only stations gridded (test years)
- **Time**: ~21s (tas), ~72s (pr) — optimized via Numba batch weight matrix
- **Method**: Groups timesteps by active station set → builds Sibson weight matrix once per group → sparse matrix multiply. Ocean cells masked to NaN.

---

## Step 5 — Variance correction

```bash
python run_varcorr_cal.py --var tas
python run_varcorr_cal.py --var pr
```

Apply 30-year moving-window standard-deviation matching to align PCR gridded variability with the driving ESM.

- **Input**: `grid_pcr_raw_{var}.nc` + `grid_esm_cal_{var}.nc`
- **Output**: `grid_pcr_cal_{var}.nc` (variance-corrected), `grid_pcr_vc_test_{var}.nc` (test years only)
- **Time**: ~5.5s (Numba fused kernel)
- **Formula**: PCR_adj = μ(PCR) + (PCR − μ(PCR)) × σ(ESM) / σ(PCR), window = 360 months

!!! info
    Variance-corrected products are produced for **evaluation purposes only**. Final paper figures use the raw `grid_pcr_raw_*` fields (no correction applied in the submitted analysis).

---

## Step 7 — Paper figures

Once the above products are generated (or downloaded from the shared dataset), all paper figures and tables can be reproduced. See the [Figures](figures.md) page for the full list.

---

## Verification

Paired Python vs. MATLAB comparison tests live in `related_scripts/verification/` (steps A–F). These are one-off development tools confirming numerical equivalence with Guaita's original MATLAB code.
