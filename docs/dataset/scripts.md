# Scripts ↔ Data mapping

## Production scripts (`related_scripts/`)

| Script | Inputs | Outputs |
|--------|--------|---------|
| `prepare_ghcn.py` | Raw GHCN `.dat` / `.csv` (external) | `GHCN/interim/ghcn_{tas,pr}_obs.parquet`, `ghcn_{tas,pr}_meta.parquet` |
| `make_static.py` | *(internal computation)* | `static/landmask_NA_020.nc` |
| `generate_split.py` | `GHCN/interim/*.parquet` | `interim/trace21k/models/split_calibration.pkl` |
| `run_calibrate.py --var {tas,pr}` | GHCN parquet + `TraCE21k/*.nc` + `split_calibration.pkl` | `interim/trace21k/models/pcr_models_{var}.pkl` |
| `run_cal_pi.py --var {tas,pr}` | `pcr_models_{var}.pkl` + GHCN obs + TraCE cal-period | `sem_model_{var}_m01–m12.pkl`, `arma_model_{var}_m01–m12.pkl` |
| `run_project.py --var {tas,pr} --cal-only` | PCR + SEM/ARMA models + TraCE cal-period | `interim/trace21k/station_cal/recon_cal_{var}.csv` |
| `run_grid_cal.py --var {tas,pr}` | `recon_cal_{var}.csv` + GHCN obs + `split_calibration.pkl` + TraCE + landmask | `grid_cal/grid_pcr_raw_{var}.nc`, `grid_esm_cal_{var}.nc`, `grid_obs_test_{var}.nc` |
| `run_varcorr_cal.py --var {tas,pr}` | `grid_pcr_raw_{var}.nc` + `grid_esm_cal_{var}.nc` | `grid_cal/grid_pcr_cal_{var}.nc` |

### Auxiliary production

| Script | Inputs | Outputs |
|--------|--------|---------|
| `run_project.py` *(no `--cal-only`)* | Same as cal-only | `output/trace21k/full/` (22 ka) |
| `run_cal_predict.py` | Models + TraCE predict windows | `output/trace21k/predict/` |

## Paper figure / table scripts (`Paper_Code/main/`)

| Script | Data consumed | Output |
|--------|---------------|--------|
| `F1_station_coverage.py` | `GHCN/interim/*_meta.parquet` | Fig. 1 |
| `F2_data_split.py` | `GHCN/interim/*_obs.parquet`, `split_calibration.pkl` | Fig. 2 |
| `F3_mean_fields.py` | `grid_cal/grid_pcr_raw_*.nc`, `grid_esm_cal_*.nc` | Fig. 3 |
| `F4_station_timeseries.py` | `station_cal/recon_cal_*.csv`, GHCN obs | Fig. 4 |
| `F5_bias_map_tas.py` | `grid_pcr_raw_tas.nc`, `grid_obs_test_tas.nc`, `grid_esm_cal_tas.nc` | Fig. 5 |
| `F6_bias_map_pr.py` | `grid_pcr_raw_pr.nc`, `grid_obs_test_pr.nc`, `grid_esm_cal_pr.nc` | Fig. 6 |
| `T1_timestep_metrics.py` | `station_cal/test_cache_*.parquet` | Table 1 |
| `T2_station_metrics.py` | `station_cal/test_cache_*.parquet` | Table 2 |
| `T45_mss.py` | `station_cal/test_cache_*.parquet` | Tables 4–5 |
| `TC1_vc_eval.py` | `grid_pcr_raw_*.nc`, `grid_pcr_cal_*.nc`, `grid_obs_test_*.nc` | Table C1 |
| `_build_test_cache.py` | GHCN obs + models + split | `station_cal/test_cache_*.parquet` |

## Reverse index: file → scripts

| File pattern | Producers | Consumers |
|--------------|-----------|-----------|
| `ghcn_*_obs.parquet` | `prepare_ghcn.py` | `generate_split.py`, `run_calibrate.py`, `run_cal_pi.py`, `run_project.py`, `run_grid_cal.py`, `F2`, `F4`, `_build_test_cache.py` |
| `ghcn_*_meta.parquet` | `prepare_ghcn.py` | `F1_station_coverage.py` |
| `TraCE21k/*.nc` | External | `run_calibrate.py`, `run_cal_pi.py`, `run_project.py`, `run_grid_cal.py` |
| `landmask_NA_020.nc` | `make_static.py` | `run_grid_cal.py` |
| `split_calibration.pkl` | `generate_split.py` | `run_calibrate.py`, `run_grid_cal.py`, `F2`, `_build_test_cache.py` |
| `pcr_models_{var}.pkl` | `run_calibrate.py` | `run_cal_pi.py`, `run_project.py`, `run_grid_cal.py`, `_build_test_cache.py` |
| `sem_model_{var}_m*.pkl` | `run_cal_pi.py` | `run_project.py` |
| `arma_model_{var}_m*.pkl` | `run_cal_pi.py` | `run_project.py` |
| `recon_cal_{var}.csv` | `run_project.py --cal-only` | `run_grid_cal.py`, `F4` |
| `test_cache_{var}.parquet` | `_build_test_cache.py` | `T1`, `T2`, `T45` |
| `grid_pcr_raw_{var}.nc` | `run_grid_cal.py` | `F3`, `F5`/`F6`, `TC1`, `run_varcorr_cal.py` |
| `grid_pcr_cal_{var}.nc` | `run_varcorr_cal.py` | `TC1` only |
| `grid_esm_cal_{var}.nc` | `run_grid_cal.py` | `F3`, `F5`/`F6`, `run_varcorr_cal.py`, `TC1` |
| `grid_obs_test_{var}.nc` | `run_grid_cal.py` | `F5`/`F6`, `TC1` |

## Execution order (minimum for paper figures)

```
prepare_ghcn.py → make_static.py → generate_split.py
→ run_calibrate.py (tas, pr)
→ run_cal_pi.py (tas, pr)          # optional if SEM/ARMA shared
→ run_project.py --cal-only (tas, pr)
→ run_grid_cal.py (tas, pr)
→ _build_test_cache.py
→ Paper_Code/main/F*.py, T*.py
```

Step 5 (`run_varcorr_cal.py`) is **only required** for `TC1_vc_eval.py`.
