# Reproducing figures and tables

All figure/table scripts live in `Paper_Code/main/`. Each script is self-contained — it reads data from `DATA_DIR` and outputs to `Results/`.

## Setup

1. Ensure the [internal dataset](dataset/index.md) is downloaded and placed under `DATA_DIR`
2. Edit the path in `Paper_Code/commons/_common.py`:

```python
DATA_DIR = Path(r"/path/to/your/DPastCliM-NA")
```

3. Run any script:

```bash
cd Paper_Code/main
python F1_station_coverage.py
```

Outputs are saved to `Results/main/` (or `Results/supplementary/`) and automatically copied to `latex_paper/Figures/`.

## Main paper

| Script | Output | Description |
|--------|--------|-------------|
| `F1_station_coverage.py` | Fig. 1 | GHCN station spatial coverage map |
| `F2_data_split.py` | Fig. 2 | Cal/val/test temporal split + PCR mode count |
| `F3_mean_fields.py` | Fig. 3 | Predict-window mean fields, PI width, and anomalies |
| `F4_station_timeseries.py` | Fig. 4 | Representative station time series with 95% PI |
| `F5_bias_map_tas.py` | Fig. 5 | Temperature bias maps: mean, P10, P90 (PCR-Obs vs ESM-Obs) |
| `F6_bias_map_pr.py` | Fig. 6 | Precipitation bias maps: mean, P10, P90 |
| `T1_timestep_metrics.py` | Table 1 | Per-timestep validation (RMSE, R², bias) |
| `T2_station_metrics.py` | Table 2 | Per-station validation metrics |
| `T45_mss.py` | Table 4–5 | Model skill score (MSS) summary |
| `TC1_vc_eval.py` | Table C1 | Variance correction evaluation |

## Appendix (MPI-ESM)

| Script | Output | Description |
|--------|--------|-------------|
| `FA1_station_coverage_mpi.py` | Fig. A1 | Station coverage (MPI-ESM) |
| `FA2_data_split_mpi.py` | Fig. A2 | Data split (MPI-ESM) |
| `FA3_mean_fields_mpi.py` | Fig. A3 | Predict-window fields (MPI-ESM) |
| `FA4_station_timeseries_mpi.py` | Fig. A4 | Station time series (MPI-ESM) |
| `FA5_bias_map_tas_mpi.py` | Fig. A5 | Temperature bias maps (MPI-ESM) |
| `FA6_bias_map_pr_mpi.py` | Fig. A6 | Precipitation bias maps (MPI-ESM) |

## Data requirements

Each script reads specific files from `DATA_DIR`. See [Internal Data > File reference](dataset/index.md#file-reference) for the full mapping of which script consumes which files.
