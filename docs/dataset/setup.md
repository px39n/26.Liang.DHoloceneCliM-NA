# Dataset setup

## 1. Configure `DATA_DIR`

All paths resolve from a single dataset root. Change **one line** in each location you use:

### Paper figures and tables

`Paper_Code/commons/_common.py`:

```python
DATA_DIR = Path(r"D:\Dataset\DPastCliM-NA")  # ← change to your path
```

Derived paths (do not edit unless restructuring):

| Variable | Path under `DATA_DIR` |
|----------|------------------------|
| `GHCN_DIR` | `GHCN/interim/` |
| `TRACE_DIR` | `TraCE21k/` |
| `MODELS_DIR` | `interim/trace21k/models/` |
| `STATION_CAL` | `interim/trace21k/station_cal/` |
| `GRID_CAL` | `interim/trace21k/grid_cal/` |

### Production scripts

Each script in `related_scripts/` defines `DATA_ROOT` near the top:

```python
DATA_ROOT = Path(r"D:\Dataset\DPastCliM-NA")  # ← change to your path
```

Files with `DATA_ROOT`: `prepare_ghcn.py`, `make_static.py`, `generate_split.py`, `run_calibrate.py`, `run_cal_pi.py`, `run_project.py`, `run_grid_cal.py`, `run_varcorr_cal.py`, and related drivers.

## 2. Conda environment

```bash
conda create -n caz python=3.12
conda activate caz
cd d:\OneDrive\Code\25.DPastCliM-NA
pip install -e ".[dev]"
```

## 3. What to download vs generate

| Data | Action |
|------|--------|
| **TraCE-21k NetCDF** (~10 GB) | Download TraCE-21k II monthly `TREFHT` and `PRECT` → place in `TraCE21k/` |
| **Raw GHCN daily** | Download from NOAA GHCN → run `prepare_ghcn.py` |
| **Land mask** | Run `make_static.py` |
| **Split + PCR models** | Run `generate_split.py`, then `run_calibrate.py` |
| **SEM / ARMA models** (~19 GB) | Run `run_cal_pi.py` locally — **regeneratable, not shared** |
| **Station + grid cal outputs** | Run Steps 3–4 (`run_project.py --cal-only`, `run_grid_cal.py`) |
| **Test cache** | Run `Paper_Code/main/_build_test_cache.py` before table scripts |
| **Variance-corrected grids** | Optional: `run_varcorr_cal.py` (only for `TC1_vc_eval.py`) |
| **Full 22 ka output** | Not in shared dataset; run full `run_project.py` when needed |

## 4. Minimal directory after setup

```
D:\Dataset\DPastCliM-NA\
├── GHCN/interim/           # 4 parquet files
├── TraCE21k/               # 2 NetCDF files
├── static/                 # landmask_NA_020.nc
└── interim/trace21k/
    ├── models/
    ├── station_cal/
    └── grid_cal/
```

## 5. Verify paths

```bash
conda activate caz
python -c "from Paper_Code.commons._common import DATA_DIR; print(DATA_DIR.exists(), DATA_DIR)"
```

Expected: `True` and your configured path.

## 6. Build docs locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview this site.
