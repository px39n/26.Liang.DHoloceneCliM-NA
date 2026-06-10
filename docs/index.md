# DHoloceneCliM-NA

**Downscaled Holocene Climate for North America**

A Python reimplementation and extension of [DPastCliM-NA](https://github.com/Env-an-Stat-group/25.Guaita.DPastCliM-NA) (Guaita et al.), producing monthly-resolution temperature and precipitation reconstructions spanning the last 22,000 years over North America using PCR downscaling from transient ESM simulations.

## Data release

For the published dataset, see [TBD — Zenodo/repository link will be placed here].

For internal shared data (pipeline inputs + calibration products), see the [Internal Data](dataset/index.md) page.

## Quick start

### Reproduce paper figures

1. Download the internal dataset and set `DPASTCLIM_DATA_DIR` — see [Internal Data](dataset/index.md) for download steps
2. Activate the `caz` environment (see below)
3. Follow the [Figures](figures.md) page for script-by-script instructions

### Re-run the full pipeline from scratch

1. Complete the setup above, plus download TraCE-21k (see [Internal Data > TraCE-21k download](dataset/index.md#trace21k))
2. Follow the [Pipeline](pipeline.md) page for step-by-step instructions

## Environment

```bash
conda create -n caz python=3.11
conda activate caz
pip install -e ".[dev]"
```

## Supported GCMs

| Model | Time span | Resolution | Reference |
|-------|-----------|------------|-----------|
| CCSM3 TraCE-21k II | 22 ka | 3.75° | He & Clark 2022 |
| MPI-ESM 1.2 CR | 26 ka | T31 (~3.75°) | Kapsch 2022 |

## Repository structure

```
src/caz/                  Core Python package (PCR, SEM, gridding, IO)
Paper_Code/
  commons/_common.py      Shared paths and utilities
  main/                   All figure and table scripts
related_scripts/          Production pipeline drivers + MATLAB verification
tests/                    Smoke tests + MATLAB comparison
25.Guaita.DPastCliM-NA/   Reference MATLAB code (git submodule)
```
