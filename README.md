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
src/caz/              Core Python package (PCR, SEM, gridding, IO)
Paper_Code/           Figure and table generation scripts
related_scripts/      Production drivers + MATLAB verification
tests/                Smoke tests + MATLAB comparison
25.Guaita.DPastCliM-NA/   Reference MATLAB code (git submodule)
```

## Installation

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/px39n/26.Liang.DHoloceneCliM-NA.git
cd 26.Liang.DHoloceneCliM-NA

# Create conda environment
conda create -n caz python=3.11
conda activate caz

# Install package in editable mode
pip install -e ".[dev]"
```

## Data

Climate data (GHCN observations, ESM outputs, gridded products) is stored externally and not included in this repository. Configure the data path in `Paper_Code/commons/_common.py` before running scripts.

## Related Work

This project builds on Guaita et al.'s MATLAB implementation ([25.Guaita.DPastCliM-NA](https://github.com/Env-an-Stat-group/25.Guaita.DPastCliM-NA)), extending it with multi-GCM support, Numba-accelerated gridding, and Python-native uncertainty quantification.

## License

See repository for license details.
