# DHoloceneCliM-NA

**Downscaled Holocene Climate for North America**

A Python reimplementation and extension of [DPastCliM-NA](https://github.com/Env-an-Stat-group/25.Guaita.DPastCliM-NA) (Guaita et al.), producing monthly-resolution temperature and precipitation reconstructions spanning the last 22,000 years over North America using PCR downscaling from transient ESM simulations.

## Documentation

Full documentation (setup, data, pipeline, figure reproduction) is available at:

**<https://px39n.github.io/26.Liang.DHoloceneCliM-NA/>**

## Quick start

```bash
git clone --recurse-submodules https://github.com/px39n/26.Liang.DHoloceneCliM-NA.git
cd 26.Liang.DHoloceneCliM-NA

conda create -n caz python=3.11
conda activate caz
pip install -e ".[dev]"
```

Set the dataset path via environment variable:

```bash
export DPASTCLIM_DATA_DIR="/path/to/DPastCliM-NA"   # Linux/macOS
$env:DPASTCLIM_DATA_DIR = "D:\Dataset\DPastCliM-NA" # PowerShell
```

See the [documentation](https://px39n.github.io/26.Liang.DHoloceneCliM-NA/) for data download, pipeline steps, and figure reproduction.

## License

See repository for license details.
