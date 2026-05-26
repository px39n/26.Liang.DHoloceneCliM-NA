"""Build NA 0.20-deg downscaling grid + land mask, save to D:\\Dataset\\DPastCliM-NA\\static.

Land mask: from Natural Earth 10m countries shapefile (rasterize onto 0.20 grid).
Free, no account, ~6 MB shapefile.
"""
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import xarray as xr

OUT = Path(r"D:\Dataset\DPastCliM-NA\static")
OUT.mkdir(parents=True, exist_ok=True)

# --- 1. Build target grid (NA 0.20 deg) ---
# Region: lon [-170, -50], lat [15, 85]; 0.20 deg cells
res = 0.20
lon = np.arange(-170 + res / 2, -50, res)        # 600 cells
lat = np.arange(15 + res / 2, 85, res)           # 350 cells
print(f"grid: {lat.size} lat x {lon.size} lon = {lat.size * lon.size} cells")

# --- 2. Build land mask via Natural Earth ---
ne_zip = OUT / "ne_10m_land.zip"
if not ne_zip.exists():
    print("downloading Natural Earth 10m land...")
    url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    urllib.request.urlretrieve(url, ne_zip)
    with zipfile.ZipFile(ne_zip) as z:
        z.extractall(OUT / "ne_10m_land")

shp = next((OUT / "ne_10m_land").glob("*.shp"))
print(f"shapefile: {shp.name}")

# Rasterize using rasterio + geopandas
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin

gdf = gpd.read_file(shp)
transform = from_origin(west=-170, north=85, xsize=res, ysize=res)
mask = rasterize(
    ((geom, 1) for geom in gdf.geometry),
    out_shape=(lat.size, lon.size),
    transform=transform,
    fill=0,
    dtype="uint8",
)
mask = np.flipud(mask).astype(bool)              # rasterize is top-down; we want lat-ascending
print(f"land cells: {mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")

# --- 3. Write Zarr + NetCDF ---
ds = xr.Dataset(
    {"mask": (("lat", "lon"), mask)},
    coords={"lat": ("lat", lat.astype("float32")),
            "lon": ("lon", lon.astype("float32"))},
    attrs={
        "title": "NA 0.20deg land mask",
        "source": "Natural Earth 10m land",
        "lon_range": "[-170, -50]",
        "lat_range": "[15, 85]",
        "resolution_deg": res,
    },
)
out_nc = OUT / "landmask_NA_020.nc"
ds.to_netcdf(out_nc)
print(f"written: {out_nc} ({out_nc.stat().st_size/1024:.1f} KB)")
