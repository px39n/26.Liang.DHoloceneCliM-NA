"""Convert raw GHCN-m v4 archives into tidy Parquet (tas + pr).

Mirrors the front half of `25.Guaita.DPastCliM-NA/preprocessing/GHCNm.m`:
  1. parse station metadata + monthly values
  2. region bbox filter (North America by default)
  3. time filter (1850-2014 by default)
  4. drop stations with < 20 years of record
  5. write `<out>/ghcn_tas_obs.parquet`, `ghcn_tas_meta.parquet`,
     `ghcn_pr_obs.parquet`, `ghcn_pr_meta.parquet`

Run:
  conda run -n caz python related_scripts/prepare_ghcn.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from caz.config import Region
from caz.io.ghcn import (
    dump_parquet,
    filter_min_record,
    filter_time,
    load_ghcn_pr,
    load_ghcn_tas,
    subset_region_df,
)


DEFAULTS = dict(
    inv_tas=r"D:\Dataset\DPastCliM-NA\GHCN\ghcnm.v4.0.1.20260512\ghcnm.tavg.v4.0.1.20260512.qcf.inv",
    dat_tas=r"D:\Dataset\DPastCliM-NA\GHCN\ghcnm.v4.0.1.20260512\ghcnm.tavg.v4.0.1.20260512.qcf.dat",
    csv_pr=r"D:\Dataset\DPastCliM-NA\GHCN\prcp_csv",
    out=r"D:\Dataset\DPastCliM-NA\GHCN\interim",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inv-tas", default=DEFAULTS["inv_tas"])
    p.add_argument("--dat-tas", default=DEFAULTS["dat_tas"])
    p.add_argument("--csv-pr", default=DEFAULTS["csv_pr"])
    p.add_argument("--out", default=DEFAULTS["out"])
    p.add_argument("--lat-min", type=float, default=7.0)
    p.add_argument("--lat-max", type=float, default=75.0)
    p.add_argument("--lon-min", type=float, default=-180.0)
    p.add_argument("--lon-max", type=float, default=-50.0)
    p.add_argument("--year-min", type=int, default=1850)
    p.add_argument("--year-max", type=int, default=2014)
    p.add_argument("--min-years", type=int, default=20)
    p.add_argument("--skip-tas", action="store_true")
    p.add_argument("--skip-pr", action="store_true")
    args = p.parse_args()

    region = Region(
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
    )
    out_dir = Path(args.out)

    if not args.skip_tas:
        t0 = time.time()
        print(f"[tas] reading {args.dat_tas}")
        obs, meta = load_ghcn_tas(args.inv_tas, args.dat_tas)
        print(f"[tas] raw: {len(obs):,} rows / {meta.shape[0]:,} stations")
        obs = subset_region_df(obs, region)
        print(f"[tas] after bbox: {len(obs):,} rows / {obs['ID'].nunique():,} stations")
        obs = filter_time(obs, args.year_min, args.year_max)
        print(f"[tas] after time {args.year_min}-{args.year_max}: {len(obs):,} rows")
        obs = filter_min_record(obs, args.min_years)
        print(f"[tas] after min-record {args.min_years}y: {len(obs):,} rows / "
              f"{obs['ID'].nunique():,} stations")
        meta = meta[meta["ID"].isin(obs["ID"].unique())].reset_index(drop=True)
        paths = dump_parquet(obs, meta, out_dir, "tas")
        print(f"[tas] wrote {paths['obs']} / {paths['meta']} in {time.time()-t0:.1f}s")

    if not args.skip_pr:
        t0 = time.time()
        print(f"[pr]  scanning {args.csv_pr}")
        obs, meta = load_ghcn_pr(None, args.csv_pr, region=region)
        print(f"[pr]  raw in-bbox: {len(obs):,} rows / {meta.shape[0]:,} stations")
        obs = filter_time(obs, args.year_min, args.year_max)
        print(f"[pr]  after time: {len(obs):,} rows")
        obs = filter_min_record(obs, args.min_years)
        print(f"[pr]  after min-record: {len(obs):,} rows / {obs['ID'].nunique():,} stations")
        meta = meta[meta["ID"].isin(obs["ID"].unique())].reset_index(drop=True)
        paths = dump_parquet(obs, meta, out_dir, "pr")
        print(f"[pr]  wrote {paths['obs']} / {paths['meta']} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
