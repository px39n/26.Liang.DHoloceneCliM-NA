"""Diff Python parquet output vs MATLAB .mat output for GHCN tas.

Run after both:
    matlab -batch "run('tests/matlab_compare/run_ghcn_tas.m')"
    python  related_scripts/prepare_ghcn.py --skip-pr
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PARQ_OBS = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_tas_obs.parquet")
ML_OBS   = Path(r"D:\Dataset\DPastCliM-NA\GHCN\interim\matlab_ghcn_tas_obs.parquet")


def main():
    if not PARQ_OBS.exists():
        sys.exit(f"missing python output: {PARQ_OBS}")
    if not ML_OBS.exists():
        sys.exit(f"missing matlab output: {ML_OBS}\nrun:\n  matlab -batch \"run('tests/matlab_compare/run_ghcn_tas.m')\"")

    py = pd.read_parquet(PARQ_OBS)
    print(f"[py]    rows={len(py):,}  stations={py['ID'].nunique():,}")
    print(f"[py]    value mean={py['value'].mean():.4f}  min={py['value'].min():.4f}  max={py['value'].max():.4f}")

    ml = pd.read_parquet(ML_OBS)
    # MATLAB cols: ID, month_since_0CE, Value, lat, lon, elev, year, month
    ml = ml.rename(columns={"Value": "value"})
    ml["ID"] = ml["ID"].astype(str)
    print(f"[ml]    rows={len(ml):,}  stations={ml['ID'].nunique():,}")
    print(f"[ml]    value mean={ml['value'].mean():.4f}  min={ml['value'].min():.4f}  max={ml['value'].max():.4f}")

    # set diff
    py_ids = set(py["ID"].unique())
    ml_ids = set(ml["ID"].unique())
    print(f"\nshared stations: {len(py_ids & ml_ids):,}")
    print(f"  py-only: {len(py_ids - ml_ids):,}  ml-only: {len(ml_ids - py_ids):,}")

    # row-level merge
    common_keys = ["ID", "year", "month"]
    merged = py.merge(ml, on=common_keys, how="inner", suffixes=("_py", "_ml"))
    print(f"\nmatched rows: {len(merged):,}")
    if len(merged):
        d = merged["value_py"].astype(np.float64) - merged["value_ml"].astype(np.float64)
        print(f"|value_py - value_ml|: max={np.nanmax(np.abs(d)):.6g}  mean={np.nanmean(np.abs(d)):.6g}")
        print(f"  exact matches: {(d == 0).sum():,}/{len(d):,}")
        for tol in (1e-6, 1e-3, 1e-2):
            print(f"  within {tol}: {(np.abs(d) <= tol).sum():,}")

    # rows in py but not in ml (and vice versa)
    only_py = len(py) - len(merged)
    only_ml = len(ml) - len(merged)
    print(f"\npy-only rows: {only_py:,}   ml-only rows: {only_ml:,}")


if __name__ == "__main__":
    main()
