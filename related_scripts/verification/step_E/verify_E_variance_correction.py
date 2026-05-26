"""Verify Python variance correction against MATLAB output.

Run the MATLAB script first to generate test_*.csv files, then run this script.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from caz.gridding import variance_correction, _movmean_2d, _movstd_2d

SCRIPT_DIR = Path(__file__).parent

def main():
    # Load shared test inputs
    pcr = np.loadtxt(SCRIPT_DIR / "test_pcr_input.csv", delimiter=",")
    esm = np.loadtxt(SCRIPT_DIR / "test_esm_input.csv", delimiter=",")
    print(f"PCR shape: {pcr.shape}, ESM shape: {esm.shape}")

    # Load MATLAB intermediates
    pcr_mm_mat = np.loadtxt(SCRIPT_DIR / "test_pcr_movmean_matlab.csv", delimiter=",")
    esm_mm_mat = np.loadtxt(SCRIPT_DIR / "test_esm_movmean_matlab.csv", delimiter=",")
    pcr_ms_mat = np.loadtxt(SCRIPT_DIR / "test_pcr_movstd_matlab.csv", delimiter=",")
    esm_ms_mat = np.loadtxt(SCRIPT_DIR / "test_esm_movstd_matlab.csv", delimiter=",")
    ratio_mat  = np.loadtxt(SCRIPT_DIR / "test_std_correction_matlab.csv", delimiter=",")
    adj_mat    = np.loadtxt(SCRIPT_DIR / "test_pcr_adjusted_matlab.csv", delimiter=",")

    window = 30

    # Step 1: Compare movmean
    pcr_mm_py = _movmean_2d(pcr, window)
    esm_mm_py = _movmean_2d(esm, window)
    print(f"\n--- movmean comparison ---")
    print(f"PCR movmean max|diff| = {np.nanmax(np.abs(pcr_mm_py - pcr_mm_mat)):.6e}")
    print(f"ESM movmean max|diff| = {np.nanmax(np.abs(esm_mm_py - esm_mm_mat)):.6e}")

    # Step 2: Compare movstd
    pcr_anom = pcr - pcr_mm_py
    esm_anom = esm - esm_mm_py
    pcr_ms_py = _movstd_2d(pcr_anom, window)
    esm_ms_py = _movstd_2d(esm_anom, window)
    print(f"\n--- movstd comparison ---")
    print(f"PCR movstd max|diff| = {np.nanmax(np.abs(pcr_ms_py - pcr_ms_mat)):.6e}")
    print(f"ESM movstd max|diff| = {np.nanmax(np.abs(esm_ms_py - esm_ms_mat)):.6e}")

    # Step 3: Compare ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_py = esm_ms_py / pcr_ms_py
    ratio_py[~np.isfinite(ratio_py)] = 1.0
    # MATLAB may produce Inf where pcr_std=0; handle consistently
    ratio_mat_clean = ratio_mat.copy()
    ratio_mat_clean[~np.isfinite(ratio_mat_clean)] = 1.0
    print(f"\n--- std_correction ratio comparison ---")
    print(f"Ratio max|diff| = {np.nanmax(np.abs(ratio_py - ratio_mat_clean)):.6e}")

    # Step 4: Compare final adjusted output
    adj_py = variance_correction(pcr, esm, window=window)
    adj_mat_clean = adj_mat.copy()
    adj_mat_clean[~np.isfinite(adj_mat_clean)] = np.nan
    print(f"\n--- Final adjusted PCR comparison ---")
    valid = np.isfinite(adj_mat_clean) & np.isfinite(adj_py)
    diff = np.abs(adj_py[valid] - adj_mat_clean[valid])
    print(f"max|diff| = {np.max(diff):.6e}")
    print(f"mean|diff| = {np.mean(diff):.6e}")
    print(f"Valid cells: {valid.sum()} / {adj_py.size}")

    if np.max(diff) < 1e-6:
        print("\n*** PASS: Variance correction matches MATLAB to machine precision ***")
    elif np.max(diff) < 1e-3:
        print(f"\n*** PASS: Variance correction matches MATLAB (max diff = {np.max(diff):.6e}) ***")
    else:
        print(f"\n*** FAIL: max diff = {np.max(diff):.6e} ***")


if __name__ == "__main__":
    main()
