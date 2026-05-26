"""Benchmark Python variance correction at production scale."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from caz.gridding import variance_correction, _movmean_2d, _movstd_2d

def benchmark(n_cells, n_time, window=30, label=""):
    print(f"--- {label} ({n_cells}x{n_time}) ---")
    np.random.seed(42)
    pcr = np.random.randn(n_cells, n_time)
    esm = np.random.randn(n_cells, n_time)

    # movmean
    t0 = time.perf_counter()
    pcr_mm = _movmean_2d(pcr, window)
    t_mm = time.perf_counter() - t0
    print(f"  movmean:  {t_mm:.3f} s")

    # movstd
    t0 = time.perf_counter()
    pcr_ms = _movstd_2d(pcr - pcr_mm, window)
    t_ms = time.perf_counter() - t0
    print(f"  movstd:   {t_ms:.3f} s")

    # full variance correction
    t0 = time.perf_counter()
    adj = variance_correction(pcr, esm, window=window)
    t_full = time.perf_counter() - t0
    print(f"  Full varcorr: {t_full:.3f} s")
    ram_gb = n_cells * n_time * 8 * 8 / 1e9
    print(f"  Peak RAM est: {ram_gb:.1f} GB\n")
    return t_mm, t_ms, t_full

if __name__ == "__main__":
    print("=== Variance Correction Benchmark (Python, cumsum-optimized) ===\n")

    results = {}
    for label, nc, nt in [
        ("Small", 50, 200),
        ("Medium", 18030, 1500),
    ]:
        results[label] = benchmark(nc, nt, label=label)

    # Full grid only if enough memory
    try:
        results["Full"] = benchmark(180901, 1500, label="Full")
    except MemoryError:
        print("--- Full (180901x1500) --- SKIPPED: MemoryError\n")

    print("Done.")
