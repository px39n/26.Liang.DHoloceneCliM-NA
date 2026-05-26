import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from caz.gridding import _movmean_2d, _movstd_2d

SCRIPT_DIR = Path(__file__).parent

x = np.loadtxt(SCRIPT_DIR / "debug_x.csv", delimiter=",").reshape(1, -1)
y6_mat = np.loadtxt(SCRIPT_DIR / "debug_movmean_k6.csv", delimiter=",").reshape(1, -1)
y5_mat = np.loadtxt(SCRIPT_DIR / "debug_movmean_k5.csv", delimiter=",").reshape(1, -1)
x30 = np.loadtxt(SCRIPT_DIR / "debug_x30.csv", delimiter=",").reshape(1, -1)
y30_mat = np.loadtxt(SCRIPT_DIR / "debug_movmean_k30.csv", delimiter=",").reshape(1, -1)

# Test k=6
y6_py = _movmean_2d(x, 6)
print("k=6 comparison:")
print(f"  MATLAB: {y6_mat[0,:5]}")
print(f"  Python: {y6_py[0,:5]}")
print(f"  max|diff|: {np.max(np.abs(y6_py - y6_mat)):.6e}")

# Test k=5
y5_py = _movmean_2d(x, 5)
print(f"\nk=5 comparison:")
print(f"  MATLAB: {y5_mat[0,:5]}")
print(f"  Python: {y5_py[0,:5]}")
print(f"  max|diff|: {np.max(np.abs(y5_py - y5_mat)):.6e}")

# Test k=30
y30_py = _movmean_2d(x30, 30)
print(f"\nk=30 comparison:")
print(f"  MATLAB first 5: {y30_mat[0,:5]}")
print(f"  Python first 5: {y30_py[0,:5]}")
print(f"  MATLAB last 5: {y30_mat[0,-5:]}")
print(f"  Python last 5: {y30_py[0,-5:]}")
print(f"  max|diff|: {np.max(np.abs(y30_py - y30_mat)):.6e}")

# Show where biggest diff is
diffs = np.abs(y30_py[0] - y30_mat[0])
idx = np.argmax(diffs)
print(f"  Biggest diff at index {idx}: py={y30_py[0,idx]:.6f}, mat={y30_mat[0,idx]:.6f}")
