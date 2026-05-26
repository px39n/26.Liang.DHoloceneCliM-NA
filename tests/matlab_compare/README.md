# Python vs MATLAB GHCN parsing parity test

Status: **Python correct, MATLAB (Caz `GHCNm.m`) has parsing bug.**

## Aggregate stats (both implementations agree)
| | Python | MATLAB |
|---|---|---|
| rows | 7,108,740 | 7,108,740 |
| stations | 11,326 | 11,326 |
| value min (°C) | -47.08 | -47.08 |
| value max (°C) | 42.68 | 42.68 |
| value mean (°C) | 10.20 | 10.20 |

Filtering (region NA bbox, time 1850-2014, ≥20 yr) identical.

## Row-level diff
* 6,808,163 (ID, year, month) keys matched between both outputs
* 300,577 keys exist only in Python output (with finite values)
* 300,577 keys exist only in MATLAB output
* **3,241,865 matched rows have value differences > 0.005 °C** (47.6 %)
* Median value diff ~0.02 °C but tails go up to ±27 °C.

## Root cause
GHCN-m v4 `.dat` is fixed-width:

```
columns 1-11  station ID
columns 12-15 year (4 digits)
columns 16-19 element ('TAVG')
columns 20-27 month 1   (value 5 chars + flags 3 chars)
columns 28-35 month 2
...
columns 108-115 month 12
```

Caz's MATLAB uses `textscan` with format `'%11s %4d %4s' + 12 * ' %5f %*3s'`
plus `'MultipleDelimsAsOne', true`. When a row contains `-9999` (the missing
sentinel, no leading space) mixed with valued months that have a single-space
prefix, the whitespace-collapsing tokeniser eats one too many delimiters and
the values shift one (sometimes more) months/years.

Sanity check — raw line for `BB000078954` year 1943:

```
BB0000789541943TAVG-9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999    2642  S 2562  S 2479  S
```

Only Oct/Nov/Dec are valid (26.42 / 25.62 / 24.79 °C).

* Python parquet at `BB000078954, 1943, Oct` = **26.42 °C** ✓
* MATLAB parquet at `BB000078954, 1944, Oct` = 26.42 °C ✗ (shifted +1 year)
* MATLAB has **no** 1943 row for this station (it was eaten).

The Python loader (`src/caz/io/ghcn.py`, function `_parse_dat_tas`) uses
explicit fixed byte positions (`start = 19 + m * 8`, width 5) and avoids the
issue.

## Reproduce
```powershell
# Python
conda run -n caz python related_scripts/prepare_ghcn.py --skip-pr

# MATLAB (uses vectorised min-record filter to avoid O(n^2))
& "C:\Program Files\MATLAB\R2026a\bin\matlab.exe" -batch "run('tests/matlab_compare/run_ghcn_tas.m')"

# Diff
conda run -n caz python tests/matlab_compare/compare_ghcn_tas.py
conda run -n caz python tests/matlab_compare/diff_analysis.py
```

## Decision
* Python parquet is the canonical GHCN tas dataset for this project.
* Notify Caz of the bug; consider switching her loader to `readmatrix` with
  explicit `'OutputType','char'` then manual fixed-width slicing.
