"""Pack dataset for internal sharing.

Collects required input + calibration product files from the dataset directory
and creates a zip archive preserving the exact directory structure.

Usage:
    python pack_dataset.py [--data-dir D:\\Dataset\\DPastCliM-NA] [--output path.zip]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from _paths import DATA_ROOT

DEFAULT_DATA_DIR = DATA_ROOT
DEFAULT_OUTPUT = DATA_ROOT.parent / "DPastCliM-NA_share.zip"

# Paths relative to the dataset root.
INPUT_FILES = [
    "GHCN/interim/ghcn_tas_obs.parquet",
    "GHCN/interim/ghcn_tas_meta.parquet",
    "GHCN/interim/ghcn_pr_obs.parquet",
    "GHCN/interim/ghcn_pr_meta.parquet",
    "static/landmask_NA_020.nc",
]

TRACE_FILES = [
    "TraCE21k/TraCE-21K-II.monthly.TREFHT.nc",
    "TraCE21k/TraCE-21K-II.monthly.PRECT.nc",
]

OUTPUT_FILES = [
    "interim/trace21k/models/pcr_models_tas.pkl",
    "interim/trace21k/models/pcr_models_pr.pkl",
    "interim/trace21k/models/split_calibration.pkl",
    "interim/trace21k/station_cal/recon_cal_tas.csv",
    "interim/trace21k/station_cal/recon_cal_pr.csv",
    "interim/trace21k/station_cal/test_cache_tas.parquet",
    "interim/trace21k/station_cal/test_cache_pr.parquet",
    "interim/trace21k/grid_cal/grid_pcr_raw_tas.nc",
    "interim/trace21k/grid_cal/grid_pcr_raw_pr.nc",
    "interim/trace21k/grid_cal/grid_esm_cal_tas.nc",
    "interim/trace21k/grid_cal/grid_esm_cal_pr.nc",
    "interim/trace21k/grid_cal/grid_obs_test_tas.nc",
    "interim/trace21k/grid_cal/grid_obs_test_pr.nc",
]

# Documented exclusions (not packed; explicit allow-list above).
EXCLUDED_PATTERNS = [
    "GHCN/ghcnm.v4.0.1.*/**",
    "GHCN/GHCN_prcp_csv/**",
    "interim/trace21k/models/sem_model_*",
    "interim/trace21k/models/arma_model_*",
    "interim/trace21k/grid_cal/grid_pcr_cal_*",
    "interim/trace21k/grid_cal/grid_pcr_vc_*",
    "output/**",
]

MANIFEST_NAME = "manifest.txt"


def fmt_size(n: int) -> str:
    """Human-readable byte size."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:,.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def zip_compress_method(path: Path) -> int:
    """Choose zip compression: store binary/NetCDF/pickle; deflate text/tabular."""
    suffix = path.suffix.lower()
    if suffix in {".nc", ".pkl"}:
        return zipfile.ZIP_STORED
    if suffix in {".csv", ".parquet"}:
        return zipfile.ZIP_DEFLATED
    return zipfile.ZIP_DEFLATED


def build_file_list(include_trace: bool) -> list[str]:
    files = list(INPUT_FILES)
    if include_trace:
        files.extend(TRACE_FILES)
    files.extend(OUTPUT_FILES)
    return files


def resolve_files(data_dir: Path, rel_paths: list[str]) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return (found files as (abs_path, arcname)), missing relative paths."""
    found: list[tuple[Path, str]] = []
    missing: list[str] = []
    for rel in rel_paths:
        arcname = rel.replace("\\", "/")
        abs_path = data_dir / rel
        if abs_path.is_file():
            found.append((abs_path, arcname))
        else:
            missing.append(rel)
    return found, missing


def build_manifest(
    data_dir: Path,
    entries: list[tuple[Path, str]],
    include_trace: bool,
    missing: list[str],
) -> str:
    lines = [
        "DPastCliM-NA dataset share manifest",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Source: {data_dir}",
        f"TraCE-21k included: {include_trace}",
        "",
        "Included files (uncompressed size):",
        "-" * 72,
    ]
    total = 0
    for abs_path, arcname in entries:
        size = abs_path.stat().st_size
        total += size
        lines.append(f"{size:>14,d}  {arcname}")
    lines.extend(
        [
            "-" * 72,
            f"{'TOTAL':>14}  {total:,d} bytes ({fmt_size(total)})",
            f"File count: {len(entries)}",
            "",
            "Excluded patterns (not in this archive):",
        ]
    )
    lines.extend(f"  - {pat}" for pat in EXCLUDED_PATTERNS)
    if missing:
        lines.extend(["", "Missing at pack time (not included):"])
        lines.extend(f"  - {rel}" for rel in missing)
    lines.append("")
    return "\n".join(lines)


def print_recipient_instructions(data_dir: Path) -> None:
    print()
    print("=" * 72)
    print("Recipient instructions")
    print("=" * 72)
    print(
        "1. Unzip the archive to a local directory, preserving paths "
        "(e.g. D:\\Dataset\\DPastCliM-NA\\)."
    )
    print(
        "2. Point the project DATA_DIR / dataset path to that directory "
        "in config or environment before running scripts."
    )
    print(f"3. Expected layout matches source root: {data_dir}")
    print(
        "4. If TraCE-21k files were omitted, obtain them separately and place under:"
    )
    print("      TraCE21k/TraCE-21K-II.monthly.TREFHT.nc")
    print("      TraCE21k/TraCE-21K-II.monthly.PRECT.nc")
    print("=" * 72)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack DPastCliM-NA dataset files for internal sharing.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output zip path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be packed; do not create the zip",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--include-trace",
        dest="include_trace",
        action="store_true",
        default=True,
        help="Include TraCE-21k NetCDF files (~10 GB; default)",
    )
    trace_group.add_argument(
        "--no-trace",
        dest="include_trace",
        action="store_false",
        help="Exclude TraCE-21k NetCDF files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()

    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    rel_paths = build_file_list(args.include_trace)
    entries, missing = resolve_files(data_dir, rel_paths)

    print(f"Data directory : {data_dir}")
    print(f"Output zip     : {output_path}")
    print(f"Include TraCE  : {args.include_trace}")
    print(f"Dry run        : {args.dry_run}")
    print()

    if missing:
        print("WARNING: missing files (will be skipped):")
        for rel in missing:
            print(f"  - {rel}")
        print()

    if not entries:
        print("ERROR: no files found to pack.", file=sys.stderr)
        return 1

    total_uncompressed = 0
    print("Files to pack:")
    for abs_path, arcname in entries:
        size = abs_path.stat().st_size
        total_uncompressed += size
        method = zip_compress_method(abs_path)
        method_name = "STORED" if method == zipfile.ZIP_STORED else "DEFLATED"
        print(f"  + {arcname}  ({fmt_size(size)}, {method_name})")

    manifest = build_manifest(data_dir, entries, args.include_trace, missing)
    manifest_bytes = manifest.encode("utf-8")
    print()
    print(f"Manifest       : {MANIFEST_NAME} ({fmt_size(len(manifest_bytes))})")
    print()
    print(f"Total files    : {len(entries)} (+ manifest)")
    print(f"Total size     : {fmt_size(total_uncompressed)} (uncompressed payload)")

    if args.dry_run:
        print()
        print("Dry run complete — no zip created.")
        print_recipient_instructions(data_dir)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print(f"Writing {output_path} ...")

    with zipfile.ZipFile(output_path, "w") as zf:
        for abs_path, arcname in entries:
            zf.write(abs_path, arcname, compress_type=zip_compress_method(abs_path))
        zf.writestr(MANIFEST_NAME, manifest, compress_type=zipfile.ZIP_DEFLATED)

    zip_size = output_path.stat().st_size
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Files packed (payload) : {len(entries)}")
    print(f"  Uncompressed total     : {fmt_size(total_uncompressed)}")
    print(f"  Zip file size          : {fmt_size(zip_size)}")
    print(f"  Output                 : {output_path}")
    if missing:
        print(f"  Missing (skipped)      : {len(missing)}")
    print("=" * 72)

    print_recipient_instructions(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
