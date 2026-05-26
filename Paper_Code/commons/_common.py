"""Shared utilities for all Paper_Code figure scripts."""
from __future__ import annotations
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_MAIN = PROJECT_ROOT / "Results" / "main"
RESULTS_SUP  = PROJECT_ROOT / "Results" / "supplementary"
LATEX_FIG    = PROJECT_ROOT / "latex_paper" / "Figures"

DATA_DIR     = Path(r"D:\Dataset\DPastCliM-NA")
GHCN_DIR     = DATA_DIR / "GHCN" / "interim"
TRACE_DIR    = DATA_DIR / "TraCE21k"

GCM          = "trace21k"
MODELS_DIR   = DATA_DIR / "interim" / GCM / "models"
STATION_CAL  = DATA_DIR / "interim" / GCM / "station_cal"
GRID_CAL     = DATA_DIR / "interim" / GCM / "grid_cal"
INTERIM_DIR  = STATION_CAL  # backward compat alias


def sync_output(fig_path: str | Path, csv_path: str | Path | None = None,
                is_supplementary: bool = False) -> None:
    """Copy figure (and optional CSV) to Results/ and latex_paper/Figures/.

    If the source file is already in the destination, the copy is skipped.
    """
    fig_path = Path(fig_path).resolve()
    dest_results = RESULTS_SUP if is_supplementary else RESULTS_MAIN
    dest_results.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.mkdir(parents=True, exist_ok=True)

    dst_res = (dest_results / fig_path.name).resolve()
    dst_latex = (LATEX_FIG / fig_path.name).resolve()

    if fig_path != dst_res:
        shutil.copy2(fig_path, dst_res)
        print(f"  -> {dst_res}")
    shutil.copy2(fig_path, dst_latex)
    print(f"  -> {dst_latex}")

    if csv_path is not None:
        csv_path = Path(csv_path).resolve()
        dst_csv = (dest_results / csv_path.name).resolve()
        if csv_path != dst_csv:
            shutil.copy2(csv_path, dst_csv)
            print(f"  -> {dst_csv}")
