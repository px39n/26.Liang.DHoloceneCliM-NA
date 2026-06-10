"""Single source of truth for the dataset root in related_scripts/.

All production scripts import DATA_ROOT from here.
Set the environment variable DPASTCLIM_DATA_DIR to override the default path.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DPASTCLIM_DATA_DIR", r"D:\Dataset\DPastCliM-NA"))
