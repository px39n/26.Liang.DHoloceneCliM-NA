"""Iterative mode selection by RMSE drop on the model-selection split (Methods §2.4)."""
from __future__ import annotations
import numpy as np

from ..config import Modes
from .pca import PCAResult


def select_modes(
    pca: PCAResult,
    O_msel: np.ndarray,
    cfg: Modes,
) -> list[int]:
    """Return list of column indices in PCA basis to retain.

    Algorithm:
      1. start with modes whose ev >= init_ev_pct
      2. consider candidates with ev >= candidate_ev_pct, ordered by ev desc
      3. for each, compute mean RMSE over n_realizations on O_msel; if it
         drops by >= rmse_drop_pct relative to current set, accept
      4. stop at max_modes
    """
    raise NotImplementedError("select_modes: implement after regression is wired")
