"""Spatial Error Model + seasonal ARMA on regression residuals (Methods §4).

This is the most numerically intensive piece (MLE on a per-month / per-pixel basis).
Likely needs sparse matrix tricks for the SEM weight matrix W.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SEMFit:
    rho: float        # spatial AR coefficient
    sigma2: float     # innovation variance
    W: np.ndarray     # (Ns, Ns) row-normalized neighbor weights


def fit_sem(residuals: np.ndarray, coords: np.ndarray, k_neighbors: int = 8) -> SEMFit:
    """Fit u = rho * W u + eps via MLE.  See inverse_SEM_season.m for reference."""
    raise NotImplementedError("fit_sem: port from inverse_SEM_season.m")
