"""Per-station per-month regression g(O)' = S beta + u (Methods §2.2).

Link functions:
    tas: identity, g(O) = O
    pr : log-shifted, g(O) = log(O + O_t),  O_t = 1 + min(O_cal)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RegressionFit:
    beta: np.ndarray       # (Ne,)
    sigma2_u: float        # residual variance
    mse: float             # in-sample MSE
    link_offset: float = 0.0   # only used by log-pr link


def identity_link(O: np.ndarray) -> tuple[np.ndarray, float]:
    return O, 0.0


def link_log_pr(O_cal: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-shift link with offset = 1 + min(O_cal)."""
    offset = 1.0 + float(np.nanmin(O_cal))
    return np.log(O_cal + offset), offset


def fit_regression(S: np.ndarray, gO_prime: np.ndarray) -> RegressionFit:
    """OLS of gO_prime on S (intercept-free; gO_prime should be centered).

    S: (Nt, Ne)
    gO_prime: (Nt,)
    """
    valid = ~np.isnan(gO_prime)
    Sv, yv = S[valid], gO_prime[valid]
    beta, *_ = np.linalg.lstsq(Sv, yv, rcond=None)
    resid = yv - Sv @ beta
    sigma2 = float(np.var(resid, ddof=Sv.shape[1]))
    mse = float(np.mean(resid ** 2))
    return RegressionFit(beta=beta, sigma2_u=sigma2, mse=mse)
