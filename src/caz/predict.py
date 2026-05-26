"""Methods §3: apply fitted PCR + mean correction to Holocene transient input."""
from __future__ import annotations
import numpy as np
import xarray as xr


def project_pca(M_past: np.ndarray, E: np.ndarray) -> np.ndarray:
    """S_past = M_past @ E   (Nt_past, Ne)."""
    return M_past @ E


def mean_correction(
    gO_pred: np.ndarray,
    gO_cal_mean: float,
    gM_mov30_mean: np.ndarray,
    gM_cal_mean: float,
) -> np.ndarray:
    """Add mean offset:  ĝ + mean(g(O)_cal) + mean(g(M)_mov30) - mean(g(M)_cal)."""
    return gO_pred + gO_cal_mean + gM_mov30_mean - gM_cal_mean


def back_transform(g_hat: np.ndarray, link: str, offset: float = 0.0) -> np.ndarray:
    if link == "identity":
        return g_hat
    if link == "log_pr":
        return np.exp(g_hat) - offset
    raise ValueError(link)
