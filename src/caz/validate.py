"""Methods §7: validation metrics and skill scores."""
from __future__ import annotations
import numpy as np


def mb(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.nanmean(pred - obs))


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.nanmean(np.abs(pred - obs)))


def kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta efficiency."""
    p, o = pred[~np.isnan(pred) & ~np.isnan(obs)], obs[~np.isnan(pred) & ~np.isnan(obs)]
    r = float(np.corrcoef(p, o)[0, 1])
    alpha = float(np.std(p) / np.std(o))
    beta = float(np.mean(p) / np.mean(o))
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def skill_score(A_target: float, A_ref: float, A_perfect: float = 0.0) -> float:
    """SS_A = 100 * (A_target - A_ref) / (A_perfect - A_ref)."""
    if A_perfect == A_ref:
        return float("nan")
    return 100.0 * (A_target - A_ref) / (A_perfect - A_ref)
