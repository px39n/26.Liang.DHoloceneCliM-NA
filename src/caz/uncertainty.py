"""Methods §6: Monte Carlo prediction intervals at the station scale."""
from __future__ import annotations
import numpy as np


def monte_carlo_pi(
    Sbeta: np.ndarray,
    mse: float,
    n_sim: int = 1000,
    quantiles: tuple[float, float] = (0.025, 0.975),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate N MC realizations, return (Nt, 2) lower/upper bounds in g-space.

    Mean correction + back-transform must be applied after this.
    """
    rng = rng or np.random.default_rng(0)
    eps = rng.normal(0.0, np.sqrt(mse), size=(n_sim,) + Sbeta.shape)
    sims = Sbeta[None, ...] + eps
    return np.quantile(sims, q=list(quantiles), axis=0).T
