"""PCA decomposition of the anomaly matrix M' = E S^T (Methods §2.1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PCAResult:
    E: np.ndarray   # (Ng, Ne) eigenvectors / spatial patterns
    S: np.ndarray   # (Nt, Ne) scores / temporal coefficients
    ev: np.ndarray  # (Ne,)    explained variance fractions


def fit_pca(M_prime: np.ndarray, n_components: int | None = None) -> PCAResult:
    """SVD-based PCA on (Nt, Ng) anomaly matrix.

    Convention: rows of M' sum to zero, max rank = Nt - 1.
    """
    Nt, Ng = M_prime.shape
    U, sigma, Vt = np.linalg.svd(M_prime, full_matrices=False)
    E = Vt.T                              # (Ng, k)
    S = U * sigma                         # (Nt, k)
    var = sigma ** 2 / (Nt - 1)
    ev = var / var.sum()
    if n_components is not None:
        E, S, ev = E[:, :n_components], S[:, :n_components], ev[:n_components]
    return PCAResult(E=E, S=S, ev=ev)
