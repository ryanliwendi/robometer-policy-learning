"""VENDORED from "Uncertainty Comes for Free" (He, Cao & Ciocarlie, 2025; arXiv:2503.01876).

Source: https://github.com/Yifeng-Cao/hitl_DiffusionPolicy --
diffusion_policy/uncertainty/gmm_uncertainty.py
"""

from typing import Sequence, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

REFERENCE_N_INIT = 10


def fit_best_gmm(vectors: np.ndarray,
                 mode_candidates: Sequence[int] = (1, 2, 3, 4, 5),
                 random_state: int = 42,
                 n_init: int = REFERENCE_N_INIT) -> GaussianMixture:
    best_bic = np.inf
    best_gmm = None

    for n_modes in mode_candidates:
        gmm = GaussianMixture(
            n_components=n_modes,
            covariance_type='full',
            random_state=random_state,
            max_iter=100,
            n_init=n_init,
        )
        try:
            gmm.fit(vectors)
            bic = gmm.bic(vectors)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            # GMM fitting can fail for some mode numbers (e.g. singular covariance)
            continue

    if best_gmm is None:
        # Fallback to single mode if all fittings failed
        best_gmm = GaussianMixture(n_components=1, random_state=random_state)
        best_gmm.fit(vectors)

    return best_gmm


def compute_divergence(gmm: GaussianMixture) -> float:
    """Inter-mode divergence D(V) = sum_i w_i * ||mu_i - mu_mean||."""
    weights = gmm.weights_  # [n_modes]
    means = gmm.means_      # [n_modes, 3]

    mean_of_means = np.average(means, axis=0, weights=weights)  # [3]

    divergence = 0.0
    for i in range(len(weights)):
        distance = np.linalg.norm(means[i] - mean_of_means)
        divergence += weights[i] * distance

    return divergence


def compute_weighted_variance(gmm: GaussianMixture, vectors: np.ndarray) -> float:
    """Intra-mode weighted variance Var_g(V) = sum_i w_i * Var(vectors assigned to mode i)."""
    labels = gmm.predict(vectors)  # [N]
    n_modes = gmm.n_components

    weighted_var = 0.0
    for i in range(n_modes):
        mode_vectors = vectors[labels == i]

        if len(mode_vectors) == 0:
            continue

        mode_mean = gmm.means_[i]  # [3]
        distances_sq = np.sum((mode_vectors - mode_mean) ** 2, axis=1)
        mode_variance = np.mean(distances_sq)

        weighted_var += gmm.weights_[i] * mode_variance

    return weighted_var


def compute_uncertainty(vectors: np.ndarray,
                        alpha: float = 0.1,
                        mode_candidates: Sequence[int] = (1, 2, 3, 4, 5),
                        random_state: int = 42,
                        n_init: int = REFERENCE_N_INIT) -> Tuple[float, GaussianMixture]:
    """``D(V) + alpha * Var_g(V)`` for one (N, 3) vector field, and the GMM it came from."""
    gmm = fit_best_gmm(vectors, mode_candidates, random_state, n_init=n_init)

    divergence = compute_divergence(gmm)
    weighted_var = compute_weighted_variance(gmm, vectors)

    uncertainty = divergence + alpha * weighted_var

    return uncertainty, gmm
