"""Intrinsic dimension estimators: TwoNN, MLE, and a unified wrapper.

Wraps scikit-dimension estimators alongside the custom PHD implementation
to provide a consistent interface for computing intrinsic dimension on
audio embedding point clouds.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

from intrinsic_ai_music_detection.config import PHDConfig
from intrinsic_ai_music_detection.features.phd import MINIMAL_CLOUD, PHD

logger = logging.getLogger(__name__)


def estimate_phd(
    embeddings: npt.NDArray[np.float32],
    metric: str = "cosine",
    cfg: PHDConfig | None = None,
) -> float:
    """Estimate intrinsic dimension via Persistent Homology Dimension.

    Parameters
    ----------
    embeddings : array of shape ``[N, D]``
    metric : distance metric (``'cosine'`` for CLAP/MERT, ``'euclidean'`` for EnCodec)
    cfg : PHD hyperparameters

    Returns
    -------
    float — estimated intrinsic dimension, or NaN if cloud is too small
    """
    if cfg is None:
        cfg = PHDConfig()

    solver = PHD(
        alpha=cfg.alpha,
        metric=metric,
        n_reruns=cfg.n_reruns,
        n_points=cfg.n_points,
        n_points_min=cfg.n_points_min,
    )
    return solver.estimate(embeddings.astype(np.float64), intermediate_points=cfg.intermediate_points)


def estimate_twonn(embeddings: npt.NDArray[np.float32]) -> float:
    """Estimate intrinsic dimension via Two Nearest Neighbors.

    Uses the ratio of 1st and 2nd nearest neighbor distances.
    Robust to noise; non-parametric.
    """
    from skdim.id import TwoNN

    if embeddings.shape[0] < 10:
        return float("nan")

    try:
        estimator = TwoNN()
        estimator.fit(embeddings)
        return float(estimator.dimension_)
    except Exception:
        logger.warning("TwoNN estimation failed", exc_info=True)
        return float("nan")


def estimate_mle(embeddings: npt.NDArray[np.float32], k1: int = 2, k2: int = 20) -> float:
    """Estimate intrinsic dimension via Maximum Likelihood Estimation.

    Native implementation of Levina & Bickel (2004), avoiding scikit-dimension's
    MLE which is broken on Python 3.13 (FrameLocalsProxy incompatibility).

    For each point x_i with k-th nearest neighbor distance T_k(x_i):
        m_k(x_i) = [ 1/(k-1) * sum_{j=1}^{k-1} log(T_k / T_j) ]^{-1}

    The global estimate averages over all points and k in [k1, k2].

    Uses KDTree for O(n log n) neighbor queries instead of O(n²) pairwise distances.
    """
    from scipy.spatial import KDTree

    n = embeddings.shape[0]
    k2 = min(k2, n - 2)  # cannot exceed N-1 neighbors
    if n < k2 + 1 or k2 < k1:
        return float("nan")

    try:
        # KDTree for fast neighbor lookup — O(n log n) vs O(n²) for cdist
        tree = KDTree(embeddings)
        # Query k2+1 neighbors (including self at distance 0)
        dists, _ = tree.query(embeddings, k=k2 + 1)
        sorted_dists = dists[:, 1:]  # [N, k2] — exclude self

        estimates = []
        for k in range(k1, k2 + 1):
            T_k = sorted_dists[:, k - 1]  # [N]
            # Vectorized: sum log(T_k / T_j) for j in [1, k-1]
            T_js = sorted_dists[:, : k - 1]  # [N, k-1]
            # Avoid log(0): clamp distances
            T_k_safe = np.maximum(T_k, 1e-30)[:, np.newaxis]
            T_js_safe = np.maximum(T_js, 1e-30)
            log_ratios = np.sum(np.log(T_k_safe / T_js_safe), axis=1)  # [N]

            valid = log_ratios > 1e-30
            per_point = np.full(n, np.nan)
            per_point[valid] = (k - 1) / log_ratios[valid]
            estimates.append(np.nanmean(per_point))

        return float(np.nanmean(estimates))
    except Exception:
        logger.warning("MLE estimation failed", exc_info=True)
        return float("nan")


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

EstimatorName = Literal["phd", "twonn", "mle"]


def estimate_id(
    embeddings: npt.NDArray[np.float32],
    method: EstimatorName = "phd",
    metric: str = "cosine",
    phd_cfg: PHDConfig | None = None,
) -> float:
    """Estimate intrinsic dimension using the specified method.

    Parameters
    ----------
    embeddings : ``[N, D]`` embedding matrix for a single track
    method : ``'phd'``, ``'twonn'``, or ``'mle'``
    metric : distance metric (only used by PHD)
    phd_cfg : PHD hyperparameters (only used by PHD)
    """
    if method == "phd":
        return estimate_phd(embeddings, metric=metric, cfg=phd_cfg)
    elif method == "twonn":
        return estimate_twonn(embeddings)
    elif method == "mle":
        return estimate_mle(embeddings)
    else:
        raise ValueError(f"Unknown ID estimator: {method!r}")


def estimate_all(
    embeddings: npt.NDArray[np.float32],
    metric: str = "cosine",
    phd_cfg: PHDConfig | None = None,
    methods: list[EstimatorName] | None = None,
    max_points: int = 2000,
) -> dict[str, float]:
    """Run all requested estimators and return a dict of results.

    Parameters
    ----------
    embeddings : ``[N, D]`` embedding matrix
    metric : distance metric for PHD
    phd_cfg : PHD hyperparameters
    methods : list of estimator names (default: all three)
    max_points : subsample to this many points for efficiency.
        ID estimation converges well before N=9000; 2000 points
        gives nearly identical results in ~1/20th the time.

    Returns
    -------
    dict mapping estimator name → intrinsic dimension value
    """
    if methods is None:
        methods = ["phd", "twonn", "mle"]

    # Subsample large point clouds for speed
    if embeddings.shape[0] > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(embeddings.shape[0], max_points, replace=False)
        embeddings = embeddings[idx]

    results: dict[str, float] = {}
    for method in methods:
        results[method] = estimate_id(embeddings, method=method, metric=metric, phd_cfg=phd_cfg)
        logger.debug("ID(%s) = %.3f", method, results[method])

    return results
