"""Persistent Homology Dimension (PHD) estimator for intrinsic dimension.

Ported and adapted from ArGintum/GPTID (MIT license):
https://github.com/ArGintum/GPTID/blob/main/IntrinsicDim.py

Original paper: "Intrinsic Dimension Estimation for Robust Detection
of AI-Generated Texts" (Tulchinskiy et al., 2023).

Adaptations for audio:
  - Support for cosine distance (CLAP/MERT) and euclidean (EnCodec)
  - Configurable distance metric via scipy.spatial.distance.cdist
  - Unchanged core algorithm: MST-based PH0 estimation via Prim's tree
"""

from __future__ import annotations

from threading import Thread

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

# Minimum number of points to attempt PHD estimation
MINIMAL_CLOUD = 80


def prim_tree(adj_matrix: npt.NDArray[np.float64], alpha: float = 1.0) -> float:
    """Compute the α-weighted total edge length of the Minimum Spanning Tree.

    Uses Prim's algorithm on a dense adjacency (distance) matrix.

    Parameters
    ----------
    adj_matrix : square distance matrix of shape ``[n, n]``
    alpha : exponent applied to each edge weight before summing

    Returns
    -------
    float — the sum  Σ_{e ∈ MST} |e|^α
    """
    n = adj_matrix.shape[0]
    infty = np.max(adj_matrix) + 10.0

    dst = np.full(n, infty)
    visited = np.zeros(n, dtype=bool)

    v = 0
    s = 0.0
    for _ in range(n - 1):
        visited[v] = True
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty
        v = int(np.argmin(dst))
        s += dst[v] ** alpha

    return float(s)


class PHD:
    """Persistence-Homology-based Intrinsic Dimension estimator.

    Estimates the intrinsic dimension *d* of a point cloud by analysing
    how the total MST edge length scales with subsample size.

    The relationship ``log E(n) ≈ κ · log n`` is fitted via OLS, and
    the intrinsic dimension is computed as ``d = 1 / (1 − κ)``.

    Parameters
    ----------
    alpha : exponent for MST edge weighting (default 1.0)
    metric : distance metric for ``scipy.spatial.distance.cdist``
             Use ``'cosine'`` for CLAP/MERT, ``'euclidean'`` for EnCodec.
    n_reruns : number of independent estimation runs (averaged)
    n_points : number of subsamples drawn at each subsample size
    n_points_min : subsamples drawn for sizes > half the cloud
    """

    def __init__(
        self,
        alpha: float = 1.0,
        metric: str = "euclidean",
        n_reruns: int = 3,
        n_points: int = 7,
        n_points_min: int = 3,
    ) -> None:
        self.alpha = alpha
        self.metric = metric
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self._is_distance_matrix = False

    def _sample(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int,
    ) -> npt.NDArray[np.float64]:
        """Draw a random subset of size *n_samples*."""
        indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
        if self._is_distance_matrix:
            return X[np.ix_(indices, indices)]
        return X[indices]

    def _calc_single(
        self,
        X: npt.NDArray[np.float64],
        test_n: range,
        output: npt.NDArray[np.float64],
        thread_id: int,
    ) -> None:
        """Single-threaded estimation run."""
        lengths = []
        for n in test_n:
            restarts = self.n_points_min if X.shape[0] <= 2 * n else self.n_points
            reruns = np.ones(restarts)
            for i in range(restarts):
                subset = self._sample(X, n)
                if self._is_distance_matrix:
                    dist_mat = subset
                else:
                    dist_mat = cdist(subset, subset, metric=self.metric)
                reruns[i] = prim_tree(dist_mat, self.alpha)
            lengths.append(float(np.median(reruns)))

        lengths_arr = np.array(lengths)
        x = np.log(np.array(list(test_n), dtype=float))
        y = np.log(lengths_arr)

        # OLS slope: κ = (N·Σxy − Σx·Σy) / (N·Σx² − (Σx)²)
        n_pts = len(x)
        output[thread_id] = (n_pts * (x * y).sum() - x.sum() * y.sum()) / (n_pts * (x**2).sum() - x.sum() ** 2)

    def fit_transform(
        self,
        X: npt.NDArray[np.float64],
        min_points: int = 50,
        max_points: int = 512,
        point_jump: int = 40,
        dist: bool = False,
    ) -> float:
        """Estimate the intrinsic dimension of point cloud *X*.

        Parameters
        ----------
        X : array of shape ``[n_points, n_features]``, or a precomputed
            distance matrix of shape ``[n, n]`` when *dist=True*.
        min_points : smallest subsample size
        max_points : largest subsample size
        point_jump : step between consecutive subsample sizes
        dist : if True, treat *X* as a precomputed distance matrix

        Returns
        -------
        float — estimated intrinsic dimension  d = 1 / (1 − κ)
        """
        self._is_distance_matrix = dist

        if X.shape[0] < MINIMAL_CLOUD:
            return float("nan")

        # Clamp max_points to the actual cloud size
        max_points = min(max_points, X.shape[0])
        if min_points >= max_points:
            return float("nan")

        test_n = range(min_points, max_points, point_jump)
        if len(test_n) < 2:
            return float("nan")

        slopes = np.zeros(self.n_reruns)
        threads: list[Thread] = []

        for i in range(self.n_reruns):
            t = Thread(target=self._calc_single, args=(X, test_n, slopes, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        kappa = float(np.mean(slopes))
        if kappa >= 1.0:
            return float("nan")
        return 1.0 / (1.0 - kappa)

    def estimate(
        self,
        X: npt.NDArray[np.float64],
        intermediate_points: int = 7,
    ) -> float:
        """Convenience wrapper with automatic parameter tuning.

        Computes ``min_points``, ``max_points``, and ``point_jump``
        from the cloud size, similar to the GPTID example notebook.
        """
        n = X.shape[0]
        if n < MINIMAL_CLOUD:
            return float("nan")

        min_pts = max(40, n // 10)
        max_pts = n - (n // 10)
        step = max(1, (max_pts - min_pts) // intermediate_points)

        return self.fit_transform(X, min_points=min_pts, max_points=max_pts, point_jump=step)
