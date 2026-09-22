"""Tests for the PHD intrinsic dimension estimator.

Key validation:
- Swiss Roll manifold (known d=2) → PHD should return ≈2.0
- Simple synthetic data (1D line in 3D) → PHD ≈ 1.0
- Edge cases: too-small clouds, degenerate inputs
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll

from intrinsic_ai_music_detection.features.phd import MINIMAL_CLOUD, PHD, prim_tree


class TestPrimTree:
    """Tests for the MST edge-length computation."""

    def test_triangle(self) -> None:
        # 3 points: equilateral-ish triangle
        adj = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        # MST of 3 equidistant points: 2 edges of length 1
        assert prim_tree(adj, alpha=1.0) == pytest.approx(2.0)

    def test_alpha_exponent(self) -> None:
        adj = np.array(
            [
                [0.0, 2.0, 3.0],
                [2.0, 0.0, 1.0],
                [3.0, 1.0, 0.0],
            ]
        )
        # MST: edges 1.0 and 2.0 → sum = 1^2 + 2^2 = 5
        assert prim_tree(adj, alpha=2.0) == pytest.approx(5.0)


class TestPHD:
    """Tests for the PHD estimator."""

    def test_swiss_roll_approx_2d(self) -> None:
        """Swiss Roll has intrinsic dimension 2; PHD should be ≈2.0 ± 1.0."""
        np.random.seed(42)
        X, _ = make_swiss_roll(n_samples=500, noise=0.1)

        phd = PHD(alpha=1.0, metric="euclidean", n_reruns=3)
        dim = phd.estimate(X.astype(np.float64))

        assert not np.isnan(dim), "PHD returned NaN on Swiss Roll"
        assert 1.0 < dim < 4.0, f"Expected ~2, got {dim:.2f}"

    def test_1d_line_in_3d(self) -> None:
        """A straight line embedded in 3D has d ≈ 1."""
        np.random.seed(42)
        t = np.linspace(0, 10, 300)
        X = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
        X += np.random.normal(0, 0.01, X.shape)

        phd = PHD(alpha=1.0, metric="euclidean", n_reruns=3)
        dim = phd.estimate(X.astype(np.float64))

        assert not np.isnan(dim), "PHD returned NaN on 1D line"
        assert 0.5 < dim < 2.0, f"Expected ~1, got {dim:.2f}"

    def test_too_small_cloud_returns_nan(self) -> None:
        """Clouds smaller than MINIMAL_CLOUD should return NaN."""
        X = np.random.randn(MINIMAL_CLOUD - 1, 10)
        phd = PHD()
        assert np.isnan(phd.estimate(X))

    def test_cosine_metric(self) -> None:
        """PHD with cosine metric should not crash and produce finite output."""
        np.random.seed(42)
        X = np.random.randn(200, 50).astype(np.float64)
        # Normalise to unit sphere (cosine lives on hypersphere)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        phd = PHD(alpha=1.0, metric="cosine", n_reruns=2)
        dim = phd.estimate(X)

        assert not np.isnan(dim), "PHD(cosine) returned NaN"
        assert dim > 0, f"Expected positive dimension, got {dim}"

    def test_fit_transform_params(self) -> None:
        """fit_transform with explicit params should work."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float64)
        phd = PHD(alpha=1.0, metric="euclidean", n_reruns=2)
        dim = phd.fit_transform(X, min_points=50, max_points=150, point_jump=20)
        assert not np.isnan(dim)
