"""Tests for the ID estimator wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.id_estimators import estimate_all, estimate_id


class TestEstimateID:
    """Tests for the unified estimate_id function."""

    def test_phd_runs(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        dim = estimate_id(X, method="phd", metric="euclidean")
        assert not np.isnan(dim)
        assert dim > 0

    def test_unknown_method_raises(self) -> None:
        X = np.random.randn(100, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown"):
            estimate_id(X, method="bogus")  # type: ignore[arg-type]

    def test_small_cloud_phd(self) -> None:
        X = np.random.randn(10, 5).astype(np.float32)
        dim = estimate_id(X, method="phd")
        assert np.isnan(dim)


class TestEstimateAll:
    """Tests for estimate_all."""

    def test_returns_dict(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        results = estimate_all(X, metric="euclidean", methods=["phd"])
        assert isinstance(results, dict)
        assert "phd" in results
