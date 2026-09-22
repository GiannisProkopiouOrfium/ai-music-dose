"""Tests for EER metric and the RealNVP one-class flow detector."""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, equal_error_rate
from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass


class TestEER:
    def test_perfect_separation_zero_eer(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert equal_error_rate(y, scores) == pytest.approx(0.0, abs=1e-6)

    def test_random_scores_near_half(self) -> None:
        rng = np.random.RandomState(0)
        y = np.r_[np.zeros(500), np.ones(500)].astype(int)
        scores = rng.rand(1000)
        eer = equal_error_rate(y, scores)
        assert 0.35 < eer < 0.65

    def test_single_class_returns_nan(self) -> None:
        assert np.isnan(equal_error_rate(np.zeros(5), np.arange(5)))

    def test_auc_and_eer_consistent(self) -> None:
        rng = np.random.RandomState(1)
        y = np.r_[np.zeros(200), np.ones(200)].astype(int)
        scores = np.r_[rng.normal(0, 1, 200), rng.normal(3, 1, 200)]
        auc, eer = auc_and_eer(y, scores)
        assert auc > 0.9
        assert eer < 0.15


class TestRealNVPOneClass:
    def test_anomaly_scores_higher_for_ood(self) -> None:
        rng = np.random.RandomState(42)
        # real manifold: tight Gaussian blob in 8-D
        x_real = rng.normal(0, 1, size=(800, 8))
        # ood: shifted far away
        x_ood = rng.normal(6, 1, size=(200, 8))

        det = RealNVPOneClass(RealNVPConfig(n_epochs=60, n_coupling_layers=6, hidden_dim=64, seed=0)).fit(x_real)

        real_score = det.score_samples(x_real).mean()
        ood_score = det.score_samples(x_ood).mean()
        assert ood_score > real_score

    def test_detects_ood_via_auc(self) -> None:
        rng = np.random.RandomState(7)
        x_real = rng.normal(0, 1, size=(600, 6))
        x_ood = rng.normal(0, 1, size=(200, 6)) + np.array([4, 0, 0, 0, 0, 0])

        det = RealNVPOneClass(RealNVPConfig(n_epochs=80, n_coupling_layers=6, hidden_dim=64, seed=1)).fit(x_real)

        y = np.r_[np.zeros(len(x_real)), np.ones(len(x_ood))].astype(int)
        scores = np.r_[det.score_samples(x_real), det.score_samples(x_ood)]
        auc, eer = auc_and_eer(y, scores)
        assert auc > 0.85
        assert eer < 0.25

    def test_one_dim_gaussian_fallback(self) -> None:
        rng = np.random.RandomState(3)
        x_real = rng.normal(0, 1, size=(300, 1))
        det = RealNVPOneClass(RealNVPConfig(n_epochs=10)).fit(x_real)
        # far-out point should be more anomalous than an in-distribution point
        assert det.score_samples(np.array([[10.0]]))[0] > det.score_samples(np.array([[0.0]]))[0]

    def test_requires_fit_before_score(self) -> None:
        det = RealNVPOneClass()
        with pytest.raises(RuntimeError):
            det.score_samples(np.zeros((2, 4)))
