"""Tests for the classification and evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.models.evaluate import cohens_d, compare_distributions, compute_binary_metrics
from intrinsic_ai_music_detection.models.train_model import (
    build_pipeline,
    train_and_evaluate,
    train_threshold_classifier,
)


class TestCohensD:
    def test_identical_groups(self) -> None:
        a = np.ones(50)
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_large_effect(self) -> None:
        a = np.random.RandomState(42).normal(0, 1, 100)
        b = np.random.RandomState(42).normal(3, 1, 100)
        d = cohens_d(a, b)
        assert abs(d) > 2.0  # Very large effect


class TestCompareDistributions:
    def test_returns_result(self) -> None:
        np.random.seed(42)
        human = np.random.normal(5.0, 1.0, 50)
        ai = np.random.normal(3.0, 1.0, 50)
        result = compare_distributions(human, ai)
        assert result.mann_whitney_p < 0.05
        assert result.cohens_d > 0


class TestBuildPipeline:
    def test_logistic_regression(self) -> None:
        pipe = build_pipeline("logistic_regression")
        assert len(pipe.steps) == 2

    def test_svm(self) -> None:
        pipe = build_pipeline("svm")
        assert len(pipe.steps) == 2


class TestTrainAndEvaluate:
    def test_separable_data(self) -> None:
        np.random.seed(42)
        X = np.vstack(
            [
                np.random.normal(0, 0.5, (50, 3)),
                np.random.normal(3, 0.5, (50, 3)),
            ]
        )
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        result = train_and_evaluate(X, y, feature_set_name="test")
        assert result.cv_accuracy > 0.8
        assert result.cv_f1 > 0.8


class TestThresholdClassifier:
    def test_perfect_separation(self) -> None:
        ids = np.array([1.0, 1.5, 2.0, 5.0, 6.0, 7.0])
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        thresh, acc = train_threshold_classifier(ids, labels, direction="lower")
        assert acc > 0.8
