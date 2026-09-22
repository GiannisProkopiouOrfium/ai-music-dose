"""Tests for the visualization module."""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.visualization.visualize import (
    plot_confusion_matrix,
    plot_effect_sizes,
    plot_id_distributions,
    plot_method_comparison_heatmap,
    plot_per_track_scatter,
    plot_roc_curve,
    plot_roc_curves_multi,
)


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def human_ids(rng: np.random.RandomState) -> np.ndarray:
    return rng.normal(18.0, 2.0, 50)


@pytest.fixture
def ai_ids(rng: np.random.RandomState) -> np.ndarray:
    return rng.normal(15.0, 2.0, 50)


class TestPlotIDDistributions:
    def test_returns_figure(self, human_ids: np.ndarray, ai_ids: np.ndarray) -> None:
        import matplotlib.pyplot as plt

        fig = plot_id_distributions(human_ids, ai_ids, method="PHD")
        assert fig is not None
        assert len(fig.axes) == 2  # violin + histogram
        plt.close(fig)

    def test_save_to_file(self, human_ids: np.ndarray, ai_ids: np.ndarray, tmp_path) -> None:
        save_path = tmp_path / "test_dist.png"
        plot_id_distributions(human_ids, ai_ids, save_path=str(save_path))
        assert save_path.exists()


class TestPlotROC:
    def test_single_roc(self) -> None:
        import matplotlib.pyplot as plt

        fpr = np.array([0, 0.1, 0.3, 1.0])
        tpr = np.array([0, 0.5, 0.8, 1.0])
        fig = plot_roc_curve(fpr, tpr, auc=0.85, label="Test")
        assert fig is not None
        plt.close(fig)

    def test_multi_roc(self) -> None:
        import matplotlib.pyplot as plt

        curves = [
            (np.array([0, 0.2, 1]), np.array([0, 0.8, 1]), 0.9, "Model A"),
            (np.array([0, 0.3, 1]), np.array([0, 0.7, 1]), 0.8, "Model B"),
        ]
        fig = plot_roc_curves_multi(curves)
        assert fig is not None
        plt.close(fig)


class TestPlotConfusionMatrix:
    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        plt.close(fig)


class TestPlotPerTrackScatter:
    def test_returns_figure(self, rng: np.random.RandomState) -> None:
        import matplotlib.pyplot as plt

        ids = rng.normal(15, 3, 40)
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
        fig = plot_per_track_scatter(ids, labels, method="TwoNN")
        assert fig is not None
        plt.close(fig)


class TestPlotMethodComparisonHeatmap:
    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        results = {
            "encodec": {"PHD": 0.85, "TwoNN": 0.82, "MLE": 0.78},
            "clap": {"PHD": 0.90, "TwoNN": 0.88, "MLE": 0.84},
        }
        fig = plot_method_comparison_heatmap(results, metric="AUC")
        assert fig is not None
        plt.close(fig)


class TestPlotEffectSizes:
    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        methods = ["PHD", "TwoNN", "MLE"]
        effect_sizes = [1.2, 0.8, -0.3]
        fig = plot_effect_sizes(methods, effect_sizes)
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmp_path) -> None:
        save_path = tmp_path / "effects.png"
        plot_effect_sizes(["A", "B"], [0.5, 1.0], save_path=str(save_path))
        assert save_path.exists()
