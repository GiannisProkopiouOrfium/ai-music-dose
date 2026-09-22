"""Visualization utilities for AI music detection experiments.

Provides publication-ready plots for:
- ID distribution comparisons (violin / box / histogram)
- ROC curves
- Confusion matrices
- Per-track ID scatter plots
- Method comparison heatmaps
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.2)
REAL_COLOR = "#2196F3"
FAKE_COLOR = "#F44336"
PALETTE = {"real": REAL_COLOR, "human": REAL_COLOR, "fake": FAKE_COLOR, "ai": FAKE_COLOR}


def plot_id_distributions(
    human_ids: npt.NDArray[np.float64],
    ai_ids: npt.NDArray[np.float64],
    method: str = "PHD",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot side-by-side violin + strip plots of ID distributions.

    Parameters
    ----------
    human_ids : ID values for human-made tracks
    ai_ids : ID values for AI-generated tracks
    method : name of ID estimator (for axis label)
    title : plot title (auto-generated if None)
    save_path : optional path to save the figure
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "Intrinsic Dimension": np.concatenate([human_ids, ai_ids]),
            "Source": ["Human"] * len(human_ids) + ["AI"] * len(ai_ids),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin plot
    sns.violinplot(
        data=df,
        x="Source",
        y="Intrinsic Dimension",
        hue="Source",
        palette={"Human": REAL_COLOR, "AI": FAKE_COLOR},
        inner="box",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title(f"{method} — Violin Plot")

    # Histogram
    axes[1].hist(human_ids, bins=20, alpha=0.6, color=REAL_COLOR, label="Human", density=True)
    axes[1].hist(ai_ids, bins=20, alpha=0.6, color=FAKE_COLOR, label="AI", density=True)
    axes[1].set_xlabel("Intrinsic Dimension")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"{method} — Distribution")
    axes[1].legend()

    fig.suptitle(title or f"ID Distribution: Human vs AI ({method})", fontsize=14)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_roc_curve(
    fpr: npt.NDArray[np.float64],
    tpr: npt.NDArray[np.float64],
    auc: float,
    label: str = "Model",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a single ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color=REAL_COLOR, lw=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_roc_curves_multi(
    curves: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, str]],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot multiple ROC curves on the same axes.

    Parameters
    ----------
    curves : list of (fpr, tpr, auc, label) tuples
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette("husl", len(curves))

    for (fpr, tpr, auc, label), color in zip(curves, colors):
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Method Comparison")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_confusion_matrix(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
    labels: Sequence[str] = ("Human", "AI"),
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_per_track_scatter(
    ids: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    method: str = "PHD",
    save_path: str | Path | None = None,
) -> Figure:
    """Scatter plot of per-track ID values colored by label.

    Parameters
    ----------
    ids : 1-D array of intrinsic dimension values
    labels : 0 = human, 1 = AI
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [REAL_COLOR if l == 0 else FAKE_COLOR for l in labels]
    ax.scatter(range(len(ids)), ids, c=colors, alpha=0.6, s=20)
    ax.set_xlabel("Track Index")
    ax.set_ylabel(f"Intrinsic Dimension ({method})")
    ax.set_title(f"Per-Track {method} — Human (blue) vs AI (red)")

    # Add mean lines
    human_mean = np.mean(ids[labels == 0])
    ai_mean = np.mean(ids[labels == 1])
    ax.axhline(human_mean, color=REAL_COLOR, ls="--", lw=1.5, label=f"Human mean: {human_mean:.2f}")
    ax.axhline(ai_mean, color=FAKE_COLOR, ls="--", lw=1.5, label=f"AI mean: {ai_mean:.2f}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_method_comparison_heatmap(
    results: dict[str, dict[str, float]],
    metric: str = "auc_roc",
    save_path: str | Path | None = None,
) -> Figure:
    """Heatmap of classification metrics across embedding × ID method combinations.

    Parameters
    ----------
    results : nested dict ``{embedding_name: {method_name: metric_value}}``
    metric : name of metric being displayed (for title)
    """
    import pandas as pd

    df = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_xlabel("ID Estimator")
    ax.set_ylabel("Embedding Model")
    ax.set_title(f"{metric.upper()} — Embedding × ID Method")
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_effect_sizes(
    methods: list[str],
    effect_sizes: list[float],
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart of Cohen's d effect sizes across methods."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [REAL_COLOR if d > 0 else FAKE_COLOR for d in effect_sizes]
    bars = ax.bar(methods, effect_sizes, color=colors, alpha=0.8)

    # Add value labels
    for bar, d in zip(bars, effect_sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{d:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size: Human vs AI Intrinsic Dimension")
    ax.axhline(0, color="gray", lw=0.8)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def _save_fig(fig: Figure, path: str | Path) -> None:
    """Save figure and log the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D1 – Embedding space visualizations (Plan Phase D1)
# ---------------------------------------------------------------------------


def _reduce_embeddings(
    X: npt.NDArray[np.float64],
    method: str = "umap",
    n_components: int = 2,
    seed: int = 42,
    pca_prewhiten: int = 50,
) -> npt.NDArray[np.float64]:
    """Reduce high-dimensional embeddings to 2D for visualization.

    Parameters
    ----------
    X             : [n_samples, dim] embedding matrix.
    method        : 'umap', 'tsne', or 'pca'.
    pca_prewhiten : Apply PCA to this many dims before UMAP/t-SNE if dim > this.
    """
    import numpy as np

    if X.shape[1] > pca_prewhiten and method in ("umap", "tsne"):
        from sklearn.decomposition import PCA

        X = PCA(n_components=pca_prewhiten, random_state=seed).fit_transform(X)

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=n_components, random_state=seed, n_jobs=1)
            return reducer.fit_transform(X)
        except ImportError:
            logger.warning("umap-learn not installed; falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=n_components, random_state=seed, perplexity=30).fit_transform(X)

    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=seed).fit_transform(X)


def plot_embedding_space(
    X: npt.NDArray[np.float64],
    labels: Sequence[str],
    hue_col: str = "label",
    extra_cols: dict | None = None,
    method: str = "umap",
    n_max: int = 10000,
    seed: int = 42,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """UMAP / t-SNE scatter of pooled window embeddings.

    Parameters
    ----------
    X         : [n_samples, dim] embeddings (pre-pooled window vectors).
    labels    : per-sample label string (e.g. 'real', 'udio-120s', 'chirp-v3').
    hue_col   : 'label', 'generator', or 'genre' — which column drives color.
    extra_cols: dict mapping additional attribute name -> per-sample list/array
                (e.g. {'genre': genres_list, 'anomaly': scores_list}).
    method    : 'umap', 'tsne', or 'pca'.
    n_max     : subsample to this many points for speed (stratified).
    """
    import pandas as pd

    labels_arr = np.asarray(labels)
    n = len(X)

    # Stratified subsample
    if n > n_max:
        rng = np.random.default_rng(seed)
        uniq, counts = np.unique(labels_arr, return_counts=True)
        per_class = max(n_max // len(uniq), 1)
        idx = np.concatenate(
            [
                rng.choice(np.where(labels_arr == u)[0], size=min(per_class, c), replace=False)
                for u, c in zip(uniq, counts)
            ]
        )
        X_sub, labels_sub = X[idx], labels_arr[idx]
        extra_sub = {k: np.asarray(v)[idx] for k, v in (extra_cols or {}).items()}
    else:
        X_sub, labels_sub = X, labels_arr
        extra_sub = {k: np.asarray(v) for k, v in (extra_cols or {}).items()}

    logger.info("Reducing %d samples to 2D via %s ...", len(X_sub), method)
    coords = _reduce_embeddings(X_sub.astype(np.float64), method=method, seed=seed)

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], hue_col: labels_sub})
    for k, v in extra_sub.items():
        df[k] = v

    # Determine number of panels: base (label) + optional anomaly score
    n_panels = 1 + int("anomaly" in extra_sub)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", n_colors=len(df[hue_col].unique()))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue=hue_col,
        palette=palette,
        s=8,
        alpha=0.6,
        linewidths=0,
        ax=axes[0],
    )
    axes[0].set_title(f"{method.upper()} — colored by {hue_col}")
    axes[0].set_xlabel(f"{method.upper()} dim 1")
    axes[0].set_ylabel(f"{method.upper()} dim 2")
    axes[0].legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.05, 1))

    if "anomaly" in extra_sub:
        sc = axes[1].scatter(
            df["x"],
            df["y"],
            c=df["anomaly"],
            cmap="RdYlGn_r",
            s=8,
            alpha=0.6,
            vmin=np.percentile(df["anomaly"].dropna(), 2),
            vmax=np.percentile(df["anomaly"].dropna(), 98),
        )
        plt.colorbar(sc, ax=axes[1], label="Anomaly score (higher = more fake)")
        axes[1].set_title(f"{method.upper()} — colored by anomaly score")
        axes[1].set_xlabel(f"{method.upper()} dim 1")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_per_dimension_nll(
    per_dim_nll: npt.NDArray[np.float64],
    dim_names: Sequence[str] | None = None,
    top_k: int = 20,
    title: str = "Per-dimension NLL contribution",
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart of per-EnCodec-dimension NLL contribution (Plan D2 support).

    Parameters
    ----------
    per_dim_nll : [n_dims] array of mean NLL contribution per dimension,
                  computed as -0.5*(z_dim^2 + log2π) averaged over windows.
    dim_names   : optional axis labels.
    top_k       : number of top anomalous dimensions to highlight.
    """
    n_dims = len(per_dim_nll)
    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(n_dims)]

    sort_idx = np.argsort(per_dim_nll)  # ascending: most anomalous (lowest NLL) first
    colors = ["#F44336" if i in sort_idx[:top_k] else "#2196F3" for i in range(n_dims)]

    fig, ax = plt.subplots(figsize=(max(12, n_dims // 4), 5))
    ax.bar(range(n_dims), -per_dim_nll, color=colors, alpha=0.8)  # negate: higher bar = more anomalous
    ax.set_xlabel("EnCodec dimension index")
    ax.set_ylabel("Mean per-dim anomaly contribution (−NLL, higher = more anomalous)")
    ax.set_title(title)
    ax.set_xticks(sort_idx[:top_k])
    ax.set_xticklabels([dim_names[i] for i in sort_idx[:top_k]], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_temporal_anomaly_trajectory(
    window_scores: npt.NDArray[np.float64],
    label: str = "Track",
    threshold: float | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot per-window anomaly score over time for a single track (Plan D2)."""
    t = np.arange(len(window_scores)) * 2.0  # 2s hop
    color = FAKE_COLOR if label.lower() == "fake" else REAL_COLOR

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, window_scores, color=color, lw=1.5, alpha=0.9)
    ax.fill_between(t, window_scores.min(), window_scores, alpha=0.15, color=color)
    if threshold is not None:
        ax.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold={threshold:.1f}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Anomaly score (−log p)")
    ax.set_title(f"Temporal anomaly trajectory — {label}")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig
