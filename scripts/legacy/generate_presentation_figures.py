"""Generate all presentation figures from the 1k pilot experiment results.

Reads the saved CSVs from data/processed/sonics_results/ and produces
publication-ready figures in reports/figures/.

Usage
-----
    python scripts/generate_presentation_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("data/processed/sonics_results")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.2)
REAL_COLOR = "#2196F3"
FAKE_COLOR = "#F44336"
PALETTE = {"Real": REAL_COLOR, "Fake": FAKE_COLOR}

EMBEDDING_ORDER = ["encodec", "mert", "clap", "muq"]
EMBEDDING_LABELS = {"encodec": "EnCodec", "mert": "MERT", "clap": "CLAP", "muq": "MuQ"}
ESTIMATOR_ORDER = ["phd", "twonn", "mle"]
ESTIMATOR_LABELS = {"phd": "PHD", "twonn": "TwoNN", "mle": "MLE"}


def load_id_results() -> dict[str, pd.DataFrame]:
    """Load per-embedding ID result CSVs."""
    dfs = {}
    for emb in EMBEDDING_ORDER:
        path = RESULTS_DIR / f"id_results_{emb}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["source_label"] = df["label"].map({"real": "Real", "fake": "Fake"})
            dfs[emb] = df
    return dfs


def load_fakeprint_results() -> pd.DataFrame | None:
    """Load fakeprint baseline results."""
    path = RESULTS_DIR / "fakeprint_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["source_label"] = df["label"].map({"real": "Real", "fake": "Fake"})
        return df
    return None


# ---------------------------------------------------------------------------
# Figure 1: Violin plots — ID distributions per embedding × estimator
# ---------------------------------------------------------------------------
def fig1_violin_plots(dfs: dict[str, pd.DataFrame]) -> None:
    """4×3 grid of violin plots: embedding × estimator."""
    available_embs = [e for e in EMBEDDING_ORDER if e in dfs]
    n_embs = len(available_embs)

    fig, axes = plt.subplots(n_embs, 3, figsize=(16, 4 * n_embs))
    if n_embs == 1:
        axes = axes[np.newaxis, :]

    for row, emb in enumerate(available_embs):
        df = dfs[emb]
        for col, est in enumerate(ESTIMATOR_ORDER):
            ax = axes[row, col]
            col_name = f"id_{est}"
            if col_name not in df.columns:
                ax.set_visible(False)
                continue

            data = df[[col_name, "source_label"]].dropna()
            if data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Filter extreme MLE outliers for visualization
            if est == "mle":
                p99 = data[col_name].quantile(0.99)
                data = data[data[col_name] <= p99]

            sns.violinplot(
                data=data,
                x="source_label",
                y=col_name,
                hue="source_label",
                palette=PALETTE,
                inner="box",
                legend=False,
                ax=ax,
                order=["Real", "Fake"],
            )

            # Add means as horizontal lines
            for label, color in PALETTE.items():
                mean_val = data[data["source_label"] == label][col_name].mean()
                ax.axhline(mean_val, color=color, linestyle="--", alpha=0.7, linewidth=1)

            ax.set_xlabel("")
            ax.set_ylabel("Intrinsic Dimension" if col == 0 else "")
            ax.set_title(f"{EMBEDDING_LABELS[emb]} — {ESTIMATOR_LABELS[est]}")

    fig.suptitle("Intrinsic Dimension Distributions: Real vs AI-Generated Music", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_violin_all.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_violin_all.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_violin_all.png/pdf")


# ---------------------------------------------------------------------------
# Figure 2: Effect size heatmap — Cohen's d per embedding × estimator
# ---------------------------------------------------------------------------
def fig2_effect_size_heatmap(dfs: dict[str, pd.DataFrame]) -> None:
    """Heatmap of Cohen's d values."""
    data = {}
    for emb in EMBEDDING_ORDER:
        if emb not in dfs:
            continue
        df = dfs[emb]
        row = {}
        for est in ESTIMATOR_ORDER:
            col_name = f"id_{est}"
            if col_name not in df.columns:
                row[ESTIMATOR_LABELS[est]] = np.nan
                continue
            real = df[df["label"] == "real"][col_name].dropna().values
            fake = df[df["label"] == "fake"][col_name].dropna().values
            if len(real) < 2 or len(fake) < 2:
                row[ESTIMATOR_LABELS[est]] = np.nan
                continue
            n1, n2 = len(real), len(fake)
            pooled = np.sqrt(((n1 - 1) * np.var(real, ddof=1) + (n2 - 1) * np.var(fake, ddof=1)) / (n1 + n2 - 2))
            d = (np.mean(real) - np.mean(fake)) / pooled if pooled > 1e-10 else 0
            row[ESTIMATOR_LABELS[est]] = d
        data[EMBEDDING_LABELS[emb]] = row

    df_heatmap = pd.DataFrame(data).T
    df_heatmap = df_heatmap[["PHD", "TwoNN", "MLE"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        vmin=-1.2,
        vmax=0.4,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Cohen's d  (negative = Fake > Real)"},
    )
    ax.set_title("Effect Size (Cohen's d) — Embedding × Estimator", fontsize=14)
    ax.set_ylabel("Embedding Model")
    ax.set_xlabel("ID Estimator")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_effect_size_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_effect_size_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_effect_size_heatmap.png/pdf")


# ---------------------------------------------------------------------------
# Figure 3: Best violin plot — EnCodec TwoNN (largest effect size)
# ---------------------------------------------------------------------------
def fig3_best_result_detail(dfs: dict[str, pd.DataFrame]) -> None:
    """Detailed plot for the best result: EnCodec × TwoNN."""
    if "encodec" not in dfs:
        return
    df = dfs["encodec"]
    col = "id_twonn"
    if col not in df.columns:
        return

    data = df[[col, "source_label"]].dropna()
    real = data[data["source_label"] == "Real"][col].values
    fake = data[data["source_label"] == "Fake"][col].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Violin + strip
    sns.violinplot(
        data=data,
        x="source_label",
        y=col,
        hue="source_label",
        palette=PALETTE,
        inner="box",
        legend=False,
        ax=axes[0],
        order=["Real", "Fake"],
    )
    axes[0].set_title("EnCodec — TwoNN  (d = -0.971)", fontsize=13)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Intrinsic Dimension (TwoNN)")

    # Overlapping histograms
    axes[1].hist(real, bins=30, alpha=0.6, color=REAL_COLOR, label=f"Real (μ={np.mean(real):.1f})", density=True)
    axes[1].hist(fake, bins=30, alpha=0.6, color=FAKE_COLOR, label=f"Fake (μ={np.mean(fake):.1f})", density=True)
    axes[1].axvline(np.mean(real), color=REAL_COLOR, linestyle="--", linewidth=2)
    axes[1].axvline(np.mean(fake), color=FAKE_COLOR, linestyle="--", linewidth=2)
    axes[1].set_xlabel("Intrinsic Dimension (TwoNN)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution Overlap", fontsize=13)
    axes[1].legend()

    fig.suptitle("Best Signal: EnCodec + TwoNN — Large Effect Size", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_best_encodec_twonn.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_best_encodec_twonn.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_best_encodec_twonn.png/pdf")


# ---------------------------------------------------------------------------
# Figure 4: ROC curves — one per embedding
# ---------------------------------------------------------------------------
def fig4_roc_curves(dfs: dict[str, pd.DataFrame]) -> None:
    """ROC curves for logistic regression on [PHD, TwoNN, MLE] per embedding."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = sns.color_palette("husl", len(dfs))

    for (emb, df), color in zip(dfs.items(), colors):
        id_cols = [c for c in ["id_phd", "id_twonn", "id_mle"] if c in df.columns]
        if not id_cols:
            continue

        valid = df.dropna(subset=id_cols)
        X = valid[id_cols].values.astype(np.float32)
        y = (valid["label"] == "fake").astype(int).values

        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        if len(np.unique(y)) < 2 or len(y) < 20:
            continue

        # Cross-validated predictions for ROC
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)

        ax.plot(fpr, tpr, color=color, lw=2, label=f"{EMBEDDING_LABELS[emb]} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — ID-Based Classification (Logistic Regression)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_roc_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_roc_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_roc_curves.png/pdf")


# ---------------------------------------------------------------------------
# Figure 5: Summary bar chart — accuracy / F1 / AUC per embedding
# ---------------------------------------------------------------------------
def fig5_classification_summary(dfs: dict[str, pd.DataFrame]) -> None:
    """Grouped bar chart of classification metrics per embedding."""
    results = []
    for emb in EMBEDDING_ORDER:
        if emb not in dfs:
            continue
        df = dfs[emb]
        id_cols = [c for c in ["id_phd", "id_twonn", "id_mle"] if c in df.columns]
        if not id_cols:
            continue

        valid = df.dropna(subset=id_cols)
        X = valid[id_cols].values.astype(np.float32)
        y = (valid["label"] == "fake").astype(int).values
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        if len(np.unique(y)) < 2 or len(y) < 20:
            continue

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        results.append({"Embedding": EMBEDDING_LABELS[emb], "Accuracy": acc, "F1": f1, "AUC": auc})

    if not results:
        return

    df_metrics = pd.DataFrame(results)
    df_melted = df_metrics.melt(id_vars="Embedding", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melted, x="Embedding", y="Score", hue="Metric", ax=ax, palette="Set2")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Classification Performance — ID Features + Logistic Regression", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xlabel("")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_classification_summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_classification_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5_classification_summary.png/pdf")


# ---------------------------------------------------------------------------
# Figure 6: Fakeprint comparison
# ---------------------------------------------------------------------------
def fig6_fakeprint(df_fp: pd.DataFrame | None) -> None:
    """Violin plot of fakeprint mean energy."""
    if df_fp is None or "fakeprint_mean" not in df_fp.columns:
        print("  Skipping fig6 — no fakeprint data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin
    sns.violinplot(
        data=df_fp,
        x="source_label",
        y="fakeprint_mean",
        hue="source_label",
        palette=PALETTE,
        inner="box",
        legend=False,
        ax=axes[0],
        order=["Real", "Fake"],
    )
    axes[0].set_title("Fakeprint Mean Energy", fontsize=13)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean Spectral Residual Energy")

    # Histogram
    real_fp = df_fp[df_fp["label"] == "real"]["fakeprint_mean"].dropna()
    fake_fp = df_fp[df_fp["label"] == "fake"]["fakeprint_mean"].dropna()
    axes[1].hist(real_fp, bins=30, alpha=0.6, color=REAL_COLOR, label=f"Real (μ={real_fp.mean():.4f})", density=True)
    axes[1].hist(fake_fp, bins=30, alpha=0.6, color=FAKE_COLOR, label=f"Fake (μ={fake_fp.mean():.4f})", density=True)
    axes[1].set_xlabel("Fakeprint Mean Energy")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution", fontsize=13)
    axes[1].legend()

    fig.suptitle("Fakeprint Baseline (Fourier Spectral Artifacts)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_fakeprint.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig6_fakeprint.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig6_fakeprint.png/pdf")


# ---------------------------------------------------------------------------
# Figure 7: Results summary table as image
# ---------------------------------------------------------------------------
def fig7_results_table(dfs: dict[str, pd.DataFrame]) -> None:
    """Render the full results table as a figure."""
    from scipy.stats import mannwhitneyu

    rows = []
    for emb in EMBEDDING_ORDER:
        if emb not in dfs:
            continue
        df = dfs[emb]
        for est in ESTIMATOR_ORDER:
            col = f"id_{est}"
            if col not in df.columns:
                continue
            real = df[df["label"] == "real"][col].dropna().values
            fake = df[df["label"] == "fake"][col].dropna().values
            if len(real) < 2 or len(fake) < 2:
                continue

            r_mean, r_std = np.mean(real), np.std(real)
            f_mean, f_std = np.mean(fake), np.std(fake)

            n1, n2 = len(real), len(fake)
            pooled = np.sqrt(((n1 - 1) * np.var(real, ddof=1) + (n2 - 1) * np.var(fake, ddof=1)) / (n1 + n2 - 2))
            d = (r_mean - f_mean) / pooled if pooled > 1e-10 else 0

            _, p = mannwhitneyu(real, fake, alternative="two-sided")

            direction = "Real > Fake" if d > 0 else "Fake > Real"

            rows.append(
                {
                    "Embedding": EMBEDDING_LABELS[emb],
                    "Estimator": ESTIMATOR_LABELS[est],
                    "Real (μ±σ)": f"{r_mean:.2f} ± {r_std:.2f}",
                    "Fake (μ±σ)": f"{f_mean:.2f} ± {f_std:.2f}",
                    "Cohen's d": f"{d:.3f}",
                    "Direction": direction,
                    "p-value": f"{p:.2e}",
                }
            )

    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, 0.4 * len(rows) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color-code Cohen's d cells
    for i, row in enumerate(rows):
        d_val = float(row["Cohen's d"])
        cell = table[i + 1, 4]  # Cohen's d column
        if abs(d_val) >= 0.8:
            cell.set_facecolor("#FFCDD2")  # red for large
        elif abs(d_val) >= 0.5:
            cell.set_facecolor("#FFF9C4")  # yellow for medium
        else:
            cell.set_facecolor("#E8F5E9")  # green for small

    # Header styling
    for j in range(len(df_table.columns)):
        table[0, j].set_facecolor("#1976D2")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Pilot Experiment Results (N=1,000 per class)", fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_results_table.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig7_results_table.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig7_results_table.png/pdf")


# ---------------------------------------------------------------------------
# Figure 8: Per-embedding comparison — PHD + TwoNN only (drop MLE)
# ---------------------------------------------------------------------------
def fig8_phd_twonn_comparison(dfs: dict[str, pd.DataFrame]) -> None:
    """Side-by-side box plots for PHD and TwoNN across all embeddings."""
    rows = []
    for emb in EMBEDDING_ORDER:
        if emb not in dfs:
            continue
        df = dfs[emb]
        for est in ["phd", "twonn"]:
            col = f"id_{est}"
            if col not in df.columns:
                continue
            for _, r in df[[col, "label"]].dropna().iterrows():
                rows.append(
                    {
                        "Embedding": EMBEDDING_LABELS[emb],
                        "Estimator": ESTIMATOR_LABELS[est],
                        "ID": r[col],
                        "Source": "Real" if r["label"] == "real" else "Fake",
                    }
                )

    if not rows:
        return

    df_plot = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, est in enumerate(["PHD", "TwoNN"]):
        sub = df_plot[df_plot["Estimator"] == est]
        if sub.empty:
            continue
        sns.boxplot(
            data=sub,
            x="Embedding",
            y="ID",
            hue="Source",
            palette=PALETTE,
            ax=axes[i],
            showfliers=False,
        )
        axes[i].set_title(f"{est} — Across Embedding Models", fontsize=13)
        axes[i].set_ylabel("Intrinsic Dimension")
        axes[i].set_xlabel("")

    fig.suptitle("ID Comparison: Stable Estimators (PHD & TwoNN)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_phd_twonn_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig8_phd_twonn_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig8_phd_twonn_comparison.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading results...")
    dfs = load_id_results()
    df_fp = load_fakeprint_results()

    if not dfs:
        print("ERROR: No ID result CSVs found in", RESULTS_DIR)
        print("Download them from EC2 first:")
        print(
            "  scp ubuntu@EC2:~/intrinsic-ai-music-detection/data/processed/sonics_results/*.csv data/processed/sonics_results/"
        )
        return

    print(f"Found results for: {', '.join(EMBEDDING_LABELS[e] for e in dfs)}")
    print(f"Fakeprint data: {'yes' if df_fp is not None else 'no'}")
    print(f"\nGenerating figures in {FIGURES_DIR}/\n")

    fig1_violin_plots(dfs)
    fig2_effect_size_heatmap(dfs)
    fig3_best_result_detail(dfs)
    fig4_roc_curves(dfs)
    fig5_classification_summary(dfs)
    fig6_fakeprint(df_fp)
    fig7_results_table(dfs)
    fig8_phd_twonn_comparison(dfs)

    print(f"\nDone! All figures saved to {FIGURES_DIR}/")
    print("\nFigure inventory:")
    for f in sorted(FIGURES_DIR.glob("fig*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
