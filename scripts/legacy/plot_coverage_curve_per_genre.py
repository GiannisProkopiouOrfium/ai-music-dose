"""Plot the per-genre manifold-coverage / data-efficiency scaling curves.

`coverage_curve.py` writes `coverage_curve_per_genre.csv` (and, for the overall
sweep only, `coverage_curve_overall.png`) but never plots the per-genre sweep —
this script fills that gap directly from the already-computed CSV, so it does
NOT require re-running any experiment (CPU-only, seconds).

Columns expected in the CSV (as produced by coverage_curve.py):
    genre, n_genre_tracks, n_total_real, heldout_anomaly_mean,
    heldout_anomaly_std, n_heldout

Usage
-----
python scripts/plot_coverage_curve_per_genre.py \\
    --csv data/processed/coverage_curve/coverage_curve_per_genre.csv \\
    --output-dir reports/figures/coverage_curve
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to coverage_curve_per_genre.csv")
    ap.add_argument("--output-dir", default="reports/figures/coverage_curve")
    ap.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.05,
        help=(
            "Relative change (fraction of the genre's overall anomaly-mean range) below which "
            "consecutive points are considered 'saturated'. Used to flag genres that have NOT "
            "yet saturated within the tested n_genre_tracks range."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = {"genre", "n_genre_tracks", "heldout_anomaly_mean", "heldout_anomaly_std"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {args.csv} is missing expected columns: {missing}")

    genres = sorted(df["genre"].unique())
    logger.info("Plotting per-genre coverage curves for %d genres: %s", len(genres), genres)

    # --- Combined multi-panel figure: anomaly mean (with std band) vs n_genre_tracks ---
    n_cols = 3
    n_rows = -(-len(genres) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

    not_saturated = []
    for i, genre in enumerate(genres):
        ax = axes[i // n_cols][i % n_cols]
        sub = df[df["genre"] == genre].sort_values("n_genre_tracks")
        x = sub["n_genre_tracks"].to_numpy()
        y = sub["heldout_anomaly_mean"].to_numpy()
        yerr = sub["heldout_anomaly_std"].to_numpy()

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, color="tab:blue")
        ax.set_xscale("log")
        ax.set_xlabel("n real tracks (this genre)")
        ax.set_ylabel("Held-out anomaly mean")
        ax.set_title(genre, fontsize=10)

        # Saturation check: relative change between the last two points vs. the
        # full observed range across all tested sizes for this genre.
        y_range = max(y.max() - y.min(), 1e-6)
        if len(y) >= 2:
            last_delta = abs(y[-1] - y[-2]) / y_range
            if last_delta > args.saturation_threshold:
                not_saturated.append(genre)
                ax.text(
                    0.02,
                    0.95,
                    "NOT SATURATED",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="red",
                    va="top",
                    fontweight="bold",
                )

    # Hide unused axes
    for j in range(len(genres), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        "Per-genre manifold-coverage scaling: held-out real anomaly score vs. training-set size\n"
        "(flat curve = saturated; still-sloping curve = genre remains data-limited at the tested sizes)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "coverage_curve_per_genre.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)

    # --- Single overlay figure: all genres on one normalized-x-axis plot ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for genre in genres:
        sub = df[df["genre"] == genre].sort_values("n_genre_tracks")
        ax2.semilogx(sub["n_genre_tracks"], sub["heldout_anomaly_mean"], "o-", label=genre, alpha=0.8)
    ax2.set_xlabel("n real tracks (this genre)")
    ax2.set_ylabel("Held-out anomaly mean (log-likelihood; less negative = better fit)")
    ax2.set_title("Per-genre coverage curves (overlay)")
    ax2.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path2 = out_dir / "coverage_curve_per_genre_overlay.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    logger.info("Saved %s", out_path2)

    if not_saturated:
        logger.warning(
            "Genres NOT clearly saturated within the tested n_genre_tracks range (last-step relative "
            "change > %.0f%% of the genre's observed range): %s -- extend the sweep to larger "
            "n_genre_tracks for these before concluding they are intrinsically hard rather than still "
            "data-limited.",
            args.saturation_threshold * 100,
            not_saturated,
        )
    else:
        logger.info("All genres appear saturated within the tested range.")


if __name__ == "__main__":
    main()
