"""Figures for the NMF fakeprint detector: the atoms, and the space they induce.

Three figures, each answering a question a reviewer will ask.

1. **atoms.png** — the learned dictionary W, sorted by how much each atom loses
   under blurring. This is the interpretability figure: if the top atoms are
   visibly periodic combs, the detector is reading a decoder signature and the
   mechanism claim is *shown*, not asserted. Afchar et al. make the same argument
   with their logistic-regression weights; ours needs no labels to produce.

2. **activations.png** — a 2-D embedding of the NMF activations H, coloured by
   generator. Structure here means the unsupervised dictionary separates
   generators without ever seeing a label, which is their Task B (clustering)
   result reproduced on FakeMusicCaps / SONICS.

3. **score_hist.png** — the score distribution per class with the reals-only
   threshold drawn on it. This is what the operating point looks like, and it
   makes the overlap (the false-positive cost) visible.

No seaborn, no styling dependencies — matplotlib only, so it runs on the box.

Usage (EC2)
-----------
    python scripts/visualize_nmf_space.py \\
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \\
        --per-stratum 300 --workers 6 --max-duration 9 --level-match \\
        --n-atoms 20 --blur-sigma 2 \\
        --out-dir reports/figures/nmf_fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_nmf_peak_detector import _collect  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("visualize_nmf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-column", default="canonical_path")
    # Required by eval_nmf_peak_detector._collect, which this script reuses. The
    # dependency was added in R10.3 (--carry-columns, for the per-genre table) and
    # this argparse was not updated, so every invocation died with
    # "AttributeError: 'Namespace' object has no attribute 'carry_columns'".
    parser.add_argument(
        "--carry-columns",
        nargs="*",
        default=["genre"],
        help="Manifest columns to copy through _collect; kept in sync with "
        "eval_nmf_peak_detector.py so the shared collector cannot drift again.",
    )
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=300,
        help="Tracks per label x algorithm. NOTE: unlike the eval_* scripts, 0 means "
        "ZERO here, not ALL — this script samples for a figure, not a corpus run.",
    )
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument("--f-min", type=float, default=1000.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--n-atoms", type=int, default=20)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--n-bins", type=int, default=0, help="0 = full resolution.")
    parser.add_argument("--top-atoms", type=int, default=6, help="How many atoms to plot.")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import NMF

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")
    rng = np.random.RandomState(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], dropna=False):
        n = min(len(grp), args.per_stratum)
        parts.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
    df = pd.concat(parts, ignore_index=True)

    per_track = _collect(df, args, args.audio_column)
    X = np.stack(per_track["profile"].to_numpy(), axis=0).astype(np.float64)
    if args.n_bins and args.n_bins != X.shape[1]:
        grid = np.linspace(0, X.shape[1] - 1, args.n_bins)
        X = np.stack([np.interp(grid, np.arange(X.shape[1]), r) for r in X], axis=0)
    logger.info("fakeprint matrix %s", X.shape)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmf = NMF(n_components=args.n_atoms, init="nndsvda", random_state=args.seed, max_iter=600)
        H = nmf.fit_transform(np.clip(X, 0.0, None))
    W = nmf.components_
    W_blur = gaussian_filter1d(W, sigma=args.blur_sigma, axis=1)
    scores = np.linalg.norm(H @ (W - W_blur), axis=1)

    freqs = np.linspace(args.f_min, args.f_max, X.shape[1])
    labels = per_track["label"].to_numpy()
    algos = per_track["algorithm"].to_numpy()
    is_real = labels == "real"

    # --- 1. the atoms, ordered by how much blurring destroys them -----------
    #
    # TWO statistics, and the distinction is the whole mechanism claim.
    #
    #   peakiness   = ||W - W*G|| / ||W||. This is the SCORE'S OWN SENSITIVITY —
    #                 how much blurring destroys the atom. A dense forest of
    #                 RANDOM spikes scores high on it. It says nothing about
    #                 whether the atom is a comb, and quoting it as evidence for
    #                 "the dictionary learns decoder combs" is a category error.
    #   periodicity = the normalised autocorrelation peak of the atom, i.e.
    #                 `comb_strength` — the same statistic the detector applies to
    #                 a track's residual. THIS is what tests the mechanism: a
    #                 decoder comb repeats at a fixed spacing, random spikes do not.
    #
    # Reported together, and the ordering is still by peakiness so the top panels
    # remain the atoms the score actually responds to.
    from intrinsic_ai_music_detection.features.comb_artifacts import comb_strength

    peakiness = np.linalg.norm(W - W_blur, axis=1) / (np.linalg.norm(W, axis=1) + 1e-9)
    bin_hz = (args.f_max - args.f_min) / max(X.shape[1] - 1, 1)
    periodicity = np.empty(len(W))
    spacing_hz = np.empty(len(W))
    for a in range(len(W)):
        s, sp, _ = comb_strength(W[a], bin_hz=bin_hz)
        periodicity[a], spacing_hz[a] = s, sp

    order = np.argsort(-peakiness)[: args.top_atoms]
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 1.5 * len(order)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, order):
        ax.plot(freqs, W[a], lw=0.6, color="#1f77b4")
        ax.plot(freqs, W_blur[a], lw=0.9, color="#d62728", alpha=0.8)
        ax.set_ylabel(
            f"atom {a}\npeak {peakiness[a]:.2f}\nper {periodicity[a]:.2f}" f"\n@{spacing_hz[a]:.0f}Hz",
            fontsize=6,
        )
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("frequency (Hz)")
    axes[0].set_title(
        "NMF dictionary atoms (blue) vs Gaussian-blurred (red).\n"
        "The score is the norm of the gap — periodic atoms are decoder combs.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "atoms.png", dpi=160)
    plt.close(fig)

    # --- 2. the activation space -------------------------------------------
    try:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=args.seed).fit_transform(H / (H.sum(axis=1, keepdims=True) + 1e-9))
        method = "PCA"
        try:
            import umap  # noqa: F401
            from umap import UMAP

            emb = UMAP(n_components=2, random_state=args.seed).fit_transform(H / (H.sum(axis=1, keepdims=True) + 1e-9))
            method = "UMAP"
        except Exception:
            logger.info("umap unavailable — using PCA for the activation embedding")

        fig, ax = plt.subplots(figsize=(7, 6))
        for name in ["real"] + sorted(set(algos[~is_real]) - {""}):
            m = is_real if name == "real" else (algos == name)
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.55, label=name)
        ax.legend(fontsize=7, markerscale=2)
        ax.set_title(
            f"NMF activations ({method}), coloured by generator.\n"
            "The dictionary is UNSUPERVISED — any separation here is label-free.",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "activations.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("activation embedding skipped: %s", exc)

    # --- 3. scores and the reals-only threshold ----------------------------
    thr = float(np.quantile(scores[is_real], 0.95))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(float(np.nanmin(scores)), float(np.nanpercentile(scores, 99.5)), 70)
    ax.hist(scores[is_real], bins=bins, alpha=0.6, label="real", color="#2ca02c", density=True)
    ax.hist(scores[~is_real], bins=bins, alpha=0.6, label="generated", color="#d62728", density=True)
    ax.axvline(thr, color="k", ls="--", lw=1.2, label="95% real quantile (reals-only threshold)")
    ax.set_xlabel("peak-energy score  r_i = || H_i W  -  H_i (W*G) ||")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("Peak-energy score. The threshold uses reals only — no fake label.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "score_hist.png", dpi=160)
    plt.close(fig)

    pd.DataFrame(
        {
            "atom": np.arange(len(peakiness)),
            "peakiness": np.round(peakiness, 4),
            "periodicity": np.round(periodicity, 4),
            "spacing_hz": np.round(spacing_hz, 1),
        }
    ).sort_values("peakiness", ascending=False).to_csv(out_dir / "atom_peakiness.csv", index=False)

    logger.info("Wrote atoms.png, activations.png, score_hist.png, atom_peakiness.csv → %s", out_dir)
    logger.info(
        "Top atoms by PEAKINESS (the score's own sensitivity): %s",
        np.round(peakiness[order], 3).tolist(),
    )
    logger.info(
        "Their PERIODICITY (comb_strength — the mechanism test): %s at spacings %s Hz",
        np.round(periodicity[order], 3).tolist(),
        np.round(spacing_hz[order], 0).tolist(),
    )
    logger.info(
        "Peakiness alone does NOT support 'the dictionary learns decoder combs' — a forest "
        "of random spikes is peaky too. Periodicity is the statistic that does, and it is "
        "the same one the detector applies to a track. Quote periodicity in the paper."
    )


if __name__ == "__main__":
    main()
