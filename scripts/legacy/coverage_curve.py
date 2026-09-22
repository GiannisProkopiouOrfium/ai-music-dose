"""Manifold-coverage / data-efficiency scaling curves for the RealNVP flow.

This script sweeps the number of real training tracks — overall and per-genre
— and measures detection AUC and FMA FPR at each point.  The result is a
"scaling law" curve that answers:
  - How many real tracks are needed for the manifold to generalize?
  - How many tracks per genre are needed for within-genre AUC to stabilize?
  - What is the minimum genre coverage that avoids the 15.3% FMA FPR issue?

This is a novel contribution (Plan Phase B3): no prior work on audio forensics
has quantified the manifold-coverage requirement for density-estimation detectors.

Protocol
--------
1. For N in {200, 500, 1000, 2000, 4000, 8000, 12722}:
   - Sample N real tracks from the full SONICS real set (or combined corpus).
   - Train a RealNVP flow on their windows.
   - Score SONICS fake tracks (LOGO: held-out by generator) -> AUC/EER.
   - Score FMA-small real tracks -> FPR.
2. For each genre G and n_tracks in {10, 20, 50, 100, 200}:
   - Sample n_tracks from genre G.
   - Train flow on those n_tracks + baseline 2000 from all other genres.
   - Score held-out real tracks of genre G -> manifold quality proxy.
   - Score fake tracks of genre G (if available) -> within-genre AUC.

Outputs
-------
  coverage_curve_overall.csv      overall n_tracks -> AUC/EER/FPR
  coverage_curve_per_genre.csv    per-genre n_tracks -> within-genre AUC/FPR
  coverage_curve_overall.png      scaling law plot
  coverage_curve_per_genre.png    per-genre plot

Usage (EC2)
-----------
# Overall sweep only (3-4 h on T4):
poetry run python scripts/coverage_curve.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --fma-scores-csv data/processed/external_scores/fma_fpr/external_scores.csv \\
    --output-dir data/processed/coverage_curve \\
    --device cuda \\
    --mode overall

# Per-genre sweep (requires genre tags from tag_all_genres.py):
poetry run python scripts/coverage_curve.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --output-dir data/processed/coverage_curve \\
    --device cuda \\
    --mode per_genre
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, equal_error_rate
from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ENCODEC_FPS = 75
ENCODEC_SR = 24_000
WINDOW_DUR = 4.0
HOP_DUR = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_cache_health(
    track_ids: list[str],
    emb_cache: Path,
    max_duration: float,
    n_probe: int = 50,
    min_hit_rate: float = 0.5,
) -> None:
    """Fail fast with a clear message if the embedding cache key doesn't match.

    Silently proceeding with a 0%-hit-rate cache (e.g. due to a max-duration or
    preprocess-mode mismatch vs. how the cache was originally built) would waste
    hours before producing empty/garbage results. Probe a small sample up front.
    """
    probe_ids = track_ids[:n_probe]
    hits = 0
    for tid in probe_ids:
        key = hashlib.md5(f"{tid}_{ENCODEC_SR}_{max_duration}_preprocessed".encode()).hexdigest()
        if (emb_cache / "encodec" / f"{key}.npy").exists():
            hits += 1
    hit_rate = hits / max(len(probe_ids), 1)
    logger.info(
        "Cache health check: %d/%d probed tracks hit the cache (%.0f%%) at %s",
        hits,
        len(probe_ids),
        100 * hit_rate,
        emb_cache / "encodec",
    )
    if hit_rate < min_hit_rate:
        logger.error(
            "Cache hit rate %.0f%% is below the %.0f%% threshold. This usually means "
            "--max-duration (currently %.1f) or the cache directory does not match how "
            "the cache was built (see run_full_sonics_pipeline.sh: --analysis-duration, "
            "--preprocess-mode preprocessed). Aborting before wasting compute. "
            "Fix --max-duration/--emb-cache and retry.",
            100 * hit_rate,
            100 * min_hit_rate,
            max_duration,
        )
        sys.exit(1)


from intrinsic_ai_music_detection.features.pooling import pool_windows as _pool_windows  # noqa: E402


def _load_track_windows(
    track_id: str,
    emb_cache: Path,
    max_duration: float,
    window_frames: int,
    hop_frames: int,
) -> np.ndarray | None:
    """Load pre-computed EnCodec windows for a track from the embedding cache."""
    key = hashlib.md5(f"{track_id}_{ENCODEC_SR}_{max_duration}_preprocessed".encode()).hexdigest()
    cp = emb_cache / "encodec" / f"{key}.npy"
    if not cp.exists():
        return None
    try:
        mat = np.load(str(cp))
        pw = _pool_windows(mat, window_frames, hop_frames)
        return pw if len(pw) >= 2 else None
    except Exception:
        return None


def _train_flow_on_tracks(
    track_ids: list[str],
    emb_cache: Path,
    max_duration: float,
    window_frames: int,
    hop_frames: int,
    device: str,
    seed: int,
    flow_epochs: int = 200,
    n_coupling_layers: int = 12,
    pca_components: int = 0,
) -> tuple[RealNVPOneClass, list[np.ndarray]]:
    """Train a RealNVP flow on the given track IDs. Returns (flow, list_of_window_arrays).

    Defaults match the ADOPTED HEADLINE configuration (K=12, 200 epochs,
    patience 20) so the curve characterizes the model actually being reported —
    the previous K=8/100-epoch defaults measured a different model.

    ``pca_components > 0`` fits a PCA on the TRAINING windows and attaches it to
    the flow (see RealNVPOneClass.attach_pca), so downstream scoring of raw-dim
    windows transforms automatically. Used to replicate e.g. the FakeMusicCaps
    K=2 recipe (--wf-pca-components 64).
    """
    windows_list: list[np.ndarray] = []
    for tid in track_ids:
        pw = _load_track_windows(tid, emb_cache, max_duration, window_frames, hop_frames)
        if pw is not None and len(pw) >= 2:
            windows_list.append(pw)

    if not windows_list:
        raise ValueError("No valid windows found for any track ID")

    x_train = np.concatenate(windows_list, axis=0)
    pca = None
    if pca_components and x_train.shape[1] > pca_components and len(x_train) > pca_components:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_components, random_state=seed)
        x_train = pca.fit_transform(x_train)
    cfg = RealNVPConfig(
        device=device,
        n_epochs=flow_epochs,
        patience=20,
        seed=seed,
        verbose=False,
        n_coupling_layers=n_coupling_layers,
    )
    flow = RealNVPOneClass(cfg).fit(x_train)
    if pca is not None:
        flow.attach_pca(pca)
    return flow, windows_list


def _score_tracks(
    track_ids: list[str],
    emb_cache: Path,
    flow: RealNVPOneClass,
    max_duration: float,
    window_frames: int,
    hop_frames: int,
) -> np.ndarray:
    """Score a list of tracks with the flow. Returns per-track anomaly scores."""
    scores = []
    for tid in track_ids:
        pw = _load_track_windows(tid, emb_cache, max_duration, window_frames, hop_frames)
        if pw is None:
            scores.append(float("nan"))
            continue
        lls = -flow.score_samples(pw)  # NLL negated = loglik
        finite = lls[np.isfinite(lls)]
        if len(finite) < 2:
            scores.append(float("nan"))
            continue
        scores.append(float(-finite.mean()))  # anomaly = -mean_loglik
    return np.array(scores)


# ---------------------------------------------------------------------------
# Overall sweep
# ---------------------------------------------------------------------------


def run_overall_sweep(
    manifest: pd.DataFrame,
    wf_df: pd.DataFrame,
    emb_cache: Path,
    output_dir: Path,
    device: str,
    seed: int,
    max_duration: float,
    n_real_sizes: list[int],
    flow_epochs: int,
    n_coupling_layers: int,
    n_heldout_real: int,
    pca_components: int = 0,
    fma_manifest: pd.DataFrame | None = None,
    fma_emb_cache: Path | None = None,
    fma_max_duration: float | None = None,
) -> pd.DataFrame:
    """Sweep overall number of real tracks -> AUC/EER/FPR curve.

    PROTOCOL (corrected): a fixed HELD-OUT real evaluation set (size
    ``n_heldout_real``) is reserved up front, disjoint from every training
    prefix. Each sweep point trains on the first ``n_real`` tracks of the
    remaining pool and evaluates real-vs-fake AUC using ONLY the held-out
    reals. (The previous version scored ALL reals — including the training
    sample — so every AUC was partly in-sample, with the bias growing with
    n_real, i.e. exactly along the measured axis. Numbers from that version
    are not citable.)

    FMA FPR (corrected): only computed when FMA tracks can be scored by THIS
    sweep point's own flow (``fma_manifest`` + ``fma_emb_cache``). The previous
    version thresholded scores produced by a DIFFERENT frozen flow against this
    flow's real distribution — NLLs across independently trained flows share no
    scale, so those fma_fpr_1pct values were meaningless.
    """
    window_frames = max(int(WINDOW_DUR * ENCODEC_FPS), 5)
    hop_frames = max(int(HOP_DUR * ENCODEC_FPS), 1)

    real_ids = manifest.loc[manifest["label"] == "real", "track_id"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    real_ids_shuffled = rng.permutation(real_ids).tolist()

    # Reserve the held-out real eval set FIRST — disjoint from all training prefixes.
    n_heldout = min(n_heldout_real, max(len(real_ids_shuffled) // 5, 100))
    heldout_real_ids = real_ids_shuffled[-n_heldout:]
    train_pool = real_ids_shuffled[:-n_heldout]
    logger.info(
        "Overall sweep: %d real tracks → %d train-pool + %d HELD-OUT eval reals (disjoint)",
        len(real_ids_shuffled),
        len(train_pool),
        len(heldout_real_ids),
    )

    fma_ids: list[str] | None = None
    if fma_manifest is not None and fma_emb_cache is not None:
        fma_ids = fma_manifest["track_id"].astype(str).tolist()
        logger.info("FMA FPR enabled: %d FMA tracks will be scored by each sweep flow", len(fma_ids))

    rows = []
    for n_real in n_real_sizes:
        if n_real > len(train_pool):
            logger.info("Skipping n_real=%d (only %d in train pool after held-out split)", n_real, len(train_pool))
            continue

        sample_ids = train_pool[:n_real]
        logger.info("\n[sweep] n_real=%d  training flow (K=%d, %d epochs)...", n_real, n_coupling_layers, flow_epochs)

        try:
            flow, _ = _train_flow_on_tracks(
                sample_ids,
                emb_cache,
                max_duration,
                window_frames,
                hop_frames,
                device,
                seed + n_real,
                flow_epochs,
                n_coupling_layers=n_coupling_layers,
                pca_components=pca_components,
            )
        except Exception as exc:
            logger.warning("Flow training failed for n_real=%d: %s", n_real, exc)
            continue

        row: dict = {"n_real_tracks": n_real, "n_heldout_real": len(heldout_real_ids)}
        real_scores = _score_tracks(heldout_real_ids, emb_cache, flow, max_duration, window_frames, hop_frames)
        real_fin = real_scores[np.isfinite(real_scores)]
        if len(real_fin) < 10:
            logger.warning("n_real=%d: only %d finite held-out real scores — skipping point", n_real, len(real_fin))
            continue

        for alg in sorted(wf_df[wf_df["label"] == "fake"]["algorithm"].dropna().unique()):
            fake_ids = (
                wf_df.loc[(wf_df["label"] == "fake") & (wf_df["algorithm"] == alg), "track_id"].astype(str).tolist()
            )
            fake_scores = _score_tracks(fake_ids, emb_cache, flow, max_duration, window_frames, hop_frames)
            fake_fin = fake_scores[np.isfinite(fake_scores)]
            if len(fake_fin) < 5:
                continue
            y = np.r_[np.zeros(len(real_fin)), np.ones(len(fake_fin))]
            sc = np.r_[real_fin, fake_fin]
            auc, eer = auc_and_eer(y, sc)
            row[f"auc_{alg}"] = round(auc, 4)
            row[f"eer_{alg}_pct"] = round(eer * 100, 2)

        # FPR on FMA — scored by THIS flow, thresholded on THIS flow's held-out reals.
        if fma_ids:
            fma_dur = fma_max_duration if fma_max_duration is not None else max_duration
            fma_anomaly = _score_tracks(fma_ids, fma_emb_cache, flow, fma_dur, window_frames, hop_frames)
            fma_fin = fma_anomaly[np.isfinite(fma_anomaly)]
            if len(fma_fin) >= 10:
                threshold_1pct = float(np.percentile(real_fin, 99))  # 1% FPR on held-out reals
                row["fma_fpr_1pct"] = round(float((fma_fin > threshold_1pct).mean()), 4)
                row["threshold_1pct"] = round(threshold_1pct, 2)
                row["n_fma_scored"] = int(len(fma_fin))

        rows.append(row)
        logger.info("  n_real=%d: %s", n_real, {k: v for k, v in row.items() if k != "n_real_tracks"})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "coverage_curve_overall.csv", index=False)
    logger.info("Saved coverage_curve_overall.csv")
    return df


# ---------------------------------------------------------------------------
# Per-genre sweep
# ---------------------------------------------------------------------------


def run_per_genre_sweep(
    manifest: pd.DataFrame,
    genres_df: pd.DataFrame,
    emb_cache: Path,
    output_dir: Path,
    device: str,
    seed: int,
    max_duration: float,
    n_per_genre_sizes: list[int],
    baseline_n: int,
    flow_epochs: int,
    n_coupling_layers: int = 12,
) -> pd.DataFrame:
    """Sweep per-genre track count -> within-genre AUC / FPR."""
    import zlib

    window_frames = max(int(WINDOW_DUR * ENCODEC_FPS), 5)
    hop_frames = max(int(HOP_DUR * ENCODEC_FPS), 1)

    real_manifest = manifest[manifest["label"] == "real"].copy()
    real_manifest["track_id"] = real_manifest["track_id"].astype(str)
    genres_df = genres_df[genres_df["label"] == "real"].copy()
    genres_df["track_id"] = genres_df["track_id"].astype(str)

    tagged = real_manifest.merge(genres_df[["track_id", "genre_top1"]], on="track_id", how="left")
    genre_groups = tagged.dropna(subset=["genre_top1"]).groupby("genre_top1")

    rows = []

    for genre, grp in genre_groups:
        # Stable per-genre seed: crc32 of the NAME (the old ``len(genre)``
        # collided for equal-length names, and a shared sequential RNG made
        # each genre's sample depend on how many genres preceded it).
        genre_seed = (seed + zlib.crc32(str(genre).encode())) % (2**32)
        g_rng = np.random.default_rng(genre_seed)

        genre_ids = grp["track_id"].tolist()
        # Require room for the largest training size PLUS the 50-track held-out
        # set, so heldout_ids can never overlap a training prefix.
        if len(genre_ids) < max(n_per_genre_sizes) + 50:
            continue

        # Baseline: all real tracks NOT in this genre
        other_ids = tagged.loc[tagged["genre_top1"] != genre, "track_id"].dropna().tolist()
        if len(other_ids) > baseline_n:
            other_ids = g_rng.choice(other_ids, size=baseline_n, replace=False).tolist()

        genre_ids_shuffled = g_rng.permutation(genre_ids).tolist()
        heldout_ids = genre_ids_shuffled[-50:]

        for n_per_genre in n_per_genre_sizes:
            if n_per_genre > len(genre_ids_shuffled) - 50:
                continue

            train_genre_ids = genre_ids_shuffled[:n_per_genre]
            train_ids = train_genre_ids + other_ids

            logger.info("  [per_genre] genre=%s  n_genre=%d  total=%d", genre, n_per_genre, len(train_ids))

            try:
                flow, _ = _train_flow_on_tracks(
                    train_ids,
                    emb_cache,
                    max_duration,
                    window_frames,
                    hop_frames,
                    device,
                    genre_seed + n_per_genre,
                    flow_epochs,
                    n_coupling_layers=n_coupling_layers,
                )
            except Exception as exc:
                logger.warning("    Flow failed: %s", exc)
                continue

            # Manifold quality: held-out real tracks of this genre should have
            # lower anomaly scores (i.e., flow recognizes them as real)
            heldout_scores = _score_tracks(heldout_ids, emb_cache, flow, max_duration, window_frames, hop_frames)
            heldout_fin = heldout_scores[np.isfinite(heldout_scores)]

            rows.append(
                {
                    "genre": genre,
                    "n_genre_tracks": n_per_genre,
                    "n_total_real": len(train_ids),
                    "heldout_anomaly_mean": float(np.mean(heldout_fin)) if len(heldout_fin) > 0 else float("nan"),
                    "heldout_anomaly_std": float(np.std(heldout_fin)) if len(heldout_fin) > 0 else float("nan"),
                    "n_heldout": len(heldout_fin),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "coverage_curve_per_genre.csv", index=False)
    logger.info("Saved coverage_curve_per_genre.csv")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_coverage_curves(overall_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the overall data-efficiency scaling curve."""
    try:
        import matplotlib.pyplot as plt

        auc_cols = [c for c in overall_df.columns if c.startswith("auc_")]
        eer_cols = [c for c in overall_df.columns if c.startswith("eer_")]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # AUC curves
        ax = axes[0]
        for col in auc_cols:
            alg = col.replace("auc_", "")
            ax.semilogx(overall_df["n_real_tracks"], overall_df[col], "o-", label=alg)
        ax.set_xlabel("Number of real training tracks")
        ax.set_ylabel("AUC")
        ax.set_title("Scaling Law: AUC vs Real Training Set Size")
        ax.legend(fontsize=7)
        ax.axhline(0.9, color="gray", ls="--", lw=1, alpha=0.5, label="AUC=0.90")
        ax.set_ylim(0.4, 1.0)

        # EER curves
        ax = axes[1]
        for col in eer_cols:
            alg = col.replace("eer_", "").replace("_pct", "")
            ax.semilogx(overall_df["n_real_tracks"], overall_df[col], "o-", label=alg)
        ax.set_xlabel("Number of real training tracks")
        ax.set_ylabel("EER (%)")
        ax.set_title("Scaling Law: EER vs Real Training Set Size")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 50)

        # FPR curve
        if "fma_fpr_1pct" in overall_df.columns:
            ax = axes[2]
            ax.semilogx(overall_df["n_real_tracks"], overall_df["fma_fpr_1pct"] * 100, "ko-")
            ax.axhline(5, color="green", ls="--", lw=1, label="5% FPR target")
            ax.axhline(15.3, color="red", ls="--", lw=1, label="SONICS-only (15.3%)")
            ax.set_xlabel("Number of real training tracks")
            ax.set_ylabel("FMA FPR at SONICS 1%-threshold (%)")
            ax.set_title("Scaling Law: FMA FPR vs Real Training Set Size")
            ax.legend(fontsize=9)
            ax.set_ylim(0, 50)

        plt.tight_layout()
        fig.savefig(output_dir / "coverage_curve_overall.png", dpi=150)
        plt.close(fig)
        logger.info("Saved coverage_curve_overall.png")
    except Exception as exc:
        logger.warning("Could not save plots: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifold-coverage / data-efficiency scaling curve for RealNVP flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--window-flow-csv", required=True, help="window_flow_eval.csv with per-track labels + algorithms"
    )
    parser.add_argument("--emb-cache", required=True, help="EnCodec embedding cache directory")
    parser.add_argument("--sonics-manifest", required=True, help="canonical_manifest.csv for SONICS")
    parser.add_argument(
        "--genre-csv", default=None, help="all_genres.csv from tag_all_genres.py (required for per_genre mode)"
    )
    parser.add_argument(
        "--fma-scores-csv",
        default=None,
        help=(
            "DEPRECATED/IGNORED: the old cross-flow FPR computed from this file was "
            "invalid (scores from a different frozen flow thresholded against each "
            "sweep flow). Use --fma-manifest + --fma-emb-cache instead."
        ),
    )
    parser.add_argument(
        "--fma-manifest",
        default=None,
        help=(
            "CSV with a track_id column for FMA tracks. Together with "
            "--fma-emb-cache, enables a VALID FMA FPR: each sweep point's own "
            "flow scores the FMA tracks and thresholds on its own held-out reals."
        ),
    )
    parser.add_argument(
        "--fma-emb-cache", default=None, help="Embedding cache root containing encodec/ for the FMA tracks."
    )
    parser.add_argument(
        "--fma-max-duration",
        type=float,
        default=None,
        help="Cache-key max-duration for FMA embeddings (default: --max-duration).",
    )
    parser.add_argument(
        "--n-heldout-real",
        type=int,
        default=2000,
        help=(
            "Held-out real evaluation set reserved BEFORE building training "
            "prefixes (disjoint at every sweep point). The old protocol scored "
            "training reals as if held out — its numbers are not citable."
        ),
    )
    parser.add_argument(
        "--n-coupling-layers",
        type=int,
        default=12,
        help="RealNVP depth — default 12 matches the ADOPTED HEADLINE config.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help=(
            "Fit a PCA on each sweep point's training windows before the flow "
            "(0 = none). Use 64 with --n-coupling-layers 2 to replicate the "
            "FakeMusicCaps K=2 recipe."
        ),
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=4.0,
        help="Pooling window (s). The dense-window ablation uses e.g. 1.0.",
    )
    parser.add_argument(
        "--hop-duration", type=float, default=2.0, help="Pooling hop (s). The dense-window ablation uses e.g. 0.5."
    )
    parser.add_argument("--output-dir", default="data/processed/coverage_curve")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode",
        choices=["overall", "per_genre", "both"],
        default="both",
    )
    parser.add_argument(
        "--n-real-sizes",
        nargs="+",
        type=int,
        default=[200, 500, 1000, 2000, 4000, 8000, 12722],
        help="Overall sweep: number of real tracks at each point",
    )
    parser.add_argument(
        "--n-per-genre-sizes",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 200, 500],
        help="Per-genre sweep: genre sample sizes",
    )
    parser.add_argument(
        "--baseline-n",
        type=int,
        default=2000,
        help="Per-genre sweep: number of non-genre real tracks in baseline training set",
    )
    parser.add_argument(
        "--flow-epochs", type=int, default=200, help="RealNVP training epochs per sweep point (200 = headline parity)"
    )
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Window/hop are module-level constants consumed by the sweeps and the
    # cache-health probe; override them from the CLI (dense-window ablation).
    global WINDOW_DUR, HOP_DUR  # noqa: PLW0603  (deliberate: CLI overrides module-level sweep constants)
    WINDOW_DUR = args.window_duration
    HOP_DUR = args.hop_duration

    emb_cache = Path(args.emb_cache)

    # Load data
    manifest = pd.read_csv(args.sonics_manifest, low_memory=False)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"]
    manifest["track_id"] = manifest["track_id"].astype(str)

    wf_df = pd.read_csv(args.window_flow_csv, low_memory=False)
    wf_df["track_id"] = wf_df["track_id"].astype(str)

    _check_cache_health(
        manifest.loc[manifest["label"] == "real", "track_id"].astype(str).tolist(),
        emb_cache,
        args.max_duration,
    )

    if args.fma_scores_csv:
        logger.warning(
            "--fma-scores-csv is DEPRECATED and ignored: thresholding another flow's scores "
            "against this sweep's flow is invalid (cross-flow NLLs share no scale). "
            "Pass --fma-manifest + --fma-emb-cache for a valid FMA FPR."
        )
    fma_manifest = None
    fma_emb_cache = None
    if args.fma_manifest and args.fma_emb_cache:
        if Path(args.fma_manifest).exists() and Path(args.fma_emb_cache).exists():
            fma_manifest = pd.read_csv(args.fma_manifest, low_memory=False)
            fma_emb_cache = Path(args.fma_emb_cache)
        else:
            logger.warning("FMA manifest/cache path missing — FMA FPR disabled.")

    if args.mode in ("overall", "both"):
        logger.info("=== Overall sweep ===")
        overall_df = run_overall_sweep(
            manifest=manifest,
            wf_df=wf_df,
            emb_cache=emb_cache,
            output_dir=output_dir,
            device=args.device,
            seed=args.seed,
            max_duration=args.max_duration,
            n_real_sizes=args.n_real_sizes,
            flow_epochs=args.flow_epochs,
            n_coupling_layers=args.n_coupling_layers,
            n_heldout_real=args.n_heldout_real,
            pca_components=args.pca_components,
            fma_manifest=fma_manifest,
            fma_emb_cache=fma_emb_cache,
            fma_max_duration=args.fma_max_duration,
        )
        plot_coverage_curves(overall_df, output_dir)

    if args.mode in ("per_genre", "both"):
        if args.genre_csv is None or not Path(args.genre_csv).exists():
            logger.error("--genre-csv required for per_genre mode. Run tag_all_genres.py first.")
            sys.exit(1)
        genres_df = pd.read_csv(args.genre_csv, low_memory=False)
        logger.info("=== Per-genre sweep ===")
        run_per_genre_sweep(
            manifest=manifest,
            genres_df=genres_df,
            emb_cache=emb_cache,
            output_dir=output_dir,
            device=args.device,
            seed=args.seed,
            max_duration=args.max_duration,
            n_per_genre_sizes=args.n_per_genre_sizes,
            baseline_n=args.baseline_n,
            flow_epochs=args.flow_epochs,
            n_coupling_layers=args.n_coupling_layers,
        )


if __name__ == "__main__":
    main()
