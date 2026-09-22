"""Score an external audio corpus against the SONICS-trained EnCodec RealNVP flow.

Two use-cases:
  (1) FPR test — pass real music (FMA, MUSDB) to measure the false-positive rate.
  (2) Generalization — pass fakes from unseen generators (FakeMusicCaps, MusicGen)
      to measure cross-dataset zero-shot detection.

The full RealNVP flow (trained on all SONICS real windows) is loaded from a saved
.pt file.  A missing checkpoint is a HARD ERROR: re-training a substitute happens
only with an explicit --allow-retrain, because the silent version of that fallback
made two runs scoring two different detectors emit byte-identical score files
(report §8.2 / §9.7.28).

Usage
-----
# FPR test on FMA-small tracks:
poetry run python scripts/score_external_corpus.py \\
    --audio-dir data/external/fma_small_sample/ \\
    --dataset-label real \\
    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\
    --sonics-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --output-dir data/processed/external_scores/fma_fpr \\
    --device cuda

# Generalization test on FakeMusicCaps (subdirs = algorithm names):
poetry run python scripts/score_external_corpus.py \\
    --audio-dir data/external/fakemusiccaps/ \\
    --dataset-label fake \\
    --algorithm-from-dirname \\
    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\
    --sonics-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --output-dir data/processed/external_scores/fakemusiccaps \\
    --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

# Make the src/ package importable when run as a script (mirrors run_balanced_ablation.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Redirect stdin to /dev/null immediately so background execution never blocks on
# a SIGTTIN signal (triggered when subprocesses, e.g. ffmpeg inside preprocess_audio,
# attempt to read from the controlling terminal).
import os as _os

_dev_null = _os.open(_os.devnull, _os.O_RDONLY)
_os.dup2(_dev_null, 0)
_os.close(_dev_null)

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
ENCODEC_TARGET_SR = 24_000
ENCODEC_FPS = 75  # approx frames per second at 24 kHz


# ---------------------------------------------------------------------------
# Window pooling (mirrors _pool_windows in run_balanced_ablation.py)
# ---------------------------------------------------------------------------


def _pool_windows(emb_matrix: np.ndarray, window_frames: int, hop_frames: int) -> np.ndarray:
    pooled: list[np.ndarray] = []
    n = len(emb_matrix)
    start = 0
    while start + window_frames <= n:
        sub = emb_matrix[start : start + window_frames]
        finite = sub[np.isfinite(sub).all(axis=1)]
        if len(finite) >= 1:
            pooled.append(finite.mean(axis=0))
        start += hop_frames
    return np.array(pooled, dtype=np.float64) if pooled else np.empty((0, emb_matrix.shape[1]))


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------


def _canonicalize_audio(path: Path, max_duration: float = 55.0) -> tuple[np.ndarray, int] | None:
    """Load, MP3 round-trip (64kbps), LUFS-normalize, resample to 24 kHz, trim."""
    try:
        from intrinsic_ai_music_detection.data.audio_preprocessing import preprocess_audio

        audio, sr, _ = preprocess_audio(
            path,
            target_sr=ENCODEC_TARGET_SR,
            mode="canonical",
            max_duration=max_duration,
            mp3_bitrate_kbps=64,
            target_lufs=-23.0,
        )
        return audio, sr
    except Exception as exc:
        logger.debug("Canonicalize failed for %s: %s", path, exc)
        return None


def _extract_encodec_embeddings(audio: np.ndarray, sr: int, extractor) -> np.ndarray | None:
    """Run EnCodec on audio, return continuous latent matrix [n_frames, 128]."""
    try:
        emb = extractor.extract(audio, sr)
        if emb is None or len(emb) == 0:
            return None
        return np.asarray(emb, dtype=np.float32)
    except Exception as exc:
        logger.debug("EnCodec extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# SONICS flow: load from disk or re-train from embedding cache
# ---------------------------------------------------------------------------


def _load_or_train_flow(
    flow_path: Path,
    sonics_cache: Path,
    sonics_manifest: Path,
    wf_window_duration: float,
    wf_hop_duration: float,
    max_duration: float,
    flow_epochs: int,
    device: str,
    seed: int,
    allow_retrain: bool = False,
    embedding: str = "encodec",
    n_coupling_layers: int = 8,
    hidden_dim: int = 128,
) -> "RealNVPOneClass":
    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    if flow_path.exists():
        # Dispatch on the checkpoint: this path is also fed banded and
        # likelihood-ratio detectors, and loading them as a plain flow would
        # score with only part of the model.
        from intrinsic_ai_music_detection.models.loading import load_detector

        return load_detector(flow_path, device=device)

    # A missing checkpoint used to fall through to a silent re-train. Two runs
    # scoring two DIFFERENT detectors (fma_fpr_ctrl and fma_fpr_lr) both took
    # that path and produced BYTE-IDENTICAL score files, because what actually
    # got scored was the same freshly-trained default flow in both cases. The
    # numbers were plausible, which is exactly why it went unnoticed. Every
    # fallback in this repo is now opt-in: producing a plausible wrong answer is
    # strictly worse than failing.
    if not allow_retrain:
        raise SystemExit(
            f"FLOW CHECKPOINT NOT FOUND: {flow_path}\n"
            "Refusing to silently re-train a substitute flow — that is how two runs with "
            "different detectors produced identical scores (report §8.2 / §9.7.28).\n"
            "Fix one of:\n"
            f"  * point --flow-path at an existing checkpoint (look for *_{embedding}.pt; "
            "run_balanced_ablation.py appends the embedding name to --wf-save-flow-path);\n"
            "  * re-run the arm with --wf-save-flow-path so a checkpoint is written;\n"
            "  * pass --allow-retrain if you genuinely want a fresh flow trained here (then "
            "also pass --wf-n-coupling-layers / --wf-hidden-dim to match the arm you are "
            "comparing against, or the retrained flow will NOT be the same model)."
        )

    logger.warning(
        "No saved flow at %s — RE-TRAINING (--allow-retrain given). embedding=%s K=%d hidden=%d. "
        "This flow is NOT the same object as any other arm's checkpoint; do not compare its "
        "scores against another arm's without saying so.",
        flow_path,
        embedding,
        n_coupling_layers,
        hidden_dim,
    )

    # Load real track IDs from the SONICS manifest
    manifest = pd.read_csv(sonics_manifest)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"]
    real_ids = manifest.loc[manifest["label"] == "real", "track_id"].astype(str).tolist()
    logger.info("SONICS real tracks: %d", len(real_ids))

    # Collect pooled window vectors from the embedding cache. The subdirectory is
    # the embedding name: this used to be hard-coded to "encodec", so a retrain
    # requested for a spectrogram arm silently trained on codec latents and then
    # scored spectrogram features with it (both are 128-d, so every shape check
    # passed).
    cache_dir = sonics_cache / embedding
    if not cache_dir.exists():
        raise SystemExit(
            f"--allow-retrain: embedding cache {cache_dir} does not exist. "
            f"--sonics-cache must point at the cache ROOT (it contains one subdirectory "
            f"per embedding, e.g. {sonics_cache}/{embedding}/)."
        )
    window_frames = max(int(wf_window_duration * ENCODEC_FPS), 5)
    hop_frames = max(int(wf_hop_duration * ENCODEC_FPS), 1)

    real_wins: list[np.ndarray] = []
    n_ok = 0
    for track_id in real_ids:
        key = hashlib.md5(f"{track_id}_{ENCODEC_TARGET_SR}_{max_duration}_preprocessed".encode()).hexdigest()
        cp = cache_dir / f"{key}.npy"
        if not cp.exists():
            continue
        try:
            mat = np.load(str(cp))
            pw = _pool_windows(mat, window_frames, hop_frames)
            if len(pw) > 0:
                real_wins.append(pw)
                n_ok += 1
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", track_id, exc)

    if not real_wins:
        raise RuntimeError(
            f"No real-track embeddings found in {cache_dir}. "
            "Check --sonics-cache points to the encodec embedding cache directory."
        )

    x_real = np.concatenate(real_wins, axis=0)
    logger.info("Collected %d real windows from %d/%d tracks for flow training", len(x_real), n_ok, len(real_ids))

    # K and hidden width must be passed through: omitting them silently produced
    # a K=8 flow whenever the arm being compared against was K=12.
    cfg = RealNVPConfig(
        device=device,
        n_epochs=flow_epochs,
        patience=20,
        seed=seed,
        verbose=True,
        n_coupling_layers=n_coupling_layers,
        hidden_dim=hidden_dim,
    )
    flow = RealNVPOneClass(cfg).fit(x_real)
    # Tag the front end so a later run cannot score spectrogram features with a
    # codec-latent flow (the mismatch guard in main() reads this attribute).
    flow.embedding_name = embedding
    flow.save(flow_path)
    logger.info("Flow saved → %s", flow_path)
    return flow


# ---------------------------------------------------------------------------
# Score a single track
# ---------------------------------------------------------------------------


def _score_track(
    emb_matrix: np.ndarray,
    flow: "RealNVPOneClass",
    window_frames: int,
    hop_frames: int,
) -> dict | None:
    pw = _pool_windows(emb_matrix, window_frames, hop_frames)
    if len(pw) < 2:
        return None
    lls = -flow.score_samples(pw)  # score_samples returns NLL; negate → loglik
    finite = lls[np.isfinite(lls)]
    if len(finite) < 2:
        return None
    x = np.arange(len(finite), dtype=float)
    slope = float(np.polyfit(x, finite, 1)[0])
    return {
        "wf_mean": float(finite.mean()),
        "wf_std": float(finite.std()),
        "wf_slope": slope,
        "wf_range": float(finite.max() - finite.min()),
        "wf_first_last": float(finite[-1] - finite[0]),
        "wf_n_windows": int(len(finite)),
        "wf_anomaly_score": float(-finite.mean()),  # negate loglik → anomaly score for AUC
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(df: pd.DataFrame, dataset_label: str, output_dir: Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    out_csv = output_dir / "external_scores.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Scores saved → %s (%d rows)", out_csv, len(df))

    if dataset_label == "real":
        # FPR test: what fraction of real tracks are flagged?
        # Use threshold = median SONICS real anomaly score or a fixed percentile
        score_col = "wf_anomaly_score"
        scores = df[score_col].dropna().values
        logger.info("\n=== FPR report on external real corpus ===")
        logger.info("  n tracks scored    : %d", len(scores))
        logger.info("  anomaly_score mean : %.3f  (higher = more AI-like)", scores.mean())
        logger.info("  anomaly_score std  : %.3f", scores.std())
        logger.info(
            "  NOTE: compare these scores against SONICS real tracks in window_flow_eval.csv "
            "(wf_anomaly_score = -wf_mean). If FMA scores < SONICS real mean, FPR will be low."
        )
        # Save percentile table
        for pct in [50, 75, 90, 95, 99]:
            logger.info("  %d-th percentile   : %.3f", pct, float(np.percentile(scores, pct)))
        return

    # Fake corpus: per-algorithm AUC/EER
    logger.info("\n=== Detection report on external fake corpus ===")
    real_col = "wf_anomaly_score"
    fake_df = df[df["label"] == "fake"]
    algorithms = sorted(fake_df["algorithm"].dropna().unique())
    rows = []
    for alg in algorithms:
        alg_scores = fake_df.loc[fake_df["algorithm"] == alg, real_col].dropna().values
        if len(alg_scores) < 5:
            continue
        # We need SONICS real scores for comparison. Load from window_flow_eval.csv if available.
        rows.append(
            {
                "algorithm": alg,
                "n": len(alg_scores),
                "mean_anomaly": float(alg_scores.mean()),
                "std_anomaly": float(alg_scores.std()),
            }
        )
        logger.info(
            "  %-25s  n=%5d  mean_anomaly=%.3f  std=%.3f", alg, len(alg_scores), alg_scores.mean(), alg_scores.std()
        )
    pd.DataFrame(rows).to_csv(output_dir / "per_generator_summary.csv", index=False)

    # Joint AUC with SONICS real scores (loaded from window_flow_eval.csv)
    wf_path = output_dir.parent.parent / "full_sonics_all" / "window_flow_eval.csv"
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        real_scores = -wf.loc[wf["label"] == "real", "encodec_wf_mean"].dropna().values
        logger.info("\n=== AUC vs SONICS real music ===")
        for alg in algorithms:
            alg_scores = fake_df.loc[fake_df["algorithm"] == alg, real_col].dropna().values
            if len(alg_scores) < 5:
                continue
            y = np.r_[np.zeros(len(real_scores)), np.ones(len(alg_scores))]
            s = np.r_[real_scores, alg_scores]
            try:
                auc = roc_auc_score(y, s)
                fpr_c, tpr_c, _ = roc_curve(y, s)
                fnr_c = 1 - tpr_c
                eer = float(fpr_c[np.nanargmin(np.abs(fnr_c - fpr_c))])
                logger.info("  %-25s  AUC=%.4f  EER=%.4f", alg, auc, eer)
            except Exception:
                pass
    else:
        logger.info(
            "  (No window_flow_eval.csv found at %s — cannot compute AUC. "
            "Run with --output-dir inside the project so the path resolves.)",
            wf_path,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score external audio against the SONICS-trained EnCodec RealNVP flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-dir", required=True, help="Directory of audio files to score.")
    parser.add_argument(
        "--dataset-label",
        choices=["real", "fake"],
        required=True,
        help="'real' for FPR test (e.g. FMA); 'fake' for generalization test (e.g. FakeMusicCaps).",
    )
    parser.add_argument("--algorithm", default=None, help="Algorithm name tag (for fake tracks with a flat dir).")
    parser.add_argument(
        "--algorithm-from-dirname",
        action="store_true",
        help="Use immediate parent directory name as the algorithm label (for FakeMusicCaps structure).",
    )
    parser.add_argument(
        "--flow-path",
        required=True,
        help=(
            "Path to saved flow .pt file (e.g. data/processed/full_sonics_all/sonics_real_flow_encodec.pt). "
            "Must EXIST unless --allow-retrain is given."
        ),
    )
    parser.add_argument(
        "--allow-retrain",
        action="store_true",
        default=False,
        help=(
            "Opt in to training a fresh flow when --flow-path is missing. OFF by default: the "
            "silent version of this fallback made two runs with different detectors emit "
            "byte-identical scores (§8.2). When on, also pass --wf-n-coupling-layers / "
            "--wf-hidden-dim to match the arm you are comparing against."
        ),
    )
    parser.add_argument(
        "--wf-n-coupling-layers",
        type=int,
        default=8,
        help="K for a --allow-retrain flow. Must match the arm under comparison (headline SONICS is 12).",
    )
    parser.add_argument(
        "--wf-hidden-dim",
        type=int,
        default=128,
        help="Coupling-net width for a --allow-retrain flow.",
    )
    parser.add_argument(
        "--sonics-cache",
        default="data/emb_cache_encodec_full",
        help="Path to the EnCodec embedding cache from the full SONICS run.",
    )
    parser.add_argument(
        "--sonics-manifest",
        default="data/processed/canonical_sonics_full/canonical_manifest.csv",
        help="Path to the SONICS canonical manifest CSV.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output CSVs.")
    parser.add_argument("--device", default="cuda", help="PyTorch device for EnCodec and flow.")
    parser.add_argument("--max-duration", type=float, default=55.0, help="Max audio duration to analyze (s).")
    parser.add_argument("--wf-window-duration", type=float, default=4.0, help="Window size for flow scoring (s).")
    parser.add_argument("--wf-hop-duration", type=float, default=2.0, help="Hop size for flow scoring (s).")
    parser.add_argument("--flow-epochs", type=int, default=200, help="Epochs for flow re-training if needed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracks", type=int, default=None, help="Limit tracks (useful for quick tests).")
    parser.add_argument(
        "--embedding",
        default=None,
        help="Front end for scoring (must match the checkpoint; auto-detected from its tag if omitted).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = Path(args.flow_path)

    # Collect audio files
    audio_dir = Path(args.audio_dir)
    audio_files: list[tuple[Path, str]] = []  # (path, algorithm)
    for p in sorted(audio_dir.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        if args.algorithm_from_dirname:
            alg = p.parent.name
        else:
            alg = args.algorithm or ""
        audio_files.append((p, alg))

    if args.max_tracks:
        audio_files = audio_files[: args.max_tracks]

    logger.info("Found %d audio files in %s", len(audio_files), audio_dir)
    if not audio_files:
        logger.error("No audio files found. Check --audio-dir.")
        sys.exit(1)

    # Load/train the SONICS flow
    flow = _load_or_train_flow(
        flow_path=flow_path,
        sonics_cache=Path(args.sonics_cache),
        sonics_manifest=Path(args.sonics_manifest),
        wf_window_duration=args.wf_window_duration,
        wf_hop_duration=args.wf_hop_duration,
        max_duration=args.max_duration,
        flow_epochs=args.flow_epochs,
        device=args.device,
        seed=args.seed,
        allow_retrain=args.allow_retrain,
        embedding=args.embedding or "encodec",
        n_coupling_layers=args.wf_n_coupling_layers,
        hidden_dim=args.wf_hidden_dim,
    )

    # Load EnCodec extractor (lazy: only loaded on first cache miss)
    extractor = None

    window_frames = max(int(args.wf_window_duration * ENCODEC_FPS), 5)
    hop_frames = max(int(args.wf_hop_duration * ENCODEC_FPS), 1)
    logger.info(
        "Window: %.0fs / hop %.0fs → %d / %d frames",
        args.wf_window_duration,
        args.wf_hop_duration,
        window_frames,
        hop_frames,
    )

    # Score each track
    rows: list[dict] = []
    n_ok = n_skip = 0
    t0 = time.time()
    for i, (path, alg) in enumerate(audio_files, 1):
        logger.debug("[%d] canonicalizing %s", i, path.name)
        result = _canonicalize_audio(path, max_duration=args.max_duration)
        if result is None:
            logger.debug("[%d] canonicalize failed, skipping", i)
            n_skip += 1
            continue
        audio, sr = result
        logger.debug("[%d] canonicalized, extracting embeddings", i)

        if extractor is None:
            from intrinsic_ai_music_detection.features.embeddings import get_extractor

            # Honour the front end the flow was trained on. EnCodec latents and
            # 128-bin log-mel are both 128-d, so a mismatched extractor passes
            # every shape check and silently yields meaningless scores.
            _trained_on = getattr(flow, "embedding_name", None) if flow is not None else None
            _req = getattr(args, "embedding", None)
            if _trained_on and _req and _trained_on != _req:
                raise SystemExit(
                    f"EMBEDDING MISMATCH: checkpoint trained on {_trained_on!r} but --embedding is "
                    f"{_req!r}. Re-run with --embedding {_trained_on}."
                )
            _name = _req or _trained_on or "encodec"
            if not _trained_on:
                logger.warning(
                    "Checkpoint has no embedding tag — assuming %r. Pass --embedding explicitly "
                    "if the flow was trained on a different front end.",
                    _name,
                )
            logger.info("Loading %s extractor (device=%s) ...", _name, args.device)
            extractor = get_extractor(_name, device=args.device)
            logger.info("%s extractor ready.", _name)

        emb = _extract_encodec_embeddings(audio, sr, extractor)
        if emb is None:
            n_skip += 1
            continue

        scores = _score_track(emb, flow, window_frames, hop_frames)
        if scores is None:
            n_skip += 1
            continue

        rows.append(
            {
                "track_id": path.stem,
                "path": str(path),
                "label": args.dataset_label,
                "algorithm": alg,
                **scores,
            }
        )
        n_ok += 1

        if i <= 5 or i % 10 == 0 or i == len(audio_files):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            logger.info("  %d/%d  ok=%d skip=%d  %.1f tracks/s", i, len(audio_files), n_ok, n_skip, rate)

    if not rows:
        logger.error("No tracks scored successfully.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    _report(df, args.dataset_label, output_dir)


if __name__ == "__main__":
    main()
