"""Small end-to-end sanity check.

Downloads a few tracks from S3, extracts EnCodec embeddings (CPU-friendly),
computes ID via PHD/TwoNN/MLE, and prints a comparison of real vs AI.

Usage (with aws-vault):
    aws-vault exec admin@innovation -- python scripts/sanity_check.py --n-pairs 5

Without aws-vault (if AWS env vars already set):
    python scripts/sanity_check.py --n-pairs 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so we can import without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.config import ExperimentConfig, S3Config
from intrinsic_ai_music_detection.data.audio_utils import get_duration, load_audio, normalize_audio
from intrinsic_ai_music_detection.data.make_dataset import download_from_s3, load_plagiarism_registry, read_s3_csv
from intrinsic_ai_music_detection.features.id_estimators import estimate_all
from intrinsic_ai_music_detection.features.phd import PHD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/raw/sanity_check")
RESULTS_DIR = Path("data/processed/sanity_check")


def download_sample_pairs(
    registry: pd.DataFrame,
    s3_cfg: S3Config,
    n_pairs: int = 5,
) -> list[dict]:
    """Download n_pairs matched real/AI tracks from S3.

    Returns list of dicts with keys: track_id, real_path, ai_path, source_model
    """
    pairs = []
    prefix = s3_cfg.prefix

    # List available AI outputs to find what actually exists
    # We'll try musicgen instrumentals first (most likely to exist)
    count = 0
    for _, row in registry.iterrows():
        if count >= n_pairs:
            break

        vid_id = str(row["original_video_id"])

        # Try to download original instrumental + musicgen AI version
        real_s3_key = f"{prefix}/inputs/no_vocals/{vid_id}.wav"
        ai_s3_key = f"{prefix}/outputs/musicgen/{vid_id}.wav"

        real_local = CACHE_DIR / "real" / f"{vid_id}.wav"
        ai_local = CACHE_DIR / "ai_musicgen" / f"{vid_id}.wav"

        try:
            download_from_s3(s3_cfg.bucket, real_s3_key, real_local, s3_cfg.region)
            download_from_s3(s3_cfg.bucket, ai_s3_key, ai_local, s3_cfg.region)
            pairs.append(
                {
                    "track_id": vid_id,
                    "real_path": real_local,
                    "ai_path": ai_local,
                    "source_model": "musicgen",
                }
            )
            count += 1
            logger.info("Downloaded pair %d/%d: %s", count, n_pairs, vid_id)
        except Exception as e:
            logger.debug("Skipping %s: %s", vid_id, e)
            continue

    if not pairs:
        # Fallback: try mixed_acestep or mixed_vevo2
        for _, row in registry.iterrows():
            if count >= n_pairs:
                break

            vid_id = str(row["original_video_id"])
            ai_s3_path = str(row.get("ai_generated_s3_path", ""))
            if not ai_s3_path:
                continue

            # Parse the s3 path to get bucket-relative key
            if ai_s3_path.startswith("s3://"):
                # Strip s3://bucket_name/
                ai_s3_key = "/".join(ai_s3_path.split("/")[3:])
            else:
                ai_s3_key = ai_s3_path

            real_s3_key = f"{prefix}/inputs/no_vocals/{vid_id}.wav"
            real_local = CACHE_DIR / "real" / f"{vid_id}.wav"
            ai_local = CACHE_DIR / "ai" / f"{vid_id}.wav"

            try:
                download_from_s3(s3_cfg.bucket, real_s3_key, real_local, s3_cfg.region)
                download_from_s3(s3_cfg.bucket, ai_s3_key, ai_local, s3_cfg.region)
                pairs.append(
                    {
                        "track_id": vid_id,
                        "real_path": real_local,
                        "ai_path": ai_local,
                        "source_model": "from_registry",
                    }
                )
                count += 1
                logger.info("Downloaded pair %d/%d: %s", count, n_pairs, vid_id)
            except Exception as e:
                logger.debug("Skipping %s: %s", vid_id, e)
                continue

    return pairs


def extract_encodec_embeddings(audio_path: Path) -> np.ndarray | None:
    """Extract EnCodec pre-quantization embeddings from an audio file."""
    from intrinsic_ai_music_detection.features.embeddings import EnCodecExtractor

    try:
        extractor = EnCodecExtractor(device="cpu")
        embeddings = extractor.extract_from_file(audio_path)
        return embeddings
    except Exception as e:
        logger.warning("Failed to extract embeddings from %s: %s", audio_path, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check: small end-to-end test")
    parser.add_argument("--n-pairs", type=int, default=5, help="Number of track pairs to test")
    parser.add_argument("--skip-download", action="store_true", help="Skip S3 download, use cached files")
    parser.add_argument("--region", type=str, default="eu-west-1", help="AWS region")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    s3_cfg = S3Config(region=args.region)

    # ---------------------------------------------------------------
    # Step 1: Download registry CSV
    # ---------------------------------------------------------------
    registry_local = CACHE_DIR / "plagiarism_dataset.csv"
    if registry_local.exists():
        logger.info("Using cached registry: %s", registry_local)
        registry = pd.read_csv(registry_local)
    else:
        logger.info("Downloading registry from S3...")
        registry = load_plagiarism_registry(s3_cfg=s3_cfg)
        registry.to_csv(registry_local, index=False)
        logger.info("Registry saved: %d rows", len(registry))

    print(f"\n{'='*60}")
    print(f"Registry: {len(registry)} entries")
    print(f"Columns: {list(registry.columns)}")
    print(f"Sample:\n{registry.head(3).to_string()}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # Step 2: Download sample pairs
    # ---------------------------------------------------------------
    if args.skip_download:
        # Use whatever is cached
        real_files = sorted((CACHE_DIR / "real").glob("*.wav")) if (CACHE_DIR / "real").exists() else []
        pairs = []
        for rf in real_files[: args.n_pairs]:
            vid_id = rf.stem
            ai_dirs = ["ai_musicgen", "ai"]
            for ai_dir in ai_dirs:
                ai_path = CACHE_DIR / ai_dir / f"{vid_id}.wav"
                if ai_path.exists():
                    pairs.append(
                        {
                            "track_id": vid_id,
                            "real_path": rf,
                            "ai_path": ai_path,
                            "source_model": ai_dir.replace("ai_", ""),
                        }
                    )
                    break
    else:
        pairs = download_sample_pairs(registry, s3_cfg, n_pairs=args.n_pairs)

    if not pairs:
        logger.error("No audio pairs available. Check S3 access and data.")
        sys.exit(1)

    print(f"Downloaded {len(pairs)} matched pairs\n")

    # ---------------------------------------------------------------
    # Step 3: Extract EnCodec embeddings + compute IDs
    # ---------------------------------------------------------------
    real_ids: list[dict] = []
    ai_ids: list[dict] = []

    for pair in pairs:
        track_id = pair["track_id"]
        real_path = Path(pair["real_path"])
        ai_path = Path(pair["ai_path"])

        print(f"Processing {track_id}...")

        # Extract embeddings
        real_emb = extract_encodec_embeddings(real_path)
        ai_emb = extract_encodec_embeddings(ai_path)

        if real_emb is None or ai_emb is None:
            print(f"  SKIP: embedding extraction failed")
            continue

        print(f"  Real embeddings: {real_emb.shape}")
        print(f"  AI embeddings:   {ai_emb.shape}")

        # Compute ID for both
        real_id_scores = estimate_all(real_emb, metric="euclidean")
        ai_id_scores = estimate_all(ai_emb, metric="euclidean")

        real_ids.append({"track_id": track_id, "type": "real", **real_id_scores})
        ai_ids.append({"track_id": track_id, "type": "ai", **ai_id_scores})

        print(
            f"  Real ID: PHD={real_id_scores.get('phd', float('nan')):.2f}, "
            f"TwoNN={real_id_scores.get('twonn', float('nan')):.2f}, "
            f"MLE={real_id_scores.get('mle', float('nan')):.2f}"
        )
        print(
            f"  AI   ID: PHD={ai_id_scores.get('phd', float('nan')):.2f}, "
            f"TwoNN={ai_id_scores.get('twonn', float('nan')):.2f}, "
            f"MLE={ai_id_scores.get('mle', float('nan')):.2f}"
        )
        print()

    # ---------------------------------------------------------------
    # Step 4: Compare distributions
    # ---------------------------------------------------------------
    if not real_ids or not ai_ids:
        logger.error("No valid results to compare.")
        sys.exit(1)

    df_results = pd.DataFrame(real_ids + ai_ids)
    results_path = RESULTS_DIR / "sanity_check_results.csv"
    df_results.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tracks processed: {len(real_ids)} real, {len(ai_ids)} AI\n")

    for method in ["phd", "twonn", "mle"]:
        real_vals = [r[method] for r in real_ids if not np.isnan(r.get(method, float("nan")))]
        ai_vals = [r[method] for r in ai_ids if not np.isnan(r.get(method, float("nan")))]

        if real_vals and ai_vals:
            real_mean = np.mean(real_vals)
            ai_mean = np.mean(ai_vals)
            diff = real_mean - ai_mean

            print(
                f"{method.upper():>6s}: Real mean={real_mean:.3f} (n={len(real_vals)}), "
                f"AI mean={ai_mean:.3f} (n={len(ai_vals)}), "
                f"diff={diff:+.3f} {'<-- REAL HIGHER (expected)' if diff > 0 else '<-- AI HIGHER (unexpected)'}"
            )
        else:
            print(f"{method.upper():>6s}: insufficient data")

    print(f"\nFull results saved to: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
