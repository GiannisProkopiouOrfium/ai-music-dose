"""Zero-shot genre tagging for real SONICS tracks via CLAP.

Real tracks in SONICS lack genre metadata (only fake tracks have `genre` /
`style` / `mood` from the SONICS fake_songs.csv).  This script infers a
genre for each real track by scoring it against text prompts derived from
the SONICS genre vocabulary — no new model is needed, only the CLAP extractor
already present in the codebase.

Approach
--------
1. Build a genre vocabulary: the unique `genre` values from fake_songs.csv
   (e.g. "punk", "r&b", "jazz", "pop rock", …).
2. Load CLAP text embeddings for each genre prompt
   (e.g. "a song in the genre of punk rock").
3. For every real track, extract the CLAP audio embedding.
4. Compute cosine similarity between the audio embedding and all genre prompts.
5. Assign the top-k genres + their confidence scores.
6. Write real_genres.csv.

Runtime: ~0.5 s per track on GPU.  For 20 k real tracks → ~3 h.
Use --max-tracks for a quick validation run.

Usage
-----
# Full run:
poetry run python scripts/tag_real_genres.py \
  --output-dir data/processed/genre_tags \
  --device cuda

# Quick validation (100 real tracks):
poetry run python scripts/tag_real_genres.py \
  --max-tracks 100 --output-dir data/processed/genre_tags_pilot \
  --device cuda

# From sampler canonical manifest (after build_canonical_for_sampler.py):
poetry run python scripts/tag_real_genres.py \
  --canonical-manifest data/processed/canonical_sampler/canonical_manifest.csv \
  --output-dir data/processed/genre_tags \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")

# Text prompt template for genre classification
GENRE_PROMPT = "a song in the genre of {genre}"

# Minimum CLAP similarity to assign a genre (avoids low-confidence assignments)
MIN_CONFIDENCE = 0.10


def build_genre_vocabulary(fake_csv: Path) -> list[str]:
    """Extract unique genre values from fake_songs.csv."""
    df = pd.read_csv(fake_csv, low_memory=False)
    genres = sorted(str(g).strip().lower() for g in df["genre"].dropna().unique() if str(g).strip())
    logger.info("Genre vocabulary: %d unique genres", len(genres))
    return genres


def embed_genre_prompts(genres: list[str], clap_extractor) -> np.ndarray:
    """Embed all genre text prompts via CLAP. Returns [n_genres, D] array."""
    prompts = [GENRE_PROMPT.format(genre=g) for g in genres]
    logger.info("Embedding %d genre prompts via CLAP text encoder...", len(prompts))
    sys.stdout.flush()
    with __import__("torch").no_grad():
        text_embs = clap_extractor.model.get_text_embedding(prompts)
    logger.info("Genre prompt embeddings done. Shape: %s", np.asarray(text_embs).shape)
    sys.stdout.flush()
    return np.asarray(text_embs, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector a [D] and matrix b [N, D]. Returns [N]."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (b_norms @ a_norm).astype(float)


def _load_real_tracks(manifest_csv: str | None) -> list[dict]:
    """Load real track list from canonical manifest or SONICS metadata."""
    if manifest_csv:
        manifest_path = Path(manifest_csv)
        if manifest_path.exists():
            df = pd.read_csv(manifest_path, low_memory=False)
            real_df = df[(df["label"] == "real") & (df.get("status", "ok") == "ok")]
            tracks = []
            for _, row in real_df.iterrows():
                path = row.get("canonical_path", row.get("src_path", ""))
                if path and Path(path).exists():
                    tracks.append(
                        {
                            "track_id": str(row.get("track_id", "")),
                            "path": path,
                            "artist": str(row.get("artist", "")),
                            "year": str(row.get("year", "")),
                        }
                    )
            logger.info("Loaded %d real tracks from canonical manifest", len(tracks))
            return tracks

    # Fall back to raw metadata
    metadata_dir = SONICS_DIR / "metadata"
    real_csv = metadata_dir / "real_songs_ready.csv"
    if not real_csv.exists():
        real_csv = metadata_dir / "real_songs.csv"

    if not real_csv.exists():
        logger.error("real_songs.csv not found. Run: python scripts/download_sonics.py --prepare")
        sys.exit(1)

    df_real = pd.read_csv(real_csv, low_memory=False)
    real_dir = SONICS_DIR / "real_songs"
    tracks = []
    for _, row in df_real.iterrows():
        yt_id = str(row.get("youtube_id", "")).strip()
        audio_path = row.get("audio_path", "")
        if audio_path and Path(audio_path).exists():
            path = audio_path
        else:
            candidates = list(real_dir.glob(f"{yt_id}.*")) if real_dir.exists() else []
            if not candidates:
                continue
            path = str(candidates[0])
        tracks.append(
            {
                "track_id": yt_id,
                "path": str(path),
                "artist": str(row.get("artist", "")),
                "year": str(row.get("year", "")),
            }
        )
    logger.info("Loaded %d real tracks from SONICS metadata", len(tracks))
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot genre tagging for real SONICS tracks via CLAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/processed/genre_tags")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top genres to record per track")
    parser.add_argument("--max-tracks", type=int, default=None, help="Limit to N real tracks (for testing)")
    parser.add_argument(
        "--canonical-manifest",
        default=None,
        help="Path to canonical_manifest.csv (uses canonical audio for embeddings)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]  # prevent CLAP from hijacking our args

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Genre vocabulary from fake metadata
    fake_csv = SONICS_DIR / "metadata" / "fake_songs.csv"
    if not fake_csv.exists():
        fake_csv = SONICS_DIR / "metadata" / "fake_songs_ready.csv"
    if not fake_csv.exists():
        logger.error("fake_songs.csv not found. Run: python scripts/download_sonics.py --download-fakes")
        sys.exit(1)

    genres = build_genre_vocabulary(fake_csv)

    # Save vocabulary for reference
    with open(output_dir / "genre_vocabulary.json", "w") as f:
        json.dump(genres, f, indent=2)
    logger.info("Saved genre_vocabulary.json (%d genres)", len(genres))

    # Load CLAP extractor (lazy-loaded on first use)
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    logger.info("Loading CLAP model on %s...", args.device)
    extractor = get_extractor("clap", device=args.device)

    # Embed genre text prompts
    genre_embs = embed_genre_prompts(genres, extractor)  # [n_genres, D]

    # Load real tracks
    real_tracks = _load_real_tracks(args.canonical_manifest)
    if args.max_tracks:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(real_tracks))[: args.max_tracks]
        real_tracks = [real_tracks[i] for i in idx]

    logger.info("Loaded %d real tracks for tagging.", len(real_tracks))
    sys.stdout.flush()
    logger.info("Tagging %d real tracks...", len(real_tracks))

    # Check for existing results to resume
    out_csv = output_dir / "real_genres.csv"
    done_ids: set[str] = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv, low_memory=False)
        done_ids = set(existing["track_id"].astype(str))
        logger.info("Resuming: %d tracks already tagged", len(done_ids))

    results: list[dict] = []
    t0 = time.time()
    n_todo = sum(1 for t in real_tracks if t["track_id"] not in done_ids)

    for i, track in enumerate(real_tracks, start=1):
        if track["track_id"] in done_ids:
            continue

        try:
            from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio

            audio, sr = load_audio(track["path"], target_sr=extractor.sample_rate, max_duration=30.0)
            audio = normalize_audio(audio)

            audio_emb = extractor.extract(audio, sr)  # [N_chunks, D]
            # Average over chunks to get a single track embedding
            track_emb = audio_emb.mean(axis=0)  # [D]

            sims = _cosine_sim(track_emb, genre_embs)  # [n_genres]
            top_k_idx = np.argsort(sims)[::-1][: args.top_k]

            row = {
                "track_id": track["track_id"],
                "artist": track.get("artist", ""),
                "year": track.get("year", ""),
                "genre_top1": genres[top_k_idx[0]],
                "confidence_top1": float(sims[top_k_idx[0]]),
                "genres_top_k": json.dumps([genres[j] for j in top_k_idx]),
                "confidences_top_k": json.dumps([float(sims[j]) for j in top_k_idx]),
            }
            results.append(row)

        except Exception as exc:
            logger.warning("Failed %s: %s", track["track_id"], exc)

        done_so_far = len(results)
        if done_so_far % 50 == 0 or done_so_far == n_todo:
            elapsed = time.time() - t0
            eta = elapsed / max(done_so_far, 1) * max(n_todo - done_so_far, 0)
            logger.info("[%d/%d] elapsed=%.0fm eta=%.0fm", done_so_far, n_todo, elapsed / 60, eta / 60)
            sys.stdout.flush()

        # Append to CSV every 500 tracks for crash resilience
        if len(results) % 500 == 0 and results:
            chunk_df = pd.DataFrame(results)
            if out_csv.exists():
                chunk_df.to_csv(out_csv, mode="a", header=False, index=False)
            else:
                chunk_df.to_csv(out_csv, index=False)
            done_ids.update(r["track_id"] for r in results)
            results = []

    # Final flush
    if results:
        chunk_df = pd.DataFrame(results)
        if out_csv.exists():
            chunk_df.to_csv(out_csv, mode="a", header=False, index=False)
        else:
            chunk_df.to_csv(out_csv, index=False)

    # Summary
    if out_csv.exists():
        final_df = pd.read_csv(out_csv, low_memory=False)
        logger.info("Done. real_genres.csv: %d tracks", len(final_df))
        logger.info("Top-5 assigned genres:")
        for genre, cnt in final_df["genre_top1"].value_counts().head(5).items():
            logger.info("  %-25s  %d", genre, cnt)


if __name__ == "__main__":
    main()
