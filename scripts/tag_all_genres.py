"""Zero-shot genre tagging for ALL tracks (real AND fake) via a single shared CLAP tagger.

Unlike tag_real_genres.py which only tags real tracks, this script tags every
track in the canonical manifest using the same CLAP model and the same genre
vocabulary derived from SONICS fake_songs.csv.  Using one consistent tagger
for both classes is required for confound analysis (A2): if we tagged real and
fake via different methods, genre-stratified AUC would conflate tagger differences
with real vs fake differences.

Outputs (all in --output-dir)
------------------------------
  all_genres.csv          per-track: track_id, label, algorithm, genre_top1,
                          confidence_top1, genres_top_k, confidences_top_k,
                          clap_genre_vector (JSON list, full posterior over vocabulary)
  genre_vocabulary.json   ordered list of genres used as the CLAP text labels

Usage (EC2)
-----------
# Full run (both classes, ~3-4 h on g4dn.xlarge for 59k tracks):
poetry run python scripts/tag_all_genres.py \\
    --canonical-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --output-dir data/processed/genre_tags_all \\
    --device cuda

# Quick validation (500 tracks):
poetry run python scripts/tag_all_genres.py \\
    --canonical-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --output-dir data/processed/genre_tags_all_pilot \\
    --max-tracks 500 --device cuda

# Include FMA tracks for FPR genre attribution:
poetry run python scripts/tag_all_genres.py \\
    --canonical-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --extra-real-dirs data/external/fma_small \\
    --extra-label fma_real \\
    --output-dir data/processed/genre_tags_all \\
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

GENRE_PROMPT = "a song in the genre of {genre}"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
MIN_CONFIDENCE = 0.05


# ---------------------------------------------------------------------------
# Genre vocabulary
# ---------------------------------------------------------------------------


def build_genre_vocabulary(fake_csv: Path) -> list[str]:
    """Extract unique genre values from SONICS fake_songs.csv."""
    df = pd.read_csv(fake_csv, low_memory=False)
    genres = sorted(str(g).strip().lower() for g in df["genre"].dropna().unique() if str(g).strip())
    logger.info("Genre vocabulary: %d unique genres from %s", len(genres), fake_csv)
    return genres


def embed_genre_prompts(genres: list[str], clap_extractor) -> np.ndarray:
    """Embed all genre text prompts via CLAP text encoder. Returns [n_genres, D]."""
    prompts = [GENRE_PROMPT.format(genre=g) for g in genres]
    logger.info("Embedding %d genre prompts ...", len(prompts))
    import torch

    with torch.no_grad():
        text_embs = clap_extractor.model.get_text_embedding(prompts)
    embs = np.asarray(text_embs, dtype=np.float32)
    logger.info("Genre embeddings: %s", embs.shape)
    return embs


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity: a [D] vs b [N, D] -> [N]."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (b_norms @ a_norm).astype(float)


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------


def load_track_list(manifest_csv: str, extra_real_dirs: list[str], extra_label: str) -> list[dict]:
    """Load tracks from canonical manifest + optional extra directories."""
    tracks: list[dict] = []

    # --- canonical manifest ---
    manifest = pd.read_csv(manifest_csv, low_memory=False)
    ok = (
        manifest[manifest.get("status", pd.Series(["ok"] * len(manifest))).astype(str) == "ok"]
        if "status" in manifest.columns
        else manifest
    )

    for _, row in ok.iterrows():
        path_col = row.get("canonical_path", row.get("src_path", ""))
        if not path_col or not Path(str(path_col)).exists():
            continue
        tracks.append(
            {
                "track_id": str(row.get("track_id", "")),
                "path": str(path_col),
                "label": str(row.get("label", "unknown")),
                "algorithm": str(row.get("algorithm", "")),
            }
        )

    logger.info("Loaded %d tracks from canonical manifest", len(tracks))

    # --- extra real dirs (e.g. FMA) ---
    for extra_dir in extra_real_dirs:
        d = Path(extra_dir)
        if not d.exists():
            logger.warning("Extra dir not found: %s", d)
            continue
        extra_count = 0
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS:
                tracks.append(
                    {
                        "track_id": p.stem,
                        "path": str(p),
                        "label": extra_label,
                        "algorithm": "",
                    }
                )
                extra_count += 1
        logger.info("Loaded %d extra tracks from %s (label=%s)", extra_count, d, extra_label)

    logger.info("Total tracks to tag: %d", len(tracks))
    return tracks


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def tag_track(
    track: dict,
    clap_extractor,
    genre_embs: np.ndarray,
    genres: list[str],
    top_k: int,
) -> dict | None:
    """Tag one track. Returns a result dict or None on failure."""
    try:
        from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio

        audio, sr = load_audio(track["path"], target_sr=clap_extractor.sample_rate, max_duration=30.0)
        audio = normalize_audio(audio)
        audio_emb = clap_extractor.extract(audio, sr)  # [N_chunks, D]
        track_emb = audio_emb.mean(axis=0)  # [D]

        sims = _cosine_sim(track_emb, genre_embs)  # [n_genres]
        top_k_idx = np.argsort(sims)[::-1][:top_k]

        return {
            "track_id": track["track_id"],
            "label": track["label"],
            "algorithm": track.get("algorithm", ""),
            "genre_top1": genres[top_k_idx[0]],
            "confidence_top1": float(sims[top_k_idx[0]]),
            "genres_top_k": json.dumps([genres[j] for j in top_k_idx]),
            "confidences_top_k": json.dumps([float(sims[j]) for j in top_k_idx]),
            # Full posterior vector (all genres) for residualization in A2
            "clap_genre_vector": json.dumps([float(s) for s in sims]),
        }
    except Exception as exc:
        logger.debug("Failed %s: %s", track["track_id"], exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genre-tag ALL tracks (real + fake) via one shared CLAP tagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--canonical-manifest",
        required=True,
        help="Path to canonical_manifest.csv (real + fake rows)",
    )
    parser.add_argument("--output-dir", default="data/processed/genre_tags_all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K genres to store per track")
    parser.add_argument("--max-tracks", type=int, default=None, help="Limit tracks for testing")
    parser.add_argument(
        "--extra-real-dirs",
        nargs="*",
        default=[],
        help="Additional directories of real audio (e.g. FMA-small) to tag",
    )
    parser.add_argument(
        "--extra-label",
        default="fma_real",
        help="Label string for tracks from --extra-real-dirs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fake-csv",
        default=None,
        help="Path to fake_songs.csv for genre vocabulary (auto-detected if not set)",
    )
    args = parser.parse_args()

    # Suppress CLAP arg-parser from hijacking our args
    sys.argv = [sys.argv[0]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Genre vocabulary ---
    if args.fake_csv:
        fake_csv = Path(args.fake_csv)
    else:
        # Try common locations
        candidates = [
            Path("data/raw/sonics/metadata/fake_songs.csv"),
            Path("data/raw/sonics/metadata/fake_songs_ready.csv"),
        ]
        fake_csv = next((c for c in candidates if c.exists()), None)
        if fake_csv is None:
            logger.error("fake_songs.csv not found. Provide --fake-csv or run download_sonics.py first.")
            sys.exit(1)

    genres = build_genre_vocabulary(fake_csv)
    with open(output_dir / "genre_vocabulary.json", "w") as f:
        json.dump(genres, f, indent=2)
    logger.info("Saved genre_vocabulary.json (%d genres)", len(genres))

    # --- CLAP model ---
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    logger.info("Loading CLAP on %s ...", args.device)
    extractor = get_extractor("clap", device=args.device)
    logger.info("CLAP ready.")

    genre_embs = embed_genre_prompts(genres, extractor)  # [n_genres, D]

    # --- Tracks ---
    tracks = load_track_list(args.canonical_manifest, args.extra_real_dirs, args.extra_label)
    if args.max_tracks:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(tracks))[: args.max_tracks]
        tracks = [tracks[i] for i in sorted(idx)]
        logger.info("Subsampled to %d tracks (--max-tracks)", len(tracks))

    # --- Resume ---
    out_csv = output_dir / "all_genres.csv"
    done_ids: set[str] = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv, low_memory=False)
        done_ids = set(existing["track_id"].astype(str))
        logger.info("Resuming: %d tracks already tagged", len(done_ids))

    results: list[dict] = []
    t0 = time.time()
    n_todo = sum(1 for t in tracks if t["track_id"] not in done_ids)
    done_count = 0

    for i, track in enumerate(tracks, start=1):
        if track["track_id"] in done_ids:
            continue

        row = tag_track(track, extractor, genre_embs, genres, args.top_k)
        if row is not None:
            results.append(row)
        done_count += 1

        if done_count % 100 == 0 or done_count == n_todo:
            elapsed = time.time() - t0
            eta = elapsed / max(done_count, 1) * max(n_todo - done_count, 0)
            logger.info(
                "[%d/%d] ok=%d  %.1f track/s  ETA %.0f min",
                done_count,
                n_todo,
                len(results),
                done_count / max(elapsed, 1e-6),
                eta / 60,
            )

        # Flush every 500 tracks
        if len(results) % 500 == 0 and results:
            chunk = pd.DataFrame(results)
            mode, header = ("a", False) if out_csv.exists() else ("w", True)
            chunk.to_csv(out_csv, mode=mode, header=header, index=False)
            done_ids.update(r["track_id"] for r in results)
            results = []

    # Final flush
    if results:
        chunk = pd.DataFrame(results)
        mode, header = ("a", False) if out_csv.exists() else ("w", True)
        chunk.to_csv(out_csv, mode=mode, header=header, index=False)

    if out_csv.exists():
        final = pd.read_csv(out_csv, low_memory=False)
        logger.info("Done. all_genres.csv: %d tracks total", len(final))
        logger.info("Label distribution:\n%s", final["label"].value_counts().to_string())
        logger.info("Top genres (all tracks):\n%s", final["genre_top1"].value_counts().head(10).to_string())
        logger.info(
            "Top genres (real only):\n%s",
            final[final["label"] == "real"]["genre_top1"].value_counts().head(10).to_string(),
        )
        logger.info(
            "Top genres (fake only):\n%s",
            final[final["label"] == "fake"]["genre_top1"].value_counts().head(10).to_string(),
        )


if __name__ == "__main__":
    main()
