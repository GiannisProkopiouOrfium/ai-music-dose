"""Build the canonical matched audio corpus.

Reads the SONICS ready-CSVs and applies unified preprocessing to every track:
  - Both real and fake pass through the same MP3 round-trip (64 kbps default)
    and LUFS normalisation (-23 dBFS), removing format and loudness confounds.
  - Outputs resampled WAV files to data/processed/canonical/{real,fake}/
  - Writes a canonical_manifest.csv with path, label, algorithm, fake_label,
    genre (if available), and measured pre-processing covariates (LUFS, peak,
    duration).

This script is designed to be run ONCE on EC2 before any analysis run.
It supports resuming: already-processed tracks are detected via the manifest
and skipped.

Usage
-----
# Full corpus:
poetry run python scripts/build_canonical_corpus.py \
  --output-dir data/processed/canonical \
  --target-sr 24000 \
  --mp3-bitrate 64 \
  --target-lufs -23.0 \
  --max-duration 120 \
  --workers 4

# Dry-run (process only first 50 per class):
poetry run python scripts/build_canonical_corpus.py \
  --output-dir data/processed/canonical \
  --max-per-class 50 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_preprocessing import (
    CANONICAL_MAX_DURATION,
    CANONICAL_MP3_KBPS,
    CANONICAL_TARGET_LUFS,
    CANONICAL_TARGET_SR,
    preprocess_audio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------


def _process_one(args: dict) -> dict | None:
    """Process a single track. Returns a manifest row dict or None on failure."""
    src_path = Path(args["src_path"])
    dst_path = Path(args["dst_path"])
    target_sr = args["target_sr"]
    mp3_bitrate = args["mp3_bitrate"]
    target_lufs = args["target_lufs"]
    max_duration = args["max_duration"]
    delete_src_after: bool = bool(args.get("delete_src_after", False))

    if dst_path.exists():
        return {
            **args,
            "canonical_path": str(dst_path),
            "status": "ok",
        }

    try:
        audio, sr, meta = preprocess_audio(
            src_path,
            target_sr=target_sr,
            mode="canonical",
            max_duration=max_duration,
            mp3_bitrate_kbps=mp3_bitrate,
            target_lufs=target_lufs,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(dst_path), sr, audio_int16)

        # Delete the source file to reclaim disk space, but only if the WAV was
        # written successfully and src != dst (safety guard).
        if delete_src_after and src_path.exists():
            if str(src_path.resolve()) != str(dst_path.resolve()):
                try:
                    src_path.unlink()
                except OSError as _del_exc:
                    import logging as _log

                    _log.getLogger(__name__).warning("Could not delete source %s: %s", src_path, _del_exc)

        return {
            **args,
            "canonical_path": str(dst_path),
            "measured_lufs": meta.get("measured_lufs", float("nan")),
            "measured_peak": meta.get("measured_peak", float("nan")),
            "actual_duration": meta.get("actual_duration", float("nan")),
            "status": "ok",
        }
    except Exception as exc:
        return {**args, "status": f"error: {exc}"}


# ---------------------------------------------------------------------------
# Track list builders
# ---------------------------------------------------------------------------


def _load_track_list(
    output_dir: Path,
    target_sr: int,
    mp3_bitrate: int,
    target_lufs: float,
    max_duration: float,
    max_per_class: int | None,
) -> list[dict]:
    metadata_dir = SONICS_DIR / "metadata"
    real_csv = metadata_dir / "real_songs_ready.csv"
    fake_csv = metadata_dir / "fake_songs_ready.csv"

    # Fall back to the unfiltered CSVs + do our own has-audio check
    if not real_csv.exists():
        real_csv = metadata_dir / "real_songs.csv"
    if not fake_csv.exists():
        fake_csv = metadata_dir / "fake_songs.csv"

    if not real_csv.exists() or not fake_csv.exists():
        logger.error("Metadata CSVs not found. Run: python scripts/download_sonics.py --prepare")
        sys.exit(1)

    df_real = pd.read_csv(real_csv, low_memory=False)
    df_fake = pd.read_csv(fake_csv, low_memory=False)

    real_dir = SONICS_DIR / "real_songs"
    fake_dir = SONICS_DIR / "fake_songs"

    tracks: list[dict] = []

    # Real tracks
    for _, row in df_real.iterrows():
        yt_id = str(row.get("youtube_id", "")).strip()
        if not yt_id:
            continue
        # Resolve audio path
        audio_path = row.get("audio_path", "")
        if audio_path and Path(audio_path).exists():
            src = Path(audio_path)
        else:
            candidates = list(real_dir.glob(f"{yt_id}.*")) if real_dir.exists() else []
            if not candidates:
                continue
            src = candidates[0]

        dst = output_dir / "real" / f"{yt_id}.wav"
        tracks.append(
            {
                "src_path": str(src),
                "dst_path": str(dst),
                "track_id": yt_id,
                "label": "real",
                "algorithm": "",
                "fake_label": "",
                "genre": "",
                "artist": str(row.get("artist", "")),
                "year": str(row.get("year", "")),
                "target_sr": target_sr,
                "mp3_bitrate": mp3_bitrate,
                "target_lufs": target_lufs,
                "max_duration": max_duration,
            }
        )
        if max_per_class and len([t for t in tracks if t["label"] == "real"]) >= max_per_class:
            break

    # Fake tracks
    # Build fake audio index once — handles SONICS subdirectory layout
    # (fake_songs/chirp-v3.5/abc.mp3) that flat glob misses.
    fake_index: dict[str, Path] = {}
    if fake_dir.exists():
        for _ff in fake_dir.rglob("*"):
            if _ff.is_file() and _ff.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
                fake_index[_ff.stem.lower()] = _ff
        logger.info("Built fake audio index: %d files under %s", len(fake_index), fake_dir)

    n_fake = 0
    for _, row in df_fake.iterrows():
        filename = str(row.get("filename", row.get("id", ""))).strip()
        if not filename:
            continue
        audio_path = row.get("audio_path", "")
        if audio_path and Path(audio_path).exists():
            src = Path(audio_path)
        else:
            stem = Path(filename).stem
            _indexed = fake_index.get(stem.lower())
            if _indexed:
                src = _indexed
            else:
                candidates = list(fake_dir.glob(f"{stem}.*")) if fake_dir.exists() else []
                if not candidates:
                    continue
                src = candidates[0]

        track_id = Path(filename).stem
        dst = output_dir / "fake" / f"{track_id}.wav"
        tracks.append(
            {
                "src_path": str(src),
                "dst_path": str(dst),
                "track_id": track_id,
                "label": "fake",
                "algorithm": str(row.get("algorithm", "")),
                "fake_label": str(row.get("label", "")),
                "genre": str(row.get("genre", "")),
                "artist": "",
                "year": "",
                "target_sr": target_sr,
                "mp3_bitrate": mp3_bitrate,
                "target_lufs": target_lufs,
                "max_duration": max_duration,
            }
        )
        n_fake += 1
        if max_per_class and n_fake >= max_per_class:
            break

    return tracks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical matched audio corpus (MP3 round-trip + LUFS normalisation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/processed/canonical", help="Output directory")
    parser.add_argument("--target-sr", type=int, default=CANONICAL_TARGET_SR, help="Target sample rate")
    parser.add_argument("--mp3-bitrate", type=int, default=CANONICAL_MP3_KBPS, help="MP3 round-trip bitrate (kbps)")
    parser.add_argument("--target-lufs", type=float, default=CANONICAL_TARGET_LUFS, help="LUFS normalisation target")
    parser.add_argument("--max-duration", type=float, default=CANONICAL_MAX_DURATION, help="Max track duration (s)")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help=(
            "Parallel worker processes. "
            "Recommended: 6-8 on g4dn.xlarge (4 vCPUs); I/O wait allows oversubscription. "
            "Each worker spawns 2 ffmpeg subprocesses. Monitor htop; reduce if load > 12."
        ),
    )
    parser.add_argument("--max-per-class", type=int, default=None, help="Limit tracks per class (for testing)")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=500,
        help="Flush manifest to disk every N completions (crash safety). Reduce if disk writes are slow.",
    )
    parser.add_argument(
        "--delete-src-after",
        action="store_true",
        default=False,
        help=(
            "Delete the source MP3/M4A immediately after its canonical WAV is written. "
            "Saves ~5 MB per track while adding ~5.5 MB, so net disk growth is only ~0.5 MB/track. "
            "WARNING: irreversible. Use --cleanup-srcs --dry-run to preview deletions first."
        ),
    )
    parser.add_argument(
        "--cleanup-srcs",
        action="store_true",
        default=False,
        help=(
            "Read the existing manifest and delete source files for all 'ok' rows. "
            "Use this to free up disk for tracks already canonicalized in a prior run "
            "(e.g. before the --delete-src-after flag existed). "
            "Combine with --dry-run to preview without deleting."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="With --cleanup-srcs: show what would be deleted without deleting anything.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "canonical_manifest.csv"

    # --- Cleanup mode: delete src files for already-canonicalized tracks ---
    if args.cleanup_srcs:
        if not manifest_path.exists():
            logger.error(
                "No manifest found at %s — nothing to clean up. "
                "Run the corpus build first (without --cleanup-srcs).",
                manifest_path,
            )
            sys.exit(1)
        df_m = pd.read_csv(manifest_path, low_memory=False)
        ok_rows = df_m[df_m.get("status", pd.Series(dtype=str)).astype(str) == "ok"]
        mode_label = "DRY RUN" if args.dry_run else "LIVE"
        logger.info("Cleanup mode (%s): examining %d ok rows in %s", mode_label, len(ok_rows), manifest_path)
        deleted = skipped = failed = 0
        saved_bytes = 0
        for _, row in ok_rows.iterrows():
            src = Path(str(row.get("src_path", ""))).resolve()
            if not src.exists():
                skipped += 1
                continue
            sz = src.stat().st_size
            if args.dry_run:
                logger.info("[DRY-RUN] would delete: %s (%.1f MB)", src, sz / 1e6)
                deleted += 1
                saved_bytes += sz
            else:
                try:
                    src.unlink()
                    deleted += 1
                    saved_bytes += sz
                except OSError as exc:
                    logger.warning("Cannot delete %s: %s", src, exc)
                    failed += 1
        prefix = "[DRY-RUN] " if args.dry_run else ""
        logger.info(
            "%sCleanup done: %d deleted, %d already gone, %d failed — %.2f GB freed",
            prefix,
            deleted,
            skipped,
            failed,
            saved_bytes / 1e9,
        )
        return

    # Load existing manifest to detect already-processed tracks
    existing_ids: set[str] = set()
    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path, low_memory=False)
        ok_mask = existing_df.get("status", pd.Series(dtype=str)) == "ok"
        existing_ids = set(existing_df.loc[ok_mask, "track_id"].astype(str))
        logger.info("Resuming: %d tracks already in manifest", len(existing_ids))

    tracks = _load_track_list(
        output_dir=output_dir,
        target_sr=args.target_sr,
        mp3_bitrate=args.mp3_bitrate,
        target_lufs=args.target_lufs,
        max_duration=args.max_duration,
        max_per_class=args.max_per_class,
    )
    # Skip already processed
    todo = [t for t in tracks if t["track_id"] not in existing_ids]
    logger.info("Tracks to process: %d  (skipping %d already done)", len(todo), len(tracks) - len(todo))

    # Inject runtime flag so the worker can delete sources after writing
    delete_src_after = bool(args.delete_src_after)
    for t in todo:
        t["delete_src_after"] = delete_src_after
    if delete_src_after:
        logger.info("--delete-src-after enabled: source files will be deleted after successful canonicalization")

    if not todo:
        logger.info("Nothing to do — corpus is up to date.")
        return

    # --- disk space pre-check ---
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    n_remaining = len(todo)
    # 24kHz mono 16-bit WAV, 120s cap → ~5.5 MB per track
    est_gb = n_remaining * 5.5 / 1024
    logger.info("Disk space: %.1f GB free, estimated %.1f GB needed for %d tracks", free_gb, est_gb, n_remaining)
    if free_gb < est_gb * 1.15:
        logger.warning(
            "LOW DISK: only %.1f GB free but ~%.1f GB needed (115%% of estimate). "
            "Consider --max-per-class to build a subsample first, or expand EBS.",
            free_gb,
            est_gb * 1.15,
        )

    results: list[dict] = []
    t0 = time.time()
    done = 0
    # Incremental flush: append new rows to manifest every FLUSH_EVERY completions.
    # This means a crash only loses the last FLUSH_EVERY tracks, not the whole run.
    FLUSH_EVERY = args.flush_every
    flushed_ids: set[str] = set(existing_ids)  # track what's already on disk

    def _flush_results(batch: list[dict]) -> None:
        """Append a batch of result rows to the manifest CSV."""
        if not batch:
            return
        new_df = pd.DataFrame(batch)
        write_header = not manifest_path.exists()
        new_df.to_csv(manifest_path, mode="a", header=write_header, index=False)
        flushed_ids.update(str(r.get("track_id", "")) for r in batch)

    flush_buffer: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in todo}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
                # Only buffer rows that are genuinely new (skip already-flushed)
                tid = str(res.get("track_id", ""))
                if tid not in flushed_ids:
                    flush_buffer.append(res)
            done += 1

            # Incremental flush
            if len(flush_buffer) >= FLUSH_EVERY:
                _flush_results(flush_buffer)
                flush_buffer = []

            if done % 100 == 0 or done == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / done * max(len(todo) - done, 0)
                ok = sum(1 for r in results if r.get("status") == "ok")
                err = sum(1 for r in results if r.get("status", "").startswith("error"))
                logger.info(
                    "[%d/%d] ok=%d err=%d  elapsed=%.0fm eta=%.0fm", done, len(todo), ok, err, elapsed / 60, eta / 60
                )

    # Final flush for any remaining rows
    _flush_results(flush_buffer)

    # Re-read the full manifest (all previous + newly written rows) for final stats
    out_df = pd.read_csv(manifest_path, low_memory=False) if manifest_path.exists() else pd.DataFrame(results)

    ok_mask = out_df.get("status", pd.Series(dtype=str)) == "ok"
    ok_n = int(ok_mask.sum())
    err_n = int(out_df["status"].astype(str).str.startswith("error").sum()) if "status" in out_df.columns else 0
    logger.info("Done. Manifest: %d ok, %d errors → %s", ok_n, err_n, manifest_path)

    # Summary stats
    ok_df = out_df[out_df["status"].astype(str) == "ok"] if "status" in out_df.columns else out_df
    if len(ok_df):
        for label in ["real", "fake"]:
            sub = ok_df[ok_df["label"] == label]
            if len(sub):
                lufs_vals = pd.to_numeric(sub.get("measured_lufs", pd.Series(dtype=float)), errors="coerce")
                dur_vals = pd.to_numeric(sub.get("actual_duration", pd.Series(dtype=float)), errors="coerce")
                logger.info(
                    "  %s: n=%d  LUFS mean=%.1f±%.1f  dur mean=%.0f±%.0fs",
                    label,
                    len(sub),
                    lufs_vals.mean(),
                    lufs_vals.std(),
                    dur_vals.mean(),
                    dur_vals.std(),
                )


if __name__ == "__main__":
    main()
