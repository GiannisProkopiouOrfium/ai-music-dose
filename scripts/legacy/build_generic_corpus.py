"""Build a canonical corpus from any directory of audio files.

Creates a canonical_manifest.csv that is directly compatible with
run_balanced_ablation.py --canonical-manifest.

Uses the same preprocess_audio pipeline as build_canonical_corpus.py:
  64 kbps MP3 round-trip → LUFS normalisation → resample → WAV int16

Usage examples
--------------
# FMA real music:
poetry run python scripts/build_generic_corpus.py \\
    --input-dir data/raw/fma_small \\
    --output-dir data/processed/canonical_fma \\
    --label real --source fma \\
    --workers 6

# FakeMusicCaps MusicGen:
poetry run python scripts/build_generic_corpus.py \\
    --input-dir data/raw/fakemusiccaps/musicgen \\
    --output-dir data/processed/canonical_fmc \\
    --label fake --source fakemusiccaps --algorithm musicgen \\
    --fake-label "full fake" \\
    --workers 6

# Merge multiple manifests (run after building each):
poetry run python scripts/build_generic_corpus.py \\
    --merge-manifests \\
      data/processed/canonical_sonics_full/canonical_manifest.csv \\
      data/processed/canonical_fma/canonical_manifest.csv \\
      data/processed/canonical_fmc/canonical_manifest.csv \\
    --output-manifest data/processed/combined_manifest.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ---------------------------------------------------------------------------
# Worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


def _process_one(args: dict) -> dict:
    src = Path(args["src_path"])
    dst = Path(args["dst_path"])

    if dst.exists():
        return {**args, "canonical_path": str(dst), "status": "ok"}

    try:
        audio, sr, meta = preprocess_audio(
            src,
            target_sr=args["target_sr"],
            mode="canonical",
            max_duration=args["max_duration"],
            mp3_bitrate_kbps=args["mp3_bitrate"],
            target_lufs=args["target_lufs"],
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(dst), sr, audio_int16)
        return {
            **args,
            "canonical_path": str(dst),
            "measured_lufs": meta.get("measured_lufs", float("nan")),
            "measured_peak": meta.get("measured_peak", float("nan")),
            "actual_duration": meta.get("actual_duration", float("nan")),
            "status": "ok",
        }
    except Exception as exc:
        return {**args, "status": f"error: {exc}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize any audio directory → canonical_manifest.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=str, default=None, help="Root directory containing audio files (recursive search)."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write canonical WAVs + manifest.")
    parser.add_argument(
        "--label", choices=["real", "fake"], default="real", help="Label for all tracks in this corpus."
    )
    parser.add_argument(
        "--source", type=str, default="", help="Source tag written to manifest (e.g. 'fma', 'fakemusiccaps')."
    )
    parser.add_argument(
        "--algorithm", type=str, default="", help="Algorithm name for fake tracks (e.g. 'musicgen', 'audioldm2')."
    )
    parser.add_argument("--fake-label", type=str, default="full fake", help="fake_label column value for fake tracks.")
    parser.add_argument(
        "--track-id-prefix",
        type=str,
        default="",
        help=(
            "Prefix for generated track_ids ('<prefix>__<relative path>'). Use 'auto' to derive it "
            "from the variant (algorithm for fakes, 'real' for reals). REQUIRED for corpora that "
            "re-generate the same source id with several systems (e.g. FakeMusicCaps: one "
            "MusicCaps YouTube id -> 1 real + 5 TTM variants): the embedding cache is keyed on "
            "track_id alone, so colliding ids make every variant silently read one variant's "
            "embeddings and invalidate all downstream scores. Default '' preserves legacy ids for "
            "corpora whose paths are already unique (e.g. FMA). Verify with "
            "scripts/diagnose_cache_collisions.py."
        ),
    )
    # Preprocessing
    parser.add_argument("--target-sr", type=int, default=CANONICAL_TARGET_SR)
    parser.add_argument("--mp3-bitrate", type=int, default=CANONICAL_MP3_KBPS)
    parser.add_argument("--target-lufs", type=float, default=CANONICAL_TARGET_LUFS)
    parser.add_argument("--max-duration", type=float, default=CANONICAL_MAX_DURATION)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--flush-every", type=int, default=200)
    # Manifest merge mode
    parser.add_argument(
        "--merge-manifests",
        nargs="+",
        default=None,
        metavar="CSV",
        help="Merge multiple canonical_manifest.csv files into one. "
        "Provide list of paths. Use --output-manifest for the destination.",
    )
    parser.add_argument("--output-manifest", type=str, default=None, help="Output path for --merge-manifests.")
    args = parser.parse_args()

    # ---- Merge mode -------------------------------------------------------
    if args.merge_manifests:
        if not args.output_manifest:
            parser.error("--output-manifest required with --merge-manifests")
        dfs = []
        for p in args.merge_manifests:
            if not Path(p).exists():
                logger.warning("Manifest not found, skipping: %s", p)
                continue
            df = pd.read_csv(p, low_memory=False)
            dfs.append(df)
            logger.info("  Loaded %d rows from %s", len(df), p)
        if not dfs:
            logger.error("No manifests loaded. Abort.")
            sys.exit(1)
        combined = pd.concat(dfs, ignore_index=True)
        # Drop exact duplicates (same track_id+canonical_path)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["track_id", "canonical_path"], keep="first")
        logger.info("Combined: %d rows (dropped %d duplicates)", len(combined), before - len(combined))
        ok = combined[combined.get("status", pd.Series("ok")) == "ok"]
        logger.info(
            "OK tracks: %d  (real=%d  fake=%d)", len(ok), (ok["label"] == "real").sum(), (ok["label"] == "fake").sum()
        )
        if "algorithm" in ok.columns:
            alg_counts = ok[ok["label"] == "fake"]["algorithm"].value_counts()
            for alg, n in alg_counts.items():
                logger.info("  %-30s %6d", alg, n)
        Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output_manifest, index=False)
        logger.info("Written → %s", args.output_manifest)
        return

    # ---- Build mode -------------------------------------------------------
    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required (or use --merge-manifests)")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "canonical_manifest.csv"

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # Collect audio files
    audio_files = sorted(f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
    logger.info("Found %d audio files in %s", len(audio_files), input_dir)
    if not audio_files:
        logger.error("No audio files found. Supported: %s", AUDIO_EXTENSIONS)
        sys.exit(1)

    # Load existing manifest to resume. Track the (label, algorithm) each id was
    # written under: an id reappearing under a DIFFERENT variant is a collision,
    # not a resume — see the guard below.
    existing_ids: set[str] = set()
    existing_variant: dict[str, tuple[str, str]] = {}
    if manifest_path.exists():
        ex = pd.read_csv(manifest_path, low_memory=False)
        ok_mask = ex.get("status", pd.Series(dtype=str)) == "ok"
        ex_ok = ex.loc[ok_mask]
        existing_ids = set(ex_ok["track_id"].astype(str))
        if "label" in ex_ok.columns:
            _alg = ex_ok.get("algorithm", pd.Series([""] * len(ex_ok))).fillna("").astype(str)
            existing_variant = dict(zip(ex_ok["track_id"].astype(str), zip(ex_ok["label"].astype(str), _alg)))
        logger.info("Resuming: %d tracks already processed", len(existing_ids))

    # Build task list
    subdirname = args.label  # real/ or fake/
    this_variant = (args.label, args.algorithm if args.label == "fake" else "")
    prefix = args.track_id_prefix
    if prefix == "auto":
        # Corpora that re-generate the SAME source id with several systems
        # (FakeMusicCaps: one MusicCaps YouTube id -> 1 real + 5 TTM variants)
        # MUST get distinct ids, because the embedding cache is keyed on
        # track_id alone — colliding ids silently make every variant read one
        # variant's embeddings. See scripts/diagnose_cache_collisions.py.
        prefix = (args.algorithm or "fake") if args.label == "fake" else "real"
    if prefix:
        logger.info("track_id prefix: %r (ids will be '<prefix>__<relative path>')", prefix)

    n_collisions = 0
    tasks: list[dict] = []
    for f in audio_files:
        # Use relative path from input_dir as track_id (collapses subdirs)
        rel = f.relative_to(input_dir)
        track_id = str(rel.with_suffix("")).replace("/", "__").replace("\\", "__")
        if prefix:
            track_id = f"{prefix}__{track_id}"
        if track_id in existing_ids:
            prev = existing_variant.get(track_id)
            if prev is not None and prev != this_variant:
                n_collisions += 1
                continue
            continue  # genuine resume: same variant already done
        dst = output_dir / subdirname / f"{track_id}.wav"
        tasks.append(
            {
                "src_path": str(f),
                "dst_path": str(dst),
                "track_id": track_id,
                "label": args.label,
                "algorithm": args.algorithm if args.label == "fake" else "",
                "fake_label": args.fake_label if args.label == "fake" else "",
                "source": args.source,
                "genre": "",
                "target_sr": args.target_sr,
                "mp3_bitrate": args.mp3_bitrate,
                "target_lufs": args.target_lufs,
                "max_duration": args.max_duration,
            }
        )

    if n_collisions:
        logger.error(
            "TRACK-ID COLLISION: %d ids from this run already exist in the manifest under a "
            "DIFFERENT (label, algorithm). Because the embedding cache is keyed on track_id "
            "alone, colliding ids make every variant silently read one variant's embeddings, "
            "invalidating all downstream scores. Re-run with --track-id-prefix auto (or an "
            "explicit prefix) and rebuild the affected embedding cache. Aborting.",
            n_collisions,
        )
        sys.exit(1)

    logger.info("Tasks to process: %d  (skipping %d already done)", len(tasks), len(audio_files) - len(tasks))

    if not tasks:
        logger.info("Nothing to do. Manifest is up to date.")
        return

    # Process
    results: list[dict] = []
    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), start=1):
            row = fut.result()
            results.append(row)
            if row.get("status") == "ok":
                n_ok += 1
            else:
                n_err += 1
                logger.warning("  %s → %s", row.get("track_id"), row.get("status"))

            if i % args.flush_every == 0 or i == len(tasks):
                # Append to manifest (read-modify-write, safe)
                new_df = pd.DataFrame(results)
                if manifest_path.exists():
                    existing_df = pd.read_csv(manifest_path, low_memory=False)
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["track_id"], keep="last")
                else:
                    combined = new_df
                combined.to_csv(manifest_path, index=False)
                results.clear()
                logger.info("  %d/%d  ok=%d  err=%d", i, len(tasks), n_ok, n_err)

    logger.info("Done: %d ok, %d errors → %s", n_ok, n_err, manifest_path)

    # Summary
    df_final = pd.read_csv(manifest_path, low_memory=False)
    ok = df_final[df_final.get("status", "ok") == "ok"]
    logger.info("Manifest: %d total rows, %d ok (%s)", len(df_final), len(ok), args.label)


if __name__ == "__main__":
    main()
