"""Build canonical corpus for exactly the tracks the balanced sampler will select.

Unlike build_canonical_corpus.py (which processes all ~69k tracks), this script:

1. Calls load_balanced_tracks() to determine which fake tracks are needed
   (stratified per algorithm × fake_label, up to --per-stratum each).
2. Expands the real pool by --real-pool-multiplier (default 3×) so the canonical
   manifest has enough real-track variety for genre / covariate matching later.
3. Checks --existing-manifest first: any track already canonicalized there is
   reused (no reprocessing, no extra disk).
4. Canonicalizes only the remaining tracks into --output-dir.
5. Writes a combined canonical_manifest.csv that the sampler can use directly.
6. Optionally deletes source files after writing (--delete-src-after).

Disk comparison (per-stratum=200, 3× real pool)
------------------------------------------------
  Full corpus build                : 69k × 5.5 MB ≈ 380 GB
  Sampler build, keep sources      :  ~15k × 5.5 MB ≈  83 GB
  Sampler build, --delete-src-after:  net +8 GB    (replaces ~5 MB source with 5.5 MB WAV)

Recommended workflow (full build already in progress)
------------------------------------------------------
# Step 1: free disk for the 6700+ tracks already canonicalized
poetry run python scripts/build_canonical_corpus.py \\
    --output-dir data/processed/canonical \\
    --cleanup-srcs

# Step 2: build canonical for sampler (reuses already-done tracks from Step 1)
poetry run python scripts/build_canonical_for_sampler.py \\
    --output-dir data/processed/canonical_sampler \\
    --existing-manifest data/processed/canonical/canonical_manifest.csv \\
    --delete-src-after \\
    --workers 8 \\
    2>&1 | tee logs/canonical_sampler.log
"""

from __future__ import annotations

import argparse
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
from intrinsic_ai_music_detection.data.balanced_sampling import load_balanced_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker (top-level so it's picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _process_one(args: dict) -> dict:
    """Canonicalize one track. Returns a manifest row dict."""
    src_path = Path(args["src_path"])
    dst_path = Path(args["dst_path"])
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
            target_sr=args["target_sr"],
            mode="canonical",
            max_duration=args["max_duration"],
            mp3_bitrate_kbps=args["mp3_bitrate"],
            target_lufs=args["target_lufs"],
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(dst_path), sr, audio_int16)

        if delete_src_after and src_path.exists():
            if str(src_path.resolve()) != str(dst_path.resolve()):
                try:
                    src_path.unlink()
                except OSError as exc:
                    import logging as _l

                    _l.getLogger(__name__).warning("Could not delete source %s: %s", src_path, exc)

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
# Helpers
# ---------------------------------------------------------------------------


def _load_existing_manifest(path: str | None) -> dict[str, dict]:
    """Return {track_id: manifest_row} for all ok rows in an existing manifest."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("--existing-manifest %s not found — starting fresh.", p)
        return {}
    df = pd.read_csv(p, low_memory=False)
    ok = df[df.get("status", pd.Series(dtype=str)).astype(str) == "ok"]
    result = {}
    for _, row in ok.iterrows():
        tid = str(row.get("track_id", "")).strip()
        canon = str(row.get("canonical_path", "")).strip()
        if tid and canon and Path(canon).exists():
            result[tid] = row.to_dict()
    logger.info("Existing manifest: %d reusable ok rows from %s", len(result), p)
    return result


def _track_to_job(
    track: dict,
    output_dir: Path,
    target_sr: int,
    mp3_bitrate: int,
    target_lufs: float,
    max_duration: float,
    delete_src_after: bool,
) -> dict:
    """Convert a balanced-sampler track dict to a _process_one job dict."""
    label = track["label"]
    track_id = track["track_id"]
    src_path = track["path"]
    dst_path = output_dir / label / f"{track_id}.wav"
    return {
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "track_id": track_id,
        "label": label,
        "algorithm": track.get("algorithm", ""),
        "fake_label": track.get("fake_label", ""),
        "genre": track.get("genre", ""),
        "artist": track.get("artist", ""),
        "year": track.get("year", ""),
        "target_sr": target_sr,
        "mp3_bitrate": mp3_bitrate,
        "target_lufs": target_lufs,
        "max_duration": max_duration,
        "delete_src_after": delete_src_after,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical corpus scoped to exactly the tracks the balanced sampler needs, "
            "reusing any already-canonicalized tracks from --existing-manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/processed/canonical_sampler")
    parser.add_argument(
        "--existing-manifest",
        default=None,
        help=(
            "Path to a canonical_manifest.csv from a prior build (e.g. "
            "data/processed/canonical/canonical_manifest.csv). "
            "Any track already there with status=ok is reused; only the rest are processed."
        ),
    )
    # Preprocessing parameters (must match the existing manifest if reusing)
    parser.add_argument("--target-sr", type=int, default=CANONICAL_TARGET_SR)
    parser.add_argument("--mp3-bitrate", type=int, default=CANONICAL_MP3_KBPS)
    parser.add_argument("--target-lufs", type=float, default=CANONICAL_TARGET_LUFS)
    parser.add_argument("--max-duration", type=float, default=CANONICAL_MAX_DURATION)
    # Sampler parameters (determines which tracks to include)
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=200,
        help="Max fake tracks per (algorithm × fake_label) stratum.",
    )
    parser.add_argument(
        "--real-pool-multiplier",
        type=float,
        default=3.0,
        help=(
            "Multiply the target real count by this factor to build a larger real pool. "
            "Needed so that genre/covariate matching has enough variety later. "
            "E.g. 3× means canonicalise 3× as many reals as fakes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    # Execution parameters
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel worker processes. Recommended 6-8 on g4dn.xlarge.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=200,
        help="Flush manifest to disk every N completions.",
    )
    parser.add_argument(
        "--delete-src-after",
        action="store_true",
        default=False,
        help=(
            "Delete each source MP3/M4A immediately after its canonical WAV is written. "
            "Net disk growth: ~0.5 MB per track (5.5 MB WAV minus ~5 MB source). "
            "WARNING: irreversible."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "canonical_manifest.csv"
    Path("logs").mkdir(exist_ok=True)

    # ---- 1. Determine which tracks the sampler needs ----
    logger.info(
        "Determining sampler track set (per_stratum=%d, real_multiplier=%.1fx)…",
        args.per_stratum,
        args.real_pool_multiplier,
    )

    # Fakes: exactly the stratified set
    _, fake_tracks = load_balanced_tracks(per_stratum=args.per_stratum, seed=args.seed)
    n_fakes = len(fake_tracks)
    logger.info("Fake tracks needed (stratified): %d", n_fakes)

    # Reals: expanded pool so genre/covariate matching has room
    n_real_pool = int(n_fakes * args.real_pool_multiplier)
    real_tracks, _ = load_balanced_tracks(per_stratum=args.per_stratum, seed=args.seed, max_real=n_real_pool)
    n_reals = len(real_tracks)
    logger.info("Real tracks in pool (%s× multiplier): %d", args.real_pool_multiplier, n_reals)
    logger.info("Total tracks needed: %d", n_fakes + n_reals)

    all_needed = real_tracks + fake_tracks

    # ---- 2. Load existing manifest (reuse already-canonicalized tracks) ----
    existing: dict[str, dict] = _load_existing_manifest(args.existing_manifest)

    # ---- 3. Determine which still need processing ----
    # Also check this script's own output manifest for resumed runs
    own_done: set[str] = set()
    if manifest_path.exists():
        own_df = pd.read_csv(manifest_path, low_memory=False)
        own_done = set(
            own_df.loc[own_df.get("status", pd.Series(dtype=str)).astype(str) == "ok", "track_id"].astype(str)
        )
        logger.info("Resuming: %d tracks already in own manifest", len(own_done))

    reuse_rows: list[dict] = []
    jobs: list[dict] = []
    for track in all_needed:
        tid = track["track_id"]
        if tid in own_done:
            continue  # already in this script's manifest
        if tid in existing:
            reuse_rows.append(existing[tid])  # borrow from existing full build
        else:
            jobs.append(
                _track_to_job(
                    track,
                    output_dir,
                    target_sr=args.target_sr,
                    mp3_bitrate=args.mp3_bitrate,
                    target_lufs=args.target_lufs,
                    max_duration=args.max_duration,
                    delete_src_after=args.delete_src_after,
                )
            )

    logger.info(
        "Plan: %d reused from existing manifest, %d own-manifest skip, %d to process",
        len(reuse_rows),
        len(own_done),
        len(jobs),
    )

    # ---- 4. Disk pre-check ----
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    est_new_gb = len(jobs) * 5.5 / 1024
    logger.info("Disk: %.1f GB free, ~%.1f GB needed for %d new tracks", free_gb, est_new_gb, len(jobs))
    if free_gb < est_new_gb * 1.15:
        logger.warning(
            "LOW DISK: %.1f GB free < %.1f GB needed. " "Consider --delete-src-after or expand EBS before continuing.",
            free_gb,
            est_new_gb * 1.15,
        )

    if args.delete_src_after:
        logger.info("--delete-src-after enabled: source files will be deleted after writing")

    # ---- 5. Write reused rows to own manifest first ----
    def _flush(batch: list[dict]) -> None:
        if not batch:
            return
        df_batch = pd.DataFrame(batch)
        write_header = not manifest_path.exists()
        df_batch.to_csv(manifest_path, mode="a", header=write_header, index=False)

    if reuse_rows:
        _flush(reuse_rows)
        logger.info("Flushed %d reused rows to manifest", len(reuse_rows))

    if not jobs:
        logger.info("Nothing new to process — all tracks already covered.")
    else:
        # ---- 6. Canonicalize missing tracks ----
        FLUSH_EVERY = args.flush_every
        flush_buffer: list[dict] = []
        t0 = time.time()
        done = 0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, j): j for j in jobs}
            for future in as_completed(futures):
                res = future.result()
                done += 1
                if res is not None:
                    flush_buffer.append(res)
                if len(flush_buffer) >= FLUSH_EVERY:
                    _flush(flush_buffer)
                    flush_buffer = []
                if done % 100 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    eta = elapsed / done * max(len(jobs) - done, 0)
                    ok_n = sum(1 for r in flush_buffer if r.get("status") == "ok")
                    logger.info("[%d/%d] elapsed=%.0fm eta=%.0fm", done, len(jobs), elapsed / 60, eta / 60)

        _flush(flush_buffer)  # final flush

    # ---- 7. Summary ----
    if manifest_path.exists():
        final_df = pd.read_csv(manifest_path, low_memory=False)
        ok_n = int((final_df.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum())
        err_n = int(final_df["status"].astype(str).str.startswith("error").sum()) if "status" in final_df.columns else 0
        logger.info("Manifest: %d ok, %d errors → %s", ok_n, err_n, manifest_path)

        for lbl in ["real", "fake"]:
            sub = final_df[(final_df["label"] == lbl) & (final_df.get("status", "ok") == "ok")]
            if len(sub):
                lufs_vals = pd.to_numeric(sub.get("measured_lufs", pd.Series(dtype=float)), errors="coerce")
                dur_vals = pd.to_numeric(sub.get("actual_duration", pd.Series(dtype=float)), errors="coerce")
                logger.info(
                    "  %s: n=%d  LUFS %.1f±%.1f  dur %.0f±%.0fs",
                    lbl,
                    len(sub),
                    lufs_vals.mean(),
                    lufs_vals.std(),
                    dur_vals.mean(),
                    dur_vals.std(),
                )

        fake_ok = final_df[(final_df["label"] == "fake") & (final_df.get("status", "ok") == "ok")]
        if len(fake_ok) and "algorithm" in fake_ok.columns:
            logger.info("Fake strata coverage:")
            strata_col = "fake_label" if "fake_label" in fake_ok.columns else "label"
            for (algo, fl), grp in fake_ok.groupby(["algorithm", strata_col]):
                logger.info("  %-20s %-12s  n=%d", algo, fl, len(grp))

        logger.info("=" * 60)
        logger.info("Next steps (run after this script completes):")
        logger.info("  1. tag_real_genres.py --canonical-manifest %s", manifest_path)
        logger.info("  2. profile_covariates.py --canonical-manifest %s --workers 4", manifest_path)
        logger.info(
            "  3. run_acoustic_analysis.py --canonical-manifest %s --preprocess-mode preprocessed", manifest_path
        )
        logger.info("  4. validate_subsample.py")
        logger.info(
            "  5. run_balanced_ablation.py --canonical-manifest %s --preprocess-mode preprocessed", manifest_path
        )


if __name__ == "__main__":
    main()
