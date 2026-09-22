"""Diagnose how much of the OFFICIAL SONICS real-music corpus we actually have.

MusicDET (and SONICS itself) trains on the *full* SONICS real-music subset:
48,090 songs from 9,096 artists (Rahman et al., 2025). Our pipeline has been
training the headline flow on only 12,722 real tracks. This script answers
the question "is 12,722 all that's available (audio download attrition), or
is it an artificial subsample of a larger locally-fetchable pool?" so we can
make a data-driven decision about scaling up before re-running the headline
experiment, rather than guessing.

It reports, with no downloads:
  1. Total rows / unique youtube_ids in the official real_songs.csv metadata.
  2. How many of those youtube_ids already have audio on disk (S3-cached).
  3. How many are in our current canonical corpus (the 12,722 used so far).
  4. Split-column breakdown (train/valid/test) for both the full metadata
     and the locally-available subset, so we know if the gap is concentrated
     in one split.
  5. (Optional, --probe-missing N) A lightweight, download-free "is this
     video still up" check via YouTube's oEmbed endpoint for a random sample
     of MISSING youtube_ids, to estimate what fraction of the gap is
     realistically recoverable vs. permanently dead (YouTube-sourced datasets
     decay over time as videos get taken down).

Usage
-----
python scripts/check_sonics_real_coverage.py
python scripts/check_sonics_real_coverage.py --probe-missing 200
"""

from __future__ import annotations

import argparse
import logging
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")
REAL_SONGS_DIR = SONICS_DIR / "real_songs"
METADATA_DIR = SONICS_DIR / "metadata"


def _probe_youtube_alive(youtube_id: str, timeout: float = 5.0) -> bool:
    """Check if a YouTube video is still reachable, without downloading it.

    Uses the public oEmbed endpoint (no API key, no yt-dlp dependency) —
    returns 200 if the video exists and is embeddable, 404/401 otherwise.
    """
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={youtube_id}&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canonical-manifest", default="data/processed/canonical_sonics_full/canonical_manifest.csv")
    parser.add_argument(
        "--probe-missing",
        type=int,
        default=0,
        help="If >0, sample this many MISSING youtube_ids and check YouTube availability via oEmbed (no download).",
    )
    parser.add_argument("--probe-sleep", type=float, default=0.3, help="Seconds between oEmbed probes (be polite).")
    args = parser.parse_args()

    real_csv = METADATA_DIR / "real_songs.csv"
    if not real_csv.exists():
        logger.error(
            "%s not found. Run: python scripts/download_sonics.py --download-fakes "
            "(fetches the HF metadata CSVs, including real_songs.csv)",
            real_csv,
        )
        return

    df_real = pd.read_csv(real_csv, low_memory=False)
    n_total = len(df_real)
    n_unique_yt = df_real["youtube_id"].dropna().nunique()

    logger.info("=" * 70)
    logger.info("OFFICIAL SONICS real_songs.csv (from HuggingFace awsaf49/sonics)")
    logger.info("=" * 70)
    logger.info("Total metadata rows      : %d", n_total)
    logger.info("Unique youtube_ids       : %d", n_unique_yt)
    if n_total < 40_000:
        logger.warning(
            "This is far short of the paper's reported 48,090 real songs. "
            "Either the HF release is a partial snapshot, or a --split filter "
            "is needed. Check df_real.columns / df_real['split'].value_counts() below."
        )
    if "split" in df_real.columns:
        logger.info("Split breakdown (full metadata):\n%s", df_real["split"].value_counts().to_string())

    # How many have audio on disk already (S3-cached)?
    existing_yt_ids: set[str] = set()
    if REAL_SONGS_DIR.exists():
        existing_yt_ids = {
            f.stem for f in REAL_SONGS_DIR.iterdir() if f.suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
        }
    df_real["youtube_id"] = df_real["youtube_id"].astype(str)
    df_real["has_audio_local"] = df_real["youtube_id"].isin(existing_yt_ids)
    n_local = int(df_real["has_audio_local"].sum())

    logger.info("-" * 70)
    logger.info("Locally cached audio (%s)", REAL_SONGS_DIR)
    logger.info("-" * 70)
    logger.info("Tracks with audio on disk : %d / %d  (%.1f%%)", n_local, n_total, 100 * n_local / max(n_total, 1))
    if "split" in df_real.columns:
        logger.info(
            "Split breakdown (locally available):\n%s",
            df_real[df_real["has_audio_local"]]["split"].value_counts().to_string(),
        )

    # How many are in our current canonical corpus (the 12,722 used so far)?
    canon_path = Path(args.canonical_manifest)
    if canon_path.exists():
        canon_df = pd.read_csv(canon_path, low_memory=False)
        canon_real_ids = set(canon_df.loc[canon_df.get("label") == "real", "track_id"].astype(str))
        logger.info("-" * 70)
        logger.info("Current canonical corpus (%s)", canon_path)
        logger.info("-" * 70)
        logger.info("Real tracks in canonical manifest : %d", len(canon_real_ids))
        n_local_not_in_canon = len(existing_yt_ids - canon_real_ids)
        if n_local_not_in_canon:
            logger.warning(
                "%d locally-cached real tracks are NOT in the canonical manifest — "
                "these can be added to the corpus for FREE (no download needed) by "
                "re-running build_canonical_corpus.py.",
                n_local_not_in_canon,
            )
    else:
        logger.info("Canonical manifest not found at %s (skipping cross-check)", canon_path)

    n_missing = n_total - n_local
    logger.info("=" * 70)
    logger.info(
        "GAP: %d official real tracks have NO local audio (%.1f%% of the full corpus)",
        n_missing,
        100 * n_missing / max(n_total, 1),
    )
    logger.info("=" * 70)

    if args.probe_missing > 0 and n_missing > 0:
        missing_ids = df_real.loc[~df_real["has_audio_local"], "youtube_id"].dropna().unique().tolist()
        sample = random.Random(42).sample(missing_ids, min(args.probe_missing, len(missing_ids)))
        logger.info("Probing %d missing youtube_ids for live availability (oEmbed, no download)...", len(sample))
        alive = 0
        for i, yt_id in enumerate(sample, 1):
            if _probe_youtube_alive(yt_id):
                alive += 1
            time.sleep(args.probe_sleep)
            if i % 50 == 0:
                logger.info("  probed %d/%d (alive so far: %d)", i, len(sample), alive)
        pct_alive = 100 * alive / len(sample)
        logger.info("-" * 70)
        logger.info(
            "RESULT: %d/%d (%.1f%%) of missing youtube_ids are still live on YouTube.",
            alive,
            len(sample),
            pct_alive,
        )
        est_recoverable = int(n_missing * pct_alive / 100)
        logger.info(
            "Extrapolated: ~%d of the %d missing real tracks are realistically "
            "recoverable via re-download; the rest are permanently gone "
            "(video removed / private / region-blocked) — a known limitation of "
            "any YouTube-sourced dataset like SONICS.",
            est_recoverable,
            n_missing,
        )
        logger.info(
            "If this number is worth pursuing, run:\n"
            "  python scripts/download_missing_sonics_reals.py --max-tracks %d",
            est_recoverable,
        )


if __name__ == "__main__":
    main()
