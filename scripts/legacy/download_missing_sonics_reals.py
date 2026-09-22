"""Best-effort downloader for SONICS real tracks missing from local cache.

Run scripts/check_sonics_real_coverage.py FIRST to see how big the gap is and
what fraction of missing youtube_ids are actually still alive before
committing to this — many will be permanently dead (YouTube-sourced datasets
decay over time), so this is opt-in / exploratory, not guaranteed to reach
the full 48,090-track official corpus.

Requires yt-dlp (``pip install yt-dlp``). Downloads audio-only, rate-limited
and resumable (skips youtube_ids already on disk or already marked dead in
the failure manifest from a previous run).

Usage
-----
pip install yt-dlp
python scripts/download_missing_sonics_reals.py --max-tracks 5000 --sleep 1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")
REAL_SONGS_DIR = SONICS_DIR / "real_songs"
METADATA_DIR = SONICS_DIR / "metadata"
FAILURE_MANIFEST = SONICS_DIR / "download_failures.jsonl"


def _load_dead_ids() -> set[str]:
    dead: set[str] = set()
    if FAILURE_MANIFEST.exists():
        with open(FAILURE_MANIFEST) as fh:
            for line in fh:
                try:
                    dead.add(json.loads(line)["youtube_id"])
                except Exception:
                    pass
    return dead


def _record_failure(youtube_id: str, reason: str) -> None:
    FAILURE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_MANIFEST, "a") as fh:
        fh.write(json.dumps({"youtube_id": youtube_id, "reason": reason[:300]}) + "\n")


# Only these patterns mean the video is GENUINELY, PERMANENTLY gone. Anything
# else (bot-check blocks, HTTP 429/network errors, transient timeouts) must
# NOT be written to the permanent dead-list — a systemic block (e.g. YouTube
# rate-limiting/blocking this EC2 IP range, or an outdated yt-dlp) would
# otherwise cause EVERY live video to be misclassified as dead on the first
# run, permanently poisoning download_failures.jsonl and making future runs
# skip videos that were never actually unavailable.
_PERMANENT_FAILURE_PATTERNS = (
    "video unavailable",
    "private video",
    "this video has been removed",
    "video is no longer available",
    "account associated with this video has been terminated",
    "copyright grounds",
    "blocked it in your country",
    "members-only content",
)


def _download_one(
    youtube_id: str,
    timeout: float,
    extractor_args: str | None = None,
    cookies_path: str | None = None,
) -> tuple[bool, str]:
    """Download best-quality audio for one YouTube video via yt-dlp.

    Returns (success, stderr_tail) — stderr_tail is always returned (even on
    success = False) so the caller can inspect it without re-reading the file.
    """
    out_tmpl = str(REAL_SONGS_DIR / f"{youtube_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        # "bestaudio" alone fails with "Requested format is not available" on
        # videos where YouTube doesn't expose an audio-only stream to this
        # client/cookie combination (common after player_client workarounds or
        # certain age/region-gated videos) — fall back to best combined stream.
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "2",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-o",
        out_tmpl,
    ]
    if extractor_args:
        cmd += ["--extractor-args", extractor_args]
    if cookies_path:
        cmd += ["--cookies", cookies_path]
    cmd.append(f"https://www.youtube.com/watch?v={youtube_id}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0 and list(REAL_SONGS_DIR.glob(f"{youtube_id}.*")):
            return True, ""
        stderr_tail = result.stderr.strip()[-300:] if result.stderr else "unknown yt-dlp failure (no stderr)"
        is_permanent = any(p in stderr_tail.lower() for p in _PERMANENT_FAILURE_PATTERNS)
        if is_permanent:
            _record_failure(youtube_id, stderr_tail)
        return False, stderr_tail
    except subprocess.TimeoutExpired:
        return False, "timeout (transient — not recorded as dead)"
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install it with: pip install yt-dlp")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-tracks", type=int, default=2000, help="Max NEW tracks to attempt this run.")
    parser.add_argument(
        "--sleep", type=float, default=1.5, help="Seconds between download attempts (be polite / avoid rate limits)."
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-track download timeout (seconds).")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--extractor-args",
        default=None,
        help="Passed through to yt-dlp --extractor-args, e.g. 'youtube:player_client=android,web' "
        "(common workaround for YouTube bot-checks on datacenter IPs).",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        dest="cookies_path",
        help="Path to a cookies.txt (Netscape format) exported from a logged-in browser session, "
        "passed through to yt-dlp --cookies. Often required when YouTube blocks the server's IP.",
    )
    args = parser.parse_args()

    real_csv = METADATA_DIR / "real_songs.csv"
    if not real_csv.exists():
        logger.error("%s not found. Run scripts/download_sonics.py --download-fakes first.", real_csv)
        return
    REAL_SONGS_DIR.mkdir(parents=True, exist_ok=True)

    df_real = pd.read_csv(real_csv, low_memory=False)
    youtube_ids = df_real["youtube_id"].dropna().astype(str).unique().tolist()

    existing = {f.stem for f in REAL_SONGS_DIR.iterdir() if f.suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}}
    dead = _load_dead_ids()
    todo = [yt for yt in youtube_ids if yt not in existing and yt not in dead]
    random.Random(args.shuffle_seed).shuffle(todo)
    todo = todo[: args.max_tracks]

    logger.info(
        "Total metadata: %d | already local: %d | known-dead: %d | attempting this run: %d",
        len(youtube_ids),
        len(existing),
        len(dead),
        len(todo),
    )
    if not todo:
        logger.info("Nothing to do — either fully downloaded or --max-tracks exhausted the dead-id-filtered pool.")
        return

    downloaded = 0
    failed = 0
    t_start = time.time()
    CIRCUIT_BREAKER_N = 20
    for i, yt_id in enumerate(todo, 1):
        ok, reason = _download_one(
            yt_id,
            timeout=args.timeout,
            extractor_args=args.extractor_args,
            cookies_path=args.cookies_path,
        )
        if ok:
            downloaded += 1
        else:
            failed += 1
            if i <= 5:
                # Print the actual yt-dlp error immediately so a systemic
                # problem (outdated yt-dlp, IP blocked/bot-check, etc.) is
                # visible right away instead of only after burning through
                # the whole queue.
                logger.info("  [%d] FAILED %s: %s", i, yt_id, reason)

        # Circuit breaker: if almost everything is failing early on, this is
        # almost certainly a systemic issue (outdated yt-dlp / YouTube
        # blocking this IP range for bot-detection — very common for AWS/GCP/
        # Azure datacenter IPs), not 20 individually-dead videos. Stop before
        # wasting hours grinding through 35k IDs for 0% yield.
        if i == CIRCUIT_BREAKER_N and downloaded == 0:
            logger.error(
                "CIRCUIT BREAKER: 0/%d succeeded. This is almost certainly a systemic "
                "issue, not dead videos (see error samples printed above). Common fixes:\n"
                "  1. pip install -U yt-dlp   (YouTube changes break old yt-dlp versions constantly)\n"
                "  2. Try --extractor-args 'youtube:player_client=android,web' (bypasses some bot-checks)\n"
                "  3. If the error mentions 'Sign in to confirm you're not a bot': YouTube is "
                "blocking this EC2 IP range. Export cookies.txt from a logged-in browser session "
                "on your own machine, upload it, and pass --cookies cookies.txt.\n"
                "Aborting now — no more IDs will be (mis)recorded as dead.",
                CIRCUIT_BREAKER_N,
            )
            return
        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - t_start
            logger.info(
                "[%d/%d] downloaded=%d failed=%d  (%.1f%% success)  elapsed=%.0fs",
                i,
                len(todo),
                downloaded,
                failed,
                100 * downloaded / i,
                elapsed,
            )
        time.sleep(args.sleep)

    logger.info("=" * 70)
    logger.info("Done. Downloaded %d new real tracks, %d failed (see %s)", downloaded, failed, FAILURE_MANIFEST)
    logger.info("Next steps:")
    logger.info("  1. python scripts/download_sonics.py --convert-m4a --prepare")
    logger.info("  2. Re-run build_canonical_corpus.py to add the new tracks to the canonical manifest")
    logger.info("  3. Re-extract EnCodec embeddings for the new tracks only (existing cache is untouched)")
    logger.info(
        "  4. Re-run run_balanced_ablation.py --window-flow-eval to retrain the headline flow on the larger real corpus"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
