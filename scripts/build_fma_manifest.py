"""Build an FMA manifest so the false-positive experiment can run.

FMA is genuine human music from a catalogue neither corpus draws on, so the rate
at which a detector flags it is the deployment-relevant number: how often does
real music get called AI? Every earlier attempt at this in the project is
INVALID — `data/processed/fma_fpr_ctrl/` and `fma_fpr_lr/` hold byte-identical
score files produced by the silent-retrain bug (report §8.2), not two different
detectors.

The audio is on the box at `data/external/fma_small` (and `fma_medium`), laid out
as `<3-digit dir>/<track_id>.mp3`. This writes a manifest in the same schema the
rest of the pipeline expects, so it can be fed straight to
`build_channel_matched_corpus.py` and put through the SAME delivery chain as the
corpus it is compared against — otherwise the false-positive rate measures the
chain rather than the detector.

Every row is labelled ``real``: there are no fakes in FMA, and that is the point.

Usage (EC2)
-----------
    python scripts/build_fma_manifest.py \
        --audio-dir data/external/fma_small \
        --out data/processed/fma_manifest.csv --limit 3000

    python scripts/build_channel_matched_corpus.py \
        --manifest data/processed/fma_manifest.csv --src-column src_path \
        --out-dir data/processed/canonical_fma_raw16 \
        --chain-sr 16000 --chain-kbps 0 --target-sr 16000 \
        --max-duration 10 --workers 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("build_fma_manifest")

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifest for an FMA directory tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tracks.")
    parser.add_argument(
        "--min-bytes", type=int, default=10_000, help="Skip truncated files — FMA ships a handful of zero-length mp3s."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.audio_dir)
    if not root.exists():
        raise SystemExit(f"--audio-dir not found: {root}")

    rows: list[dict] = []
    skipped = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        try:
            if path.stat().st_size < args.min_bytes:
                skipped += 1
                continue
        except OSError:
            skipped += 1
            continue
        rows.append(
            {
                # Prefix the id: FMA numbers collide with nothing in our corpora
                # today, but a bare integer id is exactly the kind of thing that
                # collides silently in a shared cache (report §8.4).
                "track_id": f"fma_{path.stem}",
                "src_path": str(path),
                "label": "real",
                "algorithm": "",
                "status": "ok",
            }
        )

    if not rows:
        raise SystemExit(f"no audio found under {root}")
    logger.info("found %d tracks (%d skipped as truncated)", len(rows), skipped)

    df = pd.DataFrame(rows)
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=args.seed).sort_values("track_id")
        logger.info("sampled down to %d", len(df))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("manifest -> %s (%d rows, all label=real)", out, len(df))
    logger.info(
        "NEXT: put it through the SAME chain as the corpus you compare against, or the "
        "false-positive rate measures the delivery chain instead of the detector."
    )


if __name__ == "__main__":
    main()
