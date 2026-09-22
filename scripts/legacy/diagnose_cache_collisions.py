"""CRITICAL data-integrity check: do distinct tracks share one embedding-cache entry?

The embedding cache key used by `run_balanced_ablation.py`, `coverage_curve.py`,
`score_mixture_of_flows.py` and friends is

    md5(f"{track_id}_{target_sr}_{max_duration}_{preprocess_mode}{tags}")

which contains **no label and no algorithm**. That is safe for SONICS (globally
unique track ids) but NOT for corpora that reuse a source id across variants —
FakeMusicCaps re-generates every MusicCaps caption with 5 TTM systems, so the
real clip and all 5 fakes can share one YouTube id. If they do, all six map to
ONE cache file: whichever variant was extracted first silently supplies the
embeddings for the other five, and every downstream number for that corpus is
an artifact (it would also explain per-generator AUCs that come out
bit-identical).

This script answers, for any manifest + cache directory:
  1. How many track_ids appear more than once, and in which label/algorithm mix?
  2. How many distinct manifest rows collapse onto the same cache path?
  3. Do the underlying AUDIO files actually differ for colliding rows
     (md5 of the canonical file) — i.e. is real information being lost?
  4. Which variant physically owns each cache entry (best-effort, by comparing
     a freshly-extracted embedding against the cached one) — optional, needs GPU.

Exit code is 1 when a collision that loses information is detected, so it can
gate a pipeline.

Usage:
    python scripts/diagnose_cache_collisions.py \\
        --manifest data/processed/canonical_fmc_native/combined_manifest.csv \\
        --emb-cache data/emb_cache_encodec_fmc_native \\
        --max-duration 10.0
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _cache_key(track_id: str, sr: int, max_duration: float, mode: str) -> str:
    return hashlib.md5(f"{track_id}_{sr}_{max_duration}_{mode}".encode()).hexdigest()


def _file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--emb-cache", required=True, help="cache root (containing encodec/ etc.)")
    parser.add_argument("--embedding", default="encodec")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument(
        "--max-duration", type=float, required=True, help="MUST match the --analysis-duration the cache was built with"
    )
    parser.add_argument("--preprocess-mode", default="preprocessed")
    parser.add_argument(
        "--audit-audio",
        type=int,
        default=200,
        help="Hash the canonical audio of up to N colliding groups to prove information loss",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df["track_id"] = df["track_id"].astype(str)
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("").astype(str)
    logger.info("manifest rows: %d | unique track_id: %d", len(df), df["track_id"].nunique())

    # --- 1. duplicate track_ids ---
    counts = df["track_id"].value_counts()
    dup_ids = counts[counts > 1]
    if dup_ids.empty:
        logger.info("PASS: every track_id is unique — no cache-key collision possible.")
        sys.exit(0)

    logger.error(
        "COLLISION DETECTED: %d track_ids appear more than once (%d manifest rows affected, "
        "max %d rows sharing one id).",
        len(dup_ids),
        int(dup_ids.sum()),
        int(dup_ids.max()),
    )
    dup_df = df[df["track_id"].isin(dup_ids.index)]
    mix = dup_df.groupby(["label", "algorithm"]).size().sort_values(ascending=False)
    logger.error("Variants sharing ids, by (label, algorithm):\n%s", mix.to_string())

    # --- 2. cache paths ---
    cache_dir = Path(args.emb_cache) / args.embedding
    df["cache_key"] = [_cache_key(t, args.target_sr, args.max_duration, args.preprocess_mode) for t in df["track_id"]]
    df["cache_path"] = [str(cache_dir / f"{k}.npy") for k in df["cache_key"]]
    n_paths = df["cache_path"].nunique()
    n_exist = sum(Path(p).exists() for p in df["cache_path"].unique())
    logger.error(
        "%d manifest rows map to only %d distinct cache paths (%d present on disk) — "
        "each present file is being read by ~%.1f different tracks.",
        len(df),
        n_paths,
        n_exist,
        len(df) / max(n_paths, 1),
    )

    # --- 3. does the underlying audio actually differ? ---
    if args.audit_audio > 0 and "canonical_path" in df.columns:
        groups = list(dup_df.groupby("track_id"))[: args.audit_audio]
        n_checked = n_differing = 0
        for tid, g in groups:
            hashes = {_file_md5(p) for p in g["canonical_path"].astype(str)}
            hashes.discard("MISSING")
            if len(hashes) > 1:
                n_differing += 1
            if hashes:
                n_checked += 1
        logger.error(
            "AUDIO AUDIT: %d/%d inspected colliding groups contain GENUINELY DIFFERENT audio — "
            "for those, the cache returns one variant's embeddings for all of them, so every "
            "downstream score for the other variants is WRONG.",
            n_differing,
            n_checked,
        )

    if args.output_csv:
        out = df[df["track_id"].isin(dup_ids.index)][
            ["track_id", "label", "algorithm", "cache_path"] + (["canonical_path"] if "canonical_path" in df else [])
        ]
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        logger.info("Collision detail written to %s", args.output_csv)

    logger.error(
        "\nREQUIRED FIX: make track_id unique per variant when the corpus is built "
        "(e.g. '<algorithm>__<ytid>' for fakes, 'real__<ytid>' for reals) in "
        "build_generic_corpus.py, delete the affected embedding cache, and re-extract. "
        "Until then, treat ALL results computed from this manifest/cache as INVALID."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
