"""Make ``track_id`` unique in a manifest, so the embedding cache cannot collide.

Why this exists
---------------
The embedding-cache key is

    md5(f"{track_id}_{target_sr}_{max_duration}_{preprocess_mode}{tags}")

with **no label and no algorithm** in it. Any corpus that reuses one source id
across variants therefore maps several distinct audio files onto a single cache
entry: whichever variant is extracted first silently supplies the embeddings for
all the others, and every downstream number for that corpus is an artifact.
``diagnose_cache_collisions.py`` detects this; this script repairs it.

It is a **manifest-only** rewrite. The canonical audio is untouched, so there is
no re-canonicalisation cost — only re-extraction of embeddings under the new
(now unique) keys. Run the diagnose script again afterwards: it must exit 0.

What it does
------------
1. Drops exact duplicate rows that point at the same ``canonical_path`` *and*
   carry the same label/algorithm — those are genuine duplicates, not
   collisions, and renaming them would invent data.
2. Rewrites ``track_id`` to ``<corpus>_<label>_<algorithm>_<original>``, using
   whichever of those columns exist, and appends a numeric suffix if that is
   still not unique. The original is preserved in ``source_track_id`` so any
   join back to the upstream dataset still works.
3. Refuses to write unless the result is fully unique AND one-to-one with
   ``canonical_path``.

Usage:
    python scripts/repair_manifest_track_ids.py \\
        --manifest data/processed/canonical_sonics_musiccaps_combined/combined_manifest.csv \\
        --output   data/processed/canonical_sonics_musiccaps_combined/combined_manifest_uniq.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^0-9A-Za-z]+")


def _slug(v: object) -> str:
    s = _SLUG.sub("-", str(v or "").strip()).strip("-").lower()
    return s


def repair(df: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return (repaired_df, stats). Pure function so it is unit-testable."""
    if "track_id" not in df.columns:
        raise SystemExit("manifest has no 'track_id' column")
    out = df.copy()
    out["track_id"] = out["track_id"].astype(str)
    stats = {"rows_in": len(out), "dupe_rows_dropped": 0}

    # 1. genuine duplicate rows (same audio, same provenance) -> keep one
    if "canonical_path" in out.columns:
        dedupe_on = ["canonical_path"] + [c for c in key_cols if c in out.columns]
        before = len(out)
        out = out.drop_duplicates(subset=dedupe_on, keep="first").reset_index(drop=True)
        stats["dupe_rows_dropped"] = before - len(out)

    stats["colliding_ids_before"] = int((out["track_id"].value_counts() > 1).sum())

    # 2. qualify the id with whatever provenance columns the manifest carries
    present = [c for c in key_cols if c in out.columns]
    if present:
        prefix = out[present].apply(lambda r: "_".join(p for p in (_slug(v) for v in r) if p), axis=1)
        new_id = prefix.str.cat(out["track_id"], sep="_").str.strip("_")
    else:
        new_id = out["track_id"].copy()

    # 3. last-resort numeric suffix for anything still duplicated
    dup = new_id.duplicated(keep=False)
    if dup.any():
        new_id = new_id.where(~dup, new_id + "_" + new_id.groupby(new_id).cumcount().astype(str))
        logger.warning("%d ids needed a numeric suffix after qualification", int(dup.sum()))

    out["source_track_id"] = out["track_id"]
    out["track_id"] = new_id
    stats["colliding_ids_after"] = int((out["track_id"].value_counts() > 1).sum())
    stats["rows_out"] = len(out)
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--key-cols",
        default="corpus,source,label,algorithm",
        help="comma-separated provenance columns used to qualify the id (missing ones are skipped)",
    )
    args = ap.parse_args()

    src = Path(args.manifest)
    if not src.exists():
        raise SystemExit(f"manifest not found: {src}")
    df = pd.read_csv(src, low_memory=False)
    out, stats = repair(df, [c.strip() for c in args.key_cols.split(",") if c.strip()])

    logger.info(
        "rows %d -> %d (%d exact-duplicate rows dropped); colliding ids %d -> %d",
        stats["rows_in"],
        stats["rows_out"],
        stats["dupe_rows_dropped"],
        stats["colliding_ids_before"],
        stats["colliding_ids_after"],
    )
    if stats["colliding_ids_after"]:
        logger.error("REFUSING TO WRITE: %d ids are still not unique", stats["colliding_ids_after"])
        return 1
    if "canonical_path" in out.columns:
        n_paths = out["canonical_path"].nunique()
        if n_paths != len(out):
            logger.error(
                "REFUSING TO WRITE: %d rows map to only %d distinct canonical_path values — "
                "distinct ids would still share audio, so the manifest is malformed upstream.",
                len(out),
                n_paths,
            )
            return 1

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    logger.info("wrote %s", dst)
    logger.info(
        "NEXT: re-run diagnose_cache_collisions.py against this manifest (must exit 0), "
        "and use a FRESH --emb-cache dir: old cache entries are keyed by the old ids."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
