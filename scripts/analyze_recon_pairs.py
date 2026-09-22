"""Gate C10, done as a PAIRED test — the only form in which it means anything.

Why this script exists
----------------------
``build_reconstruction_control.py`` exports content-identical pairs: the same real
track, once as itself and once re-synthesised by a decoder. Every one of them is
real music, so a corpus-level AUC is undefined — ``eval_comb_detector.py`` says so
and prints score quantiles instead:

    ALL-REAL CORPUS (1800 tracks): no AUC is defined.

The question C10 asks is not "can the detector separate two classes" but **"does
the detector fire HIGHER on the reconstruction than on its own source?"** — which
is a *within-pair* comparison, with content, genre, key, tempo, loudness and
everything else held constant by construction. That is what makes C10 the cleanest
control in the project, and it is why the number has to be computed pairwise.

What it reports, per variant
----------------------------
``frac_above``   share of pairs where the variant scores above its own source. A
                 detector indifferent to the transformation sits at 0.50.
``median_delta`` median of (variant − source) across pairs, in score units.
``wilcoxon_p``   two-sided signed-rank test against "no shift". Paired, so it does
                 not assume the score distribution is anything in particular.

The FakeMusicCaps reference this reproduces (300 pairs per variant):

    encodec_{3,24,6}kbps   96.7-98.0% of pairs, median +0.113 … +0.153
    griffinlim_{256,128}   47.3-50.0% of pairs, median -0.0005 … +0.0000

EnCodec is a transposed-convolution decoder; Griffin-Lim is phase reconstruction
with no deconvolution at all. The detector responds to **deconvolution**, not to
re-synthesis in general — and Griffin-Lim landing on exactly chance is the negative
control that makes the positive one worth anything.

Usage
-----
    python scripts/analyze_recon_pairs.py \\
        --score-csv reports/diagnostics/comb_recon_control_sonics/comb_per_track_lvl_sm5_median_db_f1000_noclip.csv \\
        --score-column comb_strength \\
        --out-dir reports/diagnostics/c10_sonics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("recon_pairs")

# The variant name the exporter gives the untouched track. Everything else is a
# reconstruction and is compared against it within its own pair_id.
SOURCE_VARIANTS = ("original", "source", "real", "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--score-column", default="comb_strength")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--pair-column",
        default="pair_id",
        help="Column identifying content-identical groups.",
    )
    parser.add_argument("--variant-column", default="variant")
    parser.add_argument(
        "--source-variant",
        default=None,
        help="Name of the untouched variant. Auto-detected from a short known list "
        "if omitted; the script fails loudly rather than guessing wrong.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.score_csv, low_memory=False)
    for col in (args.score_column, args.pair_column, args.variant_column):
        if col not in df.columns:
            raise SystemExit(f"{args.score_csv} has no column {col!r}.\n" f"available: {sorted(df.columns)}")

    variants = sorted(df[args.variant_column].dropna().unique())
    source = args.source_variant
    if source is None:
        found = [v for v in variants if str(v).lower() in SOURCE_VARIANTS]
        if len(found) != 1:
            raise SystemExit(
                f"cannot identify the untouched variant among {variants}. "
                f"Pass --source-variant explicitly — guessing it would silently "
                f"compare two reconstructions against each other."
            )
        source = found[0]
    logger.info("source variant: %r; reconstructions: %s", source, [v for v in variants if v != source])

    df = df[np.isfinite(df[args.score_column])]
    src = df[df[args.variant_column] == source].set_index(args.pair_column)[args.score_column]
    if src.empty:
        raise SystemExit(f"no rows with {args.variant_column} == {source!r}")

    rows = []
    for variant in variants:
        if variant == source:
            continue
        rec = df[df[args.variant_column] == variant].set_index(args.pair_column)[args.score_column]
        common = src.index.intersection(rec.index)
        if len(common) < 30:
            logger.warning("%s: only %d paired tracks — skipping", variant, len(common))
            continue
        delta = (rec.loc[common] - src.loc[common]).to_numpy()
        try:
            from scipy.stats import wilcoxon

            p = float(wilcoxon(delta, zero_method="zsplit").pvalue)
        except Exception:  # noqa: BLE001  - scipy absent or all-zero deltas
            p = float("nan")
        rows.append(
            {
                "variant": variant,
                "n_pairs": int(len(common)),
                "frac_above": float((delta > 0).mean()),
                "median_delta": float(np.median(delta)),
                "mean_delta": float(np.mean(delta)),
                "wilcoxon_p": p,
            }
        )

    if not rows:
        raise SystemExit("no variant had enough paired tracks to compare")

    res = pd.DataFrame(rows).sort_values("frac_above", ascending=False)
    path = out_dir / "recon_pairs.csv"
    res.to_csv(path, index=False)

    logger.info(
        "\nC10 — PAIRED within content-identical pairs (source = %r):\n%s",
        source,
        res.to_string(index=False),
    )
    logger.info(
        "\nRead it as: a detector indifferent to the transformation sits at "
        "frac_above = 0.50 with median_delta = 0. A DECONVOLUTION decoder (EnCodec) "
        "should sit far above; phase-only reconstruction (Griffin-Lim) should sit at "
        "chance. If both fire, the score is responding to re-synthesis in general and "
        "C10 does NOT support a deconvolution-specific reading — say so."
    )
    logger.info("Saved → %s", path)


if __name__ == "__main__":
    main()
