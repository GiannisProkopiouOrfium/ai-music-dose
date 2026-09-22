"""Per-track calibration of the prior-restricted comb score against decoy priors.

What this computes, and why the calibration is the whole point
-------------------------------------------------------------
``comb_priormax<M>_strength`` asks whether a track's peak residual is periodic at
a spacing some published transposed-convolution decoder could produce. It is an
**absolute** autocorrelation value, so it inherits the track's own residual
statistics — how smooth the residual is, how much peak structure it carries. That
nuisance is not hypothetical: on SONICS the Udio families' residual is flatter
than real music's in every measure, and their raw ``comb_strength`` (0.055) sits
BELOW real music's (0.072). A raw score therefore ranks Udio as *more real than
real music* — an inversion, which is worse than a miss, because it mis-orders.

The 24 decoy prior sets (``decoy_fundamentals``) were introduced as a corpus-level
null: same cardinality, deliberately wrong frequencies. But they are also a
**per-track** null, and a much better one than a permutation surrogate, because
they hold the residual FIXED and vary only the hypothesis. So for each track we
can ask the calibrated question:

    is this residual more periodic at a PLAUSIBLE decoder fundamental
    than at implausible ones?

Real music: no. Udio, whose comb lies above the band: no. Suno and the HiFi-GAN
family: emphatically yes. The track's own smoothness cancels, because it enters
the real and the decoy scores alike.

The variants, and their measured behaviour
------------------------------------------
Full corpus, signed macro AUC, ``inv`` = families inverted out of five:

| variant | definition | FMC | SONICS | inv |
|---|---|---|---|---|
| raw | ``s`` | **0.9468** | 0.6661 | 0 / **2** |
| z | ``(s - mean D) / sd D`` | 0.9241 | 0.7762 | 0 / 2 |
| pval | ``mean(D < s)`` | 0.9040 | 0.7682 | 0 / 2 |
| **margin** | ``s - max D`` | 0.9153 | **0.8313** | **0 / 0** |

(``comb_priormax4``; the incumbents are 0.9177 on FMC and 0.7546 two-sided on
SONICS.) ``margin`` is the only variant sign-consistent on all ten generator
families, which is the criterion the paper adopts from Zhou and Wang. It costs
0.002 on FakeMusicCaps — inside the incumbent's CI [0.9128, 0.9227] — and gains
0.077 on SONICS.

⚠ **Provenance discipline.** The M in ``comb_priormax<M>`` was pre-registered to be
selected on FakeMusicCaps (ledger R9). The *calibration variant* was NOT: it was
devised after SONICS fell short of the incumbent, so it is EXPLORATORY and the
ledger records it as such. ``--variant`` therefore defaults to nothing; the caller
states which one they are using and why.

Usage (EC2)
-----------
    python scripts/derive_prior_margin.py \\
        --score-csv reports/diagnostics/comb_sonics_HARM2/comb_per_track_*.csv \\
        --out-csv  reports/diagnostics/comb_sonics_HARM2/per_track_calibrated.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("prior_margin")

REAL_COL = re.compile(r"comb_(harm\d+_prior|priormax\d+)_strength")


def calibrated_variants(real: np.ndarray, decoys: np.ndarray) -> dict[str, np.ndarray]:
    """The four ways to express a score relative to its own decoy null.

    ``decoys`` is ``[n_tracks, n_decoys]``. Every variant is computed per track and
    uses no label, no corpus statistic and no fitted parameter.

    ``margin`` uses the MAXIMUM decoy rather than a central value deliberately: the
    question is whether the plausible prior beats *every* implausible one on this
    residual, which is a statement about evidence rather than about typicality. A
    quantile alternative is reported by ``--quantiles`` so the choice of ``max`` can
    be seen to be robust rather than assumed.
    """
    with np.errstate(invalid="ignore"):
        return {
            "raw": real,
            "z": (real - np.nanmean(decoys, axis=1)) / (np.nanstd(decoys, axis=1) + 1e-12),
            "margin": real - np.nanmax(decoys, axis=1),
            "pval": np.nanmean(decoys < real[:, None], axis=1),
        }


def _auc_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    is_real = (df["label"] == "real").to_numpy()
    gens = sorted(set(df.loc[~is_real, "algorithm"].dropna().astype(str)) - {""})
    rows = []
    for col in cols:
        s = df[col].to_numpy(float)
        per = {}
        for g in gens:
            sel = is_real | (df["algorithm"].astype(str) == g).to_numpy()
            ok = sel & np.isfinite(s)
            y = (~is_real)[ok].astype(float)
            if len(set(y)) < 2:
                continue
            per[g], _ = auc_and_eer(y, s[ok])
        if not per:
            continue
        rows.append(
            {
                "score": col,
                **{g: round(float(v), 4) for g, v in per.items()},
                "MACRO": round(float(np.mean(list(per.values()))), 4),
                # The column that decides admissibility: a score inverted on ANY
                # family needs a per-generator sign, and choosing one needs labels.
                "n_inverted": int(sum(v < 0.5 for v in per.values())),
            }
        )
    if not rows:
        # An ALL-REAL corpus (the C10 reconstruction control, FMA) has no AUC by
        # construction. That is a legitimate input -- the calibrated columns are
        # exactly what the paired C10 test consumes -- so return empty rather than
        # raising on a missing MACRO column.
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("MACRO", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=[0.75, 0.90],
        help="Also emit margins against these decoy quantiles, so the choice of max is testable.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.score_csv, low_memory=False)
    for req in ("label", "algorithm"):
        if req not in df.columns:
            raise SystemExit(f"{args.score_csv} has no {req!r} column")
    df["algorithm"] = df["algorithm"].fillna("").astype(str)

    added: list[str] = []
    for real_col in [c for c in df.columns if REAL_COL.fullmatch(c)]:
        stem = real_col.replace("_prior_strength", "").replace("_strength", "")
        dec_cols = sorted(c for c in df.columns if re.fullmatch(re.escape(stem) + r"_decoy\d+_strength", c))
        if len(dec_cols) < 8:
            continue
        real = pd.to_numeric(df[real_col], errors="coerce").to_numpy(float)
        decoys = df[dec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        for name, vals in calibrated_variants(real, decoys).items():
            if name == "raw":
                continue
            col = f"{stem}_{name}"
            df[col] = vals
            added.append(col)
        for q in args.quantiles:
            col = f"{stem}_marginq{int(round(q * 100))}"
            df[col] = real - np.nanquantile(decoys, q, axis=1)
            added.append(col)
        logger.info("%s: %d decoys -> %d calibrated columns", stem, len(dec_cols), len(added))

    if not added:
        raise SystemExit(
            "no prior/decoy column pairs found. This CSV must come from a run with "
            "--null-priors > 0; without the decoys there is no per-track null to "
            "calibrate against."
        )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logger.info("wrote %s (%d rows, %d new columns)", args.out_csv, len(df), len(added))

    base = [c for c in df.columns if REAL_COL.fullmatch(c)]
    table = _auc_table(df, base + added)
    if table.empty:
        logger.info(
            "ALL-REAL CORPUS (%d rows): no AUC is defined, so no table is written. "
            "The calibrated columns above are what scripts/analyze_recon_pairs.py "
            "consumes for the paired C10 test.",
            len(df),
        )
        return
    logger.info(
        "\nCALIBRATED PRIOR SCORES — signed macro AUC.\n"
        "n_inverted is the admissibility column: a training-free score inverted on\n"
        "ANY family is not usable zero-shot, whatever its macro.\n%s",
        table.to_string(index=False),
    )
    out_auc = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_auc.csv")
    table.to_csv(out_auc, index=False)
    logger.info("wrote %s", out_auc)


if __name__ == "__main__":
    main()
