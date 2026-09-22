"""Per-track calibration against the autocorrelation's OWN bulk, with no decoys.

What this computes, and why it needs no new audio pass
------------------------------------------------------
``comb_harmonic`` already returns both a peak and a floor:

    comb_harm<M>_sharpness = comb_harm<M>_strength / (floor + 1e-12)

where ``floor = median(|H|)`` over the harmonic curve (``comb_artifacts.py:493``,
:509). So the floor is recoverable from two columns that every ``--n-harm`` run has
already written, and every functional form of a bulk calibration can be derived from
an existing per-track CSV with no re-extraction -- exactly as
``derive_prior_margin.py`` derives the decoy-calibrated variants.

That floor IS a per-track null. Where the 24 decoy sets estimate "what can a WRONG
HYPOTHESIS achieve on this residual" by sampling, the floor estimates "what does a
TYPICAL LAG achieve on this residual" analytically. It needs no decoys, no seed and
no draws, and it has no provenance problem at all -- there are no frequencies to have
chosen with the answers in view.

Ratio or difference: the question this script exists to settle
---------------------------------------------------------------
``sharpness`` is the RATIO form. On a seed sweep at the FakeMusicCaps geometry
(``tests/test_null_alternatives.py``) the ratio form sits at or below chance on a
family whose residual is smoother than real music's and carries no in-band comb --
0.4483 against the difference form's 0.6631 at 200 seeds -- and the roughly 0.2 AUC
gap is stable across sweep sizes even though the absolute levels move.

The mechanism: a smoother residual raises the autocorrelation bulk, so DIVIDING by it
deflates that family more than it deflates real music, while SUBTRACTING it does not.
Subtraction is the right operation when the nuisance is additive in autocorrelation
units, which is what the decoy margin already assumes.

This matters because ledger R1 records ``comb_harm4_sharpness`` at macro 0.91986 on
FakeMusicCaps -- above the incumbent -- and notes that it fixes the sharpness
channel's sign failure on stable_audio_open (0.2370 -> 0.8475). It was never taken to
SONICS, which is where the Udio families live. The fixture predicts the ratio form is
the fragile one there; ``_nmargin`` is the form predicted to survive.

Columns written, per M
----------------------
``comb_harm<M>_floor``     the recovered per-track null.
``comb_harm<M>_nmargin``   ``strength - floor``. The difference form. The headline.
``comb_priormax<M>_xfloor``  ``comb_priormax<M>_strength - comb_harm<M>_floor``.
    APPROXIMATE, and labelled so wherever it appears. The floor is the median of the
    M-harmonic curve, whose noise is about sqrt(M) below the plain autocorrelation's,
    so this mixes two scales. It exists only to say whether an exact ``comb_acf_floor``
    column is worth a corpus pass. **Do not quote a number from it.**

``raw`` is deliberately not re-emitted: it is the pre-existing
``comb_harm<M>_strength``, and duplicating it under a second name is how a column
acquires two values.

Usage (EC2)
-----------
    python scripts/derive_analytic_null.py \\
        --score-csv reports/diagnostics/comb_sonics_HARM2/comb_per_track_*.csv \\
        --out-csv  reports/diagnostics/comb_sonics_HARM2/analytic_null.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse rather than reimplement: _auc_table is what produced the published
# per_track_calibrated_auc.csv, including the n_inverted admissibility column and the
# all-real-corpus guard added in ledger R17. A second implementation would be a
# second thing to keep in step.
from derive_prior_margin import _auc_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("analytic_null")

STRENGTH = re.compile(r"comb_harm(\d+)_strength")
PRIORMAX = re.compile(r"comb_priormax(\d+)_strength")
LATMAX = re.compile(r"comb_priormax(\d+)(_pm1)?_latmax_strength")
UNIONMAX = re.compile(r"comb_priormax(\d+)_unionmax_strength")
HARMNULL = re.compile(r"comb_priormax(\d+)_hmax_strength")
APPROXIMATE_SUFFIX = "_xfloor"


def recover_floor(strength: np.ndarray, sharpness: np.ndarray) -> np.ndarray:
    """Invert ``sharpness = strength / (floor + 1e-12)``.

    The ``1e-12`` is the guard in ``comb_harmonic`` and is far below the resolution of
    anything downstream, so it is not subtracted back off. A zero or non-finite
    sharpness leaves the floor undefined rather than infinite -- a track whose
    harmonic curve is degenerate has no null to calibrate against, and NaN is the
    honest value for that.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = strength / sharpness
    return np.where(np.isfinite(floor), floor, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.score_csv, low_memory=False)
    for req in ("label", "algorithm"):
        if req not in df.columns:
            raise SystemExit(f"{args.score_csv} has no {req!r} column")
    df["algorithm"] = df["algorithm"].fillna("").astype(str)

    added: list[str] = []

    def num(col: str):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float)

    # --- family A: the harmonic-curve floor, recovered from sharpness --------
    for strength_col in [c for c in df.columns if STRENGTH.fullmatch(c)]:
        m = STRENGTH.fullmatch(strength_col).group(1)
        sharpness_col = f"comb_harm{m}_sharpness"
        if sharpness_col not in df.columns:
            logger.warning("%s present without %s -- skipping M=%s", strength_col, sharpness_col, m)
            continue
        strength = num(strength_col)
        floor = recover_floor(strength, num(sharpness_col))
        df[f"comb_harm{m}_floor"] = floor
        df[f"comb_harm{m}_nmargin"] = strength - floor
        added += [f"comb_harm{m}_floor", f"comb_harm{m}_nmargin"]
        logger.info("M=%s harmonic floor recovered on %d / %d tracks", m, int(np.isfinite(floor).sum()), len(df))

    # --- family B: the plain-ACF floor, the dimensionally exact analytic null -
    # comb_acf_floor is the median |acf| over the SAME band comb_strength searches,
    # so subtracting it from a plain-ACF peak mixes no scales. It only exists in runs
    # made after 2026-09-17; the cross-form below is the fallback and is approximate.
    for pm in sorted(c for c in df.columns if PRIORMAX.fullmatch(c)):
        m = PRIORMAX.fullmatch(pm).group(1)
        if "comb_acf_floor" in df.columns:
            df[f"comb_priormax{m}_afmargin"] = num(pm) - num("comb_acf_floor")
            added.append(f"comb_priormax{m}_afmargin")
        elif f"comb_harm{m}_floor" in df.columns:
            df[f"comb_priormax{m}{APPROXIMATE_SUFFIX}"] = num(pm) - df[f"comb_harm{m}_floor"].to_numpy(float)
            added.append(f"comb_priormax{m}{APPROXIMATE_SUFFIX}")

    # --- family C: the deterministic lattice null ----------------------------
    # This is the professor's proposal, moved to the fundamental axis. Unlike the 24
    # random decoy sets the lattice is DISJOINT from the prior's lag set by
    # construction, which is what lets a deep prior work: see the zero-atom
    # diagnostic below for why that matters more than it sounds.
    for lat in sorted(c for c in df.columns if LATMAX.fullmatch(c)):
        g = LATMAX.fullmatch(lat)
        m, pm1 = g.group(1), g.group(2) or ""
        real_col = f"comb_priormax{m}{pm1}_strength"
        if real_col not in df.columns:
            logger.warning("%s present without %s -- skipping", lat, real_col)
            continue
        df[f"comb_priormax{m}{pm1}_latmargin"] = num(real_col) - num(lat)
        added.append(f"comb_priormax{m}{pm1}_latmargin")
        pval_col = f"comb_priormax{m}{pm1}_latpval"
        if pval_col in df.columns:
            # Low p = strong evidence of a comb. Flip it so that, like every other
            # score here, HIGHER means more generated.
            df[f"comb_priormax{m}{pm1}_latconf"] = 1.0 - num(pval_col)
            added.append(f"comb_priormax{m}{pm1}_latconf")

    # --- family D: the union null -- the decoys' coverage, the lattice's disjointness
    for uni in sorted(c for c in df.columns if UNIONMAX.fullmatch(c)):
        m = UNIONMAX.fullmatch(uni).group(1)
        real_col = f"comb_priormax{m}_strength"
        if real_col not in df.columns:
            continue
        df[f"comb_priormax{m}_unionmargin"] = num(real_col) - num(uni)
        added.append(f"comb_priormax{m}_unionmargin")

    # --- family E: the harmonic-protected null (deterministic, comb-series disjoint)
    for hm in sorted(c for c in df.columns if HARMNULL.fullmatch(c)):
        m = HARMNULL.fullmatch(hm).group(1)
        real_col = f"comb_priormax{m}_strength"
        if real_col not in df.columns:
            continue
        df[f"comb_priormax{m}_hmargin"] = num(real_col) - num(hm)
        added.append(f"comb_priormax{m}_hmargin")

    # --- family F: the 2x2 floor x ceiling probe (ledger R27.62, R27.64).
    # Each arm subtracts its OWN prior: the lowered floor must reach both sides or
    # the comparison is rigged, which is why `lo20margin` reads `_lo20_strength`
    # rather than the published `_strength`. Emitted only where the columns exist,
    # so CSVs from before the probe was added pass through unchanged.
    for tag, real_suffix in (("lo20", "lo20_strength"), ("wide", "strength"), ("lo20wide", "lo20_strength")):
        pat = re.compile(rf"comb_priormax(\d+)_{tag}_hmax_strength")
        for col in sorted(c for c in df.columns if pat.fullmatch(c)):
            m = pat.fullmatch(col).group(1)
            real_col = f"comb_priormax{m}_{real_suffix}"
            if real_col not in df.columns:
                continue
            name = f"comb_priormax{m}_{tag}margin"
            df[name] = num(real_col) - num(col)
            added.append(name)

    # --- family G: the DECOUPLED arms (ledger R27.68). M does two unrelated jobs --
    # it sets the prior's depth AND the null's ceiling -- and the measurements pull
    # them opposite ways: R27.64 says the null wants to be wide, R27.67 says the prior
    # wants to be shallow. These cross a shallow prior against the WIDEST null present
    # in the CSV, so prior depth and null ceiling can be read independently.
    #
    # Derived here rather than emitted by comb_features on purpose: it needs no new
    # feature code, so an extraction already in flight stays valid.
    for tag, null_col, real_suffix in (
        ("xwide", "wide_hmax_strength", "strength"),
        ("xlo20wide", "lo20wide_hmax_strength", "lo20_strength"),
    ):
        avail = {
            int(re.fullmatch(rf"comb_priormax(\d+)_{null_col}", c).group(1))
            for c in df.columns
            if re.fullmatch(rf"comb_priormax(\d+)_{null_col}", c)
        }
        if not avail:
            continue
        widest = max(avail)  # the largest M has the largest ceiling, n_harm * hi_hz
        shared = f"comb_priormax{widest}_{null_col}"
        for m in sorted(
            int(re.fullmatch(r"comb_priormax(\d+)_strength", c).group(1))
            for c in df.columns
            if re.fullmatch(r"comb_priormax(\d+)_strength", c)
        ):
            real_col = f"comb_priormax{m}_{real_suffix}"
            if real_col not in df.columns:
                continue
            name = f"comb_priormax{m}_{tag}margin"
            df[name] = num(real_col) - num(shared)
            added.append(name)
        logger.info(
            "decoupled %s: prior depth varies, null held at comb_priormax%d (%s)",
            tag, widest, null_col,
        )

    if not added:
        raise SystemExit(
            "nothing to derive. This CSV needs at least one of: comb_harm<M>_strength with its "
            "comb_harm<M>_sharpness (the harmonic floor), comb_acf_floor with comb_priormax<M>_strength "
            "(the exact analytic margin), or comb_priormax<M>_latmax_strength (the deterministic "
            "lattice). eval_comb_detector.py:352 passes tuple(args.n_harm or ()), so without --n-harm "
            "the whole harmonic grid is EMPTY; comb_acf_floor and the lattice columns exist only in "
            "runs made after 2026-09-17."
        )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logger.info("wrote %s (%d rows, %d new columns)", args.out_csv, len(df), len(added))

    # --- the zero-atom diagnostic -------------------------------------------
    # A margin is real - max(null). If the null contains the prior's own winning lag
    # the margin is exactly 0, and a pile of ties at 0 destroys the recall at any
    # quantile AND makes a threshold degenerate. This is what killed
    # comb_priormax8_margin (recall 0.28 pooled) and what made the per-genre conformal
    # thresholds come back all equal to 0.0. Reported for every margin-like column so
    # the decoy and lattice nulls can be compared on it directly.
    margin_cols = [
        c
        for c in df.columns
        if c.endswith(("_margin", "_latmargin", "_unionmargin", "_hmargin", "_nmargin", "_afmargin"))
    ]
    if margin_cols:
        diag = pd.DataFrame(
            [
                {
                    "score": c,
                    "frac_exactly_zero": round(float((num(c) == 0.0).mean()), 4),
                    "frac_non_positive": round(float((num(c) <= 0.0).mean()), 4),
                    "n_finite": int(np.isfinite(num(c)).sum()),
                }
                for c in sorted(margin_cols)
            ]
        ).sort_values("frac_exactly_zero", ascending=False)
        diag.to_csv(Path(args.out_csv).with_name(Path(args.out_csv).stem + "_zeroatom.csv"), index=False)
        logger.info(
            "\nZERO-ATOM DIAGNOSTIC. A margin pinned at exactly 0 means the null contained the "
            "prior's own winning lag. Ties at 0 cap the recall at every quantile and make a "
            "threshold degenerate -- a large frac_exactly_zero explains a collapse that the macro "
            "AUC will not show:\n%s",
            diag.to_string(index=False),
        )

    table = _auc_table(df, added)
    if table.empty:
        logger.info(
            "ALL-REAL CORPUS (%d rows): no AUC is defined, so no table is written. The calibrated "
            "columns above are what the FMA transfer analysis and the C10 paired test consume.",
            len(df),
        )
        return

    logger.info(
        "\nANALYTIC-NULL SCORES -- signed macro AUC.\n"
        "n_inverted is the admissibility column: a training-free score inverted on ANY family is\n"
        "not usable zero-shot, whatever its macro.\n"
        "Columns ending %s are APPROXIMATE (they mix the M-harmonic floor with a plain-ACF peak)\n"
        "and are a diagnostic only -- do not quote them.\n%s",
        APPROXIMATE_SUFFIX,
        table.to_string(index=False),
    )
    out_auc = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_auc.csv")
    table.to_csv(out_auc, index=False)
    logger.info("wrote %s", out_auc)


if __name__ == "__main__":
    main()
