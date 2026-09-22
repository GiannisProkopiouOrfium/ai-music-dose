"""False-positive rate on an unseen real catalogue, at a threshold set on reals only.

The deployment-relevant number
------------------------------
"AUC on a balanced benchmark" is not what a rightsholder cares about. They care
how often genuine music gets flagged. FMA is human music from a catalogue neither
benchmark draws on, so the rate at which our detector fires on it is the honest
false-positive rate — and it has no AUC at all, because there are no fakes in it.

The threshold is set as a **quantile of the REAL tracks of the labelled corpus**,
which is Afchar & Hennequin's own decision rule (arXiv:2607.25530: "given a small
training collection of real examples, we take a quantile of a training set of
errors, for instance, 95%"). No fake label is used to choose it, so the operating
point is as zero-shot as the score.

Two rates are reported and they answer different questions:

  in-corpus FPR   fraction of the labelled corpus's own reals above threshold.
                  Should land near 1 - quantile by construction; if it does not,
                  the threshold split is broken.
  transfer FPR    fraction of the UNSEEN catalogue above the same threshold.
                  This is the number to report. If it is much higher than the
                  in-corpus rate, the detector is keyed to the benchmark's reals
                  rather than to real music, which is a corpus confound that no
                  AUC on that benchmark would reveal.

``--calibrate-per-group``: a threshold that holds per genre
-----------------------------------------------------------
The default behaviour above MEASURES how a single global threshold lands on each
genre. It does not SET one per genre, and the spread it exposes is large: the comb
arm reads 0.236 on Electronic against 0.028 on Folk, an 8.5x spread with disjoint
Wilson intervals, and the prior-restricted score still spreads 2.5x.

A reals-only quantile is a split-conformal threshold under exchangeability -- which
is exactly the property an unseen catalogue breaks. Conditioning the calibration on
genre is Mondrian (group-conditional) conformal prediction: each group gets its own
threshold from its own calibration reals, so each group gets its own nominal
false-positive rate by construction rather than by luck.

Three arms are reported side by side so the comparison isolates one variable:

  transfer          threshold from the LABELLED corpus's reals. The published
                    behaviour, and the one whose transfer ratio the paper reports.
  conformal_global  threshold from half the UNSEEN reals, pooled. Isolates "did
                    calibrating on the deployment distribution help?" from "did
                    conditioning on genre help?".
  conformal_group   threshold from half the unseen reals, PER GROUP. Mondrian.

**The assumption this adds, stated plainly.** The two conformal arms need a sample of
real music from the deployment catalogue, labelled real and tagged by group. That is
strictly more than the transfer arm needs. It is the same assumption the published
rule already makes -- Afchar and Hennequin take "a quantile of a training set of
errors" over real examples -- applied per group instead of globally, and every
deployer screening a catalogue has real music from it. No FAKE label is used
anywhere, so the score stays zero-shot; it is the THRESHOLD that becomes
catalogue-conditional.

The threshold is the finite-sample conformal quantile, the ``ceil((n+1)(1-alpha))``-th
order statistic, not ``np.quantile``. On 3,000 pooled reals the two agree to the
fourth decimal; on a genre with forty tracks they do not, and the small groups are
precisely the ones a per-group threshold exists to serve.

Usage (EC2)
-----------
    python scripts/measure_false_positive_rate.py \\
        --labelled-csv reports/diagnostics/comb_fmc_grid/comb_per_track_lvl_sm5_hull_db_f1000_noclip.csv \\
        --unseen-csv reports/diagnostics/comb_fma/comb_per_track_lvl_sm5_median_db_f1000_noclip.csv \\
        --score-column median_db__comb_strength \\
        --out-dir reports/diagnostics/fpr_fma

    # ...and the Mondrian arm
    python scripts/measure_false_positive_rate.py ... \\
        --group-column genre --calibrate-per-group
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fpr")

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The rule-of-three upper bound, imported rather than re-implemented: an observed
# ZERO false positives on a small group is not a zero rate, and treating it as one
# makes a threshold with negligible recall report precision 1.0. That trap was
# already paid for once (analyze_detector.py:58) and must not be re-introduced here,
# where the per-group samples are far smaller than the pooled one.
from analyze_detector import fpr_upper_bound  # noqa: E402

# A group smaller than 1/alpha - 1 cannot support a conformal threshold at level
# alpha at all: the required order statistic falls outside the sample. At the 0.05
# level that is 19 calibration tracks. The floor below is deliberately higher, because
# a threshold resting on the single largest of twenty scores is valid but useless.
MIN_CALIBRATION_PER_GROUP = 40


def conformal_threshold(calibration: np.ndarray, alpha: float) -> float:
    """Split-conformal upper threshold: ``P(new real > t) <= alpha`` under exchangeability.

    The ``ceil((n+1)(1-alpha))``-th order statistic, which is the finite-sample-valid
    quantity, rather than ``np.quantile``'s interpolated plug-in estimate. They
    converge as n grows; they differ where it matters, which is a small group.

    Returns ``+inf`` when the sample cannot support the level -- an honest "no
    threshold is defensible here" rather than silently returning the maximum and
    implying a guarantee the data cannot give.
    """
    x = np.asarray(calibration, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(x)[k - 1])


def precision_at_prevalence(tpr: float, fpr: float, p: float) -> float:
    """Precision implied by a (TPR, FPR) pair at population prevalence ``p``."""
    num = p * tpr
    den = num + (1.0 - p) * fpr
    return float(num / den) if den > 0 else float("nan")


def _conformal_by_group(args, out_dir: Path, uns: pd.DataFrame, labelled_thr_half: np.ndarray, fake: np.ndarray) -> None:
    """Mondrian conformal thresholds on the unseen catalogue, against two baselines.

    The unseen reals are split in half per group: one half calibrates, the other
    measures. Splitting WITHIN each group rather than globally keeps every group
    represented on both sides, which a global shuffle does not guarantee for the
    small ones -- and the small ones are the point.
    """
    alpha = args.alpha
    score_col, group_col = args.score_column, args.group_column
    n_splits = max(int(args.n_splits), 1)

    work = uns[[score_col, group_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[np.isfinite(work[score_col])]
    work[group_col] = work[group_col].fillna("(unlabelled)").astype(str)

    thr_transfer = float(np.quantile(labelled_thr_half, 1.0 - alpha))
    finite_fake = fake[np.isfinite(fake)] if len(fake) else np.array([])

    def _recall(threshold: float) -> float:
        if not len(finite_fake) or not np.isfinite(threshold):
            return float("nan")
        return float((finite_fake > threshold).mean())

    # Average over independent calibration splits. One split is a point estimate:
    # simulated at n = 191 per group with a PERFECTLY calibrated procedure on
    # exchangeable data, the max/min FPR spread across eight groups has median 4.0x
    # and reaches 8.5x at the 90th percentile purely by chance. A single-split
    # comparison of 9.6x against 9.0x therefore measures nothing.
    per_split: list[dict] = []
    for split in range(n_splits):
        rng = np.random.RandomState(args.seed + split)
        calib_mask = np.zeros(len(work), dtype=bool)
        for _, idx in work.groupby(group_col).groups.items():
            pos = work.index.get_indexer(idx)
            rng.shuffle(pos)
            calib_mask[pos[: len(pos) // 2]] = True
        calib, evaluate = work[calib_mask], work[~calib_mask]
        thr_global = conformal_threshold(calib[score_col].to_numpy(float), alpha)

        for group, sub_eval in evaluate.groupby(group_col):
            sub_calib = calib[calib[group_col] == group][score_col].to_numpy(float)
            fallback = len(sub_calib) < MIN_CALIBRATION_PER_GROUP
            thr_group = thr_global if fallback else conformal_threshold(sub_calib, alpha)
            if not np.isfinite(thr_group):
                thr_group, fallback = thr_global, True
            v = sub_eval[score_col].to_numpy(float)
            rec = {
                "group": group,
                "split": split,
                "n_calib": len(sub_calib),
                "n_eval": len(v),
                "fallback_to_global": bool(fallback),
                "thr_conformal_global": thr_global,
                "thr_conformal_group": thr_group,
                "fpr_transfer": float((v > thr_transfer).mean()),
                "fpr_conformal_global": float((v > thr_global).mean()),
                "fpr_conformal_group": float((v > thr_group).mean()),
                "recall_transfer": _recall(thr_transfer),
                "recall_conformal_group": _recall(thr_group),
            }
            for arm, threshold in (("transfer", thr_transfer), ("conformal_group", thr_group)):
                bounded = fpr_upper_bound(int((v > threshold).sum()), len(v))
                rec[f"precision@p=0.01_{arm}"] = precision_at_prevalence(rec[f"recall_{arm}"], bounded, 0.01)
            per_split.append(rec)

    raw = pd.DataFrame(per_split)
    mean_cols = [
        "n_calib", "n_eval", "thr_conformal_global", "thr_conformal_group",
        "fpr_transfer", "fpr_conformal_global", "fpr_conformal_group",
        "recall_transfer", "recall_conformal_group",
        "precision@p=0.01_transfer", "precision@p=0.01_conformal_group",
    ]
    table = raw.groupby("group")[mean_cols].mean().round(6).reset_index()
    table.insert(1, "n_splits", n_splits)
    table.insert(2, "target_fpr", alpha)
    table.insert(3, "fallback_to_global", raw.groupby("group")["fallback_to_global"].any().to_numpy())
    table.insert(4, "thr_transfer", round(thr_transfer, 6))
    # The spread of the per-group rate ACROSS splits: if it is comparable to the
    # spread ACROSS groups, the groups are not distinguishable at this sample size.
    table["fpr_conformal_group_sd"] = (
        raw.groupby("group")["fpr_conformal_group"].std().round(6).to_numpy() if n_splits > 1 else np.nan
    )
    table = table.sort_values("fpr_transfer", ascending=False)
    table.to_csv(out_dir / "conformal_by_group.csv", index=False)

    def _spread(col: str) -> str:
        """max / min across the groups big enough to estimate a rate at all.

        Reported as "max vs min" with the ratio only where the minimum is non-zero: a
        group with zero observed false positives makes the ratio infinite, which says
        less than the two endpoints do.
        """
        v = table.loc[table["n_eval"] >= 20, col]
        if len(v) < 2:
            return "n/a (fewer than two groups)"
        hi, lo = float(v.max()), float(v.min())
        ratio = f"{hi / lo:.2f}x" if lo > 0 else "ratio undefined (a group observed zero)"
        return f"{hi:.4f} vs {lo:.4f}, {ratio}"

    n_fallback = int(table["fallback_to_global"].sum())
    if n_splits == 1:
        logger.warning(
            "--n-splits 1: this is a POINT ESTIMATE. At ~190 reals per group a perfectly "
            "calibrated procedure still yields a median max/min FPR spread of 4.0x across eight "
            "groups, and 8.5x at the 90th percentile. Do not read a single-split spread as "
            "evidence for or against group-conditional calibration; pass --n-splits 20 or more."
        )
    logger.info(
        "\nGROUP-CONDITIONAL (MONDRIAN) CONFORMAL THRESHOLDS at alpha = %.3f, averaged over "
        "%d split(s).\n" % (alpha, n_splits) + "%s" % ""
        + "Each group's threshold is the ceil((n+1)(1-alpha))-th order statistic of ITS OWN "
        "calibration half, so its false-positive rate is nominal by construction rather than "
        "by luck. %d of %d groups fell back to the pooled threshold for want of %d calibration "
        "tracks, and are flagged.\n%s",
        n_fallback,
        len(table),
        MIN_CALIBRATION_PER_GROUP,
        table.to_string(index=False),
    )
    logger.info(
        "FPR SPREAD across groups with n_eval >= 20 (worst group against best):\n"
        "  transfer         %s\n"
        "  conformal_global %s\n"
        "  conformal_group  %s\n"
        "Equalising the spread is what the per-group threshold buys; whether it also buys recall "
        "is the recall_* columns, and precision uses the rule-of-three FPR bound so a group with "
        "zero observed false positives cannot report precision 1.0.",
        _spread("fpr_transfer"),
        _spread("fpr_conformal_global"),
        _spread("fpr_conformal_group"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled-csv", required=True, help="Corpus with reals AND fakes.")
    parser.add_argument("--unseen-csv", required=True, help="All-real catalogue (e.g. FMA).")
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Real-score quantiles to use as thresholds (Afchar & Hennequin use 0.95).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-column",
        default=None,
        help="Optional column in --unseen-csv to break the FPR down by (e.g. genre).",
    )
    parser.add_argument(
        "--calibrate-per-group",
        action="store_true",
        help="Also SET a per-group conformal threshold (Mondrian), not only measure the spread of "
        "a global one. Requires --group-column. Writes conformal_by_group.csv.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Average the per-group conformal result over this many independent calibration "
        "splits. At ~190 reals per genre a SINGLE split is a point estimate: a perfectly "
        "calibrated procedure still produces a median max/min FPR spread of 4.0x and 8.5x at the "
        "90th percentile, so one split cannot resolve the effect. Use >=20.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target false-positive rate for the conformal arms.",
    )
    args = parser.parse_args()
    if args.calibrate_per_group and not args.group_column:
        raise SystemExit("--calibrate-per-group needs --group-column: there are no groups to condition on")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(args.labelled_csv, low_memory=False)
    uns = pd.read_csv(args.unseen_csv, low_memory=False)
    for name, df in (("--labelled-csv", lab), ("--unseen-csv", uns)):
        if args.score_column not in df.columns:
            raise SystemExit(
                f"{name} has no column {args.score_column!r}. Available: "
                f"{sorted(c for c in df.columns if 'comb_' in c)[:25]}"
            )
    if "label" not in lab.columns:
        raise SystemExit("--labelled-csv needs a 'label' column")
    if set(uns.get("label", pd.Series(["real"]))) - {"real"}:
        raise SystemExit("--unseen-csv must be all-real; it is the false-positive catalogue")

    real = lab.loc[lab["label"] == "real", args.score_column].to_numpy(float)
    real = real[np.isfinite(real)]
    fake = lab.loc[lab["label"] != "real", args.score_column].to_numpy(float)
    unseen = uns[args.score_column].to_numpy(float)
    unseen = unseen[np.isfinite(unseen)]
    if len(real) < 100 or len(unseen) < 100:
        raise SystemExit(f"need >=100 of each; got {len(real)} labelled reals, {len(unseen)} unseen")

    # Split the labelled reals: half set the threshold, half measure the in-corpus
    # rate. Using the same reals for both makes the in-corpus FPR optimistic by
    # construction and hides a badly-calibrated threshold.
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(real))
    thr_half, eval_half = real[idx[: len(idx) // 2]], real[idx[len(idx) // 2 :]]

    rows: list[dict] = []
    for q in args.quantiles:
        thr = float(np.quantile(thr_half, q))
        row = {
            "quantile": q,
            "threshold": round(thr, 6),
            "target_fpr": round(1 - q, 4),
            "in_corpus_fpr": round(float((eval_half > thr).mean()), 4),
            "unseen_fpr": round(float((unseen > thr).mean()), 4),
            "n_threshold_reals": len(thr_half),
            "n_eval_reals": len(eval_half),
            "n_unseen": len(unseen),
        }
        if len(fake):
            f = fake[np.isfinite(fake)]
            row["tpr_on_fakes"] = round(float((f > thr).mean()), 4)
        rows.append(row)

    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / "false_positive_rate.csv", index=False)
    logger.info("\nFALSE-POSITIVE RATE at reals-only thresholds:\n%s", rep.to_string(index=False))

    for r in rows:
        ratio = r["unseen_fpr"] / max(r["in_corpus_fpr"], 1e-9)
        if r["in_corpus_fpr"] > 0 and ratio > 2.0:
            logger.warning(
                "q=%.2f: unseen FPR %.4f is %.1fx the in-corpus FPR %.4f. The detector "
                "fires far more on an UNSEEN real catalogue than on the benchmark's own "
                "reals — that is a corpus confound no AUC on the benchmark would reveal, "
                "and it must be reported.",
                r["quantile"],
                r["unseen_fpr"],
                ratio,
                r["in_corpus_fpr"],
            )

    if args.group_column and args.group_column in uns.columns:
        thr = float(np.quantile(thr_half, 0.95))
        by = (
            uns.assign(_flag=uns[args.score_column].to_numpy(float) > thr)
            .groupby(args.group_column)["_flag"]
            .agg(["mean", "size"])
            .rename(columns={"mean": "fpr_at_q95", "size": "n"})
            .sort_values("fpr_at_q95", ascending=False)
        )
        by.to_csv(out_dir / "false_positive_rate_by_group.csv")
        logger.info("\nFPR at q=0.95 by %s (worst first):\n%s", args.group_column, by.head(20).to_string())

    if args.calibrate_per_group and args.group_column in uns.columns:
        _conformal_by_group(args, out_dir, uns, thr_half, fake)

    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
