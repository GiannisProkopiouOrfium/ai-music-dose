"""Deployment-oriented analysis: precision at realistic prevalence, and what we miss.

Why EER is the wrong headline for this problem
-----------------------------------------------
MusicDET, SpecTTTra and most of the field report EER, which assumes the two
classes are equally likely and that a false positive costs the same as a false
negative. Neither holds for a catalogue. A distributor screening uploads sees a
small minority of AI tracks, and **flagging real music is far more expensive than
missing a fake** — it accuses a human artist.

At prevalence p, precision follows from TPR and FPR:

    precision = p·TPR / (p·TPR + (1−p)·FPR)

so at p = 1% an FPR of 5% caps precision at ~17% however good the TPR is. That is
the number a deployment decision actually turns on, and it is not in any of the
papers we compare against.

This script reports, per detector score:
  * precision / recall at prevalences {10%, 5%, 1%, 0.1%}, at thresholds set from
    REALS ONLY (so the operating point stays zero-shot);
  * the threshold needed to reach a target precision, and the recall it costs;
  * **error patterns** — which generators, and which real-track properties,
    dominate the misses. That is the "where does it fail" table.

Usage (EC2)
-----------
    python scripts/analyze_detector.py \\
        --score-csv reports/diagnostics/nmf_peak_fmc/nmf_peak_per_track_lvl.csv \\
        --score-column r_k20_sigma2_bins7167 \\
        --descriptor-csv reports/diagnostics/channel_fmc_2026_08_20/channel_per_track_canonical_profile_lvl.csv \\
        --out-dir reports/diagnostics/analysis_nmf_fmc
"""

from __future__ import annotations

import argparse
import logging
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
logger = logging.getLogger("analyze_detector")

PREVALENCES = (0.10, 0.05, 0.01, 0.001)
TARGET_PRECISIONS = (0.90, 0.95, 0.99)
DEFAULT_QUANTILES = (0.90, 0.95, 0.99, 0.999)
# The left tail is where a deployment threshold actually sits: at 1% prevalence,
# precision 0.9 needs an FPR near 0.076%. See partial_auc.
TARGET_FPRS = (0.001, 0.01, 0.05)
PAUC_MAX_FPR = 0.01


def fpr_upper_bound(n_false: int, n_total: int) -> float:
    """Upper 95% bound on the FPR, so an observed ZERO is not treated as zero.

    Seeing 0 false positives in 300 held-out reals does not mean FPR = 0; by the
    rule of three the 95% upper bound is 3/n = 0.01. Reporting the point estimate
    made a threshold with recall 1.9% look like it achieved precision 0.99 at 1%
    prevalence, because precision is 1.0 whenever FPR is exactly 0. Every
    precision figure here therefore uses the CONSERVATIVE bound.
    """
    if n_total <= 0:
        return float("nan")
    point = n_false / n_total
    return max(point, 3.0 / n_total)


def precision_at_prevalence(tpr: float, fpr: float, p: float) -> float:
    """Precision implied by a (TPR, FPR) pair at population prevalence ``p``.

    Computed analytically rather than from the balanced test set, because the
    benchmark's 1:1 class ratio is not any real catalogue's.
    """
    num = p * tpr
    den = num + (1.0 - p) * fpr
    return float(num / den) if den > 0 else float("nan")


def roc_curve(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) for every distinct threshold, both starting at the origin."""
    s = np.concatenate([real, fake])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tpr = np.concatenate([[0.0], np.cumsum(y) / max(len(fake), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - y) / max(len(real), 1)])
    return fpr, tpr


def partial_auc(real: np.ndarray, fake: np.ndarray, max_fpr: float) -> float:
    """MEAN TPR over FPR in ``[0, max_fpr]`` -- the left tail of the ROC.

    Why this and not AUC. Precision 0.9 at 1% prevalence needs an FPR near 0.076%, so
    a deployment decision is made entirely inside the first percent of the FPR axis.
    Macro AUC integrates over the whole axis and is therefore almost blind to the
    region that decides whether the detector is usable: two scores can tie on AUC
    while one of them is worthless at a strict threshold.

    Read the scale carefully -- it is NOT an AUC. Dividing the partial area by
    ``max_fpr`` makes this the average recall achievable while the false-positive
    rate stays below ``max_fpr``, which is directly interpretable but has its chance
    level at ``max_fpr / 2`` (0.005 at max_fpr = 0.01), not at 0.5. A perfect score
    gives 1.0. The McClish standardisation that puts chance at 0.5 is deliberately
    not used, because the raw recall is the number a deployer acts on.
    """
    if len(real) == 0 or len(fake) == 0 or max_fpr <= 0:
        return float("nan")
    fpr, tpr = roc_curve(real, fake)
    keep = fpr <= max_fpr
    x, y = fpr[keep], tpr[keep]
    if len(x) < 2:
        return float("nan")
    if x[-1] < max_fpr:
        # Interpolate the last step so the integration limit is exactly max_fpr
        # rather than the largest observed FPR below it.
        j = np.searchsorted(fpr, max_fpr)
        if j < len(fpr):
            frac = (max_fpr - x[-1]) / max(fpr[j] - x[-1], 1e-12)
            x = np.append(x, max_fpr)
            y = np.append(y, y[-1] + frac * (tpr[j] - y[-1]))
    return float(np.trapezoid(y, x) / max_fpr) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / max_fpr)


def threshold_grid(reals: np.ndarray, mode: str = "tail", n_points: int = 400) -> np.ndarray:
    """Candidate thresholds for the precision-target search.

    ⚠ CORRECTED 2026-09-17, and it changes a published claim. The historical grid was
    ``np.quantile(reals, np.linspace(0.5, 0.99999, 400))`` -- uniform in the quantile,
    so its last step spans 0.99874 to 0.99999 with nothing in between. On SONICS' 6,361
    threshold reals that gap is eight tracks wide, and it skips the three-to-six range
    where a strict operating point lives.

    The cost was not hypothetical. On ``analysis_pm4_sonics`` the operating-point table
    reports precision 0.9092 at 1% prevalence at q = 0.999, while
    ``precision_targets.csv`` from the SAME run called precision 0.90 at 1%
    unreachable with a best of 0.8830. Two tables, one script, one run,
    contradicting each other -- because the second could not see the threshold the
    first was standing on.

    Precision at low prevalence is decided entirely by the far-left tail of the ROC:
    0.9 at 1% prevalence needs a false-positive rate near 0.076%. A grid that is
    uniform in ``q`` puts almost all of its points where precision is hopeless and
    almost none where it is decided. ``tail`` is log-spaced in ``1 - q`` down to
    ``1/n``, which is the finest threshold the sample can distinguish at all.

    ``linear`` is kept so the superseded numbers can be reproduced on demand, per the
    project's standing rule that a corrected value is shown beside the one it
    replaces -- not so it can be used.
    """
    x = np.asarray(reals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([])
    body = np.linspace(0.5, 0.99999, n_points)
    if mode == "linear":
        return np.quantile(x, body)
    # A strict SUPERSET of the historical grid: the same 400 body points plus a
    # log-spaced tail down to 1/n. Adding candidates can only raise the best
    # precision the search finds, never lower it, so 'tail' cannot regress against
    # 'linear' on any input -- which is asserted in the tests rather than assumed.
    finest = max(1.0 / len(x), 1e-6)
    tail = 1.0 - np.logspace(np.log10(0.01), np.log10(finest), n_points)
    return np.quantile(x, np.unique(np.clip(np.concatenate([body, tail]), 0.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--descriptor-csv",
        default=None,
        help="Per-track descriptors, to look for patterns in the misses.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision-grid",
        choices=("tail", "linear"),
        default="tail",
        help="Threshold grid for precision_targets.csv. 'tail' is log-spaced in (1-q) down to "
        "1/n and resolves the far-left tail where precision targets are actually met; 'linear' "
        "reproduces this script's historical uniform grid, which does not. See threshold_grid.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUANTILES),
        help="Real-score quantiles used as thresholds. The default reproduces this script's "
        "historical inline grid exactly, so existing outputs are unchanged.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.score_csv, low_memory=False)
    if args.score_column not in df.columns:
        raise SystemExit(
            f"no column {args.score_column!r}. Available: "
            f"{[c for c in df.columns if c.startswith(('r_', 'comb_', 'id_', 'fuse_'))][:25]}"
        )
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")

    s = df[args.score_column].to_numpy(float)
    y = (df["label"] != "real").astype(int).to_numpy()
    ok = np.isfinite(s)
    s, y, df = s[ok], y[ok], df[ok].reset_index(drop=True)

    # Threshold from a HELD-OUT half of the reals: the other half measures FPR, so
    # the operating point is never validated on the tracks that set it.
    rng = np.random.RandomState(args.seed)
    real_idx = np.flatnonzero(y == 0)
    rng.shuffle(real_idx)
    thr_reals, eval_reals = real_idx[: len(real_idx) // 2], real_idx[len(real_idx) // 2 :]
    fake_idx = np.flatnonzero(y == 1)

    rows: list[dict] = []
    for q in args.quantiles:
        thr = float(np.quantile(s[thr_reals], q))
        n_fp = int((s[eval_reals] > thr).sum())
        fpr_point = n_fp / len(eval_reals)
        fpr = fpr_upper_bound(n_fp, len(eval_reals))
        tpr = float((s[fake_idx] > thr).mean())
        row = {
            "real_quantile": q,
            "threshold": round(thr, 6),
            "fpr_point": round(fpr_point, 5),
            "fpr_upper95": round(fpr, 5),
            "n_false_pos": n_fp,
            "n_eval_reals": len(eval_reals),
            "tpr_recall": round(tpr, 4),
        }
        for p in PREVALENCES:
            row[f"precision@p={p:g}"] = round(precision_at_prevalence(tpr, fpr, p), 4)
        rows.append(row)
    op = pd.DataFrame(rows)
    op.to_csv(out_dir / "operating_points.csv", index=False)
    logger.info(
        "\nOPERATING POINTS (thresholds from held-out reals only — still zero-shot):\n%s",
        op.to_string(index=False),
    )

    # --- the left tail of the ROC -------------------------------------------
    # Keyed by TARGET FPR rather than by real-quantile, which is why it is its own
    # table instead of extra columns on operating_points.csv: a pAUC does not vary
    # with the quantile, and repeating one value down four rows invites it being read
    # as four measurements.
    pauc = partial_auc(s[eval_reals], s[fake_idx], PAUC_MAX_FPR)
    tail_rows: list[dict] = []
    for target in TARGET_FPRS:
        thr = float(np.quantile(s[thr_reals], 1.0 - target))
        n_fp = int((s[eval_reals] > thr).sum())
        fpr = fpr_upper_bound(n_fp, len(eval_reals))
        tpr = float((s[fake_idx] > thr).mean())
        row = {
            "target_fpr": target,
            "threshold": round(thr, 6),
            "achieved_fpr_point": round(n_fp / len(eval_reals), 5),
            "achieved_fpr_upper95": round(fpr, 5),
            "tpr_at_fpr": round(tpr, 4),
            f"pauc_fpr{PAUC_MAX_FPR:g}_meantpr": round(pauc, 4),
        }
        for pv in PREVALENCES:
            row[f"precision@p={pv:g}"] = round(precision_at_prevalence(tpr, fpr, pv), 4)
        tail_rows.append(row)
    tail = pd.DataFrame(tail_rows)
    tail.to_csv(out_dir / "left_tail.csv", index=False)
    logger.info(
        "\nTHE LEFT TAIL: TPR at a fixed FPR, and the MEAN TPR over FPR <= %g (chance for "
        "that column is max_fpr/2, not 0.5). This is the region a deployment threshold sits "
        "in, and macro AUC is almost blind to it:\n%s",
        PAUC_MAX_FPR,
        tail.to_string(index=False),
    )

    # What threshold reaches a target precision, and what recall it costs?
    grid = threshold_grid(s[thr_reals], mode=args.precision_grid)
    prec_rows: list[dict] = []
    for p in PREVALENCES:
        for target in TARGET_PRECISIONS:
            best = None
            best_prec, best_prec_recall = 0.0, 0.0
            for thr in grid:
                n_fp = int((s[eval_reals] > thr).sum())
                fpr = fpr_upper_bound(n_fp, len(eval_reals))
                tpr = float((s[fake_idx] > thr).mean())
                pr = precision_at_prevalence(tpr, fpr, p)
                if pr > best_prec:
                    best_prec, best_prec_recall = pr, tpr
                if best is None and pr >= target:
                    best = (thr, fpr, tpr)  # grid ascends: first hit = best recall
            prec_rows.append(
                {
                    "prevalence": p,
                    "target_precision": target,
                    "achievable": best is not None,
                    "threshold": round(best[0], 6) if best else None,
                    "fpr_upper95": round(best[1], 5) if best else None,
                    "recall": round(best[2], 4) if best else None,
                    # When the target is out of reach, the useful number is how close
                    # it gets — "unreachable" alone does not tell a deployer anything.
                    "best_precision_reachable": round(best_prec, 4),
                    "recall_there": round(best_prec_recall, 4),
                }
            )
    prec = pd.DataFrame(prec_rows)
    prec.to_csv(out_dir / "precision_targets.csv", index=False)
    logger.info("\nRECALL AT A TARGET PRECISION (the deployment table):\n%s", prec.to_string(index=False))
    unreachable = prec[~prec["achievable"]]
    if len(unreachable):
        logger.warning(
            "%d of %d (prevalence, precision) targets are UNREACHABLE at any threshold. "
            "At low prevalence the FPR floor caps precision regardless of recall — this is "
            "the honest limitation to state, and no EER number reveals it.",
            len(unreachable),
            len(prec),
        )

    # --- error patterns -----------------------------------------------------
    logger.info(
        "NOTE: FPR is reported as the 95%% UPPER BOUND (rule of three when zero false "
        "positives are observed). A point estimate of exactly 0 on %d reals would make any "
        "threshold look like precision 1.0, which is an artifact of the sample size.",
        len(eval_reals),
    )

    thr95 = float(np.quantile(s[thr_reals], 0.95))
    df["_flagged"] = s > thr95
    df["_error"] = np.where(
        (y == 1) & ~df["_flagged"],
        "false_negative",
        np.where((y == 0) & df["_flagged"], "false_positive", "correct"),
    )
    by_gen = (
        df[y == 1]
        .groupby("algorithm")["_flagged"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "recall_at_q95", "size": "n"})
        .sort_values("recall_at_q95")
    )
    by_gen.to_csv(out_dir / "errors_by_generator.csv")
    logger.info("\nRECALL BY GENERATOR at the 95%% real quantile (worst first):\n%s", by_gen.to_string())

    # The same breakdown at every quantile, written BESIDE the file above rather than
    # replacing it, so nothing already published moves. The q=0.95 rows here must
    # equal errors_by_generator.csv by construction.
    per_q: list[dict] = []
    for q in args.quantiles:
        thr_q = float(np.quantile(s[thr_reals], q))
        flagged = s > thr_q
        for gen, sub in df[y == 1].groupby("algorithm"):
            idx = sub.index.to_numpy()
            per_q.append(
                {
                    "real_quantile": q,
                    "algorithm": gen,
                    # NOT rounded: errors_by_generator.csv writes the raw mean, and
                    # tests/test_analyze_detector_left_tail.py asserts the two agree
                    # exactly at q=0.95. Rounding here would make them disagree in the
                    # fourth decimal for no benefit.
                    "recall": float(flagged[idx].mean()),
                    "n": int(len(idx)),
                }
            )
    by_gen_q = pd.DataFrame(per_q).sort_values(["real_quantile", "recall"])
    by_gen_q.to_csv(out_dir / "errors_by_generator_by_q.csv", index=False)
    logger.info(
        "\nRECALL BY GENERATOR AT EVERY QUANTILE. A family whose recall collapses as the "
        "threshold tightens has a score that ranks it correctly but does not separate it -- "
        "which is invisible in AUC:\n%s",
        by_gen_q.to_string(index=False),
    )

    if args.descriptor_csv and "track_id" in df.columns:
        desc = pd.read_csv(args.descriptor_csv, low_memory=False)
        if "track_id" in desc.columns:
            m = df.merge(desc, on="track_id", how="left", suffixes=("", "_d"))
            num = [
                c
                for c in m.columns
                if m[c].dtype.kind in "fc"
                and c not in (args.score_column,)
                and m[c].notna().sum() > 50
                and m[c].nunique() > 5
            ]
            pat: list[dict] = []
            for cls, mask in (("false_negative", (y == 1)), ("false_positive", (y == 0))):
                sub = m[mask]
                err = (sub["_error"] == cls).to_numpy()
                if err.sum() < 20 or (~err).sum() < 20:
                    continue
                for c in num:
                    a, b = sub.loc[err, c].to_numpy(float), sub.loc[~err, c].to_numpy(float)
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    if len(a) < 20 or len(b) < 20:
                        continue
                    pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
                    pat.append(
                        {
                            "error_class": cls,
                            "descriptor": c,
                            "cohens_d": round(float((a.mean() - b.mean()) / pooled), 3),
                            "err_median": round(float(np.median(a)), 4),
                            "ok_median": round(float(np.median(b)), 4),
                            "n_err": int(len(a)),
                        }
                    )
            if pat:
                pt = pd.DataFrame(pat)
                pt["abs_d"] = pt["cohens_d"].abs()
                pt = pt.sort_values(["error_class", "abs_d"], ascending=[True, False])
                pt.drop(columns="abs_d").to_csv(out_dir / "error_patterns.csv", index=False)
                logger.info(
                    "\nWHAT THE ERRORS LOOK LIKE (|Cohen's d| > 0.3; which track properties "
                    "separate the misses from the hits):\n%s",
                    pt[pt["abs_d"] > 0.3].drop(columns="abs_d").head(20).to_string(index=False),
                )

    df.to_csv(out_dir / "per_track_with_errors.csv", index=False)
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
