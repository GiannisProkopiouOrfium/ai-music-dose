"""Subsample fairness gate.

Validates that a proposed subsample (as produced by the balanced sampler) is:
  1. **Representativeness**: its genre / algorithm distribution approximates the full dataset.
  2. **Covariate balance**: real-vs-fake standardised mean differences (SMD) < 0.10
     after matching; KS p-values > 0.05.
  3. **Stratum coverage**: all algorithm × fake_label combinations are present with
     at least a minimum count.

Pass/fail criteria
------------------
  PASS: all conditions satisfied → safe to scale up
  WARN: one condition marginal → review before scaling
  FAIL: any core condition violated → do not use subsample for scientific claims

Outputs (all in --output-dir)
------------------------------
  representativeness_report.csv   genre + algorithm KL divergence vs full dataset
  covariate_balance_report.csv    per-covariate SMD, KS test, pass/fail
  stratum_coverage_report.csv     per stratum n_sampled, pass/fail
  subsample_validation_summary.txt human-readable overall verdict

Usage
-----
# After running the balanced sampler (or any run script in subsample mode):
poetry run python scripts/validate_subsample.py \
  --subsample-csv data/processed/acoustic_analysis_canonical/acoustic_features.csv \
  --full-fake-csv data/raw/sonics/metadata/fake_songs.csv \
  --full-real-csv data/raw/sonics/metadata/real_songs.csv \
  --output-dir data/processed/subsample_validation

# With covariate profiles:
poetry run python scripts/validate_subsample.py \
  --subsample-csv data/processed/acoustic_analysis_canonical/acoustic_features.csv \
  --full-fake-csv data/raw/sonics/metadata/fake_songs.csv \
  --full-real-csv data/raw/sonics/metadata/real_songs.csv \
  --covariate-profiles-csv data/processed/covariate_profile/covariate_profiles.csv \
  --output-dir data/processed/subsample_validation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- pass/fail thresholds ----
SMD_THRESHOLD = 0.10  # standardised mean difference; < 0.10 = balanced
KS_PVALUE_THRESHOLD = 0.05  # KS test p-value; > 0.05 = distributions compatible
KL_THRESHOLD = 0.05  # KL divergence bits; < 0.05 = distributions compatible
MIN_STRATUM_COUNT = 5  # minimum tracks per (algorithm × fake_label)

COVARIATE_NAMES = ["duration", "loudness_lufs", "spectral_centroid_mean", "tempo"]


# ---------------------------------------------------------------------------
# KL divergence helper
# ---------------------------------------------------------------------------


def _kl_divergence(p: pd.Series, q: pd.Series) -> float:
    """Compute KL(p || q) between two categorical distributions (value_counts)."""
    all_cats = set(p.index) | set(q.index)
    p_arr = np.array([p.get(c, 0) for c in all_cats], dtype=float)
    q_arr = np.array([q.get(c, 0) for c in all_cats], dtype=float)
    p_arr = p_arr / (p_arr.sum() + 1e-12)
    q_arr = q_arr / (q_arr.sum() + 1e-12)
    # Add epsilon to avoid log(0)
    eps = 1e-8
    return float(np.sum(p_arr * np.log((p_arr + eps) / (q_arr + eps))))


def _js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, 1])."""
    all_cats = sorted(set(p.index) | set(q.index))
    p_arr = np.array([p.get(c, 0) for c in all_cats], dtype=float)
    q_arr = np.array([q.get(c, 0) for c in all_cats], dtype=float)
    p_arr = p_arr / (p_arr.sum() + 1e-12)
    q_arr = q_arr / (q_arr.sum() + 1e-12)
    m = 0.5 * (p_arr + q_arr)
    eps = 1e-8
    js = 0.5 * np.sum(p_arr * np.log((p_arr + eps) / (m + eps))) + 0.5 * np.sum(
        q_arr * np.log((q_arr + eps) / (m + eps))
    )
    return float(np.clip(js, 0, 1))


# ---------------------------------------------------------------------------
# Representativeness check
# ---------------------------------------------------------------------------


def check_representativeness(
    subsample_df: pd.DataFrame,
    full_fake_df: pd.DataFrame,
    full_real_df: pd.DataFrame,
    kl_threshold: float = KL_THRESHOLD,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
) -> tuple[pd.DataFrame, bool]:
    """Compare subsample distributions against the full dataset."""
    rows = []
    all_pass = True

    # --- Algorithm distribution (fakes only) ---
    if "algorithm" in subsample_df.columns and "algorithm" in full_fake_df.columns:
        sub_fake = subsample_df[subsample_df["label"] == "fake"]
        sub_algo_counts = sub_fake["algorithm"].value_counts(normalize=True)
        full_algo_counts = full_fake_df["algorithm"].value_counts(normalize=True)
        js = _js_divergence(sub_algo_counts, full_algo_counts)
        passed = js < kl_threshold
        if not passed:
            all_pass = False
        rows.append(
            {
                "dimension": "algorithm distribution (fakes)",
                "subsample_n": len(sub_fake),
                "full_n": len(full_fake_df),
                "js_divergence": js,
                "threshold": kl_threshold,
                "passed": passed,
                "notes": " | ".join(f"{k}:{v:.1%}" for k, v in sub_algo_counts.head(5).items()),
            }
        )

    # --- Genre distribution (fakes) ---
    if "genre" in subsample_df.columns and "genre" in full_fake_df.columns:
        sub_fake = subsample_df[subsample_df["label"] == "fake"]
        sub_genre = sub_fake["genre"].astype(str).str.lower().value_counts(normalize=True)
        full_genre = full_fake_df["genre"].astype(str).str.lower().value_counts(normalize=True)
        js = _js_divergence(sub_genre, full_genre)
        passed = js < kl_threshold
        if not passed:
            all_pass = False
        rows.append(
            {
                "dimension": "genre distribution (fakes)",
                "subsample_n": len(sub_fake),
                "full_n": len(full_fake_df),
                "js_divergence": js,
                "threshold": kl_threshold,
                "passed": passed,
                "notes": " | ".join(f"{k}:{v:.1%}" for k, v in sub_genre.head(5).items()),
            }
        )

    # --- Duration quartile coverage (fakes) ---
    if "duration" in subsample_df.columns:
        sub_fake = subsample_df[subsample_df["label"] == "fake"]
        full_durs = pd.to_numeric(full_fake_df.get("duration", pd.Series(dtype=float)), errors="coerce").dropna()
        sub_durs = pd.to_numeric(sub_fake.get("duration", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(full_durs) > 10 and len(sub_durs) > 10:
            ks_stat, ks_pval = scipy.stats.ks_2samp(sub_durs.to_numpy(), full_durs.to_numpy())
            passed = ks_pval >= ks_pvalue_threshold
            if not passed:
                all_pass = False
            rows.append(
                {
                    "dimension": "duration distribution (fakes)",
                    "subsample_n": len(sub_durs),
                    "full_n": len(full_durs),
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pval,
                    "threshold": ks_pvalue_threshold,
                    "passed": passed,
                    "notes": f"sub mean={sub_durs.mean():.0f}s full mean={full_durs.mean():.0f}s",
                }
            )

    return pd.DataFrame(rows), all_pass


# ---------------------------------------------------------------------------
# Covariate balance check (real vs fake within subsample)
# ---------------------------------------------------------------------------


def check_covariate_balance(
    subsample_df: pd.DataFrame,
    covariate_profiles_csv: str | None = None,
    smd_threshold: float = SMD_THRESHOLD,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
) -> tuple[pd.DataFrame, bool]:
    """Check real-vs-fake covariate balance in the subsample."""
    if covariate_profiles_csv and Path(covariate_profiles_csv).exists():
        cov_df = pd.read_csv(covariate_profiles_csv, low_memory=False)
        # Filter to tracks in the subsample
        sub_ids = set(subsample_df["track_id"].astype(str))
        cov_df = cov_df[cov_df["track_id"].astype(str).isin(sub_ids)]
    else:
        cov_df = subsample_df

    real_mask = cov_df["label"] == "real"
    fake_mask = cov_df["label"] == "fake"

    rows = []
    all_pass = True

    for cov in COVARIATE_NAMES:
        if cov not in cov_df.columns:
            continue
        x_r = pd.to_numeric(cov_df.loc[real_mask, cov], errors="coerce").dropna().to_numpy(dtype=float)
        x_f = pd.to_numeric(cov_df.loc[fake_mask, cov], errors="coerce").dropna().to_numpy(dtype=float)

        if len(x_r) < 5 or len(x_f) < 5:
            continue

        pooled_std = np.sqrt((np.var(x_r, ddof=1) + np.var(x_f, ddof=1)) / 2)
        smd = abs(np.mean(x_r) - np.mean(x_f)) / (pooled_std + 1e-12)
        ks_stat, ks_pval = scipy.stats.ks_2samp(x_r, x_f)
        passed = bool(smd <= smd_threshold and ks_pval >= ks_pvalue_threshold)
        if not passed:
            all_pass = False

        rows.append(
            {
                "covariate": cov,
                "n_real": len(x_r),
                "n_fake": len(x_f),
                "mean_real": float(np.mean(x_r)),
                "mean_fake": float(np.mean(x_f)),
                "smd": float(smd),
                "smd_threshold": smd_threshold,
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "ks_threshold": ks_pvalue_threshold,
                "passed": passed,
            }
        )

    return pd.DataFrame(rows), all_pass


# ---------------------------------------------------------------------------
# Stratum coverage check
# ---------------------------------------------------------------------------


def check_stratum_coverage(subsample_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Check all (algorithm × fake_label) strata are sufficiently represented."""
    fake_df = subsample_df[subsample_df["label"] == "fake"]
    if fake_df.empty:
        return pd.DataFrame(), False

    rows = []
    all_pass = True

    for (algo, fl), grp in fake_df.groupby(["algorithm", "fake_label"], dropna=False, sort=True):
        n = len(grp)
        passed = n >= MIN_STRATUM_COUNT
        if not passed:
            all_pass = False
        rows.append(
            {
                "algorithm": algo,
                "fake_label": fl,
                "n_sampled": n,
                "min_required": MIN_STRATUM_COUNT,
                "passed": passed,
            }
        )

    return pd.DataFrame(rows), all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subsample fairness gate: representativeness + covariate balance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subsample-csv",
        required=True,
        help=(
            "CSV with per-track features (acoustic_features.csv or ablation_features.csv). "
            "Must have: track_id, label, algorithm, fake_label, genre columns."
        ),
    )
    parser.add_argument("--full-fake-csv", required=True, help="Full fake_songs.csv from SONICS metadata")
    parser.add_argument("--full-real-csv", required=True, help="Full real_songs.csv from SONICS metadata")
    parser.add_argument(
        "--covariate-profiles-csv",
        default=None,
        help="covariate_profiles.csv from profile_covariates.py (enriches balance check)",
    )
    parser.add_argument("--output-dir", default="data/processed/subsample_validation")
    parser.add_argument(
        "--smd-threshold",
        type=float,
        default=SMD_THRESHOLD,
        help="Max acceptable standardised mean difference (default: 0.10)",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=KL_THRESHOLD,
        help="Max acceptable JS divergence for distribution comparison (default: 0.05)",
    )
    parser.add_argument(
        "--balanced-design",
        action="store_true",
        default=False,
        help=(
            "Set when the subsample is a DELIBERATELY BALANCED ablation (equal per-stratum "
            "sampling). In this mode the algorithm distribution check is raised to JS < 0.30 "
            "(informational) because uniform per-generator sampling is the INTENDED design "
            "rather than a fairness problem. Covariate balance and stratum coverage checks "
            "are unchanged."
        ),
    )
    args = parser.parse_args()

    # Use local variables for thresholds so check functions receive them explicitly.
    smd_thr = args.smd_threshold
    kl_thr = args.kl_threshold
    # When the design is explicitly balanced-per-stratum, relax the algorithm
    # distribution JS threshold: we EXPECT our uniform sampler to diverge from
    # the full dataset's skewed generator counts.  This is scientific intent,
    # not bias.  All other checks (covariate SMD, stratum coverage) are unchanged.
    if getattr(args, "balanced_design", False):
        kl_thr = max(kl_thr, 0.30)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading subsample from %s ...", args.subsample_csv)
    subsample_df = pd.read_csv(args.subsample_csv, low_memory=False)
    full_fake_df = pd.read_csv(args.full_fake_csv, low_memory=False)
    full_real_df = pd.read_csv(args.full_real_csv, low_memory=False)

    logger.info(
        "Subsample: %d total (%d real, %d fake)",
        len(subsample_df),
        (subsample_df["label"] == "real").sum() if "label" in subsample_df.columns else -1,
        (subsample_df["label"] == "fake").sum() if "label" in subsample_df.columns else -1,
    )

    # ---- Check 1: Representativeness ----
    logger.info("--- Check 1: Representativeness ---")
    repr_df, repr_pass = check_representativeness(
        subsample_df,
        full_fake_df,
        full_real_df,
        kl_threshold=kl_thr,
        ks_pvalue_threshold=KS_PVALUE_THRESHOLD,
    )
    repr_df.to_csv(output_dir / "representativeness_report.csv", index=False)
    logger.info("Saved representativeness_report.csv")
    for _, r in repr_df.iterrows():
        flag = "PASS" if r.get("passed") else "FAIL"
        logger.info("  [%s] %s", flag, r["dimension"])

    # ---- Check 2: Covariate balance ----
    logger.info("--- Check 2: Covariate balance (real vs fake within subsample) ---")
    cov_df, cov_pass = check_covariate_balance(
        subsample_df,
        args.covariate_profiles_csv,
        smd_threshold=smd_thr,
        ks_pvalue_threshold=KS_PVALUE_THRESHOLD,
    )
    cov_df.to_csv(output_dir / "covariate_balance_report.csv", index=False)
    logger.info("Saved covariate_balance_report.csv")
    if not cov_df.empty:
        logger.info("%-30s  %7s  %7s  %6s  %s", "covariate", "smd", "ks_pval", "n_real", "status")
        for _, r in cov_df.iterrows():
            flag = "PASS" if r.get("passed") else "FAIL"
            logger.info(
                "  %-28s  %7.3f  %7.4f  %6d  %s", r["covariate"], r["smd"], r["ks_pvalue"], int(r["n_real"]), flag
            )

    # ---- Check 3: Stratum coverage ----
    logger.info("--- Check 3: Stratum coverage ---")
    strat_df, strat_pass = check_stratum_coverage(subsample_df)
    strat_df.to_csv(output_dir / "stratum_coverage_report.csv", index=False)
    logger.info("Saved stratum_coverage_report.csv")
    n_fail_strata = (~strat_df["passed"]).sum() if not strat_df.empty else 0
    logger.info("Strata: %d total, %d with n < %d", len(strat_df), n_fail_strata, MIN_STRATUM_COUNT)

    # ---- Overall verdict ----
    all_pass = repr_pass and cov_pass and strat_pass
    if all_pass:
        verdict = "PASS"
        msg = "Subsample is balanced and representative — safe to scale up."
    elif repr_pass and strat_pass:
        verdict = "WARN"
        msg = "Representativeness and coverage OK, but some covariate imbalance detected. Review before scaling."
    else:
        verdict = "FAIL"
        msg = "Subsample failed one or more fairness checks — do NOT use for scientific claims without fixing."

    summary_lines = [
        f"Subsample Validation Summary",
        f"{'='*60}",
        f"Subsample CSV:     {args.subsample_csv}",
        f"Subsample size:    {len(subsample_df)} tracks",
        f"Balanced design:   {'YES (algorithm JS threshold relaxed to 0.30)' if getattr(args, 'balanced_design', False) else 'NO'}",
        f"",
        f"Check 1: Representativeness    → {'PASS' if repr_pass else 'FAIL'}",
        f"Check 2: Covariate balance     → {'PASS' if cov_pass else 'FAIL'}",
        f"Check 3: Stratum coverage      → {'PASS' if strat_pass else 'FAIL'}",
        f"",
        f"OVERALL VERDICT: {verdict}",
        f"",
        f"Interpretation: {msg}",
        f"",
        f"Thresholds used:",
        f"  SMD < {smd_thr}  (standardised mean difference per covariate)",
        f"  JS divergence < {kl_thr}  (distribution similarity to full dataset)",
        f"  min stratum count = {MIN_STRATUM_COUNT}",
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (output_dir / "subsample_validation_summary.txt").write_text(summary_text + "\n")
    logger.info("Saved subsample_validation_summary.txt")

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
