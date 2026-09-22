"""Comprehensive confound analysis for the EnCodec+RealNVP real-manifold detector.

This is the scientific-rigor gating test (Plan Phase A2).  It answers the key
question: **does the anomaly score reflect genuine codec/vocoder artifacts, or
is it picking up genre, instrumentation, or bandwidth differences between
the SONICS real and fake populations?**

Four complementary analyses are run:
  1. Genre-stratified AUC — separation within each CLAP genre bucket.
  2. Covariate/genre residualization — partial AUC after removing linear genre
     and acoustic-covariate effects from the anomaly score.
  3. Genre-only classifier baseline — can CLAP genre posteriors alone decide
     real vs fake?  High AUC => genre is confounded; must be controlled.
  4. FMA outlier attribution — for the 15.3% FPR real tracks flagged by the
     flow on FMA-small, what genres / covariates drive the misclassification?

All results are saved to CSV/JSON in --output-dir and a summary is printed.

Usage (EC2)
-----------
poetry run python scripts/analyze_confounds.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --covariate-csv data/processed/covariate_profile_canonical/covariate_profiles.csv \\
    --output-dir reports/confound_analysis \\
    --score-col encodec_wf_mean

# With FMA external-score for FPR attribution:
poetry run python scripts/analyze_confounds.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --covariate-csv data/processed/covariate_profile_canonical/covariate_profiles.csv \\
    --fma-scores-csv data/processed/external_scores/fma_fpr/external_scores.csv \\
    --fma-genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --output-dir reports/confound_analysis \\
    --score-col encodec_wf_mean
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, equal_error_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACOUSTIC_COVARIATES = [
    "duration",
    "tempo",
    "key",
    "loudness_lufs",
    "spectral_centroid_mean",
    "effective_bandwidth_hz",
    "spectral_rolloff_85_hz",
]


def _bootstrap_auc(y: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, sb)))
    aucs = np.array(aucs)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ci_lo": float(np.percentile(aucs, 2.5)),
        "ci_hi": float(np.percentile(aucs, 97.5)),
        "n_boot": len(aucs),
    }


def _partial_auc_after_residualization(
    df: pd.DataFrame,
    score_col: str,
    feature_cols: list[str],
) -> dict:
    """Regress score_col on feature_cols, return AUC on residuals.

    If the confounders fully explain the anomaly score, the residual AUC
    should collapse toward 0.5.  If it remains high, the signal is genuine.
    """
    sub = df[[score_col] + feature_cols + ["is_fake"]].dropna()
    if len(sub) < 50 or sub["is_fake"].nunique() < 2:
        return {"residual_auc": float("nan"), "n": len(sub)}

    X = sub[feature_cols].values.astype(float)
    # score_col (wf_mean) is mean log-likelihood: higher = more real-like. Negate
    # up front so every quantity below lives in "anomaly-score" space (higher =
    # more fake-like), matching the convention used everywhere else in this
    # script (genre_stratified_auc, Analysis 1 overall AUC, etc.). Without this,
    # full_auc/residual_auc come out inverted (~1 - true_auc), which is what was
    # happening before this fix (full_auc was reported as ~0.098 instead of ~0.90).
    y_score = -sub[score_col].values.astype(float)
    y_label = sub["is_fake"].values.astype(int)

    # Standardize features
    X = StandardScaler().fit_transform(X)

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(X, y_score)
    residuals = y_score - reg.predict(X)

    # AUC on residuals (residual > 0 means "more anomalous than explained by covariates")
    try:
        auc_res = float(roc_auc_score(y_label, residuals))
    except Exception:
        auc_res = float("nan")

    full_auc = float(roc_auc_score(y_label, y_score)) if sub["is_fake"].nunique() >= 2 else float("nan")

    return {
        "full_auc": full_auc,
        "residual_auc": auc_res,
        "auc_drop": full_auc - auc_res,
        "n": len(sub),
        "features_used": feature_cols,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Genre-stratified AUC
# ---------------------------------------------------------------------------


def genre_stratified_auc(df: pd.DataFrame, score_col: str, min_per_class: int = 20) -> pd.DataFrame:
    """Compute real-vs-fake AUC/EER within each CLAP genre bucket."""
    rows = []
    genres = df["genre_top1"].dropna().unique()

    for genre in sorted(genres):
        sub = df[df["genre_top1"] == genre]
        real_s = sub.loc[sub["is_fake"] == 0, score_col].dropna().values
        fake_s = sub.loc[sub["is_fake"] == 1, score_col].dropna().values

        if len(real_s) < min_per_class or len(fake_s) < min_per_class:
            continue

        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        # score_col is wf_mean (log-likelihood, higher = more real) -> negate for AUC
        s = np.r_[-real_s, -fake_s]
        auc, eer = auc_and_eer(y, s)

        rows.append(
            {
                "genre": genre,
                "n_real": len(real_s),
                "n_fake": len(fake_s),
                "auc": auc,
                "eer": eer,
                "real_mean_score": float(np.mean(real_s)),
                "fake_mean_score": float(np.mean(fake_s)),
                "effect_size_d": float(
                    (np.mean(fake_s) - np.mean(real_s)) / (np.std(np.r_[real_s, fake_s], ddof=1) + 1e-9)
                ),
            }
        )

    df_out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    logger.info(
        "\nGenre-stratified AUC (%d genres with >=%d real AND >=%d fake):\n%s",
        len(df_out),
        min_per_class,
        min_per_class,
        df_out[["genre", "n_real", "n_fake", "auc", "eer"]].to_string(index=False),
    )
    return df_out


# ---------------------------------------------------------------------------
# Analysis 2: Covariate & genre residualization
# ---------------------------------------------------------------------------


def residualization_analysis(df: pd.DataFrame, score_col: str) -> list[dict]:
    """Partial AUC under several residualization scenarios."""
    # One-hot encode genre
    if "genre_top1" in df.columns:
        genre_dummies = pd.get_dummies(df["genre_top1"], prefix="g", drop_first=True)
        df = pd.concat([df, genre_dummies], axis=1)
        genre_cols = list(genre_dummies.columns)
    else:
        genre_cols = []

    # Some acoustic covariates can be almost entirely missing (e.g. a broken
    # extractor silently producing NaN for every track). Including such a
    # column in the residualization feature set means a single all-NaN column
    # makes the row-wise dropna() wipe out the ENTIRE dataframe (n=0), hiding
    # every other (working) covariate's signal too. Guard against this by
    # requiring a minimum non-null coverage before a covariate is eligible.
    min_coverage = 0.7
    usable_acoustic = []
    for c in ACOUSTIC_COVARIATES:
        if c not in df.columns:
            continue
        cov = df[c].notna().mean()
        if cov >= min_coverage:
            usable_acoustic.append(c)
        else:
            logger.warning(
                "Excluding acoustic covariate '%s' from residualization: only %.1f%% non-null "
                "(< %.0f%% threshold) — likely a broken extractor; check profile_covariates.py.",
                c,
                cov * 100,
                min_coverage * 100,
            )
    if usable_acoustic:
        logger.info("Acoustic covariates used for residualization: %s", usable_acoustic)

    scenarios = [
        ("acoustic_only", usable_acoustic),
        ("genre_top1_dummies", ["genre_top1"]),  # converted below
        ("acoustic_plus_genre", usable_acoustic + ["genre_top1"]),
    ]

    results = []
    for name, feats in scenarios:
        if "genre_top1_dummies" in name:
            feats = genre_cols
        elif "genre_top1" in feats:
            feats = [f for f in feats if f != "genre_top1"] + genre_cols

        # Keep only features that actually exist
        feats = [f for f in feats if f in df.columns]
        if not feats:
            logger.warning("Scenario %s: no valid features found, skipping", name)
            continue

        res = _partial_auc_after_residualization(df, score_col, feats)
        res["scenario"] = name
        results.append(res)
        logger.info(
            "  [residualize=%s]  full_auc=%.4f  residual_auc=%.4f  drop=%.4f  n=%d",
            name,
            res.get("full_auc", float("nan")),
            res.get("residual_auc", float("nan")),
            res.get("auc_drop", float("nan")),
            res.get("n", 0),
        )

    # Additionally: full CLAP genre posterior vector residualization
    if "clap_genre_vector" in df.columns:
        try:
            gvec = np.vstack(df["clap_genre_vector"].apply(json.loads).values)
            tmp = df[[score_col, "is_fake"]].copy()
            tmp = tmp[tmp[score_col].notna() & tmp["is_fake"].notna()].copy()
            idx = tmp.index
            gvec_sub = gvec[df.index.get_indexer(idx)]  # align
            gvec_sub = StandardScaler().fit_transform(gvec_sub)
            # Negate for the same reason as _partial_auc_after_residualization above:
            # wf_mean is higher-for-real; flip to anomaly-score convention (higher=fake-like).
            score_sub = -tmp[score_col].values.astype(float)
            y_sub = tmp["is_fake"].values.astype(int)

            from sklearn.linear_model import LinearRegression

            reg = LinearRegression().fit(gvec_sub, score_sub)
            residuals = score_sub - reg.predict(gvec_sub)
            auc_res = float(roc_auc_score(y_sub, residuals)) if len(np.unique(y_sub)) > 1 else float("nan")
            full_auc = float(roc_auc_score(y_sub, score_sub)) if len(np.unique(y_sub)) > 1 else float("nan")
            res = {
                "scenario": "clap_full_posterior",
                "full_auc": full_auc,
                "residual_auc": auc_res,
                "auc_drop": full_auc - auc_res,
                "n": len(score_sub),
            }
            results.append(res)
            logger.info(
                "  [residualize=clap_full_posterior]  full_auc=%.4f  residual_auc=%.4f  drop=%.4f  n=%d",
                full_auc,
                auc_res,
                full_auc - auc_res,
                len(score_sub),
            )
        except Exception as exc:
            logger.warning("CLAP posterior residualization failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Genre-only classifier baseline
# ---------------------------------------------------------------------------


def genre_only_classifier(df: pd.DataFrame) -> dict:
    """Train a logistic classifier on CLAP genre posteriors -> real/fake.

    If this classifier achieves high AUC, the genre distribution differs
    between real and fake and must be explicitly controlled.
    """
    sub = (
        df[df["clap_genre_vector"].notna() & df["is_fake"].notna()].copy()
        if "clap_genre_vector" in df.columns
        else pd.DataFrame()
    )

    if len(sub) < 50 or sub["is_fake"].nunique() < 2:
        logger.warning("Not enough data for genre-only classifier (need clap_genre_vector column)")
        return {"genre_only_auc": float("nan"), "n": len(sub)}

    try:
        X = np.vstack(sub["clap_genre_vector"].apply(json.loads).values)
        y = sub["is_fake"].values.astype(int)
        X = StandardScaler().fit_transform(X)

        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = cross_val_predict(
            LogisticRegression(max_iter=500, C=1.0),
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        auc = float(roc_auc_score(y, proba))
        eer = equal_error_rate(y, proba)
        logger.info(
            "Genre-only classifier (logistic on CLAP posterior, 5-fold CV):  AUC=%.4f  EER=%.4f  n=%d",
            auc,
            eer,
            len(y),
        )
        logger.info(
            "  Interpretation: if AUC > 0.7, genre distribution differs between "
            "real and fake => must control for genre in confound analysis."
        )
        return {"genre_only_auc": auc, "genre_only_eer": eer, "n": len(y)}
    except Exception as exc:
        logger.warning("Genre-only classifier failed: %s", exc)
        return {"genre_only_auc": float("nan"), "n": len(sub)}


# ---------------------------------------------------------------------------
# Analysis 4: FMA FPR outlier attribution
# ---------------------------------------------------------------------------


def fma_outlier_attribution(
    fma_scores: pd.DataFrame,
    fma_genres: pd.DataFrame,
    sonics_real_anomaly_threshold: float,
    fma_covariate_csv: str | None,
) -> pd.DataFrame:
    """Attribute FMA real-track false positives to genre / covariate clusters.

    Parameters
    ----------
    sonics_real_anomaly_threshold : the SONICS-calibrated anomaly-score threshold
        (already in `wf_anomaly_score` units, i.e. `-wf_mean`; higher = more anomalous).
        Compute this from the SONICS real distribution's own percentile
        (see `main()` — NOT a raw log-likelihood value, and NOT negated here).
    """
    if fma_scores.empty or fma_genres.empty:
        logger.warning("FMA scores or genres empty — skipping FPR attribution")
        return pd.DataFrame()

    # Merge scores + genres
    merged = fma_scores.merge(fma_genres[["track_id", "genre_top1", "confidence_top1"]], on="track_id", how="left")
    # Both wf_anomaly_score (FMA) and the threshold are in the SAME anomaly-score
    # units (higher = more anomalous). No sign flip needed here.
    merged["is_flagged"] = (merged["wf_anomaly_score"] > sonics_real_anomaly_threshold).astype(int)

    logger.info(
        "FMA FPR attribution: %d tracks, %d flagged (FPR=%.1f%%)",
        len(merged),
        merged["is_flagged"].sum(),
        100 * merged["is_flagged"].mean(),
    )

    genre_fpr = (
        merged.groupby("genre_top1")
        .agg(
            n=("is_flagged", "count"),
            n_flagged=("is_flagged", "sum"),
            fpr=("is_flagged", "mean"),
        )
        .sort_values("fpr", ascending=False)
        .reset_index()
    )

    logger.info(
        "\nFMA FPR by genre (top 10 highest-FPR genres):\n%s",
        genre_fpr.head(10).to_string(index=False),
    )

    # Optional covariate merge
    if fma_covariate_csv and Path(fma_covariate_csv).exists():
        cov = pd.read_csv(fma_covariate_csv, low_memory=False)
        merged = merged.merge(cov[["track_id"] + ACOUSTIC_COVARIATES], on="track_id", how="left")
        for cov_name in ACOUSTIC_COVARIATES:
            if cov_name not in merged.columns:
                continue
            flagged_mean = merged.loc[merged["is_flagged"] == 1, cov_name].mean()
            ok_mean = merged.loc[merged["is_flagged"] == 0, cov_name].mean()
            if np.isnan(flagged_mean) or np.isnan(ok_mean):
                continue
            logger.info(
                "  %s: flagged=%.3f  ok=%.3f  diff=%.3f",
                cov_name,
                flagged_mean,
                ok_mean,
                flagged_mean - ok_mean,
            )

    return genre_fpr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive confound analysis for the EnCodec+RealNVP detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--window-flow-csv",
        required=True,
        help="window_flow_eval.csv from the full SONICS run (per-track anomaly scores)",
    )
    parser.add_argument(
        "--genre-csv",
        required=True,
        help="all_genres.csv from tag_all_genres.py (per-track CLAP genre tags)",
    )
    parser.add_argument(
        "--covariate-csv",
        default=None,
        help="covariate_profiles.csv from profile_covariates.py",
    )
    parser.add_argument(
        "--fma-scores-csv",
        default=None,
        help="external_scores.csv for FMA tracks (FPR test output)",
    )
    parser.add_argument(
        "--fma-genre-csv",
        default=None,
        help="all_genres.csv for FMA tracks (can be same as --genre-csv if FMA rows are included)",
    )
    parser.add_argument(
        "--fma-covariate-csv",
        default=None,
        help="covariate_profiles.csv for FMA tracks",
    )
    parser.add_argument(
        "--score-col",
        default="encodec_wf_mean",
        help="Column in window_flow_eval.csv holding the per-track log-likelihood score "
        "(higher = more real). Will be negated for AUC.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/confound_analysis",
        help="Directory for output CSVs/JSON",
    )
    parser.add_argument(
        "--min-per-genre",
        type=int,
        default=20,
        help="Minimum real AND fake tracks per genre for genre-stratified AUC",
    )
    parser.add_argument(
        "--fpr-percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile of the SONICS real anomaly-score distribution used to "
            "calibrate the FPR threshold for FMA outlier attribution (e.g. 99 -> "
            "1%% FPR threshold on SONICS real, matching the handover's reported "
            "FPR-calibration convention). Threshold is computed dynamically from "
            "--window-flow-csv, NOT hardcoded, to avoid sign/unit ambiguity."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap iterations for AUC CIs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load window-flow scores ----------
    wf = pd.read_csv(args.window_flow_csv, low_memory=False)
    logger.info("Window-flow CSV: %d rows, cols: %s", len(wf), list(wf.columns[:10]))

    if args.score_col not in wf.columns:
        # Try to find the wf_mean column
        candidates = [c for c in wf.columns if "wf_mean" in c]
        if candidates:
            args.score_col = candidates[0]
            logger.info("score_col not found, using: %s", args.score_col)
        else:
            logger.error("Score column %s not found. Available: %s", args.score_col, list(wf.columns))
            sys.exit(1)

    wf["is_fake"] = (wf["label"] == "fake").astype(int)

    # ---------- Merge genre tags ----------
    genres_df = pd.read_csv(args.genre_csv, low_memory=False)
    genres_df["track_id"] = genres_df["track_id"].astype(str)
    wf["track_id"] = wf["track_id"].astype(str)
    df = wf.merge(
        genres_df[["track_id", "genre_top1", "confidence_top1", "clap_genre_vector"]].rename(
            columns={"clap_genre_vector": "clap_genre_vector"}
        ),
        on="track_id",
        how="left",
    )
    logger.info("After genre merge: %d rows, genre matched: %d", len(df), df["genre_top1"].notna().sum())

    # ---------- Merge covariates ----------
    if args.covariate_csv and Path(args.covariate_csv).exists():
        cov = pd.read_csv(args.covariate_csv, low_memory=False)
        cov["track_id"] = cov["track_id"].astype(str)
        avail_covs = [c for c in ACOUSTIC_COVARIATES if c in cov.columns]
        df = df.merge(cov[["track_id"] + avail_covs], on="track_id", how="left")
        logger.info("Covariates merged: %s", avail_covs)

    # =========================================================
    # ANALYSIS 1: Overall AUC with bootstrap CI
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1: Overall AUC (bootstrap 95%% CI)")
    logger.info("=" * 70)
    real_s = df.loc[df["is_fake"] == 0, args.score_col].dropna().values
    fake_s = df.loc[df["is_fake"] == 1, args.score_col].dropna().values
    if len(real_s) >= 10 and len(fake_s) >= 10:
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        scores = np.r_[-real_s, -fake_s]  # negate: higher anomaly = more fake
        overall = _bootstrap_auc(y, scores, n_boot=args.n_bootstrap)
        overall["eer"] = equal_error_rate(y, scores)
        logger.info(
            "Overall AUC: %.4f [%.4f, %.4f] (95%% CI)  EER: %.4f",
            overall["auc"],
            overall["ci_lo"],
            overall["ci_hi"],
            overall["eer"],
        )
    else:
        overall = {"auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    with open(output_dir / "overall_auc.json", "w") as f:
        json.dump(overall, f, indent=2)

    # =========================================================
    # ANALYSIS 2: Per-generator AUC with bootstrap CI
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: Per-generator AUC with bootstrap CI")
    logger.info("=" * 70)
    alg_col = "algorithm" if "algorithm" in df.columns else None
    per_gen_rows = []
    if alg_col:
        for alg in sorted(df[df["is_fake"] == 1][alg_col].dropna().unique()):
            real_s = df.loc[df["is_fake"] == 0, args.score_col].dropna().values
            fake_s = df.loc[(df["is_fake"] == 1) & (df[alg_col] == alg), args.score_col].dropna().values
            if len(real_s) < 10 or len(fake_s) < 10:
                continue
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            sc = np.r_[-real_s, -fake_s]
            boot = _bootstrap_auc(y, sc, n_boot=args.n_bootstrap)
            boot["algorithm"] = alg
            boot["eer"] = equal_error_rate(y, sc)
            per_gen_rows.append(boot)
            logger.info(
                "  %-30s  AUC=%.4f [%.4f, %.4f]  EER=%.4f",
                alg,
                boot["auc"],
                boot["ci_lo"],
                boot["ci_hi"],
                boot["eer"],
            )
    pd.DataFrame(per_gen_rows).to_csv(output_dir / "per_generator_auc_ci.csv", index=False)

    # =========================================================
    # ANALYSIS 3: Genre-stratified AUC
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3: Genre-stratified AUC")
    logger.info("=" * 70)
    genre_macro_auc = float("nan")
    genre_macro_auc_weighted = float("nan")
    if "genre_top1" in df.columns:
        genre_auc = genre_stratified_auc(df, args.score_col, min_per_class=args.min_per_genre)
        genre_auc.to_csv(output_dir / "genre_stratified_auc.csv", index=False)
        if len(genre_auc):
            # Macro-average AUC: every genre bucket weighted equally, regardless of how
            # many tracks it has. This is a confound-robust headline number — the pooled
            # AUC from Analysis 1 is dominated by heavily-represented "easy" genres
            # (e.g. grunge/doom metal at ~10k tracks) and can look better than the
            # detector's *typical* per-genre performance actually is.
            genre_macro_auc = float(genre_auc["auc"].mean())
            n_tot = genre_auc["n_real"] + genre_auc["n_fake"]
            genre_macro_auc_weighted = float(np.average(genre_auc["auc"], weights=n_tot))
            logger.info(
                "Genre macro-avg AUC (unweighted across %d genres): %.4f  "
                "(vs. pooled AUC %.4f from Analysis 1 — a gap here means pooled AUC is "
                "inflated by over-represented easy genres)",
                len(genre_auc),
                genre_macro_auc,
                overall.get("auc", float("nan")),
            )
    else:
        logger.warning("No genre_top1 column — skipping genre-stratified analysis (run tag_all_genres.py first)")

    # =========================================================
    # ANALYSIS 4: Covariate / genre residualization
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 4: Covariate / genre residualization (partial AUC)")
    logger.info("=" * 70)
    resid_results = residualization_analysis(df, args.score_col)
    pd.DataFrame(resid_results).to_csv(output_dir / "residualization_results.csv", index=False)

    # =========================================================
    # ANALYSIS 5: Genre-only classifier baseline
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 5: Genre-only classifier (CLAP posterior -> real/fake)")
    logger.info("=" * 70)
    genre_clf_result = genre_only_classifier(df)
    with open(output_dir / "genre_only_classifier.json", "w") as f:
        json.dump(genre_clf_result, f, indent=2)

    # =========================================================
    # ANALYSIS 6: FMA FPR outlier attribution
    # =========================================================
    if args.fma_scores_csv and Path(args.fma_scores_csv).exists():
        logger.info("\n" + "=" * 70)
        logger.info("ANALYSIS 6: FMA FPR outlier attribution by genre")
        logger.info("=" * 70)
        fma_scores = pd.read_csv(args.fma_scores_csv, low_memory=False)
        fma_scores["track_id"] = fma_scores["track_id"].astype(str)

        fma_genres_path = args.fma_genre_csv or args.genre_csv
        fma_genres = (
            pd.read_csv(fma_genres_path, low_memory=False) if Path(fma_genres_path).exists() else pd.DataFrame()
        )
        if not fma_genres.empty:
            fma_genres = fma_genres[fma_genres["label"].isin(["real", "fma_real"])].copy()
            fma_genres["track_id"] = fma_genres["track_id"].astype(str)

        # Compute the anomaly-score threshold DYNAMICALLY from the SONICS real
        # distribution in `df` (already in wf_anomaly_score-equivalent units:
        # anomaly = -wf_mean). This avoids any sign/unit ambiguity — both the
        # threshold and the FMA scores being compared are anomaly scores where
        # higher = more anomalous.
        sonics_real_anomaly = -df.loc[df["is_fake"] == 0, args.score_col].dropna().values
        if len(sonics_real_anomaly) >= 10:
            sonics_threshold = float(np.percentile(sonics_real_anomaly, args.fpr_percentile))
            logger.info(
                "SONICS real anomaly-score threshold at %.1f-th percentile "
                "(-> %.1f%% nominal FPR on SONICS real itself): %.4f",
                args.fpr_percentile,
                100 - args.fpr_percentile,
                sonics_threshold,
            )
        else:
            logger.warning("Not enough SONICS real scores to calibrate FPR threshold; skipping.")
            sonics_threshold = float("nan")

        genre_fpr = fma_outlier_attribution(
            fma_scores,
            fma_genres,
            sonics_real_anomaly_threshold=sonics_threshold,
            fma_covariate_csv=args.fma_covariate_csv,
        )
        if not genre_fpr.empty:
            genre_fpr.to_csv(output_dir / "fma_fpr_by_genre.csv", index=False)

    # =========================================================
    # Summary verdict
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFOUND ANALYSIS SUMMARY")
    logger.info("=" * 70)
    verdict = {
        "overall_auc": overall.get("auc"),
        "genre_macro_auc": genre_macro_auc,
        "genre_macro_auc_weighted": genre_macro_auc_weighted,
        "genre_only_auc": genre_clf_result.get("genre_only_auc"),
        "is_genre_confounded": genre_clf_result.get("genre_only_auc", 0) > 0.70,
    }
    for res in resid_results:
        verdict[f"residual_auc_{res['scenario']}"] = res.get("residual_auc")
        verdict[f"auc_drop_{res['scenario']}"] = res.get("auc_drop")

    if genre_clf_result.get("genre_only_auc", 0) > 0.70:
        logger.warning(
            "⚠  Genre-only AUC=%.4f > 0.70 — genre distribution differs between "
            "real and fake. Genre is a CONFOUNDER. Check residualized AUC to see if "
            "signal survives after controlling for it.",
            genre_clf_result.get("genre_only_auc", float("nan")),
        )
    else:
        logger.info(
            "✓  Genre-only AUC=%.4f ≤ 0.70 — genre does NOT strongly separate real "
            "from fake on its own. Genre confound is not the primary driver.",
            genre_clf_result.get("genre_only_auc", float("nan")),
        )

    # NOTE: auc_drop can be negative (residualizing moves the — inverted-baseline —
    # AUC *toward* 0.5, i.e. explains away signal). We need the largest *magnitude*
    # drop across scenarios, not the algebraic max (which would silently pick the
    # least-confounded scenario and mask a large drop in, e.g., clap_full_posterior).
    max_drop = max(abs(res.get("auc_drop", 0) or 0) for res in resid_results)
    if max_drop < 0.05:
        logger.info(
            "✓  Maximum AUC drop after residualization = %.4f (< 0.05). "
            "Anomaly signal is robust to covariate/genre control.",
            max_drop,
        )
    else:
        logger.warning(
            "⚠  AUC drops by %.4f after residualization. "
            "Confounders explain some of the signal. See residualization_results.csv.",
            max_drop,
        )

    with open(output_dir / "summary_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    logger.info("\nAll outputs saved to: %s", output_dir)
    logger.info(
        "Key outputs:\n"
        "  overall_auc.json            – AUC with 95%% bootstrap CI\n"
        "  per_generator_auc_ci.csv    – per-generator AUC CI\n"
        "  genre_stratified_auc.csv    – AUC within each genre bucket\n"
        "  residualization_results.csv – partial AUC after covariate/genre control\n"
        "  genre_only_classifier.json  – genre-distribution confound test\n"
        "  fma_fpr_by_genre.csv        – (if FMA) FPR outlier attribution\n"
        "  summary_verdict.json        – machine-readable verdict"
    )


if __name__ == "__main__":
    main()
