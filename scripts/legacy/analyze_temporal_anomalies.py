"""Temporal ID sequence analysis: anomalies, transition entropy, HMM, change-points.

Loads ablation_{emb}.csv (or ablation_features.csv) with per-window ID sequences
and computes:
  - Anomaly detection: z-score spikes, jerk
  - Transition-matrix entropy (k-means regime clustering)
  - State-space / HMM features (hmmlearn — optional)
  - Change-point detection (ruptures — optional)

All features are compared between real and fake tracks with Mann-Whitney U tests.
A logistic-regression AUC is also computed over all recovered features.

Usage
-----
# Full analysis (both PHD and TwoNN):
poetry run python scripts/analyze_temporal_anomalies.py \
  --features-csv data/processed/mert_subsample_lp8k_d55/ablation_mert-95m.csv \
  --output-dir data/processed/temporal_anomalies

# Specify estimator + exclude a confounded stratum:
poetry run python scripts/analyze_temporal_anomalies.py \
  --features-csv data/processed/mert_subsample_lp8k_d55/ablation_mert-95m.csv \
  --estimator phd \
  --exclude-algorithms udio-30s \
  --min-windows 6 \
  --output-dir data/processed/temporal_anomalies

# Quick run (skip slow HMM/changepoint):
poetry run python scripts/analyze_temporal_anomalies.py \
  --features-csv data/processed/mert_subsample_lp8k_d55/ablation_mert-95m.csv \
  --no-hmm --no-changepoint \
  --output-dir data/processed/temporal_anomalies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.temporal_features import (
    ALL_TEMPORAL_EXTENSION_NAMES,
    CHANGEPOINT_FEATURE_NAMES,
    HMM_FEATURE_NAMES,
    TRANSITION_FEATURE_NAMES,
    all_temporal_features,
)

Z_THRESHOLD = 2.0  # |z| > 2 = anomalous window


# ---------------------------------------------------------------------------
# Prefix detection
# ---------------------------------------------------------------------------


def _detect_prefix(df: pd.DataFrame, estimator: str) -> str:
    """Return the embedding-name prefix for temporal columns (e.g. 'mert-95m_').

    Scans all column names for ``*temporal_{estimator}_window_ids`` and extracts
    whatever comes before ``temporal_``.  Falls back to a list of well-known
    prefixes, then to the empty string.
    """
    suffix = f"temporal_{estimator}_window_ids"
    for col in df.columns:
        if col.endswith(suffix):
            return col[: -len(suffix)]  # everything before "temporal_"

    # Fallback: look for any temporal_ column and infer prefix
    for col in df.columns:
        for est in ("phd", "twonn", "mle"):
            s = f"temporal_{est}_"
            idx = col.find(s)
            if idx >= 0:
                return col[:idx]

    return ""


# ---------------------------------------------------------------------------
# Anomaly stats
# ---------------------------------------------------------------------------


def anomaly_stats(window_values: list, z_thresh: float = Z_THRESHOLD) -> dict:
    """Compute anomaly features from a per-window ID sequence."""
    arr = np.array([v for v in window_values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return {}

    z = np.abs(stats.zscore(arr))
    n_anomalous = int((z > z_thresh).sum())
    anomaly_rate = float(n_anomalous / len(arr))
    max_spike = float(np.max(z))
    spike_position = float(np.argmax(z) / max(len(arr) - 1, 1))

    if len(arr) >= 3:
        jerk = np.abs(np.diff(arr, n=2))
        max_jerk = float(np.max(jerk))
        mean_jerk = float(np.mean(jerk))
    else:
        max_jerk = mean_jerk = np.nan

    return {
        "n_windows": len(arr),
        "n_anomalous": n_anomalous,
        "anomaly_rate": anomaly_rate,
        "max_z_score": max_spike,
        "spike_position": spike_position,
        "max_id_jerk": max_jerk,
        "mean_id_jerk": mean_jerk,
    }


# ---------------------------------------------------------------------------
# Compare real vs fake: Mann-Whitney U test
# ---------------------------------------------------------------------------


def compare_real_fake(result_df: pd.DataFrame, features: list[str], label: str) -> None:
    """Print real-vs-fake comparison for a list of features."""
    avail = [f for f in features if f in result_df.columns]
    if not avail:
        print(f"\n=== {label} === (no columns available)")
        return
    print(f"\n=== {label} ===")
    print(f"{'feature':<35}  {'real_mean':>9}  {'fake_mean':>9}  {'p_value':>10}  status")
    print("-" * 82)
    for feat in avail:
        r = result_df[result_df["label"] == "real"][feat].dropna()
        f = result_df[result_df["label"] == "fake"][feat].dropna()
        if len(r) < 5 or len(f) < 5:
            continue
        _, p = stats.mannwhitneyu(r, f, alternative="two-sided")
        flag = "**" if p < 0.001 else ("*" if p < 0.05 else "")
        print(f"  {feat:<33}  {np.mean(r):9.4f}  {np.mean(f):9.4f}  {p:10.4e}  {flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal ID sequence analysis: anomalies, transition entropy, HMM, change-points",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-csv", required=True, help="ablation_{emb}.csv (has window_ids) or ablation_features.csv"
    )
    parser.add_argument("--estimator", default="phd", choices=["phd", "twonn", "mle"])
    parser.add_argument("--z-threshold", type=float, default=Z_THRESHOLD)
    parser.add_argument("--output-dir", default="data/processed/temporal_anomalies")
    parser.add_argument("--no-hmm", action="store_true", help="Skip HMM fitting (requires hmmlearn; slower)")
    parser.add_argument(
        "--no-changepoint", action="store_true", help="Skip change-point detection (requires ruptures; slower)"
    )
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of ID regimes for transition-matrix entropy")
    parser.add_argument(
        "--exclude-algorithms",
        nargs="+",
        default=None,
        help=(
            "Exclude specific algorithm strata from analysis. "
            "Use to remove duration-confounded strata (e.g. udio-30s) for a "
            "fair comparison. Excluded tracks are saved separately."
        ),
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=4,
        help=(
            "Skip tracks with fewer than this many valid windows. "
            "Prevents degenerate statistics from short-duration strata."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv, low_memory=False)
    print(f"Loaded {len(df)} tracks from {args.features_csv}")

    # --- Robust prefix detection ---
    emb_prefix = _detect_prefix(df, args.estimator)
    window_col = f"{emb_prefix}temporal_{args.estimator}_window_ids"
    n_windows_col = f"{emb_prefix}temporal_{args.estimator}_n_windows"

    print(f"Detected embedding prefix: '{emb_prefix}'")
    print(f"Window column: '{window_col}' — {'FOUND' if window_col in df.columns else 'NOT FOUND'}")

    if window_col not in df.columns:
        # Fallback: check if extension columns already computed
        ext_cols = [c for c in df.columns if f"{emb_prefix}temporal_{args.estimator}_trans_" in c]
        if ext_cols:
            print(f"Extension features already present ({len(ext_cols)} trans_* cols). " "Analysing directly.")
            result_df = df.copy()
        else:
            print(
                f"\nColumn '{window_col}' not found and no extension columns detected.\n"
                "Make sure to use the per-embedding CSV (ablation_{{emb}}.csv) which "
                "contains the window_ids column, not the wide ablation_features.csv."
            )
            # List candidate columns for debugging
            cands = [c for c in df.columns if "temporal" in c and "window" in c]
            if cands:
                print("Candidate columns found:", cands[:5])
            return
    else:
        # --- Excluded stratum handling ---
        excluded_df = pd.DataFrame()
        if args.exclude_algorithms:
            mask = df["algorithm"].isin(args.exclude_algorithms)
            excluded_df = df[mask].copy()
            print(f"Excluding {mask.sum()} tracks from: {args.exclude_algorithms}")
            print(
                f"  Window stats for excluded: {excluded_df[n_windows_col].describe().to_dict() if n_windows_col in excluded_df else 'n/a'}"
            )
            df = df[~mask].copy()
            print(f"Analysis set after exclusion: {len(df)} tracks")

        # --- Check HMM/changepoint deps ---
        run_hmm = not args.no_hmm
        run_changepoint = not args.no_changepoint
        if run_hmm:
            try:
                import hmmlearn  # noqa: F401
            except ImportError:
                print("WARNING: hmmlearn not installed — skipping HMM. " "Install: pip install hmmlearn")
                run_hmm = False
        if run_changepoint:
            try:
                import ruptures  # noqa: F401
            except ImportError:
                print("WARNING: ruptures not installed — skipping change-point. " "Install: pip install ruptures")
                run_changepoint = False

        rows = []
        n_skipped_short = 0
        for _, row in df.iterrows():
            raw = row.get(window_col)
            if not raw or (isinstance(raw, float) and np.isnan(raw)):
                continue
            try:
                window_values = json.loads(raw)
            except Exception:
                continue

            # Anomaly stats
            astats = anomaly_stats(window_values, args.z_threshold)
            if not astats:
                continue

            # Skip tracks with too few windows (would give degenerate stats)
            if astats["n_windows"] < args.min_windows:
                n_skipped_short += 1
                continue

            # Temporal extension features
            ext = all_temporal_features(
                window_values,
                n_clusters=args.n_clusters,
                run_hmm=run_hmm,
                run_changepoint=run_changepoint,
            )

            rows.append(
                {
                    "track_id": row.get("track_id", ""),
                    "label": row.get("label", ""),
                    "algorithm": row.get("algorithm", ""),
                    "fake_label": row.get("fake_label", ""),
                    "path": row.get("path", ""),
                    "estimator": args.estimator,
                    "window_sequence": str(window_values),
                    **astats,
                    **ext,
                }
            )

        if n_skipped_short > 0:
            print(
                f"Skipped {n_skipped_short} tracks with < {args.min_windows} windows " "(use --min-windows to adjust)"
            )

        if not rows:
            print("No window_ids data found.")
            return

        result_df = pd.DataFrame(rows)

        # Save excluded stratum separately for reference
        if not excluded_df.empty:
            excluded_df.to_csv(output_dir / f"excluded_{args.estimator}.csv", index=False)
            print(f"Excluded tracks saved: excluded_{args.estimator}.csv ({len(excluded_df)} rows)")

    # --- Save results ---
    out_csv = output_dir / f"temporal_anomalies_{args.estimator}.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  ({len(result_df)} tracks)")

    n_real = (result_df["label"] == "real").sum()
    n_fake = (result_df["label"] == "fake").sum()
    print(f"  real: {n_real}   fake: {n_fake}")

    # --- Comparison tables ---
    anomaly_features = ["n_windows", "anomaly_rate", "max_z_score", "max_id_jerk", "mean_id_jerk", "spike_position"]
    compare_real_fake(result_df, anomaly_features, f"Anomaly statistics (z_thresh={args.z_threshold:.1f})")
    compare_real_fake(result_df, TRANSITION_FEATURE_NAMES, "Transition-matrix entropy features")
    if run_hmm:
        compare_real_fake(result_df, HMM_FEATURE_NAMES, "HMM / state-space features")
    else:
        print("\n(HMM features skipped — hmmlearn not available)")
    if run_changepoint:
        compare_real_fake(result_df, CHANGEPOINT_FEATURE_NAMES, "Change-point detection features")
    else:
        print("(Change-point features skipped — ruptures not available)")

    # --- Per-algorithm summary ---
    print("\n=== Per-algorithm summary (anomaly_rate + transition entropy) ===")
    print(f"{'algorithm':<22}  {'n':>5}  {'n_windows':>9}  " f"{'anomaly_rate':>12}  {'trans_entropy':>13}")
    real_sub = result_df[result_df["label"] == "real"]
    for algo in sorted(result_df["algorithm"].dropna().unique()):
        if not str(algo).strip():
            continue
        sub = result_df[result_df["algorithm"] == algo]
        anom = sub["anomaly_rate"].mean() if "anomaly_rate" in sub else float("nan")
        trans = sub["trans_entropy_conditional"].mean() if "trans_entropy_conditional" in sub else float("nan")
        nwin = sub["n_windows"].mean() if "n_windows" in sub else float("nan")
        print(f"  {algo:<22}  {len(sub):>5}  {nwin:9.2f}  {anom:12.4f}  {trans:13.4f}")
    real_anom = real_sub["anomaly_rate"].mean() if "anomaly_rate" in real_sub else float("nan")
    real_trans = (
        real_sub["trans_entropy_conditional"].mean() if "trans_entropy_conditional" in real_sub else float("nan")
    )
    real_nwin = real_sub["n_windows"].mean() if "n_windows" in real_sub else float("nan")
    print(f"  {'[real]':<22}  {len(real_sub):>5}  {real_nwin:9.2f}  {real_anom:12.4f}  {real_trans:13.4f}")

    # --- Top anomalous tracks ---
    if "max_z_score" in result_df.columns:
        print(f"\n=== Top 10 most anomalous fake tracks ===")
        top_fake = result_df[result_df["label"] == "fake"].nlargest(10, "max_z_score")
        show_cols = [
            "track_id",
            "algorithm",
            "fake_label",
            "n_windows",
            "max_z_score",
            "anomaly_rate",
            "trans_entropy_conditional",
            "hmm_n_states_bic",
            "cp_n_changepoints",
        ]
        show_cols = [c for c in show_cols if c in top_fake.columns]
        print(top_fake[show_cols].to_string(index=False))

        print(f"\n=== Top 10 most anomalous real tracks (false-positive candidates) ===")
        top_real = result_df[result_df["label"] == "real"].nlargest(10, "max_z_score")
        show_r = [
            "track_id",
            "n_windows",
            "max_z_score",
            "anomaly_rate",
            "trans_entropy_conditional",
            "cp_n_changepoints",
        ]
        show_r = [c for c in show_r if c in top_real.columns]
        print(top_real[show_r].to_string(index=False))

    # --- Classification AUC ---
    all_feat_cols = anomaly_features + ALL_TEMPORAL_EXTENSION_NAMES
    all_feat_cols = [c for c in all_feat_cols if c in result_df.columns]
    if len(all_feat_cols) >= 3 and "label" in result_df.columns:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            y = (result_df["label"] == "fake").astype(int).to_numpy()
            X = result_df[all_feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            row_mask = np.isfinite(X).all(axis=1)
            if row_mask.sum() >= 20:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, C=0.1)),
                    ]
                )
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                probs = cross_val_predict(pipe, X[row_mask], y[row_mask], cv=cv, method="predict_proba")[:, 1]
                auc = roc_auc_score(y[row_mask], probs)
                print(
                    f"\n=== Anomaly + extension features  AUC: {auc:.3f} "
                    f"(n={row_mask.sum()}, features={len(all_feat_cols)}) ==="
                )
        except Exception as exc:
            print(f"\n(Classification failed: {exc})")

    # --- Sensitivity: excluded vs included AUC side-by-side note ---
    if args.exclude_algorithms:
        print(
            f"\nNOTE: Results above exclude {args.exclude_algorithms}. "
            "Compare against a run without --exclude-algorithms to quantify "
            "the impact of those strata on pooled metrics."
        )


if __name__ == "__main__":
    main()
