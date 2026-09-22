"""eval_extended_features.py — Extended-feature classification & generator-shift eval.

Merges the base ablation features (ablation_features.csv) with the
changepoint / HMM extensions produced by analyze_temporal_anomalies.py,
then re-runs the *exact* bundle-classification logic used in
run_balanced_ablation.py so results are directly comparable.

New bundles added on top of the original ablation:
  temporal:<emb>:cp            changepoint features only  (cp_*)
  temporal:<emb>:hmm           HMM features only          (hmm_*)
  temporal:<emb>:all_trajectory summary + trans + hmm + cp  (was skipped
                                when original ablation ran without hmmlearn/ruptures)
  combined:<emb>               static_full + all_trajectory  (updated)
  fusion:all                   all usable columns            (updated)

Outputs (written to --output-dir):
  classification_extended.csv       per-bundle CV metrics, sorted by AUC
  stratified_eval_extended.csv      per-algorithm AUC with the best bundle
  generator_shift_extended.csv      leave-one-generator-out AUC
  interpretability_extended.json    per-bundle logistic-regression coefficients
  incremental_auc_table.csv         incremental AUC table: +cp, +hmm vs baseline
  merge_stats.txt                   merge diagnostics

Usage (on EC2):
  poetry run python scripts/eval_extended_features.py \\
    --features-csv  data/processed/mert_subsample_lp8k_d55/ablation_features.csv \\
    --phd-anomaly-csv   data/processed/mert_subsample_lp8k_d55/temporal_anomalies/temporal_anomalies_phd.csv \\
    --twonn-anomaly-csv data/processed/mert_subsample_lp8k_d55/temporal_anomalies_twonn/temporal_anomalies_twonn.csv \\
    --output-dir    data/processed/mert_subsample_lp8k_d55/extended_eval \\
    --exclude-algorithms udio-30s
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: feature name registry (matches run_balanced_ablation.py exactly)
# ---------------------------------------------------------------------------

TEMPORAL_STATS = ["mean", "std", "slope", "range", "first_last"]
TRANS_STATS = ["trans_entropy_flat", "trans_entropy_conditional", "trans_self_prob", "trans_n_distinct"]
HMM_STATS = [
    "hmm_n_states_bic",
    "hmm_log_likelihood",
    "hmm_bic",
    "hmm_viterbi_entropy",
    "hmm_dwell_mean",
    "hmm_dwell_cv",
]
CP_STATS = ["cp_n_changepoints", "cp_mean_segment_len", "cp_first_change_rel", "cp_last_change_rel", "cp_density"]

# Cols to import from anomaly CSVs (unprefixed names)
ANOMALY_IMPORT_COLS = CP_STATS + HMM_STATS


# ---------------------------------------------------------------------------
# Pipeline helpers  (identical to run_balanced_ablation.py)
# ---------------------------------------------------------------------------


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ],
        memory=None,
    )


def _usable_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(arr).any():
            out.append(c)
    return out


def _eval_bundle(name: str, df: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    """Stratified 5-fold CV + per-fold coefficient extraction (mirrors ablation)."""
    feature_cols = _usable_cols(df, feature_cols)
    if not feature_cols:
        return None

    valid = df.dropna(subset=feature_cols).copy()
    X = valid[feature_cols].to_numpy(dtype=np.float32)
    y = (valid["label"] == "fake").astype(int).to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 20 or len(np.unique(y)) < 2:
        return None

    pipe = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tp = int(((y == 1) & (y_pred == 1)).sum())

    fold_coefs: list[list[float]] = []
    for tr, _ in cv.split(X, y):
        pf = _build_pipeline()
        pf.fit(X[tr], y[tr])
        fold_coefs.append(pf.named_steps["clf"].coef_.ravel().tolist())

    coef_arr = np.array(fold_coefs)
    return {
        "name": name,
        "n_samples": int(len(y)),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "auc": float(roc_auc_score(y, y_prob)),
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp": tp,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "coef_mean": coef_arr.mean(axis=0).tolist(),
        "coef_std": coef_arr.std(axis=0).tolist(),
        "coef_sign_consistency": (np.sign(coef_arr) == np.sign(coef_arr.mean(axis=0))).mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Bundle definition  (matches + extends run_classification_and_eval)
# ---------------------------------------------------------------------------


def _define_bundles(
    merged: pd.DataFrame,
    emb_names: list[str],
    estimators: list[str],
) -> dict[str, list[str]]:
    """Return ordered bundle dict matching ablation naming exactly, plus new cp/hmm bundles."""
    bundles: dict[str, list[str]] = {}
    scales = ["full", "half", "quarter"]

    for emb in emb_names:
        # ── static ──────────────────────────────────────────────────────────
        static_full = _usable_cols(merged, [f"{emb}_id_{est}_full" for est in estimators])
        if static_full:
            bundles[f"static:{emb}:full"] = static_full

        static_multi = _usable_cols(merged, [f"{emb}_id_{est}_{sc}" for est in estimators for sc in scales])
        if len(static_multi) > len(static_full):
            bundles[f"static:{emb}:multiscale"] = static_multi

        # ── temporal summary ─────────────────────────────────────────────────
        temporal = _usable_cols(
            merged, [f"{emb}_temporal_{est}_{stat}" for est in estimators for stat in TEMPORAL_STATS]
        )
        if temporal:
            bundles[f"temporal:{emb}"] = temporal

        # ── transition entropy ───────────────────────────────────────────────
        trans = _usable_cols(merged, [f"{emb}_temporal_{est}_{s}" for est in estimators for s in TRANS_STATS])
        if trans:
            bundles[f"temporal:{emb}:transition"] = trans
            temporal_plus_trans = _usable_cols(merged, temporal + trans)
            if len(temporal_plus_trans) > len(temporal):
                bundles[f"temporal:{emb}:extended"] = temporal_plus_trans

        # ── changepoint only (new) ───────────────────────────────────────────
        cp = _usable_cols(merged, [f"{emb}_temporal_{est}_{s}" for est in estimators for s in CP_STATS])
        if cp:
            bundles[f"temporal:{emb}:cp"] = cp

        # ── HMM only (new) ───────────────────────────────────────────────────
        hmm = _usable_cols(merged, [f"{emb}_temporal_{est}_{s}" for est in estimators for s in HMM_STATS])
        if hmm:
            bundles[f"temporal:{emb}:hmm"] = hmm

        # ── summary + cp (key incremental test) ─────────────────────────────
        if cp:
            temporal_plus_cp = _usable_cols(merged, temporal + cp)
            if len(temporal_plus_cp) > len(temporal):
                bundles[f"temporal:{emb}:summary_cp"] = temporal_plus_cp

        # ── all trajectory (summary + trans + hmm + cp) ──────────────────────
        all_trajectory = _usable_cols(merged, temporal + trans + hmm + cp)
        if len(all_trajectory) > len(temporal):
            bundles[f"temporal:{emb}:all_trajectory"] = all_trajectory
        # ── model-agnostic geometric window descriptors (non-ID) ──────────
        wd_all = _usable_cols(merged, [c for c in merged.columns if c.startswith(f"{emb}_wd_")])
        wd_geom = _usable_cols(merged, [c for c in merged.columns if c.startswith(f"{emb}_wd_") and "_traj_" not in c])
        wd_traj = _usable_cols(merged, [c for c in merged.columns if c.startswith(f"{emb}_wd_traj_")])
        if wd_geom:
            bundles[f"descriptors:{emb}:geometry"] = wd_geom
        if wd_traj:
            bundles[f"descriptors:{emb}:trajectory"] = wd_traj
        if wd_all:
            bundles[f"descriptors:{emb}"] = wd_all
            id_plus_desc = _usable_cols(merged, temporal + wd_all)
            if len(id_plus_desc) > len(wd_all):
                bundles[f"temporal+descriptors:{emb}"] = id_plus_desc
        # ── combined (static full + best temporal set) ───────────────────────
        best_temporal = (
            all_trajectory if all_trajectory else (temporal + trans + hmm + cp if (trans or hmm or cp) else temporal)
        )
        combined = _usable_cols(merged, static_full + best_temporal)
        if combined and best_temporal:
            bundles[f"combined:{emb}"] = combined

    # ── cross-embedding fusions ──────────────────────────────────────────────
    fusion_full = _usable_cols(merged, [f"{emb}_id_{est}_full" for emb in emb_names for est in estimators])
    if fusion_full:
        bundles["fusion:static:all_emb"] = fusion_full

    fusion_multi = _usable_cols(
        merged, [f"{emb}_id_{est}_{sc}" for emb in emb_names for est in estimators for sc in scales]
    )
    if len(fusion_multi) > len(fusion_full):
        bundles["fusion:multiscale:all_emb"] = fusion_multi

    all_temporal = _usable_cols(
        merged, [f"{emb}_temporal_{est}_{stat}" for emb in emb_names for est in estimators for stat in TEMPORAL_STATS]
    )
    if all_temporal:
        bundles["fusion:temporal:all_emb"] = all_temporal

    all_trans = _usable_cols(
        merged, [f"{emb}_temporal_{est}_{s}" for emb in emb_names for est in estimators for s in TRANS_STATS]
    )
    fusion_extended = _usable_cols(merged, all_temporal + all_trans)
    if len(fusion_extended) > len(all_temporal):
        bundles["fusion:temporal:extended"] = fusion_extended

    all_descriptors = _usable_cols(merged, [c for c in merged.columns if "_wd_" in c])
    if all_descriptors:
        bundles["fusion:descriptors:all_emb"] = all_descriptors

    all_feat = _usable_cols(merged, list({c for cols in bundles.values() for c in cols}))
    if all_feat:
        bundles["fusion:all"] = all_feat

    return bundles


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


def run_classification(
    merged: pd.DataFrame,
    emb_names: list[str],
    estimators: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Evaluate all bundles, save results. Returns (clf_df, bundles_dict)."""
    bundles = _define_bundles(merged, emb_names, estimators)
    rows: list[dict] = []
    interpretability: dict[str, dict] = {}

    for bundle_name, cols in bundles.items():
        res = _eval_bundle(bundle_name, merged, cols)
        if res is None:
            logger.warning("  [skip] %s — not enough data or features", bundle_name)
            continue
        interpretability[bundle_name] = {
            k: res[k] for k in ("feature_names", "coef_mean", "coef_std", "coef_sign_consistency")
        }
        rows.append(
            {
                k: v
                for k, v in res.items()
                if k not in ("coef_mean", "coef_std", "coef_sign_consistency", "feature_names")
            }
        )
        logger.info(
            "  [bundle] %-52s  AUC=%.3f  F1=%.3f  n=%d",
            bundle_name,
            res["auc"],
            res["f1"],
            res["n_samples"],
        )

    clf_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    clf_df.to_csv(output_dir / "classification_extended.csv", index=False)
    with open(output_dir / "interpretability_extended.json", "w") as f:
        json.dump(interpretability, f, indent=2)
    logger.info("Saved classification_extended.csv (%d bundles)", len(clf_df))

    return clf_df, bundles


# ---------------------------------------------------------------------------
# Stratified eval (per-algorithm AUC with best bundle)
# ---------------------------------------------------------------------------


def run_stratified_eval(
    merged: pd.DataFrame,
    best_bundle_name: str,
    best_cols: list[str],
    output_dir: Path,
) -> None:
    strat_rows: list[dict] = []
    for dim, col in [("source", "source"), ("fake_type", "fake_label"), ("algorithm", "algorithm")]:
        if col not in merged.columns:
            continue
        for val in sorted(v for v in merged[col].dropna().unique() if str(v).strip()):
            sub = merged[(merged["label"] == "real") | ((merged["label"] == "fake") & (merged[col] == val))]
            res = _eval_bundle(f"{dim}:{val}", sub, best_cols)
            if res:
                strat_rows.append(
                    {
                        "dimension": dim,
                        "value": val,
                        "n_real": int((sub["label"] == "real").sum()),
                        "n_fake": int((sub["label"] == "fake").sum()),
                        "best_bundle": best_bundle_name,
                        "auc": res["auc"],
                        "f1": res["f1"],
                        "fpr": res["fpr"],
                        "fnr": res["fnr"],
                        "accuracy": res["accuracy"],
                    }
                )
                logger.info(
                    "  strat [%s=%s]  n_real=%d n_fake=%d  AUC=%.3f",
                    dim,
                    val,
                    strat_rows[-1]["n_real"],
                    strat_rows[-1]["n_fake"],
                    res["auc"],
                )

    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(output_dir / "stratified_eval_extended.csv", index=False)
        logger.info("Saved stratified_eval_extended.csv (%d rows)", len(strat_rows))


# ---------------------------------------------------------------------------
# Generator-shift eval
# ---------------------------------------------------------------------------


def run_generator_shift(
    merged: pd.DataFrame,
    emb_names: list[str],
    estimators: list[str],
    output_dir: Path,
) -> None:
    if "algorithm" not in merged.columns:
        logger.warning("No 'algorithm' column — skipping generator-shift eval")
        return

    # Build the same bundles the ablation uses for generator-shift
    all_bundles: dict[str, list[str]] = {}
    for emb in emb_names:
        for est in estimators:
            prefix = f"{emb}_"
            static_cols = [c for c in merged.columns if c.startswith(f"{prefix}id_{est}_")]
            if static_cols:
                all_bundles[f"{emb}_{est}_static"] = static_cols

            temporal_summary = [
                c
                for c in merged.columns
                if c.startswith(f"{prefix}temporal_{est}_")
                and "window_ids" not in c
                and not any(c.endswith(s) for s in TRANS_STATS + HMM_STATS + CP_STATS)
                and "trans_" not in c
                and "hmm_" not in c
                and "cp_" not in c
            ]
            if temporal_summary:
                all_bundles[f"{emb}_{est}_temporal"] = static_cols + temporal_summary

            # NEW: temporal + changepoint
            cp_cols = [
                c
                for c in merged.columns
                if c.startswith(f"{prefix}temporal_{est}_") and any(c.endswith(s) for s in CP_STATS)
            ]
            if temporal_summary and cp_cols:
                all_bundles[f"{emb}_{est}_temporal_cp"] = static_cols + temporal_summary + cp_cols

            # NEW: all trajectory
            all_traj = [c for c in merged.columns if c.startswith(f"{prefix}temporal_{est}_") and "window_ids" not in c]
            if all_traj:
                all_bundles[f"{emb}_{est}_all_trajectory"] = static_cols + all_traj

    algorithms = sorted(str(a) for a in merged["algorithm"].dropna().unique() if str(a).strip())
    shift_rows: list[dict] = []

    for held_out in algorithms:
        test_mask = (merged["label"] == "fake") & (merged["algorithm"] == held_out)
        train_mask = (merged["label"] == "real") | ((merged["label"] == "fake") & (merged["algorithm"] != held_out))
        if test_mask.sum() < 5:
            continue

        test_df = pd.concat([merged[test_mask], merged[merged["label"] == "real"]], ignore_index=True)
        train_df = merged[train_mask]

        for bundle_name, feat_cols in all_bundles.items():
            available = [c for c in feat_cols if c in merged.columns]
            if not available:
                continue

            X_tr = train_df[available].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            y_tr = (train_df["label"] == "fake").astype(int).to_numpy()
            X_te = test_df[available].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            y_te = (test_df["label"] == "fake").astype(int).to_numpy()

            m_tr = np.isfinite(X_tr).all(axis=1)
            m_te = np.isfinite(X_te).all(axis=1)

            if m_tr.sum() < 20 or m_te.sum() < 5 or len(np.unique(y_tr[m_tr])) < 2:
                continue
            try:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, C=0.1)),
                    ]
                )
                pipe.fit(X_tr[m_tr], y_tr[m_tr])
                probs = pipe.predict_proba(X_te[m_te])[:, 1]
                y_t = y_te[m_te]
                if len(np.unique(y_t)) < 2:
                    continue
                auc = roc_auc_score(y_t, probs)
                preds = (probs >= 0.5).astype(int)
                shift_rows.append(
                    {
                        "held_out_generator": held_out,
                        "bundle": bundle_name,
                        "n_train": int(m_tr.sum()),
                        "n_test": int(m_te.sum()),
                        "auc": auc,
                        "f1": float(f1_score(y_t, preds, zero_division=0)),
                    }
                )
            except Exception as exc:
                logger.debug("Generator-shift failed %s / %s: %s", held_out, bundle_name, exc)

    if shift_rows:
        shift_df = pd.DataFrame(shift_rows)
        shift_df.to_csv(output_dir / "generator_shift_extended.csv", index=False)
        logger.info("Saved generator_shift_extended.csv (%d rows)", len(shift_rows))
        logger.info("=== Cross-generator zero-shot AUC summary ===")
        for gen, grp in shift_df.groupby("held_out_generator"):
            best_row = grp.loc[grp["auc"].idxmax()]
            logger.info(
                "  held-out %-20s  best_AUC=%.3f  bundle=%s",
                gen,
                best_row["auc"],
                best_row["bundle"],
            )

        # Per-bundle mean AUC across generators
        logger.info("=== Generator-shift mean AUC per bundle ===")
        bundle_summary = shift_df.groupby("bundle")["auc"].mean().sort_values(ascending=False)
        for bname, mauc in bundle_summary.items():
            logger.info("  %-40s  mean_AUC=%.3f", bname, mauc)


# ---------------------------------------------------------------------------
# Incremental AUC table
# ---------------------------------------------------------------------------


def print_incremental_table(clf_df: pd.DataFrame) -> pd.DataFrame:
    """Print and return a compact table of AUC by bundle family."""
    key_bundles = [
        # Original ablation baselines
        "static:mert-95m:full",
        "static:mert-95m:multiscale",
        "temporal:mert-95m",  # summary (10 cols)
        "temporal:mert-95m:transition",  # trans only
        "temporal:mert-95m:extended",  # summary + trans
        # New
        "temporal:mert-95m:cp",  # cp only
        "temporal:mert-95m:hmm",  # hmm only
        "temporal:mert-95m:summary_cp",  # summary + cp   ← KEY
        "temporal:mert-95m:all_trajectory",  # summary + trans + hmm + cp
        "combined:mert-95m",  # static + all_trajectory
        "fusion:all",  # everything
    ]
    rows = []
    for bname in key_bundles:
        row = clf_df[clf_df["name"] == bname]
        if row.empty:
            rows.append({"bundle": bname, "AUC": "—", "n": "—", "n_features": "—"})
        else:
            r = row.iloc[0]
            rows.append(
                {
                    "bundle": bname,
                    "AUC": f"{r['auc']:.3f}",
                    "n": int(r["n_samples"]),
                    "n_features": int(r["n_features"]),
                }
            )
    inc_df = pd.DataFrame(rows)
    logger.info("\n=== Incremental AUC table ===\n%s", inc_df.to_string(index=False))
    return inc_df


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def _rename_anomaly_cols(
    df: pd.DataFrame,
    emb_prefix: str,
    estimator: str,
    cols_to_import: list[str],
) -> pd.DataFrame:
    """Rename unprefixed anomaly columns → {emb_prefix}_temporal_{estimator}_{col}."""
    rename_map = {col: f"{emb_prefix}_temporal_{estimator}_{col}" for col in cols_to_import if col in df.columns}
    return df[["track_id"] + list(rename_map.keys())].rename(columns=rename_map)


def build_merged_df(
    features_csv: Path,
    phd_anomaly_csv: Path | None,
    twonn_anomaly_csv: Path | None,
    exclude_algorithms: list[str],
    emb_prefix: str = "mert-95m",
) -> pd.DataFrame:
    logger.info("Loading %s", features_csv)
    df = pd.read_csv(features_csv)
    logger.info("  %d tracks, %d cols", len(df), len(df.columns))

    # Merge PHD anomaly extensions
    if phd_anomaly_csv and phd_anomaly_csv.exists():
        phd = pd.read_csv(phd_anomaly_csv)
        avail = [c for c in ANOMALY_IMPORT_COLS if c in phd.columns]
        missing = [c for c in ANOMALY_IMPORT_COLS if c not in phd.columns]
        if missing:
            logger.warning("PHD anomaly CSV missing cols: %s", missing)
        ext = _rename_anomaly_cols(phd, emb_prefix, "phd", avail)
        before = len(df)
        df = df.merge(ext, on="track_id", how="left")
        n_merged = df[ext.columns.drop("track_id", errors="ignore")[0]].notna().sum() if len(ext.columns) > 1 else 0
        logger.info(
            "  PHD anomaly merge: %d/%d tracks got cp/hmm features (%d new cols)",
            n_merged,
            before,
            len(avail),
        )
    else:
        logger.warning("PHD anomaly CSV not found or not provided — skipping PHD cp/hmm features")

    # Merge TwoNN anomaly extensions
    if twonn_anomaly_csv and twonn_anomaly_csv.exists():
        twonn = pd.read_csv(twonn_anomaly_csv)
        avail = [c for c in ANOMALY_IMPORT_COLS if c in twonn.columns]
        missing = [c for c in ANOMALY_IMPORT_COLS if c not in twonn.columns]
        if missing:
            logger.warning("TwoNN anomaly CSV missing cols: %s", missing)
        ext = _rename_anomaly_cols(twonn, emb_prefix, "twonn", avail)
        before = len(df)
        df = df.merge(ext, on="track_id", how="left")
        n_merged = df[ext.columns.drop("track_id", errors="ignore")[0]].notna().sum() if len(ext.columns) > 1 else 0
        logger.info(
            "  TwoNN anomaly merge: %d/%d tracks got cp/hmm features (%d new cols)",
            n_merged,
            before,
            len(avail),
        )
    else:
        logger.warning("TwoNN anomaly CSV not found or not provided — skipping TwoNN cp/hmm features")

    # Exclude algorithms
    if exclude_algorithms:
        before = len(df)
        df = df[~df["algorithm"].isin(exclude_algorithms)].copy()
        logger.info(
            "Excluded %s: %d → %d tracks",
            exclude_algorithms,
            before,
            len(df),
        )

    logger.info(
        "Final dataset: %d tracks  (real=%d  fake=%d)",
        len(df),
        (df["label"] == "real").sum(),
        (df["label"] == "fake").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge extended features and re-run classification matching ablation bundles."
    )
    parser.add_argument(
        "--features-csv",
        required=True,
        type=Path,
        help="ablation_features.csv from run_balanced_ablation.py",
    )
    parser.add_argument(
        "--phd-anomaly-csv",
        type=Path,
        default=None,
        help="temporal_anomalies_phd.csv from analyze_temporal_anomalies.py (--estimator phd)",
    )
    parser.add_argument(
        "--twonn-anomaly-csv",
        type=Path,
        default=None,
        help="temporal_anomalies_twonn.csv from analyze_temporal_anomalies.py (--estimator twonn)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--exclude-algorithms",
        nargs="*",
        default=["udio-30s"],
        help="Algorithm(s) to exclude (default: udio-30s). Pass empty string to include all.",
    )
    parser.add_argument(
        "--emb-prefix",
        default="mert-95m",
        help="Embedding prefix in column names (default: mert-95m)",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["phd", "twonn"],
        help="ID estimators used (default: phd twonn)",
    )
    parser.add_argument(
        "--no-generator-shift",
        action="store_true",
        help="Skip generator-shift evaluation",
    )
    args = parser.parse_args()

    # Clean up exclude list (handle empty-string sentinel)
    exclude = [a for a in (args.exclude_algorithms or []) if a]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Merge ─────────────────────────────────────────────────────────────
    merged = build_merged_df(
        features_csv=args.features_csv,
        phd_anomaly_csv=args.phd_anomaly_csv,
        twonn_anomaly_csv=args.twonn_anomaly_csv,
        exclude_algorithms=exclude,
        emb_prefix=args.emb_prefix,
    )

    # Save merge diagnostics
    col_presence = {c: int(merged[c].notna().sum()) for c in merged.columns if merged[c].notna().sum() > 0}
    with open(args.output_dir / "merge_stats.txt", "w") as f:
        f.write(f"n_tracks: {len(merged)}\n")
        f.write(f"n_real: {int((merged['label']=='real').sum())}\n")
        f.write(f"n_fake: {int((merged['label']=='fake').sum())}\n")
        f.write(f"excluded_algorithms: {exclude}\n\n")
        f.write("Column non-null counts:\n")
        for col, cnt in sorted(col_presence.items()):
            f.write(f"  {col}: {cnt}\n")

    # ── 2. Classification ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION (5-fold stratified CV)")
    logger.info("=" * 60)
    clf_df, bundles = run_classification(merged, [args.emb_prefix], args.estimators, args.output_dir)

    # ── 3. Incremental AUC table ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("INCREMENTAL AUC TABLE")
    logger.info("=" * 60)
    inc_df = print_incremental_table(clf_df)
    inc_df.to_csv(args.output_dir / "incremental_auc_table.csv", index=False)

    # ── 4. Stratified eval ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STRATIFIED EVAL (best bundle)")
    logger.info("=" * 60)
    if not clf_df.empty:
        # Select the best bundle among those evaluated on a *comparable* sample.
        # Bundles that span many features drop more rows to NaN (smaller n), which
        # inflates AUC on an easier, non-comparable subset. Restrict candidates to
        # those retaining >=95% of the maximum sample size before ranking by AUC,
        # so a small-n fusion bundle cannot outrank a larger like-for-like one.
        max_n = int(clf_df["n_samples"].max())
        eligible = clf_df[clf_df["n_samples"] >= 0.95 * max_n]
        if eligible.empty:
            eligible = clf_df
        best_row = eligible.sort_values("auc", ascending=False).iloc[0]
        best_name = best_row["name"]
        best_cols = bundles.get(best_name, [])
        logger.info(
            "Best comparable bundle: %s (AUC=%.3f, n=%d / max n=%d)",
            best_name,
            best_row["auc"],
            int(best_row["n_samples"]),
            max_n,
        )
        run_stratified_eval(merged, best_name, best_cols, args.output_dir)

    # ── 5. Generator-shift ───────────────────────────────────────────────────
    if not args.no_generator_shift:
        logger.info("\n" + "=" * 60)
        logger.info("GENERATOR-SHIFT EVAL (leave-one-out)")
        logger.info("=" * 60)
        run_generator_shift(merged, [args.emb_prefix], args.estimators, args.output_dir)

    logger.info("\nAll outputs written to %s", args.output_dir)
    logger.info("  classification_extended.csv")
    logger.info("  stratified_eval_extended.csv")
    logger.info("  generator_shift_extended.csv")
    logger.info("  interpretability_extended.json")
    logger.info("  incremental_auc_table.csv")
    logger.info("  merge_stats.txt")


if __name__ == "__main__":
    main()
