"""End-to-end experiment runner.

Loads pre-computed ID and fakeprint features, trains classifiers
across multiple feature configurations, and produces evaluation reports.

Usage
-----
    python -m scripts.run_experiment \
        --id-results data/processed/id_results.csv \
        --fakeprints data/processed/fakeprints.npz \
        --labels data/processed/labels.csv \
        --output-dir reports/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from intrinsic_ai_music_detection.models.evaluate import (
    compare_distributions,
    compute_binary_metrics,
    format_results_table,
    generate_report,
)
from intrinsic_ai_music_detection.models.train_model import (
    ClassificationResult,
    save_model,
    train_and_evaluate,
    train_threshold_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--id-results", type=str, required=True, help="CSV with ID estimation results")
    parser.add_argument("--fakeprints", type=str, default=None, help="NPZ with fakeprint features")
    parser.add_argument("--labels", type=str, required=True, help="CSV with track_id and label columns")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    id_df = pd.read_csv(args.id_results)
    labels_df = pd.read_csv(args.labels)

    # Merge on track_id
    df = id_df.merge(labels_df[["track_id", "label"]], on="track_id", how="inner")
    logger.info("Merged dataset: %d tracks (%d human, %d AI)", len(df), (df.label == 0).sum(), (df.label == 1).sum())

    y = df["label"].values.astype(np.int64)

    # --- ID feature columns ---
    id_cols = [c for c in df.columns if c.startswith("id_")]
    results_summary: list[dict] = []

    # ---- A) Statistical tests on each ID feature ----
    stats_results = []
    for col in id_cols:
        valid = df[col].notna()
        if valid.sum() < 10:
            continue
        human_ids = df.loc[valid & (df.label == 0), col].values.astype(np.float64)
        ai_ids = df.loc[valid & (df.label == 1), col].values.astype(np.float64)
        comp = compare_distributions(human_ids, ai_ids)
        stats_results.append(
            {
                "feature": col,
                "human_mean": comp.human_mean,
                "ai_mean": comp.ai_mean,
                "cohens_d": comp.cohens_d,
                "u_test_p": comp.mann_whitney_p,
            }
        )
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_df.to_csv(output_dir / "statistical_tests.csv", index=False)
        logger.info("Statistical tests saved")

    # ---- B) Single-ID threshold classifiers ----
    for col in id_cols:
        valid = df[col].notna()
        if valid.sum() < 20:
            continue
        vals = df.loc[valid, col].values.astype(np.float64)
        labels = df.loc[valid, "label"].values.astype(np.int64)
        thresh, acc = train_threshold_classifier(vals, labels, direction="lower")
        results_summary.append(
            {
                "method": f"threshold_{col}",
                "accuracy": acc,
                "f1": float("nan"),
                "auc_roc": float("nan"),
            }
        )
        logger.info("Threshold %s: t=%.3f acc=%.3f", col, thresh, acc)

    # ---- C) Single-ID classifiers (LR + SVM) ----
    all_classification_results: list[ClassificationResult] = []
    for col in id_cols:
        valid = df[col].notna()
        if valid.sum() < 20:
            continue
        X_single = df.loc[valid, [col]].values.astype(np.float64)
        y_single = df.loc[valid, "label"].values.astype(np.int64)

        for clf_type in ["logistic_regression", "svm"]:
            result = train_and_evaluate(
                X_single,
                y_single,
                classifier_type=clf_type,
                feature_set_name=f"{col}_{clf_type}",
                n_folds=args.n_folds,
            )
            all_classification_results.append(result)
            results_summary.append(
                {
                    "method": f"{col}_{clf_type}",
                    "accuracy": result.cv_accuracy,
                    "f1": result.cv_f1,
                    "auc_roc": result.cv_auc,
                }
            )

    # ---- D) Multi-ID classifier ----
    if len(id_cols) > 1:
        valid = df[id_cols].notna().all(axis=1)
        if valid.sum() >= 20:
            X_multi = df.loc[valid, id_cols].values.astype(np.float64)
            y_multi = df.loc[valid, "label"].values.astype(np.int64)

            for clf_type in ["logistic_regression", "svm"]:
                result = train_and_evaluate(
                    X_multi,
                    y_multi,
                    classifier_type=clf_type,
                    feature_set_name=f"multi_id_{clf_type}",
                    n_folds=args.n_folds,
                )
                all_classification_results.append(result)
                results_summary.append(
                    {
                        "method": f"multi_id_{clf_type}",
                        "accuracy": result.cv_accuracy,
                        "f1": result.cv_f1,
                        "auc_roc": result.cv_auc,
                    }
                )

    # ---- E) Fakeprint classifier ----
    if args.fakeprints:
        fp_data = np.load(args.fakeprints, allow_pickle=True)
        fp_ids = fp_data["track_ids"]
        fp_features = fp_data["features"]

        fp_df = pd.DataFrame({"track_id": fp_ids})
        fp_df = fp_df.merge(labels_df[["track_id", "label"]], on="track_id", how="inner")

        # Get matching indices
        fp_mask = np.isin(fp_ids, fp_df.track_id.values)
        X_fp = fp_features[fp_mask]
        y_fp = fp_df["label"].values.astype(np.int64)

        if len(y_fp) >= 20:
            for clf_type in ["logistic_regression", "svm"]:
                result = train_and_evaluate(
                    X_fp,
                    y_fp,
                    classifier_type=clf_type,
                    feature_set_name=f"fakeprint_{clf_type}",
                    n_folds=args.n_folds,
                )
                all_classification_results.append(result)
                results_summary.append(
                    {
                        "method": f"fakeprint_{clf_type}",
                        "accuracy": result.cv_accuracy,
                        "f1": result.cv_f1,
                        "auc_roc": result.cv_auc,
                    }
                )

    # ---- F) Hybrid: ID + fakeprint ----
    if args.fakeprints and len(id_cols) > 0:
        # Merge ID features with fakeprints
        fp_ids_list = list(fp_ids)
        common_ids = set(df.track_id.values) & set(fp_ids_list)
        if len(common_ids) >= 20:
            id_sub = df[df.track_id.isin(common_ids)].sort_values("track_id")
            fp_idx = [fp_ids_list.index(tid) for tid in id_sub.track_id.values]
            fp_sub = fp_features[fp_idx]

            valid = id_sub[id_cols].notna().all(axis=1)
            if valid.sum() >= 20:
                X_hybrid = np.hstack(
                    [
                        id_sub.loc[valid, id_cols].values.astype(np.float64),
                        fp_sub[valid.values],
                    ]
                )
                y_hybrid = id_sub.loc[valid, "label"].values.astype(np.int64)

                for clf_type in ["logistic_regression", "svm"]:
                    result = train_and_evaluate(
                        X_hybrid,
                        y_hybrid,
                        classifier_type=clf_type,
                        feature_set_name=f"hybrid_{clf_type}",
                        n_folds=args.n_folds,
                    )
                    all_classification_results.append(result)
                    results_summary.append(
                        {
                            "method": f"hybrid_{clf_type}",
                            "accuracy": result.cv_accuracy,
                            "f1": result.cv_f1,
                            "auc_roc": result.cv_auc,
                        }
                    )

    # ---- Save results ----
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_dir / "experiment_results.csv", index=False)

    table = format_results_table(results_summary, caption="Experiment Results")
    (output_dir / "results_table.md").write_text(table)

    # Save best model
    if all_classification_results:
        best = max(all_classification_results, key=lambda r: r.cv_f1)
        if best.model is not None:
            save_model(best.model, output_dir / "best_model.pkl")
        logger.info("Best model: %s (F1=%.3f)", best.feature_set, best.cv_f1)

    logger.info("Experiment complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
