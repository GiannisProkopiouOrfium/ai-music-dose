"""Per-algorithm specialised logistic regression classifiers.

Trains one classifier per generator on real vs that algorithm's fake tracks,
using the existing ablation_features.csv. Compares specialised vs universal AUC.

Usage:
  poetry run python scripts/run_per_algorithm_classifiers.py \
    --features-csv data/processed/sonics_mert_temporal_all/ablation_features.csv \
    --output-dir data/processed/per_algorithm_classifiers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )


def _cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[0] < 20 or len(np.unique(y)) < 2:
        return np.nan
    pipe = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def _cv_interpretability(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    if X.shape[0] < 20:
        return {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_coefs = []
    for tr, _ in cv.split(X, y):
        p = _build_pipeline()
        p.fit(X[tr], y[tr])
        fold_coefs.append(p.named_steps["clf"].coef_.ravel().tolist())
    coef_arr = np.array(fold_coefs)
    return {
        "feature_names": feature_names,
        "coef_mean": coef_arr.mean(axis=0).tolist(),
        "coef_std": coef_arr.std(axis=0).tolist(),
        "coef_sign_consistency": (np.sign(coef_arr) == np.sign(coef_arr.mean(axis=0))).mean(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-dir", default="data/processed/per_algorithm_classifiers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv, low_memory=False)
    real = df[df["label"] == "real"].copy()

    # Detect available feature columns
    emb_prefix = next((p for p in ("mert_", "encodec_") if any(c.startswith(p) for c in df.columns)), "")
    static_full_cols = [c for c in df.columns if c.startswith(f"{emb_prefix}id_") and c.endswith("_full")]
    multiscale_cols = [c for c in df.columns if c.startswith(f"{emb_prefix}id_")]
    temporal_cols = [c for c in df.columns if c.startswith(f"{emb_prefix}temporal_") and not c.endswith("_n_windows")]
    all_cols = multiscale_cols + temporal_cols

    bundles = {
        "static_full": static_full_cols,
        "multiscale": multiscale_cols,
        "temporal": temporal_cols,
        "all_features": all_cols,
    }
    # drop empty bundles
    bundles = {k: v for k, v in bundles.items() if v}

    algorithms = sorted(a for a in df["algorithm"].dropna().unique() if str(a).strip())
    print(f"Algorithms: {algorithms}")
    print(f"Bundles:    {list(bundles)}\n")

    rows = []
    interp_out = {}

    for bundle_name, feat_cols in bundles.items():
        # ---- UNIVERSAL classifier (all algorithms together) ----
        fakes_all = df[df["label"] == "fake"].copy()
        sub_all = pd.concat([real, fakes_all], ignore_index=True)
        y_all = (sub_all["label"] == "fake").astype(int).to_numpy()
        X_all = sub_all[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask_all = np.isfinite(X_all).all(axis=1)
        auc_universal = _cv_auc(X_all[mask_all], y_all[mask_all])
        rows.append(
            {
                "algorithm": "UNIVERSAL",
                "bundle": bundle_name,
                "n_real": int(real.shape[0]),
                "n_fake": int(fakes_all.shape[0]),
                "n_total": int(mask_all.sum()),
                "auc": auc_universal,
            }
        )
        print(f"[{bundle_name}] UNIVERSAL  n={mask_all.sum()}  AUC={auc_universal:.3f}")

        # ---- PER-ALGORITHM specialised classifiers ----
        for algo in algorithms:
            fakes_algo = df[(df["label"] == "fake") & (df["algorithm"] == algo)].copy()
            sub = pd.concat([real, fakes_algo], ignore_index=True)
            y = (sub["label"] == "fake").astype(int).to_numpy()
            X = sub[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(X).all(axis=1)
            auc = _cv_auc(X[mask], y[mask])
            delta = (auc - auc_universal) if not np.isnan(auc_universal) else np.nan
            rows.append(
                {
                    "algorithm": algo,
                    "bundle": bundle_name,
                    "n_real": int(real.shape[0]),
                    "n_fake": int(fakes_algo.shape[0]),
                    "n_total": int(mask.sum()),
                    "auc": auc,
                    "delta_vs_universal": delta,
                }
            )
            print(f"  [{bundle_name}] {algo:<22}  n={mask.sum()}  AUC={auc:.3f}  Δ={delta:+.3f}")

            # Interpretability for the best bundle only (avoid output bloat)
            if bundle_name == "all_features":
                usable = [c for c in feat_cols if pd.to_numeric(sub[c], errors="coerce").notna().any()]
                X_u = sub[usable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                mu = np.isfinite(X_u).all(axis=1)
                interp_out[algo] = _cv_interpretability(X_u[mu], y[mu], usable)

        print()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "per_algorithm_auc.csv", index=False)
    print(f"\nSaved per_algorithm_auc.csv ({len(results_df)} rows)")

    with open(output_dir / "per_algorithm_interpretability.json", "w") as f:
        json.dump(interp_out, f, indent=2)
    print("Saved per_algorithm_interpretability.json")

    # ---- Summary: specialised gain per algorithm ----
    print("\n=== Specialisation gain (all_features bundle) ===")
    sub_df = results_df[results_df["bundle"] == "all_features"].copy()
    univ = sub_df[sub_df["algorithm"] == "UNIVERSAL"]["auc"].values[0]
    print(f"  Universal AUC: {univ:.3f}")
    for _, r in sub_df[sub_df["algorithm"] != "UNIVERSAL"].iterrows():
        marker = "↑" if r["delta_vs_universal"] > 0.02 else ("↓" if r["delta_vs_universal"] < -0.02 else "~")
        print(f"  {r['algorithm']:<22}  {r['auc']:.3f}  {r['delta_vs_universal']:+.3f} {marker}")


if __name__ == "__main__":
    main()
