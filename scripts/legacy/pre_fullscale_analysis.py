"""Pre-full-scale analysis on existing SONICS pilot outputs.

Runs high-value checks before full-scale extraction/training:
- Data/estimator audit (missing/non-finite/outliers)
- Feature-set ablations per embedding
- Cross-embedding fusion + fakeprint hybrid
- Error diagnostics (FP/FN asymmetry)
- Stratified checks (source, fake_label)
- Cross-generator zero-shot (Suno -> Udio and reverse)
- Logistic coefficient interpretability (mean/std across CV folds)
- Upper-bound plot (AUC/F1 vs feature budget)

Usage
-----
poetry run python scripts/pre_fullscale_analysis.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("data/processed/sonics_results")
OUT_DIR = Path("reports/pre_fullscale")
FIG_DIR = OUT_DIR / "figures"

EMBEDDINGS = ["encodec", "clap", "mert", "muq"]
ID_COLS = ["id_phd", "id_twonn", "id_mle"]


@dataclass
class EvalResult:
    name: str
    n_samples: int
    accuracy: float
    f1: float
    auc: float
    fp: int
    fn: int
    tn: int
    tp: int
    fpr: float
    fnr: float


def _mkdirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_embedding_results() -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for emb in EMBEDDINGS:
        path = RESULTS_DIR / f"id_results_{emb}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Canonical target
        df["y"] = (df["label"] == "fake").astype(int)
        dfs[emb] = df
    return dfs


def load_fakeprint_results() -> pd.DataFrame | None:
    path = RESULTS_DIR / "fakeprint_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["y"] = (df["label"] == "fake").astype(int)
    return df


def audit_estimators(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for emb, df in dfs.items():
        for col in ID_COLS:
            if col not in df.columns:
                continue
            s = df[col]
            finite = np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))
            non_na = s.notna().sum()
            finite_n = int(finite.sum())
            missing = int(s.isna().sum())
            non_finite = int(non_na - finite_n)

            finite_vals = s[np.isfinite(pd.to_numeric(s, errors="coerce"))]
            q99 = float(np.nan) if len(finite_vals) == 0 else float(np.quantile(finite_vals, 0.99))
            q999 = float(np.nan) if len(finite_vals) == 0 else float(np.quantile(finite_vals, 0.999))
            rows.append(
                {
                    "embedding": emb,
                    "estimator": col.replace("id_", ""),
                    "n_total": int(len(df)),
                    "n_missing": missing,
                    "n_non_finite": non_finite,
                    "finite_rate": finite_n / max(len(df), 1),
                    "mean": float(np.nanmean(pd.to_numeric(s, errors="coerce"))),
                    "std": float(np.nanstd(pd.to_numeric(s, errors="coerce"))),
                    "q99": q99,
                    "q999": q999,
                }
            )
    out = pd.DataFrame(rows).sort_values(["embedding", "estimator"])
    out.to_csv(OUT_DIR / "audit_estimators.csv", index=False)
    return out


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ],
        memory=None,
    )


def _usable_feature_cols(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Keep only feature columns that exist and contain at least one finite value."""
    usable: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        arr = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isfinite(arr).any():
            usable.append(col)
    return usable


def evaluate_feature_set(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str = "y",
    n_splits: int = 5,
) -> EvalResult | None:
    feature_cols = _usable_feature_cols(df, feature_cols)
    if not feature_cols:
        return None

    valid = df.dropna(subset=feature_cols).copy()
    if valid.empty:
        return None

    X = valid[feature_cols].to_numpy(dtype=np.float32)
    y = valid[y_col].to_numpy(dtype=np.int64)

    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    y = y[finite_mask]

    if len(X) < n_splits * 2 or len(np.unique(y)) < 2:
        return None

    pipe = _build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tp = int(((y == 1) & (y_pred == 1)).sum())

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return EvalResult(name, len(y), float(acc), float(f1), float(auc), fp, fn, tn, tp, float(fpr), float(fnr))


def coefficient_stability(df: pd.DataFrame, feature_cols: list[str], y_col: str = "y") -> pd.DataFrame:
    feature_cols = _usable_feature_cols(df, feature_cols)
    if not feature_cols:
        return pd.DataFrame()

    valid = df.dropna(subset=feature_cols).copy()
    if valid.empty:
        return pd.DataFrame()

    X = valid[feature_cols].to_numpy(dtype=np.float32)
    y = valid[y_col].to_numpy(dtype=np.int64)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(np.unique(y)) < 2 or len(y) < 20:
        return pd.DataFrame()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    coefs = []
    for tr, _ in cv.split(X, y):
        pipe = _build_pipeline()
        pipe.fit(X[tr], y[tr])
        coef = pipe.named_steps["clf"].coef_.ravel()
        coefs.append(coef)

    coef_arr = np.vstack(coefs)
    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_mean": coef_arr.mean(axis=0),
            "coef_std": coef_arr.std(axis=0),
            "coef_sign_consistency": (np.sign(coef_arr) == np.sign(coef_arr.mean(axis=0))).mean(axis=0),
        }
    ).sort_values("coef_mean", key=np.abs, ascending=False)
    return out


def run_embedding_ablations(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []

    sets = {
        "phd": ["id_phd"],
        "twonn": ["id_twonn"],
        "mle": ["id_mle"],
        "phd_twonn": ["id_phd", "id_twonn"],
        "all_three": ["id_phd", "id_twonn", "id_mle"],
    }

    coef_artifacts: dict[str, dict[str, list[dict]]] = {}

    for emb, df in dfs.items():
        coef_artifacts[emb] = {}
        for set_name, cols in sets.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            result = evaluate_feature_set(f"{emb}:{set_name}", df, cols)
            if result is None:
                continue
            rows.append(result.__dict__)

            coef_df = coefficient_stability(df, cols)
            if not coef_df.empty:
                coef_artifacts[emb][set_name] = coef_df.to_dict(orient="records")

    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not out.empty:
        out["embedding"] = out["name"].str.split(":").str[0]
        out["feature_set"] = out["name"].str.split(":").str[1]
        out = out.sort_values(["embedding", "auc"], ascending=[True, False])
        out = out[
            [
                "embedding",
                "feature_set",
                "n_samples",
                "accuracy",
                "f1",
                "auc",
                "fp",
                "fn",
                "tn",
                "tp",
                "fpr",
                "fnr",
            ]
        ]

    out.to_csv(OUT_DIR / "ablation_per_embedding.csv", index=False)
    with open(OUT_DIR / "logreg_coefficients_per_embedding.json", "w") as f:
        json.dump(coef_artifacts, f, indent=2)

    return out


def _available_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def build_feature_bundles(dfs: dict[str, pd.DataFrame], fakeprint: pd.DataFrame | None) -> dict[str, list[str]]:
    """Construct feature bundles spanning embeddings, estimators, and fakeprint."""
    if not dfs:
        return {}

    df = next(iter(dfs.values()))
    bundles: dict[str, list[str]] = {}

    single_map = {f"single:{emb}": _available_cols(df, [f"{emb}_{c}" for c in ID_COLS]) for emb in EMBEDDINGS}
    for name, cols in single_map.items():
        if cols:
            bundles[name] = cols

    for est in ["phd", "twonn", "mle"]:
        cols = _available_cols(df, [f"{emb}_id_{est}" for emb in EMBEDDINGS])
        if cols:
            bundles[f"bundle:{est}_all_embeddings"] = cols

    stable = _available_cols(df, [f"{emb}_id_phd" for emb in EMBEDDINGS] + [f"{emb}_id_twonn" for emb in EMBEDDINGS])
    if stable:
        bundles["fusion:phd_twonn_all_embeddings"] = stable

    all_id = _available_cols(df, [f"{emb}_{est}" for emb in EMBEDDINGS for est in ID_COLS])
    if all_id:
        bundles["fusion:all_id_all_embeddings"] = all_id

    fp_cols = []
    if fakeprint is not None:
        fp_cols = _available_cols(fakeprint, ["fakeprint_mean", "fakeprint_max", "fakeprint_std", "fakeprint_n_peaks"])
    if fp_cols and stable:
        bundles["fusion:phd_twonn_all_embeddings_plus_fakeprint"] = stable + fp_cols
    if fp_cols and all_id:
        bundles["fusion:all_id_all_embeddings_plus_fakeprint"] = all_id + fp_cols

    return bundles


def make_wide_fusion_df(dfs: dict[str, pd.DataFrame], fakeprint: pd.DataFrame | None) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for emb, df in dfs.items():
        keep = ["track_id", "label", "y", "split", "source", "fake_label"] + [c for c in ID_COLS if c in df.columns]
        temp = df[keep].copy()
        rename = {c: f"{emb}_{c}" for c in ID_COLS if c in temp.columns}
        temp = temp.rename(columns=rename)

        if merged is None:
            merged = temp
        else:
            merged = merged.merge(
                temp[["track_id", "label"] + list(rename.values())],
                on=["track_id", "label"],
                how="inner",
            )

    if merged is None:
        return pd.DataFrame()

    if fakeprint is not None:
        fp_cols = ["track_id", "label", "fakeprint_mean", "fakeprint_max", "fakeprint_std", "fakeprint_n_peaks"]
        fp_cols = [c for c in fp_cols if c in fakeprint.columns]
        merged = merged.merge(fakeprint[fp_cols], on=["track_id", "label"], how="left")

    merged.to_csv(OUT_DIR / "fusion_features.csv", index=False)
    return merged


def run_fusion_evals(df_fusion: pd.DataFrame) -> pd.DataFrame:
    if df_fusion.empty:
        return pd.DataFrame()

    scenarios = build_feature_bundles({"fusion": df_fusion}, df_fusion)

    # Use a richer fusion map when the full merged table is available.
    if scenarios:
        # Keep only those bundles whose columns are actually present in the fusion table.
        scenarios = {
            name: _usable_feature_cols(df_fusion, cols)
            for name, cols in scenarios.items()
            if _usable_feature_cols(df_fusion, cols)
        }

    rows = []
    coef_payload = {}
    for name, cols in scenarios.items():
        result = evaluate_feature_set(name, df_fusion, cols)
        if result is None:
            continue
        rows.append(result.__dict__)
        coef_df = coefficient_stability(df_fusion, cols)
        if not coef_df.empty:
            coef_payload[name] = coef_df.to_dict(orient="records")

    out = pd.DataFrame(rows).sort_values("auc", ascending=False) if rows else pd.DataFrame()
    if not out.empty:
        out["feature_count"] = out["name"].map(lambda n: len(scenarios.get(n, [])))
    out.to_csv(OUT_DIR / "fusion_scenarios.csv", index=False)
    with open(OUT_DIR / "logreg_coefficients_fusion.json", "w") as f:
        json.dump(coef_payload, f, indent=2)

    # Flatten coefficient payload to a ranking table for slide-friendly feature importance.
    ranking_rows: list[dict] = []
    for scenario_name, feats in coef_payload.items():
        for item in feats:
            ranking_rows.append(
                {
                    "scenario": scenario_name,
                    "feature": item["feature"],
                    "coef_mean": item["coef_mean"],
                    "coef_abs_mean": abs(item["coef_mean"]),
                    "coef_std": item["coef_std"],
                    "coef_sign_consistency": item["coef_sign_consistency"],
                }
            )
    if ranking_rows:
        df_rank = pd.DataFrame(ranking_rows).sort_values(
            ["coef_abs_mean", "coef_sign_consistency"], ascending=[False, False]
        )
        df_rank.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

    return out


def _eval_subset(df: pd.DataFrame, name: str, cols: list[str]) -> dict | None:
    r = evaluate_feature_set(name, df, cols)
    return None if r is None else r.__dict__


def _zero_shot_score_split(
    fake: pd.DataFrame,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    stable: list[str],
    train_src: str,
    test_src: str,
) -> dict | None:
    fake_train = fake[fake["source"] == train_src]
    fake_test = fake[fake["source"] == test_src]
    if len(fake_train) < 50 or len(fake_test) < 50 or len(real_train) < 50 or len(real_test) < 50:
        return None

    train_df = pd.concat([real_train, fake_train], ignore_index=True)
    test_df = pd.concat([real_test, fake_test], ignore_index=True)

    X_train = train_df[stable].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    X_test = test_df[stable].to_numpy(dtype=np.float32)
    y_test = test_df["y"].to_numpy(dtype=np.int64)

    train_mask = np.isfinite(X_train).all(axis=1)
    test_mask = np.isfinite(X_test).all(axis=1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tp = int(((y_test == 1) & (y_pred == 1)).sum())

    return {
        "scenario": f"train_{train_src}_test_{test_src}",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def run_stratified_checks(df_fusion: pd.DataFrame) -> pd.DataFrame:
    if df_fusion.empty:
        return pd.DataFrame()

    stable = [c for c in df_fusion.columns if c.endswith(("id_phd", "id_twonn"))]
    stable = _usable_feature_cols(df_fusion, stable)
    if not stable:
        return pd.DataFrame()

    rows = []

    # Source-based subsets (real + one fake source)
    if "source" in df_fusion.columns:
        sources = sorted(x for x in df_fusion["source"].dropna().unique() if str(x).strip())
        for src in sources:
            sub = df_fusion[
                (df_fusion["label"] == "real") | ((df_fusion["label"] == "fake") & (df_fusion["source"] == src))
            ]
            res = _eval_subset(sub, f"source_subset:{src}", stable)
            if res:
                rows.append(res)

    # Fake label subsets (real + one fake type)
    if "fake_label" in df_fusion.columns:
        labels = sorted(x for x in df_fusion["fake_label"].dropna().unique() if str(x).strip())
        for fl in labels:
            sub = df_fusion[
                (df_fusion["label"] == "real") | ((df_fusion["label"] == "fake") & (df_fusion["fake_label"] == fl))
            ]
            res = _eval_subset(sub, f"fake_label_subset:{fl}", stable)
            if res:
                rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "stratified_checks.csv", index=False)
    return out


def run_cross_generator_zeroshot(df_fusion: pd.DataFrame) -> pd.DataFrame:
    if df_fusion.empty or "source" not in df_fusion.columns:
        return pd.DataFrame()

    stable = [c for c in df_fusion.columns if c.endswith(("id_phd", "id_twonn"))]
    stable = _usable_feature_cols(df_fusion, stable)
    if not stable:
        return pd.DataFrame()

    fake_sources = sorted(
        x for x in df_fusion[df_fusion["label"] == "fake"]["source"].dropna().unique() if str(x).strip()
    )
    if len(fake_sources) < 2:
        return pd.DataFrame()

    # Keep only two major sources for clarity
    major = [s for s in ["suno", "udio"] if s in fake_sources]
    if len(major) == 2:
        src_a, src_b = major
    else:
        src_a, src_b = fake_sources[:2]

    base = df_fusion.dropna(subset=stable).copy()

    # Split real songs into train/test once for fair zero-shot comparisons
    real = base[base["label"] == "real"].copy()
    fake = base[base["label"] == "fake"].copy()
    if real.empty or fake.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(real))
    cut = int(0.7 * len(idx))
    real_train = real.iloc[idx[:cut]]
    real_test = real.iloc[idx[cut:]]

    rows = []
    r1 = _zero_shot_score_split(fake, real_train, real_test, stable, src_a, src_b)
    r2 = _zero_shot_score_split(fake, real_train, real_test, stable, src_b, src_a)
    if r1:
        rows.append(r1)
    if r2:
        rows.append(r2)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "cross_generator_zeroshot.csv", index=False)
    return out


def plot_upper_bound_curve(fusion_df: pd.DataFrame) -> None:
    if fusion_df.empty:
        return

    # performance vs feature budget
    tmp = fusion_df.copy()
    tmp = tmp[tmp["feature_count"] > 0].sort_values("feature_count")
    if tmp.empty:
        return

    # Reduce duplicate feature-count collisions by keeping the best score per budget.
    tmp = (
        tmp.sort_values(["feature_count", "auc"], ascending=[True, False])
        .groupby("feature_count", as_index=False)
        .head(1)
    )

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(tmp["feature_count"], tmp["auc"], marker="o", linewidth=2, label="AUC")
    ax1.plot(tmp["feature_count"], tmp["f1"], marker="s", linewidth=2, label="F1")

    for _, r in tmp.iterrows():
        ax1.annotate(r["name"], (r["feature_count"], r["auc"]), fontsize=8, alpha=0.8)

    ax1.set_xlabel("Feature count")
    ax1.set_ylabel("Score")
    ax1.set_title("Upper bound: Performance vs feature budget")
    ax1.set_ylim(0.45, 0.9)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "upper_bound_feature_budget.png", dpi=200)
    plt.close(fig)


def write_summary(
    audit_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    fusion_df: pd.DataFrame,
    strat_df: pd.DataFrame,
    zeroshot_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("Pre-full-scale Analysis Summary")
    lines.append("")

    if not audit_df.empty:
        lines.append("Estimator audit")
        lines.append(f"- Rows: {len(audit_df)}")
        low_finite = audit_df[audit_df["finite_rate"] < 0.95]
        lines.append(f"- Estimator cells with finite_rate < 0.95: {len(low_finite)}")
        lines.append("")

    if not ablation_df.empty:
        lines.append("Best per-embedding ablations (by AUC)")
        best = ablation_df.sort_values("auc", ascending=False).groupby("embedding", as_index=False).head(1)
        for _, r in best.iterrows():
            lines.append(
                f"- {r['embedding']}: {r['feature_set']} | AUC={r['auc']:.3f} F1={r['f1']:.3f} "
                f"FPR={r['fpr']:.3f} FNR={r['fnr']:.3f}"
            )
        lines.append("")

    if not fusion_df.empty:
        lines.append("Fusion scenarios (top 5 by AUC)")
        top = fusion_df.sort_values("auc", ascending=False).head(5)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['name']}: features={int(r.get('feature_count', 0))} | AUC={r['auc']:.3f}, F1={r['f1']:.3f}, FPR={r['fpr']:.3f}, FNR={r['fnr']:.3f}"
            )
        lines.append("")

    if not strat_df.empty:
        lines.append("Stratified checks")
        for _, r in strat_df.sort_values("auc", ascending=False).head(8).iterrows():
            lines.append(f"- {r['name']}: AUC={r['auc']:.3f}, F1={r['f1']:.3f}")
        lines.append("")

    if not zeroshot_df.empty:
        lines.append("Cross-generator zero-shot")
        for _, r in zeroshot_df.iterrows():
            lines.append(
                f"- {r['scenario']}: AUC={r['auc']:.3f}, F1={r['f1']:.3f}, FPR={r['fpr']:.3f}, FNR={r['fnr']:.3f}"
            )
        lines.append("")

    (OUT_DIR / "summary.txt").write_text("\n".join(lines))


def main() -> None:
    _mkdirs()
    dfs = load_embedding_results()
    fakeprint = load_fakeprint_results()

    if not dfs:
        raise SystemExit("No id_results_*.csv found in data/processed/sonics_results")

    audit_df = audit_estimators(dfs)
    ablation_df = run_embedding_ablations(dfs)

    fusion_features = make_wide_fusion_df(dfs, fakeprint)
    fusion_eval = run_fusion_evals(fusion_features)

    strat_df = run_stratified_checks(fusion_features)
    zeroshot_df = run_cross_generator_zeroshot(fusion_features)

    plot_upper_bound_curve(fusion_eval)
    write_summary(audit_df, ablation_df, fusion_eval, strat_df, zeroshot_df)

    print("Saved outputs to:", OUT_DIR)
    print("- audit_estimators.csv")
    print("- ablation_per_embedding.csv")
    print("- fusion_features.csv")
    print("- fusion_scenarios.csv")
    print("- stratified_checks.csv")
    print("- cross_generator_zeroshot.csv")
    print("- logreg_coefficients_per_embedding.json")
    print("- logreg_coefficients_fusion.json")
    print("- summary.txt")
    print("- figures/upper_bound_feature_budget.png")


if __name__ == "__main__":
    main()
