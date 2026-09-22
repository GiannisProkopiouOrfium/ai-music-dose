"""Run experiments on the SONICS dataset — full-scale edition.

Design decisions locked from pilot analysis (2026-06-11)
---------------------------------------------------------
* EnCodec + TwoNN is the strongest single-embedding feature (AUC 0.763).
* PHD + TwoNN fusion across all embeddings raises AUC to 0.812 (+4.9 pts).
* Temporal ID summaries raise AUC another ~4 pts (temporal:twonn 0.806 vs static 0.763).
* MLE is noisy and sign-inconsistent; excluded by default (--include-mle to re-enable).
* Fakeprint AUC=1.0 on pilot was an imbalance artefact; kept as a separate signal.
* All metadata (source, fake_label, split) is preserved for stratified eval.

Outputs
-------
  id_results_{emb}.csv          — per-track ID + metadata (one row per track)
  fakeprint_results.csv         — per-track fakeprint features
  fusion_features.csv           — wide-format join of all embeddings + fakeprint
  classification_{emb}.json     — CV metrics per embedding
  classification_fusion.json    — CV metrics for fusion feature sets
  interpretability.json         — logistic coef mean/std/sign_consistency per fold
  stratified_eval.csv           — per-source / per-fake-label AUC/F1/FPR/FNR
  analysis_{emb}.json           — Cohen's d, Mann-Whitney per estimator
  all_id_results.csv            — combined multi-embedding result table

Usage
-----
    # Full-scale run — all embeddings, PHD+TwoNN only, balanced stratified sample
    python scripts/run_sonics_experiment.py \\
        --embedding all --device cuda \\
        --include-fakeprint --include-mle false \\
        --stratified --output-dir data/processed/sonics_results_fullscale

    # Resume interrupted run (checkpoints survive)
    python scripts/run_sonics_experiment.py --embedding all --device cuda \\
        --output-dir data/processed/sonics_results_fullscale

    # Pilot / smoke-test
    python scripts/run_sonics_experiment.py \\
        --embedding encodec --max-real 100 --max-fake 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio
from intrinsic_ai_music_detection.features.embeddings import get_extractor
from intrinsic_ai_music_detection.features.id_estimators import estimate_all
from intrinsic_ai_music_detection.models.evaluate import cohens_d, compare_distributions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")
RESULTS_DIR = Path("data/processed/sonics_results")

# Estimators used by default; MLE excluded based on pilot sign-inconsistency findings.
DEFAULT_ESTIMATORS: list[str] = ["phd", "twonn"]
ALL_ESTIMATORS: list[str] = ["phd", "twonn", "mle"]

EMBEDDING_CONFIGS = {
    "encodec": {"target_sr": 24000, "metric": "euclidean"},
    "clap": {"target_sr": 48000, "metric": "cosine"},
    "mert": {"target_sr": 24000, "metric": "cosine"},
    "muq": {"target_sr": 24000, "metric": "cosine"},
}

# Stable feature policy from pilot: sign_consistency = 1.0 for these
STABLE_FEATURES = [
    "encodec_id_phd",
    "encodec_id_twonn",
    "mert_id_phd",
    "mert_id_twonn",
    "muq_id_twonn",
    "clap_id_twonn",
]
BUNDLE_STABLE = "fusion:phd_twonn_all"


def _row_to_real_track(row: pd.Series) -> dict | None:
    p = row.get("audio_path", "")
    if not p or not Path(p).exists():
        return None
    return {
        "path": p,
        "track_id": str(row.get("youtube_id", row.get("id", ""))),
        "split": str(row.get("split", "")),
        "label": "real",
        "source": "",
        "fake_label": "",
    }


def _row_to_fake_track(row: pd.Series) -> dict | None:
    p = row.get("audio_path", "")
    if not p or not Path(p).exists():
        return None
    return {
        "path": p,
        "track_id": str(row.get("id", row.get("filename", ""))),
        "split": str(row.get("split", "")),
        "source": str(row.get("source", "")),
        "fake_label": str(row.get("label", "full fake")),
        "algorithm": str(row.get("algorithm", "")),
        "label": "fake",
    }


def _sample_fake_stratified(fake_tracks: list[dict], max_fake: int, rng: np.random.Generator) -> list[dict]:
    df_ft = pd.DataFrame(fake_tracks)
    strata = df_ft.groupby(["source", "fake_label"], dropna=False)
    per_stratum = max(max_fake // max(strata.ngroups, 1), 1)
    sampled: list[dict] = []
    for _, grp in strata:
        idx = rng.permutation(len(grp))[:per_stratum]
        sampled.extend(grp.iloc[idx].to_dict("records"))
    return sampled[:max_fake]


def collect_audio_files(
    max_real: int | None = None,
    max_fake: int | None = None,
    split: str | None = None,
    stratified: bool = False,
    seed: int = 42,
    fake_types: list[str] | None = None,
    algorithms: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Collect real and fake audio file paths from SONICS.

    When ``stratified=True`` fake tracks are sampled equally across
    (source × fake_label) strata instead of first-N slicing.
    Use ``fake_types`` to restrict to specific fake_label values
    (e.g. ['full fake', 'half fake']) and ``algorithms`` to restrict
    to specific generator versions (e.g. ['chirp-v3.5', 'udio-120s']).
    """
    metadata_dir = SONICS_DIR / "metadata"
    real_csv = metadata_dir / "real_songs_ready.csv"
    fake_csv = metadata_dir / "fake_songs_ready.csv"

    if not real_csv.exists() or not fake_csv.exists():
        logger.error("Prepared metadata not found. Run:\n  python scripts/download_sonics.py --prepare")
        sys.exit(1)

    df_real = pd.read_csv(real_csv, low_memory=False)
    df_fake = pd.read_csv(fake_csv, low_memory=False)

    if split:
        if "split" in df_real.columns:
            df_real = df_real[df_real["split"] == split]
        if "split" in df_fake.columns:
            df_fake = df_fake[df_fake["split"] == split]

    real_tracks = [t for _, row in df_real.iterrows() if (t := _row_to_real_track(row))]
    fake_tracks = [t for _, row in df_fake.iterrows() if (t := _row_to_fake_track(row))]

    if fake_types:
        ft_lower = {f.lower() for f in fake_types}
        fake_tracks = [t for t in fake_tracks if t.get("fake_label", "").lower() in ft_lower]
        logger.info("Filtered to fake_types=%s → %d fake tracks", fake_types, len(fake_tracks))
    if algorithms:
        alg_lower = {a.lower() for a in algorithms}
        fake_tracks = [t for t in fake_tracks if t.get("algorithm", "").lower() in alg_lower]
        logger.info("Filtered to algorithms=%s → %d fake tracks", algorithms, len(fake_tracks))

    rng = np.random.default_rng(seed)

    if stratified and max_fake is not None:
        fake_tracks = _sample_fake_stratified(fake_tracks, max_fake, rng)
    elif max_fake is not None:
        idx = rng.permutation(len(fake_tracks))[:max_fake]
        fake_tracks = [fake_tracks[i] for i in idx]

    if max_real is not None:
        idx = rng.permutation(len(real_tracks))[:max_real]
        real_tracks = [real_tracks[i] for i in idx]

    if fake_tracks:
        df_ft = pd.DataFrame(fake_tracks)
        for (src, fl), grp in df_ft.groupby(["source", "fake_label"], dropna=False):
            logger.info("  fake stratum (%s / %s): %d tracks", src, fl, len(grp))

    logger.info("Collected %d real, %d fake tracks", len(real_tracks), len(fake_tracks))
    return real_tracks, fake_tracks


def run_id_experiment(
    tracks: list[dict],
    embedding_name: str,
    device: str = "cpu",
    max_duration: float = 120.0,
    checkpoint_dir: Path | None = None,
    estimators: list[str] | None = None,
) -> list[dict]:
    """Extract embeddings and compute ID for a list of tracks.

    Parameters
    ----------
    estimators : which ID methods to run.  Defaults to DEFAULT_ESTIMATORS
                 (PHD + TwoNN).  Pass ALL_ESTIMATORS to include MLE.

    Returns list of dicts with track info + id_{est} columns + n_embeddings.
    Supports checkpoint/resume: saves progress every 100 tracks.
    """
    if estimators is None:
        estimators = DEFAULT_ESTIMATORS

    emb_cfg = EMBEDDING_CONFIGS[embedding_name]
    target_sr = emb_cfg["target_sr"]
    metric = emb_cfg["metric"]

    # Resume from checkpoint if available
    processed_ids: set[str] = set()
    results: list[dict] = []
    if checkpoint_dir:
        ckpt_file = checkpoint_dir / f"checkpoint_{embedding_name}.json"
        if ckpt_file.exists():
            with open(ckpt_file) as f:
                results = json.load(f)
            processed_ids = {r["track_id"] for r in results}
            logger.info("Resuming %s from checkpoint: %d tracks done", embedding_name, len(results))

    logger.info("Loading %s extractor on %s (estimators: %s)...", embedding_name, device, estimators)
    extractor = get_extractor(embedding_name, device=device)

    for i, track in enumerate(tracks):
        track_id = track["track_id"]
        if track_id in processed_ids:
            continue

        track["embedding"] = embedding_name
        result = _process_one_track(track, extractor, target_sr, metric, max_duration, estimators)
        if result is None:
            continue

        results.append(result)
        ids = {k.replace("id_", ""): result[k] for k in result if k.startswith("id_")}

        if (i + 1) % 25 == 0 or i == 0:
            est_str = "  ".join(f"{k}={ids.get(k, float('nan')):.2f}" for k in estimators)
            logger.info("[%d/%d] %s %s: %s", i + 1, len(tracks), track["label"], track_id, est_str)

        if checkpoint_dir and len(results) % 100 == 0:
            _save_checkpoint(results, checkpoint_dir, embedding_name)

    if checkpoint_dir and results:
        _save_checkpoint(results, checkpoint_dir, embedding_name)

    del extractor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return results


def run_fakeprint_experiment(tracks: list[dict]) -> list[dict]:
    """Run fakeprint baseline on tracks."""
    from intrinsic_ai_music_detection.features.fakeprints import compute_fakeprint

    results = []
    for i, track in enumerate(tracks):
        track_id = track["track_id"]
        try:
            audio, sr = load_audio(track["path"], target_sr=44100, max_duration=180.0)
            fp = compute_fakeprint(audio, sr)
            result = {
                **track,
                "fakeprint_mean": float(np.mean(fp)),
                "fakeprint_max": float(np.max(fp)),
                "fakeprint_std": float(np.std(fp)),
                "fakeprint_n_peaks": int(np.sum(fp > np.mean(fp) + 2 * np.std(fp))),
            }
            results.append(result)

            if (i + 1) % 20 == 0:
                logger.info("[%d/%d] fakeprint %s", i + 1, len(tracks), track_id)
        except Exception as e:
            logger.warning("Fakeprint failed for %s: %s", track_id, e)

    return results


def run_fst_experiment(tracks: list[dict], device: str = "cpu") -> list[dict]:
    """Run FST baseline on tracks."""
    from intrinsic_ai_music_detection.baselines.fst import FSTBaseline, FSTConfig

    cfg = FSTConfig(device=device)
    fst = FSTBaseline(cfg)

    # Use batch prediction for efficiency
    paths = [track["path"] for track in tracks]
    try:
        predictions = fst.predict_batch(paths)
    except Exception:
        logger.warning("FST predict_batch failed, falling back to sequential")
        predictions = []
        for track in tracks:
            try:
                predictions.append(fst.predict(track["path"]))
            except Exception as e:
                logger.warning("FST failed for %s: %s", track["track_id"], e)
                predictions.append(None)

    results = []
    for i, (track, pred) in enumerate(zip(tracks, predictions)):
        if pred is None:
            continue
        result = {**track, **pred}
        results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(
                "[%d/%d] FST %s: %s (p=%.3f)",
                i + 1,
                len(tracks),
                track["track_id"],
                pred["prediction"],
                pred["fake_prob"],
            )

    return results


def analyze_results(results: list[dict], method_prefix: str = "id") -> dict:
    """Analyze ID estimation results: compare distributions, compute effect sizes."""
    df = pd.DataFrame(results)

    id_cols = [c for c in df.columns if c.startswith(f"{method_prefix}_")]
    if not id_cols:
        return {}

    analysis = {}
    for col in id_cols:
        method = col.replace(f"{method_prefix}_", "")
        real_vals = df[df["label"] == "real"][col].dropna().values
        fake_vals = df[df["label"] == "fake"][col].dropna().values

        if len(real_vals) < 2 or len(fake_vals) < 2:
            continue

        d = cohens_d(real_vals, fake_vals)
        stats = compare_distributions(real_vals, fake_vals)

        analysis[method] = {
            "real_mean": float(np.mean(real_vals)),
            "real_std": float(np.std(real_vals)),
            "real_n": len(real_vals),
            "fake_mean": float(np.mean(fake_vals)),
            "fake_std": float(np.std(fake_vals)),
            "fake_n": len(fake_vals),
            "cohens_d": float(d),
            "diff": float(np.mean(real_vals) - np.mean(fake_vals)),
            "mann_whitney_u": stats.mann_whitney_u,
            "mann_whitney_p": stats.mann_whitney_p,
            "p_value": stats.mann_whitney_p,
            "ks_statistic": stats.ks_statistic,
            "ks_p": stats.ks_p,
        }

    return analysis


def _save_checkpoint(results: list[dict], checkpoint_dir: Path, embedding_name: str) -> None:
    ckpt_file = checkpoint_dir / f"checkpoint_{embedding_name}.json"
    with open(ckpt_file, "w") as f:
        json.dump(results, f, default=str)
    logger.info("Checkpoint saved: %d tracks", len(results))


def print_summary(analysis: dict, embedding_name: str) -> None:
    """Pretty-print analysis summary."""
    print(f"\n{'='*70}")
    print(f"SONICS Experiment Results — Embedding: {embedding_name.upper()}")
    print(f"{'='*70}")

    for method, stats in analysis.items():
        direction = "REAL > FAKE (expected)" if stats["diff"] > 0 else "FAKE > REAL (unexpected)"
        d_abs = abs(stats["cohens_d"])
        if d_abs >= 0.8:
            effect = "large"
        elif d_abs >= 0.5:
            effect = "medium"
        else:
            effect = "small"

        print(f"\n  {method.upper()}:")
        print(f"    Real:  mean={stats['real_mean']:.3f} ± {stats['real_std']:.3f}  (n={stats['real_n']})")
        print(f"    Fake:  mean={stats['fake_mean']:.3f} ± {stats['fake_std']:.3f}  (n={stats['fake_n']})")
        print(f"    Δ = {stats['diff']:+.3f}  |  Cohen's d = {stats['cohens_d']:.3f} ({effect})")
        print(f"    Direction: {direction}")
        if "p_value" in stats:
            print(f"    p-value: {stats['p_value']:.2e}")

    print(f"{'='*70}")


def _build_logreg_pipeline() -> Pipeline:
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


def _eval_feature_set(name: str, df: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    """Stratified 5-fold CV with per-fold coef extraction for interpretability."""
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

    pipe = _build_logreg_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tp = int(((y == 1) & (y_pred == 1)).sum())

    # Per-fold coefficient stability
    fold_coefs = []
    for tr, _ in cv.split(X, y):
        pipe_fold = _build_logreg_pipeline()
        pipe_fold.fit(X[tr], y[tr])
        fold_coefs.append(pipe_fold.named_steps["clf"].coef_.ravel().tolist())

    coef_arr = np.array(fold_coefs)
    coef_means = coef_arr.mean(axis=0).tolist()
    coef_stds = coef_arr.std(axis=0).tolist()
    sign_consistency = (np.sign(coef_arr) == np.sign(coef_arr.mean(axis=0))).mean(axis=0).tolist()

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
        "coef_means": coef_means,
        "coef_stds": coef_stds,
        "coef_sign_consistency": sign_consistency,
    }


def run_classification(results: list[dict]) -> dict:
    """Train per-embedding classifiers and save interpretability outputs."""
    from intrinsic_ai_music_detection.models.train_model import train_and_evaluate

    df = pd.DataFrame(results)
    id_cols = [c for c in df.columns if c.startswith("id_")]

    if not id_cols or len(df) < 10:
        return {}

    valid = df.dropna(subset=id_cols)
    X = valid[id_cols].values.astype(np.float32)
    y = (valid["label"] == "fake").astype(int).values
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 10 or len(np.unique(y)) < 2:
        logger.warning("Only one class or too few samples, skipping classification")
        return {}

    clf_result = train_and_evaluate(X, y)
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": id_cols,
        **clf_result.__dict__,
    }


def _process_one_track(
    track: dict,
    extractor,
    target_sr: int,
    metric: str,
    max_duration: float,
    estimators: list[str],
) -> dict | None:
    """Load, embed, estimate ID for one track. Returns None on failure."""
    try:
        audio, sr = load_audio(track["path"], target_sr=target_sr, max_duration=max_duration)
        audio = normalize_audio(audio)
        emb = extractor.extract(audio, sr)
        del audio

        finite_mask = np.isfinite(emb).all(axis=1)
        emb = emb[finite_mask]
        if emb.shape[0] < 10:
            logger.warning("Too few embeddings (%d) for %s, skipping", emb.shape[0], track["track_id"])
            return None

        ids = estimate_all(emb, metric=metric, methods=estimators)
        result = {**track, "n_embeddings": emb.shape[0], "embedding": track.get("embedding", "")}
        result.update({f"id_{k}": v for k, v in ids.items()})
        return result
    except Exception as e:
        logger.warning("Failed to process %s: %s: %s", track["track_id"], type(e).__name__, e)
        return None


def _run_strat_dimension(
    merged: pd.DataFrame,
    cols: list[str],
    dimension: str,
    col: str,
) -> list[dict]:
    rows = []
    values = sorted(v for v in merged[col].dropna().unique() if str(v).strip())
    for val in values:
        sub = merged[(merged["label"] == "real") | ((merged["label"] == "fake") & (merged[col] == val))]
        res = _eval_feature_set(f"{dimension}:{val}", sub, cols)
        if res:
            rows.append(
                {
                    "dimension": dimension,
                    "value": val,
                    "n_real": int((sub["label"] == "real").sum()),
                    "n_fake": int((sub["label"] == "fake").sum()),
                    "auc": res["auc"],
                    "f1": res["f1"],
                    "fpr": res["fpr"],
                    "fnr": res["fnr"],
                    "accuracy": res["accuracy"],
                }
            )
    return rows


def run_interpretability_and_fusion(
    dfs: dict[str, pd.DataFrame],
    fakeprint_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Build cross-embedding fusion table and run logistic interpretability analysis.

    Produces:
    - fusion_features.csv          wide-format per-track feature table
    - classification_fusion.json   CV metrics + coef interpretability per bundle
    - stratified_eval.csv          per-source / per-fake-label breakdown
    - interpretability.json        full coef mean/std/sign_consistency table
    """
    if not dfs:
        return

    # --- build wide fusion table ---
    merged: pd.DataFrame | None = None
    id_cols_base = ["id_phd", "id_twonn", "id_mle"]
    for emb, df in dfs.items():
        keep_base = ["track_id", "label", "split", "source", "fake_label", "algorithm"]
        keep_id = [c for c in id_cols_base if c in df.columns]
        keep = keep_base + keep_id
        temp = df[[c for c in keep if c in df.columns]].copy()
        temp = temp.rename(columns={c: f"{emb}_{c}" for c in keep_id})
        if merged is None:
            merged = temp
        else:
            id_renamed = [f"{emb}_{c}" for c in keep_id]
            merged = merged.merge(
                temp[["track_id", "label"] + id_renamed],
                on=["track_id", "label"],
                how="inner",
            )

    if merged is None:
        return

    if fakeprint_df is not None:
        fp_cols = ["track_id", "label"] + [
            c
            for c in ["fakeprint_mean", "fakeprint_max", "fakeprint_std", "fakeprint_n_peaks"]
            if c in fakeprint_df.columns
        ]
        merged = merged.merge(fakeprint_df[fp_cols], on=["track_id", "label"], how="left")

    merged["y"] = (merged["label"] == "fake").astype(int)
    merged.to_csv(output_dir / "fusion_features.csv", index=False)
    logger.info("Fusion table: %d rows, %d cols — saved fusion_features.csv", len(merged), len(merged.columns))

    # --- define feature bundles (mirrors pilot analysis decisions) ---
    all_embs = list(dfs.keys())
    phd_twonn_cols = [f"{e}_id_phd" for e in all_embs] + [f"{e}_id_twonn" for e in all_embs]
    stable_cols = _usable_cols(merged, phd_twonn_cols)
    all_id_cols = _usable_cols(merged, [f"{e}_{c}" for e in all_embs for c in id_cols_base])
    fp_summary = _usable_cols(merged, ["fakeprint_mean", "fakeprint_max", "fakeprint_std", "fakeprint_n_peaks"])

    bundles: dict[str, list[str]] = {}
    for emb in all_embs:
        cols = _usable_cols(merged, [f"{emb}_id_phd", f"{emb}_id_twonn", f"{emb}_id_mle"])
        if cols:
            bundles[f"single:{emb}"] = cols
    if stable_cols:
        bundles[BUNDLE_STABLE] = stable_cols
    if all_id_cols:
        bundles["fusion:all_id"] = all_id_cols
    if stable_cols and fp_summary:
        bundles["fusion:phd_twonn_plus_fakeprint"] = stable_cols + fp_summary
    if all_id_cols and fp_summary:
        bundles["fusion:all_id_plus_fakeprint"] = all_id_cols + fp_summary

    # --- evaluate each bundle ---
    fusion_rows = []
    interpretability: dict[str, dict] = {}

    for bundle_name, cols in bundles.items():
        res = _eval_feature_set(bundle_name, merged, cols)
        if res is None:
            continue
        fusion_rows.append(
            {
                k: v
                for k, v in res.items()
                if k not in ("coef_means", "coef_stds", "coef_sign_consistency", "feature_names")
            }
        )
        interpretability[bundle_name] = {
            "features": res["feature_names"],
            "coef_mean": res["coef_means"],
            "coef_std": res["coef_stds"],
            "coef_sign_consistency": res["coef_sign_consistency"],
        }
        logger.info(
            "  [fusion] %-45s  AUC=%.3f  F1=%.3f  FPR=%.3f  FNR=%.3f",
            bundle_name,
            res["auc"],
            res["f1"],
            res["fpr"],
            res["fnr"],
        )

    if fusion_rows:
        pd.DataFrame(fusion_rows).sort_values("auc", ascending=False).to_csv(
            output_dir / "classification_fusion.json".replace(".json", ".csv"), index=False
        )

    with open(output_dir / "interpretability.json", "w") as f:
        json.dump(interpretability, f, indent=2)
    logger.info("Interpretability saved to interpretability.json")

    # --- stratified evaluation ---
    strat_rows = []
    if BUNDLE_STABLE in bundles:
        best_bundle: str | None = BUNDLE_STABLE
    elif bundles:
        best_bundle = next(iter(bundles))
    else:
        best_bundle = None
    if best_bundle and "source" in merged.columns and "fake_label" in merged.columns:
        cols = bundles[best_bundle]
        strat_rows += _run_strat_dimension(merged, cols, "source", "source")
        strat_rows += _run_strat_dimension(merged, cols, "fake_type", "fake_label")
        if "algorithm" in merged.columns:
            strat_rows += _run_strat_dimension(merged, cols, "algorithm", "algorithm")

    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(output_dir / "stratified_eval.csv", index=False)
        logger.info("Stratified eval saved: %d rows", len(strat_rows))
        for r in strat_rows:
            logger.info(
                "  strat [%s=%s]  n_real=%d n_fake=%d  AUC=%.3f F1=%.3f FPR=%.3f FNR=%.3f",
                r["dimension"],
                r["value"],
                r["n_real"],
                r["n_fake"],
                r["auc"],
                r["f1"],
                r["fpr"],
                r["fnr"],
            )


def _run_one_embedding(
    emb_name: str,
    all_tracks: list[dict],
    device: str,
    max_duration: float,
    output_dir: Path,
    estimators: list[str],
) -> tuple[list[dict], pd.DataFrame] | None:
    """Run ID extraction + per-embedding classification for one embedding.  Returns (id_results, df)."""
    logger.info("=" * 60)
    logger.info("Running ID experiment: %s", emb_name)
    try:
        id_results = run_id_experiment(
            all_tracks,
            emb_name,
            device=device,
            max_duration=max_duration,
            checkpoint_dir=output_dir,
            estimators=estimators,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("GPU OOM on %s — try --device cpu or smaller --max-duration", emb_name)
            torch.cuda.empty_cache()
            return None
        raise

    if not id_results:
        return None

    analysis = analyze_results(id_results)
    print_summary(analysis, emb_name)

    df_emb = pd.DataFrame(id_results)
    df_emb.to_csv(output_dir / f"id_results_{emb_name}.csv", index=False)
    with open(output_dir / f"analysis_{emb_name}.json", "w") as f:
        json.dump(analysis, f, indent=2)

    clf_result = run_classification(id_results)
    if clf_result:
        logger.info(
            "  [%s] Acc=%.3f  F1=%.3f  AUC=%.3f",
            emb_name,
            clf_result.get("cv_accuracy", 0),
            clf_result.get("cv_f1", 0),
            clf_result.get("cv_auc", 0),
        )
        with open(output_dir / f"classification_{emb_name}.json", "w") as f:
            json.dump(clf_result, f, indent=2, default=str)

    return id_results, df_emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SONICS experiments — full-scale edition")
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["encodec", "clap", "mert", "muq", "all"],
        default="all",
        help="Embedding model(s) to run (default: all)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-fake", type=int, default=None)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Stratify fake sample across (source x fake_label) strata instead of first-N slicing",
    )
    parser.add_argument(
        "--include-mle",
        action="store_true",
        help="Include MLE estimator (noisy; excluded by default from pilot findings)",
    )
    parser.add_argument("--include-fakeprint", action="store_true")
    parser.add_argument("--include-fst", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fake-types",
        nargs="+",
        default=None,
        help="Restrict to specific fake_label values, e.g. 'full fake' 'half fake'",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Restrict to specific algorithm versions, e.g. chirp-v3.5 udio-120s",
    )
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)  # ensure tee target exists

    estimators = ALL_ESTIMATORS if args.include_mle else DEFAULT_ESTIMATORS
    logger.info("Estimators: %s", estimators)

    real_tracks, fake_tracks = collect_audio_files(
        max_real=args.max_real,
        max_fake=args.max_fake,
        split=args.split,
        stratified=args.stratified,
        seed=args.seed,
        fake_types=args.fake_types,
        algorithms=args.algorithms,
    )
    all_tracks = real_tracks + fake_tracks
    if not all_tracks:
        logger.error("No audio files found. Run download_sonics.py first.")
        sys.exit(1)

    embeddings_to_run = ["encodec", "clap", "mert", "muq"] if args.embedding == "all" else [args.embedding]
    all_id_results: list[dict] = []
    embedding_dfs: dict[str, pd.DataFrame] = {}

    for emb_name in embeddings_to_run:
        out = _run_one_embedding(emb_name, all_tracks, args.device, args.max_duration, output_dir, estimators)
        if out is not None:
            id_results, df_emb = out
            all_id_results.extend(id_results)
            embedding_dfs[emb_name] = df_emb

    fakeprint_df: pd.DataFrame | None = None
    if args.include_fakeprint:
        logger.info("Running fakeprint baseline...")
        fp_results = run_fakeprint_experiment(all_tracks)
        if fp_results:
            fakeprint_df = pd.DataFrame(fp_results)
            fakeprint_df.to_csv(output_dir / "fakeprint_results.csv", index=False)
            real_fp = fakeprint_df[fakeprint_df["label"] == "real"]["fakeprint_mean"]
            fake_fp = fakeprint_df[fakeprint_df["label"] == "fake"]["fakeprint_mean"]
            logger.info("  Fakeprint mean: Real=%.4f  Fake=%.4f", real_fp.mean(), fake_fp.mean())

    if args.include_fst:
        logger.info("Running FST baseline...")
        fst_results = run_fst_experiment(all_tracks, device=args.device)
        if fst_results:
            df_fst = pd.DataFrame(fst_results)
            df_fst.to_csv(output_dir / "fst_results.csv", index=False)
            correct = sum(1 for r in fst_results if (r["prediction"] == "fake") == (r["label"] == "fake"))
            logger.info("  FST Accuracy: %d/%d = %.3f", correct, len(fst_results), correct / len(fst_results))

    if embedding_dfs:
        logger.info("=" * 60)
        logger.info("Running fusion + interpretability + stratified eval...")
        run_interpretability_and_fusion(embedding_dfs, fakeprint_df, output_dir)

    if all_id_results:
        pd.DataFrame(all_id_results).to_csv(output_dir / "all_id_results.csv", index=False)

    logger.info("=" * 60)
    logger.info("All results saved to %s", output_dir)
    logger.info("Key outputs:")
    logger.info("  id_results_{emb}.csv        per-track ID + full metadata")
    logger.info("  fusion_features.csv         wide-format feature table (MLP-ready)")
    logger.info("  classification_fusion.csv   CV metrics per feature bundle")
    logger.info("  interpretability.json       coef mean/std/sign_consistency per fold")
    logger.info("  stratified_eval.csv         per-source and per-fake-type breakdown")
    logger.info("  analysis_{emb}.json         Cohen's d + Mann-Whitney per estimator")


if __name__ == "__main__":
    main()
