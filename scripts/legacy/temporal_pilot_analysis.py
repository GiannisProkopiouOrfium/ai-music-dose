"""Temporal-ID pilot analysis on a small balanced SONICS subset.

Purpose
-------
Test whether temporal evolution of intrinsic dimension adds signal beyond the
static per-track ID values already computed in the pilot experiment.

What it does
------------
- Loads a pilot result CSV (default: id_results_encodec.csv)
- Samples a balanced subset of real/fake tracks
- Splits each track into overlapping windows
- Computes ID per window (PHD / TwoNN / MLE)
- Summarizes each ID trajectory with mean/std/min/max/range/slope
- Compares static vs temporal feature sets with logistic regression
- Saves CSVs + plots for slide-ready inspection

Usage
-----
poetry run python scripts/temporal_pilot_analysis.py --embedding encodec --n-real 20 --n-fake 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio, sliding_window
from intrinsic_ai_music_detection.features.embeddings import get_extractor
from intrinsic_ai_music_detection.features.id_estimators import estimate_all

RESULTS_DIR = Path("data/processed/sonics_results")
OUT_DIR = Path("reports/pre_fullscale/temporal_pilot")
FIG_DIR = OUT_DIR / "figures"

EMBEDDING_CONFIGS = {
    "encodec": {"target_sr": 24_000, "metric": "euclidean"},
    "mert": {"target_sr": 24_000, "metric": "cosine"},
    "clap": {"target_sr": 48_000, "metric": "cosine"},
    "muq": {"target_sr": 24_000, "metric": "cosine"},
}

ESTIMATORS = ["phd", "twonn", "mle"]


def _mkdirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
            ),
        ],
        memory=None,
    )


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _log_progress(i: int, total_tracks: int, start_time: float, usable_tracks: int) -> None:
    elapsed = time.time() - start_time
    avg = elapsed / max(i, 1)
    eta = avg * max(total_tracks - i, 0)
    print(
        f"[temporal-pilot] progress {i}/{total_tracks} "
        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m "
        f"usable_tracks={usable_tracks}"
    )


def load_pilot_tracks(embedding: str, n_real: int, n_fake: int, seed: int = 42) -> pd.DataFrame:
    path = RESULTS_DIR / f"id_results_{embedding}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pilot CSV: {path}")

    df = pd.read_csv(path)
    if "path" not in df.columns:
        raise ValueError(f"Pilot CSV {path} does not contain a path column")

    df = df[df["path"].notna()].copy()
    df["y"] = (df["label"] == "fake").astype(int)

    real = df[df["label"] == "real"].copy().sample(n=min(n_real, (df["label"] == "real").sum()), random_state=seed)
    fake = df[df["label"] == "fake"].copy().sample(n=min(n_fake, (df["label"] == "fake").sum()), random_state=seed)
    out = pd.concat([real, fake], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def summarize_trajectory(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    values = _finite(values)
    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "slope": np.nan,
            "first_last_diff": np.nan,
            "n_windows": 0,
        }

    if len(times) != len(values):
        times = np.arange(len(values), dtype=float)

    if len(values) >= 2:
        slope = float(np.polyfit(times[: len(values)], values, deg=1)[0])
    else:
        slope = np.nan

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "slope": slope,
        "first_last_diff": float(values[-1] - values[0]) if len(values) >= 2 else np.nan,
        "n_windows": int(len(values)),
    }


def compute_temporal_features(
    row: pd.Series,
    extractor,
    embedding: str,
    estimators: list[str],
    window_duration: float,
    hop_duration: float,
    max_duration: float,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    cfg = EMBEDDING_CONFIGS[embedding]
    target_sr = cfg["target_sr"]
    metric = cfg["metric"]

    audio, sr = load_audio(row["path"], target_sr=target_sr, max_duration=max_duration)
    audio = normalize_audio(audio)

    windows = sliding_window(audio, sr, window_size=window_duration, hop_size=hop_duration)
    if len(windows) < 2:
        return None

    track_row: dict[str, float] = {
        "track_id": str(row["track_id"]),
        "label": row["label"],
        "y": int(row["y"]),
        "split": str(row.get("split", "")),
        "source": str(row.get("source", "")),
        "fake_label": str(row.get("fake_label", "")),
        "embedding": embedding,
        "n_windows": len(windows),
    }

    window_rows: list[dict] = []
    times = np.arange(len(windows), dtype=float) * hop_duration + window_duration / 2.0

    for idx, window in enumerate(windows):
        try:
            emb = extractor.extract(window, sr)
            ids = estimate_all(emb, metric=metric, methods=estimators)
            window_rows.append({"window_idx": idx, "time": float(times[idx]), **{f"id_{k}": v for k, v in ids.items()}})
        except Exception:
            window_rows.append(
                {"window_idx": idx, "time": float(times[idx]), **{f"id_{k}": np.nan for k in estimators}}
            )

    window_df = pd.DataFrame(window_rows)
    for est in estimators:
        col = f"id_{est}"
        if col not in window_df.columns:
            continue
        summary = summarize_trajectory(window_df["time"].to_numpy(), window_df[col].to_numpy())
        for k, v in summary.items():
            track_row[f"{est}_{k}"] = v

    # static reference from the full-track pilot CSV if available
    for est in estimators:
        static_col = f"id_{est}"
        if static_col in row.index:
            track_row[f"static_{est}"] = float(row[static_col])

    return track_row, window_df


def evaluate_features(df: pd.DataFrame, cols: list[str]) -> dict[str, float] | None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None

    valid = df.dropna(subset=cols).copy()
    if valid.empty:
        return None

    X = valid[cols].to_numpy(dtype=np.float32)
    y = valid["y"].to_numpy(dtype=np.int64)
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
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "auc": float(roc_auc_score(y, y_prob)),
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp": tp,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def plot_temporal_examples(
    features: pd.DataFrame, windows: dict[str, pd.DataFrame], embedding: str, estimator: str
) -> None:
    if not windows:
        return

    chosen = []
    # one real, one fake if possible
    for label in ["real", "fake"]:
        subset = features[features["label"] == label]
        if not subset.empty:
            chosen.append(subset.iloc[0]["track_id"])

    if not chosen:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {"real": "#2196F3", "fake": "#F44336"}
    for track_id in chosen:
        wdf = windows.get(track_id)
        if wdf is None or f"id_{estimator}" not in wdf.columns:
            continue
        row = features[features["track_id"] == track_id].iloc[0]
        ax.plot(
            wdf["time"],
            wdf[f"id_{estimator}"],
            marker="o",
            linewidth=2,
            label=f"{track_id} ({row['label']})",
            color=palette[row["label"]],
        )

    ax.set_title(f"Temporal ID evolution — {embedding.upper()} / {estimator.upper()}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ID per window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"temporal_{embedding}_{estimator}.png", dpi=200)
    plt.close(fig)


def run_temporal_extraction(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tracks = load_pilot_tracks(args.embedding, args.n_real, args.n_fake)
    extractor = get_extractor(args.embedding, device=args.device)
    out_rows: list[dict] = []
    window_tables: dict[str, pd.DataFrame] = {}

    total_tracks = len(tracks)
    print(
        "[temporal-pilot] starting "
        f"embedding={args.embedding} device={args.device} tracks={total_tracks} "
        f"window={args.window_duration}s hop={args.hop_duration}s max_duration={args.max_duration}s"
    )
    start_time = time.time()
    progress_every = 25

    for i, (_, row) in enumerate(tracks.iterrows(), start=1):
        result = compute_temporal_features(
            row,
            extractor=extractor,
            embedding=args.embedding,
            estimators=args.estimators,
            window_duration=args.window_duration,
            hop_duration=args.hop_duration,
            max_duration=args.max_duration,
        )
        if result is not None:
            track_row, window_df = result
            out_rows.append(track_row)
            window_tables[track_row["track_id"]] = window_df

        if i % progress_every == 0 or i == total_tracks:
            _log_progress(i, total_tracks, start_time, len(out_rows))

    if not out_rows:
        raise SystemExit("No temporal features were computed")

    return pd.DataFrame(out_rows), window_tables


def build_temporal_summary(df: pd.DataFrame, estimators: list[str]) -> tuple[pd.DataFrame, str | None]:
    summary_rows = []
    for est in estimators:
        static_cols = [f"static_{est}"]
        temporal_cols = [f"{est}_{k}" for k in ["mean", "std", "min", "max", "range", "slope", "first_last_diff"]]
        static_res = evaluate_features(df, static_cols)
        temporal_res = evaluate_features(df, temporal_cols)
        if static_res is not None:
            summary_rows.append({"feature_set": f"static:{est}", **static_res})
        if temporal_res is not None:
            summary_rows.append({"feature_set": f"temporal:{est}", **temporal_res})

    summary_df = pd.DataFrame(summary_rows)
    best_est = None
    if not summary_df.empty:
        best = summary_df.sort_values("auc", ascending=False).iloc[0]
        best_est = best["feature_set"].split(":")[1]

    return summary_df, best_est


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal-ID pilot analysis on a balanced SONICS subset")
    parser.add_argument("--embedding", choices=list(EMBEDDING_CONFIGS.keys()), default="encodec")
    parser.add_argument("--n-real", type=int, default=20)
    parser.add_argument("--n-fake", type=int, default=20)
    parser.add_argument("--window-duration", type=float, default=30.0)
    parser.add_argument("--hop-duration", type=float, default=15.0)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--estimators", nargs="+", default=["phd", "twonn", "mle"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _mkdirs()
    df, window_tables = run_temporal_extraction(args)
    df.to_csv(OUT_DIR / f"temporal_features_{args.embedding}.csv", index=False)

    summary_df, best_est = build_temporal_summary(df, args.estimators)
    summary_df.to_csv(OUT_DIR / f"temporal_summary_{args.embedding}.csv", index=False)

    if best_est:
        plot_temporal_examples(df, window_tables, args.embedding, best_est)

    print(f"Saved temporal pilot outputs to {OUT_DIR}")
    print(f"- temporal_features_{args.embedding}.csv")
    print(f"- temporal_summary_{args.embedding}.csv")
    if best_est:
        print(f"- figures/temporal_{args.embedding}_{best_est}.png")


if __name__ == "__main__":
    main()
