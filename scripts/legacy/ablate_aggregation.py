"""Trajectory-based scoring ablation: aggregators (I1) + window-level typicality.

CPU-only post-processing — no GPU, no re-scoring. Consumes the per-window
ANOMALY trajectories persisted by `run_balanced_ablation.py
--wf-save-trajectories <dir>` (wf_trajectories_<emb>.npz + _meta.csv).

Two families of track scores are evaluated:

1. **Aggregators** (`intrinsic_ai_music_detection.models.aggregate`):
   mean / median / trimmed_mean_10 / top_10pct / top_25pct / max / logsumexp.
   Motivation: if AI artifacts are localized in a minority of windows, top-k%
   aggregation beats the mean; if evidence is diffuse, the mean wins.

2. **Window-level typicality** (Nalisnick et al., arXiv:1906.02994). Their
   typicality test is an M-SAMPLE statistic: an input batch is in-distribution
   when its empirical entropy rate is CLOSE to the model's, i.e. score
   |(1/M) Σ_i NLL(x_i) − H| with H estimated on training data. A track's
   windows are exactly such a batch (M = n_windows), so this is the correct
   form of the test for us — as opposed to applying |·−H| to an
   already-aggregated track score, which discards within-track dispersion.
   Variants: `typ_M` (the M-sample statistic), `typ_window_mean` (mean of
   per-window deviations), `typ_window_top25`.
   The reference H is estimated from a REFERENCE HALF of the real trajectories,
   disjoint from the evaluated reals (split-half, label-free).

Usage:
    python scripts/ablate_aggregation.py \\
        --trajectories .../trajectories/wf_trajectories_encodec.npz \\
        --meta .../trajectories/wf_trajectories_encodec_meta.csv \\
        --output-csv .../aggregation_ablation.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.aggregate import AGGREGATOR_NAMES, aggregate_trajectory  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

TYPICALITY_NAMES = ("typ_M", "typ_window_mean", "typ_window_top25")


def _typicality(traj: np.ndarray, ref_h: float, variant: str) -> float:
    t = np.asarray(traj, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return float("nan")
    if variant == "typ_M":
        # Nalisnick M-sample statistic: |empirical entropy rate − model entropy|
        return float(abs(t.mean() - ref_h))
    dev = np.abs(t - ref_h)  # per-window deviation from the typical NLL
    if variant == "typ_window_mean":
        return float(dev.mean())
    if variant == "typ_window_top25":
        k = max(1, int(np.ceil(0.25 * len(dev))))
        return float(np.sort(dev)[-k:].mean())
    raise ValueError(variant)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectories", required=True, help="wf_trajectories_<emb>.npz (ANOMALY-oriented)")
    parser.add_argument("--meta", required=True, help="wf_trajectories_<emb>_meta.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--emit-scorer",
        default=None,
        help=(
            "Also write per-track scores for this scorer (e.g. 'top_25pct') to --emit-csv, in "
            "window_flow_eval format so it feeds delong_vs_musicdet.py / "
            "analyze_production_confound.py / fuse_labelfree_scores.py directly. The emitted "
            "'*_wf_mean' column is +log-likelihood-oriented (= -anomaly), matching the "
            "convention every downstream consumer expects."
        ),
    )
    parser.add_argument("--emit-csv", default=None)
    args = parser.parse_args()

    logger.info("Loading trajectories: %s", args.trajectories)
    npz = np.load(args.trajectories)
    meta = pd.read_csv(args.meta, low_memory=False)
    meta["track_id"] = meta["track_id"].astype(str)
    # Collision-safe key (see run_balanced_ablation.py). Older runs stored only
    # track_id — which COLLIDES on corpora that reuse ids across generators
    # (FakeMusicCaps): detect and refuse rather than silently mis-score.
    if "key" in meta.columns:
        meta["key"] = meta["key"].astype(str)
    else:
        meta["key"] = meta["track_id"]
        if meta["key"].duplicated().any() or len(npz.files) != len(meta):
            logger.error(
                "LEGACY trajectory file with COLLIDING track_id keys (%d keys for %d meta rows). "
                "Results from this file are INVALID — re-run run_balanced_ablation.py with the "
                "fixed --wf-save-trajectories (writes a unique 'key' column).",
                len(npz.files),
                len(meta),
            )
            sys.exit(1)
    meta = meta[meta["key"].isin(set(npz.files))]
    logger.info("%d trajectories, %d matched meta rows", len(npz.files), len(meta))

    # --- split-half reference for the typicality statistic (label-free) ---
    meta_real = meta[meta["label"] == "real"]
    meta_fake = meta[meta["label"] == "fake"]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(meta_real))
    ref_real = meta_real.iloc[idx[: len(meta_real) // 2]]
    eval_real = meta_real.iloc[idx[len(meta_real) // 2 :]]
    ref_windows = (
        np.concatenate([np.asarray(npz[k]).ravel() for k in ref_real["key"]]) if len(ref_real) else np.array([])
    )
    ref_windows = ref_windows[np.isfinite(ref_windows)]
    ref_h = float(np.mean(ref_windows)) if len(ref_windows) else float("nan")
    logger.info(
        "typicality reference H (mean per-window NLL of %d reference-real windows from %d tracks): %.4f",
        len(ref_windows),
        len(ref_real),
        ref_h,
    )

    algos = sorted(str(a) for a in meta_fake["algorithm"].dropna().unique() if str(a).strip())
    rows = []

    # ---- family 1: aggregators (all reals evaluated; no reference needed) ----
    for m in AGGREGATOR_NAMES:
        sc = {k: aggregate_trajectory(npz[k], m) for k in meta["key"]}
        real_s = np.array([sc[k] for k in meta_real["key"]])
        real_s = real_s[np.isfinite(real_s)]
        if len(real_s) < args.min_per_group:
            continue
        aucs = []
        for alg in algos:
            fake_s = np.array([sc[k] for k in meta_fake.loc[meta_fake["algorithm"] == alg, "key"]])
            fake_s = fake_s[np.isfinite(fake_s)]
            if len(fake_s) < args.min_per_group:
                continue
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
            rows.append(
                {
                    "family": "aggregator",
                    "scorer": m,
                    "algorithm": alg,
                    "auc": round(auc, 4),
                    "eer_pct": round(eer * 100, 2),
                    "n_real": len(real_s),
                    "n_fake": len(fake_s),
                }
            )
            aucs.append(auc)
        if aucs:
            rows.append(
                {
                    "family": "aggregator",
                    "scorer": m,
                    "algorithm": "MACRO_AVG",
                    "auc": round(float(np.mean(aucs)), 4),
                    "eer_pct": float("nan"),
                    "n_real": len(real_s),
                    "n_fake": len(meta_fake),
                }
            )

    # ---- family 2: window-level typicality (evaluated on held-out real half) ----
    if np.isfinite(ref_h):
        for m in TYPICALITY_NAMES:
            sc = {k: _typicality(npz[k], ref_h, m) for k in meta["key"]}
            real_s = np.array([sc[k] for k in eval_real["key"]])
            real_s = real_s[np.isfinite(real_s)]
            if len(real_s) < args.min_per_group:
                continue
            aucs = []
            for alg in algos:
                fake_s = np.array([sc[k] for k in meta_fake.loc[meta_fake["algorithm"] == alg, "key"]])
                fake_s = fake_s[np.isfinite(fake_s)]
                if len(fake_s) < args.min_per_group:
                    continue
                y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
                auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
                rows.append(
                    {
                        "family": "typicality",
                        "scorer": m,
                        "algorithm": alg,
                        "auc": round(auc, 4),
                        "eer_pct": round(eer * 100, 2),
                        "n_real": len(real_s),
                        "n_fake": len(fake_s),
                    }
                )
                aucs.append(auc)
            if aucs:
                rows.append(
                    {
                        "family": "typicality",
                        "scorer": m,
                        "algorithm": "MACRO_AVG",
                        "auc": round(float(np.mean(aucs)), 4),
                        "eer_pct": float("nan"),
                        "n_real": len(real_s),
                        "n_fake": len(meta_fake),
                    }
                )

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.info("Saved %s", args.output_csv)

    if args.emit_scorer:
        if not args.emit_csv:
            raise SystemExit("--emit-scorer requires --emit-csv")
        name = args.emit_scorer
        if name in AGGREGATOR_NAMES:
            per_track = {k: aggregate_trajectory(npz[k], name) for k in meta["key"]}
        elif name in TYPICALITY_NAMES:
            if not np.isfinite(ref_h):
                raise SystemExit(f"cannot emit {name}: no finite typicality reference")
            per_track = {k: _typicality(npz[k], ref_h, name) for k in meta["key"]}
        else:
            raise SystemExit(f"unknown scorer {name!r}; choose from {AGGREGATOR_NAMES + TYPICALITY_NAMES}")
        emit = meta[["track_id", "label", "algorithm", "fold"]].copy()
        # anomaly -> +log-likelihood orientation expected by downstream consumers
        emit["encodec_wf_mean"] = [-per_track[k] for k in meta["key"]]
        Path(args.emit_csv).parent.mkdir(parents=True, exist_ok=True)
        emit.to_csv(args.emit_csv, index=False)
        logger.info("Emitted per-track '%s' scores (%d rows) → %s", name, len(emit), args.emit_csv)

    pivot = out[out["algorithm"] != "MACRO_AVG"].pivot_table(index="scorer", columns="algorithm", values="auc")
    pivot["MACRO_AVG"] = out[out["algorithm"] == "MACRO_AVG"].set_index("scorer")["auc"]
    logger.info("\nAUC by scorer (rows) x generator (cols):\n%s", pivot.to_string())
    macro = out[out["algorithm"] == "MACRO_AVG"].set_index("scorer")["auc"]
    if len(macro):
        logger.info(
            "Best macro aggregator/scorer: %s (AUC %.4f vs mean %.4f). Note typicality rows are "
            "evaluated on the HELD-OUT real half (reference-disjoint), aggregator rows on all reals.",
            macro.idxmax(),
            macro.max(),
            macro.get("mean", float("nan")),
        )


if __name__ == "__main__":
    main()
