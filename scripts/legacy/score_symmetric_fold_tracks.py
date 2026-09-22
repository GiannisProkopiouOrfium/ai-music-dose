"""Produce SYMMETRIC per-track scores using the saved K-fold flows.

Why this exists: the headline eval scores held-out reals with their own fold
flow but all fakes with a separate full-corpus flow, so real and fake scores
carry different calibration (different mu/sd, different convergence). §2 of the
comparison report fixes this for AUC by computing per-fold AUCs, but the
resulting *per-track* score table (window_flow_eval.csv) is still asymmetric —
which means any downstream analysis built on it (score fusion, aggregation
ablations) inherits the asymmetry and cannot be compared to MusicDET's pooled
EER on equal footing.

This script emits a symmetric table: every track — real or fake — is scored by
exactly ONE fold flow, and never by a flow that saw it in training.
  - Real tracks: scored by the flow of the fold they were HELD OUT from
    (read from the trajectory meta's `fold` column, or recomputed with the same
    KFold(shuffle=True, random_state=seed) over the sorted real ids).
  - Fake tracks: deterministically assigned to a fold (stable hash of the track
    key) and scored by that fold's flow. Fakes are never in training, so any
    fold flow is valid for them; fixing the assignment makes the real/fake
    comparison calibration-consistent and reproducible.

Output is column-compatible with `window_flow_eval.csv` ({emb}_wf_mean is the
per-track mean LOG-LIKELIHOOD, {emb}_wf_std its dispersion), so it drops
straight into `fuse_labelfree_scores.py` / `eval_typicality.py`.

Usage (EC2):
    python scripts/score_symmetric_fold_tracks.py \\
        --manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
        --emb-cache data/emb_cache_encodec_full \\
        --fold-flow-glob 'data/processed/full_sonics_k12_symmetric/sonics_real_flow_k12_encodec_fold*.pt' \\
        --traj-meta data/processed/full_sonics_k12_symmetric/trajectories/wf_trajectories_encodec_meta.csv \\
        --max-duration 55.0 --device cuda \\
        --output-csv data/processed/full_sonics_k12_symmetric/window_flow_eval_symmetric.csv
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.pooling import ENCODEC_FPS, pool_windows  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402
from intrinsic_ai_music_detection.models.flow import RealNVPOneClass  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

ENCODEC_SR = 24_000


def _cache_path(cache_root: Path, track_id: str, max_duration: float, embedding: str) -> Path:
    key = hashlib.md5(f"{track_id}_{ENCODEC_SR}_{max_duration}_preprocessed".encode()).hexdigest()
    return cache_root / embedding / f"{key}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--emb-cache", required=True)
    parser.add_argument("--fold-flow-glob", required=True, help="glob matching the saved *_fold<i>.pt flows")
    parser.add_argument(
        "--traj-meta", default=None, help="trajectory meta CSV providing each real track's held-out fold (recommended)"
    )
    parser.add_argument("--embedding", default="encodec")
    parser.add_argument("--max-duration", type=float, required=True)
    parser.add_argument("--window-duration", type=float, default=4.0)
    parser.add_argument("--hop-duration", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--save-trajectories",
        default=None,
        help=(
            "Directory for per-window ANOMALY trajectories under the SYMMETRIC protocol "
            "(wf_trajectories_<emb>.npz + _meta.csv, keys 'label|algorithm|track_id'). Needed to "
            "compute aggregation variants (top-k%%) on symmetric scores — the trajectories saved "
            "by run_balanced_ablation.py are asymmetric (fold-scored reals, full-flow fakes)."
        ),
    )
    args = parser.parse_args()

    flow_paths = sorted(glob.glob(args.fold_flow_glob))
    if not flow_paths:
        raise SystemExit(f"no fold flows matched {args.fold_flow_glob!r}")
    flows = [RealNVPOneClass.load(p, device=args.device) for p in flow_paths]
    logger.info("loaded %d fold flows: %s", len(flows), [Path(p).name for p in flow_paths])

    df = pd.read_csv(args.manifest, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df["track_id"] = df["track_id"].astype(str)
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("").astype(str)

    # --- fold assignment ---
    fold_of: dict[str, int] = {}
    if args.traj_meta and Path(args.traj_meta).exists():
        meta = pd.read_csv(args.traj_meta, low_memory=False)
        meta["track_id"] = meta["track_id"].astype(str)
        if "fold" in meta.columns:
            for r in meta.itertuples():
                if r.label == "real" and int(r.fold) >= 0:
                    fold_of[str(r.track_id)] = int(r.fold)
            logger.info("real fold assignment read from trajectory meta for %d tracks", len(fold_of))
    n_folds = len(flows)
    n_missing_real_fold = 0

    def _fold_for(track_id: str, label: str) -> int:
        if label == "real":
            f = fold_of.get(track_id)
            if f is not None:
                return f % n_folds
            return -1  # cannot score a real track without knowing which flow held it out
        # fakes: deterministic, stable assignment (never in training either way)
        return int(hashlib.md5(track_id.encode()).hexdigest(), 16) % n_folds

    cache_root = Path(args.emb_cache)
    win = max(int(args.window_duration * ENCODEC_FPS), 5)
    hop = max(int(args.hop_duration * ENCODEC_FPS), 1)
    mean_col = f"{args.embedding}_wf_mean"
    std_col = f"{args.embedding}_wf_std"

    rows = []
    traj_store: dict[str, np.ndarray] = {}
    traj_meta: list[dict] = []
    n_missing_cache = 0
    for i, r in enumerate(df.itertuples(index=False), start=1):
        tid, label = str(r.track_id), str(r.label)
        fold = _fold_for(tid, label)
        if fold < 0:
            n_missing_real_fold += 1
            continue
        cp = _cache_path(cache_root, tid, args.max_duration, args.embedding)
        if not cp.exists():
            n_missing_cache += 1
            continue
        try:
            mat = np.load(str(cp))
            pw = pool_windows(mat, win, hop)
            if len(pw) < 2:
                continue
            ll = -flows[fold].score_samples(pw)  # score_samples = NLL -> negate for log-lik
            ll = ll[np.isfinite(ll)]
            if len(ll) < 2:
                continue
            alg = getattr(r, "algorithm", "")
            rows.append(
                {
                    "track_id": tid,
                    "label": label,
                    "algorithm": alg,
                    "fold": fold,
                    mean_col: float(ll.mean()),
                    std_col: float(ll.std()),
                    f"{args.embedding}_wf_n_windows": int(len(ll)),
                }
            )
            if args.save_trajectories:
                key = f"{label}|{alg or ''}|{tid}"
                traj_store[key] = -ll  # anomaly-oriented, matching ablate_aggregation.py
                traj_meta.append(
                    {
                        "key": key,
                        "track_id": tid,
                        "label": label,
                        "algorithm": alg,
                        "fold": fold,
                        "n_windows": int(len(ll)),
                    }
                )
        except Exception as exc:
            logger.debug("scoring failed for %s: %s", tid, exc)
        if i % 5000 == 0:
            logger.info("[%d/%d] scored=%d", i, len(df), len(rows))

    if n_missing_real_fold:
        logger.warning(
            "%d real tracks had no known held-out fold (no traj-meta entry) and were SKIPPED — "
            "scoring them with an arbitrary fold would leak training data.",
            n_missing_real_fold,
        )
    if n_missing_cache:
        logger.warning("%d tracks missing from the embedding cache", n_missing_cache)

    if args.save_trajectories and traj_store:
        td = Path(args.save_trajectories)
        td.mkdir(parents=True, exist_ok=True)
        if len(traj_store) != len(traj_meta):
            logger.error("trajectory key collision: %d keys for %d rows", len(traj_store), len(traj_meta))
        np.savez_compressed(td / f"wf_trajectories_{args.embedding}.npz", **traj_store)
        pd.DataFrame(traj_meta).to_csv(td / f"wf_trajectories_{args.embedding}_meta.csv", index=False)
        logger.info("saved %d symmetric trajectories → %s", len(traj_store), td)

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.info(
        "Saved %s (%d rows: %d real, %d fake)",
        args.output_csv,
        len(out),
        int((out["label"] == "real").sum()),
        int((out["label"] == "fake").sum()),
    )

    # --- sanity: per-generator AUC under this symmetric table ---
    real_s = -out.loc[out["label"] == "real", mean_col].to_numpy(float)
    aucs = []
    for alg in sorted(a for a in out.loc[out["label"] == "fake", "algorithm"].unique() if str(a).strip()):
        fake_s = -out.loc[(out["label"] == "fake") & (out["algorithm"] == alg), mean_col].to_numpy(float)
        if len(fake_s) < 5:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
        logger.info("  %-22s AUC=%.4f  EER=%.2f%%  (n_fake=%d)", alg, auc, eer * 100, len(fake_s))
        aucs.append(auc)
    if aucs:
        fake_all = -out.loc[out["label"] == "fake", mean_col].to_numpy(float)
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_all))]
        auc, eer = auc_and_eer(y, np.r_[real_s, fake_all])
        logger.info(
            "  MACRO AUC=%.4f | POOLED AUC=%.4f EER=%.2f%% " "(compare MusicDET pooled EER 9.72%% on this corpus)",
            float(np.mean(aucs)),
            auc,
            eer * 100,
        )


if __name__ == "__main__":
    main()
