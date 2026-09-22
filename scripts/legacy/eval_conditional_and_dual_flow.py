"""Evaluate Conditional RealNVP (genre-conditioned) and Dual Flow (real+fake).

Plan Phase C1 + C2 experiment runner.

C1 – Conditional flow:
  Trains ConditionalRealNVPOneClass conditioned on CLAP genre posteriors.
  Compares per-generator AUC: conditional vs unconditional.
  Uses the same 5-fold LOGO protocol as the base window-flow.

  Scientific purpose:
  - If AUC is similar to unconditional: genre conditioning does NOT help.
    This confirms the flow is learning codec artifacts, not genre mapping.
  - If AUC improves within-genre: genre-awareness is beneficial for FPR control.

C2 – Dual flow (supervised upper-bound ablation):
  Trains a second RealNVP on FAKE windows (4 generators per LOGO fold).
  Detection score = logp_fake - logp_real (likelihood ratio).
  Reports headroom above real-only flow (the headline method).
  The real-only flow remains the headline contribution.

C3 (optional) – Alternative codec: runs the same window-flow on
  X-Codec or MERT latents specifically for udio-30s.

Usage (EC2)
-----------
# C1 + C2 (requires genre tags from tag_all_genres.py):
poetry run python scripts/eval_conditional_and_dual_flow.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --output-dir data/processed/conditional_dual_flow \\
    --device cuda \\
    --run-c1 --run-c2

# C1 only:
poetry run python scripts/eval_conditional_and_dual_flow.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --output-dir data/processed/conditional_dual_flow \\
    --device cuda --run-c1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, bootstrap_auc
from intrinsic_ai_music_detection.models.flow import (
    ConditionalRealNVPConfig,
    ConditionalRealNVPOneClass,
    DualFlowDetector,
    RealNVPConfig,
    RealNVPOneClass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ENCODEC_FPS = 75
WINDOW_DUR = 4.0
HOP_DUR = 2.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_cache_health(
    track_ids: list[str],
    emb_cache: Path,
    max_duration: float,
    n_probe: int = 50,
    min_hit_rate: float = 0.5,
) -> None:
    """Fail fast if the embedding cache key doesn't match (see coverage_curve.py)."""
    probe_ids = track_ids[:n_probe]
    hits = 0
    for tid in probe_ids:
        key = hashlib.md5(f"{tid}_{24000}_{max_duration}_preprocessed".encode()).hexdigest()
        if (emb_cache / "encodec" / f"{key}.npy").exists():
            hits += 1
    hit_rate = hits / max(len(probe_ids), 1)
    logger.info(
        "Cache health check: %d/%d probed tracks hit the cache (%.0f%%) at %s",
        hits,
        len(probe_ids),
        100 * hit_rate,
        emb_cache / "encodec",
    )
    if hit_rate < min_hit_rate:
        logger.error(
            "Cache hit rate %.0f%% is below the %.0f%% threshold. Check --max-duration "
            "(currently %.1f) and --emb-cache match how the cache was built "
            "(run_full_sonics_pipeline.sh: --analysis-duration, --preprocess-mode preprocessed). "
            "Aborting before wasting compute.",
            100 * hit_rate,
            100 * min_hit_rate,
            max_duration,
        )
        sys.exit(1)


def _pool_windows(emb: np.ndarray, wf: int, hf: int) -> np.ndarray:
    pooled = []
    n = len(emb)
    start = 0
    while start + wf <= n:
        sub = emb[start : start + wf]
        fin = sub[np.isfinite(sub).all(axis=1)]
        if len(fin) >= 1:
            pooled.append(fin.mean(axis=0))
        start += hf
    return np.array(pooled, dtype=np.float64) if pooled else np.empty((0, emb.shape[1]))


def _load_windows(track_id: str, emb_cache: Path, max_dur: float) -> np.ndarray | None:
    wf = max(int(WINDOW_DUR * ENCODEC_FPS), 5)
    hf = max(int(HOP_DUR * ENCODEC_FPS), 1)
    key = hashlib.md5(f"{track_id}_{24000}_{max_dur}_preprocessed".encode()).hexdigest()
    cp = emb_cache / "encodec" / f"{key}.npy"
    if not cp.exists():
        return None
    try:
        mat = np.load(str(cp))
        pw = _pool_windows(mat, wf, hf)
        return pw if len(pw) >= 2 else None
    except Exception:
        return None


def _collect_windows(
    track_ids: list[str],
    emb_cache: Path,
    max_dur: float,
) -> tuple[np.ndarray, list[str]]:
    """Returns (stacked_windows [N, D], track_ids_for_each_window)."""
    all_wins, all_ids = [], []
    for tid in track_ids:
        pw = _load_windows(tid, emb_cache, max_dur)
        if pw is not None:
            all_wins.append(pw)
            all_ids.extend([tid] * len(pw))
    if not all_wins:
        return np.empty((0, 128)), []
    return np.concatenate(all_wins, axis=0), all_ids


def _score_tracks_with_flow(
    track_ids: list[str],
    emb_cache: Path,
    flow,
    max_dur: float,
    context_map: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Score tracks using a RealNVPOneClass or ConditionalRealNVPOneClass."""
    scores = []
    is_conditional = hasattr(flow, "log_likelihood") and context_map is not None
    for tid in track_ids:
        pw = _load_windows(tid, emb_cache, max_dur)
        if pw is None:
            scores.append(float("nan"))
            continue
        try:
            if is_conditional and tid in context_map:
                ctx = np.tile(context_map[tid], (len(pw), 1))
                lls = -flow.score_samples(pw, ctx)
            else:
                lls = -flow.score_samples(pw)
            fin = lls[np.isfinite(lls)]
            scores.append(float(-fin.mean()) if len(fin) >= 2 else float("nan"))
        except Exception:
            scores.append(float("nan"))
    return np.array(scores)


# ---------------------------------------------------------------------------
# C1 — Conditional flow LOGO evaluation
# ---------------------------------------------------------------------------


def eval_conditional_flow(
    manifest: pd.DataFrame,
    genres_df: pd.DataFrame,
    emb_cache: Path,
    output_dir: Path,
    device: str,
    max_dur: float,
    flow_epochs: int,
    n_bootstrap: int = 1000,
    generators_filter: list[str] | None = None,
    limit_real_tracks: int | None = None,
    n_coupling_layers: int = 12,
) -> pd.DataFrame:
    """5-fold LOGO of ConditionalRealNVP vs unconditional RealNVP."""
    logger.info("=" * 60)
    logger.info("C1: Conditional Flow LOGO evaluation")
    logger.info("=" * 60)

    # NOTE: intentionally keep BOTH real and fake rows here. tag_all_genres.py used
    # the same shared CLAP tagger for real and fake tracks precisely so fakes can be
    # conditioned on their own measured genre at inference (matching the docstring's
    # "context is the CLAP genre posterior of the track being scored"). Restricting
    # this to real-only silently forces every fake track onto an out-of-distribution
    # all-zero context vector, which biases (and can numerically destabilize) the
    # conditional-flow comparison.
    genres_df = genres_df.copy()
    genres_df["track_id"] = genres_df["track_id"].astype(str)
    manifest["track_id"] = manifest["track_id"].astype(str)

    # Build context map: track_id -> CLAP genre posterior vector [n_genres]
    context_map: dict[str, np.ndarray] = {}
    n_bad_vec = 0
    for _, row in genres_df.iterrows():
        if "clap_genre_vector" in row and pd.notna(row["clap_genre_vector"]):
            try:
                vec = np.array(json.loads(row["clap_genre_vector"]), dtype=np.float32)
                if not np.isfinite(vec).all():
                    n_bad_vec += 1
                    continue
                context_map[row["track_id"]] = vec
            except Exception:
                pass
    if n_bad_vec:
        logger.warning("Skipped %d tracks with non-finite CLAP genre vectors", n_bad_vec)
    n_real_ctx = sum(1 for tid in context_map if tid in set(manifest.loc[manifest["label"] == "real", "track_id"]))
    logger.info(
        "Genre context available for %d tracks (%d real, %d fake)",
        len(context_map),
        n_real_ctx,
        len(context_map) - n_real_ctx,
    )

    if not context_map:
        logger.error("No CLAP genre vectors found. Run tag_all_genres.py with clap_genre_vector output first.")
        return pd.DataFrame()

    real_ids = manifest.loc[manifest["label"] == "real", "track_id"].tolist()
    if limit_real_tracks is not None:
        rng = np.random.default_rng(42)
        real_ids = list(rng.choice(real_ids, size=min(limit_real_tracks, len(real_ids)), replace=False))
        logger.warning(
            "SMOKE-TEST MODE: limited to %d real tracks — AUC numbers are NOT final results.",
            len(real_ids),
        )
    generators = sorted(manifest.loc[manifest["label"] == "fake", "algorithm"].dropna().unique())
    if generators_filter:
        generators = [g for g in generators if g in set(generators_filter)]
        logger.info("Restricting to generators: %s", generators)
    n_ctx = len(next(iter(context_map.values())))

    rows = []
    for held_out_gen in generators:
        logger.info("\n  Held-out generator: %s", held_out_gen)

        fake_ids = manifest.loc[
            (manifest["label"] == "fake") & (manifest["algorithm"] == held_out_gen), "track_id"
        ].tolist()

        # --- Collect real windows + context ---
        real_wins, real_win_ids = _collect_windows(real_ids, emb_cache, max_dur)
        ctx_matrix = np.array([context_map.get(tid, np.zeros(n_ctx)) for tid in real_win_ids], dtype=np.float32)

        if len(real_wins) < 100:
            logger.warning("    Not enough real windows; skipping")
            continue

        # --- 5-fold LOGO for real tracks (to score held-out real) ---
        real_ids_arr = np.array(real_ids)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        real_logo_scores: list[tuple[str, float, float]] = []  # (track_id, uncond, cond)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(real_ids_arr)):
            train_real = real_ids_arr[train_idx].tolist()
            test_real = real_ids_arr[test_idx].tolist()

            # Gather train windows
            tr_wins, tr_wids = _collect_windows(train_real, emb_cache, max_dur)
            tr_ctx = np.array([context_map.get(tid, np.zeros(n_ctx)) for tid in tr_wids], dtype=np.float32)
            if len(tr_wins) < 50:
                continue

            # Unconditional flow
            ucfg = RealNVPConfig(
                device=device,
                n_epochs=flow_epochs,
                patience=15,
                seed=42 + fold_idx,
                n_coupling_layers=n_coupling_layers,
            )
            uf = RealNVPOneClass(ucfg).fit(tr_wins)

            # Conditional flow
            ccfg = ConditionalRealNVPConfig(
                device=device,
                n_epochs=flow_epochs,
                patience=15,
                seed=42 + fold_idx,
                context_dim=min(64, n_ctx),
                n_coupling_layers=n_coupling_layers,
            )
            cf = ConditionalRealNVPOneClass(ccfg).fit(tr_wins, tr_ctx)

            # Score held-out real
            for tid in test_real:
                pw = _load_windows(tid, emb_cache, max_dur)
                if pw is None:
                    continue
                # Unconditional
                u_ll = -uf.score_samples(pw)
                u_score = float(-u_ll[np.isfinite(u_ll)].mean()) if len(u_ll) >= 2 else float("nan")
                # Conditional
                ctx = np.tile(context_map.get(tid, np.zeros(n_ctx)), (len(pw), 1))
                c_ll = -cf.score_samples(pw, ctx)
                c_score = float(-c_ll[np.isfinite(c_ll)].mean()) if len(c_ll) >= 2 else float("nan")
                real_logo_scores.append((tid, u_score, c_score))

        # --- Score fake tracks (full real flow) ---
        # Use all real tracks for training (fake generator is held out by construction)
        ucfg_full = RealNVPConfig(
            device=device, n_epochs=flow_epochs, patience=15, seed=42, n_coupling_layers=n_coupling_layers
        )
        uf_full = RealNVPOneClass(ucfg_full).fit(real_wins)
        ccfg_full = ConditionalRealNVPConfig(
            device=device,
            n_epochs=flow_epochs,
            patience=15,
            seed=42,
            context_dim=min(64, n_ctx),
            n_coupling_layers=n_coupling_layers,
        )
        cf_full = ConditionalRealNVPOneClass(ccfg_full).fit(real_wins, ctx_matrix)

        fake_u_scores = _score_tracks_with_flow(fake_ids, emb_cache, uf_full, max_dur)
        fake_c_scores = _score_tracks_with_flow(fake_ids, emb_cache, cf_full, max_dur, context_map)

        # AUC computation
        real_u = np.array([s[1] for s in real_logo_scores if np.isfinite(s[1])])
        real_c = np.array([s[2] for s in real_logo_scores if np.isfinite(s[2])])
        fake_u = fake_u_scores[np.isfinite(fake_u_scores)]
        fake_c = fake_c_scores[np.isfinite(fake_c_scores)]

        if len(fake_u) >= 5 and len(real_u) >= 5:
            y_u = np.r_[np.zeros(len(real_u)), np.ones(len(fake_u))]
            s_u = np.r_[real_u, fake_u]
            auc_u, eer_u = auc_and_eer(y_u, s_u)
        else:
            auc_u = eer_u = float("nan")

        if len(fake_c) >= 5 and len(real_c) >= 5:
            y_c = np.r_[np.zeros(len(real_c)), np.ones(len(fake_c))]
            s_c = np.r_[real_c, fake_c]
            auc_c, eer_c = auc_and_eer(y_c, s_c)
        else:
            auc_c = eer_c = float("nan")

        logger.info(
            "  %-25s  Uncond AUC=%.4f EER=%.1f%%  Cond AUC=%.4f EER=%.1f%%  ΔAUC=%+.4f",
            held_out_gen,
            auc_u,
            eer_u * 100,
            auc_c,
            eer_c * 100,
            auc_c - auc_u,
        )
        rows.append(
            {
                "algorithm": held_out_gen,
                "unconditional_auc": round(auc_u, 4),
                "unconditional_eer_pct": round(eer_u * 100, 2),
                "conditional_auc": round(auc_c, 4),
                "conditional_eer_pct": round(eer_c * 100, 2),
                "delta_auc": round(auc_c - auc_u, 4),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "c1_conditional_vs_unconditional.csv", index=False)
    logger.info("\nC1 results saved to c1_conditional_vs_unconditional.csv")
    return df


# ---------------------------------------------------------------------------
# C2 — Dual flow LOGO evaluation
# ---------------------------------------------------------------------------


def eval_dual_flow(
    manifest: pd.DataFrame,
    emb_cache: Path,
    output_dir: Path,
    device: str,
    max_dur: float,
    flow_epochs: int,
    n_coupling_layers: int = 12,
) -> pd.DataFrame:
    """5-fold LOGO: dual flow (real+fake) vs real-only flow."""
    logger.info("=" * 60)
    logger.info("C2: Dual Flow LOGO evaluation (real+fake likelihood ratio)")
    logger.info("=" * 60)

    manifest["track_id"] = manifest["track_id"].astype(str)
    real_ids = manifest.loc[manifest["label"] == "real", "track_id"].tolist()
    generators = sorted(manifest.loc[manifest["label"] == "fake", "algorithm"].dropna().unique())

    rows = []
    for held_out_gen in generators:
        logger.info("\n  Held-out generator: %s", held_out_gen)

        test_fake_ids = manifest.loc[
            (manifest["label"] == "fake") & (manifest["algorithm"] == held_out_gen),
            "track_id",
        ].tolist()
        train_fake_ids = manifest.loc[
            (manifest["label"] == "fake") & (manifest["algorithm"] != held_out_gen),
            "track_id",
        ].tolist()

        fake_wins, _ = _collect_windows(train_fake_ids, emb_cache, max_dur)
        if len(fake_wins) < 100:
            logger.warning("    Not enough fake windows for training; skipping")
            continue

        # --- Proper 5-fold LOGO for REAL tracks: retrain per fold so held-out
        # real tracks are NEVER seen during training of the flow that scores
        # them (avoids train/test leakage; matches the plan's guardrail
        # "5-fold LOGO with held-out real scoring"). ---
        real_ids_arr = np.array(real_ids)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        real_real_scores: list[float] = []
        real_dual_scores: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(real_ids_arr)):
            train_real = real_ids_arr[train_idx].tolist()
            test_real = real_ids_arr[test_idx].tolist()

            tr_wins, _ = _collect_windows(train_real, emb_cache, max_dur)
            if len(tr_wins) < 50:
                continue

            rcfg = RealNVPConfig(
                device=device,
                n_epochs=flow_epochs,
                patience=15,
                seed=42 + fold_idx,
                n_coupling_layers=n_coupling_layers,
            )
            fold_real_flow = RealNVPOneClass(rcfg).fit(tr_wins)

            fold_dual = DualFlowDetector(
                real_cfg=RealNVPConfig(
                    device=device,
                    n_epochs=flow_epochs,
                    patience=15,
                    seed=42 + fold_idx,
                    n_coupling_layers=n_coupling_layers,
                ),
                fake_cfg=RealNVPConfig(
                    device=device,
                    n_epochs=flow_epochs,
                    patience=15,
                    seed=43 + fold_idx,
                    n_coupling_layers=n_coupling_layers,
                ),
            )
            fold_dual.fit_real(tr_wins).fit_fake(fake_wins)

            for tid in test_real:
                pw = _load_windows(tid, emb_cache, max_dur)
                if pw is None:
                    continue
                ll = -fold_real_flow.score_samples(pw)
                fin = ll[np.isfinite(ll)]
                real_real_scores.append(float(-fin.mean()) if len(fin) >= 2 else float("nan"))
                dl = fold_dual.score_samples(pw)
                dfin = dl[np.isfinite(dl)]
                real_dual_scores.append(float(dfin.mean()) if len(dfin) >= 2 else float("nan"))

        # --- Score held-out fake generator with flows trained on ALL real
        # tracks (fine here: the held-out GENERATOR, not a real track, is what
        # must be unseen — matches the base window-flow's own LOGO protocol). ---
        real_wins_full, _ = _collect_windows(real_ids, emb_cache, max_dur)
        if len(real_wins_full) < 100:
            logger.warning("    Not enough real windows for full-corpus flow; skipping fake scoring")
            fake_real_scores = np.array([])
            fake_dual_scores = np.array([])
        else:
            rcfg_full = RealNVPConfig(
                device=device, n_epochs=flow_epochs, patience=15, seed=42, n_coupling_layers=n_coupling_layers
            )
            real_flow_full = RealNVPOneClass(rcfg_full).fit(real_wins_full)
            dual_full = DualFlowDetector(
                real_cfg=RealNVPConfig(
                    device=device, n_epochs=flow_epochs, patience=15, seed=42, n_coupling_layers=n_coupling_layers
                ),
                fake_cfg=RealNVPConfig(
                    device=device, n_epochs=flow_epochs, patience=15, seed=43, n_coupling_layers=n_coupling_layers
                ),
            )
            dual_full.fit_real(real_wins_full).fit_fake(fake_wins)

            fake_real_scores = _score_tracks_with_flow(test_fake_ids, emb_cache, real_flow_full, max_dur)
            fake_dual_scores_list = []
            for tid in test_fake_ids:
                pw = _load_windows(tid, emb_cache, max_dur)
                if pw is None:
                    fake_dual_scores_list.append(float("nan"))
                    continue
                dl = dual_full.score_samples(pw)
                dfin = dl[np.isfinite(dl)]
                fake_dual_scores_list.append(float(dfin.mean()) if len(dfin) >= 2 else float("nan"))
            fake_dual_scores = np.array(fake_dual_scores_list)

        rr = np.array([s for s in real_real_scores if np.isfinite(s)])
        rd = np.array([s for s in real_dual_scores if np.isfinite(s)])
        fr = fake_real_scores[np.isfinite(fake_real_scores)] if len(fake_real_scores) else np.array([])
        fd = fake_dual_scores[np.isfinite(fake_dual_scores)] if len(fake_dual_scores) else np.array([])

        auc_r = eer_r = auc_d = eer_d = float("nan")
        if len(rr) >= 5 and len(fr) >= 5:
            auc_r, eer_r = auc_and_eer(np.r_[np.zeros(len(rr)), np.ones(len(fr))], np.r_[rr, fr])
        if len(rd) >= 5 and len(fd) >= 5:
            auc_d, eer_d = auc_and_eer(np.r_[np.zeros(len(rd)), np.ones(len(fd))], np.r_[rd, fd])

        logger.info(
            "  %-25s  Real-only AUC=%.4f EER=%.1f%%  Dual AUC=%.4f EER=%.1f%%  ΔAUC=%+.4f",
            held_out_gen,
            auc_r,
            eer_r * 100,
            auc_d,
            eer_d * 100,
            auc_d - auc_r,
        )
        rows.append(
            {
                "algorithm": held_out_gen,
                "realonly_auc": round(auc_r, 4),
                "realonly_eer_pct": round(eer_r * 100, 2),
                "dual_auc": round(auc_d, 4),
                "dual_eer_pct": round(eer_d * 100, 2),
                "delta_auc": round(auc_d - auc_r, 4),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "c2_dual_vs_realonly.csv", index=False)
    logger.info("\nC2 results saved to c2_dual_vs_realonly.csv")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Conditional (C1) and Dual (C2) flow detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--window-flow-csv", required=True)
    parser.add_argument("--emb-cache", required=True)
    parser.add_argument("--sonics-manifest", required=True)
    parser.add_argument("--genre-csv", default=None, help="Required for C1")
    parser.add_argument("--output-dir", default="data/processed/conditional_dual_flow")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--flow-epochs", type=int, default=100, help="Training epochs per flow (100 sufficient for ablation)"
    )
    parser.add_argument(
        "--n-coupling-layers",
        type=int,
        default=12,
        help=(
            "RealNVP depth for every flow trained here. Default 12 = the adopted headline "
            "config. The original C1/C2 results were produced at the then-default K=8 — "
            "pass 8 to reproduce those."
        ),
    )
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--run-c1", action="store_true", help="Run C1 conditional flow experiment")
    parser.add_argument("--run-c2", action="store_true", help="Run C2 dual flow experiment")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help="Restrict LOGO loop to these held-out generators only (fast smoke test).",
    )
    parser.add_argument(
        "--limit-real-tracks",
        type=int,
        default=None,
        help="Subsample to N real tracks for a fast smoke test (results are NOT final).",
    )
    args = parser.parse_args()

    if not args.run_c1 and not args.run_c2:
        parser.error("Specify --run-c1 and/or --run-c2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.sonics_manifest, low_memory=False)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"]
    manifest["track_id"] = manifest["track_id"].astype(str)

    emb_cache = Path(args.emb_cache)
    _check_cache_health(
        manifest.loc[manifest["label"] == "real", "track_id"].astype(str).tolist(),
        emb_cache,
        args.max_duration,
    )

    if args.run_c1:
        if args.genre_csv is None or not Path(args.genre_csv).exists():
            logger.error("--genre-csv required for C1. Run tag_all_genres.py first.")
            sys.exit(1)
        genres_df = pd.read_csv(args.genre_csv, low_memory=False)
        eval_conditional_flow(
            manifest=manifest,
            genres_df=genres_df,
            emb_cache=emb_cache,
            output_dir=output_dir,
            device=args.device,
            max_dur=args.max_duration,
            flow_epochs=args.flow_epochs,
            n_bootstrap=args.n_bootstrap,
            generators_filter=args.generators,
            limit_real_tracks=args.limit_real_tracks,
            n_coupling_layers=args.n_coupling_layers,
        )

    if args.run_c2:
        eval_dual_flow(
            manifest=manifest,
            emb_cache=emb_cache,
            output_dir=output_dir,
            device=args.device,
            max_dur=args.max_duration,
            flow_epochs=args.flow_epochs,
            n_coupling_layers=args.n_coupling_layers,
        )

    logger.info("\nAll outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
