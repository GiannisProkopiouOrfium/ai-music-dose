"""Leave-one-subdomain-out evaluation (MusicDET Table 4 analog).

MusicDET's Table 4 excludes a subdomain (a "jazz" genre subset and a
"piano"-focused subset, respectively) from training and evaluates the
resulting model on that subdomain's held-out real + AI-generated tracks, per
generator, to test whether the detector generalizes to unseen musical content
rather than overfitting to the training genre mix.

This is our closest analog on the SONICS domain, reusing the CLAP genre tags
already computed for the confound analysis (A1/A2) and the cached EnCodec
embeddings from the full pipeline run (no re-extraction needed -- retraining a
window-flow on a filtered subset of already-cached real windows is fast):

  1. Take the SONICS canonical manifest + CLAP genre tags.
  2. For --exclude-genre (default: jazz, then classical -- see note below):
     drop every REAL track whose CLAP top-1 genre matches the excluded genre
     from the training manifest (fakes are left in the manifest so
     run_balanced_ablation.py's window-flow-eval still scores them normally).
  3. Call run_balanced_ablation.py as a subprocess to train + save a flow on
     the filtered real pool (reusing the existing embedding cache).
  4. Directly score the EXCLUDED genre's real tracks (truly held out -- never
     seen during training) from the same embedding cache using the saved
     flow, replicating run_balanced_ablation.py's own window-pooling/cache-key
     logic exactly so the numbers are apples-to-apples with the headline.
  5. Report per-generator AUC/EER restricted to the excluded genre's real vs.
     fake tracks -- the direct analog of MusicDET's Table 4 columns.

NOTE on genre choice: MusicDET excludes "jazz" (a genre) and "piano" (an
instrumentation-focused subset). Our CLAP genre vocabulary has a "jazz" bucket
(exact match) but no "piano" bucket (we tag genre, not instrumentation) -- we
substitute "classical" as the second excluded subdomain, the closest available
analog (small, acoustic, non-electronic, and distinct from the bulk of SONICS'
rock/pop/electronic-leaning training distribution, same as piano would be).
This substitution is stated explicitly in the paper writeup, not hidden.

Usage
-----
python scripts/run_leave_one_genre_out.py \\
    --canonical-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --embedding-cache-dir data/emb_cache_encodec_full \\
    --exclude-genre jazz \\
    --output-dir data/processed/leave_one_genre_out/jazz \\
    --n-coupling-layers 12 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _pool_windows(mat: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = mat.shape[0]
    if n < win:
        return mat.mean(axis=0, keepdims=True)
    starts = list(range(0, n - win + 1, hop))
    return np.stack([mat[s : s + win].mean(axis=0) for s in starts], axis=0)


def _cache_path(cache_dir: Path, track_id: str, target_sr: int, max_duration: float, preprocess_mode: str) -> Path:
    # Must match run_balanced_ablation.py's _run_window_flow_eval cache-key formula
    # exactly (lp_tag/fd_tag/reg_tag empty here since we don't use lowpass/fixed-
    # duration/region variants for this experiment).
    key = hashlib.md5(f"{track_id}_{target_sr}_{max_duration}_{preprocess_mode}".encode()).hexdigest()
    return cache_dir / f"{key}.npy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical-manifest", required=True)
    ap.add_argument("--genre-csv", required=True, help="track_id, genre_top1 columns (all_genres.csv).")
    ap.add_argument("--embedding-cache-dir", required=True, help="Base cache dir (contains an 'encodec' subdir).")
    ap.add_argument("--exclude-genre", required=True, help="e.g. jazz or classical")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--analysis-duration", type=float, default=55.0)
    ap.add_argument("--wf-pca-components", type=int, default=128)
    ap.add_argument("--wf-flow-epochs", type=int, default=200)
    ap.add_argument(
        "--n-coupling-layers", type=int, default=12, help="Matches the K=12 headline (see run_headline_k12_suite.sh)."
    )
    ap.add_argument("--wf-window-duration", type=float, default=4.0)
    ap.add_argument("--wf-hop-duration", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.canonical_manifest, low_memory=False)
    manifest = manifest[manifest.get("status", "ok") == "ok"] if "status" in manifest.columns else manifest
    genres = pd.read_csv(args.genre_csv, low_memory=False)
    genres["track_id"] = genres["track_id"].astype(str)
    manifest["track_id"] = manifest["track_id"].astype(str)
    genre_by_track = dict(zip(genres["track_id"], genres["genre_top1"].astype(str).str.lower()))

    target = args.exclude_genre.lower()
    is_real = manifest["label"] == "real"
    manifest["_genre"] = manifest["track_id"].map(genre_by_track).fillna("")
    excluded_real_mask = is_real & (manifest["_genre"].str.lower() == target)
    n_excluded = int(excluded_real_mask.sum())
    logger.info(
        "Excluding %d real tracks tagged genre=%r from training (out of %d total real tracks)",
        n_excluded,
        target,
        int(is_real.sum()),
    )
    if n_excluded < 10:
        raise SystemExit(
            f"Only {n_excluded} real tracks found for genre={target!r} — too few for a meaningful "
            "held-out subdomain test. Check --genre-csv coverage / genre name spelling."
        )

    filtered_manifest_path = out_dir / f"canonical_manifest_excl_{target}.csv"
    manifest[~excluded_real_mask].drop(columns=["_genre"]).to_csv(filtered_manifest_path, index=False)
    logger.info("Wrote filtered training manifest -> %s (%d rows)", filtered_manifest_path, (~excluded_real_mask).sum())

    # --- Step 1: train + save the flow on the genre-excluded real pool -----
    flow_path_arg = out_dir / "loso_flow.pt"  # run_balanced_ablation.py appends _encodec
    flow_path_final = out_dir / "loso_flow_encodec.pt"
    if not flow_path_final.exists():
        cmd = [
            sys.executable,
            "scripts/run_balanced_ablation.py",
            "--embeddings",
            "encodec",
            "--per-stratum",
            "50000",
            "--analysis-duration",
            str(args.analysis_duration),
            "--window-duration",
            "4",
            "--hop-duration",
            "2",
            "--estimators",
            "twonn",
            "--canonical-manifest",
            str(filtered_manifest_path),
            "--preprocess-mode",
            "preprocessed",
            "--embedding-cache-dir",
            args.embedding_cache_dir,
            "--device",
            args.device,
            "--window-flow-eval",
            "--wf-window-duration",
            str(args.wf_window_duration),
            "--wf-hop-duration",
            str(args.wf_hop_duration),
            "--wf-pca-components",
            str(args.wf_pca_components),
            "--wf-flow-epochs",
            str(args.wf_flow_epochs),
            "--wf-n-coupling-layers",
            str(args.n_coupling_layers),
            "--wf-save-flow-path",
            str(flow_path_arg),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(out_dir),
        ]
        logger.info("Training genre-excluded flow: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        logger.info("Flow checkpoint already exists at %s — skipping training.", flow_path_final)

    # --- Step 2: directly score the EXCLUDED genre's held-out real tracks --
    from intrinsic_ai_music_detection.models.flow import RealNVPOneClass

    flow = RealNVPOneClass.load(str(flow_path_final), device=args.device)
    cache_dir = Path(args.embedding_cache_dir) / "encodec"
    target_sr = 24_000

    excluded_real_ids = manifest.loc[excluded_real_mask, "track_id"].tolist()
    real_scores = []
    for tid in excluded_real_ids:
        cp = _cache_path(cache_dir, tid, target_sr, args.analysis_duration, "preprocessed")
        if not cp.exists():
            continue
        try:
            mat = np.load(str(cp))
            fps = mat.shape[0] / max(args.analysis_duration, 1.0)
            wf = max(int(args.wf_window_duration * fps), 5)
            hf = max(int(args.wf_hop_duration * fps), 1)
            pw = _pool_windows(mat, wf, hf)
            # RealNVPOneClass.score_samples() returns the anomaly score directly
            # (negative log-likelihood; higher = more anomalous/AI-like) — no
            # extra negation needed here (unlike wf_mean in window_flow_eval.csv,
            # which stores raw log-likelihood and must be negated for AUC).
            anomaly = flow.score_samples(pw)
            real_scores.append(float(np.nanmean(anomaly)))
        except Exception as exc:
            logger.debug("Scoring failed for held-out real %s: %s", tid, exc)
    real_scores = np.array([s for s in real_scores if np.isfinite(s)])
    logger.info(
        "Scored %d/%d held-out genre=%r real tracks (never seen in training)",
        len(real_scores),
        len(excluded_real_ids),
        target,
    )

    # --- Step 3: fakes for this genre, from the already-produced window_flow_eval.csv
    wf_eval_path = out_dir / "window_flow_eval.csv"
    wf_eval = pd.read_csv(wf_eval_path, low_memory=False)
    wf_eval["track_id"] = wf_eval["track_id"].astype(str)
    mean_col = next((c for c in wf_eval.columns if c.endswith("wf_mean")), None)
    wf_eval["_genre"] = wf_eval["track_id"].map(genre_by_track).fillna("")
    fake_genre = wf_eval[(wf_eval["label"] == "fake") & (wf_eval["_genre"].str.lower() == target)]

    from sklearn.metrics import roc_auc_score, roc_curve

    rows = []
    for alg in sorted(fake_genre["algorithm"].dropna().unique()):
        fake_scores = -fake_genre.loc[fake_genre["algorithm"] == alg, mean_col].dropna().to_numpy()
        if len(fake_scores) < 5 or len(real_scores) < 5:
            continue
        y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
        s = np.r_[real_scores, fake_scores]
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        rows.append(
            {
                "excluded_genre": target,
                "algorithm": alg,
                "n_real_heldout": len(real_scores),
                "n_fake": len(fake_scores),
                "auc": round(auc, 4),
                "eer_pct": round(eer * 100, 2),
            }
        )
        logger.info(
            "  [genre=%s excluded] %-22s AUC=%.4f EER=%.2f%% (n_real=%d n_fake=%d)",
            target,
            alg,
            auc,
            eer * 100,
            len(real_scores),
            len(fake_scores),
        )

    out_csv = out_dir / f"loso_{target}_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved -> %s", out_csv)


if __name__ == "__main__":
    main()
