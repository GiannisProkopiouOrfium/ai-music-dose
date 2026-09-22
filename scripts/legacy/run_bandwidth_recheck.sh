#!/usr/bin/env bash
# =============================================================================
# run_bandwidth_recheck.sh
#
# Phase A4: Re-run the EnCodec+RealNVP window-flow on bandwidth-equalized audio
# (8 kHz low-pass filter applied before encoding) to confirm that the AUC
# survives the bandwidth confound that broke Phase-2 spectral features.
#
# Motivation:
#   Phase 2 (spectral probe) showed a 6-8 kHz band-edge artifact in the fake
#   delivery chain which drove AUC 0.96-0.97 on spectral features.  That was
#   a bandwidth confound.  The question for Phase 4 (EnCodec flow) is:
#   does the AUC also depend on bandwidth differences, or is it detecting
#   codec-quantization artifacts that persist after bandwidth equalization?
#
#   Expected outcome: AUC should remain > 0.85 on udio-120s after LP-8kHz
#   (the udio-120s detection via codec latents is a different mechanism from
#   the spectral band-edge).
#
# This runs run_balanced_ablation.py with --lowpass-hz 8000, producing a
# separate output directory for comparison with the canonical (no LP) results.
#
# Usage:
#   bash scripts/run_bandwidth_recheck.sh
#   # (run in a screen session on EC2)
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"

CANON_DIR="data/processed/canonical_sonics_full"
EMB_CACHE_LP="data/emb_cache_encodec_lp8k"   # separate cache for LP audio
OUT_DIR="data/processed/bw_recheck_lp8k"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/bw_recheck_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO"
mkdir -p "$LOG_DIR" "$EMB_CACHE_LP" "$OUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

log "Bandwidth recheck: EnCodec flow with --lowpass-hz 8000"
log "Output dir: $OUT_DIR"
log "Compare AUC against data/processed/full_sonics_all/window_flow_eval.csv"

# Run on a 5k real / 4k-per-generator subset (same as the bitrate sweep)
# to keep runtime tractable (~6-8h on T4).
# Full re-run on all 59k tracks can be done if pilot confirms the hypothesis.
python scripts/run_balanced_ablation.py \
    --embeddings         encodec \
    --per-stratum        4000 \
    --analysis-duration  55 \
    --window-duration    4 \
    --hop-duration       2 \
    --estimators         twonn \
    --canonical-manifest "$CANON_DIR/canonical_manifest.csv" \
    --preprocess-mode    preprocessed \
    --embedding-cache-dir "$EMB_CACHE_LP" \
    --device             cuda \
    --lowpass-hz         8000 \
    --window-flow-eval \
    --wf-window-duration  4.0 \
    --wf-hop-duration     2.0 \
    --wf-pca-components   128 \
    --wf-flow-epochs      200 \
    --seed               42 \
    --output-dir         "$OUT_DIR" \
    2>&1

log "Bandwidth recheck complete"

# Quick comparison report
python3 - <<'PYEOF'
import pandas as pd
import sys
from pathlib import Path

lp_csv = Path("data/processed/bw_recheck_lp8k/window_flow_eval.csv")
full_csv = Path("data/processed/full_sonics_all/window_flow_eval.csv")

if not lp_csv.exists():
    print("LP results not found — check for errors above.")
    sys.exit(1)

from sklearn.metrics import roc_auc_score
import numpy as np

def get_aucs(csv_path, score_col="encodec_wf_mean"):
    df = pd.read_csv(csv_path)
    real_s = df.loc[df["label"] == "real", score_col].dropna().values
    results = {}
    for alg in df[df["label"] == "fake"]["algorithm"].unique():
        fake_s = df.loc[(df["label"] == "fake") & (df["algorithm"] == alg), score_col].dropna().values
        if len(fake_s) < 5:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        sc = np.r_[-real_s, -fake_s]
        results[alg] = round(float(roc_auc_score(y, sc)), 4)
    return results

print("\n=== AUC comparison: no LP vs LP-8kHz ===")
print(f"{'Generator':<30} {'No LP (64kbps)':>16} {'LP-8kHz':>10}")
print("-" * 60)

lp_aucs = get_aucs(lp_csv)
full_aucs = get_aucs(full_csv) if full_csv.exists() else {}

for alg in sorted(lp_aucs.keys()):
    full_v = full_aucs.get(alg, float("nan"))
    lp_v = lp_aucs.get(alg, float("nan"))
    delta = lp_v - full_v if not (pd.isna(full_v) or pd.isna(lp_v)) else float("nan")
    print(f"{alg:<30} {full_v:>16.4f} {lp_v:>10.4f}  (Δ={delta:+.4f})")

print()
print("Interpretation: if |Δ| < 0.05 for all generators, the bandwidth")
print("confound does NOT drive the EnCodec flow results (signal is genuine).")
PYEOF

# Sync results to S3
aws s3 sync "$OUT_DIR/" "$S3/processed/bw_recheck_lp8k/" --no-progress || \
    echo "S3 sync failed (non-fatal)"

log "Done. Results in $OUT_DIR"
