#!/usr/bin/env bash
# =============================================================================
# rescore_fakemusiccaps_crossdataset.sh
#
# Re-scores the F1 cross-dataset zero-shot comparison (MusicCaps-real vs
# FakeMusicCaps-fake, scored against a FROZEN SONICS-trained flow — no
# retraining on FakeMusicCaps/MusicCaps data at all) against a given flow
# checkpoint. Use this to refresh F1 under the new K=12 headline flow, since
# the cached data/processed/external_scores/fakemusiccaps{,_real}/ scores were
# produced against the original K=8 flow (data/processed/full_sonics_all/
# sonics_real_flow_encodec.pt) and have never been re-scored since.
#
# IMPORTANT CONTEXT (2026-08-05): the first correctly-sourced run of this exact
# comparison (K=8 flow) gave pooled AUC=0.5371 / EER=46.76% — near chance, NOT
# the AUC 0.94-0.98 figure quoted earlier in project notes/canvas (that figure
# predates the proper MusicCaps-real download and should be treated as
# retracted/incorrect). Re-scoring under K=12 tells us whether the deeper flow
# closes any of this cross-dataset gap, or whether it's an intrinsic
# out-of-domain-generalization limitation independent of flow depth.
#
# Usage:
#   FLOW_PATH=data/processed/full_sonics_all_k12/sonics_real_flow_k12_encodec.pt \
#   TAG=k12 \
#   bash scripts/rescore_fakemusiccaps_crossdataset.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

FLOW_PATH="${FLOW_PATH:-data/processed/full_sonics_all_k12/sonics_real_flow_k12_encodec.pt}"
TAG="${TAG:-k12}"
RAW_REAL_DIR="data/raw/fakemusiccaps_real_only"
RAW_FAKE_DIR="data/external/fakemusiccaps"
REAL_OUT_DIR="data/processed/external_scores/fakemusiccaps_real_${TAG}"
FAKE_OUT_DIR="data/processed/external_scores/fakemusiccaps_${TAG}"

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -f "$FLOW_PATH" ]]; then
    echo "ERROR: $FLOW_PATH not found. Point FLOW_PATH at an existing flow checkpoint"
    echo "(do NOT let score_external_corpus.py silently retrain a new one here)."
    exit 1
fi
if [[ ! -d "$RAW_REAL_DIR" ]] || [[ "$(find "$RAW_REAL_DIR" -name '*.wav' | wc -l)" -lt 100 ]]; then
    echo "ERROR: $RAW_REAL_DIR missing/too small. Run: bash scripts/build_fakemusiccaps_native_comparison.sh (Step 1)."
    exit 1
fi
if [[ ! -d "$RAW_FAKE_DIR" ]]; then
    echo "ERROR: $RAW_FAKE_DIR missing. Run: bash scripts/download_external_data.sh --fakemusiccaps"
    exit 1
fi

log "Step 1 — Score MusicCaps-real against $FLOW_PATH (tag=$TAG)"
python scripts/score_external_corpus.py \
    --audio-dir "$RAW_REAL_DIR" \
    --dataset-label real \
    --flow-path "$FLOW_PATH" \
    --sonics-cache data/emb_cache_encodec_full \
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
    --output-dir "$REAL_OUT_DIR" \
    --device cuda

log "Step 2 — Score FakeMusicCaps-fake (all 5 generators) against $FLOW_PATH (tag=$TAG)"
python scripts/score_external_corpus.py \
    --audio-dir "$RAW_FAKE_DIR" \
    --dataset-label fake \
    --algorithm-from-dirname \
    --flow-path "$FLOW_PATH" \
    --sonics-cache data/emb_cache_encodec_full \
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
    --output-dir "$FAKE_OUT_DIR" \
    --device cuda

log "Step 3 — Native-domain AUC/EER (MusicCaps-real vs FakeMusicCaps-fake), tag=$TAG"
python3 - <<PYEOF
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer

real_df = pd.read_csv("$REAL_OUT_DIR/external_scores.csv")
fake_df = pd.read_csv("$FAKE_OUT_DIR/external_scores.csv")

real_scores = real_df["wf_anomaly_score"].dropna().to_numpy()
print(f"Real (MusicCaps) n={len(real_scores)}  [flow=$FLOW_PATH]")

rows = []
for alg in sorted(fake_df["algorithm"].dropna().unique()):
    fake_scores = fake_df.loc[fake_df["algorithm"] == alg, "wf_anomaly_score"].dropna().to_numpy()
    if len(fake_scores) < 5:
        continue
    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    s = np.r_[real_scores, fake_scores]
    auc_res = bootstrap_auc(y, s)
    eer_res = bootstrap_eer(y, s)
    rows.append({
        "algorithm": alg, "n_real": len(real_scores), "n_fake": len(fake_scores),
        "auc": round(auc_res["auc"], 4),
        "auc_ci": f"[{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]",
        "eer_pct": round(eer_res["eer"] * 100, 2),
    })
    print(f"  {alg:25s}  AUC={auc_res['auc']:.4f} [{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]  "
          f"EER={eer_res['eer']*100:.2f}%  n_fake={len(fake_scores)}")

all_fake = fake_df["wf_anomaly_score"].dropna().to_numpy()
y = np.r_[np.zeros(len(real_scores)), np.ones(len(all_fake))]
s = np.r_[real_scores, all_fake]
auc_res = bootstrap_auc(y, s)
eer_res = bootstrap_eer(y, s)
print(f"\n  {'POOLED (all generators)':25s}  AUC={auc_res['auc']:.4f} [{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]  "
      f"EER={eer_res['eer']*100:.2f}%  n_fake={len(all_fake)}")
rows.append({
    "algorithm": "POOLED", "n_real": len(real_scores), "n_fake": len(all_fake),
    "auc": round(auc_res["auc"], 4),
    "auc_ci": f"[{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]",
    "eer_pct": round(eer_res["eer"] * 100, 2),
})

out = Path("data/processed/external_scores/fakemusiccaps_native_auc_$TAG.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved -> {out}")
PYEOF

log "Done."
