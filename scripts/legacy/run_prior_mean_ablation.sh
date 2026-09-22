#!/usr/bin/env bash
# =============================================================================
# run_prior_mean_ablation.sh
#
# Reproduces MusicDET's (arXiv 2605.18072, Sec 4.3) "Effect of the Prior Mean
# mu" ablation on OUR flow. They shift their one-class flow's Gaussian prior
# to N(mu_real, I) (default mu_real=5, vs our default 0 = standard normal) and
# report EER *monotonically decreasing* as mu_real increases, attributed to
# increased latent-space margin between the real mode and everything else.
#
# This sweeps --wf-prior-mean over {0, 1, 2, 3, 5, 8} on the same SONICS-real
# corpus/cache as the headline run and reports AUC/EER per value, so we can
# see whether the same trend holds for our EnCodec-latent flow. If it does,
# mu>0 is a free win we should adopt as the new default; if not, that's a
# genuine, reportable architectural difference vs. spectrogram-band flows.
#
# Prereqs: same canonical manifest + embedding cache as run_full_sonics_pipeline.sh.
#
# Usage:
#   bash scripts/run_prior_mean_ablation.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
EMB_CACHE="data/emb_cache_encodec_full"
OUT_ROOT="data/processed/prior_mean_ablation"
LOG_DIR="logs"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

MU_VALUES=(0 1 2 3 5 8)

for MU in "${MU_VALUES[@]}"; do
    OUT_DIR="$OUT_ROOT/mu_${MU}"
    if [[ -f "$OUT_DIR/window_flow_eval.csv" ]]; then
        echo "[mu=$MU] already done — skipping."
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "============================================================"
    echo "  Prior-mean ablation: mu_real=$MU"
    echo "============================================================"
    python scripts/run_balanced_ablation.py \
        --embeddings          encodec \
        --per-stratum         50000 \
        --analysis-duration   55 \
        --window-duration     4 \
        --hop-duration        2 \
        --estimators          twonn \
        --canonical-manifest  "$SONICS_MANIFEST" \
        --preprocess-mode     preprocessed \
        --embedding-cache-dir "$EMB_CACHE" \
        --device              cuda \
        --window-flow-eval \
        --wf-window-duration  4.0 \
        --wf-hop-duration     2.0 \
        --wf-pca-components   128 \
        --wf-flow-epochs      200 \
        --wf-prior-mean       "$MU" \
        --seed                42 \
        --output-dir          "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/prior_mean_ablation_mu${MU}.log"
done

echo
echo "============================================================"
echo "  Recomputing AUC/EER per mu value"
echo "============================================================"
python3 - "$OUT_ROOT" "${MU_VALUES[@]}" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

root = sys.argv[1]
mu_values = sys.argv[2:]

rows = []
for mu in mu_values:
    path = f"{root}/mu_{mu}/window_flow_eval.csv"
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        print(f"[mu={mu}] MISSING {path}")
        continue
    col = next((c for c in df.columns if c.endswith("wf_mean")), None)
    if col is None:
        print(f"[mu={mu}] no *_wf_mean column found in {path}, skipping")
        continue
    sub = df.dropna(subset=[col])
    y = (sub["label"] != "real").astype(int).to_numpy()
    s = -sub[col].to_numpy()  # wf_mean is log-likelihood (higher = more real-like) -> negate for anomaly-AUC
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    rows.append({"prior_mean": mu, "auc": round(auc, 4), "eer_pct": round(eer * 100, 2), "n": len(sub)})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(f"{root}/prior_mean_ablation_summary.csv", index=False)
print(f"\nSaved -> {root}/prior_mean_ablation_summary.csv")
print("\nCompare trend to MusicDET Fig 5b: EER should MONOTONICALLY DECREASE as prior_mean increases, if the same mechanism transfers to our latent space.")
PYEOF
