#!/usr/bin/env bash
# =============================================================================
# run_flow_depth_ablation.sh
#
# Our analog of MusicDET's (arXiv 2605.18072, Sec 4.3) "Effect of Flow Steps K
# and Number of Frequency Bands" ablation. Their K = number of Glow-style flow
# steps per frequency band (default K=2, 2 bands). We don't have a frequency-
# band axis (we operate on EnCodec's single 128-dim continuous latent, not a
# banded STFT spectrogram) — that's a genuine architectural difference worth
# stating plainly in the paper, not something to force-fit. What DOES map
# directly is flow depth: our n_coupling_layers (RealNVP coupling steps) is
# the same knob as their K. This sweeps it to show whether our headline
# default (8) is past the point of diminishing returns, undertrained, or about
# right — the same accuracy-vs-compute trade-off story as their Fig 5a.
#
# Usage:
#   bash scripts/run_flow_depth_ablation.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
EMB_CACHE="data/emb_cache_encodec_full"
OUT_ROOT="data/processed/flow_depth_ablation"
LOG_DIR="logs"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

K_VALUES=(2 4 8 12 16)

for K in "${K_VALUES[@]}"; do
    OUT_DIR="$OUT_ROOT/k_${K}"
    if [[ -f "$OUT_DIR/window_flow_eval.csv" ]]; then
        echo "[K=$K] already done — skipping."
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "============================================================"
    echo "  Flow-depth ablation: K (n_coupling_layers)=$K"
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
        --wf-window-duration     4.0 \
        --wf-hop-duration        2.0 \
        --wf-pca-components      128 \
        --wf-flow-epochs         200 \
        --wf-n-coupling-layers   "$K" \
        --seed                   42 \
        --output-dir             "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/flow_depth_ablation_k${K}.log"
done

echo
echo "============================================================"
echo "  Recomputing AUC/EER per K value"
echo "============================================================"
python3 - "$OUT_ROOT" "${K_VALUES[@]}" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

root = sys.argv[1]
k_values = sys.argv[2:]

rows = []
for k in k_values:
    path = f"{root}/k_{k}/window_flow_eval.csv"
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        print(f"[K={k}] MISSING {path}")
        continue
    col = next((c for c in df.columns if c.endswith("wf_mean")), None)
    if col is None:
        continue
    sub = df.dropna(subset=[col])
    y = (sub["label"] != "real").astype(int).to_numpy()
    s = -sub[col].to_numpy()
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    rows.append({"K_coupling_layers": k, "auc": round(auc, 4), "eer_pct": round(eer * 100, 2), "n": len(sub)})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(f"{root}/flow_depth_ablation_summary.csv", index=False)
print(f"\nSaved -> {root}/flow_depth_ablation_summary.csv")
PYEOF
