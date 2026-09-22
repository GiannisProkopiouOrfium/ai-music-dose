#!/usr/bin/env bash
# =============================================================================
# run_window_count_confound_check.sh
#
# WHY: udio-30s tracks are natively ~30s and therefore yield exactly 15
# EnCodec windows (4s window / 2s hop) under --analysis-duration 55, while
# every other generator (chirp-*, udio-120s) yields ~26 windows (capped by
# the 55s analysis window). udio-30s has the weakest AUC (0.733) of all 5
# generators. Before concluding udio-30s is a "genuinely hard" generator, we
# must rule out that it is simply scored on ~42% less evidence per track,
# which would make its wf_mean/wf_std/wf_slope trajectory summary stats
# noisier and less discriminative — an artifact, not a generator property.
#
# WHAT THIS DOES:
#   Re-runs window-flow-eval with --wf-max-windows 15, which truncates EVERY
#   track's window trajectory (real held-out AND fake) to the first 15
#   windows before computing summary stats. This equalizes the evidence
#   budget across all generators to match udio-30s exactly.
#     - udio-30s AUC should be ~unchanged (it never had >15 windows anyway;
#       this is the built-in sanity check).
#     - If chirp-v3.5 / udio-120s AUC drops sharply toward ~0.73 when capped
#       to 15 windows  → window-count IS a major confound; report a
#       duration-matched AUC table in the paper and discuss it explicitly.
#     - If they stay close to their uncapped AUC (~0.90-0.99) → udio-30s is
#       genuinely harder acoustically, and the window-count difference is a
#       minor/non-confound.
#
# Reuses the existing checkpoint_encodec.json + embedding cache in
# full_sonics_all — this should take minutes, not hours (no new GPU
# extraction, no PHD/TwoNN recompute, only flow retrain + rescoring).
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
CANON_DIR="data/processed/canonical_sonics_full"
EMB_CACHE="data/emb_cache_encodec_full"
OUT_DIR="data/processed/full_sonics_all"

cd "$REPO"

# --- backup the existing (uncapped) headline result before overwriting ---
if [[ -f "$OUT_DIR/window_flow_eval.csv" && ! -f "$OUT_DIR/window_flow_eval_uncapped.csv" ]]; then
    cp "$OUT_DIR/window_flow_eval.csv" "$OUT_DIR/window_flow_eval_uncapped.csv"
    echo "Backed up existing window_flow_eval.csv -> window_flow_eval_uncapped.csv"
else
    echo "Backup already exists (or nothing to back up) — not overwriting backup."
fi

echo "=== Re-running window-flow-eval with --wf-max-windows 15 (evidence-budget cap) ==="

python scripts/run_balanced_ablation.py \
    --embeddings         encodec \
    --per-stratum        50000 \
    --analysis-duration  55 \
    --canonical-manifest "$CANON_DIR/canonical_manifest.csv" \
    --preprocess-mode    preprocessed \
    --embedding-cache-dir "$EMB_CACHE" \
    --device             cuda \
    --window-flow-eval \
    --wf-window-duration  4.0 \
    --wf-hop-duration     2.0 \
    --wf-pca-components   128 \
    --wf-flow-epochs      200 \
    --wf-max-windows      15 \
    --seed               42 \
    --output-dir         "$OUT_DIR" \
    2>&1

# --- rename the freshly-produced (capped) result so it doesn't get clobbered later ---
cp "$OUT_DIR/window_flow_eval.csv" "$OUT_DIR/window_flow_eval_capped15.csv"

echo
echo "=== Comparing per-generator AUC: uncapped (~26 windows) vs capped (15 windows) ==="
python3 - <<'PYEOF'
import pandas as pd
from sklearn.metrics import roc_auc_score

unc = pd.read_csv("data/processed/full_sonics_all/window_flow_eval_uncapped.csv")
cap = pd.read_csv("data/processed/full_sonics_all/window_flow_eval_capped15.csv")

def per_gen_auc(df, score_col="encodec_wf_mean"):
    # score_col (wf_mean) is mean log-likelihood: higher = more real-like.
    # Negate so higher = more anomalous/fake-like, matching the AUC convention
    # used everywhere else (fake label = 1).
    df = df.dropna(subset=[score_col])
    real = df[df["label"] == "real"]
    out = {}
    for gen, sub in df[df["label"] == "fake"].groupby("algorithm"):
        y = [0] * len(real) + [1] * len(sub)
        s = -pd.concat([real[score_col], sub[score_col]]).to_numpy()
        out[gen] = roc_auc_score(y, s)
    return out

a = per_gen_auc(unc)
b = per_gen_auc(cap)
print(f"{'generator':22s} {'uncapped(~26w)':>15s} {'capped(15w)':>13s} {'delta':>8s}")
for gen in sorted(a):
    d = b.get(gen, float("nan")) - a[gen]
    print(f"{gen:22s} {a[gen]:15.4f} {b.get(gen, float('nan')):13.4f} {d:8.4f}")
PYEOF
