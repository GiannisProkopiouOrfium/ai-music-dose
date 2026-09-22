#!/usr/bin/env bash
# =============================================================================
# run_combined_flow.sh
#
# Train the EnCodec RealNVP flow on the COMBINED real corpus (SONICS real +
# FMA-medium + MTG-Jamendo) and evaluate against the SONICS LOGO splits.
#
# Hypothesis (Plan Phase B2): more/broader real data -> better manifold
# mapping -> lower FMA FPR and improved udio-30s AUC.
#
# Comparison points:
#   SONICS-only flow  (existing): data/processed/full_sonics_all/
#   Combined flow     (new):      data/processed/combined_flow_eval/
#
# Key metrics to compare:
#   1. FMA-small FPR (false-positive rate on out-of-distribution real)
#   2. Per-generator AUC on SONICS LOGO splits
#   3. udio-30s AUC specifically (expected to benefit most from broader real)
#
# Usage:
#   # First run build_combined_real_corpus.sh, then:
#   bash scripts/run_combined_flow.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"
COMBINED_MANIFEST="data/processed/combined_real_corpus/combined_manifest.csv"
SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
EMB_CACHE="data/emb_cache_encodec_full"   # reuse the existing cache — cache keys are
# md5(track_id + target_sr + max_duration + preprocess_mode), not the file path, and both
# runs use target_sr=24000 / --analysis-duration 55 / --preprocess-mode preprocessed. The
# 46,558 SONICS-fake + 12,722 SONICS-real track_ids already exist in this cache, so only the
# ~24,985 new FMA-medium tracks require fresh GPU extraction (saves ~113GB + hours of GPU time
# versus re-extracting all ~84k tracks into a brand-new cache dir).
OUT_DIR="data/processed/combined_flow_eval"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/combined_flow_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$EMB_CACHE" "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -f "$COMBINED_MANIFEST" ]]; then
    echo "ERROR: Combined manifest not found: $COMBINED_MANIFEST"
    echo "Run scripts/build_combined_real_corpus.sh first."
    exit 1
fi

python3 -c "
import pandas as pd
df = pd.read_csv('$COMBINED_MANIFEST')
print(f'Combined manifest: {len(df)} tracks')
print(df['corpus'].value_counts().to_string())
"

# ---------------------------------------------------------------------------
# Step 1: Train the flow on combined real and score SONICS fake tracks
# ---------------------------------------------------------------------------
log "Step 1 — Train combined flow + score SONICS LOGO splits"

# NOTE: The combined manifest contains ONLY real tracks.
# We use the SONICS canonical manifest for fake tracks (for LOGO scoring).
# Strategy: run run_balanced_ablation.py with the combined manifest as
# the real source and the SONICS canonical manifest for fakes.

# The run_balanced_ablation.py script uses --canonical-manifest to load
# both real and fake. We handle the combined case by creating a merged
# manifest that has combined reals + SONICS fakes.

python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

combined_real = pd.read_csv("data/processed/combined_real_corpus/combined_manifest.csv", low_memory=False)
combined_real["label"] = "real"

sonics = pd.read_csv("data/processed/canonical_sonics_full/canonical_manifest.csv", low_memory=False)
if "status" in sonics.columns:
    sonics = sonics[sonics["status"] == "ok"]
sonics_fakes = sonics[sonics["label"] == "fake"].copy()

merged = pd.concat([
    combined_real[["track_id", "canonical_path", "label", "algorithm"] if "algorithm" in combined_real.columns
                  else ["track_id", "canonical_path", "label"]],
    sonics_fakes[["track_id", "canonical_path", "label", "algorithm"]],
], ignore_index=True)

if "algorithm" not in merged.columns:
    merged["algorithm"] = ""

merged["status"] = "ok"
out = Path("data/processed/combined_flow_eval/combined_flow_manifest.csv")
out.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out, index=False)
print(f"Combined flow manifest: {len(merged)} rows")
print(merged["label"].value_counts().to_string())
print("Algorithms:", merged[merged["label"] == "fake"]["algorithm"].value_counts().to_string())
PYEOF

python scripts/run_balanced_ablation.py \
    --embeddings         encodec \
    --per-stratum        50000 \
    --analysis-duration  55 \
    --window-duration    4 \
    --hop-duration       2 \
    --estimators         twonn \
    --canonical-manifest "$OUT_DIR/combined_flow_manifest.csv" \
    --preprocess-mode    preprocessed \
    --embedding-cache-dir "$EMB_CACHE" \
    --device             cuda \
    --generator-shift-eval \
    --window-flow-eval \
    --wf-window-duration  4.0 \
    --wf-hop-duration     2.0 \
    --wf-pca-components   128 \
    --wf-flow-epochs      200 \
    --wf-save-flow-path   "$OUT_DIR/combined_real_flow.pt" \
    --seed               42 \
    --output-dir         "$OUT_DIR" \
    2>&1

log "Combined flow eval complete"

# ---------------------------------------------------------------------------
# Step 2: FMA-small FPR with combined flow
# ---------------------------------------------------------------------------
log "Step 2 — FMA-small FPR with combined flow"

python scripts/score_external_corpus.py \
    --audio-dir         data/external/fma_small \
    --dataset-label     real \
    --flow-path         "$OUT_DIR/combined_real_flow_encodec.pt" \
    --sonics-cache      "$EMB_CACHE" \
    --sonics-manifest   "$SONICS_MANIFEST" \
    --output-dir        "$OUT_DIR/fma_fpr_combined" \
    --device            cuda \
    2>&1

log "FMA FPR with combined flow done"

# ---------------------------------------------------------------------------
# Step 3: Comparison report
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
"""
Compare SONICS-only flow vs combined flow on:
  1. Per-generator AUC/EER
  2. FMA-small FPR
"""
import pandas as pd, numpy as np, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

def get_aucs_from_wf_csv(csv_path, score_col_substr="wf_mean"):
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path, low_memory=False)
    score_col = next((c for c in df.columns if score_col_substr in c), None)
    if score_col is None:
        return {}
    real_s = df[df["label"] == "real"][score_col].dropna().values
    aucs = {}
    for alg in sorted(df[df["label"] == "fake"]["algorithm"].dropna().unique()):
        fake_s = df[(df["label"] == "fake") & (df["algorithm"] == alg)][score_col].dropna().values
        if len(fake_s) < 5:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        s = np.r_[-real_s, -fake_s]
        fpr_c, tpr_c, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y, s)
        eer = float(fpr_c[np.nanargmin(np.abs((1-tpr_c) - fpr_c))])
        aucs[alg] = {"auc": round(float(roc_auc_score(y, s)), 4), "eer_pct": round(eer*100, 2)}
    return aucs

sonics_aucs = get_aucs_from_wf_csv("data/processed/full_sonics_all/window_flow_eval.csv")
combined_aucs = get_aucs_from_wf_csv("data/processed/combined_flow_eval/window_flow_eval.csv")

print("\n=== AUC Comparison: SONICS-only vs Combined Real Corpus ===")
print(f"{'Generator':<30} {'SONICS-only AUC':>16} {'SONICS EER':>10} "
      f"{'Combined AUC':>12} {'Combined EER':>12} {'ΔAUC':>8}")
print("-" * 92)

for alg in sorted(set(list(sonics_aucs.keys()) + list(combined_aucs.keys()))):
    s = sonics_aucs.get(alg, {})
    c = combined_aucs.get(alg, {})
    s_auc = s.get("auc", float("nan"))
    c_auc = c.get("auc", float("nan"))
    delta = c_auc - s_auc if not (pd.isna(c_auc) or pd.isna(s_auc)) else float("nan")
    print(f"  {alg:<28} {s_auc:>16.4f} {s.get('eer_pct',float('nan')):>9.1f}%"
          f" {c_auc:>12.4f} {c.get('eer_pct',float('nan')):>11.1f}% {delta:>+8.4f}")

# FMA FPR comparison.
# IMPORTANT: `wf_anomaly_score` (external_scores.csv) is ALREADY the correctly
# signed anomaly score (= -wf_mean; higher = more anomalous). The threshold
# must be computed in the SAME units from EACH flow's own real training
# distribution (window_flow_eval.csv, `encodec_wf_mean`, negated) — NOT a
# hardcoded raw-loglik guess. This makes the FPR self-calibrated per flow,
# as required by the guardrails ("FPR always reported with the
# threshold-calibration source stated").
def _real_anomaly_threshold(wf_csv, percentile=99.0, score_col_substr="wf_mean"):
    if not Path(wf_csv).exists():
        return float("nan")
    df = pd.read_csv(wf_csv, low_memory=False)
    score_col = next((c for c in df.columns if score_col_substr in c), None)
    if score_col is None:
        return float("nan")
    real_anomaly = -df.loc[df["label"] == "real", score_col].dropna().values
    if len(real_anomaly) < 10:
        return float("nan")
    return float(np.percentile(real_anomaly, percentile))

def get_fma_fpr(scores_csv, threshold, score_col="wf_anomaly_score"):
    if not Path(scores_csv).exists() or pd.isna(threshold):
        return float("nan"), 0
    df = pd.read_csv(scores_csv)
    scores = df[score_col].dropna().values
    fpr = float((scores > threshold).mean())
    return fpr, len(scores)

sonics_threshold = _real_anomaly_threshold("data/processed/full_sonics_all/window_flow_eval.csv")
combined_threshold = _real_anomaly_threshold("data/processed/combined_flow_eval/window_flow_eval.csv")
print(f"\n  SONICS-only 99th-pct anomaly threshold : {sonics_threshold:.4f}")
print(f"  Combined    99th-pct anomaly threshold : {combined_threshold:.4f}")

fma_sonics_fpr, n1 = get_fma_fpr("data/processed/external_scores/fma_fpr/external_scores.csv", sonics_threshold)
fma_combined_fpr, n2 = get_fma_fpr("data/processed/combined_flow_eval/fma_fpr_combined/external_scores.csv", combined_threshold)

print(f"\n=== FMA-small FPR (each flow's own 99th-pct real-anomaly threshold => nominal 1% FPR) ===")
print(f"  SONICS-only flow   : FPR={fma_sonics_fpr:.1%}  n={n1}")
print(f"  Combined flow      : FPR={fma_combined_fpr:.1%}  n={n2}")
delta_fpr = fma_combined_fpr - fma_sonics_fpr
print(f"  ΔFPR               : {delta_fpr:+.1%}")
if fma_combined_fpr < fma_sonics_fpr:
    print("  ✓ Combined flow reduces FPR — broader real manifold generalizes better.")
else:
    print("  ⚠ Combined flow does not reduce FPR significantly.")
PYEOF

# Sync to S3
aws s3 sync "$OUT_DIR/" "$S3/processed/combined_flow_eval/" --no-progress || true

log "Done. Compare SONICS-only vs combined flow results above."
