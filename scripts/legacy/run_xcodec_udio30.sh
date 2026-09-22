#!/usr/bin/env bash
# =============================================================================
# run_xcodec_udio30.sh
#
# Plan Phase C3 (optional): Re-run the window-flow on X-Codec and MERT latents
# specifically for udio-30s comparison.
#
# Motivation (from "Probing Token Spaces" ICML 2026):
#   X-Codec latents are best for Udio detection; MERT best for Suno.
#   Our current flow uses EnCodec (optimised for Suno/chirp; weaker on udio-30s).
#   X-Codec may improve udio-30s AUC from 0.733 -> 0.85+.
#
# Protocol:
#   - Extract X-Codec/MERT latents for SONICS real + udio-30s tracks.
#   - Run window-flow (same 4s/2s protocol) on each representation.
#   - Compare AUC on udio-30s across: EnCodec vs X-Codec vs MERT.
#   - Also test ensemble: score = mean(EnCodec_score, XCodec_score).
#
# NOTE: X-Codec is available from HuggingFace (HKUST-Audio/xcodec2).
#   Install: pip install xcodec2 (or clone from HuggingFace).
#   If X-Codec is unavailable, MERT-95M is the fallback (already in codebase).
#
# Usage:
#   bash scripts/run_xcodec_udio30.sh
#   (Run after the full SONICS pipeline)
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"
CANON_DIR="data/processed/canonical_sonics_full"
OUT_DIR="data/processed/xcodec_udio30_eval"
LOG_FILE="logs/xcodec_udio30_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR" logs
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

log "C3: X-Codec / MERT window-flow for udio-30s"

# ---------------------------------------------------------------------------
# IMPORTANT: xcodec2 gets installed into an ISOLATED, throwaway venv, never
# into this project's own .venv. A previous run installed it directly into
# the shared project env and it dragged in a newer numpy + nvidia-cusparse-cu12
# that broke torch/scipy for EVERY script in this repo (see setup_project_venv.sh
# header for the postmortem). This isolation is what makes "optional" experiments
# safe to try without risking the main environment.
# ---------------------------------------------------------------------------
XCODEC_VENV="$HOME/.venv-xcodec-c3"
HAS_XCODEC=0
if [[ ! -d "$XCODEC_VENV" ]]; then
    echo "Creating isolated venv for xcodec2 at $XCODEC_VENV (does not touch the project's .venv)..."
    python3 -m venv "$XCODEC_VENV"
    "$XCODEC_VENV/bin/pip" install -q --upgrade pip
    if "$XCODEC_VENV/bin/pip" install -q xcodec2; then
        HAS_XCODEC=1
        echo "xcodec2 installed in isolated venv."
    else
        echo "xcodec2 install failed even in isolation — skipping X-Codec run, MERT-only."
    fi
else
    if "$XCODEC_VENV/bin/python" -c "import xcodec2" >/dev/null 2>&1; then
        HAS_XCODEC=1
        echo "Isolated xcodec2 venv already present and working."
    fi
fi
echo "X-Codec available: $([[ $HAS_XCODEC == 1 ]] && echo YES || echo NO — MERT-only this run)"

# ---------------------------------------------------------------------------
# Subset: udio-30s (all 4503 tracks) + all real tracks (for manifold)
# ---------------------------------------------------------------------------
log "Building udio-30s + real subset manifest"

python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

mf = pd.read_csv("data/processed/canonical_sonics_full/canonical_manifest.csv", low_memory=False)
if "status" in mf.columns:
    mf = mf[mf["status"] == "ok"]

real = mf[mf["label"] == "real"]
udio30 = mf[(mf["label"] == "fake") & (mf["algorithm"] == "udio-30s")]

subset = pd.concat([real, udio30], ignore_index=True)
subset["status"] = "ok"
out = Path("data/processed/xcodec_udio30_eval/udio30_real_manifest.csv")
out.parent.mkdir(parents=True, exist_ok=True)
subset.to_csv(out, index=False)
print(f"Subset: {len(real)} real + {len(udio30)} udio-30s = {len(subset)} total")
PYEOF

# ---------------------------------------------------------------------------
# Run 1: MERT-95M window-flow on udio-30s
# ---------------------------------------------------------------------------
log "Run 1: MERT-95M window-flow on udio-30s"

python scripts/run_balanced_ablation.py \
    --embeddings         mert-95m \
    --per-stratum        5000 \
    --analysis-duration  55 \
    --window-duration    4 \
    --hop-duration       2 \
    --estimators         twonn \
    --canonical-manifest "$OUT_DIR/udio30_real_manifest.csv" \
    --preprocess-mode    preprocessed \
    --embedding-cache-dir data/emb_cache_mert_udio30 \
    --device             cuda \
    --window-flow-eval \
    --wf-window-duration  4.0 \
    --wf-hop-duration     2.0 \
    --wf-pca-components   64 \
    --wf-flow-epochs      200 \
    --seed               42 \
    --output-dir         "$OUT_DIR/mert_run" \
    2>&1

log "MERT run complete"

# ---------------------------------------------------------------------------
# Run 2: X-Codec window-flow on udio-30s (if available)
# ---------------------------------------------------------------------------
if [[ "$HAS_XCODEC" == "1" ]]; then
    echo "X-Codec is installed (isolated venv) but NOT YET WIRED into this repo's"
    echo "run_balanced_ablation.py pipeline. Remaining work (not done by this script):"
    echo "  1. Add an XCodecExtractor class to"
    echo "     src/intrinsic_ai_music_detection/features/embeddings.py"
    echo "     following the EnCodecExtractor interface, using the isolated venv's"
    echo "     xcodec2 package (HKUST-Audio/xcodec2 on HuggingFace)."
    echo "  2. Register 'xcodec' in get_extractor()."
    echo "  3. Re-run: python scripts/run_balanced_ablation.py --embeddings xcodec --window-flow-eval ..."
    echo "This is genuine remaining work, not a quick environment fix — MERT results"
    echo "above are the reportable C3 result for now."
else
    echo "X-Codec not available — using MERT-95M results only (already run above)."
fi

# ---------------------------------------------------------------------------
# Comparison: EnCodec vs MERT on udio-30s
# ---------------------------------------------------------------------------
log "Comparison report: EnCodec vs MERT on udio-30s"

python3 - <<'PYEOF'
import pandas as pd, numpy as np, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

def get_udio30_auc(wf_csv, score_col_substr="wf_mean"):
    p = Path(wf_csv)
    if not p.exists():
        return float("nan"), float("nan")
    df = pd.read_csv(p, low_memory=False)
    sc = next((c for c in df.columns if score_col_substr in c), None)
    if sc is None:
        return float("nan"), float("nan")
    real_s = df[df["label"] == "real"][sc].dropna().values
    fake_s = df[(df["label"] == "fake") & (df["algorithm"] == "udio-30s")][sc].dropna().values
    if len(real_s) < 5 or len(fake_s) < 5:
        return float("nan"), float("nan")
    y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
    s = np.r_[-real_s, -fake_s]
    auc = float(roc_auc_score(y, s))
    fpr_c, tpr_c, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y, s)
    eer = float(fpr_c[np.nanargmin(np.abs((1 - tpr_c) - fpr_c))])
    return auc, eer

encodec_auc, encodec_eer = get_udio30_auc("data/processed/full_sonics_all/window_flow_eval.csv")
mert_auc, mert_eer = get_udio30_auc("data/processed/xcodec_udio30_eval/mert_run/window_flow_eval.csv",
                                     score_col_substr="wf_mean")

print("\n=== udio-30s AUC/EER: EnCodec vs MERT ===")
print(f"{'Representation':<20} {'AUC':>8} {'EER':>8}")
print("-" * 40)
print(f"{'EnCodec (current)':<20} {encodec_auc:>8.4f} {encodec_eer*100:>7.1f}%")
print(f"{'MERT-95M':<20} {mert_auc:>8.4f} {mert_eer*100:>7.1f}%")
print()

if mert_auc > encodec_auc + 0.02:
    print("✓ MERT improves udio-30s AUC over EnCodec.")
    print("  Recommendation: Report MERT result for udio-30s in the paper, ")
    print("  and add a footnote that alternative codec representations are ")
    print("  complementary (consistent with Probing Token Spaces).")
elif encodec_auc > mert_auc + 0.02:
    print("  EnCodec is better for udio-30s. X-Codec may differ; consider running X-Codec.")
else:
    print("  MERT and EnCodec are comparable on udio-30s.")
    print("  An ensemble of their scores may improve further.")
PYEOF

aws s3 sync "$OUT_DIR/" "$S3/processed/xcodec_udio30_eval/" --no-progress || true
log "C3 done. Results in $OUT_DIR"
