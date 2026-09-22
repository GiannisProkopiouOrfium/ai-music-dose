#!/usr/bin/env bash
# MusicCaps-specific coverage curve (handover §5 / detailed report §12.2) +
# dense-window ablation arm (I3).
#
# Question this answers: F2 (fresh K=2 flow on 5,355 MusicCaps-real tracks)
# fails at AUC 0.286 on FakeMusicCaps — is that DATA STARVATION (AUC should
# climb steadily with n) or a DOMAIN-COMPLEXITY MISMATCH (flat/degenerate AUC
# at every n)? A second sweep with dense 1s/0.5s windows tests the
# window-redundancy hypothesis (10s clips yield only ~4 heavily-overlapping
# 4s/2s windows; denser windows recover effective sample count).
#
# Uses the CORRECTED coverage_curve.py protocol: a held-out real set disjoint
# from every training prefix (the old in-sample protocol is not citable), K/PCA
# matching the F2 recipe (K=2, PCA-64), 10s cache keys.
#
# Prerequisites (already on EC2 from the F2 run):
#   data/processed/canonical_fmc_native/combined_manifest.csv
#   data/emb_cache_encodec_fmc_native/encodec/*.npy   (built at --analysis-duration 10)
#   data/processed/fmc_native_flow_k2/window_flow_eval.csv
#
# Usage: bash scripts/run_musiccaps_coverage_curve.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

MANIFEST="${MANIFEST:-data/processed/canonical_fmc_native/combined_manifest.csv}"
EMB_CACHE="${EMB_CACHE:-data/emb_cache_encodec_fmc_native}"
WF_CSV="${WF_CSV:-data/processed/fmc_native_flow_k2/window_flow_eval.csv}"
OUT_ROOT="${OUT_ROOT:-data/processed/musiccaps_coverage_curve}"
K="${K:-2}"
PCA="${PCA:-64}"
SIZES="${SIZES:-500 1000 2000 4284}"
N_HELDOUT="${N_HELDOUT:-1000}"

log() { echo "[$(date '+%F %T')] $*"; }

missing=0
for f in "$MANIFEST" "$WF_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING prerequisite: $f"
        missing=1
    fi
done
if [[ "$missing" -eq 1 ]]; then
    echo
    echo "--- What IS present (to locate the right paths) ---"
    echo "manifest candidates:"
    find data/processed -maxdepth 3 -name '*manifest*.csv' -newermt '2026-08-01' 2>/dev/null \
        | sed 's/^/  /' | head -20 || true
    echo "window_flow_eval candidates:"
    find data/processed -maxdepth 3 -name 'window_flow_eval*.csv' 2>/dev/null \
        | sed 's/^/  /' | head -20 || true
    echo "FakeMusicCaps embedding cache: $(ls -1 "$EMB_CACHE/encodec" 2>/dev/null | wc -l) cached .npy files"
    echo
    echo "If the canonical corpus is gone, rebuild it (it is the expensive part):"
    echo "  bash scripts/run_fakemusiccaps_native_flow.sh"
    echo "If the files exist under different names, re-run this script with overrides, e.g.:"
    echo "  MANIFEST=<path> WF_CSV=<path> bash scripts/run_musiccaps_coverage_curve.sh"
    exit 1
fi

# --- Sweep 1: standard 4s/2s windows (the F2 recipe, at increasing n) ---
log "Sweep 1/2 — standard 4s/2s windows, K=${K}, PCA=${PCA}, n in {${SIZES}}"
python scripts/coverage_curve.py \
    --window-flow-csv "$WF_CSV" \
    --emb-cache       "$EMB_CACHE" \
    --sonics-manifest "$MANIFEST" \
    --output-dir      "$OUT_ROOT/windows_4s2s" \
    --device          cuda \
    --mode            overall \
    --n-real-sizes    $SIZES \
    --n-heldout-real  "$N_HELDOUT" \
    --n-coupling-layers "$K" \
    --pca-components  "$PCA" \
    --max-duration    10.0 \
    --window-duration 4.0 --hop-duration 2.0 \
    --seed 42

# --- Sweep 2: dense 1s/0.5s windows (window-redundancy hypothesis, I3) ---
log "Sweep 2/2 — DENSE 1s/0.5s windows (same corpus, same K/PCA)"
python scripts/coverage_curve.py \
    --window-flow-csv "$WF_CSV" \
    --emb-cache       "$EMB_CACHE" \
    --sonics-manifest "$MANIFEST" \
    --output-dir      "$OUT_ROOT/windows_1s05s" \
    --device          cuda \
    --mode            overall \
    --n-real-sizes    $SIZES \
    --n-heldout-real  "$N_HELDOUT" \
    --n-coupling-layers "$K" \
    --pca-components  "$PCA" \
    --max-duration    10.0 \
    --window-duration 1.0 --hop-duration 0.5 \
    --seed 42

log "Done. Interpret:"
log "  - windows_4s2s/coverage_curve_overall.csv climbing with n  => data starvation (more MusicCaps-like real data would fix F2)"
log "  - flat/degenerate at all n                                 => domain-complexity mismatch (windowing/architecture, not volume)"
log "  - windows_1s05s markedly better at the same n              => window redundancy was the binding constraint (I3 confirmed)"
