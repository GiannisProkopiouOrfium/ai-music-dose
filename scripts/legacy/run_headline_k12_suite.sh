#!/usr/bin/env bash
# =============================================================================
# run_headline_k12_suite.sh
#
# Adopts K=12 coupling layers (n_coupling_layers) as the new headline flow
# config, based on the flow-depth ablation (scripts/run_flow_depth_ablation.sh,
# MusicDET Sec 4.3 "Effect of Flow Steps K" analog): K=12 beat our K=8 default
# on the full 59,280-track SONICS eval (AUC 0.9246 vs 0.9021, EER 14.38% vs
# 17.03%), non-monotonically (K=16 was worse) — a genuine depth sweet spot,
# not "more is always better".
#
# WHY A FULL RETRAIN (not just reusing k_12/window_flow_eval.csv from the
# ablation): that run did NOT pass --wf-save-flow-path, so no checkpoint was
# persisted. A3 (reconstruction control) and the robustness battery both
# REQUIRE a saved flow checkpoint (--flow-path). This script retrains the
# identical K=12 config with --wf-save-flow-path set, reusing the existing
# EnCodec embedding cache (data/emb_cache_encodec_full) so it's flow-training
# time only, not re-extraction.
#
# Order:
#   1. Retrain + save the K=12 headline flow (GPU, ~3-4h — reuses cached embeddings)
#   2. Recompute confound analysis against the new window_flow_eval.csv (fast, CPU)
#   3. Recompute the MusicDET comparison table against the new window_flow_eval.csv
#      (fast — reuses already-trained/tested MusicDET result files, no MusicDET retrain)
#   4. Re-run A3 reconstruction control with the K=12 checkpoint (GPU, ~1-3h)
#   5. Re-run robustness battery with the K=12 checkpoint (GPU+CPU, ~4-8h — NOTE:
#      run_robustness_battery.py applies each manipulation + EnCodec-extracts +
#      scores in memory per track, with no on-disk cache of manipulated audio,
#      so this costs the same as the original run, not a fast re-score)
#   6. Re-run efficiency measurement with the K=12 checkpoint (~5 min; Param
#      count changes slightly since K=12 has more coupling layers than K=8,
#      still tiny relative to MusicDET's 8.13M)
#
# All steps are resume-safe / skip-if-done except step 1 (flow training is not
# checkpointed epoch-to-epoch; if interrupted, just re-run step 1 alone).
#
# Usage:
#   bash scripts/run_headline_k12_suite.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
EMB_CACHE="data/emb_cache_encodec_full"
OUT_DIR="data/processed/full_sonics_all_k12"
FLOW_PATH="$OUT_DIR/sonics_real_flow_k12.pt"   # run_balanced_ablation.py appends _encodec
FLOW_PATH_FINAL="$OUT_DIR/sonics_real_flow_k12_encodec.pt"
LOG_DIR="logs"

GENRE_CSV="data/processed/genre_tags_all/all_genres.csv"
COVARIATE_CSV="data/processed/covariate_profile/covariate_profiles.csv"
CONFOUND_OUT="reports/confound_analysis_k12"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CONFOUND_OUT"

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

# ---------------------------------------------------------------------------
# Step 1: Retrain + save the K=12 headline flow (reuses cached embeddings)
# ---------------------------------------------------------------------------
log "Step 1 — Retrain headline flow with K=12 coupling layers (reuses cached embeddings)"
if [[ -f "$OUT_DIR/window_flow_eval.csv" ]] && [[ -f "$FLOW_PATH_FINAL" ]]; then
    echo "Already done: $OUT_DIR/window_flow_eval.csv + $FLOW_PATH_FINAL — skipping."
else
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
        --wf-n-coupling-layers   12 \
        --wf-save-flow-path      "$FLOW_PATH" \
        --seed                   42 \
        --output-dir             "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/headline_k12_train.log"
fi

if [[ ! -f "$FLOW_PATH_FINAL" ]]; then
    echo "ERROR: expected flow checkpoint not found at $FLOW_PATH_FINAL" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Recompute confound analysis against the K=12 window_flow_eval.csv
# ---------------------------------------------------------------------------
log "Step 2 — Confound analysis (K=12 flow)"
python scripts/analyze_confounds.py \
    --window-flow-csv "$OUT_DIR/window_flow_eval.csv" \
    --genre-csv "$GENRE_CSV" \
    --covariate-csv "$COVARIATE_CSV" \
    --output-dir "$CONFOUND_OUT" \
    --min-per-genre 20 \
    --fpr-percentile 99.0 \
    2>&1 | tee "$LOG_DIR/headline_k12_confounds.log"

# ---------------------------------------------------------------------------
# Step 3: Recompute the MusicDET comparison table against the K=12 eval
# (reuses already-computed MusicDET test.py result files — no MusicDET retrain)
# ---------------------------------------------------------------------------
log "Step 3 — MusicDET comparison table (K=12 flow vs. existing MusicDET results)"
python scripts/recompute_musicdet_comparison_report.py \
    --window-flow-csv "$OUT_DIR/window_flow_eval.csv" \
    --output-dir data/processed/musicdet_comparison_k12 \
    --musicdet-result-dir ~/MusicDET/output_sonics_matched/result \
    2>&1 | tee "$LOG_DIR/headline_k12_musicdet_compare.log"

# ---------------------------------------------------------------------------
# Step 4: A3 reconstruction control with the K=12 checkpoint
# ---------------------------------------------------------------------------
log "Step 4 — A3 reconstruction control (K=12 flow)"
python scripts/build_reconstruction_control.py \
    --real-dir data/processed/canonical_sonics_full/real \
    --flow-path "$FLOW_PATH_FINAL" \
    --output-dir data/processed/reconstruction_control_k12 \
    --n-tracks 500 --device cuda \
    2>&1 | tee "$LOG_DIR/headline_k12_a3.log"

# ---------------------------------------------------------------------------
# Step 5: Robustness battery with the K=12 checkpoint
# ---------------------------------------------------------------------------
log "Step 5 — Robustness battery (K=12 flow)"
python scripts/run_robustness_battery.py \
    --flow-path "$FLOW_PATH_FINAL" \
    --subset-manifest data/processed/bitrate_sweep_subset.csv \
    --output-dir data/processed/robustness_battery_k12 \
    --device cuda \
    2>&1 | tee "$LOG_DIR/headline_k12_robustness.log"

# ---------------------------------------------------------------------------
# Step 6: Efficiency measurement with the K=12 checkpoint
# ---------------------------------------------------------------------------
log "Step 6 — Efficiency measurement (K=12 flow)"
python scripts/measure_efficiency.py \
    --flow-path "$FLOW_PATH_FINAL" \
    --audio-dir data/processed/canonical_sonics_full/real \
    --n-tracks 60 --device cuda \
    --output-json reports/efficiency_comparison_k12.json \
    2>&1 | tee "$LOG_DIR/headline_k12_efficiency.log"

log "ALL DONE — K=12 headline suite complete"
echo "Key outputs:"
echo "  $OUT_DIR/window_flow_eval.csv                 — new headline (K=12)"
echo "  $FLOW_PATH_FINAL                                — reusable K=12 flow checkpoint"
echo "  $CONFOUND_OUT/summary_verdict.json             — confound re-check under K=12"
echo "  data/processed/musicdet_comparison_k12/wf_vs_musicdet_auc_ci.csv"
echo "  data/processed/reconstruction_control_k12/     — A3 under K=12"
echo "  data/processed/robustness_battery_k12/         — Table-6-parity under K=12"
echo "  reports/efficiency_comparison_k12.json         — Table-3-parity under K=12"
