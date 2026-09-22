#!/usr/bin/env bash
# =============================================================================
# run_full_sonics_pipeline.sh
#
# Full-scale SONICS AI-music detection pipeline — EnCodec RealNVP LOGO.
# Designed to run unattended in a screen session on the EC2 g4dn.xlarge.
#
# PRE-REQUISITE (do ONCE before running this script):
#   1. AWS console → EC2 → Volumes → Modify → set to 800 GB → Modify Volume
#   2. Wait ~2 min until state shows "in-use-optimizing" (usable immediately)
#   3. On the instance, grow the partition and filesystem:
#
#        ROOT_DEV=$(findmnt -n -o SOURCE / | sed 's|/dev/||')
#        DISK=$(echo "$ROOT_DEV" | sed 's/p[0-9]*$//')
#        PART=$(echo "$ROOT_DEV" | grep -oP '[0-9]+$')
#        sudo growpart /dev/"$DISK" "$PART"
#        sudo resize2fs /dev/"$ROOT_DEV"
#        df -h /   # should show ~784 GB total
#
# USAGE:
#   screen -S pipeline
#   bash scripts/run_full_sonics_pipeline.sh
#   # Ctrl-A D to detach   |   screen -r pipeline to reattach
#
# RESUME SAFETY:
#   - Canonical build: skips WAVs that already exist on disk
#   - Experiment:      resumes from checkpoint_encodec.json (skips processed tracks)
#   - S3 sync:         aws s3 sync is idempotent
#   Safe to re-run after any failure.
#
# OUTPUTS (all saved locally AND synced to S3):
#   data/processed/full_sonics_all/window_flow_eval.csv     ← HEADLINE RESULT
#   data/processed/full_sonics_all/classification_phase5.csv
#   data/processed/full_sonics_all/generator_shift_eval.csv
#   data/processed/full_sonics_all/ablation_features.csv
#   data/emb_cache_encodec_full/                            ← cache for future runs
#   logs/
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO="$HOME/intrinsic-ai-music-detection"
S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"

CANON_DIR="data/processed/canonical_sonics_full"
EMB_CACHE="data/emb_cache_encodec_full"
OUT_DIR="data/processed/full_sonics_all"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/full_sonics_pipeline_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO"
mkdir -p "$LOG_DIR" "$CANON_DIR" "$EMB_CACHE" "$OUT_DIR"

# Tee all output to a timestamped log file
exec > >(tee -a "$LOG_FILE") 2>&1

log()  { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }
die()  { echo "!!! FATAL [$(date '+%F %T')]: $*" >&2; exit 1; }
sync_s3() {
    # Best-effort: never aborts the pipeline if S3 is momentarily unavailable
    local src="$1" dst="$2"
    aws s3 sync "$src" "$dst" --no-progress 2>&1 || \
        echo "  WARN: s3 sync $src → $dst failed (non-fatal, will retry at end)"
}

trap 'echo; echo "!!! Pipeline failed at line $LINENO. Re-run to resume from checkpoint." >&2' ERR

log "Pipeline start — log: $LOG_FILE"

# ---------------------------------------------------------------------------
# Disk check — must have been expanded before running
# ---------------------------------------------------------------------------
AVAIL_GB=$(df --output=avail -BG / | tail -1 | tr -d 'G ')
log "Disk: $(df -h / | tail -1)"
if (( AVAIL_GB < 150 )); then
    die "Only ${AVAIL_GB} GB free. Expand EBS volume to 800 GB first (see script header)."
fi

# ---------------------------------------------------------------------------
# Step 1: Pre-sync existing results and cache to S3
# ---------------------------------------------------------------------------
log "Step 1 — Pre-sync existing results to S3"

sync_s3 "data/processed/full_scale_sonics/"  "$S3/processed/full_scale_sonics/"
sync_s3 "data/emb_cache_encodec_200/"        "$S3/emb_cache_encodec_200/"
sync_s3 "$LOG_DIR/"                          "$S3/logs/"

echo "Pre-sync complete."

# ---------------------------------------------------------------------------
# Step 2: Build full canonical corpus
# 59280 tracks (12722 real + 46558 fake) → canonical WAVs
# Parameters: 64 kbps MP3 round-trip → LUFS -23 → 24 kHz → trim ≤120 s → WAV int16
# Duration note: udio-30s (~32 s) → canonical WAVs are ~32 s (no padding).
#               In the experiment, --analysis-duration 55 takes min(track_len, 55s).
#               This is correct: udio-30s just gets fewer windows; no special handling.
# Resume-safe: skips any track whose output WAV already exists.
# Expected time: ~8-10 h on g4dn.xlarge with 6 workers.
# ---------------------------------------------------------------------------
log "Step 2 — Build full canonical corpus → $CANON_DIR"

python scripts/build_canonical_corpus.py \
    --output-dir     "$CANON_DIR" \
    --target-sr      24000 \
    --mp3-bitrate    64 \
    --target-lufs    -23.0 \
    --max-duration   120 \
    --workers        6 \
    --flush-every    200 \
    2>&1

# Verify manifest
log "Verifying canonical manifest"
python3 - <<'PYEOF'
import pandas as pd, sys
df = pd.read_csv("data/processed/canonical_sonics_full/canonical_manifest.csv")
ok = df[df["status"] == "ok"] if "status" in df.columns else df
n_real = (ok["label"] == "real").sum()
n_fake = (ok["label"] == "fake").sum()
print(f"OK tracks  : {len(ok):,}  (real={n_real:,}  fake={n_fake:,})")
alg = ok[ok["label"] == "fake"]["algorithm"].value_counts()
print("By algorithm:")
for g, n in alg.items():
    print(f"  {g:30s} {n:6,}")
if n_real < 12000:
    print("ERROR: fewer than 12000 real tracks", file=sys.stderr); sys.exit(1)
if n_fake < 45000:
    print("ERROR: fewer than 45000 fake tracks", file=sys.stderr); sys.exit(1)
print("Manifest OK")
PYEOF

# Sync manifest (WAVs are NOT synced — they are reproducible from raw MP3s on root EBS)
aws s3 cp "$CANON_DIR/canonical_manifest.csv" \
    "$S3/processed/canonical_sonics_full/canonical_manifest.csv"
echo "Manifest uploaded to S3."

# ---------------------------------------------------------------------------
# Step 3: Full SONICS experiment — EnCodec + RealNVP window-flow LOGO
#
# WHAT THIS RUNS (and why):
#   --embeddings encodec          EnCodec only (best for window-flow; MuQ can be added
#                                 later with --classify-only on the cached features)
#   --per-stratum 50000           Effectively all 46558 fakes (largest stratum <20k)
#                                 + all 12722 reals (3.7:1 imbalance, handled by
#                                 class_weight='balanced' in the classifier)
#   --analysis-duration 55        First 55 s per track (removes duration confound;
#                                 consistent with stratum experiments → fair comparison)
#   --with-temporal               Temporal windowed ID features (computed inline during
#                                 extraction — no extra GPU pass; adds ~2h for supervised
#                                 ablation bundles in the paper)
#   --window-duration 4           4 s temporal ID windows (same as MusicDET clip size)
#   --hop-duration 2              2 s hop
#   --estimators twonn            TwoNN only (PHD is O(n²) and 10× slower; TwoNN gave
#                                 equivalent discrimination in all stratum experiments)
#   --embedding-cache-dir         Cache all EnCodec frame arrays to disk (~125 GB).
#                                 Required for window-flow AND enables future experiments
#                                 (add MuQ, change window sizes) without re-extraction.
#   --generator-shift-eval        Cross-generator zero-shot AUC — core paper claim
#   --window-flow-eval            THE headline: RealNVP trained on real windows,
#                                 5-fold LOGO so every real is scored held-out
#   --wf-pca-components 128       EnCodec is 128-dim → condition (128 > 128) = False
#                                 → PCA is skipped internally; full 128-d used for flow
#   --wf-flow-epochs 200          With patience=20 early stopping; more data means
#                                 faster convergence, not slower — 200 is the ceiling
#   NO --real-anomaly-eval        One-class Mahalanobis proved worse than window-flow
#                                 at every scale; excluded to keep run lean
#   NO --lowpass-hz               Canonical preprocessing already applies the 64 kbps
#                                 MP3 round-trip that removes the bandwidth confound;
#                                 no additional lowpass needed
#
# RESUME: If interrupted, re-run Step 3 alone.  checkpoint_encodec.json records every
# completed track; the loop skips them.  Window-flow re-runs from scratch (it's fast
# relative to extraction) but uses the already-populated embedding cache.
#
# Expected time breakdown (g4dn.xlarge T4 16 GB):
#   EnCodec extraction + caching : ~10 h (59280 tracks, ~0.6 s/track GPU)
#   Temporal ID + descriptors    :  ~2 h (CPU, overlaps with GPU)
#   Window-flow 5-fold LOGO      :  ~8 h (330k real windows × 5 folds × RealNVP)
#   Full-flow + fake scoring     :  ~1 h
#   Classification + shift eval  :  ~1 h
#   Total                        : ~20-22 h
# ---------------------------------------------------------------------------
log "Step 3 — Full SONICS experiment (EnCodec RealNVP window-flow LOGO)"

python scripts/run_balanced_ablation.py \
    --embeddings         encodec \
    --per-stratum        50000 \
    --analysis-duration  55 \
    --with-temporal \
    --window-duration    4 \
    --hop-duration       2 \
    --estimators         twonn \
    --canonical-manifest "$CANON_DIR/canonical_manifest.csv" \
    --preprocess-mode    preprocessed \
    --embedding-cache-dir "$EMB_CACHE" \
    --device             cuda \
    --generator-shift-eval \
    --window-flow-eval \
    --wf-window-duration  4.0 \
    --wf-hop-duration     2.0 \
    --wf-pca-components   128 \
    --wf-flow-epochs      200 \
    --seed               42 \
    --output-dir         "$OUT_DIR" \
    2>&1

log "Experiment complete"

# ---------------------------------------------------------------------------
# Step 4: Post-experiment S3 sync
# Saves everything needed for future local analysis and insurance against
# instance termination. The embedding cache (~125 GB) is the most critical:
# it enables future experiments (add MuQ, test different window sizes, etc.)
# without re-running 10 h of GPU extraction.
# ---------------------------------------------------------------------------
log "Step 4 — Syncing all results and cache to S3"

# Results (small: ~100 MB — sync first for quick access from local machine)
sync_s3 "$OUT_DIR/"   "$S3/processed/full_sonics_all/"

# Logs
sync_s3 "$LOG_DIR/"   "$S3/logs/"

# Embedding cache (~125 GB — this will take ~30-60 min to upload)
log "Syncing embedding cache to S3 (~125 GB, ~30-60 min)..."
sync_s3 "$EMB_CACHE/" "$S3/emb_cache_encodec_full/"

# Manifest backup (safety copy)
aws s3 cp "$CANON_DIR/canonical_manifest.csv" \
    "$S3/processed/canonical_sonics_full/canonical_manifest.csv"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "ALL DONE"
echo "Key output files:"
echo "  $OUT_DIR/window_flow_eval.csv         ← HEADLINE: per-generator AUC/EER"
echo "  $OUT_DIR/classification_phase5.csv    ← supervised fusion (descriptors + flow)"
echo "  $OUT_DIR/generator_shift_eval.csv     ← zero-shot cross-generator AUC"
echo "  $OUT_DIR/ablation_features.csv        ← full per-track feature table"
echo ""
echo "S3 locations:"
echo "  $S3/processed/full_sonics_all/"
echo "  $S3/emb_cache_encodec_full/"
echo "  $S3/processed/canonical_sonics_full/canonical_manifest.csv"
echo ""
echo "Expected headline (based on scaling trend from stratum 200):"
echo "  chirp-v2-xxl-alpha  ~0.997   chirp-v3  ~0.975"
echo "  chirp-v3.5          ~0.960   udio-120s  ~0.950"
echo "  udio-30s            ~0.860   suno (new) ~?"
