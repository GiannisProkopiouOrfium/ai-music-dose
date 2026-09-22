#!/usr/bin/env bash
# =============================================================================
# run_next_steps.sh  —  Sequential experiment pipeline for intrinsic-ai-music-detection
#
# Runs all planned phases in order:
#   Step 0 : Aggregate existing results
#   Step 1 : pilot_fixed30_temporal  (udio-30s duration-confound validation)
#   Step 2 : pilot_muq               (EnCodec + MuQ, music-specific SSL)
#   Step 3 : region sweep            (intro / middle / outro / random, fixed-30s)
#   Step 4 : Final aggregation + S3 sync
#
# Usage (from repo root, leave running overnight):
#   chmod +x scripts/run_next_steps.sh
#   nohup bash scripts/run_next_steps.sh 2>&1 | tee logs/next_steps_$(date +%Y%m%d_%H%M).log &
#   echo $! > logs/next_steps.pid          # save PID if you need to kill it
# =============================================================================
set -euo pipefail

SCRIPT_START=$(date +%s)
echo "=== run_next_steps.sh started at $(date) ==="

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BASE="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"
NVME="/opt/dlami/nvme"
WORK_DIR="$(pwd)"
CANONICAL="data/processed/canonical_sampler/canonical_manifest.csv"

# Existing caches (from pilot_fair: encodec+xls-r at 55s, preprocessed)
EMB_CACHE_55S="data/processed/emb_cache_pilot"

# New NVMe caches (large models → NVMe to avoid root-disk pressure)
EMB_CACHE_30S="${NVME}/emb_cache_30s"        # encodec at fixed-duration 30s
EMB_CACHE_MUQ="${NVME}/emb_cache_muq"        # encodec(55s symlink) + muq(55s)
EMB_CACHE_REGION="${NVME}/emb_cache_region"  # encodec at fixed-duration 30s per-region

SEED=42
DEVICE=cuda
mkdir -p logs data/processed/region_sweep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_step() { echo ""; echo "========================================"; echo "=== $1"; echo "========================================"; echo "Started: $(date)"; }
elapsed_min() { echo $(( ($(date +%s) - SCRIPT_START) / 60 )); }

s3_sync() {
    # Sync a result directory to S3, excluding large binary files
    local local_dir="$1"
    local s3_path="$2"
    aws s3 sync "${local_dir}/" "${S3_BASE}/${s3_path}/" \
        --exclude "*.npy" \
        --quiet \
    && echo "[S3] synced ${local_dir} → ${s3_path}" \
    || echo "[S3-WARN] sync failed for ${local_dir} (continuing)"
}

check_disk() {
    # Usage: check_disk /path minimum_gb label
    local path="$1" min_gb="$2" label="$3"
    local free_gb
    free_gb=$(df --output=avail -k "${path}" 2>/dev/null | tail -1)
    free_gb=$(( free_gb / 1048576 ))
    echo "Disk check [${label}]: ${free_gb} GB free (need ${min_gb} GB)"
    if [[ ${free_gb} -lt ${min_gb} ]]; then
        echo "ERROR: insufficient disk space on ${path} — need ${min_gb} GB, have ${free_gb} GB"
        return 1
    fi
    return 0
}

run_aggregate() {
    python scripts/aggregate_results.py \
        --dirs \
            "data/processed/pilot_fair" \
            "data/processed/pilot_fixed30" \
            "data/processed/pilot_fixed30_temporal" \
            "data/processed/pilot_muq" \
            "data/processed/region_sweep/intro" \
            "data/processed/region_sweep/middle" \
            "data/processed/region_sweep/outro" \
            "data/processed/region_sweep/random" \
            "data/processed/sonics_balanced_ablation_mert_temporal_all" \
            "data/processed/mert_subsample_lp8k_d55" \
        --output data/processed/results_summary.csv \
    2>/dev/null || true
    echo "[Aggregated] results_summary.csv updated"
    aws s3 cp data/processed/results_summary.csv \
        "${S3_BASE}/results_summary.csv" --quiet \
    && echo "[S3] results_summary.csv uploaded" || true
}

# ---------------------------------------------------------------------------
# Disk space pre-flight
# ---------------------------------------------------------------------------
log_step "Pre-flight disk checks"
echo "Root disk:"
df -h . | grep -v Filesystem
echo "NVMe disk:"
df -h "${NVME}" | grep -v Filesystem

# Abort if less than 5 GB on root (for CSV outputs + misc)
check_disk "." 5 "root"

# Check NVMe: need ~55 GB (MuQ 44 GB + fixed30 3 GB + region 6 GB + margin)
NVME_OK=1
check_disk "${NVME}" 55 "nvme" || NVME_OK=0

if [[ ${NVME_OK} -eq 0 ]]; then
    echo ""
    echo "WARNING: NVMe has < 55 GB free."
    echo "  → MuQ (44 GB) will be placed on root disk instead."
    echo "  → This may fail if root disk fills up. Monitor with: watch df -h"
    EMB_CACHE_MUQ="${WORK_DIR}/data/processed/emb_cache_muq"
    EMB_CACHE_30S="${WORK_DIR}/data/processed/emb_cache_30s"
    EMB_CACHE_REGION="${WORK_DIR}/data/processed/emb_cache_region"
fi

mkdir -p "${EMB_CACHE_30S}" "${EMB_CACHE_MUQ}" "${EMB_CACHE_REGION}"

# Symlink the existing 55s encodec cache into the MuQ cache dir so
# encodec embeddings are reused (saves ~5 GB re-extraction and ~70 min).
if [[ ! -e "${EMB_CACHE_MUQ}/encodec" ]]; then
    ln -s "$(realpath "${EMB_CACHE_55S}/encodec")" "${EMB_CACHE_MUQ}/encodec" \
    && echo "[Cache] Symlinked encodec 55s cache → ${EMB_CACHE_MUQ}/encodec" \
    || echo "[Cache-WARN] symlink failed; encodec will be re-extracted"
fi

# ---------------------------------------------------------------------------
# Step 0: Aggregate existing results before doing anything new
# ---------------------------------------------------------------------------
log_step "Step 0: Aggregate existing results"
run_aggregate

# ---------------------------------------------------------------------------
# Step 1: pilot_fixed30_temporal
#   Adds --with-temporal and --window-flow-eval to the fixed-duration 30s run.
#   Reuses the encodec 30s cache already built by pilot_fixed30 (if present).
#   Establishes a CLEAN udio-30s number without the duration confound.
# ---------------------------------------------------------------------------
log_step "Step 1: pilot_fixed30_temporal (udio-30s confound validation)"

python scripts/run_balanced_ablation.py \
    --embeddings encodec \
    --per-stratum 100 \
    --fixed-duration 30 \
    --with-temporal \
    --window-duration 4 --hop-duration 2 \
    --estimators twonn \
    --canonical-manifest "${CANONICAL}" \
    --preprocess-mode preprocessed \
    --embedding-cache-dir "${EMB_CACHE_30S}" \
    --device ${DEVICE} \
    --real-anomaly-eval --one-class-method mahalanobis \
    --window-flow-eval --wf-window-duration 4.0 --wf-hop-duration 2.0 \
    --wf-pca-components 64 --wf-flow-epochs 200 \
    --seed ${SEED} \
    --output-dir data/processed/pilot_fixed30_temporal \
    2>&1 | tee logs/pilot_fixed30_temporal.log

s3_sync data/processed/pilot_fixed30_temporal pilot_fixed30_temporal
aws s3 cp logs/pilot_fixed30_temporal.log "${S3_BASE}/pilot_fixed30_temporal.log" --quiet || true
run_aggregate
echo "Step 1 done at $(date) — elapsed $(elapsed_min) min"

# ---------------------------------------------------------------------------
# Step 2: pilot_muq (EnCodec + MuQ — music-specific SSL)
#   MuQ (OpenMuQ/MuQ-large-msd-iter, ICASSP 2025) is the top-priority
#   bibliography-aligned music SSL model. EnCodec is reused from cache.
#   Expected runtime: ~3-4h (MuQ model download + extraction for 2600 tracks).
#   Expected disk usage: ~44 GB (MuQ cache on NVMe).
# ---------------------------------------------------------------------------
log_step "Step 2: pilot_muq (EnCodec + MuQ, music-specific SSL, 55s)"

# Disk check before MuQ (it writes the largest cache)
check_disk "${NVME}" 44 "nvme-before-muq" || {
    echo "WARNING: Not enough NVMe space for MuQ. Trying root disk."
    EMB_CACHE_MUQ="${WORK_DIR}/data/processed/emb_cache_muq"
    mkdir -p "${EMB_CACHE_MUQ}"
    if [[ ! -e "${EMB_CACHE_MUQ}/encodec" ]]; then
        ln -s "$(realpath "${EMB_CACHE_55S}/encodec")" "${EMB_CACHE_MUQ}/encodec" || true
    fi
}

python scripts/run_balanced_ablation.py \
    --embeddings encodec muq \
    --per-stratum 100 \
    --with-temporal \
    --window-duration 4 --hop-duration 2 \
    --analysis-duration 55 \
    --estimators twonn \
    --canonical-manifest "${CANONICAL}" \
    --preprocess-mode preprocessed \
    --embedding-cache-dir "${EMB_CACHE_MUQ}" \
    --device ${DEVICE} \
    --real-anomaly-eval --one-class-method mahalanobis \
    --window-flow-eval --wf-window-duration 4.0 --wf-hop-duration 2.0 \
    --wf-pca-components 64 --wf-flow-epochs 200 \
    --seed ${SEED} \
    --output-dir data/processed/pilot_muq \
    2>&1 | tee logs/pilot_muq.log

s3_sync data/processed/pilot_muq pilot_muq
aws s3 cp logs/pilot_muq.log "${S3_BASE}/pilot_muq.log" --quiet || true
# Save MuQ embedding cache to S3 for future recovery
echo "[S3] Syncing MuQ cache to S3 (this may take a while)..."
aws s3 sync "${EMB_CACHE_MUQ}/muq/" "${S3_BASE}/emb_cache_muq/muq/" --quiet \
    && echo "[S3] MuQ cache synced" || echo "[S3-WARN] MuQ cache sync failed"
run_aggregate
echo "Step 2 done at $(date) — elapsed $(elapsed_min) min"

# ---------------------------------------------------------------------------
# Step 3: Region sweep (intro / middle / outro / random)
#   Tests which temporal region of a track is most discriminative.
#   Uses encodec only, fixed-duration 30s, per-stratum 50 (fast ablation).
#   Cache is keyed per-region, built incrementally on NVMe.
#   Hypothesis: outro should be most discriminative ("AI forgets late").
# ---------------------------------------------------------------------------
log_step "Step 3: Region sweep (intro / middle / outro / random)"

for REGION in intro middle outro random; do
    echo "--- Region: ${REGION} ---"

    python scripts/run_balanced_ablation.py \
        --embeddings encodec \
        --per-stratum 50 \
        --fixed-duration 30 \
        --analysis-region "${REGION}" \
        --with-temporal \
        --window-duration 4 --hop-duration 2 \
        --estimators twonn \
        --canonical-manifest "${CANONICAL}" \
        --preprocess-mode preprocessed \
        --embedding-cache-dir "${EMB_CACHE_REGION}" \
        --device ${DEVICE} \
        --real-anomaly-eval --one-class-method mahalanobis \
        --seed ${SEED} \
        --output-dir "data/processed/region_sweep/${REGION}" \
        2>&1 | tee "logs/region_sweep_${REGION}.log"

    s3_sync "data/processed/region_sweep/${REGION}" "region_sweep/${REGION}"
    aws s3 cp "logs/region_sweep_${REGION}.log" "${S3_BASE}/region_sweep_${REGION}.log" --quiet || true
    echo "Region ${REGION} done at $(date)"
done

# ---------------------------------------------------------------------------
# Step 4: Final aggregation + S3 sync
# ---------------------------------------------------------------------------
log_step "Step 4: Final aggregation"
run_aggregate

# Print the comparison table for quick review
echo ""
echo "=== FINAL RESULTS SUMMARY ==="
python scripts/aggregate_results.py \
    --dirs \
        "data/processed/pilot_fair" \
        "data/processed/pilot_fixed30_temporal" \
        "data/processed/pilot_muq" \
        "data/processed/region_sweep/intro" \
        "data/processed/region_sweep/middle" \
        "data/processed/region_sweep/outro" \
        "data/processed/region_sweep/random" \
        "data/processed/sonics_balanced_ablation_mert_temporal_all" \
        "data/processed/mert_subsample_lp8k_d55" \
    --output data/processed/results_summary_final.csv

aws s3 cp data/processed/results_summary_final.csv \
    "${S3_BASE}/results_summary_final.csv" --quiet \
    && echo "[S3] results_summary_final.csv uploaded" || true

SCRIPT_END=$(date +%s)
TOTAL_MIN=$(( (SCRIPT_END - SCRIPT_START) / 60 ))
echo ""
echo "=== All steps complete at $(date) ==="
echo "=== Total elapsed: ${TOTAL_MIN} minutes ==="
echo ""
echo "Review results:"
echo "  cat data/processed/results_summary_final.csv"
echo "  aws s3 cp ${S3_BASE}/results_summary_final.csv ."
