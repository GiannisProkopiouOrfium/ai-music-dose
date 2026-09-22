#!/usr/bin/env bash
#
# Holistic, resumable AI-music-detection ablation pipeline (EnCodec + MERT-95M + XLS-R).
#
# Designed to be left running under screen/tmux on the EC2 box:
#     screen -S ablation
#     bash scripts/run_full_pipeline.sh
#     # detach: Ctrl-A then D     reattach: screen -r ablation
#
# Safe to re-run: every phase resumes from its checkpoint / cached embeddings,
# nothing destructive is done, and a failure aborts before later phases run.
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REPO="$HOME/intrinsic-ai-music-detection"
OUT="data/processed/multiembed_descriptors"
NVME_CACHE="/opt/dlami/nvme/emb_cache_multiembed"   # EnCodec + MERT-95M (ephemeral, fast)
XLSR_CACHE="data/processed/emb_cache_xlsr"          # XLS-R on EBS (separate device)
MANIFEST="data/processed/canonical_sampler/canonical_manifest.csv"
S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"
AWS="aws"                                           # creds available directly on EC2
LOG="logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
# set BACKUP_NVME_CACHE=1 to also back up the ephemeral NVMe cache to S3 at the end
# (insurance: NVMe is wiped if the instance is *stopped*/terminated)
BACKUP_NVME_CACHE="${BACKUP_NVME_CACHE:-0}"

# flags shared by every invocation
COMMON=(--estimators twonn --with-temporal
        --window-duration 8 --hop-duration 4
        --analysis-duration 55 --lowpass-hz 8000)
# extra flags only for the GPU extraction phases
EXTRACT=(--max-points 2000 --temporal-max-points 300 --per-stratum 200
         --device cuda --canonical-manifest "$MANIFEST")
EVAL=(--generator-shift-eval --real-anomaly-eval)

cd "$REPO"
mkdir -p logs
# mirror everything to a timestamped logfile
exec > >(tee -a "$LOG") 2>&1

log()  { echo; echo "=== [$(date '+%F %T')] $* ==="; }
warn() { echo "!!! [$(date '+%F %T')] WARN: $*"; }
trap 'echo "!!! FAILED at line $LINENO (exit $?). Nothing destructive done — re-run this script to resume." >&2' ERR

# best-effort S3 sync: never aborts the pipeline (creds can expire on long runs)
sync_results() {
  log "Sync results -> S3 (best-effort)"
  if $AWS s3 sync "$OUT" "$S3/$OUT"; then
    echo "  synced $OUT"
  else
    warn "S3 sync failed (continuing). Re-sync later: $AWS s3 sync $OUT $S3/$OUT"
  fi
}

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
log "Preflight"
grep -q "_atomic_write_json" scripts/run_balanced_ablation.py \
  || { echo "ERROR: atomic-checkpoint fix missing — run 'git pull' first."; exit 1; }
test -f "$MANIFEST"        || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }
test -d /opt/dlami/nvme    || { echo "ERROR: NVMe not mounted at /opt/dlami/nvme"; exit 1; }
df -h / /opt/dlami/nvme

# --------------------------------------------------------------------------- #
# 0. Move existing cache to NVMe (idempotent) + back up source audio once
# --------------------------------------------------------------------------- #
if [ -d "data/processed/emb_cache_multiembed" ] && [ ! -e "$NVME_CACHE" ]; then
  log "Moving embedding cache EBS -> NVMe"
  mv data/processed/emb_cache_multiembed "$NVME_CACHE"
elif [ -e "$NVME_CACHE" ]; then
  log "Cache already on NVMe — skipping move"
else
  log "No existing cache — will extract fresh into NVMe"
  mkdir -p "$NVME_CACHE"
fi

log "Back up canonical source audio -> S3 (one-time safety, best-effort)"
$AWS s3 sync data/processed/canonical_sampler "$S3/data/processed/canonical_sampler" \
  || warn "canonical_sampler sync failed (continuing)"

# --------------------------------------------------------------------------- #
# 1. EnCodec + MERT-95M  (GPU; resumes from checkpoint, cache on NVMe)
# --------------------------------------------------------------------------- #
log "Phase 1: EnCodec + MERT-95M extraction"
python scripts/run_balanced_ablation.py \
  --embeddings encodec mert-95m \
  --output-dir "$OUT" "${COMMON[@]}" "${EXTRACT[@]}" \
  --embedding-cache-dir "$NVME_CACHE" "${EVAL[@]}"
sync_results

# --------------------------------------------------------------------------- #
# 2. Structural descriptors, uniform across both  (CPU; near-instant)
# --------------------------------------------------------------------------- #
log "Phase 2: structural descriptors (descriptors-only)"
python scripts/run_balanced_ablation.py \
  --embeddings encodec mert-95m \
  --output-dir "$OUT" "${COMMON[@]}" \
  --embedding-cache-dir "$NVME_CACHE" \
  --descriptors-only "${EVAL[@]}"
sync_results

# --------------------------------------------------------------------------- #
# 3. XLS-R  (GPU; cache on EBS, same output dir so it joins the fusion)
# --------------------------------------------------------------------------- #
log "Phase 3: XLS-R extraction"
python scripts/run_balanced_ablation.py \
  --embeddings xls-r \
  --output-dir "$OUT" "${COMMON[@]}" "${EXTRACT[@]}" \
  --embedding-cache-dir "$XLSR_CACHE" "${EVAL[@]}"
sync_results

# --------------------------------------------------------------------------- #
# 4. Cross-embedding fusion  (CPU; reads CSVs only)
# --------------------------------------------------------------------------- #
log "Phase 4: cross-embedding fusion (classify-only)"
python scripts/run_balanced_ablation.py \
  --embeddings encodec mert-95m xls-r \
  --output-dir "$OUT" "${COMMON[@]}" \
  --classify-only "${EVAL[@]}"
sync_results

# --------------------------------------------------------------------------- #
# 5. (optional) Back up the ephemeral NVMe cache to S3
#    NVMe is wiped on instance stop/terminate (survives reboot). Enable with:
#        BACKUP_NVME_CACHE=1 bash scripts/run_full_pipeline.sh
#    Restore later with:
#        aws s3 sync $S3/nvme_cache/emb_cache_multiembed /opt/dlami/nvme/emb_cache_multiembed
# --------------------------------------------------------------------------- #
if [ "$BACKUP_NVME_CACHE" = "1" ]; then
  log "Phase 5: backing up NVMe cache -> S3 (best-effort)"
  if $AWS s3 sync "$NVME_CACHE" "$S3/nvme_cache/emb_cache_multiembed"; then
    echo "  backed up $NVME_CACHE"
  else
    warn "NVMe cache backup failed (continuing)."
  fi
else
  log "Phase 5: NVMe cache backup skipped (set BACKUP_NVME_CACHE=1 to enable)"
fi

log "ALL DONE"
echo "Results : $OUT  (classification_ablation.csv, generator_shift_*.csv, real_anomaly_*.csv)"
echo "S3      : $S3/$OUT"
echo "Logfile : $LOG"
