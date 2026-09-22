#!/usr/bin/env bash
# =============================================================================
# run_external_corpus_eval.sh
#
# *** METHODOLOGICAL WARNING (found during environment audit, not yet fixed) ***
# Step 5/6 below merges FMA + FakeMusicCaps "real" tracks INTO the combined
# manifest and re-runs run_balanced_ablation.py's K-fold LOGO window-flow on
# top of it. Because the K-fold split is drawn from ALL real tracks in that
# combined manifest, FMA/FMC real tracks end up in the flow's TRAINING fold
# in 4 of 5 folds -- this is NOT a clean FPR/generalization measurement, it's
# a corpus-broadening experiment (similar in spirit to Phase B2), and it also
# uses 2s/1s windows here instead of the canonical 4s/2s protocol used
# everywhere else, so its numbers are not directly comparable to headline
# results either. For an honest, leakage-free FPR test and zero-shot
# generalization test, use score_external_corpus.py instead (it scores
# against a FROZEN flow trained only on SONICS reals, never retrained on
# external data) -- see download_external_data.sh's printed next-steps, or
# scripts/check_external_corpus_status.py for a full audit of what's on disk.
# =============================================================================
#
# Evaluates the SONICS-trained EnCodec RealNVP flow on external corpora:
#   1. FMA-small  (8000 real tracks, 30s, diverse genres) → FPR measurement
#   2. FakeMusicCaps (27605 fake tracks, 10s, 5 unseen generators) → generalization
#
# Both are added to the existing SONICS manifest and run through the window-flow.
# The LOGO evaluation includes FMA/FakeMusicCaps naturally:
#   - FMA real tracks: scored out-of-fold → honest FPR
#   - FakeMusicCaps fakes: scored vs full flow → per-generator AUC
#
# PRE-REQUISITES:
#   - Full SONICS run complete (data/processed/canonical_sonics_full/ + emb cache)
#   - 2s/1s window re-run complete (or run in parallel)
#   - ~60 GB free disk space (FMA-small ~18 GB + FMC ~18 GB + caches ~15 GB)
#   - huggingface_hub installed: pip install huggingface_hub
#
# USAGE:
#   screen -S external
#   bash scripts/run_external_corpus_eval.sh
#   # Ctrl-A D to detach
#
# OUTPUTS:
#   data/processed/external_eval/window_flow_eval.csv  ← per-generator AUC incl. FMC
#   data/processed/external_eval/fpr_fma.csv           ← FPR on FMA real tracks
# =============================================================================
set -euo pipefail

S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"
REPO="$HOME/intrinsic-ai-music-detection"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/external_eval_$(date +%Y%m%d_%H%M%S).log"

# Directories
RAW_FMA="data/raw/fma_small"
RAW_FMC="data/raw/fakemusiccaps"
CANON_FMA="data/processed/canonical_fma"
CANON_FMC="data/processed/canonical_fmc"
COMBINED_MANIFEST="data/processed/combined_external_manifest.csv"
EMB_CACHE="data/emb_cache_encodec_full"   # REUSE existing SONICS cache (new IDs just get added)
OUT_DIR="data/processed/external_eval"
SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"

cd "$REPO"
mkdir -p "$LOG_DIR" "$RAW_FMA" "$RAW_FMC" "$CANON_FMA" "$CANON_FMC" "$OUT_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

log()  { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }
warn() { echo "!!! WARN [$(date '+%F %T')]: $*"; }

log "External corpus eval start — log: $LOG_FILE"

# ---------------------------------------------------------------------------
# Disk check
# ---------------------------------------------------------------------------
AVAIL_GB=$(df --output=avail -BG / | tail -1 | tr -d 'G ')
log "Disk: $(df -h / | tail -1)"
if (( AVAIL_GB < 60 )); then
    echo "!!! Only ${AVAIL_GB} GB free — need ~60 GB. Consider deleting canonical WAVs:"
    echo "    rm -rf data/processed/canonical_sonics_full/real data/processed/canonical_sonics_full/fake"
    echo "    (embedding cache is preserved; WAVs are reproducible from raw SONICS MP3s)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Download FMA-small (8000 real tracks, 30s each, ~7.2 GB MP3)
# ---------------------------------------------------------------------------
log "Step 1 — Download FMA-small"

FMA_ZIP="data/raw/fma_small.zip"
FMA_META_ZIP="data/raw/fma_metadata.zip"

if [ ! -d "$RAW_FMA/000" ]; then
    echo "Downloading FMA metadata (~342 MB)..."
    wget -c "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" \
         -O "$FMA_META_ZIP" --progress=dot:giga

    echo "Extracting FMA metadata..."
    unzip -q "$FMA_META_ZIP" -d data/raw/fma_metadata_raw
    mv data/raw/fma_metadata_raw/fma_metadata/* data/raw/ 2>/dev/null || true
    rm -rf data/raw/fma_metadata_raw

    echo "Downloading FMA-small (~7.2 GB, 8000 tracks at 30s)..."
    wget -c "https://os.unil.cloud.switch.ch/fma/fma_small.zip" \
         -O "$FMA_ZIP" --progress=dot:giga

    echo "Extracting FMA-small..."
    unzip -q "$FMA_ZIP" -d data/raw/
    rm -f "$FMA_ZIP"
    echo "FMA-small extracted to $RAW_FMA"
else
    echo "FMA-small already extracted at $RAW_FMA"
fi

# ---------------------------------------------------------------------------
# Step 2: Download FakeMusicCaps via HuggingFace
# ---------------------------------------------------------------------------
log "Step 2 — Download FakeMusicCaps"

if [ ! -d "$RAW_FMC/musicgen" ]; then
    python3 - <<'PYEOF'
import sys
from pathlib import Path

# FakeMusicCaps: 27,605 tracks, 5 generators, 10s each
# Available at: https://huggingface.co/datasets/lucacoma/FakeMusicCaps
# The dataset includes both real (MusicCaps) and fake tracks

raw_fmc = Path("data/raw/fakemusiccaps")

try:
    from datasets import load_dataset
    print("Downloading FakeMusicCaps via HuggingFace datasets library...")

    # The dataset has an 'audio' column and a 'generator' column
    # We process in streaming mode to avoid loading everything at once
    ds = load_dataset("lucacoma/FakeMusicCaps", split="train", streaming=False)
    print(f"Dataset loaded: {len(ds)} rows")
    print("Columns:", ds.column_names)

    # Organize by generator + label
    import soundfile as sf
    import numpy as np

    generators_seen = set()
    real_count = 0
    fake_count = 0

    for i, row in enumerate(ds):
        if i % 1000 == 0:
            print(f"  Processing row {i}/{len(ds)}...", flush=True)

        # Identify whether real or fake and which generator
        is_fake = row.get("is_fake", row.get("label", 0))
        gen = str(row.get("generator", row.get("model", "unknown"))).lower().replace(" ", "_")

        if is_fake:
            subdir = raw_fmc / gen
            track_id = f"fmc_{gen}_{i:06d}"
            fake_count += 1
        else:
            subdir = raw_fmc / "real"
            track_id = f"fmc_real_{i:06d}"
            real_count += 1

        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f"{track_id}.wav"

        if not out_path.exists():
            audio = row["audio"]
            arr = np.array(audio["array"])
            sr = audio["sampling_rate"]
            sf.write(str(out_path), arr, sr)

    print(f"FakeMusicCaps written: {real_count} real, {fake_count} fake")
    print("Generators:", list(raw_fmc.iterdir()))

except ImportError:
    print("'datasets' library not found. Install with: pip install datasets soundfile")
    print("Trying huggingface_hub snapshot download as fallback...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="lucacoma/FakeMusicCaps",
            repo_type="dataset",
            local_dir="data/raw/fakemusiccaps_hf",
        )
        print("Downloaded to data/raw/fakemusiccaps_hf — check structure and adjust paths")
    except Exception as e:
        print(f"HuggingFace download also failed: {e}")
        print("Manual download: https://huggingface.co/datasets/lucacoma/FakeMusicCaps")
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
PYEOF
else
    echo "FakeMusicCaps already downloaded at $RAW_FMC"
fi

# ---------------------------------------------------------------------------
# Step 3: Canonicalize FMA-small (real tracks)
# ---------------------------------------------------------------------------
log "Step 3 — Canonicalize FMA-small (real tracks)"

if [ ! -f "$CANON_FMA/canonical_manifest.csv" ]; then
    poetry run python scripts/build_generic_corpus.py \
        --input-dir  "$RAW_FMA" \
        --output-dir "$CANON_FMA" \
        --label real \
        --source fma \
        --algorithm "" \
        --target-sr 24000 --mp3-bitrate 64 --target-lufs -23.0 --max-duration 120 \
        --workers 6 --flush-every 200
else
    echo "FMA canonical corpus already built. Checking..."
    python3 -c "
import pandas as pd
df = pd.read_csv('$CANON_FMA/canonical_manifest.csv')
ok = df[df.get('status','ok') == 'ok'] if 'status' in df.columns else df
print(f'FMA canonical: {len(ok)} ok tracks (real={( ok[\"label\"]==\"real\").sum()})')
"
fi

# ---------------------------------------------------------------------------
# Step 4: Canonicalize FakeMusicCaps (per-generator fake tracks + real tracks)
# ---------------------------------------------------------------------------
log "Step 4 — Canonicalize FakeMusicCaps"

FMC_MANIFESTS=()

# Detect generators from subdirectory structure
for gen_dir in "$RAW_FMC"/*/; do
    gen=$(basename "$gen_dir")
    if [ "$gen" == "real" ]; then
        label="real"
        algo=""
        source_tag="musiccaps"
    else
        label="fake"
        algo="$gen"
        source_tag="fakemusiccaps"
    fi

    out_dir="${CANON_FMC}/${gen}"
    if [ ! -f "${out_dir}/canonical_manifest.csv" ]; then
        echo "Canonicalizing FakeMusicCaps/$gen ($label, algorithm=$algo)..."
        poetry run python scripts/build_generic_corpus.py \
            --input-dir  "$gen_dir" \
            --output-dir "$out_dir" \
            --label "$label" \
            --source "$source_tag" \
            --algorithm "$algo" \
            --fake-label "full fake" \
            --target-sr 24000 --mp3-bitrate 64 --target-lufs -23.0 --max-duration 120 \
            --workers 6 --flush-every 200
    else
        echo "Already built: FakeMusicCaps/$gen"
    fi
    FMC_MANIFESTS+=("${out_dir}/canonical_manifest.csv")
done

# ---------------------------------------------------------------------------
# Step 5: Build combined manifest (SONICS + FMA + FakeMusicCaps)
# ---------------------------------------------------------------------------
log "Step 5 — Building combined manifest"

# Collect all manifests
ALL_MANIFESTS=("$SONICS_MANIFEST" "$CANON_FMA/canonical_manifest.csv")
for m in "${FMC_MANIFESTS[@]}"; do
    ALL_MANIFESTS+=("$m")
done

echo "Merging ${#ALL_MANIFESTS[@]} manifests..."
poetry run python scripts/build_generic_corpus.py \
    --merge-manifests "${ALL_MANIFESTS[@]}" \
    --output-manifest "$COMBINED_MANIFEST"

python3 - <<'PYEOF'
import pandas as pd
df = pd.read_csv("data/processed/combined_external_manifest.csv")
ok = df[df.get("status","ok") == "ok"] if "status" in df.columns else df
print(f"\nCombined manifest: {len(ok):,} tracks")
print(f"  Real: {(ok['label']=='real').sum():,}")
print(f"  Fake: {(ok['label']=='fake').sum():,}")
print("\nBy algorithm (fake only):")
for alg, n in ok[ok['label']=='fake']['algorithm'].value_counts().items():
    print(f"  {alg:35s} {n:6,}")
if 'source' in ok.columns:
    print("\nBy source:")
    for src, n in ok['source'].value_counts().items():
        print(f"  {src:20s} {n:6,}")
PYEOF

# ---------------------------------------------------------------------------
# Step 6: Run evaluation on combined manifest
# Existing SONICS embeddings are reused from the cache.
# FMA + FakeMusicCaps get extracted and added to the same cache dir.
# ---------------------------------------------------------------------------
log "Step 6 — Running evaluation (EnCodec RealNVP LOGO, combined corpus)"
echo "Note: existing SONICS cache (~59k .npy files) is REUSED."
echo "      FMA (8k) + FMC (27k) = ~35k new extractions → ~1-2 hours GPU"

poetry run python scripts/run_balanced_ablation.py \
    --embeddings         encodec \
    --per-stratum        50000 \
    --analysis-duration  55 \
    --with-temporal \
    --window-duration    2 \
    --hop-duration       1 \
    --estimators         twonn \
    --canonical-manifest "$COMBINED_MANIFEST" \
    --preprocess-mode    preprocessed \
    --embedding-cache-dir "$EMB_CACHE" \
    --device             cuda \
    --generator-shift-eval \
    --window-flow-eval \
    --wf-window-duration  2.0 \
    --wf-hop-duration     1.0 \
    --wf-pca-components   128 \
    --wf-flow-epochs      200 \
    --seed               42 \
    --output-dir         "$OUT_DIR" \
    2>&1

log "Evaluation complete"

# ---------------------------------------------------------------------------
# Step 7: Post-process — extract FPR on FMA, per-generator AUC
# ---------------------------------------------------------------------------
log "Step 7 — Analysis"

python3 - <<'PYEOF'
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

wf = pd.read_csv("data/processed/external_eval/window_flow_eval.csv")

# wf_mean is log-likelihood; negate for anomaly score
score_col = "encodec_wf_mean"
wf["anomaly"] = -wf[score_col]

real_mask   = wf["label"] == "real"
fake_mask   = wf["label"] == "fake"

# ── FPR on FMA real tracks ─────────────────────────────────────────────────
# Calibrate threshold at 5% FPR on SONICS held-out reals, apply to FMA
sonics_real = wf[real_mask & (wf.get("source", "sonics") == "sonics" if "source" in wf.columns
                               else real_mask)]["anomaly"].dropna()
fma_real    = wf[real_mask & (wf.get("source", "") == "fma" if "source" in wf.columns
                               else pd.Series([False]*len(wf)))]["anomaly"].dropna()

if len(sonics_real) > 0 and len(fma_real) > 0:
    threshold_5pct = np.percentile(sonics_real, 95)   # 5% of SONICS reals above this
    fpr_fma = (fma_real > threshold_5pct).mean()
    print(f"\n=== FPR on FMA real tracks (threshold @ 5% SONICS FPR) ===")
    print(f"  FMA tracks flagged as AI: {fpr_fma:.1%}  (n={len(fma_real):,})")
    print(f"  SONICS held-out FPR: 5.0% (calibration)")

    fpr_df = pd.DataFrame({
        "source": ["sonics_real_calibration", "fma_real"],
        "n_tracks": [len(sonics_real), len(fma_real)],
        "threshold_5pct": [threshold_5pct, threshold_5pct],
        "fpr_at_5pct_threshold": [0.05, fpr_fma],
    })
    fpr_df.to_csv("data/processed/external_eval/fpr_fma.csv", index=False)
    print("  Saved → fpr_fma.csv")
else:
    print("  NOTE: 'source' column not in window_flow_eval.csv — run analysis on ablation_features.csv")

# ── Per-generator AUC (all sources) ───────────────────────────────────────
print(f"\n=== Window-Flow LOGO per generator (all sources) ===")
all_real_anom = wf.loc[real_mask, "anomaly"].dropna().values

rows = []
for alg in sorted(wf.loc[fake_mask, "algorithm"].dropna().unique()):
    fs = wf.loc[fake_mask & (wf["algorithm"] == alg), "anomaly"].dropna().values
    if len(fs) < 5:
        continue
    y = np.r_[np.zeros(len(all_real_anom)), np.ones(len(fs))]
    s = np.r_[all_real_anom, fs]
    auc = roc_auc_score(y, s)
    fpr_c, tpr_c, _ = roc_curve(y, s)
    fnr_c = 1 - tpr_c
    eer = float(fpr_c[np.nanargmin(np.abs(fnr_c - fpr_c))])
    rows.append({"algorithm": alg, "n_fake": len(fs), "auc": auc, "eer": eer})
    src = "SONICS" if alg in {"chirp-v2-xxl-alpha","chirp-v3","chirp-v3.5","udio-120s","udio-30s"} else "FMC"
    print(f"  [{src:5s}] {alg:30s}  AUC={auc:.4f}  EER={eer:.4f}  n={len(fs):,}")

pd.DataFrame(rows).to_csv("data/processed/external_eval/per_generator_auc.csv", index=False)
print("\n  Saved → per_generator_auc.csv")
PYEOF

# ---------------------------------------------------------------------------
# Step 8: Sync results to S3
# ---------------------------------------------------------------------------
log "Step 8 — Sync to S3"

aws s3 sync "$OUT_DIR/"       "$S3/processed/external_eval/"           --no-progress
aws s3 sync "$LOG_DIR/"       "$S3/logs/"                              --no-progress
aws s3 cp   "$COMBINED_MANIFEST" "$S3/processed/combined_external_manifest.csv"
# FMA + FMC canonical WAVs are NOT synced (reproducible; 35 GB)
# Their manifests ARE synced
aws s3 cp "$CANON_FMA/canonical_manifest.csv" "$S3/processed/canonical_fma_manifest.csv"
for gen_dir in "$CANON_FMC"/*/; do
    gen=$(basename "$gen_dir")
    aws s3 cp "${gen_dir}/canonical_manifest.csv" \
        "$S3/processed/canonical_fmc_${gen}_manifest.csv" 2>/dev/null || true
done

log "ALL DONE — external corpus eval"
echo ""
echo "Key outputs:"
echo "  data/processed/external_eval/window_flow_eval.csv    ← AUC for FMC unseen generators"
echo "  data/processed/external_eval/fpr_fma.csv             ← FPR on FMA real tracks"
echo "  data/processed/external_eval/per_generator_auc.csv   ← all generators"
