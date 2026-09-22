#!/usr/bin/env bash
# =============================================================================
# run_bitrate_sweep.sh
#
# Tests robustness of the EnCodec window-flow detector across MP3 bitrates.
#
# Design:
#   - Re-encode a balanced 5000-track subset at 32kbps and 128kbps by
#     running an MP3 round-trip on the EXISTING 64kbps canonical WAVs
#     (faster than rebuilding from raw audio, no raw files needed)
#   - Extract EnCodec embeddings at each bitrate
#   - Run window-flow evaluation
#   - Compare AUC/EER vs the existing 64kbps results
#
# Claim it supports:
#   "Our method is robust to MP3 bitrate; STFT-based methods collapse under
#    the same compression (MusicDET EER 41.75% at 64kbps vs ours 16.8%)."
#
# Expected runtime: ~30 min re-encode + ~4 h experiment per bitrate
# Expected disk: ~14 GB canonical WAVs + ~3 GB embeddings per bitrate
#
# Usage:
#   bash scripts/run_bitrate_sweep.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

mkdir -p logs data/processed

CANONICAL_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"

# 5000-track balanced subset manifest — build once if not present
SUBSET_MANIFEST="data/processed/bitrate_sweep_subset.csv"
if [[ ! -f "$SUBSET_MANIFEST" ]]; then
    echo "Building balanced 5000-track subset manifest..."
    python3 - <<'PYEOF'
import pandas as pd
import numpy as np

mf = pd.read_csv("data/processed/canonical_sonics_full/canonical_manifest.csv")
mf = mf[mf["status"] == "ok"]

real_sample = mf[mf["label"] == "real"].sample(1000, random_state=42)
fake_parts = []
for alg, grp in mf[mf["label"] == "fake"].groupby("algorithm"):
    fake_parts.append(grp.sample(min(len(grp), 800), random_state=42))

fake_concat = pd.concat(fake_parts)
fake_sample = fake_concat.sample(min(4000, len(fake_concat)), random_state=42)
subset = pd.concat([real_sample, fake_sample]).sample(frac=1, random_state=42)
subset.to_csv("data/processed/bitrate_sweep_subset.csv", index=False)
print(f"Subset: {len(subset)} tracks  real={len(real_sample)}  fake={len(fake_sample)}")
print(subset["algorithm"].value_counts())
PYEOF
fi

# ---------------------------------------------------------------------------
# Function: run one bitrate
# ---------------------------------------------------------------------------
run_bitrate() {
    local BITRATE="$1"
    echo ""
    echo "================================================================"
    echo "  Running bitrate sweep: ${BITRATE}kbps"
    echo "================================================================"

    OUT_DIR="data/processed/bitrate_sweep_${BITRATE}kbps"
    CANON_DIR="data/processed/canonical_bitrate_${BITRATE}kbps"
    EMB_CACHE="data/emb_cache_bitrate_${BITRATE}kbps"
    LOG="logs/bitrate_${BITRATE}kbps.log"

    mkdir -p "$CANON_DIR" "$EMB_CACHE" "$OUT_DIR"

    # Step 1: re-encode existing 64kbps canonical WAVs at the target bitrate
    # (MP3 round-trip: WAV → MP3@Xkbps → WAV, using ffmpeg via subprocess)
    if [[ ! -f "$CANON_DIR/canonical_manifest.csv" ]]; then
        echo "  [${BITRATE}kbps] Re-encoding canonical WAVs at ${BITRATE}kbps..."
        BITRATE_VAL="$BITRATE" poetry run python3 - <<'PYEOF'
import os, subprocess, sys, pandas as pd, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BITRATE = int(os.environ["BITRATE_VAL"])
subset = pd.read_csv("data/processed/bitrate_sweep_subset.csv")
out_dir = Path(f"data/processed/canonical_bitrate_{BITRATE}kbps")
out_dir.mkdir(parents=True, exist_ok=True)

def reencode(row):
    src = Path(str(row.canonical_path))
    if not src.exists():
        return None
    dst = out_dir / row.label / f"{row.track_id}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return {**row._asdict(), "canonical_path": str(dst), "status": "ok"}
    tmp_mp3 = dst.with_suffix(".mp3")
    try:
        # encode to MP3 at target bitrate
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-b:a", f"{BITRATE}k", str(tmp_mp3)],
            check=True, capture_output=True)
        # decode back to WAV (24kHz mono, same as canonical pipeline)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_mp3),
             "-ar", "24000", "-ac", "1", str(dst)],
            check=True, capture_output=True)
        tmp_mp3.unlink(missing_ok=True)
        return {**row._asdict(), "canonical_path": str(dst), "status": "ok"}
    except Exception as exc:
        tmp_mp3.unlink(missing_ok=True)
        return {**row._asdict(), "status": f"error: {exc}"}

rows_out = []
n_ok = n_err = 0
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(reencode, row): i for i, row in enumerate(subset.itertuples())}
    for i, fut in enumerate(as_completed(futures), 1):
        result = fut.result()
        if result is not None:
            rows_out.append(result)
            if result["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
        if i % 500 == 0:
            print(f"  {i}/{len(subset)} ok={n_ok} err={n_err}", flush=True)

mf_out = pd.DataFrame(rows_out)
mf_out.to_csv(out_dir / "canonical_manifest.csv", index=False)
print(f"\nDone: {n_ok} ok, {n_err} errors → {out_dir}/canonical_manifest.csv")
PYEOF
        echo "  [${BITRATE}kbps] Re-encoding done."
    else
        echo "  [${BITRATE}kbps] Canonical corpus already exists — skipping."
    fi

    # Step 2: run window-flow experiment
    if [[ ! -f "$OUT_DIR/window_flow_eval.csv" ]]; then
        echo "  [${BITRATE}kbps] Running window-flow experiment..."
        poetry run python scripts/run_balanced_ablation.py \
            --embeddings encodec \
            --per-stratum 50000 \
            --analysis-duration 55 \
            --window-duration 4 \
            --hop-duration 2 \
            --estimators twonn \
            --canonical-manifest "$CANON_DIR/canonical_manifest.csv" \
            --preprocess-mode preprocessed \
            --embedding-cache-dir "$EMB_CACHE" \
            --device cuda \
            --window-flow-eval \
            --wf-window-duration 4.0 \
            --wf-hop-duration 2.0 \
            --wf-pca-components 128 \
            --wf-flow-epochs 200 \
            --wf-save-flow-path "$OUT_DIR/sonics_real_flow.pt" \
            --seed 42 \
            --output-dir "$OUT_DIR" \
            2>&1 | tee "$LOG"
        echo "  [${BITRATE}kbps] Experiment done."
    else
        echo "  [${BITRATE}kbps] Results already exist at $OUT_DIR — skipping."
    fi

    echo "  [${BITRATE}kbps] Results → $OUT_DIR"
}

# ---------------------------------------------------------------------------
# Run both bitrates
# ---------------------------------------------------------------------------
run_bitrate 32
run_bitrate 128

# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Bitrate sweep comparison"
echo "================================================================"
python3 - <<'PYEOF'
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path

BITRATES = [32, 64, 128]
# 64kbps is the existing full-scale run
DIRS = {
    32: Path("data/processed/bitrate_sweep_32kbps"),
    64: Path("data/processed/full_sonics_all"),
    128: Path("data/processed/bitrate_sweep_128kbps"),
}
WF_FILE = "window_flow_eval.csv"

results = []
for br, d in DIRS.items():
    wf_path = d / WF_FILE
    if not wf_path.exists():
        print(f"  [{br}kbps] No results at {wf_path}")
        continue
    df = pd.read_csv(wf_path)
    # Find wf_mean column (may be prefixed with embedder name)
    mean_col = next((c for c in df.columns if "wf_mean" in c), None)
    if mean_col is None:
        print(f"  [{br}kbps] No wf_mean column found")
        continue
    real = df[df["label"] == "real"][mean_col].dropna()
    for alg in sorted(df["algorithm"].dropna().unique()):
        fake = df[(df["label"] == "fake") & (df["algorithm"] == alg)][mean_col].dropna()
        if len(fake) < 5:
            continue
        y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
        s = np.r_[-real, -fake]   # negate loglik → anomaly score
        try:
            auc = roc_auc_score(y, s)
            fpr_c, tpr_c, _ = roc_curve(y, s)
            fnr_c = 1 - tpr_c
            eer = float(fpr_c[np.nanargmin(np.abs(fnr_c - fpr_c))])
            results.append({"bitrate_kbps": br, "algorithm": alg, "AUC": round(auc, 4), "EER": round(eer, 4)})
        except Exception:
            pass

if results:
    pivot = pd.DataFrame(results).pivot(index="algorithm", columns="bitrate_kbps", values="AUC")
    print("\n=== AUC by generator and bitrate ===")
    print(pivot.to_string())
    out = Path("reports") / "bitrate_sweep_auc.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nFull results → {out}")
else:
    print("  No results to compare yet. Run both bitrates first.")
PYEOF

echo ""
echo "Done. Results in data/processed/bitrate_sweep_*/"
echo "Sync to S3:"
echo "  aws s3 sync data/processed/bitrate_sweep_32kbps/ s3://orfium-dev-research-projects/ai-music-intrinsic-dimension/processed/bitrate_sweep_32kbps/"
echo "  aws s3 sync data/processed/bitrate_sweep_128kbps/ s3://orfium-dev-research-projects/ai-music-intrinsic-dimension/processed/bitrate_sweep_128kbps/"
