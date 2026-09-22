#!/usr/bin/env bash
# =============================================================================
# run_fakemusiccaps_native_flow.sh
#
# EXACT protocol match to MusicDET Table 1 (FakeMusicCaps, zero-shot EER 4.51%).
#
# Verified against the actual MusicDET paper text (arXiv 2605.18072): their
# Table 1 zero-shot model is trained real-only on MusicCaps-real ONLY (~5,500
# clips — the real captions/audio the 5 generators were prompted from), NOT
# on SONICS-real, and NOT on any "combined" cross-benchmark corpus. Table 1
# and Table 2 are two entirely separate real-only-trained models.
#
# This is DIFFERENT from build_fakemusiccaps_native_comparison.sh, which scores
# MusicCaps-real + FakeMusicCaps-fake against our SONICS-trained frozen flow
# (a true cross-DATASET zero-shot generalization test — a claim MusicDET's own
# paper never makes, since they never train on one dataset and test on another).
# That script answers "does our method generalize beyond its training domain?"
# THIS script answers "apples-to-apples, does our architecture match MusicDET's
# reported 4.51% EER when trained on the identical small real corpus?"
#
# Report BOTH numbers in the paper — they test different, complementary claims.
#
# Prerequisite: run build_fakemusiccaps_native_comparison.sh first (or at least
# its Step 1) so data/raw/fakemusiccaps_real_only/ exists.
#
# Usage:
#   bash scripts/run_fakemusiccaps_native_flow.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

RAW_REAL_DIR="data/raw/fakemusiccaps_real_only"
RAW_FAKE_DIR="data/external/fakemusiccaps"   # Zenodo mirror, 5 generator subdirs, fake-only
CANON_REAL="data/processed/canonical_fmc_native/real"
CANON_FAKE="data/processed/canonical_fmc_native/fake"
COMBINED_MANIFEST="data/processed/canonical_fmc_native/combined_manifest.csv"
EMB_CACHE="data/emb_cache_encodec_fmc_native"

# K (n_coupling_layers) — see the "Diagnosing a catastrophic-looking result" note
# above Step 4. Override with: K=2 bash scripts/run_fakemusiccaps_native_flow.sh
# Default lowered to 2 (from run_balanced_ablation.py's global default of 8):
# MusicCaps-real has only ~5,355 tracks / ~21k total windows, i.e. ~17k windows
# per LOGO training fold vs. SONICS' ~264k+ per fold (12x more) — an 8-layer
# RealNVP that generalizes fine on SONICS can catastrophically overfit on a
# corpus this thin, producing degenerate LOGO-held-out real scores. The K=2/K=4
# points in run_flow_depth_ablation.sh's SONICS sweep (AUC 0.899/0.920) show a
# shallower flow is still perfectly usable — safer starting point for 1/12th the
# data. If a first run at K=8 already produced near-0 AUC across ALL generators
# identically (the tell for this pathology, not a real per-generator signal),
# re-run with a lower K before assuming the result is meaningful.
K="${K:-2}"
OUT_DIR="data/processed/fmc_native_flow_k${K}"

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -d "$RAW_REAL_DIR" ]] || [[ "$(find "$RAW_REAL_DIR" -name '*.wav' | wc -l)" -lt 100 ]]; then
    echo "ERROR: $RAW_REAL_DIR missing or too small. Run:"
    echo "  bash scripts/build_fakemusiccaps_native_comparison.sh   (Step 1 downloads this)"
    exit 1
fi
if [[ ! -d "$RAW_FAKE_DIR" ]]; then
    echo "ERROR: $RAW_FAKE_DIR missing. Run: bash scripts/download_external_data.sh --fakemusiccaps"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Canonicalize MusicCaps-real (10s clips -> our standard 24kHz/64kbps
# MP3-roundtrip pipeline, matching every other corpus in this repo)
# ---------------------------------------------------------------------------
log "Step 1 — Canonicalize MusicCaps-real"
# NOTE (bug fix): this used to write to a "..._realtmp" output-dir and then `mv`
# the resulting "real/" subfolder + manifest into place. build_generic_corpus.py
# bakes the *output-dir-relative* canonical_path into canonical_manifest.csv
# BEFORE the mv happens, so every canonical_path ended up pointing at the
# now-deleted "..._realtmp/real/..." location. Path(canonical_path).exists()
# then silently failed for every real row downstream in balanced_sampling.py,
# which fell back to its hardcoded SONICS-real corpus (12,722 tracks) instead
# of MusicCaps-real (~5,355 tracks) — invalidating the whole Table-1 comparison
# (identical AUC/EER across all 5 "generators" was the tell). Fix: write
# directly to the final parent dir so --label real creates "real/" in place,
# with NO move step, so canonical_path values recorded in the manifest are
# correct from the start. Only the manifest *filename* is renamed afterwards
# (in the same directory — renaming the CSV file itself doesn't touch the
# audio paths recorded inside it).
if [[ ! -f "$(dirname "$CANON_REAL")/canonical_manifest_real.csv" ]]; then
    python scripts/build_generic_corpus.py \
        --input-dir "$RAW_REAL_DIR" \
        --output-dir "$(dirname "$CANON_REAL")" \
        --label real --source musiccaps \
        --track-id-prefix auto \
        --target-sr 24000 --mp3-bitrate 64 --target-lufs -23.0 --max-duration 10 \
        --workers 6 --flush-every 200
    mv "$(dirname "$CANON_REAL")/canonical_manifest.csv" "$(dirname "$CANON_REAL")/canonical_manifest_real.csv"
else
    echo "Already canonicalized: $(dirname "$CANON_REAL")/canonical_manifest_real.csv"
fi

# ---------------------------------------------------------------------------
# Step 2: Canonicalize each FakeMusicCaps generator (fake)
# ---------------------------------------------------------------------------
log "Step 2 — Canonicalize FakeMusicCaps generators (fake)"
FAKE_MANIFESTS=()
for gen_dir in "$RAW_FAKE_DIR"/*/; do
    gen=$(basename "$gen_dir")
    out_dir="data/processed/canonical_fmc_native/fake_${gen}"
    if [[ ! -f "${out_dir}/canonical_manifest.csv" ]]; then
        echo "Canonicalizing $gen ..."
        python scripts/build_generic_corpus.py \
            --input-dir "$gen_dir" \
            --output-dir "$out_dir" \
            --label fake --source fakemusiccaps --algorithm "$gen" \
            --fake-label "full fake" \
            --track-id-prefix auto \
            --target-sr 24000 --mp3-bitrate 64 --target-lufs -23.0 --max-duration 10 \
            --workers 6 --flush-every 200
    else
        echo "Already canonicalized: $gen"
    fi
    FAKE_MANIFESTS+=("${out_dir}/canonical_manifest.csv")
done

# ---------------------------------------------------------------------------
# Step 3: Merge into one manifest (MusicCaps-real + all FakeMusicCaps-fake)
# ---------------------------------------------------------------------------
log "Step 3 — Merge manifests"
python scripts/build_generic_corpus.py \
    --merge-manifests "$(dirname "$CANON_REAL")/canonical_manifest_real.csv" "${FAKE_MANIFESTS[@]}" \
    --output-manifest "$COMBINED_MANIFEST"

# GATE: FakeMusicCaps re-generates every MusicCaps caption with 5 TTM systems, so
# the raw YouTube id is shared by 1 real + 5 fakes. The embedding cache is keyed on
# track_id alone, so colliding ids would make all 6 variants read ONE variant's
# embeddings and silently invalidate every number below (this happened: it produced
# bit-identical per-generator AUCs). --track-id-prefix auto prevents it; this check
# proves it for the merged manifest before any compute is spent.
log "Step 3b — Verify no track_id / embedding-cache collisions"
python scripts/diagnose_cache_collisions.py \
    --manifest "$COMBINED_MANIFEST" \
    --emb-cache "$EMB_CACHE" \
    --max-duration 10.0

# ---------------------------------------------------------------------------
# Step 4: Run the SAME window-flow LOGO pipeline used for the SONICS headline
# result, but on THIS manifest. The 5-fold LOGO trains a fresh flow on
# MusicCaps-real ONLY within each fold (never touches SONICS), and scores
# FakeMusicCaps-fake against it — this is the exact MusicDET Table 1 protocol.
# ---------------------------------------------------------------------------
log "Step 4 — Train + evaluate window-flow (MusicCaps-real only, LOGO, K=${K})"
python scripts/run_balanced_ablation.py \
    --embeddings          encodec \
    --per-stratum         50000 \
    --analysis-duration   10 \
    --window-duration     4 \
    --hop-duration        2 \
    --estimators          twonn \
    --canonical-manifest  "$COMBINED_MANIFEST" \
    --preprocess-mode     preprocessed \
    --embedding-cache-dir "$EMB_CACHE" \
    --device              cuda \
    --window-flow-eval \
    --wf-window-duration     4.0 \
    --wf-hop-duration        2.0 \
    --wf-pca-components      64 \
    --wf-flow-epochs         200 \
    --wf-n-coupling-layers   "$K" \
    --wf-save-flow-path      "$OUT_DIR/musiccaps_real_flow.pt" \
    --seed                42 \
    --output-dir          "$OUT_DIR" \
    2>&1

# ---------------------------------------------------------------------------
# Step 5: Report per-generator AUC/EER, format matching MusicDET Table 1
# ---------------------------------------------------------------------------
log "Step 5 — Comparison table vs MusicDET Table 1"
export FMC_OUT_DIR="$OUT_DIR"
python3 - <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer

import os

out_dir = os.environ.get("FMC_OUT_DIR", "data/processed/fmc_native_flow")
wf = pd.read_csv(f"{out_dir}/window_flow_eval.csv", low_memory=False)
mean_col = next((c for c in wf.columns if "wf_mean" in c), None)
real = wf[wf["label"] == "real"][mean_col].dropna().to_numpy()
print(f"MusicCaps-real (held-out, LOGO) n={len(real)}")

# Diagnostic: if AUC comes out near 0 (or near 1) IDENTICALLY across every fake
# generator, that's the tell for LOGO-fold overfitting on a thin corpus (real
# scores uniformly worse than ~any fake distribution) rather than a genuine,
# generator-specific signal. Print percentiles to see the effect size directly.
fake_all = wf[wf["label"] == "fake"][mean_col].dropna().to_numpy()
print(f"\nDiagnostic — wf_mean (raw log-likelihood; higher = more 'real-like' under the flow):")
print(f"  real  (n={len(real):5d}): p5={np.percentile(real, 5):8.2f}  p50={np.percentile(real, 50):8.2f}  p95={np.percentile(real, 95):8.2f}")
print(f"  fake  (n={len(fake_all):5d}): p5={np.percentile(fake_all, 5):8.2f}  p50={np.percentile(fake_all, 50):8.2f}  p95={np.percentile(fake_all, 95):8.2f}")
if np.median(fake_all) > np.median(real):
    print("  -> WARNING: median fake wf_mean > median real wf_mean — the held-out real windows score MORE")
    print("     anomalous than the fakes under this flow. If this persists at low K, it's a real data-")
    print("     starvation failure (report as a 'minimum viable corpus size' finding), not a code bug.")

# MusicDET's own published Table 1 zero-shot EER, for direct side-by-side reference
MUSICDET_TABLE1_EER = {
    "MusicGen_medium": 5.64, "musicgen": 5.64,
    "musicldm": 6.55,
    "audioldm2": 2.36,
    "stable_audio_open": 3.82,
    "mustango": 4.18,
}

rows = []
for alg in sorted(wf[wf["label"] == "fake"]["algorithm"].dropna().unique()):
    fake = wf[(wf["label"] == "fake") & (wf["algorithm"] == alg)][mean_col].dropna().to_numpy()
    if len(fake) < 5:
        continue
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    s = np.r_[-real, -fake]  # negate: higher anomaly = more fake
    auc_res = bootstrap_auc(y, s)
    eer_res = bootstrap_eer(y, s)
    mdet_eer = MUSICDET_TABLE1_EER.get(alg, float("nan"))
    rows.append({
        "algorithm": alg,
        "n_fake": len(fake),
        "our_auc": round(auc_res["auc"], 4),
        "our_eer_pct": round(eer_res["eer"] * 100, 2),
        "musicdet_table1_eer_pct": mdet_eer,
    })
    print(f"  {alg:25s}  our AUC={auc_res['auc']:.4f}  our EER={eer_res['eer']*100:.2f}%  "
          f"MusicDET Table 1 EER={mdet_eer:.2f}%")

df_out = pd.DataFrame(rows)
avg_eer = df_out["our_eer_pct"].mean()
print(f"\n  {'Avg (ours)':25s}  EER={avg_eer:.2f}%   MusicDET Table 1 Avg EER=4.51%")
out = Path(out_dir) / "native_comparison_vs_table1.csv"
df_out.to_csv(out, index=False)
print(f"\nSaved -> {out}")
PYEOF

log "Done."
