#!/usr/bin/env bash
# =============================================================================
# run_combined_sonics_musiccaps_flow.sh
#
# Plan item: "Train all reals (SONICS + MusicCaps) and rerun, rather than only
# SONICS-real -> SONICS-eval and separately MusicCaps-real -> FakeMusicCaps-eval."
#
# Everything so far trains the flow on ONE real corpus and tests it against
# fakes from the SAME dataset the reals came from (SONICS-real -> SONICS-fake
# headline; MusicCaps-real -> FakeMusicCaps-fake in run_fakemusiccaps_native_flow.sh),
# plus a cross-dataset zero-shot test in ONE direction only (SONICS-trained flow
# scored on FakeMusicCaps, in build_fakemusiccaps_native_comparison.sh). This
# script does the thing neither of those does: pool SONICS-real (12,722) AND
# MusicCaps-real (~5,355) into ONE ~18,077-track real training corpus, train a
# single flow on it (K=12 — our new best depth, see run_headline_k12_suite.sh),
# and evaluate it against BOTH SONICS-fake (5 generators, chirp/udio) AND
# FakeMusicCaps-fake (5 generators, MusicGen/AudioLDM2/MusicLDM/StableAudioOpen/
# Mustango) in the SAME run. This directly tests whether broadening the real
# manifold across two independently-collected real-music corpora improves (or
# degrades) detection on either fake population — the true test of "does more
# real data generalize" that B2 (SONICS + FMA-medium) started, but with a real
# corpus (MusicCaps) that is topically matched to one of our two fake test sets
# rather than an unrelated third corpus (FMA).
#
# Honest caveat to carry into the writeup: SONICS canonical clips are analysed
# at up to 55s (26 4s/2s windows) while FakeMusicCaps/MusicCaps clips are only
# 10s natively (~4 windows) — a real evidence-budget asymmetry between the two
# fake populations in this eval, not an artifact of this script (MusicDET's own
# protocol also crops everything to 4s, so this is a property of the source
# datasets, not something we introduced).
#
# Prerequisites:
#   bash scripts/build_fakemusiccaps_native_comparison.sh   (downloads MusicCaps-real)
#   bash scripts/run_fakemusiccaps_native_flow.sh           (canonicalizes MusicCaps-real
#                                                             + all 5 FakeMusicCaps fake
#                                                             generators; FIXED this round
#                                                             to no longer corrupt canonical_path)
#   Full SONICS canonical corpus + embedding cache (from run_full_sonics_pipeline.sh)
#
# Usage:
#   bash scripts/run_combined_sonics_musiccaps_flow.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
FMC_REAL_MANIFEST="data/processed/canonical_fmc_native/canonical_manifest_real.csv"
FMC_FAKE_DIR="data/processed/canonical_fmc_native"
COMBINED_MANIFEST="data/processed/canonical_sonics_musiccaps_combined/combined_manifest.csv"
# IMPORTANT: reuse the SAME cache dir as the SONICS headline (data/emb_cache_encodec_full),
# not a fresh one. Cache keys are track_id-based (see run_balanced_ablation.py's
# _cache_path()), so all 59,280 SONICS tracks HIT the existing cache here and only the
# ~33k new MusicCaps-real + FakeMusicCaps-fake tracks need fresh extraction — instead of
# re-extracting the full ~92k-track union from scratch.
EMB_CACHE="data/emb_cache_encodec_full"
OUT_DIR="data/processed/sonics_musiccaps_combined_flow"
FLOW_PATH="$OUT_DIR/combined_real_flow.pt"   # -> combined_real_flow_encodec.pt
LOG_DIR="logs"

mkdir -p "$(dirname "$COMBINED_MANIFEST")" "$OUT_DIR" "$LOG_DIR"

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -f "$SONICS_MANIFEST" ]]; then
    echo "ERROR: $SONICS_MANIFEST not found. Run run_full_sonics_pipeline.sh first." >&2
    exit 1
fi
if [[ ! -f "$FMC_REAL_MANIFEST" ]]; then
    echo "ERROR: $FMC_REAL_MANIFEST not found. Run:" >&2
    echo "  bash scripts/build_fakemusiccaps_native_comparison.sh" >&2
    echo "  bash scripts/run_fakemusiccaps_native_flow.sh   (Steps 1-3 build this manifest)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Merge SONICS (real+fake) + MusicCaps-real + FakeMusicCaps-fake (x5)
# into one combined manifest.
# ---------------------------------------------------------------------------
log "Step 1 — Merge SONICS + MusicCaps-real + FakeMusicCaps-fake manifests"
FMC_FAKE_MANIFESTS=()
for gen_dir in "$FMC_FAKE_DIR"/fake_*/; do
    m="${gen_dir}canonical_manifest.csv"
    if [[ -f "$m" ]]; then
        FMC_FAKE_MANIFESTS+=("$m")
    fi
done
if [[ ${#FMC_FAKE_MANIFESTS[@]} -lt 5 ]]; then
    echo "ERROR: expected 5 FakeMusicCaps fake-generator manifests under $FMC_FAKE_DIR/fake_*/, found ${#FMC_FAKE_MANIFESTS[@]}." >&2
    echo "Run run_fakemusiccaps_native_flow.sh (Step 2) first." >&2
    exit 1
fi
echo "Found ${#FMC_FAKE_MANIFESTS[@]} FakeMusicCaps fake-generator manifests: ${FMC_FAKE_MANIFESTS[*]}"

python scripts/build_generic_corpus.py \
    --merge-manifests "$SONICS_MANIFEST" "$FMC_REAL_MANIFEST" "${FMC_FAKE_MANIFESTS[@]}" \
    --output-manifest "$COMBINED_MANIFEST"

python3 - "$COMBINED_MANIFEST" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
n_real = (ok["label"] == "real").sum()
n_fake = (ok["label"] == "fake").sum()
print(f"Combined manifest: {len(ok):,} ok rows (real={n_real:,}, fake={n_fake:,})")
if n_real < 15000:
    raise SystemExit(f"ERROR: expected ~18,077 combined real tracks (SONICS 12,722 + MusicCaps ~5,355), got {n_real}. "
                      "Check that FMC_REAL_MANIFEST's canonical_path values actually resolve (see the F2 bugfix note "
                      "in run_fakemusiccaps_native_flow.sh) before proceeding.")
print("By algorithm (fake):")
print(ok[ok["label"] == "fake"]["algorithm"].value_counts().to_string())
PYEOF

# ---------------------------------------------------------------------------
# Step 2: Train the K=12 flow on the FULL combined real pool (SONICS-real +
# MusicCaps-real, ~18,077 tracks — --max-real set high so sampling never caps
# below the full available real pool regardless of the (much larger) combined
# fake count), then LOGO-score both fake populations in the same pass.
# ---------------------------------------------------------------------------
log "Step 2 — Train K=12 flow on combined SONICS+MusicCaps real corpus; score both fake populations"
python scripts/run_balanced_ablation.py \
    --embeddings          encodec \
    --per-stratum         50000 \
    --max-real            100000 \
    --analysis-duration   55 \
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
    --wf-pca-components      128 \
    --wf-flow-epochs         200 \
    --wf-n-coupling-layers   12 \
    --wf-save-flow-path      "$FLOW_PATH" \
    --seed                   42 \
    --output-dir             "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/combined_sonics_musiccaps_flow.log"

# ---------------------------------------------------------------------------
# Step 3: Per-dataset, per-generator AUC/EER breakdown, and comparison against
# (a) the SONICS-only headline (does adding MusicCaps-real help or hurt SONICS
#     detection?) and (b) the cross-dataset zero-shot test from
#     build_fakemusiccaps_native_comparison.sh (does adding MusicCaps-real to
#     TRAINING beat scoring FakeMusicCaps with a flow that never saw it?).
# ---------------------------------------------------------------------------
log "Step 3 — Per-dataset / per-generator comparison"
python3 - <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer

SONICS_ALGS = {"chirp-v2-xxl-alpha", "chirp-v3", "chirp-v3.5", "udio-120s", "udio-30s"}
FMC_ALGS = {"MusicGen_medium", "audioldm2", "musicldm", "mustango", "stable_audio_open"}

wf = pd.read_csv("data/processed/sonics_musiccaps_combined_flow/window_flow_eval.csv", low_memory=False)
mean_col = next((c for c in wf.columns if c.endswith("wf_mean")), None)
real = wf[wf["label"] == "real"][mean_col].dropna().to_numpy()
print(f"Combined real (held-out, LOGO) n={len(real)} (SONICS + MusicCaps pooled)\n")

# Reference numbers for side-by-side printing (fill in from prior runs if available)
sonics_headline_auc = {
    "chirp-v2-xxl-alpha": 0.9946, "chirp-v3": 0.9610, "chirp-v3.5": 0.9221,
    "udio-120s": 0.9036, "udio-30s": 0.7334,
}
fmc_crossdataset_auc_path = Path("data/processed/external_scores/fakemusiccaps_native_auc.csv")
fmc_crossdataset = {}
if fmc_crossdataset_auc_path.exists():
    d = pd.read_csv(fmc_crossdataset_auc_path)
    fmc_crossdataset = dict(zip(d["algorithm"], d["auc"]))

rows = []
for alg in sorted(wf[wf["label"] == "fake"]["algorithm"].dropna().unique()):
    fake = wf[(wf["label"] == "fake") & (wf["algorithm"] == alg)][mean_col].dropna().to_numpy()
    if len(fake) < 5:
        continue
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    s = np.r_[-real, -fake]
    auc_res = bootstrap_auc(y, s)
    eer_res = bootstrap_eer(y, s)
    dataset = "SONICS" if alg in SONICS_ALGS else ("FakeMusicCaps" if alg in FMC_ALGS else "unknown")
    ref = sonics_headline_auc.get(alg) if dataset == "SONICS" else fmc_crossdataset.get(alg)
    rows.append({
        "dataset": dataset, "algorithm": alg, "n_fake": len(fake),
        "combined_train_auc": round(auc_res["auc"], 4),
        "combined_train_eer_pct": round(eer_res["eer"] * 100, 2),
        "reference_auc_single_corpus_train": ref,
    })
    ref_str = f"{ref:.4f}" if ref is not None else "n/a"
    print(f"  [{dataset:14s}] {alg:22s}  combined-train AUC={auc_res['auc']:.4f}  EER={eer_res['eer']*100:.2f}%  "
          f"(single-corpus-train reference: {ref_str})")

out = Path("data/processed/sonics_musiccaps_combined_flow/combined_vs_single_corpus.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved -> {out}")
print("\nInterpretation guide: combined_train_auc > reference means broadening the real manifold to a second,")
print("independently-collected real corpus HELPED that generator's detection; combined_train_auc < reference means")
print("it hurt (specificity/coverage tradeoff, same pattern seen in B2 with SONICS+FMA-medium).")
PYEOF

log "Done."
