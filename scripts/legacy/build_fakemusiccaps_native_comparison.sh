#!/usr/bin/env bash
# =============================================================================
# build_fakemusiccaps_native_comparison.sh
#
# Fills the ONE missing piece for an exact MusicDET-protocol FakeMusicCaps
# comparison. We already have (per check_external_corpus_status.py):
#   - data/processed/external_scores/fakemusiccaps/external_scores.csv
#     -> all 27,605 FakeMusicCaps fakes (5 generators) scored against our
#        frozen SONICS-trained flow. This is a valid zero-shot generalization
#        number, but its "real" reference class is SONICS-real, not MusicCaps-real.
#   - data/processed/external_scores/fma_fpr/external_scores.csv
#     -> FPR test on 999 FMA real tracks against the same frozen flow.
#
# MusicDET's own published numbers (EER 4.51%/2.89%) are computed ENTIRELY
# within FakeMusicCaps' domain: MusicCaps-real vs FakeMusicCaps-fake. Our local
# FakeMusicCaps mirror (Zenodo, via download_external_data.sh) is fake-only —
# and so is EVERY FakeMusicCaps mirror (there is no is_fake/label column to
# filter on; confirmed against the paper, arXiv 2409.10684 — they train on
# FakeMusicCaps PLUS "the real music signals belonging to MusicCaps" as a
# SEPARATE dataset/class). This script:
#   1. Downloads the real MusicCaps audio from a HF community mirror that
#      bundles actual audio bytes (mahendra0203/musiccaps_processed_full) —
#      avoids a second YouTube/yt-dlp download battle, since google/MusicCaps
#      itself ships only YouTube ids + timestamps, no audio.
#   2. Scores it against the same frozen flow, same protocol, same output schema
#      as the existing fakemusiccaps fake scores.
#   3. Combines the two into a native-domain AUC/EER table directly comparable
#      to MusicDET Table 1/2.
#
# Usage:
#   bash scripts/build_fakemusiccaps_native_comparison.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

RAW_REAL_DIR="data/raw/fakemusiccaps_real_only"
FAKE_SCORES="data/processed/external_scores/fakemusiccaps/external_scores.csv"
REAL_OUT_DIR="data/processed/external_scores/fakemusiccaps_real"
FLOW_PATH="data/processed/full_sonics_all/sonics_real_flow_encodec.pt"

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -f "$FAKE_SCORES" ]]; then
    echo "ERROR: $FAKE_SCORES not found. Run download_external_data.sh + score_external_corpus.py"
    echo "for FakeMusicCaps fakes first (per check_external_corpus_status.py)."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Download the real MusicCaps audio.
#
# IMPORTANT (fixed): FakeMusicCaps (Zenodo / any of its HF mirrors, incl. the
# now-nonexistent "lucacoma/FakeMusicCaps") is 100% FAKE — it is Comanducci et
# al.'s TTM-regenerated captions, with NO real audio bundled and no
# is_fake/label column to filter on (confirmed against the paper, arXiv
# 2409.10684: they train on FakeMusicCaps PLUS "the real music signals
# belonging to MusicCaps" as a *separate* class/dataset). The real MusicCaps
# audio has to come from MusicCaps itself, which google/MusicCaps on HF
# ships as YouTube ids/timestamps only (no audio) — the same yt-dlp/bot-check
# problem we're already fighting for SONICS. To avoid a second YouTube
# download battle, use mahendra0203/musiccaps_processed_full, a community HF
# mirror with the actual audio already bundled (5,355 rows, ~3.3GB, verified
# schema: audio/caption/youtube_id/start_time/end_time/aspect_list — no fakes,
# every row is real MusicCaps by construction, no filtering needed).
# ---------------------------------------------------------------------------
log "Step 1 — Download real MusicCaps audio (mahendra0203/musiccaps_processed_full mirror)"
if [[ -d "$RAW_REAL_DIR" ]] && [[ "$(find "$RAW_REAL_DIR" -name '*.wav' | wc -l)" -gt 100 ]]; then
    echo "Already downloaded: $(find "$RAW_REAL_DIR" -name '*.wav' | wc -l) real WAVs at $RAW_REAL_DIR"
else
    mkdir -p "$RAW_REAL_DIR"
    python3 - <<'PYEOF'
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

out = Path("data/raw/fakemusiccaps_real_only")
out.mkdir(parents=True, exist_ok=True)

print("Loading mahendra0203/musiccaps_processed_full from HuggingFace "
      "(real MusicCaps audio, community mirror with bundled audio bytes)...")
ds = load_dataset("mahendra0203/musiccaps_processed_full", split="train", streaming=False)
# decode=False: get raw {"bytes": ..., "path": ...} instead of letting `datasets`
# auto-decode via its Audio feature, which on newer `datasets` versions requires
# torchcodec — torchcodec's compiled extension can mismatch the installed CUDA
# stack (e.g. "libnvrtc.so.13: cannot open shared object file") and hard-crashes
# rather than falling back gracefully. Decoding the raw bytes with soundfile
# ourselves sidesteps torchcodec entirely, regardless of what's installed.
ds = ds.cast_column("audio", Audio(decode=False))
print(f"Dataset loaded: {len(ds)} rows. Columns: {ds.column_names}")

n_real = 0
for i, row in enumerate(ds):
    if i % 1000 == 0:
        print(f"  {i}/{len(ds)}  (written so far: {n_real})", flush=True)
    track_id = str(row.get("youtube_id") or f"musiccaps_real_{i:06d}")
    out_path = out / f"{track_id}.wav"
    if out_path.exists():
        n_real += 1
        continue
    audio = row["audio"]
    if audio.get("bytes"):
        arr, sr = sf.read(io.BytesIO(audio["bytes"]))
    elif audio.get("path"):
        arr, sr = sf.read(audio["path"])
    else:
        continue
    sf.write(str(out_path), arr, sr)
    n_real += 1

print(f"Done: {n_real} real MusicCaps clips written to {out}")
if n_real == 0:
    raise SystemExit("No rows found — check the dataset schema/column names.")
PYEOF
fi

# ---------------------------------------------------------------------------
# Step 2: Score the real subset against the frozen flow (same protocol as the
# existing fake scores — never retrains the flow on external data).
# ---------------------------------------------------------------------------
log "Step 2 — Score real/MusicCaps subset against frozen SONICS flow"
if [[ -f "$REAL_OUT_DIR/external_scores.csv" ]]; then
    echo "Already scored: $REAL_OUT_DIR/external_scores.csv — skipping."
else
    python scripts/score_external_corpus.py \
        --audio-dir "$RAW_REAL_DIR" \
        --dataset-label real \
        --flow-path "$FLOW_PATH" \
        --sonics-cache data/emb_cache_encodec_full \
        --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
        --output-dir "$REAL_OUT_DIR" \
        --device cuda
fi

# ---------------------------------------------------------------------------
# Step 3: Combine real (MusicCaps) + fake (FakeMusicCaps) scores into the
# native-domain AUC/EER table, directly comparable to MusicDET Table 1/2.
# ---------------------------------------------------------------------------
log "Step 3 — Native-domain AUC/EER (MusicCaps-real vs FakeMusicCaps-fake)"
python3 - <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer

real_df = pd.read_csv("data/processed/external_scores/fakemusiccaps_real/external_scores.csv")
fake_df = pd.read_csv("data/processed/external_scores/fakemusiccaps/external_scores.csv")

real_scores = real_df["wf_anomaly_score"].dropna().to_numpy()
print(f"Real (MusicCaps) n={len(real_scores)}")

rows = []
for alg in sorted(fake_df["algorithm"].dropna().unique()):
    fake_scores = fake_df.loc[fake_df["algorithm"] == alg, "wf_anomaly_score"].dropna().to_numpy()
    if len(fake_scores) < 5:
        continue
    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    s = np.r_[real_scores, fake_scores]
    auc_res = bootstrap_auc(y, s)
    eer_res = bootstrap_eer(y, s)
    rows.append({
        "algorithm": alg,
        "n_real": len(real_scores),
        "n_fake": len(fake_scores),
        "auc": round(auc_res["auc"], 4),
        "auc_ci": f"[{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]",
        "eer_pct": round(eer_res["eer"] * 100, 2),
    })
    print(f"  {alg:25s}  AUC={auc_res['auc']:.4f} [{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]  "
          f"EER={eer_res['eer']*100:.2f}%  n_fake={len(fake_scores)}")

# Pooled across all generators (matches MusicDET's single reported EER/AUC number)
all_fake = fake_df["wf_anomaly_score"].dropna().to_numpy()
y = np.r_[np.zeros(len(real_scores)), np.ones(len(all_fake))]
s = np.r_[real_scores, all_fake]
auc_res = bootstrap_auc(y, s)
eer_res = bootstrap_eer(y, s)
print(f"\n  {'POOLED (all generators)':25s}  AUC={auc_res['auc']:.4f} [{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]  "
      f"EER={eer_res['eer']*100:.2f}%  n_fake={len(all_fake)}")
rows.append({
    "algorithm": "POOLED",
    "n_real": len(real_scores),
    "n_fake": len(all_fake),
    "auc": round(auc_res["auc"], 4),
    "auc_ci": f"[{auc_res['ci_lo']:.4f},{auc_res['ci_hi']:.4f}]",
    "eer_pct": round(eer_res["eer"] * 100, 2),
})

out = Path("data/processed/external_scores/fakemusiccaps_native_auc.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved -> {out}")
print("\nThis pooled EER is now directly comparable to MusicDET's own published "
      "FakeMusicCaps EER (4.51% zero-shot / 2.89% in-domain).")
PYEOF

log "Done."
