#!/usr/bin/env bash
# =============================================================================
# run_musicdet_comparison_corrected.sh  (v2 — verified against actual MusicDET source)
#
# Head-to-head: our EnCodec+RealNVP window-flow vs. MusicDET's spec-nf, BOTH
# trained real-only (zero-shot) on the IDENTICAL SONICS real/fake split.
#
# KEY CORRECTIONS vs. the handover AND vs. the first version of this script:
# ============================================================================
# 1. The handover's claim that MusicDET trains on FMA-full is WRONG — MusicDET
#    trains spec-nf on the benchmark's OWN real split (SONICS real /
#    FakeMusicCaps real), same principle as our method. This makes it a much
#    more direct baseline than the handover assumed.
# 2. MusicDET's test.py has NO --manipulation/--mp3_bitrate flag (verified
#    against github.com/Chaolei98/MusicDET test.py — that flag never existed).
#    MP3-robustness must be tested by feeding test.py separate MP3-round-tripped
#    eval CSVs, not a CLI flag.
# 3. MusicDET's default --train_dataframe/--valid_dataframe/--sonics_eval_csvs
#    point at label CSVs that ship via Google Drive, NOT in the git repo. But
#    these are plain --train_dataframe / --valid_dataframe / --sonics_eval_csvs
#    CLI args accepting *any* CSV with (filepath, target) columns (target:
#    0=real, 1=fake) — verified against dataset.py's load_sonics_dataframe().
#    So we build our OWN CSVs pointing directly at our canonical SONICS WAVs.
#    normalize_sonics_filepath() only remaps paths containing the literal
#    substring "real_songs_wav/" or "fake_songs_wav/" — our canonical paths
#    (.../canonical_sonics_full/real/...) don't match that, so absolute paths
#    pass through unchanged. No copying/symlinking of audio needed.
# 4. MusicDET's own dataloader resamples to 16kHz, crops/pads to 64600 samples
#    (~4.04s), and RMS-normalizes on the fly — so we can point it directly at
#    our existing 24kHz canonical WAVs with zero preprocessing on our side.
#
# Usage:
#   bash scripts/run_musicdet_comparison_corrected.sh
#   (Run in a screen/tmux session on EC2 — training + eval takes a few hours)
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

MUSICDET_DIR="$HOME/MusicDET"
CANON_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
OUT_DIR="data/processed/musicdet_comparison_corrected"
MDET_OUT="$MUSICDET_DIR/output_sonics_matched"
LOG="logs/musicdet_comparison_corrected_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR" logs
exec > >(tee -a "$LOG") 2>&1

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

log "Corrected MusicDET comparison v2 (see script header)"

# ---------------------------------------------------------------------------
# Step 1: Clone MusicDET + venv (spec-nf needs no SSL backbones, no pretrained/ downloads)
# ---------------------------------------------------------------------------
log "Step 1 — MusicDET setup"
if [[ ! -d "$MUSICDET_DIR" ]]; then
    git clone https://github.com/Chaolei98/MusicDET "$MUSICDET_DIR"
fi
cd "$MUSICDET_DIR"
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    [[ -f requirements.txt ]] && .venv/bin/pip install -q -r requirements.txt || \
        .venv/bin/pip install -q torch torchaudio librosa numpy scipy scikit-learn pandas tqdm
fi
cd "$REPO"

# ---------------------------------------------------------------------------
# Step 2: Build (filepath, target) CSVs pointing directly at our canonical WAVs.
#   - train:  80% of SONICS real tracks (real-only, matches MusicDET's own
#             zero-shot training paradigm — same principle as our method)
#   - valid:  10% of SONICS real (held out) + a small mixed-fake sample,
#             used only for MusicDET's internal checkpoint selection
#   - test:   remaining 10% of SONICS real (held out from train/valid) +
#             ALL fake tracks, split into one CSV per generator for
#             apples-to-apples comparison against our per-generator LOGO AUCs
#   - test_mp3: same test tracks, MP3 64kbps round-tripped (our own
#             preprocess_audio, matching our canonicalization exactly),
#             capped per generator to keep runtime bounded
# ---------------------------------------------------------------------------
log "Step 2 — Build train/valid/test CSVs from our canonical manifest"

python3 - <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.data.audio_preprocessing import preprocess_audio

REPO = Path.cwd()
OUT_DIR = REPO / "data/processed/musicdet_comparison_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = OUT_DIR / "csvs"
CSV_DIR.mkdir(exist_ok=True)
MP3_DIR = OUT_DIR / "mp3_64kbps_wav"
MP3_DIR.mkdir(exist_ok=True)

manifest = pd.read_csv("data/processed/canonical_sonics_full/canonical_manifest.csv", low_memory=False)
if "status" in manifest.columns:
    manifest = manifest[manifest["status"] == "ok"]

def abspath(p):
    return str((REPO / p).resolve())

real = manifest[manifest["label"] == "real"].copy()
real["abspath"] = real["canonical_path"].map(abspath)
fake = manifest[manifest["label"] == "fake"].copy()
fake["abspath"] = fake["canonical_path"].map(abspath)

rng = np.random.default_rng(42)
real_ids = real["track_id"].tolist()
rng.shuffle(real_ids)
n = len(real_ids)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)
train_ids = set(real_ids[:n_train])
valid_ids = set(real_ids[n_train:n_train + n_valid])
test_ids = set(real_ids[n_train + n_valid:])
print(f"Real split: train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)}")

def write_csv(df_rows, path):
    df_out = pd.DataFrame(df_rows, columns=["filepath", "target"])
    df_out.to_csv(path, index=False)
    print(f"  {path.name}: {len(df_out)} rows ({(df_out['target']==0).sum()} real, {(df_out['target']==1).sum()} fake)")

# --- train: real-only (target=0) ---
train_real = real[real["track_id"].isin(train_ids)]
write_csv([{"filepath": p, "target": 0} for p in train_real["abspath"]],
          CSV_DIR / "train_real_only.csv")

# --- valid: held-out real + a small mixed-fake sample (for checkpoint selection) ---
valid_real = real[real["track_id"].isin(valid_ids)]
valid_fake_sample = fake.sample(n=min(500, len(fake)), random_state=42)
valid_rows = [{"filepath": p, "target": 0} for p in valid_real["abspath"]] + \
             [{"filepath": p, "target": 1} for p in valid_fake_sample["abspath"]]
write_csv(valid_rows, CSV_DIR / "valid_mixed.csv")

# --- test: held-out real (target=0, shared across all per-generator CSVs) + per-generator fakes ---
test_real = real[real["track_id"].isin(test_ids)]
test_real_rows = [{"filepath": p, "target": 0} for p in test_real["abspath"]]

algorithms = sorted(fake["algorithm"].dropna().unique())
print(f"Generators found: {algorithms}")

eval_csv_paths = []
for alg in algorithms:
    alg_fake = fake[fake["algorithm"] == alg]
    rows = test_real_rows + [{"filepath": p, "target": 1} for p in alg_fake["abspath"]]
    safe_name = alg.replace("/", "_")
    csv_path = CSV_DIR / f"eval_{safe_name}.csv"
    write_csv(rows, csv_path)
    eval_csv_paths.append(str(csv_path))

with open(OUT_DIR / "eval_csv_paths_clean.txt", "w") as f:
    f.write("\n".join(eval_csv_paths))

# --- test_mp3: MP3 64kbps round-trip of held-out real + capped fake sample per generator ---
print("\nBuilding MP3 64kbps round-tripped eval set (capped at 2000/generator for runtime)...")

def mp3_roundtrip_path(src_path: str, dst_dir: Path) -> str:
    dst = dst_dir / (Path(src_path).stem + "_mp3.wav")
    if dst.exists():
        return str(dst)
    try:
        # target_lufs left at its canonical default (MusicDET's own dataloader
        # RMS-normalizes on load anyway, so this only needs to match our MP3
        # round-trip step, not a specific loudness target).
        audio, sr, _meta = preprocess_audio(
            Path(src_path), target_sr=16_000, mode="canonical",
            max_duration=10.0, mp3_bitrate_kbps=64,
        )
        import soundfile as sf
        sf.write(str(dst), audio, sr, subtype="PCM_16")
        return str(dst)
    except Exception as exc:
        print(f"  MP3 roundtrip failed for {src_path}: {exc}")
        return ""

test_real_mp3 = [mp3_roundtrip_path(p, MP3_DIR) for p in test_real["abspath"]]
test_real_mp3_rows = [{"filepath": p, "target": 0} for p in test_real_mp3 if p]
print(f"  MP3 real test tracks: {len(test_real_mp3_rows)}/{len(test_real)}")

mp3_eval_csv_paths = []
for alg in algorithms:
    alg_fake = fake[fake["algorithm"] == alg].sample(
        n=min(2000, len(fake[fake["algorithm"] == alg])), random_state=42
    )
    fake_mp3 = [mp3_roundtrip_path(p, MP3_DIR) for p in alg_fake["abspath"]]
    rows = test_real_mp3_rows + [{"filepath": p, "target": 1} for p in fake_mp3 if p]
    safe_name = alg.replace("/", "_")
    csv_path = CSV_DIR / f"eval_{safe_name}_mp3.csv"
    write_csv(rows, csv_path)
    mp3_eval_csv_paths.append(str(csv_path))

with open(OUT_DIR / "eval_csv_paths_mp3.txt", "w") as f:
    f.write("\n".join(mp3_eval_csv_paths))

print("\nCSV build complete.")
PYEOF

# ---------------------------------------------------------------------------
# Step 3: Train MusicDET spec-nf real-only on our SONICS real train split
# ---------------------------------------------------------------------------
log "Step 3 — Train MusicDET spec-nf (real-only, 10 epochs, matched split)"
cd "$MUSICDET_DIR"

if [[ -f "$MDET_OUT/anti-spoofing_feat_model.pt" ]]; then
    echo "  Trained model already exists at $MDET_OUT — skipping."
else
    .venv/bin/python train.py \
        --task sonics \
        --model spec-nf \
        --batch_size 64 \
        --epochs 10 \
        --num_workers 2 \
        --only_real \
        --train_dataframe "$REPO/$OUT_DIR/csvs/train_real_only.csv" \
        --valid_dataframe "$REPO/$OUT_DIR/csvs/valid_mixed.csv" \
        --out_fold "$MDET_OUT" \
        2>&1 | tee "$REPO/logs/musicdet_train_matched.log"
fi
cd "$REPO"

# ---------------------------------------------------------------------------
# Step 4: Test — clean, then MP3 64kbps (separate eval CSVs, not a CLI flag)
# ---------------------------------------------------------------------------
log "Step 4a — MusicDET test: clean audio, per-generator"
cd "$MUSICDET_DIR"
mapfile -t CLEAN_CSVS < "$REPO/$OUT_DIR/eval_csv_paths_clean.txt"
.venv/bin/python test.py \
    --task sonics \
    --model_path "$MDET_OUT" \
    --sonics_eval_csvs "${CLEAN_CSVS[@]}" \
    2>&1 | tee "$REPO/logs/musicdet_test_clean.log"

log "Step 4b — MusicDET test: MP3 64kbps round-tripped audio, per-generator"
mapfile -t MP3_CSVS < "$REPO/$OUT_DIR/eval_csv_paths_mp3.txt"
.venv/bin/python test.py \
    --task sonics \
    --model_path "$MDET_OUT" \
    --sonics_eval_csvs "${MP3_CSVS[@]}" \
    2>&1 | tee "$REPO/logs/musicdet_test_mp3.log"
cd "$REPO"

# ---------------------------------------------------------------------------
# Step 5: Compare Window-Flow vs MusicDET (clean + MP3) with bootstrap CI
# ---------------------------------------------------------------------------
log "Step 5 — Comparison report with bootstrap CIs"

python3 - <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer

WF_CSV = Path("data/processed/full_sonics_all/window_flow_eval.csv")
OUT_DIR = Path("data/processed/musicdet_comparison_corrected")
RESULT_DIR = Path.home() / "MusicDET/output_sonics_matched/result"

wf = pd.read_csv(WF_CSV, low_memory=False)
mean_col = next((c for c in wf.columns if "wf_mean" in c), None)
wf_real = wf[wf["label"] == "real"][mean_col].dropna().values

def load_mdet_scores(txt_path: Path):
    """MusicDET result .txt: 'basename score label_str' per line, space-separated.
    score = -nll (higher = MORE real, per test.py's compute_scores)."""
    if not txt_path.exists():
        return None
    rows = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append({"id": parts[0], "score": float(parts[1]), "label": parts[2]})
    return pd.DataFrame(rows)

rows = []
for alg in sorted(wf[wf["label"] == "fake"]["algorithm"].dropna().unique()):
    wf_fake = wf[(wf["label"] == "fake") & (wf["algorithm"] == alg)][mean_col].dropna().values
    if len(wf_fake) < 5:
        continue
    y_wf = np.r_[np.zeros(len(wf_real)), np.ones(len(wf_fake))]
    s_wf = np.r_[-wf_real, -wf_fake]  # negate: higher anomaly score = more fake
    wf_boot = bootstrap_auc(y_wf, s_wf)
    wf_eer = bootstrap_eer(y_wf, s_wf)

    row = {
        "algorithm": alg,
        "wf_auc": round(wf_boot["auc"], 4),
        "wf_auc_ci": f"[{wf_boot['ci_lo']:.4f},{wf_boot['ci_hi']:.4f}]",
        "wf_eer_pct": round(wf_eer["eer"] * 100, 2),
    }

    safe_name = alg.replace("/", "_")
    for tag, suffix in [("clean", ""), ("mp3_64k", "_mp3")]:
        mdet = load_mdet_scores(RESULT_DIR / f"eval_{safe_name}{suffix}.txt")
        if mdet is None or mdet["label"].nunique() < 2:
            continue
        y_m = (mdet["label"] == "fake").astype(int).values
        s_m = -mdet["score"].values  # negate: their score is higher=more real too
        m_boot = bootstrap_auc(y_m, s_m)
        m_eer = bootstrap_eer(y_m, s_m)
        row[f"mdet_{tag}_auc"] = round(m_boot["auc"], 4)
        row[f"mdet_{tag}_auc_ci"] = f"[{m_boot['ci_lo']:.4f},{m_boot['ci_hi']:.4f}]"
        row[f"mdet_{tag}_eer_pct"] = round(m_eer["eer"] * 100, 2)

    rows.append(row)

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "wf_vs_musicdet_auc_ci.csv", index=False)
print(f"\nComparison table:\n{df_out.to_string(index=False)}")
print(f"\nSaved to {OUT_DIR / 'wf_vs_musicdet_auc_ci.csv'}")
PYEOF

log "Done. Results in $OUT_DIR/wf_vs_musicdet_auc_ci.csv"
