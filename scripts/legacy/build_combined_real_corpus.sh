#!/usr/bin/env bash
# =============================================================================
# build_combined_real_corpus.sh
#
# Downloads, canonicalizes, and indexes a combined real-music corpus for
# training a broader RealNVP flow (Plan Phase B2):
#
#   SONICS real        ~12,722 tracks  (already canonical)
#   FMA-medium         ~25,000 tracks  (Zenodo / HF)
#   MTG-Jamendo        ~55,000 tracks  (Zenodo)
#
# All tracks are canonicalized identically to the SONICS pipeline:
#   24 kHz mono → 64 kbps MP3 round-trip → LUFS −23 → trim ≤ 120 s
#
# Outputs a combined_manifest.csv that can be passed directly to
# run_balanced_ablation.py (--canonical-manifest) or to run_combined_flow.sh.
#
# Usage:
#   bash scripts/build_combined_real_corpus.sh
#   # (~10 h on g4dn.xlarge: download + canonicalize + manifest build)
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"

cd "$REPO"

COMBINED_DIR="data/processed/combined_real_corpus"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/combined_real_corpus_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$COMBINED_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

log "Building combined real corpus for manifold scaling"

# ---------------------------------------------------------------------------
# Part A: SONICS real (already canonical — just index it)
# ---------------------------------------------------------------------------
log "Part A — SONICS real (already canonical)"

python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

manifest = pd.read_csv("data/processed/canonical_sonics_full/canonical_manifest.csv", low_memory=False)
if "status" in manifest.columns:
    manifest = manifest[manifest["status"] == "ok"]
real = manifest[manifest["label"] == "real"].copy()
real["corpus"] = "sonics"
print(f"SONICS real: {len(real)} tracks")
real.to_csv("data/processed/combined_real_corpus/sonics_real_index.csv", index=False)
PYEOF

# ---------------------------------------------------------------------------
# Part B: FMA-medium (~25,000 tracks)
# Download from Zenodo: https://zenodo.org/records/1247102
# FMA-medium is ~22 GB of MP3s.
# ---------------------------------------------------------------------------
log "Part B — FMA-medium (Zenodo)"

FMA_DIR="data/external/fma_medium"
FMA_META_DIR="data/external/fma_metadata"

if [[ ! -d "$FMA_DIR" || -z "$(ls -A $FMA_DIR 2>/dev/null)" ]]; then
    mkdir -p "$FMA_DIR" "$FMA_META_DIR"
    FMA_AUDIO_URL="https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    FMA_META_URL="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

    # Use wget -c (resumable): safe to Ctrl+C and re-run without losing progress.
    echo "Downloading FMA-medium audio (~22 GB, resumable)..."
    wget -c -O data/external/fma_medium.zip "$FMA_AUDIO_URL"
    echo "Downloading FMA metadata (resumable)..."
    wget -c -O data/external/fma_metadata.zip "$FMA_META_URL"

    python3 - <<'PYEOF'
import sys
import zipfile
from pathlib import Path

# Validate before extracting — a partial/interrupted download would otherwise
# raise a hard-to-diagnose BadZipFile mid-extraction.
for name in ["fma_medium.zip", "fma_metadata.zip"]:
    p = Path("data/external") / name
    if not zipfile.is_zipfile(p):
        print(
            f"ERROR: {p} is not a valid zip (incomplete/corrupt download). "
            f"Delete it and re-run this script to redownload.",
            file=sys.stderr,
        )
        sys.exit(1)

print("Extracting FMA-medium audio...")
with zipfile.ZipFile("data/external/fma_medium.zip") as zf:
    zf.extractall("data/external/")
Path("data/external/fma_medium.zip").unlink(missing_ok=True)

print("Extracting FMA metadata...")
with zipfile.ZipFile("data/external/fma_metadata.zip") as zf:
    zf.extractall("data/external/")
Path("data/external/fma_metadata.zip").unlink(missing_ok=True)

print("FMA download complete.")
PYEOF
else
    echo "FMA-medium already present at $FMA_DIR"
fi

# ---------------------------------------------------------------------------
# Part C: MTG-Jamendo (~55,000 tracks)
# Subset: autotagging_moodtheme split (55k tracks) from Zenodo DOI 10.5281/zenodo.3826813
# Requires accepting MTG-Jamendo terms — download via their script.
# ---------------------------------------------------------------------------
log "Part C — MTG-Jamendo"

JAMENDO_DIR="data/external/mtg_jamendo"
if [[ ! -d "$JAMENDO_DIR" || -z "$(ls -A $JAMENDO_DIR 2>/dev/null)" ]]; then
    echo "Downloading MTG-Jamendo..."
    mkdir -p "$JAMENDO_DIR"
    # MTG-Jamendo provides a downloader script
    pip install -q zenodo_get 2>/dev/null || true
    python3 -m zenodo_get 3826813 -o "$JAMENDO_DIR" 2>&1 || {
        echo "  zenodo_get failed. Trying wget..."
        # Fallback to direct download of one release split
        wget -q -P "$JAMENDO_DIR" \
            "https://zenodo.org/records/3826813/files/raw_30s.tsv" \
            "https://zenodo.org/records/3826813/files/raw_30s_spec.tsv" || true
        echo "  NOTE: Audio files must be downloaded separately from MTG-Jamendo."
        echo "  See https://mtg.github.io/mtg-jamendo-dataset/"
        echo "  After download, re-run this script to continue."
        # For now, skip and continue with what we have
    }
else
    echo "MTG-Jamendo already present at $JAMENDO_DIR"
fi

# ---------------------------------------------------------------------------
# Part D: Canonicalize all additional real tracks
# Use build_canonical_corpus.py or a direct Python loop
# ---------------------------------------------------------------------------
log "Part D — Canonicalize FMA-medium + MTG-Jamendo"

python3 - <<'PYEOF'
"""
Canonicalize FMA-medium (and Jamendo if available) using the same pipeline
as build_canonical_corpus.py: 24kHz → 64kbps MP3 round-trip → LUFS-23 → WAV.
Saves progress in a checkpoint CSV; resume-safe.
"""
import sys, os, time, json, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "src")

try:
    from intrinsic_ai_music_detection.data.audio_preprocessing import preprocess_audio
except ImportError:
    print("ERROR: Cannot import audio_preprocessing. Is the src/ path correct?")
    sys.exit(1)

COMBINED_DIR = Path("data/processed/combined_real_corpus")
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = COMBINED_DIR / "canonicalize_checkpoint.jsonl"
DONE_IDS: set = set()
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        for line in f:
            try:
                DONE_IDS.add(json.loads(line)["track_id"])
            except Exception:
                pass

def collect_audio_files(dirs_and_labels: list[tuple]) -> list[dict]:
    """Collect all audio file paths from multiple source directories."""
    exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    tracks = []
    for src_dir, corpus_label in dirs_and_labels:
        src = Path(src_dir)
        if not src.exists():
            print(f"  Source dir not found: {src} — skipping")
            continue
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in exts:
                tracks.append({
                    "track_id": f"{corpus_label}_{p.stem}",
                    "src_path": str(p),
                    "corpus": corpus_label,
                })
    return tracks

SOURCE_DIRS = [
    ("data/external/fma_medium", "fma_medium"),
    ("data/external/mtg_jamendo", "jamendo"),
]

tracks = collect_audio_files(SOURCE_DIRS)
todo = [t for t in tracks if t["track_id"] not in DONE_IDS]
print(f"  Total additional real tracks: {len(tracks)}, todo: {len(todo)}")

OUT_AUDIO = COMBINED_DIR / "audio"
OUT_AUDIO.mkdir(exist_ok=True)

def _process_one(track: dict) -> dict:
    """Canonicalize one track. Returns row dict for manifest."""
    try:
        out_path = OUT_AUDIO / f"{track['track_id']}.wav"
        if out_path.exists():
            return {"track_id": track["track_id"], "canonical_path": str(out_path),
                    "label": "real", "corpus": track["corpus"], "status": "ok"}
        audio, sr, meta = preprocess_audio(
            Path(track["src_path"]),
            target_sr=24_000,
            mode="canonical",
            max_duration=120.0,
            mp3_bitrate_kbps=64,
            target_lufs=-23.0,
        )
        import soundfile as sf
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
        return {"track_id": track["track_id"], "canonical_path": str(out_path),
                "label": "real", "corpus": track["corpus"], "status": "ok", **meta}
    except Exception as exc:
        return {"track_id": track["track_id"], "src_path": track["src_path"],
                "label": "real", "corpus": track["corpus"],
                "status": "error", "error": str(exc)[:200]}

N_WORKERS = 4
results = []
t0 = time.time()

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(_process_one, t): t for t in todo}
    for i, future in enumerate(as_completed(futures), 1):
        try:
            row = future.result(timeout=120)
        except Exception as exc:
            row = {"status": "error", "error": str(exc)[:200]}
        results.append(row)
        with open(CHECKPOINT, "a") as f:
            f.write(json.dumps(row) + "\n")
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            print(f"  [{i}/{len(todo)}] ok={sum(1 for r in results if r.get('status')=='ok')} "
                  f"  {rate:.1f} track/s", flush=True)

# Build combined manifest
print("\nBuilding combined manifest...")
sonics = pd.read_csv(COMBINED_DIR / "sonics_real_index.csv", low_memory=False)

new_rows = pd.DataFrame([r for r in results if r.get("status") == "ok"])
if not new_rows.empty:
    new_rows = new_rows[["track_id", "canonical_path", "label", "corpus"]].copy()

combined = pd.concat([sonics[["track_id", "canonical_path", "label", "corpus"]],
                      new_rows], ignore_index=True)
combined.to_csv(COMBINED_DIR / "combined_manifest.csv", index=False)
print(f"Combined manifest: {len(combined)} real tracks")
print(combined["corpus"].value_counts().to_string())
PYEOF

log "Combined real corpus build complete"
echo "Manifest: $COMBINED_DIR/combined_manifest.csv"

# Sync to S3
aws s3 sync "$COMBINED_DIR/" "$S3/processed/combined_real_corpus/" \
    --exclude "audio/*" --no-progress || \
    echo "S3 sync of manifest failed (non-fatal)"
