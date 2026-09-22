#!/usr/bin/env bash
# =============================================================================
# download_external_data.sh
#
# Downloads FMA-small and FakeMusicCaps for FPR testing and cross-dataset
# generalization evaluation.
#
# FMA-small:  8000 tracks, 30s clips, diverse genres (3.4 GB compressed)
# FakeMusicCaps: 27605 tracks, 10s clips, 5 generators (MusicGen, MusicLDM,
#                AudioLDM2, Stable Audio Open, Mustango)
#
# Usage:
#   bash scripts/download_external_data.sh [--fma] [--fakemusiccaps] [--both]
#
# Default: downloads both.
# =============================================================================
set -euo pipefail

# Derive the repo root from this script's own location rather than assuming it
# sits at $HOME/intrinsic-ai-music-detection, so a clone under any path works.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DO_FMA=0
DO_FMC=0

for arg in "$@"; do
    case "$arg" in
        --fma)          DO_FMA=1 ;;
        --fakemusiccaps) DO_FMC=1 ;;
        --both)         DO_FMA=1; DO_FMC=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Default: both
if [[ $DO_FMA -eq 0 && $DO_FMC -eq 0 ]]; then
    DO_FMA=1; DO_FMC=1
fi

mkdir -p data/external logs

# ---------------------------------------------------------------------------
# FMA-small
# ---------------------------------------------------------------------------
if [[ $DO_FMA -eq 1 ]]; then
    echo ""
    echo "=== Downloading FMA-small (7.2 GiB, 8000 tracks, 30s MP3 clips) ==="
    echo "    Source: https://github.com/mdeff/fma"

    FMA_DIR="data/external/fma_small"
    FMA_ZIP="data/external/fma_small.zip"
    FMA_META_ZIP="data/external/fma_metadata.zip"

    if [[ -d "$FMA_DIR" ]]; then
        echo "  FMA-small already extracted at $FMA_DIR — skipping download."
    else
        # Download
        if [[ ! -f "$FMA_ZIP" ]]; then
            echo "  Downloading fma_small.zip (~7.2 GB)..."
            wget -q --show-progress -O "$FMA_ZIP" \
                "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
        fi
        echo "  Extracting via Python zipfile (avoids unzip OOM on large archives)..."
        python3 - <<'PYEOF'
import zipfile, sys
from pathlib import Path
zp = Path("data/external/fma_small.zip")
print(f"  Archive: {zp.stat().st_size / 1e9:.1f} GB, extracting...")
with zipfile.ZipFile(zp) as zf:
    members = zf.namelist()
    print(f"  Files in archive: {len(members)}")
    for i, m in enumerate(members):
        zf.extract(m, "data/external/")
        if i % 1000 == 0:
            print(f"  {i}/{len(members)} extracted...", flush=True)
print(f"  Extraction complete ({len(members)} files).")
PYEOF
        echo "  FMA-small extracted → $FMA_DIR"
        rm -f "$FMA_ZIP"  # delete zip to reclaim ~7 GB
        echo "  Deleted $FMA_ZIP (zip) to reclaim space."
    fi

    # Download metadata (for track genres/info)
    if [[ ! -f "data/external/fma_metadata/tracks.csv" ]]; then
        if [[ ! -f "$FMA_META_ZIP" ]]; then
            echo "  Downloading fma_metadata.zip (~342 MB)..."
            wget -q --show-progress -O "$FMA_META_ZIP" \
                "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
        fi
        echo "  Unzipping metadata..."
        unzip -q "$FMA_META_ZIP" -d data/external/
        rm -f "$FMA_META_ZIP"
        echo "  FMA metadata → data/external/fma_metadata/"
    fi

    # Sample 1000 tracks (stratified by genre) for FPR test
    echo "  Sampling 1000 tracks for FPR test..."
    python3 - <<'PYEOF'
import os, random, shutil
from pathlib import Path

src = Path("data/external/fma_small")
dst = Path("data/external/fma_sample_1000")
dst.mkdir(parents=True, exist_ok=True)

all_mp3s = sorted(src.rglob("*.mp3"))
print(f"  Total FMA-small tracks: {len(all_mp3s)}")

random.seed(42)
sample = random.sample(all_mp3s, min(1000, len(all_mp3s)))

for p in sample:
    shutil.copy2(p, dst / p.name)

print(f"  Sampled {len(sample)} tracks → {dst}")
PYEOF

    echo "  FMA-small sample ready at data/external/fma_sample_1000/"
fi

# ---------------------------------------------------------------------------
# FakeMusicCaps
# ---------------------------------------------------------------------------
if [[ $DO_FMC -eq 1 ]]; then
    echo ""
    echo "=== Downloading FakeMusicCaps from Zenodo (12.9 GB) ==="
    echo "    Comanducci et al. 2025, Journal of Imaging"
    echo "    DOI: 10.5281/zenodo.15063698"
    echo "    5 generators × 5507 prompts = 27605 WAV clips"

    FMC_DIR="data/external/fakemusiccaps"
    FMC_ZIP="data/external/FakeMusicCaps.zip"

    if [[ -d "$FMC_DIR" ]] && [[ "$(find "$FMC_DIR" -name "*.wav" | wc -l)" -gt 1000 ]]; then
        echo "  FakeMusicCaps already extracted ($(find "$FMC_DIR" -name "*.wav" | wc -l) wavs) — skipping."
    else
        # Download
        if [[ ! -f "$FMC_ZIP" ]]; then
            echo "  Downloading FakeMusicCaps.zip from Zenodo..."
            wget -q --show-progress \
                -O "$FMC_ZIP" \
                "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1"
        fi

        echo "  Extracting via Python zipfile (avoids unzip OOM on large archives)..."
        python3 - <<'PYEOF'
import zipfile, sys
from pathlib import Path
zp = Path("data/external/FakeMusicCaps.zip")
print(f"  Archive: {zp.stat().st_size / 1e9:.1f} GB")
out = Path("data/external/fakemusiccaps")
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zp) as zf:
    # The zip root contains generator folders directly: audioldm2/track.wav
    # (no top-level FakeMusicCaps/ prefix). Preserve the generator subdirs.
    members = [m for m in zf.namelist()
               if not m.startswith("__MACOSX") and "/._" not in m and m.endswith(".wav")]
    print(f"  WAV files to extract: {len(members)}")
    # Sanity-check: confirm generator dirs are present
    generators = {Path(m).parts[0] for m in members if len(Path(m).parts) >= 2}
    print(f"  Generators detected: {sorted(generators)}")
    for i, m in enumerate(members):
        # Preserve full relative path: audioldm2/track.wav -> out/audioldm2/track.wav
        dst = out / m
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src_f, open(dst, "wb") as dst_f:
            dst_f.write(src_f.read())
        if i % 2000 == 0:
            print(f"  {i}/{len(members)} extracted...", flush=True)
print(f"  Done — {len(members)} WAV files → {out}")
PYEOF
        rm -f "$FMC_ZIP"
        echo "  Deleted zip to reclaim 12.9 GB."
    fi

    # Show what was downloaded
    if [[ -d "$FMC_DIR" ]]; then
        echo ""
        echo "  FakeMusicCaps structure:"
        ls "$FMC_DIR" 2>/dev/null | head -20 || true
        echo "  File count: $(find "$FMC_DIR" -name "*.wav" -o -name "*.mp3" | wc -l)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== External data ready ==="
echo ""
echo "FMA-small sample (FPR test):"
echo "  data/external/fma_sample_1000/   → 1000 real tracks, diverse genres"
echo ""
echo "FakeMusicCaps (generalization test):"
echo "  data/external/fakemusiccaps/     → 5 unseen generators"
echo ""
echo "Next step — score both against the SONICS flow:"
echo ""
echo "  # FPR test (real music)"
echo "  poetry run python scripts/score_external_corpus.py \\"
echo "    --audio-dir data/external/fma_sample_1000/ \\"
echo "    --dataset-label real \\"
echo "    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\"
echo "    --sonics-cache data/emb_cache_encodec_full \\"
echo "    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\"
echo "    --output-dir data/processed/external_scores/fma_fpr \\"
echo "    --device cuda"
echo ""
echo "  # Generalization test (unseen generators)"
echo "  poetry run python scripts/score_external_corpus.py \\"
echo "    --audio-dir data/external/fakemusiccaps/ \\"
echo "    --dataset-label fake \\"
echo "    --algorithm-from-dirname \\"
echo "    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\"
echo "    --sonics-cache data/emb_cache_encodec_full \\"
echo "    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\"
echo "    --output-dir data/processed/external_scores/fakemusiccaps \\"
echo "    --device cuda"
