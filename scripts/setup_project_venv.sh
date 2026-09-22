#!/usr/bin/env bash
# =============================================================================
# setup_project_venv.sh
#
# Creates a TRUE, isolated virtual environment for THIS project at
# ~/intrinsic-ai-music-detection/.venv — separate from ~/MusicDET/.venv.
#
# Root cause this fixes: this box has only ever had ONE shared Python
# environment (MusicDET's), used by both projects. `scripts/run_xcodec_udio30.sh`'s
# `pip install xcodec2` silently upgraded numpy (breaking scipy) and pulled in
# a newer nvidia-cusparse-cu12 that no longer matches the installed torch build
# (breaking `import torch` everywhere). This script:
#   1. Detaches from whatever venv is currently active in the shell.
#   2. Creates a dedicated .venv inside this repo (poetry.toml already sets
#      virtualenvs.in-project=true, so `poetry install` will do this from now on).
#   3. Installs project deps via poetry, then a matched CUDA torch/torchvision/
#      torchaudio triplet + numpy<2 pin (resolved together by pip, not guessed
#      package-by-package).
#   4. Verifies torch+CUDA, scipy, librosa all import cleanly before declaring success.
#
# After this, MusicDET's venv is left untouched (still usable for MusicDET-only
# work), and this project never touches it again.
#
# Usage:
#   bash scripts/setup_project_venv.sh
#   source .venv/bin/activate     # for all subsequent work in this project
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

echo "=== Step 1: detach from any currently-active venv ==="
unset VIRTUAL_ENV || true
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "/MusicDET/.venv/" | paste -sd: -)"
echo "PATH cleaned of MusicDET/.venv references."

echo ""
echo "=== Step 2: pick a Python interpreter ==="
PYBIN=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYBIN="$cand"
        break
    fi
done
if [[ -z "$PYBIN" ]]; then
    echo "ERROR: no python3.x interpreter found on this box." >&2
    exit 1
fi
echo "Using: $PYBIN ($($PYBIN --version))"

echo ""
echo "=== Step 3: create isolated venv at $REPO/.venv ==="
if [[ -d ".venv" ]]; then
    echo ".venv already exists — checking if it's healthy..."
    if .venv/bin/python -c "import torch, scipy.stats, numpy" 2>/dev/null; then
        echo ".venv looks healthy already. Skipping recreation. (Delete .venv manually to force a clean rebuild.)"
        SKIP_CREATE=1
    else
        echo ".venv exists but is broken — removing and recreating."
        rm -rf .venv
        SKIP_CREATE=0
    fi
else
    SKIP_CREATE=0
fi

if [[ "${SKIP_CREATE:-0}" == "0" ]]; then
    "$PYBIN" -m venv .venv
    echo "Created $REPO/.venv"
fi

source .venv/bin/activate
echo "Active interpreter: $(which python)  ($(python --version))"

echo ""
echo "=== Step 4: install project dependencies ==="
python -m pip install -q --upgrade pip
if command -v poetry >/dev/null 2>&1; then
    poetry env use "$(pwd)/.venv/bin/python"
    poetry install --no-root
else
    echo "poetry not found on PATH — installing from pyproject.toml deps via pip is not exact; installing poetry first."
    pip install -q poetry
    poetry env use "$(pwd)/.venv/bin/python"
    poetry install --no-root
fi

echo ""
echo "=== Step 5: install a matched CUDA torch/torchvision/torchaudio triplet ==="
NVIDIA_SMI_OUT="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)"
echo "Detected driver CUDA version: ${NVIDIA_SMI_OUT:-unknown}"
# cu121 wheels are broadly compatible with driver CUDA 12.1-12.6; adjust the
# index URL below if nvidia-smi reports something very different (e.g. 11.x).
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
pip install --force-reinstall "numpy<2.0,>=1.23.5"
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$TORCH_INDEX"

echo ""
echo "=== Step 6: verify ==="
python - <<'PYEOF'
import sys
ok = True
try:
    import numpy
    print(f"numpy {numpy.__version__} OK")
except Exception as e:
    print(f"numpy FAILED: {e}"); ok = False
try:
    import scipy.stats
    print(f"scipy OK")
except Exception as e:
    print(f"scipy FAILED: {e}"); ok = False
try:
    import torch
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available — check nvidia-smi / driver / TORCH_INDEX in this script.")
except Exception as e:
    print(f"torch FAILED: {e}"); ok = False
try:
    import librosa
    print(f"librosa {librosa.__version__} OK")
except Exception as e:
    print(f"librosa FAILED: {e}"); ok = False
sys.exit(0 if ok else 1)
PYEOF

echo ""
echo "=== Done ==="
echo "This project now has its OWN isolated venv at $REPO/.venv, separate from ~/MusicDET/.venv."
echo "Activate it in every new shell with:"
echo "  cd $REPO && source .venv/bin/activate"
echo ""
echo "IMPORTANT: never 'pip install' experiment-only packages (e.g. xcodec2) into this"
echo "venv directly — use scripts/run_xcodec_udio30.sh, which now installs xcodec2 into"
echo "its own disposable venv instead."
