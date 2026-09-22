#!/usr/bin/env bash
# =============================================================================
# run_robustness_battery.sh
#
# Extends run_bitrate_sweep.sh (MP3 32/64/128kbps) to the FULL manipulation
# battery from MusicDET Table 6: pitch shift, time stretch, EQ, reverb, white
# noise, AAC 64kbps, Opus 64kbps. Tests the SAME already-trained frozen flow
# used for the headline result — matches MusicDET's own protocol of measuring
# a fixed detector's degradation, not retraining per manipulation.
#
# Usage:
#   bash scripts/run_robustness_battery.sh
# =============================================================================
set -euo pipefail

# Derive the repo root from this script's own location rather than assuming it
# sits at $HOME/intrinsic-ai-music-detection, so a clone under any path works.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

FLOW_PATH="data/processed/full_sonics_all/sonics_real_flow_encodec.pt"
SUBSET_MANIFEST="data/processed/bitrate_sweep_subset.csv"

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -f "$SUBSET_MANIFEST" ]]; then
    echo "$SUBSET_MANIFEST not found — building it now (same subset run_bitrate_sweep.sh uses)."
    python3 - <<'PYEOF'
import pandas as pd

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
print(f"Subset: {len(subset)} tracks")
PYEOF
fi

if [[ ! -f "$FLOW_PATH" ]]; then
    echo "ERROR: frozen flow not found at $FLOW_PATH"
    exit 1
fi

log "Running robustness battery (pitch/stretch/EQ/reverb/noise/AAC/Opus)"
python scripts/run_robustness_battery.py \
    --flow-path "$FLOW_PATH" \
    --subset-manifest "$SUBSET_MANIFEST" \
    --output-dir data/processed/robustness_battery \
    --device cuda

log "Done. See data/processed/robustness_battery/robustness_auc_eer.csv"
