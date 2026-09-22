#!/usr/bin/env bash
# =============================================================================
# run_core_comparison_suite.sh
#
# TIER 1 — everything needed to be FULLY, defensibly comparable with MusicDET's
# published results, in priority order. Run this to completion before spending
# more GPU time on exploratory/improvement work (genre-balancing, C3 X-Codec,
# corpus scaling — those are TIER 2, decide on them AFTER seeing these results).
#
# Order matters:
#   1. A3 reconstruction control  — foundational: proves the detector reacts to
#      codec artifacts specifically, not genre/content. Everything below rests
#      on this being solid, so it goes first.
#   2. FakeMusicCaps cross-dataset zero-shot — our SONICS-trained flow, scored
#      zero-shot on MusicCaps-real + FakeMusicCaps-fake (never trained on either).
#   3. FakeMusicCaps in-domain exact match — fresh flow trained ONLY on
#      MusicCaps-real (~5.5k clips), scored on FakeMusicCaps-fake. Exact
#      apples-to-apples with MusicDET Table 1 (EER 4.51%).
#   4. Robustness battery — pitch/stretch/EQ/reverb/noise/AAC/Opus @ 64kbps,
#      exact apples-to-apples with MusicDET Table 6.
#
# Each step is independently checkpointed/resumable (see each script's header),
# and each is run with `;` rather than `&&` so one failure doesn't block the
# rest of the overnight run — check logs/core_suite_*.log afterward for any
# step that failed and re-run just that step.
#
# Usage (inside tmux/screen):
#   bash scripts/run_core_comparison_suite.sh
# =============================================================================
set -uo pipefail   # NOTE: no -e — see rationale above, each step logs its own failure

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"
mkdir -p logs

MASTER_LOG="logs/core_suite_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

FLOW_PATH="data/processed/full_sonics_all/sonics_real_flow_encodec.pt"

step() { echo; echo "############################################################"; echo "# $(date '+%F %T')  STEP: $*"; echo "############################################################"; echo; }
step_done() { echo; echo ">>> STEP DONE (exit=$1): $2"; echo; }

# ---------------------------------------------------------------------------
step "1/4 — A3: reconstruction control (Afchar/MusicDET Appendix A.2 protocol)"
# ---------------------------------------------------------------------------
python scripts/build_reconstruction_control.py \
    --real-dir data/processed/canonical_sonics_full/real \
    --flow-path "$FLOW_PATH" \
    --output-dir data/processed/reconstruction_control \
    --n-tracks 500 \
    --device cuda
step_done $? "A3 reconstruction control"

# ---------------------------------------------------------------------------
step "2/4 — FakeMusicCaps cross-dataset zero-shot (SONICS-trained flow -> MusicCaps-real + FakeMusicCaps-fake)"
# ---------------------------------------------------------------------------
bash scripts/build_fakemusiccaps_native_comparison.sh
step_done $? "FakeMusicCaps cross-dataset zero-shot"

# ---------------------------------------------------------------------------
step "3/4 — FakeMusicCaps in-domain exact Table 1 match (fresh flow trained on MusicCaps-real only)"
# ---------------------------------------------------------------------------
bash scripts/run_fakemusiccaps_native_flow.sh
step_done $? "FakeMusicCaps in-domain exact match"

# ---------------------------------------------------------------------------
step "4/4 — Robustness battery (pitch/stretch/EQ/reverb/noise/AAC/Opus, MusicDET Table 6 parity)"
# ---------------------------------------------------------------------------
bash scripts/run_robustness_battery.sh
step_done $? "Robustness battery"

echo
echo "============================================================"
echo "TIER 1 (core MusicDET comparability) suite finished. Check $MASTER_LOG for any"
echo "non-zero exit codes above and re-run just that step if needed. Key outputs:"
echo "  data/processed/reconstruction_control/reconstruction_auc.csv"
echo "  data/processed/external_scores/fakemusiccaps_native_auc.csv"
echo "  data/processed/fmc_native_flow/native_comparison_vs_table1.csv"
echo "  data/processed/robustness_battery/robustness_auc_eer.csv"
echo
echo "Once these look good, move to TIER 2 (exploratory/improvement, decide based on"
echo "the above + confound_analysis findings):"
echo "  bash scripts/run_genre_balanced_flow.sh      # finish the interrupted balancing run"
echo "  bash scripts/run_xcodec_udio30.sh             # C3: alternative embeddings for udio-30s"
echo "  python scripts/download_missing_sonics_reals.py --max-tracks 200  # small test batch only, see below"
echo "============================================================"
