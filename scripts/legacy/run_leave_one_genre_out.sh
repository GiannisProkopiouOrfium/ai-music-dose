#!/usr/bin/env bash
# =============================================================================
# run_leave_one_genre_out.sh
#
# Wrapper for run_leave_one_genre_out.py — our analog of MusicDET Table 4
# ("Leave-one-subdomain-out evaluation"). Runs it for both excluded subdomains
# (jazz = exact match to MusicDET's protocol; classical = closest available
# analog to their "piano" subset, since our CLAP genre vocabulary tags genre
# not instrumentation — see run_leave_one_genre_out.py docstring for why).
#
# Usage:
#   bash scripts/run_leave_one_genre_out.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
GENRE_CSV="data/processed/genre_tags_all/all_genres.csv"
EMB_CACHE="data/emb_cache_encodec_full"

for GENRE in jazz classical; do
    echo
    echo "============================================================"
    echo "  Leave-one-genre-out: excluding '$GENRE' from training"
    echo "============================================================"
    python scripts/run_leave_one_genre_out.py \
        --canonical-manifest "$SONICS_MANIFEST" \
        --genre-csv "$GENRE_CSV" \
        --embedding-cache-dir "$EMB_CACHE" \
        --exclude-genre "$GENRE" \
        --output-dir "data/processed/leave_one_genre_out/$GENRE" \
        --n-coupling-layers 12 \
        --device cuda \
        2>&1 | tee "logs/loso_${GENRE}.log"
done

echo
echo "Done. Summaries:"
echo "  data/processed/leave_one_genre_out/jazz/loso_jazz_summary.csv"
echo "  data/processed/leave_one_genre_out/classical/loso_classical_summary.csv"
