#!/usr/bin/env bash
# =============================================================================
# print_paper_numbers.sh
#
# Dump every submission-critical result to stdout, in full, on the EC2 box.
#
#   cd ~/intrinsic-ai-music-detection && git pull
#   bash scripts/print_paper_numbers.sh 2>&1 | tee /tmp/paper_numbers.txt
#
# then paste /tmp/paper_numbers.txt back.
#
# WHY THIS EXISTS
# ---------------
# Four claims in this project have been retracted, every one because a number was
# carried forward instead of re-read from its CSV. Nothing can enter the ICASSP
# paper that has not been read from the file that produced it.
#
# Two rules this script enforces, both learned the hard way:
#
#   1. NEVER `tail`. Result tables were truncated six separate times by `tail -30`,
#      cutting exactly the header rows that carry the MACRO column. Small tables
#      are printed WHOLE.
#
#   2. Per-track CSVs are tens of thousands of rows, so those get their HEADER and
#      row count only. That header is the point: the paper's later commands need
#      the REAL column names (`r_k20_sigma2_bins7167_pooled`, not a guessed
#      suffix). Guessed suffixes cost a round in R4, R5 and R11.
#
# A MISSING file is printed as MISSING rather than skipped silently — knowing a
# result does not exist is itself provenance.
# =============================================================================

# Deliberately NOT `set -e`: one missing file must not abort the whole dump.
set -uo pipefail
shopt -s nullglob

cd "$(dirname "$0")/.." || exit 1

DIAG="reports/diagnostics"
PROC="data/processed"

rule() { printf '%s\n' "-------------------------------------------------------------------------------"; }

section() {
    printf '\n\n'
    printf '%s\n' "==============================================================================="
    printf '## %s\n' "$1"
    printf '%s\n' "==============================================================================="
}

# A literal path is NOT removed by nullglob, so every candidate is -f tested.
# Without that guard a missing file printed "(-1 rows)" and a `cat` error.

# full <glob...> — print the entire file. For AUC / gate / summary tables.
full() {
    local pat found f
    for pat in "$@"; do
        found=0
        for f in $pat; do
            [[ -f "$f" ]] || continue
            found=1
            rule
            printf '### FULL: %s   (%s rows)\n' "$f" "$(( $(wc -l < "$f") - 1 ))"
            rule
            cat "$f"
            printf '\n'
        done
        if [[ $found -eq 0 ]]; then
            rule
            printf '### MISSING: %s\n\n' "$pat"
        fi
    done
}

# header <glob...> — print header + row count only. For per-track CSVs.
header() {
    local pat found f
    for pat in "$@"; do
        found=0
        for f in $pat; do
            [[ -f "$f" ]] || continue
            found=1
            rule
            printf '### HEADER ONLY: %s   (%s data rows)\n' "$f" "$(( $(wc -l < "$f") - 1 ))"
            rule
            head -1 "$f"
            printf '\n'
        done
        if [[ $found -eq 0 ]]; then
            rule
            printf '### MISSING: %s\n\n' "$pat"
        fi
    done
}

printf '%s\n' "==============================================================================="
printf '# PAPER NUMBERS DUMP — %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf '# host=%s  commit=%s\n' "$(hostname)" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf '# branch=%s\n' "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
printf '%s\n' "==============================================================================="

# -----------------------------------------------------------------------------
section "1. HEADLINE — full corpus, both benchmarks"
# -----------------------------------------------------------------------------
full "$DIAG/nmf_fmc_FULL/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_fmc_FULL/nmf_peak_operating_point_lvl.csv" \
     "$DIAG/nmf_sonics_FULL/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_sonics_FULL/nmf_peak_operating_point_lvl.csv" \
     "$DIAG/comb_fmc_FULL/comb_auc_"*.csv
header "$DIAG/nmf_fmc_FULL/nmf_peak_per_track_lvl.csv" \
       "$DIAG/nmf_sonics_FULL/nmf_peak_per_track_lvl.csv" \
       "$DIAG/comb_fmc_FULL/comb_per_track_"*.csv

# -----------------------------------------------------------------------------
section "2. CONTRIBUTION 1 — the transductive boundary"
# -----------------------------------------------------------------------------
full "$DIAG/nmf_fmc_fitreals/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_fmc_nreals/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_sonics_nreals/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_sonics_fitfmc/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_fmc_fitsonics/nmf_peak_auc_lvl.csv"
header "$DIAG/nmf_sonics_fitfmc/nmf_peak_per_track_lvl.csv" \
       "$DIAG/nmf_fmc_fitsonics/nmf_peak_per_track_lvl.csv"

# -----------------------------------------------------------------------------
section "3. CONTRIBUTION 3 — sign consistency (SIGNED vs |AUC|)"
# -----------------------------------------------------------------------------
full "$DIAG/comb_fmc_grid_signed/comb_auc_"*.csv \
     "$DIAG/comb_sonics_grid_120s_signed/comb_auc_"*.csv

# -----------------------------------------------------------------------------
section "4. CONTRIBUTION 4 — the two-sided four-cell ablation"
# -----------------------------------------------------------------------------
full "$DIAG/fusion_fmc_twosided/fusion_auc.csv" \
     "$DIAG/fusion_sonics_twosided/fusion_auc.csv" \
     "$DIAG/fusion_sonics_onesided/fusion_auc.csv" \
     "$DIAG/nmf_sonics_twosided/fusion_auc.csv" \
     "$DIAG/nmf_sonics_fitfmc_twosided/fusion_auc.csv"

# -----------------------------------------------------------------------------
section "5. THE CHANNEL-CONTROL PROTOCOL (incl. the 0.9993 row)"
# -----------------------------------------------------------------------------
full "$DIAG/channel_sonics_UNCONTROLLED/channel_alone_auc_"*.csv \
     "$DIAG/comb_sonics_UNCONTROLLED/comb_auc_"*.csv \
     "$DIAG/comb_fmc_grid_rs15k/comb_auc_"*.csv \
     "$DIAG/comb_sonics_grid_120s_rs15k/comb_auc_"*.csv \
     "$DIAG/channel_fmc_2026_08_20/channel_alone_auc_"*.csv \
     "$DIAG/channel_fmc_harm/channel_alone_auc_"*.csv
header "$DIAG/comb_sonics_grid_120s_rs15k/comb_per_track_"*.csv \
       "$DIAG/channel_fmc_harm/channel_per_track_"*.csv

# -----------------------------------------------------------------------------
section "6. GATES C1-C11 — every one of them"
# -----------------------------------------------------------------------------
# The shell expands this before `full` sees it, so an empty expansion would print
# nothing at all. Silence is the wrong answer for provenance — say so explicitly.
GATE_FILES=( "$DIAG"/gate_*/gate_*.csv )
if [[ ${#GATE_FILES[@]} -eq 0 ]]; then
    rule; printf '### MISSING: no %s/gate_*/gate_*.csv on this box\n' "$DIAG"
else
    printf '### %s gate files found\n' "${#GATE_FILES[@]}"
    full "${GATE_FILES[@]}"
fi

# -----------------------------------------------------------------------------
section "7. C10 — content-identical pairs (report v3, NOT v2)"
# -----------------------------------------------------------------------------
full "$DIAG/comb_recon_control/comb_auc_"*.csv \
     "$PROC/recon_control_v3/reconstruction_auc.csv" \
     "$PROC/recon_control_v2/reconstruction_auc.csv"
header "$DIAG/comb_recon_control/comb_per_track_"*.csv
printf '\n### listing: %s\n' "$PROC/recon_control_v3/"
ls -la "$PROC/recon_control_v3/" 2>/dev/null || printf 'MISSING\n'
printf '\n### csv headers under recon_control_v3/\n'
for f in "$PROC"/recon_control_v3/*.csv; do printf '%s :: ' "$f"; head -1 "$f"; done

# -----------------------------------------------------------------------------
section "8. DEPLOYMENT — precision, transfer FPR, per-genre"
# -----------------------------------------------------------------------------
full "$DIAG/analysis_nmf_sonics_FULL/"*.csv \
     "$DIAG/analysis_nmf_fmc/"*.csv \
     "$DIAG/analysis_comb_fmc/"*.csv \
     "$DIAG/fpr_fma_by_genre/"*.csv \
     "$DIAG/fpr_fma_nmf/"*.csv \
     "$DIAG/fpr_fma/"*.csv

# -----------------------------------------------------------------------------
section "9. EFFICIENCY AND ROBUSTNESS"
# -----------------------------------------------------------------------------
full "$PROC/robustness_comb_median_db/robustness_auc_eer.csv"
for f in "$PROC"/efficiency_*.json; do
    rule; printf '### FULL: %s\n' "$f"; rule; cat "$f"; printf '\n'
done

# -----------------------------------------------------------------------------
section "10. THE SUPERVISED BASELINE (Afchar LR) AND ITS TRANSFER COLLAPSE"
# -----------------------------------------------------------------------------
full "$DIAG/fakeprint_lr_fmc/fakeprint_lr_lvl.csv" \
     "$DIAG/fakeprint_lr_sonics_120s/fakeprint_lr_lvl.csv"

# -----------------------------------------------------------------------------
section "11. NEGATIVES — incl. the f-min sweep whose SIGNED udio values were never read"
# -----------------------------------------------------------------------------
full "$DIAG/comb_sonics_fmin200/comb_auc_"*.csv \
     "$DIAG/comb_sonics_fmin500/comb_auc_"*.csv \
     "$DIAG/comb_sonics_fmin1000/comb_auc_"*.csv \
     "$DIAG/nmf_sonics_fmin200/nmf_peak_auc_lvl.csv" \
     "$DIAG/phd_fmc_v2/phd_auc_"*.csv \
     "$DIAG/phd_fmc_encodec/phd_auc_"*.csv \
     "$DIAG/flow_terms_fmc_holdout/flow_terms_auc_lvl.csv"

# -----------------------------------------------------------------------------
section "12. FIGURE 2 SUPPORT — atom peakiness"
# -----------------------------------------------------------------------------
full "reports/figures/nmf_fmc/"*.csv "reports/figures/nmf_sonics/"*.csv
printf '\n### figure files present\n'
ls -la reports/figures/nmf_fmc/ reports/figures/nmf_sonics/ 2>/dev/null || printf 'MISSING\n'

# -----------------------------------------------------------------------------
section "13. CORPUS SIZES — the n behind every claim"
# -----------------------------------------------------------------------------
for m in "$PROC/canonical_fmc_raw16/canonical_manifest.csv" \
         "$PROC/canonical_sonics_chain/canonical_manifest.csv" \
         "$PROC/canonical_fma_genre/canonical_manifest.csv" \
         "$PROC/canonical_sonics_full/canonical_manifest.csv"; do
    rule
    if [[ -f "$m" ]]; then
        printf '### %s :: %s rows\n' "$m" "$(( $(wc -l < "$m") - 1 ))"
        printf 'header: '; head -1 "$m"
        printf 'label x algorithm counts:\n'
        python3 - "$m" <<'PYEOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
cols = [c for c in ("label", "algorithm") if c in df.columns]
print(df.groupby(cols).size().to_string() if cols else "no label/algorithm columns")
PYEOF
    else
        printf '### MISSING: %s\n' "$m"
    fi
done

# -----------------------------------------------------------------------------
section "14. WHAT ELSE IS ON THE BOX — every diagnostics dir, so nothing is overlooked"
# -----------------------------------------------------------------------------
ls -1 "$DIAG" 2>/dev/null || printf 'MISSING\n'

printf '\n### the same dirs by modification time, newest LAST\n'
ls -1rtd "$DIAG"/*/ 2>/dev/null || printf 'MISSING\n'

# -----------------------------------------------------------------------------
section "15. ROUNDS 5-10 — the leak-free, held-out, multi-seed results the paper reports"
# -----------------------------------------------------------------------------
# Added 2026-08-29. Sections 1-14 predate the hold-out fix (--holdout-reals),
# the row-order fix and the 3-seed sweeps, so NONE of the numbers the paper's
# contribution 1 and Figure 1 actually use appear above. Everything here is a
# result the provenance ledger cites from §I onward.

# --- Figure 1: the contamination curves, held out, 3 seeds -------------------
full "$DIAG/nmf_fmc_curve_FINAL/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_fmc_curve_FINAL/nmf_peak_operating_point_lvl.csv" \
     "$DIAG/nmf_sonics_curve_FINAL/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_sonics_curve_FINAL/nmf_peak_operating_point_lvl.csv"

# --- Contribution 1: reals-only, HELD OUT (supersedes nmf_*_fitreals) --------
full "$DIAG/nmf_fmc_reals_heldout/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_sonics_reals_heldout/nmf_peak_auc_lvl.csv"

# --- Cell 4 of the ablation, at full corpus (nmf_sonics_fitfmc was n=600) ----
full "$DIAG/nmf_sonics_fitfmc_FULL/nmf_peak_auc_lvl.csv" \
     "$DIAG/nmf_sonics_fitfmc_FULL_onesided/fusion_auc.csv" \
     "$DIAG/nmf_sonics_fitfmc_FULL_twosided/fusion_auc.csv" \
     "$DIAG/fusion_sonics_FULL_rs15k_onesided/fusion_auc.csv" \
     "$DIAG/fusion_sonics_FULL_rs15k_twosided/fusion_auc.csv" \
     "$DIAG/comb_sonics_FULL_rs15k/comb_auc_"*.csv

# --- C10, the WITHIN-PAIR analysis (a corpus AUC does not exist here) --------
full "$DIAG/c10_sonics/recon_pairs.csv" \
     "$DIAG/c10_fmc/recon_pairs.csv" \
     "$PROC/recon_control_sonics/reconstruction_auc.csv"

# --- The chain-matched channel table (C1-C4 on SONICS at max-duration 120) ---
full "$DIAG/channel_sonics_chain_120s/channel_alone_auc_"*.csv \
     "$DIAG/channel_sonics_chain_2026_08_23/channel_alone_auc_"*.csv

# --- Dictionary learning is transductive: sparse coding and k-means ----------
full "$DIAG/dict_sparse_fmc_pooled/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_sparse_fmc_reals/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_kmeans_fmc_pooled/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_kmeans_fmc_reals/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_sparse_sonics_pooled/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_sparse_sonics_reals/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_kmeans_sonics_pooled/nmf_peak_auc_lvl.csv" \
     "$DIAG/dict_kmeans_sonics_reals/nmf_peak_auc_lvl.csv"

# --- FIGURE 2: per-atom PERIODICITY and SPACING -----------------------------
# NOT the same files as section 12. Section 12's `atom_peakiness.csv` is the
# SUPERSEDED peakiness-only run; ledger §F4 shows peakiness is the score's own
# sensitivity and is NOT evidence of a comb. The `*_periodicity/` dirs carry the
# `periodicity` and `spacing_hz` columns Figure 2 is drawn from.
full "reports/figures/nmf_fmc_periodicity/"*.csv \
     "reports/figures/nmf_sonics_periodicity/"*.csv
printf '\n### figure files present (periodicity runs)\n'
ls -la reports/figures/nmf_fmc_periodicity/ reports/figures/nmf_sonics_periodicity/ 2>/dev/null \
    || printf 'MISSING\n'

# --- Every gate produced after 2026-08-23, by mtime, so none is overlooked ---
# Section 6's glob prints every gate file it finds, but it cannot tell you WHICH
# are new. This does, so a gate written in Round 7 is not read as a Round 1 one.
printf '\n### gate dirs newer than the 2026-08-23 dump\n'
find "$DIAG" -maxdepth 1 -type d -name 'gate_*' -newermt '2026-08-23' 2>/dev/null \
    | sort || printf '(find -newermt unavailable)\n'

printf '\n\n=== END OF DUMP ===\n'
