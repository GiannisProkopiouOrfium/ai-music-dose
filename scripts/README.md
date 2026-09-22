# `scripts/` — what is live, and what each live script produces

Two tiers. **Everything at this level is live**: it either produces a number in
the ICASSP paper, builds a corpus the paper depends on, or is pinned by a test.
**`legacy/` is the archive** — 85 superseded scripts, moved rather than deleted
because several produced numbers that are still cited, including retracted ones
kept deliberately (`legacy/eval_nmf_reconstruction.py` produced retraction #1).

Nothing in `legacy/` is run any more, and its repo-root path derivation is one
directory off since the move. Fixing it is not planned; if you need one, read it
where it is or move it back.

---

## Detectors and scoring

| script | produces | paper |
|---|---|---|
| `eval_nmf_peak_detector.py` | the NMF blurred-atom arm. `--fit-on {pooled,external,reals,mixture}` is the protocol axis; `--holdout-reals`, `--fit-real-frac`, `--fit-seeds`, `--dictionary {nmf,sparse,kmeans}` | Table 1, Fig. 1, Table 3 |
| `eval_comb_detector.py` | the training-free comb arm; `--operator-grid` sweeps residual/average operators | §6, robustness |
| `fuse_physics_scores.py` | two-sided real-only scoring and the four-cell ablation | §4.3 |

## Confound gates and controls

| script | produces | paper |
|---|---|---|
| `run_confound_gate.py` | gates C1–C11 for any per-track score | §4.4 |
| `diagnose_channel_confound.py` | the channel-descriptor tables (`hf_floor_frac`, `frac_power_above_8k`, …) | Table 2 |
| `build_reconstruction_control.py` | content-identical pairs: EnCodec and Griffin-Lim variants of the same source audio | C10 |
| `analyze_recon_pairs.py` | the **within-pair** C10 test. A reconstruction control is all real music, so a corpus-level AUC does not exist; this computes the paired statistic that does | §4.4 |

## Deployment analysis

| script | produces | paper |
|---|---|---|
| `analyze_detector.py` | operating points, precision at prevalence, per-generator recall | §6 prevalence |
| `measure_false_positive_rate.py` | transfer FPR on unseen real music, and the per-genre breakdown | §6 transfer, fairness |
| `measure_efficiency.py` | throughput and parameter count | §6 cost |
| `run_robustness_battery.py` / `.sh` | the eight-manipulation battery against MusicDET Table 6 | §6 stability |
| `apply_audio_manipulations.py` | the manipulations themselves (imported by the battery) | — |

## Figures and provenance

| script | produces |
|---|---|
| `make_paper_figures.py` | Fig. 1 and Fig. 2 as PDF into `icassp/figs/`, plus a `*_plotted.csv` sidecar recording exactly what reached the axes. Refuses to draw Fig. 2 from the ledger excerpt, which lists only the atoms that matched a decoder spacing |
| `visualize_nmf_space.py` | learned atoms with per-atom `periodicity` and `spacing_hz` |
| `print_paper_numbers.sh` | dumps every result CSV on EC2 for paste-back. §15 covers the Rounds 5–10 held-out, multi-seed results the paper actually reports |

## Corpus construction

| script | produces |
|---|---|
| `build_canonical_corpus.py` | `canonical_fmc_raw16` (32,960 tracks) |
| `build_channel_matched_corpus.py` | `canonical_sonics_chain` (59,280) — identical sample rate, codec, loudness and duration cap for real and generated audio |
| `build_fma_manifest.py`, `tag_real_genres.py`, `tag_all_genres.py` | `canonical_fma_genre` (3,000) with genre labels, for the transfer and fairness numbers |
| `download_sonics.py`, `download_external_data.sh` | acquisition |

## Pinned by tests, otherwise superseded

`score_flow_terms.py`, `score_external_corpus.py`, `fuse_labelfree_scores.py`
and `run_balanced_ablation.py` belong to the withdrawn normalising-flow arm.
They stay at this level because `tests/test_silent_fallback_guards.py`,
`test_flow_terms.py`, `test_combprint.py` and `test_flow_only_path.py` load them
by the literal path `scripts/<name>.py`. Those tests pin the silent-corruption
regressions that invalidated results four times; do not move these files without
updating the loaders.

---

## Reproducing the headline

```bash
# the published transductive protocol
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \
  --per-stratum 0 --max-duration 120 --level-match \
  --n-atoms 20 --blur-sigma 5 --fit-on pooled \
  --out-dir reports/diagnostics/nmf_sonics_FULL

# what a deployer actually gets: real music only, dictionary rows held out
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \
  --per-stratum 0 --max-duration 120 --level-match \
  --n-atoms 20 --blur-sigma 5 --fit-on reals --holdout-reals \
  --fit-seeds 42 43 44 \
  --out-dir reports/diagnostics/nmf_sonics_reals_heldout
```

**No number enters a report without a `run_confound_gate.py` output beside it,
and a SKIP is not a PASS.** Report signed AUC, never `|AUC|`, as detector
performance. Full claim-to-CSV map: `reports/provenance_ledger.md`.
