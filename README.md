# Label-Free or Zero-Shot? The Transductive Dose of an AI Music Detector and a Detector Without One

**Ioannis Prokopiou**<sup>1,2</sup>, **Pantelis Vikatos**<sup>2</sup>,
**Maximos Kaliakatsos-Papakostas**<sup>3</sup>, **Theodoros Giannakopoulos**<sup>4</sup>,
**Themos Stafylakis**<sup>1,5</sup>

<sup>1</sup>Athens University of Economics and Business, Athens, Greece ·
<sup>2</sup>Orfium, Athens, Greece ·
<sup>3</sup>Hellenic Mediterranean University, Rethymno, Greece ·
<sup>4</sup>NCSR "Demokritos", Athens, Greece ·
<sup>5</sup>Archimedes/Athena R.C., Athens, Greece

Correspondence: [gian.prokopiou@aueb.gr](mailto:gian.prokopiou@aueb.gr) ·
Submitted to IEEE ICASSP 2027 ·
**Project page: <https://giannisprokopiouorfium.github.io/ai-music-dose>**

---

Code and full result record for the paper. It does two things. It **measures**
what a published label-free detector's protocol is worth: the detector's dictionary
is fitted on the corpus being scored, and that is worth 0.25 to 0.56 AUC. It then
**proposes** a detector that never pays that price, because it learns no dictionary
at all.

## The result, in one table

A published label-free detector (Afchar and Hennequin) learns a non-negative
matrix factorisation (NMF) dictionary over spectral "fakeprints" and scores a track
by the peak energy its reconstruction loses when the atoms are blurred. **No fake
label enters the estimator.** But the dictionary is learned from a matrix that
contains the corpus being scored. We varied only *which unlabelled audio the
dictionary sees*:

| the dictionary is learned from | FakeMusicCaps (32,960 tracks) | SONICS (59,280 tracks) |
|---|---|---|
| the scored corpus itself (the published protocol) | **0.9917** | **0.9389** |
| a different corpus (inductive transfer) | 0.5733 | 0.7123 |
| **real music only, held out, 3 draws** | **0.744 ± 0.020** | **0.382 ± 0.043** |
| **cost of the protocol** | **−0.248** | **−0.557** |

Signed macro AUC (one AUC per generator family against all real music, oriented
in advance, averaged without weights), full corpus, fitted rows excluded from
scoring. **FakeMusicCaps degrades; SONICS inverts**: real music scores higher than
generated music.

The gap closes at a small dose. On SONICS, **about 50 generated tracks in a
6,400-track fit set (0.7%) flip the detector**, and the draws at that dose are
bimodal: per draw the Suno families score either about 0.31 or about 0.98 with
nothing between, so the standard deviation over draws peaks *at* the transition
(0.216) and collapses after it (0.0006). With ten draws, 0 of 10 flip at 23
tracks, 7 of 10 at 47 and 10 of 10 at 70.

| generated tracks in the fit set | 0 | 9 | 23 | **47** | **70** | 233 | 931 | 11,640 |
|---|---|---|---|---|---|---|---|---|
| mean AUC | 0.382 | 0.400 | 0.493 | 0.661 | **0.787** | 0.793 | 0.901 | 0.932 |
| SD over draws | 0.043 | 0.054 | 0.137 | **0.216** | **0.0006** | 0.003 | 0.002 | 0.001 |
| draws flipped (of 3) | 0 | 0 | 0 | **2** | **3** | 3 | 3 | 3 |

The flipping dose falls as the dictionary grows (168 to 341 tracks at 5 atoms, 66
to 168 at 10, about 47 at 20), consistent with the dictionary needing enough
generated audio to spend one atom on the decoder comb.

## What we propose: a detector with no fit set

A dictionary must **discover** that periodicity is the signal, and discovery is
what costs unlabelled examples. But a transposed-convolution decoder leaves a
spectral comb whose spacing is `Δ = f_s / ∏ strides`, an architecture constant,
and published decoders' stride products form a short list. A detector *told* where
to look learns nothing and owes no dose.

1. **The residual.** Take the time-averaged log power spectrum (16,384-point
   short-time Fourier transform, mean of dB over frames), subtract a five-bin running
   median, and keep the residual `e` above 1 kHz. Remove its mean.
2. **The autocorrelation along frequency.** By the Wiener–Khinchin theorem,
   `ρ(τ) = r(τ)/r(0)` with `r = IDFT(|DFT(e)|²)`, zero-padded so it is linear rather
   than circular. A comb of spacing `Δ` peaks at lag `Δ/b` for bin width `b`.
3. **The prior `P`.** The first `M = 4` multiples of six published decoder
   fundamentals (21.53, 50, 75, 86.13, 93.75 and 100 Hz), above a 40 Hz floor:
   18 lags instead of about 4,000, so noise rarely wins.
4. **The null `N`.** Every lag in 40–600 Hz (`M × 150` Hz, 150 Hz being the highest
   plausible decoder fundamental) *except* ±1 bin around `k·Δ` for every fundamental
   and **every** integer `k`: 441 lags on FakeMusicCaps and 252 on SONICS.
5. **The score.** `m = max_P ρ − max_N ρ`. Whatever else is true of the track (how
   smooth, how peaky) raises both terms and cancels.

**Which null is the contribution.** A decoder emits energy at every multiple of
`Δ`, so a null holding the comb's own higher teeth makes the score subtract its own
evidence: **−0.128** on a clean synthetic 50 Hz comb, where the harmonic-protected
null scores **+0.511**. An earlier version drew 24 random decoy sets as the null
instead; a drawn lag was sometimes one of the prior's own, so **19.5% of
FakeMusicCaps margins were exactly 0**, and a quantile threshold landed inside that
tie: **3% recall on the worst generator while the AUC read 0.915**. The enumerated
null needs no seed and cannot contain a comb tooth by construction.

| arm | trainable parameters | FakeMusicCaps | SONICS | inverted families |
|---|---|---|---|---|
| NMF blurred-atom (the method we measure) | transductive dictionary | 0.9917 | 0.9389 | — |
| comb statistic (no prior, no null) | 0 | 0.9177 | 0.6864 one-sided / 0.7546 two-sided | 2 of 10 |
| + decoder prior, random decoy null (superseded) | 0 | 0.9153 | 0.8272 | 0 of 10 |
| **+ decoder prior, deterministic null (proposed)** | **0** | **0.9445** | **0.8286** | **0 of 10** |

On FakeMusicCaps the proposed score's 95% interval `[0.9416, 0.9474]` is
**disjoint** from the comb statistic's `[0.9128, 0.9227]`, and it beats the decoy
form at DeLong *p* = 5.6×10⁻¹⁰⁴. On SONICS it is **+0.074** macro over the comb
statistic but a **statistical tie** with the decoy form (*p* = 0.106). SONICS macro
0.8286 corresponds to pooled 0.7925 `[0.7887, 0.7963]`: the three Suno families read
0.99, the two Udio families 0.586 and 0.579. The claim is the last column: **the only
score we tested correctly oriented on all ten generator families.**

| setting (change in macro AUC) | FakeMusicCaps | SONICS |
|---|---|---|
| `M = 2` instead of 4 | −0.001 | −0.007 |
| `M = 1` instead of 4 | −0.015 | −0.029 |
| floor 20 Hz instead of 40 Hz | −0.006 | +0.000 |

Only `M = 1` costs accuracy, because Stable Audio's 21.5 Hz series then has no
multiple above the floor.

**Validation.** The eleven-gate confound battery gives the comb statistic's own
profile (channel and level fail marginally at 0.61–0.63 against a pre-registered
0.60 on both corpora; the other nine pass). The content-identical control passes on
both corpora: a transposed-convolution codec (EnCodec) scores above its own source
on 93–98% of pairs, against 47–58% for phase-only Griffin-Lim reconstruction, which
sits at chance on FakeMusicCaps and is barely detectable on SONICS (*p* ≈ 0.0015) at
a median shift 33–40× below the codec's. The score responds to *deconvolution*, not
to re-synthesis.

**Deployment.** 378 audio-seconds per second on four CPU cores, within 3% of the
uncalibrated comb statistic (the decoy form was 2.4× slower). A threshold set on
benchmark reals and applied to 2,998 unseen Free Music Archive (FMA) tracks lands at
0.72× the intended false-positive rate while catching 87.6% of generated tracks; the
comb statistic's rate grows 2.75×. Under group-conditional conformal calibration all
eight FMA genres sit at 4.4–5.1% against a 5% target, where the comb statistic's raw
rate spans 0.236 (Electronic) to 0.028 (Folk). Precision is a base-rate property:
reaching 0.9 needs a prevalence near 6% at the 99.9th-percentile threshold, or 30%
at the 95th.

**What it does not do.** It assumes a strided transposed-convolution decoder. It
is defeated by **pitch-shift, which drives it below chance (0.4665)**: resampling
moves `Δ` off all six candidate frequencies (time-stretch, a phase vocoder, leaves
`Δ` in place and only blurs the teeth). And **it does not detect Udio**: three
independent statistics agree that Udio leaves no comb inside these 16 kHz
benchmarks' band, so the calibration turns an inversion into a miss, a smaller and
different thing.

**0.9917 is not our zero-shot number and is never presented as one.**

## How to read this repository

| start here | what it is |
|---|---|
| **`reports/provenance_ledger.md`** | **The authority on every number.** Each figure re-read from its output and marked verified, corrected or retracted. If a number is not in here, it is not in the paper. |
| `icassp/paper.tex` | the submission; every number carries a `% src:` comment naming its source |
| `scripts/README.md` | what each live script produces, and which paper number it backs |
| `reports/project_narrative.md` | intrinsic dimension, then one-class normalising flows, then this: what failed and why |
| `docs/index.html` | the project page, including an interactive version of the dose figure |

## Withdrawn claims, kept visible

Every one was withdrawn because a value or a criterion was carried forward instead
of re-derived. They are listed rather than buried, because the protocol lessons are
the reason the surviving numbers can be trusted.

| # | claim | status and cause |
|---|---|---|
| 1 | "NMF reconstruction error inverts" (0.4998 / 0.2564) | **Retracted**: wrong quantity (`‖x−HW‖` rather than `‖HW−H(W∗G)‖`) and wrong fit set |
| 2 | "We extend the fakeprint below 16 kHz" | **Withdrawn**: the reference code already takes a frequency range |
| 3 | SONICS 0.8468 | **Retracted**: that was `\|AUC\|`; signed it is 0.6203. `\|AUC\|` hid a per-generator inversion |
| 4 | "The comb statistic uses Afchar's operator" | **Corrected**: the winning operator (median residual) is in neither reference codebase; operator and readout are coupled |
| 5 | "Reals-only inverts to 0.167" (FakeMusicCaps) | **Retracted**: the fit reals *were* the scored reals. Held out at full corpus it is **0.744 ± 0.020** |
| 6 | "At 1% prevalence, an FPR of 5% caps precision at 0.161" | **Retracted**: untraceable to any output. The measured value is **0.127** at FPR 4.70% on 6,361 held-out reals |
| 7 | "The atoms recover the decoder strides", on both corpora | **Narrowed**: holds on FakeMusicCaps (*p* = 0.020 against a cross-corpus null), not shown on SONICS (*p* = 0.45), whose comb sits within 0.12% of a 2:1 ratio to FakeMusicCaps' |
| 8 | "The published descriptor is 445-dimensional" | **Retracted**: it is 4,458. The resolution finding that replaced it is real and smaller |
| 9 | "An exhaustive prior falls to chance" (0.5044) | **Retracted**: the sweep moved the prior and the null together, and the fixture was weaker than its neighbours'. Controlled, 2 → 40 harmonics costs about 0.02 AUC |
| — | A calibration built from 24 randomly drawn decoy sets | **Superseded**: 19.5% of FakeMusicCaps margins tied at exactly 0, giving 3% recall on the worst generator at AUC 0.915. Replaced by the deterministic null |

Two software defects were also found and fixed, both pinned by tests: parallel
workers returned rows in completion order, so budgeted fits drew different samples
despite a fixed seed; and results were written only at the end of a sweep, so three
multi-hour runs saved nothing.

### The rules these earned

* **Whenever a score is read off a fitted object, check whether the fitted rows are
  also scored.**
* **Report signed AUC.** `|AUC|` measures how much class information a nuisance
  variable carries; it does not measure detector performance.
* **No number enters a report without a confound-gate output beside it, and a SKIP
  is not a PASS.**
* **Put a reproducibility anchor, a repeated cell, in every sweep.**
* **Run several draws.** The bimodality in the dose is invisible at one.
* **Choose the null that discriminates, not the one that is generous** (retraction 7).
* **Before recording a negative result, diff against the source repository's actual
  operator**, not its abstract.

## Layout

```text
icassp/            paper.tex, refs.bib, figs/ and the ICASSP style files
scripts/           the live scripts; scripts/README.md maps each to the numbers it backs
scripts/legacy/    superseded code, moved rather than deleted: some of it produced
                   numbers still cited, including retracted ones kept deliberately
src/intrinsic_ai_music_detection/
  features/        comb_artifacts.py holds every spectral operator and the detector
  data/            audio_preprocessing.py holds the channel-matching chain
  models/          evaluate.py holds AUC, EER, bootstrap intervals and the DeLong test
tests/             511 tests; several pin silent-corruption regressions
reports/           the provenance ledger and the project narrative
docs/              the project page
```

The package name, `intrinsic_ai_music_detection`, dates from the project's first
phase.

## Getting started

```bash
git clone https://github.com/GiannisProkopiouOrfium/ai-music-dose.git
cd ai-music-dose

# Poetry, with the virtual environment inside the repository (poetry.toml)
poetry install
source .venv/bin/activate
```

Python `>=3.10,<3.14`. The audio stack (`librosa`, `soxr`) is needed to compute
features; the figures and the paper build without it.

```bash
python3 -m pytest tests/ -q          # 510 passed, 1 skipped, with the audio stack
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so no `PYTHONPATH` is
needed. Without the audio stack the audio-dependent tests skip, so run the suite
with it installed before trusting a result.

**Data and compute.** The datasets are not redistributed (see below). The heavy
runs used a single 4-vCPU, 15 GB machine; the detector itself needs no GPU, and the
full-corpus NMF factorisations are the memory-bound step. Result directories
(`data/processed/`, `reports/diagnostics/`) are not version-controlled; they are
regenerated by the commands below, and `bash scripts/print_paper_numbers.sh` prints
every paper number from them.

## Reproducing the argument: three commands that differ in one flag

```bash
# 1. the published protocol: the dictionary sees the corpus being scored
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \
  --per-stratum 0 --max-duration 120 --level-match \
  --n-atoms 20 --blur-sigma 5 --fit-on pooled \
  --out-dir reports/diagnostics/nmf_sonics_FULL            # -> 0.9389

# 2. what a deployer has: real music only, dictionary rows held out
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \
  --per-stratum 0 --max-duration 120 --level-match \
  --n-atoms 20 --blur-sigma 5 --fit-on reals --holdout-reals \
  --fit-seeds 42 43 44 \
  --out-dir reports/diagnostics/nmf_sonics_reals_heldout   # -> 0.382 +/- 0.043

# 3. the dose: half the reals plus a known number of unlabelled generated tracks
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \
  --per-stratum 0 --max-duration 120 --level-match \
  --n-atoms 20 --blur-sigma 5 --fit-on mixture --fit-real-frac 0.5 \
  --holdout-reals --fit-seeds 42 43 44 \
  --out-dir reports/diagnostics/nmf_sonics_curve_FINAL     # -> Figure 1
```

FakeMusicCaps is the same with `canonical_fmc_raw16`, `--max-duration 9` and
`--blur-sigma 2`. `--holdout-reals` is not optional: without it the same command
returns an inverted score for reasons unrelated to generated audio (retraction 5).

## Reproducing the proposed detector

Two commands per corpus. The first extracts the comb features, the prior and the
harmonic-protected null; the second forms the calibrated score and its per-family
AUC table. The null is deterministic, so `--null-priors 0`: there are no decoy sets
to draw and no seed to record.

```bash
python scripts/eval_comb_detector.py \
  --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \
  --out-dir reports/diagnostics/comb_fmc_HNULL \
  --per-stratum 0 --workers 4 --max-duration 9 --level-match \
  --resample-hz 0 --smooth-bins 5 --residual median --average db \
  --f-min 1000 --hull-clip-db 0 --span-duration 4 \
  --n-harm 2 4 --null-priors 0

python scripts/derive_analytic_null.py \
  --score-csv reports/diagnostics/comb_fmc_HNULL/comb_per_track_*.csv \
  --out-csv  reports/diagnostics/comb_fmc_HNULL/analytic_null.csv
# -> the proposed score is the column comb_priormax4_hmargin; the *_zeroatom.csv
#    beside it reports the fraction of margins exactly 0 (0.0000 for this score)
```

For SONICS use `canonical_sonics_chain`, `--max-duration 120` and
`--resample-hz 15000` (decimate to 15 kHz and return to 24 kHz, removing the band
above 7.5 kHz that carries the channel label).

The controls that must accompany it:

```bash
# the eleven-gate battery, with the comb statistic as rival
python scripts/run_confound_gate.py \
  --score-csv reports/diagnostics/comb_fmc_HNULL/analytic_null.csv \
  --score-column comb_priormax4_hmargin \
  --descriptor-csv reports/diagnostics/channel_fmc_harm/channel_per_track_canonical_profile_lvl.csv \
  --rival-column median_db__comb_strength \
  --out-dir reports/diagnostics/gate_h4_fmc

# content-identical control: build the reconstruction pairs, score them with the two
# commands above, then compare within pairs
python scripts/analyze_recon_pairs.py \
  --score-csv <analytic_null.csv of the reconstruction-control extraction> \
  --score-column comb_priormax4_hmargin \
  --out-dir reports/diagnostics/c10_h4_fmc

# threshold transfer to FMA, then group-conditional calibration by genre
python scripts/measure_false_positive_rate.py \
  --labelled-csv reports/diagnostics/comb_fmc_HNULL/analytic_null.csv \
  --unseen-csv <analytic_null.csv of the FMA extraction> \
  --score-column comb_priormax4_hmargin \
  --out-dir reports/diagnostics/fpr_fma_h4
python scripts/measure_false_positive_rate.py \
  --labelled-csv reports/diagnostics/comb_fmc_HNULL/analytic_null.csv \
  --unseen-csv <analytic_null.csv of the FMA extraction> \
  --score-column comb_priormax4_hmargin \
  --group-column genre --calibrate-per-group --n-splits 50 --alpha 0.05 \
  --out-dir reports/diagnostics/mondrian50_h4

# robustness to manipulation, and cost
python scripts/run_robustness_battery.py --training-free comb_priormax4_hmargin \
  --output-dir data/processed/robustness_hmargin
python scripts/measure_efficiency.py --training-free comb_priormax4_hmargin \
  --null-priors 0 --audio-dir <directory of canonical real audio> --n-tracks 60 --n-trials 5
```

The descriptor table for the gate must be built at the duration the detector
actually saw; building it at 25 s and gating a 120 s score made the duration gate
read FAIL 0.6143 where the matched configuration gives PASS 0.5516 (ledger §H1).

## Building the paper

```bash
python3 scripts/make_paper_figures.py --out-dir icassp/figs
cd icassp && latexmk -pdf paper.tex     # 4 body pages + references on page 5
```

## Datasets and the method measured

* **SONICS**: Rahman et al., ICLR 2025 · [arXiv:2408.14080](https://arxiv.org/abs/2408.14080)
* **FakeMusicCaps**: Comanducci et al., *J. Imaging* 2025 · [arXiv:2409.10684](https://arxiv.org/abs/2409.10684)
* **FMA**: Defferrard et al., ISMIR 2017, for threshold transfer and per-genre false-positive rates
* **The method measured**: Afchar and Hennequin, [arXiv:2607.25530](https://arxiv.org/abs/2607.25530),
  building on Afchar et al., [arXiv:2506.19108](https://arxiv.org/abs/2506.19108) (ISMIR 2025).
  Their result stands; what we measure is what its *protocol* contributes.

## Funding

This research was funded by the European Union's Horizon Europe research and
innovation programme under the AIXPERT project (Grant Agreement No. 101214389).

## Licence and citation

MIT, see [`LICENSE`](LICENSE). The datasets above carry their own licences and are
not redistributed here. If you use this code or its measurements, please cite the
paper; [`CITATION.cff`](CITATION.cff) has the metadata.

```bibtex
@inproceedings{prokopiou2027dose,
  author    = {Ioannis Prokopiou and Pantelis Vikatos and Maximos Kaliakatsos-Papakostas
               and Theodoros Giannakopoulos and Themos Stafylakis},
  title     = {Label-Free or Zero-Shot? The Transductive Dose of an AI Music Detector
               and a Detector Without One},
  booktitle = {Submitted to IEEE ICASSP 2027},
  year      = {2027},
  url       = {https://giannisprokopiouorfium.github.io/ai-music-dose}
}
```
