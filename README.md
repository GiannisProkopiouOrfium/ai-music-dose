# Label-Free or Zero-Shot? The Transductive Dose, and a Detector Without One

Code and full result record for the ICASSP 2027 submission *"Label-Free or
Zero-Shot? The Transductive Dose of an AI-Music Detector, and a Detector Without
One."*

The paper does two things. It **measures** what a published label-free detector's
protocol is worth — the dictionary is fitted on the corpus being scored, and that
is worth 0.25–0.56 AUC — and it **proposes** a detector that never pays that
price, because it learns no dictionary at all.

**Project page: [`docs/index.html`](docs/index.html)** — the finding as an
interactive figure, how the detector works, the commands that reproduce it, and
every withdrawn claim. Served by GitHub Pages from `/docs` on the default branch;
see [Publishing the project page](#publishing-the-project-page).

> **Note on the package name.** The Python package is still
> `intrinsic_ai_music_detection`. The project began as an intrinsic-dimension
> detector, spent its longest phase on one-class normalising flows, and both were
> falsified by our own measurements. The flow and intrinsic-dimension code is
> preserved in `scripts/legacy/` rather than deleted.

---

## The result, in one table

A published label-free detector learns an NMF dictionary over spectral
"fakeprints" and scores a track by the peak energy its reconstruction loses
under blurring. **No fake label enters the estimator.** But the dictionary is
learned from a matrix that contains the corpus being scored. We varied only
*which unlabelled audio the dictionary sees*:

| the dictionary is learned from | FakeMusicCaps (32,960) | SONICS (59,280) |
|---|---|---|
| the scored corpus itself (the published protocol) | **0.9917** | **0.9389** |
| a different corpus (inductive transfer) | 0.5733 | 0.7123 |
| **real music only, held out, 3 seeds** | **0.744 ± 0.020** | **0.382 ± 0.043** |
| **cost of the protocol** | **−0.248** | **−0.557** |

Signed macro AUC, full corpus, held out. **FakeMusicCaps degrades; SONICS
inverts** — real music scores higher than generated music.

And the gap closes at a startlingly small dose. On SONICS, **~50 generated
tracks in a 6,400-track dictionary (0.7%) flips the detector**, and sharply: per
draw the Suno families score either ~0.31 or ~0.98 with nothing between, so the
seed SD peaks *at* the transition (0.216) and collapses after it (0.0006). Three
draws per cell fix where the transition is and how abrupt it is, but not its
width or its mechanism — see the limitations in the paper.

| generated tracks in dict | 0 | 9 | 23 | **47** | **70** | 233 | 931 | 11,640 |
|---|---|---|---|---|---|---|---|---|
| mean AUC | 0.382 | 0.400 | 0.493 | 0.661 | **0.787** | 0.793 | 0.901 | 0.932 |
| seed SD | 0.043 | 0.054 | 0.137 | **0.216** | **0.0006** | 0.003 | 0.002 | 0.001 |
| draws flipped | 0/3 | 0/3 | 0/3 | **2/3** | **3/3** | 3/3 | 3/3 | 3/3 |

### What we propose: a detector with no fit set

A dictionary must **discover** that periodicity is the signal, and discovery is
what costs unlabelled examples. The comb's spacing is `f_s / ∏ strides`, an
architecture constant, and published decoders' stride products form a short list —
so a detector *told* where to look learns nothing and owes no dose. Two steps:

1. **Restrict the search.** Score the autocorrelation only at the six
   `f_s / ∏ strides` values published decoders imply, and their first four
   multiples above a 40 Hz floor, instead of taking the largest of ~4,000 noisy
   candidates. Call that set the prior, `P`.
2. **Calibrate per track.** Subtract a null `N` read on the same residual.
   Whatever else is true of that track — how smooth, how loud, how peaky — raises
   both terms and cancels.

**Which null is the contribution.** `N` is every lag in 40–600 Hz *except* ±1 bin
around `k·Δ`, for every candidate spacing `Δ` and **every** integer `k` — 441
surviving lags on FakeMusicCaps, 253 on SONICS. Deleting only `P` is not enough: a
decoder emits energy at every multiple, so a null holding the comb's own higher
teeth makes the score subtract its own evidence, scoring **−0.128** on a clean
synthetic 50 Hz comb where the protected null scores **+0.511**.

The first version drew 24 decoy candidate sets at random instead, and that has a
defect: a drawn lag is sometimes one of the prior's own, so **19.5% of
FakeMusicCaps margins were pinned at exactly 0**. A deployment threshold is a
quantile, so it lands inside the tie and takes all of it — **3% recall on the worst
generator while the AUC read 0.915**. The enumeration needs no seed and cannot
contain a real comb tooth by construction.

| arm | trained? | FakeMusicCaps | SONICS | inverted families |
|---|---|---|---|---|
| NMF blurred-atom (the method we measure) | **transductive** | 0.9917 | 0.9389 | — |
| comb statistic (training-free baseline) | 0 parameters | 0.9177 | 0.6864 / 0.7546 | 2 of 10 |
| + decoder prior, random decoy null | 0 parameters | 0.9153 | 0.8272 | 0 of 10 |
| **+ decoder prior, deterministic null (proposed)** | **0 parameters** | **0.9445** | **0.8286** | **0 of 10** |

On FakeMusicCaps the proposed score's `[0.9416, 0.9474]` is **disjoint** from the
baseline's `[0.9128, 0.9227]`, and it beats the decoy form at DeLong
*p* = 5.6×10⁻¹⁰⁴. On SONICS it is **+0.074** macro over the baseline but a
**statistical tie** with the decoy form (*p* = 0.106) — said plainly, because the
table invites the other reading. Udio moves from 0.19 and 0.18 (actively
mis-ordering real music above AI) to 0.586 and 0.579. The claim is the last column:
**the only score we tested correctly oriented on all ten generator families.**

Validation: the gate battery is the comb arm's own (C1/C2 marginal, everything else
passing), and the content-identical control passes on both corpora — a
transposed-convolution codec scores above its own source on 93–98% of pairs against
47–58% for phase-only reconstruction, which sits at chance on FakeMusicCaps and is
barely detectable on SONICS (*p* ≈ 0.0015) at a median shift 33–40× below the
codec's. So it responds to *deconvolution*, not to re-synthesis.

Deployment: 378 audio-seconds per second on four CPU cores (within 3% of the
uncalibrated arm; the decoy form costs 2.4× more), a threshold that transfers to
unseen FMA reals at 0.72× the intended rate while catching 87.6% of generated
tracks, and per-genre false-positive rates within 4.4–5.1% of a 5% target under
group-conditional conformal calibration. **Pitch-shift drives it below chance
(0.4665)**: resampling moves Δ off all six candidate frequencies, so the prior
misses on fakes specifically. That is a signed inversion, not a degradation.

**It does not detect Udio**, and does not claim to: three independent statistics
agree Udio leaves no comb inside these 16 kHz benchmarks' band. It converts an
inversion into a miss, which is a smaller and different thing.

**0.9917 is not our zero-shot number and is never presented as one.** Deployment
figures name their arm.

---

## How to read this repository

| start here | what it is |
|---|---|
| **`reports/provenance_ledger.md`** | **The authority on every number.** Each figure re-read from its CSV and marked `VERIFIED` / `CORRECTED` / `RETRACTED`. If a number is not in here, it does not go in the paper. |
| `scripts/README.md` | what each live script produces, and which paper number it backs |
| `reports/project_narrative.md` | intrinsic dimension → flows → here, with what failed and why |
| `icassp/paper.tex` | the submission. Every number carries a `% src:` comment naming its CSV |

---

## Withdrawn claims, kept visible

Every one was withdrawn because a value was carried forward instead of
re-derived. They are listed here rather than buried, because the protocol
lessons are the reason the surviving numbers can be trusted.

| # | claim | status and cause |
|---|---|---|
| 1 | "NMF reconstruction error inverts" (0.4998 / 0.2564) | **RETRACTED** — wrong quantity (`‖x−HW‖` rather than `‖HW−H(W∗G)‖`) and wrong fit set |
| 2 | "we extend the fakeprint below 16 kHz" | **WITHDRAWN** — the source already takes `--fmin`/`--fmax` |
| 3 | SONICS 0.8468 | **RETRACTED** — that was `\|AUC\|`; signed is 0.6203. `\|AUC\|` hid a per-generator inversion |
| 4 | "`comb_strength` uses Afchar's operator" | **CORRECTED** — the winning operator (`median_db`) is in neither codebase; operator and readout are coupled |
| 5 | "reals-only inverts to 0.167" (FMC) | **RETRACTED** — dictionary-level train-on-test: the fit reals *were* the scored reals. Held out at full corpus it is **0.744 ± 0.020**, and the FMC dose–response runs the opposite way to what we had |
| 6 | "at p=1%, FPR 5% caps precision at 0.161" | **RETRACTED 2026-08-29** — untraceable to any CSV. The measured value is **0.1265** at FPR 4.70% on 6,361 reals (ledger §O1) |
| 7 | "the atoms recover the decoder strides" — on *both* corpora | **NARROWED 2026-08-30** — holds on FakeMusicCaps (*p* = 0.020 against a cross-corpus null) but **not shown on SONICS** (*p* = 0.45): the two corpora's combs sit within 0.12% of a 2:1 ratio, so sub-multiple matches cannot attribute an atom to a decoder (ledger §O8b) |
| 8 | "the published descriptor is 445-dimensional" | **RETRACTED** — it is 4,458. We had recorded the wrong figure for someone else's paper and built a contribution on the difference. The resolution finding that replaced it is real and smaller (ledger §Q1) |
| 9 | "an exhaustive prior falls to chance" (0.5044) | **RETRACTED** — two defects, both ours: the sweep moved the prior and the null together, and the fixture was weaker than the neighbouring tables'. Controlled, 2 → 40 harmonics costs ≈0.02 AUC, not 0.46 (ledger R27.67) |
| — | a calibration built from 24 randomly drawn decoy sets | **SUPERSEDED** — a drawn decoy is sometimes one of the prior's own lags, pinning 19.5% of FakeMusicCaps margins at exactly 0; a quantile threshold then takes the whole tie, giving 3% recall on the worst generator at AUC 0.915. Replaced by the deterministic null above (ledger R27.20, R27.42) |

Plus two plumbing defects found in the final week, both of which had already
corrupted results: `as_completed` returned rows in worker-completion order, so
every budgeted fit drew a different sample despite a fixed seed; and results
were written only after a whole sweep, so three multi-hour runs computed
everything and saved nothing. Both fixed and pinned by tests.

### The rules these earned

* **Whenever a score is read off a fitted object, check whether the fitted rows
  are also scored.** This cost a headline twice.
* **Report signed AUC.** `|AUC|` is a legitimate measure of how much class
  information a nuisance variable carries and an illegitimate measure of
  detector performance.
* **No number enters a report without a gate output beside it, and a SKIP is not
  a PASS.**
* **Put a reproducibility anchor — a repeated cell — in every sweep.** Ours
  caught a defect that had already changed a headline.
* **Run three seeds.** The transition in the dose–response is invisible at one.
* **Choose the null that discriminates, not the one that is generous.** The
  atom-matching rule was picked because it caught many atoms; the same
  generosity is what stopped it attributing them (retraction 7).
* **Before recording a negative result, diff against the source repository's
  actual operator**, not its abstract.

---

## Layout

```
icassp/            paper.tex, refs.bib, figs/ (+ the ICASSP style files)
scripts/           28 live scripts — see scripts/README.md
scripts/legacy/    84 superseded, moved not deleted; several produced numbers
                   still cited, including retracted ones kept deliberately
src/               the library: features/, models/, data/
tests/             263 tests. Several pin silent-corruption regressions and
                   load scripts by literal path — read scripts/README.md first
reports/           the provenance ledger and the project narrative
```

## Getting started

```bash
git clone git@github.com:GiannisProkopiouOrfium/ai-music-dose.git
cd intrinsic-ai-music-detection

# Poetry, with the venv inside the repo (poetry.toml sets virtualenvs.in-project)
poetry install
source .venv/bin/activate

# or, on a fresh GPU box, which also pins the CUDA-adjacent packages:
bash scripts/setup_project_venv.sh
```

Python `>=3.10,<3.14`. The audio stack (`librosa`, `soxr`) is needed to compute
features; everything else — the tests, the figures, the paper — runs without it.

**Tests.**

```bash
# full suite, needs the audio stack
python3 -m pytest tests/ -q                                # 262 passed, 1 skipped

# without librosa/soxr: three files cannot be imported, so skip them
python3 -m pytest tests/ -q \
  --ignore=tests/test_audio_utils.py \
  --ignore=tests/test_embeddings.py \
  --ignore=tests/test_fakeprints.py                        # 235 passed, 20 skipped
```

No `PYTHONPATH=src` needed: `pyproject.toml` sets `pythonpath = ["src"]` under
`[tool.pytest.ini_options]`, so the `src/` layout resolves from a bare checkout.

Both counts are correct; they differ because **20 tests that skip without the
audio stack actually run with it**. Four failures hid in that gap for weeks, so
treat the first number as the real one and run it before trusting any result.

**Where the compute happens.** All heavy work runs on a single T4-class EC2
instance (4 vCPU, 15 GB). `data/processed/` and `reports/diagnostics/` are not
version-controlled and largely do not exist in a fresh checkout: code moves by
git, results move by `bash scripts/print_paper_numbers.sh` and paste-back. Long
jobs go under `screen` — three multi-hour runs were lost to shell exits.

## Reproducing the argument — the three commands differ in one flag

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

# 3. the dose: a catalogue with a known fraction of unlabelled generated audio
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \
  --per-stratum 0 --max-duration 120 --level-match \
  --n-atoms 20 --blur-sigma 5 --fit-on mixture --fit-real-frac 0.5 \
  --holdout-reals --fit-seeds 42 43 44 \
  --out-dir reports/diagnostics/nmf_sonics_curve_FINAL     # -> Figure 1
```

### Reproducing the proposed detector

Two commands per corpus. The first extracts every comb channel and the harmonic
null; the second turns them into the calibrated score and its AUC table. The null
is deterministic, so `--null-priors 0` — there are no decoy sets to draw and no
seed to record.

```bash
# FakeMusicCaps (~25 min on 4 cores); SONICS is the same with
#   --manifest data/processed/canonical_sonics_chain/canonical_manifest.csv
#   --max-duration 120 --resample-hz 15000     (and no --operator-grid)
python scripts/eval_comb_detector.py \
  --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \
  --out-dir reports/diagnostics/comb_fmc_HNULL \
  --per-stratum 0 --workers 4 --max-duration 9 --level-match \
  --resample-hz 0 --smooth-bins 5 --residual median --average db \
  --f-min 1000 --hull-clip-db 0 --span-duration 4 \
  --n-harm 2 4 --null-priors 0 --operator-grid

python scripts/derive_analytic_null.py \
  --score-csv reports/diagnostics/comb_fmc_HNULL/comb_per_track_*.csv \
  --out-csv  reports/diagnostics/comb_fmc_HNULL/analytic_null.csv
# -> comb_priormax4_hmargin is the proposed score; the AUC table names every
#    calibration variant and how many families each one inverts, and the
#    *_zeroatom.csv beside it reports what fraction of margins are exactly 0.
#    For this score that fraction is 0.0000; for the decoy form it is 0.1952.
```

Then the controls that must accompany it — a gate battery and the
content-identical test, neither of which may be skipped:

```bash
python scripts/run_confound_gate.py \
  --score-csv reports/diagnostics/comb_fmc_HNULL/analytic_null.csv \
  --score-column comb_priormax4_hmargin \
  --descriptor-csv reports/diagnostics/channel_fmc_harm/channel_per_track_canonical_profile_lvl.csv \
  --rival-column median_db__comb_strength \
  --out-dir reports/diagnostics/gate_margin_fmc

python scripts/analyze_recon_pairs.py \
  --score-csv reports/diagnostics/comb_recon_HARM2/per_track_calibrated.csv \
  --score-column comb_priormax2_margin \
  --out-dir reports/diagnostics/c10_margin_fmc
```

`--holdout-reals` is not optional. Without it the same command returns an
inverted score for reasons that have nothing to do with generated audio — that
is retraction 5, and it moved a headline by 0.58 AUC.

Every score is then gated:

```bash
python scripts/run_confound_gate.py \
  --score-csv reports/diagnostics/nmf_sonics_FULL/nmf_peak_per_track_lvl.csv \
  --score-column r_k20_sigma5_bins4779_pooled \
  --descriptor-csv <the chain-matched SONICS descriptor table, built at the SAME \
                    --max-duration as the score; see provenance ledger §H1> \
  --out-dir reports/diagnostics/gate_nmf_sonics_FULL
```

The descriptor table must be built at the duration the detector actually saw.
Building it at 25 s and gating a 120 s score made C3 read FAIL 0.6143 when the
matched configuration gives PASS 0.5516 — a retraction avoided by checking
(ledger §G4, §H1).

## Building the paper

```bash
python3 scripts/make_paper_figures.py --out-dir icassp/figs
cd icassp && latexmk -pdf paper.tex     # 4 body pages + references on page 5
```

## Publishing the project page

`docs/index.html` is self-contained — no build step, no external assets, no
network requests. To serve it:

1. Push `docs/` to the default branch (`main`).
2. **Settings → Pages → Build and deployment → Source: _Deploy from a branch_**,
   then **Branch: `main`, folder: `/docs`**, and Save.
3. It appears at `https://giannisprokopiouorfium.github.io/ai-music-dose/` after a
   minute or two — the URL printed in the paper.

`docs/index.html` also opens correctly from a local checkout by double-clicking
it, since it loads nothing over the network.

## Datasets and the method measured

* **SONICS** — Rahman et al., ICLR 2025 · [arXiv:2408.14080](https://arxiv.org/abs/2408.14080)
* **FakeMusicCaps** — Comanducci et al., *J. Imaging* 2025 · [arXiv:2409.10684](https://arxiv.org/abs/2409.10684)
* **FMA** — Defferrard et al., ISMIR 2017, for the transfer and per-genre false-positive rates
* The method measured — Afchar & Hennequin, [arXiv:2607.25530](https://arxiv.org/abs/2607.25530),
  building on [arXiv:2506.19108](https://arxiv.org/abs/2506.19108) (ISMIR 2025).
  Their result stands; what we measure is what its *protocol* contributes.

## Licence and citation

MIT, see [`LICENSE`](LICENSE). The datasets above carry their own licences and are
not redistributed here. If you use this code or its measurements, cite the paper —
[`CITATION.cff`](CITATION.cff) has the metadata.
