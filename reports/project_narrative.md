# How we got here: intrinsic dimension → normalizing flows → the transductive boundary

*Written 2026-08-29. The walkthrough version of the project. Every number traces to
`reports/provenance_ledger.md`; retracted numbers are kept visible and marked.*

---

## 0. The one-sentence version

We set out to detect AI-generated music by measuring the **geometry** of real music
and flagging anything unusual. Every version of that idea failed, for one reason.
What we have instead is a measurement of **where a published "zero-shot" detector's
performance actually comes from** — and the answer is roughly fifty unlabelled
generated tracks sitting in its dictionary.

---

## 1. Act one: intrinsic dimension (the original thesis)

**The idea.** GPTID (arXiv:2306.04723) detects AI-generated *text* by estimating the
intrinsic dimension of a sample's token-embedding cloud: human text lives on a
higher-dimensional manifold than generated text. It transfers across domains where
supervised classifiers collapse. Music is a natural next domain.

**What we built.** A port of their persistent-homology dimension estimator
(`features/phd.py`), applied to two kinds of point cloud per track: EnCodec latent
frames, and per-span comb profiles.

**What happened.** The claim does not transfer. ΔID (fake − real) is:

| generator | MusicGen | stable_audio | audioldm2 | musicldm | mustango |
|---|---|---|---|---|---|
| ΔID | −4.44 ✅ | −5.75 ✅ | **+2.55** ❌ | **+0.52** ❌ | **+0.98** ❌ |

Macro AUC 0.559 on EnCodec clouds, **0.474** on comb-profile clouds — below chance.
**The sign is generator-dependent**, so there is no threshold that works.

An earlier Phase-1 run had shown 0.608–0.684, which looked promising. Under channel
controls it fell to 0.559: **most of the original signal was the delivery chain**,
not the generator. That was the project's first hard lesson.

---

## 2. Act two: one-class normalizing flows (the longest detour)

**The idea.** Fit a density on real music only, score `−log p(x)`, flag the
unlikely. No fake labels, no generator assumptions — genuinely zero-shot.

**What we built.** RealNVP on four front ends (EnCodec latents, log-mel, linear STFT
in MusicDET's configuration, comb profiles), with banded and global priors, window
flows, symmetric-fold scoring, and every score correction in the OOD literature.

**What happened — six formulations, all inverted.**

| arm | reference | result |
|---|---|---|
| plain `−log p` | — | inverts |
| typicality | Nalisnick 1906.02994 | FMC 0.545 vs 0.617 |
| complexity compensation | Serrà ICLR 2020 | 0.604 vs 0.620 — a wash |
| background likelihood ratio | Ren 1906.02845 | **catastrophic**: 0.469 vs 0.633 |
| mixture-of-flows | — | negative |
| proxy negatives | Afchar 2501.10111 | 0.534 vs 0.640 — worse |

Three further pieces of evidence pinned the cause:

* **A capacity ladder.** Widening the pooled input made it *monotonically worse*:
  256-d 0.621 → 512-d 0.632 → 1024-d 0.531 → **2048-d 0.413**. Better density
  estimation, worse detection.
* **A likelihood decomposition.** `log p = log p_Z + log|det J|`, scored with a
  held-out refit: base density **0.5127**, log-det Jacobian **0.4416**, sum
  **0.5096**. *The flow is at chance.* An earlier run had reported 0.74 — that was
  train-on-test, and the whole gap was the leak.
* **A compression check.** FLAC bitrate: SONICS real 251,723 bits/s vs fake
  209,093. Generated music is literally more compressible.

**The single mechanism behind every failure.** *Generated audio is simpler and more
regular than real music.* So every statistic of the form "is this unusual for real
music?" ranks it as **more** real. Density, typicality, likelihood ratio, intrinsic
dimension — they all inherit the same sign error, and no reweighting of the same
likelihood escapes it. We falsified our own guiding hypothesis and recorded it.

**The rule this act produced:** *do not propose another density unless you have a
mechanism for why its sign would be right.*

---

## 3. Act three: physics instead of statistics

**The turn.** If a fitted statistic cannot get its sign right, take the sign from
the *mechanism* instead. Afchar et al. (ISMIR 2025) supply one: a transposed-
convolution decoder imprints periodic spectral peaks whose spacing is set by the
network's **strides**, not by its weights or training data — the audio instance of
the checkerboard artifact (Odena 2016; Pons et al., ICASSP 2021). More comb means
more decoder. The direction is fixed a priori and cannot be mis-fitted.

**What we built.** A training-free comb statistic (autocorrelation-in-frequency of a
peak residual) and a faithful reimplementation of Afchar & Hennequin's blurred-atom
NMF score, `r_i = ‖H_iW − H_i(W∗G)‖₂`.

**What it cost to get right.** Two source-diff audits, both after we had already
recorded results:

* Our NMF arm computed `‖x − HW‖` with a **reals-only** basis. The paper computes
  the difference between **two reconstructions** on a **pooled** basis. Ours is a
  novelty score; theirs is a *peak-energy* score. The retracted result (FMC 0.4998 /
  SONICS 0.2564) was a property of our code.
* Our comb statistic used a median envelope and a power-domain mean where they use a
  lower hull and a mean in dB. Fixing it *made things worse* — their 5 dB clip
  saturates an autocorrelation readout. The winning operator, `median_db`, is in
  **neither** codebase. Operator and readout are coupled.

**The standing rule:** *before recording a negative result, diff the implementation
against the source repository's actual operator.* `tests/test_afchar_parity.py`
now checks our lower hull index-for-index against theirs.

---

## 4. Act four: the result — where the performance actually comes from

The corrected NMF score reaches **0.9917** on FakeMusicCaps (32,960 tracks) and
**0.9389** on SONICS (59,280) with no label anywhere. The obvious question is what
it is using. The paper it comes from is titled *"Finding the noise: Zero-shot AI
Music Detection"*, and it factorises a matrix "containing real and synthetic
examples". So we varied only **which unlabelled audio the dictionary sees**:

| the dictionary is learned from | FakeMusicCaps | SONICS |
|---|---|---|
| the scored corpus itself (their protocol) | **0.9917** | **0.9389** |
| a different corpus (inductive transfer) | 0.5733 | 0.7123 |
| **real music only, held out** | **0.744 ± 0.020** | **0.382 ± 0.043** |

**The published protocol is worth 0.25–0.56 AUC**, and no fake *label* is used in
any row. It is transductive, not zero-shot.

**And then the dose.** Fitting the dictionary on a simulated catalogue holding a
known fraction of unlabelled generated audio, three seeds, held out:

| fakes in a 6,400-track SONICS dictionary | 0 | 9 | 23 | **47** | **70** |
|---|---|---|---|---|---|
| draws that flipped to working | 0/3 | 0/3 | 0/3 | **2/3** | **3/3** |
| macro AUC | 0.382 | 0.400 | 0.493 | 0.661 | **0.787** |
| seed SD | 0.043 | 0.054 | 0.137 | **0.216** | **0.0006** |

**It is a phase transition.** Per draw, Suno/chirp is either ~0.31 or ~0.98 —
nothing between. With K = 20 atoms, either enough generated audio is present for the
factorisation to spend one atom on the decoder comb, or it is not. The seed SD peaks
*at* the switch and collapses immediately after: the error bar is the transition
width.

**~50 generated tracks — 0.7% of the dictionary — is the whole difference between an
inverted detector and a working one.**

The mechanism is visible directly. The learned atoms reproduce the *measured*
decoder spacings to better than 0.05%: FMC atoms at 200.3 Hz against a measured
200.195 Hz cluster and 250.1 Hz against MusicGen's 250.0; SONICS atoms at 400.0 Hz
against chirp's 399.902.

---

## 5. What each paper contributed

| paper | what we took | what happened |
|---|---|---|
| **GPTID** 2306.04723 | intrinsic dimension as a domain-transferring signal | ported; **negative for music**, mixed-sign ΔID |
| Nalisnick, Ren, Serrà | three OOD score corrections | all implemented, all negative, one catastrophic |
| **Pons et al., ICASSP 2021**; Odena 2016 | why a transposed convolution leaves periodic peaks | the DSP mechanism the whole detector rests on |
| **Afchar et al.** 2506.19108 (ISMIR 2025) | the fakeprint and the comb | reimplemented byte-for-byte |
| **Afchar & Hennequin** 2607.25530 | the blurred-atom NMF score | the method we measure; first evaluation on FMC and SONICS |
| Afchar 2501.10111 | proxy negatives | implemented, negative (0.534) |
| **MusicDET** 2605.18072 | the closest competitor | non-reproduction documented; its benchmark is channel-labelled |
| Dugelay 2607.27454 | log-frequency invariance | `comb_log_strength` is sign-inverted; the log family is not |
| **2606.20488** (images) | *"a training-free score is not a detector until its sign is verified"* | criterion **ceded**; we supply the audio measurement (+0.23) |
| **2601.18900** RealStats (images) | two-sided real-only p-values | criterion **ceded**; we supply the four-cell ablation they do not do |
| **2604.16254** ArtifactNet | codec confound by augmentation | we measure it at corpus level instead; they do no per-genre FPR |
| 2606.08663 Probing Token Spaces | X-Codec tokens for Udio | cited; not taken — supervised there, a density here |
| 2507.10447, 2412.00571, ASVspoof 5, VoxENES | protocol and evaluation | adopted / cited |

---

## 6. Five retractions, and what each taught

| # | claim | cause | lesson |
|---|---|---|---|
| 1 | "NMF reconstruction error inverts" | wrong quantity, wrong fit set | diff against the source operator |
| 2 | "we extend the fakeprint below 16 kHz" | their code already takes `--fmin/--fmax` | read the code, not the abstract |
| 3 | SONICS 0.8468 | it was `\|AUC\|`; signed is 0.6203 | report signed AUC; `\|AUC\|` hides per-generator inversions |
| 4 | "comb_strength uses Afchar's operator" | median filter + power mean, not hull + dB mean | operator and readout are coupled |
| 5 | **"reals-only inverts to 0.167" (FMC)** | **dictionary-level train-on-test**: the fit reals *were* the scored reals | **whenever a score is read off a fitted object, check whether the fitted rows are also scored** |

Plus two plumbing defects found in the final week, both of which had already
corrupted results: `as_completed` made every budgeted fit non-reproducible (row
order changed the drawn sample despite a fixed seed), and results were written only
after a whole sweep, so three multi-hour runs computed everything and saved nothing.

**Retraction 5 is the one to tell.** It moved a headline, it was found by a
reproducibility anchor we deliberately built into a runbook, and it is the same
defect class as retraction-adjacent leak found earlier in `score_flow_terms.py`.
Twice in one project.

---

## 7. The contributions, one slide each

1. **The transductive boundary.** 0.99 → 0.74 (FMC) / 0.94 → 0.38 (SONICS), held
   out, three seeds, three dictionary learners, two corpora.
2. **The dose–response, and the phase transition.** ~50 generated tracks flip it.
   Near the boundary the detector is not merely worse but *unpredictable* — one
   generator swings 0.21 AUC across draws below 5% contamination, and 0.003 above it.
3. **The published 445-bin descriptor is aliased at 16 kHz.** Full resolution is
   worth **+0.046 / +0.102**, with a DSP mechanism.
4. **Deployment and fairness.** Precision at prevalence; **2.75×** transfer FPR on
   unseen real music; **8.46×** per-genre spread (Electronic 0.236, Folk 0.028).
5. **Two criteria transferred from image forensics and quantified in audio** —
   sign consistency and two-sided real-only scoring — with a four-cell controlled
   ablation neither source paper performs.

Plus the protocol: on the corpus the field publishes on, a single channel descriptor
scores **0.9842 signed**, and our own detector inflates by **+0.217** without the
control. We report that against ourselves.

---

## 8. What we would tell someone starting this

* A statistic whose sign is fitted will get its sign wrong on the case you care
  about. Take the sign from the mechanism.
* "Label-free" and "zero-shot" are not the same claim, and the difference is
  measurable.
* `|AUC|` is a legitimate measure of how much class information a nuisance variable
  carries, and an illegitimate measure of detector performance.
* Put reproducibility anchors — a repeated cell — in every sweep. Ours caught a
  defect that had already changed a headline.
* Run three seeds. The finding in §4 is invisible at one.
