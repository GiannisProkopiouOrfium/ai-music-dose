# Provenance ledger — every number destined for the ICASSP paper

**Built 2026-08-23 from `icassp/paper_numbers.txt`** (10.7 MB dump produced by
`scripts/print_paper_numbers.sh` on EC2, commit on `feat/musicdet-comp`), plus the
three Batch-1 runs of 2026-08-23.

Nothing here was copied from a handover. Every row was recomputed from the CSV
named in it. **Nothing in the dump was MISSING** — every file the result map
expects exists on the box.

Status: `VERIFIED` (CSV matches the text) · `CORRECTED` (both values shown) ·
`UNVERIFIED` (no CSV backs it) · `PENDING` (not yet measured).

---

## A. VERIFIED — recomputed from the CSV, matches the text exactly

| claim | value | source | n |
|---|---|---|---|
| NMF peak, FMC, full resolution | **0.99168** | `nmf_fmc_FULL/nmf_peak_auc_lvl.csv` σ=2 bins7167 | 32,960 |
| NMF peak, FMC, published 445-d | **0.94568** | same file, bins445 | 32,960 |
| **445-d penalty, FMC** | **+0.04600** | difference of the two | — |
| NMF peak, SONICS, full resolution | **0.93894** | `nmf_sonics_FULL/…` σ=5 bins4779 | 59,280 |
| NMF peak, SONICS, published 445-d | **0.83676** | same file, bins445 | 59,280 |
| **445-d penalty, SONICS** | **+0.10218** | difference of the two | — |
| FMC operating point | TPR **0.9872** @ FPR **0.0448** | `nmf_fmc_FULL/nmf_peak_operating_point_lvl.csv` | 2,677 / 2,678 reals |
| SONICS operating point | TPR **0.6957** @ FPR **0.0497** | `nmf_sonics_FULL/…` | 6,361 / 6,361 |
| chirp on full SONICS | 0.9987 / 0.9997 / 0.9984 | `nmf_sonics_FULL` σ=5 | — |
| **Reals-only dictionary inverts (FMC)** | **0.16686** | `nmf_fmc_fitreals/nmf_peak_auc_lvl.csv` | 600 reals in dict |
| per-generator inversion | audioldm2 **0.0637**, mustango 0.1000, musicldm 0.1125 | same | — |
| comb scalar, FMC | **0.9177** | `comb_fmc_FULL/comb_auc_…csv`, `median_db__comb_strength` | 32,960 |
| Two-sided **cell 1** (FMC comb) | 0.9177 → 0.9128, **−0.0049** | `fusion_fmc_twosided/fusion_auc.csv` | 32,960 |
| Two-sided **cell 2** (SONICS NMF transductive) | 0.9389 → 0.9214, **−0.0175** | `nmf_sonics_twosided/fusion_auc.csv` | 59,280 |
| Transfer FPR, comb, q=0.95 | 0.0419 → 0.1151 = **2.75×** | `fpr_fma_by_genre/false_positive_rate.csv` | 2,675 / 2,998 |
| **Per-genre FPR spread** | Electronic **0.2356**, Folk **0.0279** = **8.46×** | `fpr_fma_by_genre/false_positive_rate_by_group.csv` | 2,998 |
| Robustness, all 8 rows | baseline 0.9187 / EER 14.6% | `robustness_comb_median_db/robustness_auc_eer.csv` | 5,354 / 27,605 |
| Efficiency | **42.6 ± 4.02** tracks/s, **379.7** audio-s/s, **0 parameters** | `efficiency_comb_detector_only.json` | 60 tracks × 5 trials |
| Gates on the FMC NMF headline | C1 FAIL, C2 FAIL, **C8 SKIP**, all others PASS (incl. C9, C10) | `gate_nmf_FULL_c10/gate_…csv` | — |

### Dose–response, exact values recomputed from the CSVs

| reals in dict | 25 | 50 | 100 | 200 | 400 | 600 |
|---|---|---|---|---|---|---|
| FMC (`nmf_fmc_nreals`) | 0.4811 | 0.5000 | 0.4805 | 0.4238 | 0.2875 | **0.1679** |
| SONICS (`nmf_sonics_nreals`) | 0.3550 | 0.3293 | 0.3013 | 0.2267 | 0.2540 | **0.2044** |

---

## B. CORRECTED — the CSV disagrees with the text. Both values shown.

### B1. ⚠ Atom peakiness: the Figure-2 numbers are wrong, and half of them do not exist

| | claimed in outline §5b | **actual** |
|---|---|---|
| FakeMusicCaps | "peakiness **0.70–0.77**" | **0.2198 – 0.5998** (`reports/figures/nmf_fmc/atom_peakiness.csv`, 20 atoms) |
| SONICS | "peakiness **0.53–0.60**" | **no such file** — `reports/figures/nmf_sonics/` contains only `activations.png` and `atoms.png` |

The SONICS figure run (Aug 21 11:56) predates the version of
`visualize_nmf_space.py` that emits `atom_peakiness.csv` (FMC run, Aug 21 17:29).
So **the FMC/SONICS peakiness contrast — the quantitative half of the mechanism
figure — is unsupported in both directions.** Where "0.70–0.77" came from is
unknown; no CSV on the box contains it.

The *visual* claim (FMC atoms are periodic combs) may still hold from `atoms.png`,
but no number may be quoted until §R2.2 below re-derives both.
**This is the most serious correction in this round.**

### B2. Two-sided cell 3, now at full corpus

| | one-sided | two-sided | Δ | n |
|---|---|---|---|---|
| as reported (`fusion_sonics_twosided`) | 0.6790 | 0.7628 | **+0.0838** | 3,600 |
| **full corpus** (`fusion_sonics_FULL_rs15k_*`) | **0.6864** | **0.7546** | **+0.0682** | **59,280** |

Best one-sided feature is `comb_stat_strength` in both; best two-sided is
`comb_residual_kurtosis__abs` in both. The gain is smaller at full corpus but
unambiguous. udio one-sided → two-sided: 0.2225/0.2897 → **0.4966/0.5112**.

### B3. Cross-corpus NMF, FMC scored / SONICS dictionary, now at full corpus

| | macro | musicldm | n scored |
|---|---|---|---|
| as reported (`nmf_fmc_fitsonics`) | 0.5678 | 0.4075 ⚠ inverted | 3,600 |
| **full corpus** (`nmf_fmc_fitsonics_FULL`) | **0.5733** | **0.4028** ⚠ inverted | **32,960** |

σ=5 in both. Gate: C5/C6/C7/C11 PASS, AUC 0.5733 [0.5653, 0.5809]; C1–C4, C8–C10
SKIP. The claim (cross-corpus transfer collapses and is sign-inconsistent) holds.

### B4. ⚠ The 0.9993 protocol row is an `|AUC|` — and there is a better number

`channel_sonics_UNCONTROLLED/channel_alone_auc_canonical_lvl.csv`, macro over the
five generators:

| descriptor | macro **signed** | macro **\|AUC\|** | sign-consistent? |
|---|---|---|---|
| `frac_power_above_8k` | **0.0007** | **0.9993** | inverted on all 5 |
| `frac_power_8k_11k` | 0.0008 | 0.9992 | inverted on all 5 |
| **`hf_floor_frac`** | **0.9842** | 0.9842 | **yes — 0.9406 to 1.0000** |
| `spectral_flatness_hf` | **0.9756** | 0.9756 | **yes — 0.9177 to 1.0000** |
| `duration_s` | 0.6156 | 0.7650 | no |

**This matters twice over.** First, the paper criticises the field for reporting
`|AUC|` (contribution 3) while its own strongest protocol number *is* an `|AUC|`.
Second, and better: **we do not need it.** `hf_floor_frac` reaches **signed 0.9842,
correctly oriented on every generator** — a deployable detector built from nothing
but the delivery chain, which survives our own sign-consistency criterion.

**Recommended wording change:** lead with `hf_floor_frac` **0.9842 signed**, and
report `frac_power_above_8k` `|AUC|` 0.9993 as the *information* measure with the
distinction stated explicitly — `|AUC|` is a legitimate measure of *how much class
information a nuisance variable carries* and an illegitimate measure of *detector
performance*. Making that distinction converts an apparent inconsistency into a
sharpening of contribution 3.

### B5. The f-min sweep: signed udio, read from the CSVs for the first time

The outline inferred these from `|AUC|` tails. The signed truth (`comb_strength`):

| f_min | chirp-v2 | chirp-v3 | chirp-v3.5 | udio-120s | udio-30s | MACRO |
|---|---|---|---|---|---|---|
| 200 Hz | 0.8804 | 0.9254 | 0.8392 | **0.2283** | **0.1908** | 0.6128 |
| 500 Hz | 0.8853 | 0.9266 | 0.8513 | **0.2359** | **0.1783** | 0.6155 |
| 1000 Hz | 0.8857 | 0.9268 | 0.8553 | **0.2588** | **0.1751** | 0.6203 |

The handover reported "udio 0.772 / 0.809 at 200 Hz" — that was `|AUC|`.

**The signed result is a stronger statement than the one it replaces.** Widening
the band does not merely fail to *find* udio's comb; **udio stays inverted at every
band searched**, at 0.175–0.259. udio's residual is consistently *smoother* than
real music's, in all three analysis bands. That is a measured, three-band physical
justification for two-sided scoring — better evidence than the outline claims.

### B6. "Monotone" is an overstatement

FMC: n25 0.4811 **<** n50 0.5000. SONICS: n200 0.2267 **<** n400 0.2540. Neither
curve is monotone. The **trend** and the reported **r = −0.88 in log n** stand.
Write "decreasing, r = −0.88 in log n", never "monotone".

### B7. The NMF arm's 1.10× transfer FPR rests on n = 300

`fpr_fma_nmf/false_positive_rate.csv`: `n_threshold_reals = 300`, `n_eval_reals =
300` — against the comb arm's 2,675. The 0.067 → 0.073 ratio is a 300-real
estimate and must be reported with its n or re-run at full corpus.

---

## C. NEW — results this round produced that are not in any outline

### C1. ⚡ `comb_spacing_hz` is sign-consistent *within* each corpus and **flips between them**

| corpus | chirp/gen 1–3 | gen 4–5 | MACRO | real median spacing |
|---|---|---|---|---|
| **SONICS** @15 kHz, full | 0.5944 / 0.6501 / 0.6187 | udio 0.6793 / 0.6652 | **0.6415** — all 5 correct | 282.7 Hz |
| **FakeMusicCaps**, full | 0.3054 / 0.3150 / 0.3034 | 0.3136 / 0.2692 | **0.3013** — all 5 inverted | 474.6 Hz |

On SONICS every generator's comb is *wider*-spaced than real music; on FMC every
generator's is *narrower*. Both are internally sign-consistent, so a per-corpus
sign check passes — **and the feature is still unusable, because the sign is a
property of which decoders and which reals the corpus happens to contain.**

Mechanistically exact: spacing is `f_s / ∏ strides`, an architecture constant, and
real music's estimated spacing is a corpus statistic. Neither is fixed a priori.

**This is a stronger form of contribution 3 than the outline has**, and it is new:
it shows that *within-corpus sign consistency is necessary but not sufficient*. It
also justifies, after the fact, the project's decision to use spacing only as the
C8 mechanism check and never as a detector.

Note this is also the only feature on SONICS that is correctly oriented on udio —
worth one sentence, immediately followed by the corpus flip that disqualifies it.

### C2. The log-axis family is not uniformly inverted

On full-corpus SONICS @15 kHz: `comb_log_residual_kurtosis` **0.6883** — the best
one-sided macro of any feature there, above `comb_stat_strength`'s 0.6864. Only
`comb_log_strength` (0.3332) is inverted. The blanket claim "our log-axis variant
is sign-inverted" (outline §7, Dugelay row) should name the feature.

---

## D. PENDING — not yet measured

| # | item | why it matters | est. |
|---|---|---|---|
| P1 | **`nmf_sonics_fitfmc_FULL` never ran** — the directory does not exist; the `_onesided`/`_twosided` dirs beside it are empty shells created by the failed fusion calls | **cell 4 of the four-cell ablation**, and the paper's honest inductive SONICS number. The only cell still at n=600 | ~2–3 h |
| P2 | atom peakiness, both corpora (see B1) | the mechanism figure | ~40 min |
| P3 | chain-matched SONICS channel descriptor table | C1–C4 SKIP on **every** SONICS arm right now | ~1 h |
| P4 | second dictionary learner (sweep item R1) | turns "NMF is transductive" into "dictionary learning is transductive" | ~1 h |

---

## E. Two script defects found by this round

1. **`visualize_nmf_space.py` is broken.** It calls
   `eval_nmf_peak_detector._collect`, which gained an `args.carry_columns`
   dependency in R10.3, but its own argparse never gained the flag:
   `AttributeError: 'Namespace' object has no attribute 'carry_columns'`.
   Both figure commands died on it. One-line fix.
2. **`fuse_physics_scores.py` and `run_confound_gate.py` `mkdir` the output
   directory before reading the input**, so a missing input leaves an empty
   directory that later looks like a completed run. Two empty dirs exist now
   (`nmf_sonics_fitfmc_FULL_{one,two}sided`) and must be removed before they are
   mistaken for results.

---

## F. ROUND 2 RESULTS — 2026-08-24

### F1. ✅ The four-cell ablation is complete at matched n

| cell | inverted? | one-sided | two-sided | Δ | n | source |
|---|---|---|---|---|---|---|
| FMC, comb | no | 0.9177 | 0.9128 | −0.0049 | 32,960 | `fusion_fmc_twosided/` |
| SONICS, NMF transductive | no | 0.9389 | 0.9214 | −0.0175 | 59,280 | `nmf_sonics_twosided/` |
| SONICS, comb @15 kHz | **yes** | 0.6864 | **0.7546** | **+0.0682** | 59,280 | `fusion_sonics_FULL_rs15k_*` |
| SONICS, NMF cross-corpus | **yes** | 0.7123 | **0.7345** | **+0.0222** | 59,280 | `nmf_sonics_fitfmc_FULL_*` |

**§7b's weakest-significance concern is CLOSED.** The cross-corpus two-sided lift
was DeLong p = 0.0148 at n=600; at full corpus it is **p = 3.07e-57** (z = 15.95,
pooled AUC 0.6318 → 0.6747).

Cross-corpus macro **CORRECTED** 0.7152 (n=3,600) → **0.7123** (n=59,280), and
two-sided 0.7378 → **0.7345**. Both within noise of the small-n estimates.

### F2. ✅ NEW — the transductive boundary holds under three dictionary learners

FakeMusicCaps, 600/stratum, full resolution, σ=2:

| learner | pooled | reals-only | source |
|---|---|---|---|
| NMF | 0.9900 | **0.1679** | `nmf_fmc_nreals` n600 |
| sparse coding (L1, non-negative) | **0.9790** | **0.2974** | `dict_sparse_fmc_{pooled,reals}` |
| k-means atoms | **0.9166** | **0.3761** | `dict_kmeans_fmc_{pooled,reals}` |

The claim generalises from "NMF is transductive" to **"dictionary learning is
transductive"**. Monotone in dictionary structure: the more constrained the
learner, the harder the inversion.

### F3. ⚠ CORRECTED — the atom figures were attributed to the wrong corpora

| | outline §5b | **measured, n=1200** |
|---|---|---|
| FakeMusicCaps | "0.70–0.77, periodic combs, cut off at ~7350 Hz" | peakiness **0.154–0.616**, band runs to the full 8 kHz |
| SONICS | "0.53–0.60, unstructured spike forests" | peakiness **0.220–0.776**, band cuts at ~7350 Hz |

Verified atom-for-atom against both `atom_peakiness.csv` files and both images.
The ~7350 Hz cutoff is the **SONICS** MP3-64k chain edge; `canonical_fmc_raw16`
has no codec stage and runs to 8 kHz. R7.1 read the two figures the wrong way
round, and the peakiness ordering is the reverse of what the outline claims.

**Consequence for the narrative:** peakiness does *not* explain the FMC/SONICS
performance gap — SONICS atoms are peakier and SONICS performs worse. Any Figure-2
caption built on that correlation must be rewritten.

### F4. ⚠ CORRECTED — peakiness is the wrong statistic for the mechanism claim

`peakiness = ||W − W∗G|| / ||W||` is the **score's own sensitivity**, not evidence
of a comb. Measured on a fixture with identical spike counts:

| atom | peakiness | periodicity (`comb_strength`) | recovered spacing |
|---|---|---|---|
| periodic comb (true 84 Hz) | 0.850 | **0.988** | **84 Hz** ✓ |
| random spike forest | 0.839 | **0.073** | 1802 Hz ✗ |

Separation: peakiness 0.011, periodicity 0.915. **"The dictionary learns decoder
combs" has never been tested.** `visualize_nmf_space.py` now emits `periodicity`
and `spacing_hz` per atom via `comb_strength`; four tests pin the distinction
(`TestPeakinessIsNotEvidenceOfACombButPeriodicityIs`). Re-run pending.

### F5. ⚠ NEW THREAT — `duration_s` reaches 0.7650 on the CHAIN-MATCHED SONICS corpus

`channel_sonics_chain_2026_08_23`: bandwidth family best **0.6000**
(`hf_floor_frac`), level/silence family best **0.7650** (`duration_s`), script
verdict *"0.70–0.90 a material confound"*.

**0.7650 exceeds our SONICS detector** (0.6864 one-sided / 0.7546 two-sided).
Chain-matching fixes sample rate and codec; it does not touch duration. **Every
SONICS gate to date reported `[SKIP] C3`**, so this has never been tested against
any SONICS number. Gating is Round 3 item 2 and it is the highest-priority command
outstanding.

### F6. ✅ The protocol table, both halves, sign-consistent

| | uncontrolled | chain-matched |
|---|---|---|
| `hf_floor_frac` **signed** | **0.9842** (0.9406–1.0000, all five correct) | **0.6000** |
| `frac_power_above_8k` \|AUC\| | 0.9993 | 0.5795 |
| our detector | 0.8374 | 0.6311 |

### F7. Gate status on cell 4

`gate_nmf_sonics_fitfmc_FULL`: C5/C6/C7/C11 PASS (AUC 0.6318 [0.6271, 0.6368],
59,280 rows, smallest stratum 1,568); **C1–C4, C8, C9, C10 SKIP** — seven
untested confounds. Round 3 item 2 closes C1–C4.

---

## G. ROUND 3 RESULTS — 2026-08-25

### G1. ⚡ THE CONTAMINATION CURVE — the paper's headline figure

Dictionary fitted on a simulated catalogue holding a known fraction of *unlabelled*
generated audio. Labels build the corpus; the method never sees one.
`nmf_{fmc,sonics}_contamination/nmf_peak_auc_lvl.csv`, 600/stratum, full resolution.

| contamination of the fit set | 0% | 0.99% | 2.44% | 4.76% | 9.09% | 20% | 33.3% | 55.6% | 71.4% | 83.3% |
|---|---|---|---|---|---|---|---|---|---|---|
| **FakeMusicCaps** (σ=2) | 0.158 | 0.186 | **0.505** | 0.698 | 0.809 | 0.896 | 0.916 | 0.980 | 0.987 | **0.991** |
| **SONICS** (σ=5) | 0.191 | 0.276 | **0.665** | 0.675 | 0.691 | 0.721 | 0.822 | 0.871 | 0.911 | **0.935** |

**Strictly monotone on both corpora** — unlike the reals-count sweep, which was not.

**Break-even (AUC = 0.5): 2.42% on FMC, 1.82% on SONICS.** Below ~2% unlabelled AI
content the detector is *actively inverted*; above ~20% it is essentially the
published number. Endpoints reproduce the independently measured values (FMC 0% →
0.158 vs `nmf_fmc_fitreals` 0.167; 83.3% → 0.991 vs pooled 0.990. SONICS 0% → 0.191
vs `nmf_sonics_nreals` n600 0.204; 83.3% → 0.935 vs `nmf_sonics_FULL` 0.939).

This is the deployment statement the field does not have: *a label-free artifact
detector needs roughly one part in fifty of its unlabelled fitting corpus to be
generated before it works at all.*

### G2. ✅ THE MECHANISM IS CONFIRMED — atom spacings match the measured decoder combs

`reports/figures/nmf_{fmc,sonics}_periodicity/atom_peakiness.csv`, n=1200:

| corpus | atoms near the **measured** decoder spacing | periodicity |
|---|---|---|
| **FMC** | atoms 15, 12, 5 at **200.3 Hz** vs measured audioldm2 / musicldm / mustango **200.195 Hz** (0.05%) | 0.78, 0.86, 0.68 |
| **FMC** | atoms 13, 6, 18 at **250.1 Hz** vs measured MusicGen **250.0 Hz** (0.04%) | 0.87, 0.81, 0.74 |
| **SONICS** | atoms 16, 4, 1, 13, 14, 2 at **400.0 Hz** vs measured chirp v2/v3/v3.5 **399.902 Hz** (0.03%) | 0.71, 0.59, 0.47, 0.40, 0.40, 0.41 |

**The dictionary learns atoms whose periodic spacing reproduces the decoder combs
to better than 0.05%, from unlabelled audio.** That is the mechanism, shown
quantitatively rather than asserted — and it is exactly why removing the generated
audio inverts the score.

**And it confirms F4 on real data**: `corr(peakiness, periodicity)` is **+0.035**
on FMC and **−0.309** on SONICS. The two statistics are unrelated. Peakiness could
never have supported this claim.

The corpus contrast now runs the correct way: FMC mean periodicity **0.700**
(AUC 0.992) vs SONICS **0.464** (AUC 0.939). Peakiness ran the opposite way.

### G3. ✅ Three learners × two corpora — six cells, all inverted on reals-only

| learner | FMC pooled | FMC reals-only | SONICS pooled | SONICS reals-only |
|---|---|---|---|---|
| NMF | 0.9900 | **0.1679** | 0.9352 | **0.2044** |
| sparse coding | 0.9790 | **0.2974** | 0.8852 | **0.3124** |
| k-means atoms | 0.9166 | **0.3761** | 0.7535 | **0.2864** |

"Dictionary learning is transductive" is now a two-corpus, three-objective result.

### G4. ⚠ C2 AND C3 **FAIL** ON EVERY SONICS NUMBER — including the 0.93894 headline

With `channel_sonics_chain_2026_08_23` as the descriptor table:

| gate | verdict | value |
|---|---|---|
| C1 channel-alone | **PASS** ✅ *(new — the bandwidth confound is cleared on SONICS)* | `hf_floor_frac` 0.5568 |
| **C2 level** | **FAIL** | `peak_dbfs` **0.6322** (threshold 0.60) |
| **C3 duration** | **FAIL** | duration AUC **0.6143**; full-length fraction gap 0.016 (within the 0.05 max) |
| C4 silence | PASS | 0.5422 |
| C9 harmonicity | PASS | within-real ≤ 0.254 |
| C10 | **SKIP** | no `--recon-csv` on any SONICS arm |

Identical across `comb_strength`, `comb_stat_strength`, `comb_residual_kurtosis`,
`nmf …_pooled` (0.9388) and `nmf …_external_ext` (0.6318) — as expected, since the
descriptors are a property of the corpus.

> ⚠ **C3 is measured against the wrong truncation.** The descriptor table was built
> with `--max-duration 25`; every score it is gated against was computed at
> `--max-duration 120`. So `duration_s` there largely encodes *"is this track
> shorter than 25 s"*, not the duration the detector actually saw. **C3 must be
> re-run against a 120 s descriptor table before it is reported as a failure** —
> this is exactly the class of mismatch that produced four earlier retractions.

C2 is real and survives `--level-match`: level matching sets a reference level, and
`peak_dbfs` still separates at 0.6322, i.e. the residual difference is crest factor.
Note FMC fails C2 too (0.6207), so this is a consistent, modest, reportable
limitation on both corpora rather than a SONICS-specific defect.

---

## H. ROUND 4 RESULTS — 2026-08-26. One gate cleared, one number in serious doubt.

### H1. ✅ C3 was a configuration mismatch, as suspected — CLEARED

Rebuilding the descriptor at the duration the detector actually used:

| descriptor built at | C1 channel | C2 level | **C3 duration** |
|---|---|---|---|
| `--max-duration 25` (mismatched) | PASS 0.5568 | FAIL 0.6322 | **FAIL 0.6143** |
| **`--max-duration 120`** (matches the score) | **FAIL 0.6082** | FAIL 0.6202 | **PASS 0.5516** |

**C3 PASSES at the correct configuration** (duration AUC 0.5516, full-length gap
0.010 against a 0.05 max). The failure was the mismatch, exactly as flagged — a
retraction avoided.

C1 flips the other way and now fails marginally at 0.6082. **The honest SONICS gate
status, at the matched configuration, is C1 FAIL 0.6082 / C2 FAIL 0.6202, everything
else PASS except C8 (SKIP for the NMF arm) and C10.** That is the *same* picture as
FMC (C1 0.6284, C2 0.6207) — both corpora fail the same two gates, marginally, and
both have bandwidth twins as the C1 mitigation (FMC Δ −0.0096, SONICS Δ −0.0025).

**A note that belongs in the paper.** `diagnose_channel_confound.py` reports
`duration_s` at macro **|AUC| 0.8266** while the gate's signed pooled AUC on the
same data is **0.5516**. The descriptor is sign-inconsistent across generators —
udio and chirp durations differ in opposite directions — so its `|AUC|` overstates
the confound by 0.28. That is contribution 3's criterion applied to a *nuisance*
variable, and it is a second, independent demonstration of the same point.

### H2. ⚠⚠ THE REALS-ONLY NUMBER REVERSES AT FULL CORPUS — contribution 1's headline is in doubt

Same script, same flags, only `--per-stratum` changed:

| contamination | 600/stratum (3,600 scored) | **FULL corpus (32,960 scored)** |
|---|---|---|
| **0% (reals-only)** | **0.1580 — inverted** | **0.6099 — correctly oriented** |
| 1% | 0.1860 | 0.7964 |
| 2.4–2.5% | 0.5050 | 0.9527 |
| 9% | 0.8086 | 0.9736 |
| 33% | 0.9159 | 0.9909 |
| 83% | 0.9909 | 0.9917 |

**A swing of +0.45 and a sign change on the number contribution 1 quotes as
"0.167 — inverted".** Against the published dose–response (25 → 600 reals going
0.481 → 0.168, "the better you model real music, the worse the detector"), 5,355
reals gives **0.610** — the trend reverses.

**Two variables moved at once** — dictionary size (600 → 5,355 reals) *and* scored
set (3,600 → 32,960). They must be disentangled before either number is written.

**What survives regardless:** the transductive boundary itself. At full corpus it is
**pooled 0.9917 vs reals-only 0.6099 — a drop of 0.38**, and the contamination curve
is still monotone across the whole range. The *boundary* is intact; what is in doubt
is the word **"inverts"**, the 0.167 figure, and the phrase *"the better you model
real music, the worse the detector"*.

**Working hypothesis, to be tested not assumed.** A 600-sample dictionary in 7,167
dimensions is badly under-determined; the inversion may be a small-sample property
of the *dictionary estimate* rather than of modelling real music well. If so, the
dose–response measures estimation error, not the mechanism it is claimed to. The
test is in Round 5 §1 and it includes a 445-bin arm, because 600 samples in 445
dimensions is far better determined — **if the inversion disappears at 445 bins it
is a sample-to-dimension effect and the claim must be rewritten.**

**Treat this as a fifth retraction candidate until measured.**

### H3. ⚠ C10 on SONICS scored the FLOW again, and it reproduces the known negative

`recon_control_sonics/reconstruction_auc.csv`: EnCodec-24 kbps **0.9731**,
Griffin-Lim-128 **0.9997**, Griffin-Lim-256 **0.9981**. Griffin-Lim fires *harder*
than EnCodec, i.e. **no deconvolution specificity** — reproducing the claim
withdrawn on 2026-08-18 for the flow on SONICS.

This does **not** close C10 for the reported detectors, because it scored
`flow_combprint.pt`, not `comb_strength`. The FMC C10 with the *reported* detector
(EnCodec 96.7–98.0% of content-identical pairs vs Griffin-Lim 47.3–50.0%, median
shift +0.11…+0.15 vs ~0.0000) stands and is the one to report; the SONICS flow
result is reported beside it as the contrary evidence, per the honesty discipline.

`reconstruction_scores.csv` carries no `path` column, so the audio cannot be
re-scored from it. `build_reconstruction_control.py --save-audio-dir` writes
`recon_audio_manifest.csv`; that is the route to closing C10 on SONICS with the
reported detector.

### H4. The 25 s duration twin (no longer needed for C3, still informative)

`comb_sonics_FULL_rs15k_dur25`, |AUC| macro: `comb_strength` 0.8410,
`comb_stat_strength` 0.8289. Spacing cluster holds exactly — chirp v2/v3/v3.5 all at
**399.902 Hz**, udio at 651.9 / 596.2, real music **354.5 Hz**. Since C3 passes at
the matched configuration, this twin is supporting evidence rather than a required
mitigation. **Its signed macro was truncated from the paste and must be read from
the CSV before use.**

---

## I. ROUND 5 — 2026-08-27. ⚠ RETRACTION #5, with its mechanism identified.

### I1. ⚠⚠ RETRACTED: "the reals-only dictionary inverts to 0.1669 on FakeMusicCaps"

**Both values, per the honesty discipline:**

| | value | protocol |
|---|---|---|
| as published | **0.1669 / 0.1679 — "inverted"** | dictionary of 600 reals; **those 600 were 100% of the scored reals** (3,600-track subsample) |
| **corrected** | **0.5690 (n=600) … 0.5940 (n=5,355)** | same dictionary sizes; scored set = the **full 32,960** corpus |

**Mechanism — a train-on-test leak at the dictionary level.** `r_i` is read off a
reconstruction through learned atoms. A track that was *in* the fit set is
reconstructed through atoms it helped create, so its `r_i` is inflated. When the fit
reals are also the scored reals, real music scores systematically higher and the AUC
inverts for reasons unrelated to AI music.

Reproduced on a symmetric synthetic fixture, dictionary fixed at 60 reals:

| fit reals as a share of scored reals | 100% | 30% | 10% | 3% | held out |
|---|---|---|---|---|---|
| AUC | **0.0000** | 0.2368 | 0.3195 | 0.3549 | **0.3205** |

Same defect class as the `score_flow_terms.py` train-on-test caught in R2.3, one
level down. Pinned by `TestDictionaryLevelTrainOnTestLeak` (2 tests).
`--holdout-reals` now excludes the fit rows from evaluation.

**Also retracted: the FMC dose–response** (25→600 reals: 0.481 → 0.168) and the
phrase *"the better you model real music, the worse the detector"*. At the full
scored set the FMC trend is **flat-to-rising** at full resolution (0.500 → 0.594)
and only declines at the published 445-d (0.537 → 0.445). What the old curve
measured was **the growing overlap between fit and scored reals**, not the quality
of the real-music model.

### I2. ✅ The SONICS inversion is REAL and survives everything

Scored set fixed at 59,280, dictionary swept 25 → 12,722 reals, both dimensionalities:

| n_reals | 25 | 100 | 600 | 2400 | 6361 | 12722 |
|---|---|---|---|---|---|---|
| bins4779 | 0.359 | 0.323 | **0.335** | 0.328 | 0.364 | 0.342 |
| bins445 | — | — | — | 0.310 | 0.294 | 0.297 |

**Inverted at every dictionary size and both dimensionalities (0.29–0.40)** —
including n=600, where only 4.7% of the scored reals are in the fit set. The
synthetic fixture agrees: the true held-out inversion there is ~0.32 while the
leak drives it to 0.000. **SONICS inverts; FMC does not.**

### I3. What contribution 1 becomes — still large, now corpus-differentiated

| dictionary | FakeMusicCaps | SONICS |
|---|---|---|
| **pooled** (the published transductive protocol) | **0.9917** | **0.9389** |
| cross-corpus (inductive) | 0.5733 | 0.7123 |
| **reals only** | **≈0.59 — chance** | **≈0.34 — inverted** |
| **collapse** | **−0.40** | **−0.60** |

The boundary is intact and larger on SONICS than previously claimed. The honest
sentence is *"removing the generated audio from the factorisation costs 0.40–0.60
AUC and takes the detector to chance (FMC) or below it (SONICS)"* — and the fact
that the **sign** of the collapse is corpus-dependent is itself an instance of the
paper's own sign-consistency theme.

### I4. ⚠ The FMC contamination curve's 0% cell inherits the same correction

The full-corpus FMC curve starts at **0.610**, not 0.158, so **there is no
break-even on FMC** — it never goes below chance. The useful statement replaces it
and is stronger: **the first 2.5% of unlabelled generated audio buys 0.610 → 0.953,
which is 85% of the total 0.610 → 0.992 range.** The 600/stratum curve (0.158 →
0.991, break-even 2.42%) is leak-contaminated at its low end and must not be used.

SONICS full-corpus contamination has **not been produced** —
`nmf_sonics_contamination_FULL/nmf_peak_auc_lvl.csv` does not exist; the job
reached the 59,280 × 4,779 factorisation stage and stopped. Most likely OOM (15 GB
box; ten dictionary fits over a 2.3 GB float64 matrix, each with clipped copies).

### I5. ✅ C10 on SONICS is now runnable with the reported detector

`build_reconstruction_control.py --save-audio-dir` exported **1,800 variant WAVs
across 300 content-identical pairs** and wrote
`recon_control_sonics_audio/wav/recon_audio_manifest.csv` with columns
`track_id,pair_id,variant,label,algorithm,path,canonical_path`. The script printed
the exact scoring command. The flow arm on the same pairs again shows **no
deconvolution specificity** (min reconstruction AUC 0.9618, Griffin-Lim above
EnCodec), reproducing the 2026-08-18 withdrawal — report both.

---

## J. ROUND 6 — 2026-08-28. The leak-free numbers, and a second instance of the same leak.

### J1. ✅ Reals-only, HELD OUT — the deployer protocol, measured properly

`nmf_{fmc,sonics}_reals_heldout/nmf_peak_auc_lvl.csv`, full corpus, dictionary rows
excluded from evaluation.

| n_reals in dictionary | FMC bins7167 | FMC bins445 | SONICS bins4779 | SONICS bins445 |
|---|---|---|---|---|
| 25 | 0.5064 | 0.5244 | 0.3658 | 0.3728 |
| 100 | 0.5222 | 0.5294 | 0.3177 | 0.3856 |
| 600 | 0.5587 | 0.5313 | **0.4607** | 0.3842 |
| 2400 | **0.7345** | 0.5819 | 0.3885 | 0.3588 |

**FMC rises with dictionary size (0.506 → 0.735); SONICS stays inverted (0.318 –
0.461) at every budget and both dimensionalities.**

⚠ **The same quantity has now had three values under three protocols** — this is
the number to state carefully and only once:

| value | protocol | status |
|---|---|---|
| 0.167 | leaked, 3,600-track subsample | **RETRACTED** (§I1) |
| 0.594 | full corpus, not held out | superseded |
| **0.735** (FMC) / **0.461** (SONICS) | **full corpus, held out** | **the number to report** |

### J2. Contribution 1, leak-free and final

| dictionary | FMC | SONICS |
|---|---|---|
| **pooled with the scored corpus** (the published protocol) | **0.9917** | **0.9389** |
| cross-corpus (inductive) | 0.5733 | 0.7123 |
| **reals only, held out** | **0.7345** | **0.4607** |
| **gap** | **−0.257** | **−0.478** |

The claim: *the published protocol beats what a deployer can actually do by
0.26–0.48 AUC.* Still large, now leak-free and held-out. Two wording changes are
forced: **"inverts" is a SONICS-only statement**, and the FMC dose–response runs
the **opposite** way to what was published (more real music makes the FMC detector
*better*, 0.506 → 0.735, presumably because 16 kHz real music already carries
periodic structure the atoms can learn).

### J3. ⚠ The contamination curve carries the SAME leak — do not ship it as measured

`--fit-on mixture` puts **every** real in the fit set, and every real is then
scored. Overlap is 100% at every point of the curve. Measured cost of that leak at
only 45% overlap on FMC: **0.5813 (leaked) vs 0.7345 (held out) = 0.15 AUC.**

**Fixed:** `--fit-real-frac` added. At 0.5, with `--holdout-reals`, half the
catalogue fits the dictionary and the other half is scored. Both FMC and SONICS
curves must be re-measured that way before Figure 1 is drawn.

### J4. ⚠ C10 on SONICS produced no comparison — wrong analysis, not a wrong result

`eval_comb_detector.py` correctly reported *"ALL-REAL CORPUS (1800 tracks): no AUC
is defined"* — every track in a reconstruction control is real music, so a
corpus-level AUC does not exist. The FMC C10 figure was a **within-pair** test, and
that is the only form in which C10 means anything.

**Fixed:** `scripts/analyze_recon_pairs.py` computes it — per variant, the share of
content-identical pairs where the reconstruction scores above its own source, the
median within-pair delta, and a Wilcoxon signed-rank test. Smoke-tested on a
fixture with a known answer. The scores are already on disk
(`comb_recon_control_sonics/comb_per_track_lvl_sm5_median_db_f1000_noclip.csv`,
1,800 rows, `variant` and `pair_id` carried), so this costs seconds, not a re-run.

### J5. The SONICS contamination halves did not complete

Both `_lo` and `_hi` reached *"pooled fakeprint matrix: (59280, 4779)"* and stopped
with empty output directories. `dmesg` shows **no OOM kill**, so the likeliest cause
is the foreground process ending with the shell session — neither was run under
`screen`. Re-run detached.

---

## K. ROUND 7 — 2026-08-28. C10 closes on both corpora. The knee is unresolved.

### K1. ✅ C10 PASSES for the reported detector on BOTH corpora — the last SKIP is closed

Within content-identical pairs, `comb_strength`, 299–300 pairs per variant:

| variant | FMC frac_above | FMC median Δ | **SONICS frac_above** | **SONICS median Δ** |
|---|---|---|---|---|
| encodec_3kbps | 0.980 | +0.153 | **0.9498** | **+0.1896** |
| encodec_6kbps | 0.967 | +0.113 | **0.9465** | **+0.1382** |
| encodec_24kbps | 0.980 | +0.117 | **0.9532** | **+0.1408** |
| griffinlim_128mel | 0.473 | −0.0005 | **0.5084** | **+0.00008** |
| griffinlim_256mel | 0.500 | 0.0000 | **0.4682** | **−0.0009** |

Source: `c10_sonics/recon_pairs.csv`. Wilcoxon p < 2.1e-45 for every EnCodec variant.

**A transposed-convolution decoder fires above its own source on 94.7–98.0% of pairs
across both corpora; phase-only reconstruction sits at chance with a zero median
shift.** Same content, same genre, every covariate held constant by construction.
**The detector responds to deconvolution, not to re-synthesis in general** — and it
is now a two-corpus result.

Report beside it, per the honesty discipline: the **flow** on SONICS shows the
opposite ordering (Griffin-Lim 0.9997/0.9981 above EnCodec 0.9731), reproducing the
2026-08-18 withdrawal. The claim is about the reported detector, not the flow.

### K2. ⚡ SONICS contamination, held out — and the knee is narrower than we sampled

`nmf_sonics_contamination_HELDOUT_lo`, `--fit-real-frac 0.5 --holdout-reals`,
6,361 reals scored at every point:

| contamination | 0% | 1.44% | 3.53% | 6.83% | 12.8% |
|---|---|---|---|---|---|
| **MACRO** | **0.3417** | **0.7877** | 0.7950 | 0.8617 | **0.9006** |
| TPR @ 5% FPR | 0.0072 | **0.4689** | 0.4769 | 0.4795 | 0.5098 |
| chirp v2/v3/v3.5 | 0.27/0.21/0.24 | **0.97/1.00/0.98** | — | — | 0.99/1.00/0.99 |

**+0.446 AUC from 1.44% contamination**, and chirp goes from inverted to solved.
The 0% cell (0.3417) matches the held-out reals-only prediction — sanity check PASSES.

⚠ **The entire transition happens between 0% and 1.44%, and there is no measured
point inside it.** Figure 1's most important feature is currently a straight line
between two points. Round 8 samples it.

### K3. ⚠ The FMC held-out curve computed but wrote nothing

The log shows all nine configs completing (last at 23:24:51, `mix83.8pct`), then the
output directory is empty. No OOM in `dmesg`; the run was not under `screen` and the
shell session ended before the CSVs were written. **Re-run detached — the compute is
~30 min and the numbers were never persisted.**

The SONICS `_hi` half reached `mix78.5pct` at 09:27 and its table was not in the
paste; its CSV needs to be sent.

---

## L. ROUND 8 — 2026-08-28. Two code defects found. Both affect published cells.

### L1. ⚠⚠ Row order was worker-completion order — every budgeted fit was non-reproducible

`_collect` gathered results with `as_completed(futures)`, which yields in **worker
completion order**, not submission order. Row order in `X` therefore changed between
runs, so `rng.choice(real_rows, size=budget)` drew a **different set of tracks** each
time despite `--seed 42` being fixed.

**Caught by the reproducibility anchor built into Round 8.** The 0%-contamination
cell of the SONICS curve, an identical configuration run twice:

| run | 0% cell | 1.44% cell |
|---|---|---|
| `_lo` | **0.3417** | 0.7877 |
| `KNEE` | **0.3892** | 0.7894 |

The post-transition cell reproduces to 0.002; the pre-transition cell moves by
**0.047**. On a synthetic fixture, changing *only* the row order moved a held-out
AUC across **0.343 → 0.667**.

**Scope.** Full-corpus results are unaffected — `pooled` and `external` use every
row, and AUC is a rank statistic over the same set. **Affected: every budgeted fit**
— `--n-reals` (the dose–response, the held-out reals-only numbers) and
`--fit-real-frac` (the contamination curves).

**Fixed:** rows are sorted by `track_id` after collection, making order a
deterministic function of the manifest.

**Consequence for the paper.** Determinism is necessary but not sufficient: which
reals you draw still matters, so a single seed is a point estimate with unknown
spread. **`--fit-seeds` added** — sweeps the draw within one extraction pass, so
three seeds cost three NMF fits rather than three corpus passes. Every budgeted cell
in the paper is to be reported as **mean ± SD over ≥3 seeds**.

Note the spread is structured, not uniform: post-transition cells are stable
(0.787–0.792 across draws) while pre-transition cells vary (0.342–0.404). Once
comb-like atoms are in the dictionary the score is robust; without them it is
driven by whatever the draw happened to contain. That is worth one sentence.

### L2. ⚠ Results were written only after the whole sweep — three runs lost everything

`nmf_fmc_contamination_HELDOUT` (twice) and `nmf_sonics_contamination_HELDOUT_hi`
each computed **every** configuration and wrote **nothing**: the CSVs were saved
only after the full loop, and the process did not survive to that point. No OOM in
`dmesg`; the runs were not detached. ~2 hours of compute lost three times.

**Fixed:** the AUC and operating-point CSVs are now written after **every**
configuration. A sweep that dies on its last cell keeps the ones before it.

**Also reduced peak memory:** a fakeprint is `clip(curve − hull, 0, None)/max`, i.e.
already non-negative, so `np.clip(X, 0, None)` was a wasted full copy — 1.9 GB on
the 32,960 × 7,167 FMC matrix, on a 15 GB box that also holds `X` and the collected
profiles. It now copies only when the data would actually change.

### L3. The knee, as measured (to be re-run under the fixes)

`nmf_sonics_contamination_KNEE`, held out, single seed:

| contamination | 0% | 0.141% | 0.36% | **0.733%** | 1.09% | 1.44% |
|---|---|---|---|---|---|---|
| MACRO | 0.389 | 0.404 | 0.369 | **0.787** | 0.792 | 0.789 |
| TPR @ 5% FPR | 0.009 | 0.009 | 0.010 | **0.447** | 0.476 | 0.476 |
| chirp v2 | 0.371 | 0.390 | 0.303 | **0.960** | 0.969 | 0.968 |

**The transition is a step, not a ramp**: everything happens between 0.36% and
0.73% — roughly 140 to 280 generated tracks in a 6,400-track fit set. Below it the
detector is inverted and its TPR at a 5% FPR is 1%; above it chirp is essentially
solved and TPR is 45%.

The step (+0.42) is an order of magnitude larger than the seed spread (0.047), so
the finding is not an artefact of L1 — but **the exact location and the values must
be re-measured with the fix and multi-seed error bars before this becomes Figure 1.**

---

## M. ROUND 9 — 2026-08-28. The final curve, with error bars. FMC complete.

### M1. ✅ FakeMusicCaps contamination curve — held out, 3 seeds, full corpus

`nmf_fmc_curve_FINAL/nmf_peak_auc_lvl.csv`. Dictionary = half the real catalogue
plus a known fraction of unlabelled generated audio; the other half of the reals
and all unused fakes are scored.

| contamination | mean AUC | SD | mean TPR @ 5% FPR |
|---|---|---|---|
| **0%** | **0.7444** | 0.0203 | 0.278 |
| 0.52% | 0.8357 | 0.0180 | 0.472 |
| 1.03% | 0.8749 | 0.0142 | 0.597 |
| 2.01% | 0.9401 | 0.0178 | 0.790 |
| 4.90% | 0.9771 | 0.0017 | 0.927 |
| 17.1% | 0.9873 | 0.0008 | 0.970 |
| 34% | 0.9917 | 0.0006 | 0.986 |
| 72% | 0.9919 | 0.0004 | 0.989 |

Monotone, tight error bars, and the 72% cell reproduces the pooled full-corpus
headline (0.9917) exactly — an independent confirmation of the headline number by a
different code path.

### M2. Contribution 1 — final, with error bars

| | FMC | SONICS |
|---|---|---|
| **pooled** (published protocol) | **0.9917** | **0.9389** |
| **reals only, held out** | **0.744 ± 0.020** | **0.382 ± 0.043** `[partial]` |
| **gap** | **0.248** | **0.557** |

**FMC does not invert at any dictionary composition** — the honest FMC statement is
*"a deployer with only real music gets 0.74 where the published protocol reports
0.99"*. **SONICS inverts** (0.382 ± 0.043 at 0%). The corpus-dependent sign is now
a stable, multi-seed finding rather than a single-draw curiosity.

### M3. ⚡ NEW — the low-dose regime is UNSTABLE, not merely weaker

Per-generator spread across the three draws, FMC `stable_audio_open`:

| contamination | 0% | 0.52% | 1.03% | 2.01% | **4.9%** | 17.1% |
|---|---|---|---|---|---|---|
| spread across seeds | 0.125 | **0.206** | 0.167 | **0.209** | **0.003** | 0.001 |

Below ~5% contamination a single generator swings **0.21 AUC** depending on which
few hundred tracks happen to be in the dictionary; above 5% the same generator is
stable to 0.003. **A deployer near the boundary does not merely get a worse
detector, they get an unpredictable one.** That is a deployment finding, it costs
one sentence, and it is only visible because the cells were run at three seeds.

### M4. ⚠ Two protocol notes

* **The 100%-of-fakes cell is degenerate under hold-out.** `--fake-frac 1.0` puts
  every fake in the dictionary and then holds them all out, leaving only reals to
  score. It silently emitted an empty AUC row and a NaN TPR. **Fixed:** the cell is
  now skipped with a warning. The held-out curve simply has no 100% point, and that
  is a property of the design rather than a missing measurement.
* The SONICS run was still in progress at the paste; the partial values are
  0% = 0.382 ± 0.043, 0.141% = 0.400 ± 0.054, 0.36% = 0.568 ± 0.062 (two seeds).
  With the row-order fix the transition begins **earlier and more gradually** than
  the single-draw KNEE run suggested — that run's apparent cliff between 0.36% and
  0.73% was partly the sampling defect.

---

## N. ROUND 10 — 2026-08-29. The SONICS curve completes. It is a PHASE TRANSITION.

`nmf_sonics_curve_FINAL/nmf_peak_auc_lvl.csv`, held out, `--fit-real-frac 0.5`,
3 seeds, 59,280 tracks, 6,361 reals scored at every point.

| contam % | **fakes in the dictionary** | s42 | s43 | s44 | **mean** | **SD** | chirp mean |
|---|---|---|---|---|---|---|---|
| 0 | **0** | 0.402 | 0.411 | 0.333 | **0.382** | 0.043 | 0.310 |
| 0.141 | **9** | 0.437 | 0.425 | 0.338 | 0.400 | 0.054 | 0.342 |
| 0.36 | **23** | 0.611 | 0.524 | 0.344 | 0.493 | 0.137 | 0.493 |
| **0.733** | **47** | **0.785** | **0.786** | **0.411** | 0.661 | **0.216** | 0.774 |
| **1.09** | **70** | 0.787 | 0.786 | 0.788 | **0.787** | **0.0006** | **0.984** |
| 1.44 | 93 | 0.787 | 0.790 | 0.787 | 0.788 | 0.002 | 0.986 |
| 3.53 | 233 | 0.796 | 0.794 | 0.790 | 0.793 | 0.003 | 0.987 |
| 12.8 | 931 | 0.899 | 0.903 | 0.900 | 0.901 | 0.002 | 0.992 |
| 26.8 | 2,328 | 0.913 | 0.915 | 0.914 | 0.914 | 0.001 | 0.996 |
| 64.7 | 11,640 | 0.931 | 0.933 | 0.931 | **0.932** | 0.001 | 0.998 |

### ⚡ N1. It is a SWITCH, not a ramp — and this is the paper's best result

Per draw, chirp is either **~0.31** (no comb atom in the dictionary) or **~0.98**
(one is) — **nothing in between**. The mean curve's apparent smoothness between
0.36% and 1.09% is an artefact of averaging seeds that are on opposite sides:

| fakes in the dictionary | 9 | 23 | **47** | **70** |
|---|---|---|---|---|
| draws that flipped | 0 / 3 | 0 / 3 | **2 / 3** | **3 / 3** |

**The seed SD peaks at exactly the switch — 0.216 at 47 fakes — and collapses to
0.0006 at 70.** The error bar *is* the transition width. Running three seeds is
what made this visible at all; a single-seed curve would have shown a smooth ramp
and the finding would have been missed.

**The quotable sentence: ~50 generated tracks in a 6,400-track dictionary — 0.7% —
is the whole difference between an inverted detector and a working one.**

Mechanism: K = 20 atoms. Either enough generated audio is present for the
factorisation to spend one atom on the decoder comb, or it is not. Below threshold
the dictionary is entirely real-music atoms and a comb projects onto them poorly,
so fakes score *low* and the AUC inverts. That is the transductive boundary made
mechanical, and it is consistent with §G2 (the atoms reproduce decoder spacings to
better than 0.05%).

### N2. FakeMusicCaps, for contrast — smooth, because it never inverts

| contam % | 0 | 0.52 | 1.03 | 2.01 | 4.9 | 17.1 | 34 | 72 |
|---|---|---|---|---|---|---|---|---|
| fakes in dict | 0 | 14 | 28 | 55 | 138 | 552 | 1,380 | 6,901 |
| mean AUC | 0.744 | 0.836 | 0.875 | 0.940 | 0.977 | 0.987 | 0.992 | 0.992 |
| SD | 0.020 | 0.018 | 0.014 | 0.018 | 0.002 | 0.001 | 0.001 | 0.000 |

FMC starts at 0.744 — already usable — so there is no switch to observe, only a
smooth climb. **The two corpora show the two regimes**, which is a better pair of
panels for Figure 1 than either alone.

### N3. Contribution 1 — FINAL

| | FakeMusicCaps | SONICS |
|---|---|---|
| **pooled** (the published protocol) | **0.9917** | **0.9389** |
| cross-corpus (inductive) | 0.5733 | 0.7123 |
| **reals only, held out, 3 seeds** | **0.744 ± 0.020** | **0.382 ± 0.043** |
| **gap** | **0.248** | **0.557** |

The 72%/64.7% cells reproduce the pooled headlines (0.9919 vs 0.9917; 0.932 vs
0.939) through an independent code path.

**THE EXPERIMENTAL PROGRAMME IS CLOSED.** No open measurement remains.

---

## O. WRITE-UP AUDIT — 2026-08-29. Cross-check of every headline against the dump.

Before drafting `icassp/paper.tex`, every number in `HANDOVER_2026-08-29_FINAL.md`
§2 was re-read from `icassp/paper_numbers.txt`. Three things did not hold.

### O1. ⚠ RETRACTED — "at p = 1%, an FPR of 5% caps precision at 0.161"

**No CSV on the box contains 0.161, and no section of this ledger carries it.**
It traces only to `HANDOVER_2026-08-22.md:111`, from where the handover and
`paper_outline.md` §6 copied it forward. This is the sixth instance of the
project's standing failure mode — a value carried instead of re-derived — and it
was caught only because the write-up required a CSV path per number.

What the operating-point CSVs actually say, at the real-only q = 0.95 threshold:

| arm | file | FPR | TPR | **precision @ p = 1%** | n reals |
|---|---|---|---|---|---|
| **NMF, SONICS, full corpus** | `analysis_nmf_sonics_FULL/operating_points.csv` | 0.0470 | 0.6737 | **0.1265** | **6,361** |
| NMF, FakeMusicCaps | `analysis_nmf_fmc/operating_points.csv` | 0.0300 | 0.9573 | 0.2438 | 300 |
| comb, FakeMusicCaps | `analysis_comb_fmc/operating_points.csv` | 0.0500 | 0.4890 | 0.0899 | 300 |

**Use 0.1265** — it is the only one of the three measured at full corpus. The
analytic ceiling at FPR = 5%, TPR = 1, p = 1% is 0.168; 0.161 is near it but is
not a measurement of anything, and the two must not be conflated. The paper
states the measured value and the bound separately.

Note also that the two FakeMusicCaps deployment rows rest on **300** evaluation
reals against SONICS's 6,361 — the same small-n caveat as §B7. Do not mix arms
or corpora in one deployment sentence.

### O2. ⚠ CORRECTED — the gate status table must have an ARM axis

Handover §2.8 prints one gate column per corpus. The CSVs show C8 and C9 are
**complementary across the two arms** and never both PASS for one number:

| gate | comb arm (`gate_comb_fmc_v2/…median_db__comb_strength.csv`) | NMF arm (`gate_nmf_FULL/…bins7167_pooled.csv`) |
|---|---|---|
| C1 channel-alone | FAIL 0.6284 | FAIL 0.6284 |
| C2 level | FAIL 0.6207 | FAIL 0.6207 |
| C3 duration | PASS 0.5228 | PASS 0.5228 |
| C8 decoder-spacing | **PASS** | **SKIP** — no `comb_spacing_hz` on this arm |
| C9 harmonicity | **SKIP** — no descriptor | **PASS** — within-real \|corr\| 0.184 |
| C11 bootstrap CI | PASS 0.9225 [0.9075, 0.9373] | PASS |

C1/C2 are corpus properties, so they are identical across arms as expected.

⚠ Also worth stating in any writeup: the superseded `gate_comb_fmc` (v1) records
**C8 FAIL** on the *same detector*, under the old between-spread-vs-within-scatter
criterion (83.5 Hz vs 295.3 Hz). v2 PASSes under a different criterion —
per-generator concentration within ±5% of own median (fake 0.436 vs real 0.035).
**That is a criterion change, not a measurement change**, and reporting "C8 PASS"
without naming the criterion would be the same class of error as reporting
`|AUC|` without saying so.

### O3. ⚠ `paper_outline.md` v4 §1 is STALE — do not read numbers from it

The outline's headline table quotes reals-only held-out **0.7345 / 0.4607**
(Round 6, §J1, single seed). §M2 and §N3 supersede it with **0.744 ± 0.020 /
0.382 ± 0.043** (Rounds 9–10, three seeds). The SONICS value moves by 0.079 and
the gap with it. The outline's SONICS dose table (0% = 0.342, 1.44% = 0.788) is
the single-seed `_lo` run superseded by §N.

**The outline is a skeleton. §N3 and §N are the authority.** The paper uses those.

### O4. ✅ Verified, no action

* **Efficiency 42.6 ± 4.0 tracks/s, 379.7 audio-s/s.** Two efficiency JSONs exist
  and disagree by 5.8×; `efficiency_comb_median_db.json` (7.34 tracks/s, 66
  audio-s/s) is the **superseded** timing, whose correction and cause are
  recorded at `RUNBOOK_2026-08-20_corrections.md:1848` — the old path ran the
  hull Python loop plus unused diagnostics. `efficiency_comb_detector_only.json`
  is the one to cite. Checked because the filenames invite the opposite reading.
* Transfer FPR 0.0419 → 0.1151 (2.75×) and per-genre 0.2356 / 0.0279 (8.46×):
  both re-read from `fpr_fma_by_genre/*`, n = 2,675 threshold / 2,998 unseen.
  **These are the comb arm**; the NMF FPR arm is n = 300 (§B7).
* Corpus sizes 32,960 / 59,280 / 3,000 re-read from the manifests.
* Robustness: our baseline EER is **14.6%** against MusicDET's reported 4.51%,
  so absolute post-manipulation EERs are not comparable. Report degradation from
  each system's own baseline (AAC +0.3 vs +31.3; Opus +0.4 vs +17.6) and concede
  time-stretch, where they report 2.44% absolute against our 29.26%.

### O5. Three paper numbers that were NOT verbatim in this ledger — now derived

A mechanical diff of every numeral in `paper.tex` against this file left three
unmatched. All three are legitimate, and the derivations are recorded here so
they are traceable rather than trusted.

**`0.8468` — the `|AUC|` SONICS headline (retraction #3).** Present in the
handover, absent here. It is the macro over the same five per-generator rows that
give the signed 0.6203, from
`comb_sonics_fmin1000/comb_auc_lvl_sm5_median_db_f1000_noclip.csv`,
feature `comb_strength`:

| generator | chirp-v2 | chirp-v3 | chirp-v3.5 | udio-120s | udio-30s | macro |
|---|---|---|---|---|---|---|
| `auc_signed` | 0.8857 | 0.9268 | 0.8553 | **0.2588** | **0.1751** | **0.6203** |
| `auc_abs` | 0.8857 | 0.9268 | 0.8553 | 0.7412 | 0.8249 | **0.8468** |

Inflation **0.2264**, which the paper rounds to +0.23. The two udio families are
the whole of it, and taking the absolute value is exactly what conceals them.

**`0.510` — the held-out flow likelihood decomposition.** Recomputed from
`flow_terms_fmc_holdout/flow_terms_auc_lvl.csv`, macro over five generators:
base density `neg_log_pz` **0.5127**, `neg_log_det_jac` **0.4416**, their sum
`neg_log_prob` **0.5096**. Confirms `project_narrative.md` §2 exactly. The paper
rounds the sum to 0.510.

**`0.028` — Folk false-positive rate.** 0.0279 rounded
(`fpr_fma_by_genre/false_positive_rate_by_group.csv`). Electronic likewise
0.2356 → 0.236. Ratio 8.46×.

### O6. ✅ Per-genre FPR novelty claim — RESOLVED by reading 2506.18488 in full

Frohmann, Epure, Meseguer-Brocal, Schedl & Hennequin, *"AI-Generated Song
Detection via Lyrics Transcripts"*, **ISMIR 2025**. Previously `SEARCH`-only and
flagged as blocking; now `SRC`.

They break down **recall** by genre, not false-positive rate, and mention
higher false positives for Alternative / Electronic / Pop / R&B / Rock only in
prose, with no numbers. Their detector is **text** (Whisper transcription →
LLM2Vec → MLP) and never touches the audio signal.

So the narrowed claim is verified, not merely safe — **and the citation now works
for us**: two disjoint modalities over-accuse the same genres, which argues the
bias is a property of the music rather than of either feature set. Cited in §6.
See `bibliography_review.md` §2.3.

### O7. ✅ C9 CLOSED on the comb arm — the last untested gate on a reported number

`gate_comb_fmc_c9/`, run 2026-08-29 against
`channel_fmc_harm/channel_per_track_canonical_profile_lvl.csv`, which already
carried `harmonicity` / `pitch_salience` / `spectral_peak_count`. No
re-extraction was needed.

| gate | verdict | value |
|---|---|---|
| C1 channel-alone | FAIL | `spectral_flatness_hf` 0.6284 |
| C2 level | FAIL | `peak_dbfs` 0.6207 |
| C3 duration | PASS | 0.5228, full-length gap 0.007 |
| C4 silence | PASS | 0.5104 |
| C5 label shuffle | PASS | [0.5001, 0.5101] |
| C6 / C7 | PASS | 32,951 finite / smallest stratum 5,355 |
| C8 decoder spacing | PASS | fake concentration 0.431 vs real 0.030 |
| **C9 harmonicity** | **PASS** | **within-real \|ρ\| 0.019** on `pitch_salience` |
| C10 | SKIP here | closed separately by the within-pair test (§K1) |
| C11 | PASS | **AUC 0.9177 [0.9128, 0.9227]** |

**C9 now passes on every arm and both corpora**: 0.019 (FMC comb), 0.048–0.056
(SONICS comb), 0.184 (FMC NMF), 0.253 (SONICS NMF), all against a 0.40
threshold. The within-*fake* correlations are much larger (0.30 on harmonicity,
**0.66** on spectral peak count for the FMC comb) — that is the mechanism, and
the gate deliberately reports it unthresholded.

Note this also gives the C11 bootstrap CI for the headline comb number at full
corpus: **0.9177 [0.9128, 0.9227]**, n = 32,951.

### O8. ✅ FIGURE 2 — all 20 atoms, with a null model

`reports/figures/nmf_{fmc,sonics}_periodicity/atom_peakiness.csv` (batch-2 dump).
The ledger's §G2 quoted only the matching atoms; the full dictionaries are much
more informative.

An atom counts as **on a decoder comb** if its recovered spacing is within **1%**
of a measured decoder spacing *or of its half or quarter*. Sub-multiples are
admissible because an autocorrelation-in-frequency locks onto any period of the
comb: every second or fourth tooth is still periodic.

| corpus | atoms on a comb | null mean | p | mean periodicity |
|---|---|---|---|---|
| **FakeMusicCaps** | **15 / 20** | 0.64 | **1×10⁻⁴** | **0.700** |
| **SONICS** | **9 / 20** | 0.34 | **0.010** | **0.464** |

The null draws the same number of fundamentals uniformly from 40–700 Hz and
recounts, 20,000 times (`comb_match_null` in `scripts/make_paper_figures.py`).
**Without it the match count is not a result** — a tolerance band plus
sub-multiples covers enough of the axis to guarantee some hits.

Two further contrasts, both new:

* On FMC the on-comb atoms are also the *more periodic* ones (0.752 vs 0.544);
  on SONICS that separation vanishes (0.474 vs 0.455).
* `corr(peakiness, periodicity)` = **+0.035** (FMC), **−0.309** (SONICS),
  reproducing §G2 exactly from the full 20-atom tables and confirming §F4:
  peakiness could never have supported this claim.

The FMC hits are exact where the estimate is well resolved — 200.3 Hz against a
measured 200.195 (0.05%), 250.1 against MusicGen's 250.0 (0.04%) — and the
remaining hits are at 49.8 and 99.6 Hz, the quarter and half of 200.195.

### O8b. ⚠⚠ CORRECTED, same day — the atom-matching null was too weak, and the SONICS half of the claim does not survive

§O8's null draws a fundamental uniformly from 40–700 Hz. A **cross-corpus null**
is far more demanding and more honest: count how many atoms match a *real*
decoder spacing taken from the **other** corpus.

| | own fundamental(s) | the other corpus's | Fisher exact |
|---|---|---|---|
| FMC atoms, m ∈ {1,2,4} | **15/20** | **7/20** | p = 0.025 |
| SONICS atoms, m ∈ {1,2,4} | **9/20** | **4/20** | **p = 0.176 — n.s.** |
| FMC atoms, **m = 1 only** | **6/20** | **0/20** | p = 0.020 |
| SONICS atoms, **m = 1 only** | **6/20** | **3/20** | **p = 0.451 — n.s.** |

**Why.** FMC's decoder comb is at **200.195 Hz** and SONICS's at **399.902 Hz** —
`399.902/2 = 199.95`, which is **0.12%** from 200.195, and `399.902/4 = 99.98` is
0.12% from 200.195/2. **The two corpora's decoder combs are themselves within
0.12% of a 2:1 ratio**, so admitting sub-multiples makes them mutually
indistinguishable. The rule that made the count large is the same rule that
destroys its power to attribute.

**What survives, and what does not:**

* ✅ **FakeMusicCaps holds, and for an identifiable reason.** Six atoms sit at a
  measured fundamental (m = 1) and **none** at SONICS's, p = 0.020. The
  discriminating evidence is the **250.0 Hz MusicGen cluster** — atoms 13, 6, 18
  at 250.1 Hz — because 250 Hz is *not* a sub-multiple of 399.902 and so cannot
  be confused with the other corpus.
* ❌ **SONICS does not.** Six atoms at 399.902, but three more sit within 1% of
  200.195 (they are at ~200.7 Hz), and spacing alone cannot say which comb they
  belong to. p = 0.45. **This must be reported as a limitation, not a result.**

**Consequence for the paper.** The mechanism claim is stated for FakeMusicCaps
with the cross-corpus null beside it, and explicitly *withdrawn* for SONICS.
Reported alongside is the corpus contrast that does survive and does not depend
on attribution: mean atom periodicity **0.700** vs **0.464**, ordered the same
way as the AUCs.

Caught by running the null the reviewer would have run. Same failure mode as the
other six: a criterion chosen because it was generous rather than because it
discriminated.

### O9. ⚠ The held-out three-learner run FAILS as commanded — use `--n-reals`

```
--holdout-reals left only 0 real tracks to score for config
k20_sigma2_bins445_reals_n5355_kmeans_heldout_s42
```

`--fit-on reals` without `--n-reals` puts **every** real in the dictionary, so
holding them out leaves nothing to score. The guard fired correctly — this is
the degenerate-cell guard added in Round 9 (§M4) doing its job rather than
silently emitting an empty AUC row. **The dictionary budget must be set
explicitly:**

```bash
for D in sparse kmeans; do
python scripts/eval_nmf_peak_detector.py \
  --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \
  --per-stratum 0 --workers 3 --max-duration 9 --level-match \
  --n-atoms 20 --blur-sigma 2 --dictionary $D --n-reals 2400 \
  --fit-on reals --holdout-reals --fit-seeds 42 43 44 \
  --out-dir reports/diagnostics/dict_${D}_fmc_reals_heldout; done
```

`--n-reals 2400` matches the largest budget in `nmf_fmc_reals_heldout` (§J1), so
the result is directly comparable to NMF's 0.7345 at that budget.

**Until this runs, the sparse/k-means reals-only cells in §F2/§G3 remain at the
leaked 600/stratum protocol** and the paper claims only the qualitative
replication ("the collapse reproduces under sparse coding and k-means atoms"),
not the values.

### O10. Figure 2 is no longer blocked — superseded note kept for the record

`scripts/make_paper_figures.py` draws Fig. 2 as a per-atom scatter of recovered
spacing against periodicity and **refuses to draw it from this ledger's §G2
excerpt**, because §G2 lists only the atoms that matched a decoder spacing. A
scatter of those alone would put every atom on a decoder line and imply a result
the dictionary does not support. It needs all K = 20 atoms per corpus from
`reports/figures/nmf_{fmc,sonics}_periodicity/atom_peakiness.csv`, which are on
EC2 and not in the 2026-08-23 dump. `print_paper_numbers.sh` §15 now collects
them.


---

## P. ROUND 11 — 2026-08-30. The atom-budget hypothesis is TESTED, and two citation errors.

### P1. ⚡ The contamination threshold scales with the atom budget K — hypothesis confirmed

`nmf_sonics_K{5,10,40}_knee/nmf_peak_auc_lvl.csv`, held out, `--fit-real-frac 0.5`,
3 seeds, bins4779, σ=5. **NOTE the x-axis:** `--fake-frac` is a fraction of the
FAKE POOL, so the achieved doses here are 0 / 66 / 168 / 341 / 507 / 670 fakes
(0, 1.03, 2.57, 5.09, 7.38, 9.53% of the fit set) — a coarser and higher grid
than the 3-seed curve of §N, which sampled 0–93 fakes.

| K | 0 | 66 | 168 | 341 | 507 | 670 fakes |
|---|---|---|---|---|---|---|
| 5 | 0.334 | 0.356 | **0.429** | **0.774** | 0.775 | 0.778 |
| 10 | 0.352 | **0.449** | **0.798** | 0.800 | 0.803 | 0.814 |
| 20 (§N) | 0.382 | — flips at **47** — | 0.793 | | | 0.932 |
| 40 | 0.359 | **0.780** | 0.791 | 0.820 | 0.864 | **0.873** |

**The dose that flips the detector falls monotonically as K grows:** 168–341
tracks at K=5, 66–168 at K=10, ≈47 at K=20, and K=40 already flipped at the
smallest dose sampled (66). That is the atom-budget reading — the factorisation
must be able to afford one atom for the decoder comb — turned from a hypothesis
into a measured scaling. §3.2 of the paper states it as such.

**Two caveats the paper keeps:**
* K=40's threshold is only bounded (≤66), not located; the grid does not resolve
  below 1.03%. The monotone claim rests on K=5 > K=10 > K=20.
* At K=40 **udio becomes correctly oriented for the first time** (0.748 / 0.686
  at 9.53%, against ≈0.49/0.55 at K≤10). Capacity, not dose, is what buys udio.
  Not yet in the paper; it is a lead worth one line if space appears.

### P2. ⚠ The 10-seed knee run was KILLED — re-run alone

`nmf_sonics_knee_10seed` reached `mix7.38pct_..._s51` and was killed. It ran
concurrently with the K sweep on a 15 GB box; two full 59,280 × 4,779 float64
matrices do not co-exist. **Re-run detached and alone.** Not required for the
paper — three draws already fix the location and sharpness, and the limitation
is stated.

### P3. ⚠⚠ TWO CITATION ERRORS found by re-verification, both from search summaries

The write-up rule "verify from the source, not the abstract" applies to
*bibliography* too, and two entries had been built from search-result summaries
rather than a direct fetch:

| entry | wrong | correct |
|---|---|---|
| DOUST / transductive OD | *"About Test-Time Training for Outlier Detection"* | **"Deep Transductive Outlier Detection"** — the search returned the superseded v1 title; arXiv:2404.03495 was renamed in v2 |
| ASVspoof 5 (2502.08857) | 13 authors | **29 authors** — the 13-name list belongs to arXiv:**2408.08739**, a *different* ASVspoof 5 paper. Now truncated with `and others` |

Every one of the 29 entries in `icassp/refs.bib` has now been resolved by a
direct fetch of its arXiv or publisher page, not a search summary.
Cros Vila et al. (TISMIR 8(1):179–194) and Li et al. (*Sci. Rep.* 16:13757)
re-confirmed against the publisher pages.


### P4. ✅ The 10-seed knee run — the bimodality claim, properly supported

`nmf_sonics_knee_10seed/nmf_peak_auc_lvl.csv`, seeds 42–51, bins4779, held out.

| contamination | fakes | flipped | macro range | TPR@5%FPR |
|---|---|---|---|---|
| 0.36% | 23 | **0/10** | 0.344–0.611 | 0.008–0.027 |
| 0.733% | 47 | **7/10** | 0.411–0.787 | see below |
| 1.09% | 70 | **10/10** | 0.783–0.790 | 0.465–0.483 |
| 1.44% | 93 | **10/10** | 0.785–0.792 | 0.470–0.487 |

**At 47 fakes the two groups are strictly disjoint.** TPR at a 5% FPR is
**0.446–0.468** in the seven draws that flipped and **0.0085–0.0124** in the
three that did not — a **36×** gap with nothing in it. Macro AUC shows the same
split (seven at 0.780–0.787, three at 0.411–0.547).

This replaces the 3-seed evidence for the flagship claim. Note 23 fakes is *not*
bimodal — the macro spreads continuously over 0.344–0.611 — so the transition
begins between 23 and 47 and completes by 70. The paper now states flip counts
out of ten and quotes the TPR separation rather than the seed SD.

---

## Q. Write-up audit round 2 (2026-09-06) — reading the source paper in full

### Q1. ⚠⚠ RETRACTION 8: "the published 445-bin descriptor" — the published
dimension is **4458**, and 445 was ours

`context/finding_the_noise.md` (arXiv:2607.25530v1) §4.2, verbatim: *"We use the
similar parameters to configure the fakeprints as in the original paper (e.g., a
n_fft of 2^14 for the STFT), and a frequency band of [3kHz, 15kHz]. **This
results in fakeprint vectors of length 4458.**"*

Internally consistent: 44100/16384 = 2.6917 Hz/bin; (15000−3000)/2.6917 = 4458.1.
**445 is a dropped digit on our side.** It propagated to
`scripts/eval_nmf_peak_detector.py:112`, `features/comb_artifacts.py`,
`config.py`, the `afchar-source-of-truth` memory, ledger §A, and to paper.tex's
contribution (iii) and §6, where it was billed as a DSP finding against prior
work. We had even rationalised it ("chosen for 44.1 kHz where a bin is ~27 Hz"),
which requires n_fft ≈ 1638 against their stated 2^14. The rationalisation should
have caught it.

| withdrawn | replaces it |
|---|---|
| "the published 445-bin descriptor is aliased at 16 kHz, worth +0.046/+0.102" | "the published descriptor transfers to 16 kHz essentially intact; resolution binds only below ~1500 bins" |

**The measurement stands; only the attribution was wrong.** Resolution sweep,
full corpus, `reports/diagnostics/nmf_{fmc,sonics}_binsweep/nmf_peak_auc_lvl.csv`,
signed macro AUC:

| bins | FakeMusicCaps (σ=2) | SONICS (σ=5) |
|---|---|---|
| native (7167 / 4779) | **0.99168** | **0.93894** |
| **4458 — their published dimension** | **0.98910** (−0.0026) | **0.93864** (−0.0003) |
| 2600 — their physical resolution (2.69 Hz/bin) | 0.98782 | 0.92926 |
| 1500 | 0.97934 | 0.91846 |
| 1000 | 0.97382 | 0.89860 |
| 445 — ours | 0.94568 (−0.046) | 0.83676 (−0.102) |

The requirement is a **resolution** (≈2.7 Hz/bin), not a dimension, and it is
generator-dependent: at 445 bins the three chirp families hold 0.9004–0.9823
while both Udio families fall to 0.6745/0.6783.

### Q2. Framing corrected — their paper states the limitation we measure

§5: *"One limitation of our work is the need for a sufficiently large sample
collection for a synthetic class to detect it (otherwise, it is treated as noise
and thus as real)."* §3.2: *"We assume the class cardinality not to be too
imbalanced."* §3.3, of Task A: *"This is where this method turns into a one-class
classification."*

So the claim "they call it zero-shot but it is transductive" was too strong; their
§3.3 is careful, even though title, abstract and conclusion frame the method as
zero-shot. **The contribution is now quantifying an acknowledged limitation** —
0.7% of the unlabelled set, as a threshold not a ramp, with inversion below it.
Paper title changed accordingly.

### Q3. They rejected 16 kHz corpora; that becomes a contribution, not a threat

§4.1: *"all audios in SONICS have been resampled to 16kHz, which results in a
unsuited cutoff in the frequency bands of interest for fakeprints."* They use
FMA-AE, Echoes and PopularAISet at 44.1 kHz and never evaluate on SONICS or
FakeMusicCaps. With the band re-derived to [1k, 8k] the method reaches
0.9917/0.9389 at 16 kHz, so the cutoff is survivable — which is what makes the
whole boundary measurement possible on the public benchmarks.

### Q4. Udio: an open prediction, not a defect

They report Udio among their best-detected at 44.1 kHz (0.2% EER supervised;
99.4%/96.4% one-class on PopularAISet). Udio is our weakest family at every
resolution and inverts in the comb arm at 16 kHz. **Prediction: the Udio comb
lies largely above the 8 kHz Nyquist limit of both public benchmarks rather than
being absent.** Untestable on SONICS (16 kHz only); needs wideband Udio audio.
Now in the paper's future work.

### Q5. Disclosed: we use UNREGULARISED NMF where they use elastic net

Their §3.3 minimises `||X − HW||² + λ_H||H||_{1,2} + λ_W||W||_{1,2}`. Our
`eval_nmf_peak_detector.py:194` is
`NMF(n_components=20, init="nndsvda", random_state=seed, max_iter=600)` —
sklearn defaults `alpha_W=0.0`, `l1_ratio=0.0`, i.e. **no penalty at all**. The
λ values are not in their paper ("The rest of the chosen parameters may be found
on our code repository"), so we could not match them without inventing.

This matters because the penalty is the mechanism our whole paper is about: it
exists, in their words, so that an atom is learned "only if a sufficient
collection of fakeprints follow a given pattern". Regularisation can only make it
*harder* for a small synthetic class to earn one of the 20 atoms, so the
threshold under their formulation sits at or above ours. **The 0.7% dose is
therefore a lower bound on theirs, and the paper now says so** (§3.1). The claim
"we reimplement without modification" was withdrawn from the paper.

Optional sensitivity run if time allows: sweep `alpha_W ∈ {1e-4, 1e-3, 1e-2}` at
`l1_ratio=0.5` on the SONICS knee doses and check the flip point only moves up.
Not required for submission; the argument direction is already favourable.

---

## R. ROUND 12 — 2026-09-07/08. A constructive arm, and its null PRE-REGISTERED

### R0. Why this section exists before its numbers do

The paper measures an existing detector and proposes nothing. This round adds a
proposed arm. The first FakeMusicCaps result is **positive and not yet honest**,
for a reason recorded here *before* the control was run, so that the control's
outcome cannot be reinterpreted after the fact.

### R1. ⚡ The first measurement — `comb_fmc_HARM`, full corpus, n = 32,960

`reports/diagnostics/comb_fmc_HARM/comb_auc_lvl_sm5_median_db_f1000_noclip_harm248.csv`,
signed macro AUC, same audio and same operator as the published
`median_db__comb_strength` = 0.91766 (computed in the same pass as a control):

| feature | MusicGen | audioldm2 | musicldm | mustango | stable_audio_open | MACRO |
|---|---|---|---|---|---|---|
| **`comb_harm2_prior_strength`** | 0.9697 | 0.9340 | 0.8969 | 0.9635 | 0.9487 | **0.94256** |
| `comb_harm4_prior_strength` | 0.9671 | 0.9052 | 0.8491 | 0.9473 | 0.9600 | 0.92574 |
| `comb_harm4_sharpness` | 0.9119 | 0.9595 | 0.8935 | 0.9869 | 0.8475 | 0.91986 |
| **`median_db__comb_strength` (incumbent)** | 0.9565 | 0.9123 | 0.8545 | 0.9504 | 0.9146 | **0.91766** |
| `comb_harm2_strength` | 0.9575 | 0.9137 | 0.8492 | 0.9491 | 0.9178 | 0.91746 |
| `comb_surrogate_z` | 0.9568 | 0.9111 | 0.8547 | 0.9452 | 0.9154 | 0.91664 |
| `comb_harm8_strength` | 0.9620 | 0.8812 | 0.8276 | 0.9146 | 0.9192 | 0.90092 |

**+0.0249 over the incumbent, and every one of the five families improves.**

Three negatives from the same table, recorded because they kill three of this
round's four hypotheses:

* **The harmonic sum alone does nothing**: `comb_harm2_strength` 0.91746 vs
  `comb_strength` 0.91766, and it *degrades* with M (0.91078 at M=4, 0.90092 at
  M=8). The fixture predicted a gain in the weak-comb regime; FakeMusicCaps has no
  family in that regime (all five already ≥ 0.85).
* **The per-track surrogate null does nothing**: 0.91664 vs 0.91766.
* **The overlapping-span stationarity does not rescue the stationarity channel**:
  `comb_harm2_stat_strength` 0.85850 against `comb_stat_strength` 0.84778.

Two genuine side-results:

* `comb_harm4_sharpness` **fixes the sharpness channel's sign failure**:
  stable_audio_open goes 0.2370 → 0.8475 and the channel becomes correctly
  oriented on all five (macro 0.72388 → 0.91986).
* `comb_harm2_spacing_dispersion` reaches **|AUC| 0.865** against
  `comb_spacing_dispersion`'s 0.756. Its direction is fixed a priori by physics —
  a decoder's spacing is stationary, so LOW dispersion ⇒ generated — and that
  orientation was documented in `comb_stationarity`'s docstring before this run.

### R2. ⚠⚠ Why R1's headline is NOT yet reportable — a leak of our own making

`comb_harm2_prior_strength` restricts the autocorrelation search to
`DECODER_FUNDAMENTALS_HZ = {21.53, 50, 75, 86.13, 93.75, 100}`. **That list was
assembled partly by working backwards from spacings measured on FakeMusicCaps**:
100 Hz because the measured comb is at 200.195 = 2 × 100, 50 Hz because MusicGen's
is at 250 = 5 × 50, 21.53 Hz because Stable Audio Open's is at 107.42 ≈ 5 × 21.5.
The derivations are written in the source comment in our own hand.

A prior chosen with the answers in view is not a prior. There are three live
explanations for +0.0249 and they are not distinguishable from R1 alone:

1. **Set SIZE.** The max of a noisy autocorrelation over K lags grows like
   `σ√(2 ln K)`, so going from ~4,000 lags to 6 suppresses real music's baseline
   by `√(ln 6 / ln 4000) ≈ 0.47` **whatever the candidates are**. Legitimate and
   content-free.
2. **Set CONTENT, as physics.** The candidates are genuine published stride
   products and the restriction forces the search onto the *fundamental* rather
   than the loudest harmonic.
3. **Set CONTENT, as leak.** We enumerated the decoders in our own test set.

### R3. The controls, and what each can and cannot settle

**Control A — the decoy null (`decoy_fundamentals`, 24 sets, seed 20260907).**
Same cardinality, drawn log-uniformly from [20, 150] Hz, rejecting anything within
5% of a real prior. Separates (1) from (2)+(3). Emitted on the base operator only,
as `comb_harm{M}_decoy{NN}_strength` and `comb_priormax{M}_decoy{NN}_strength`.

**Control B — SONICS as a holdout for the prior.** Suno's and Udio's decoders are
proprietary and were **not** used to build the list, so SONICS separates (2) from
(3). This is the only control that can, and it is why the SONICS run is required
rather than optional. Note the standing arithmetic hazard of §O8b: chirp's measured
399.902 Hz is 4 × 99.98, and 100 Hz *is* in the list, so a hit on chirp is
consistent with a genuine 100 Hz fundamental and must be checked against
`comb_priormax{M}_hz` rather than assumed.

### R4. PRE-REGISTERED PREDICTIONS — recorded 2026-09-07, before either run

From a synthetic fixture at the FakeMusicCaps geometry (bin 0.9766 Hz, 7,167 bins,
3-bin teeth, 120 draws per class). **These are predictions, not measurements.**

| # | prediction | what it would mean if WRONG |
|---|---|---|
| P1 | `comb_harm2_prior_strength` decoys collapse to ≈0.37–0.49 macro, well below the real prior's 0.9426 | the gain is set SIZE, not content — the prior claim is void and the honest statement is "restricting the candidate set helps, and the frequencies are irrelevant" |
| P2 | `comb_priormax8_*` decoys reach ≈1.0, i.e. **no specificity at M = 8** | our reading of why is wrong; at 6 × 8 = 48 lags the candidate set is dense enough to hit any comb by chance, which is §O8b's sub-multiple hazard again |
| P3 | `comb_priormax2_strength` is WEAK on MusicGen and stable_audio_open (≈0.2–0.5) because no prior's first two multiples reach 250 Hz or 107.4 Hz | the mechanism is not the one stated |
| P4 | `comb_harm{M}_prior_hz` is CONCENTRATED per generator for fakes and SCATTERED for reals | the score reads residual smoothness at short lags, not a comb, and is a nuisance |
| P5 | On SONICS, if the prior fires on chirp it fires at 99.98 Hz or a multiple | a hit at an unrelated candidate means the prior is not reading Suno's stride |

**The criterion, fixed now:** the prior arm ships **only if** the decoys collapse
(P1) **and** it transfers to SONICS, where the list had no opportunity to leak. If
the decoys match, the reportable claim shrinks to the content-free one. If SONICS
collapses while FakeMusicCaps holds, the arm is withdrawn and this section records
it as retraction 9.

### R5. Code and its own falsification

`src/.../features/comb_artifacts.py` gains `comb_harmonic`, `comb_prior_strength`,
`comb_prior_max`, `comb_surrogate_null`, `comb_harmonic_stationarity`,
`decoy_fundamentals` and an overlap factor on `_span_residuals`. **`comb_strength`
is untouched** and every pre-existing column keeps its value: the span hop is
`span/k` for an integer k *dividing* the span, so `spans[::k]` reproduces the
non-overlapping set bin-for-bin, and on SONICS k = 1.

A first implementation used `span_samples // k` for arbitrary k, which at k = 7 put
the subsampled spans at 63,994 instead of 64,000 — the "unchanged" published
`comb_stat_strength` would have moved with nothing reporting an error. Caught by
`test_overlap_reproduces_the_non_overlapping_set_exactly`. Same family as §L1/§L2.

**A hypothesis of ours that a fixture falsified, kept visible:** we proposed that
harmonic summing would resolve the sub-multiple ambiguity and stabilise the spacing
estimate. It does not — if teeth sit at every multiple of 50 with the multiples of
150 tallest, 150 is *itself* a period, so its own multiples are all elevated and it
wins the harmonic sum too. Measured over 200 noise draws, spacing concentration is
0.055 (argmax) vs 0.060 (harmonic). Pinned by
`test_harmonic_sum_does_NOT_resolve_the_sub_multiple`. Consequence:
`comb_harm_spacing_hz` is **not** a better identity than `comb_spacing_hz`, and any
cross-track consensus built on it inherits the same instability.

### R6. OUTCOMES against the §R4 pre-registration — 2 confirmed, 2 refuted, 1 pending

`comb_fmc_HARM2/` (n = 32,960) and `comb_sonics_HARM2/` (n = 59,280), full corpus,
same audio and operator as the published arms, 24 decoy sets, seed 20260907.

| # | prediction | outcome |
|---|---|---|
| P1 | decoys collapse to ≈0.37–0.49 | **CONFIRMED in direction, WRONG in magnitude.** 0/24 decoys reach the real prior on **either** corpus, but the decoy median is 0.69 (FMC) / 0.59 (SONICS), not 0.37–0.49 |
| P2 | `priormax8` decoys reach ≈1.0 — no specificity at M=8 | **REFUTED.** decoy median 0.799, max 0.936, still 0/24 ≥ real (0.9383). The fixture over-predicted decoy performance because its comb had a single spacing where a real fakeprint has richer structure |
| P3 | `priormax2` weak (≈0.2–0.5) on MusicGen and stable_audio_open | **REFUTED.** 0.9681 and 0.9362. Both fire on a true sub-harmonic (MusicGen at 49.8, SAO at 43.0 = 2 × 21.5) |
| P4 | `prior_hz` concentrated for fakes, scattered for reals | **CONFIRMED, and it is the round's strongest result.** See R7 |
| P5 | on SONICS the prior fires on chirp at 99.98 Hz or a multiple | **CONFIRMED**: chirp fires at 99.6 Hz (M=2) and 49.8 Hz (M=4, M=8) |

**The magnitude error in P1 matters and is not cosmetic.** Decoys at 0.69 rather
than chance means restricting the candidate set is worth ≈0.19 AUC **whatever the
frequencies are** — the extreme-value effect is real and large. The correct
frequencies are worth a further ≈0.25. Both must be reported; attributing the
whole +0.029 over the incumbent to the physics would overstate it.

### R7. ⚡ The mechanism, per architecture — `comb_harm2_prior_hz` mode and its share

| generator | fires at | share | published stride product |
|---|---|---|---|
| MusicGen | **49.80 Hz** | 0.918 | EnCodec 32 kHz, ∏s = 640 → 50.00 |
| audioldm2 | **99.60 Hz** | 0.937 | HiFi-GAN 16 kHz, mel hop 160 → 100.0 |
| musicldm | 99.60 Hz | 0.847 | as above |
| mustango | 99.60 Hz | **0.991** | as above |
| stable_audio_open | **21.50 Hz** | 0.857 | Stable Audio 44.1 kHz, ∏s = 2048 → 21.53 |
| **real music** | 21.50 Hz | **0.189** | — no fundamental; scattered |
| **chirp-v2 / v3 / v3.5** | **49.80 Hz** | **0.971 / 0.990 / 0.976** | proprietary (Suno) |
| udio-120s / udio-30s | 93.80 Hz | **0.290 / 0.217** | — barely above real's 0.189 |

Two things follow.

**The prior transfers to decoders it was not built from.** `DECODER_FUNDAMENTALS_HZ`
was assembled from public architectures, three of whose values coincide with
FakeMusicCaps' measured spacings (§R2), so FakeMusicCaps alone could not settle
whether the list was a prior or a fit. Suno's decoder is proprietary and did not
inform the list, and chirp concentrates on 49.8 Hz in **97–99%** of tracks. Its
visible comb at 399.902 Hz is therefore the **8th harmonic of a ≈49.9 Hz
fundamental**, not a 400 Hz spacing. That also softens §O8b: the SONICS comb is a
harmonic of a low fundamental, which is why spacing alone could not attribute it.

**Udio has no fundamental in band.** Concentration 0.217–0.290 against real music's
0.189, on the statistic where every other family reaches 0.85–0.99. Together with
`comb_strength` (0.055) sitting *below* real music's (0.072) and
`comb_spacing_dispersion` showing Udio *less* stationary than real music, three
independent statistics agree. This is convergent support for the paper's standing
prediction (§Q4) that Udio's comb lies above the 8 kHz Nyquist of both benchmarks,
and it means **no in-band readout can recover Udio** — a data limitation, not a
method failure. Any Udio number must come from the two-sided branch.

### R8. Where the arm stands, one-sided, against the incumbents

| | FakeMusicCaps | SONICS |
|---|---|---|
| `comb_priormax2_strength` | **0.9468** (0/5 inverted) | 0.6661 |
| `comb_harm2_prior_strength` | 0.9426 (0/5) | 0.6914 (2/5 inverted — both Udio) |
| `comb_harm8_prior_strength` | 0.8513 | **0.7313** (2/5) |
| `comb_strength` (incumbent) | 0.9177 (0/5) | 0.6311 (2/5) |
| `comb_stat_strength` (incumbent) | 0.8478 | 0.6833 (2/5) |
| two-sided fusion (paper's SONICS headline) | — | 0.7546 |

**FakeMusicCaps clears the bar (+0.029 over 0.9177, no family inverted). SONICS
does not yet**: the best one-sided prior figure (0.7313) is below the paper's
existing two-sided 0.7546, and Udio inverts. The two-sided run is what decides it,
and it is pre-registered in §R9 rather than chosen after the fact.

`comb_priormax4_hz` reaches 0.8016 on SONICS with 0/5 inverted, and must NOT be
reported: it is a *spacing identity*, and §C1 established that spacing's sign is a
property of which decoders and which reals a corpus happens to contain — it is
0.30 (all five inverted) on FakeMusicCaps. Sign-consistency within one corpus is
necessary and not sufficient.

### R9. PRE-REGISTERED — the two-sided cell, before it is run

Features fixed by MECHANISM, not by AUC, and identical on both corpora:
`comb_priormax2_strength` (periodicity at a plausible fundamental),
`comb_harm8_prior_strength`, `comb_stat_strength` (stationarity),
`comb_residual_std`, `comb_residual_kurtosis` (peak amount) — the last three being
the branches `fuse_physics_scores.py`'s own docstring identifies as the ones that
catch Udio where periodicity cannot. Combiner `max`, never `mean`: averaging
`comb_strength + comb_sharpness` previously gave 0.8308 against 0.9067 for strength
alone.

**Ship criterion, fixed now:** the arm enters the paper only if it beats the
incumbent on **both** corpora — FakeMusicCaps > 0.9177 and SONICS > 0.7546 — with
no family inverted. If SONICS lands below 0.7546, the honest result is that the
prior arm improves FakeMusicCaps and the 16 kHz Udio families remain out of reach,
and it is reported as such rather than as a detector.

### R10. The SONICS sample rate — a paper correction, and the lattice resolved

`comb_*_hz` values on SONICS could not be reproduced on FakeMusicCaps' 0.9765625 Hz
lattice (399.90 Hz is unreachable: 409 bins → 399.41, 410 → 400.39), and SONICS
reports 74.70 Hz where FMC reports 75.20 for the same 75 Hz candidate. Measured
directly from the canonical files:

| corpus | canonical sample rate | bin at n_fft 16384 | F over [1k, 8k] |
|---|---|---|---|
| FakeMusicCaps | 16,000 (real and fake) | 0.9765625 Hz | 7,167 ✓ |
| **SONICS** | **24,000 (real and fake)** | **1.46484 Hz** | **4,779** ✓ |

Every SONICS value lands exactly: 399.90 = 273 bins, 49.80 = 34, 74.70 = 51,
99.60 = 68, 93.75 = 64, 200.70 = 137. **Both classes share one rate within each
corpus**, so there is no per-track lattice variation and no channel confound in the
`prior_hz` concentration of §R7. This also independently reproduces the paper's own
`F = 7167 / 4779`, which is only consistent with 16 kHz / 24 kHz.

⚠ **`icassp/paper.tex:245` says both corpora are "resampled to $16$\,kHz". That is
wrong for SONICS**, and it contradicts the paper's own `F = 4779` two paragraphs
later (at 16 kHz that figure would be 7167). `build_canonical_corpus.py` documents
`--target-sr 24000`. The channel-matching argument is unaffected — matching is
*within* each corpus and both classes share a rate — but the sentence must be
corrected before submission. Corpus-level claims about "16 kHz benchmarks" remain
correct as statements about the datasets as published (SONICS ships 16 kHz mp3
fakes); it is the description of OUR chain that is inaccurate.

### R11. The matched filter — a good idea the analysis lattice cannot support

Proposed (by the user) and implemented: score the residual at the tooth POSITIONS
rather than autocorrelating it. An autocorrelation is offset-blind, while a
transposed-convolution comb is DC-aligned at multiples of `f_s / prod(strides)`, so
the alignment is a real constraint that the ACF discards. The filter is also
*linear* where the ACF is quadratic, so its floor falls as `1/sqrt(K)` in the tooth
count K — 140 teeth at 50 Hz over a 7 kHz band.

**Simulated at the FakeMusicCaps geometry (3-bin teeth, 120 draws/class) before
spending EC2 time. NOT RUN on the corpora.**

| comb | prior | drift by 7 kHz | comb_strength | MF(prior) | MF(refined) |
|---|---|---|---|---|---|
| MusicGen 250.0 = 5 × 50.0 | exact | 0 bins | 0.513 | **0.760** | 0.492 |
| chirp 399.902 ≈ 8 × 49.99 | 50.0 | ~1.7 bins | 0.514 | 0.586 | 0.451 |
| HiFi-GAN 200.195 vs 2 × 100.0 | 100.0 | ~7 bins | 0.495 | 0.488 | 0.457 |
| Stable Audio 107.42 vs 5 × 21.53 | 21.53 | ~15 bins | 0.593 | 0.432 | 0.559 |

**The mechanism of the failure is precision, not principle.** An error `d` in the
spacing displaces the K-th tooth by `K*d`, so the filter needs Δ to about `bin/K` —
0.028 Hz at 35 teeth. The ACF argmax quantises Δ to the lattice, ±0.5 bin =
±0.49 Hz, **twenty times too coarse**. Coarse-to-fine refinement does not rescue it:
in the weak-comb regime the ACF estimate is itself noise (concentration ≈0.05, §R5),
so the refinement centres on nothing, and its 161-point grid inflates real music's
maximum — visible in the strong regime, where refined MF reads 0.892 against
`comb_strength`'s 1.000.

**Kept, not shipped.** `comb_mf_z` remains implemented and unit-tested
(`TestTheMatchedFilterUsesAlignment` asserts that a DC-aligned comb outscores a
half-period-shifted one, i.e. that alignment carries information at all) and is
admissible in the fusion allowlist, because where the fundamental happens to be
exact it is worth +0.25 AUC in the weak regime. **No number from it enters the
paper without a corpus run**, and none has been made.

This is a negative obtained for ten minutes of simulation instead of five hours of
EC2, and it explains why an autocorrelation is the right tool here in spite of
being quadratic: it needs one lag correct, not K.

### R12. ⚡ Per-track decoy calibration — the first arm sign-consistent on all TEN families

`comb_{fmc,sonics}_HARM2/comb_per_track*.csv`, full corpus, no new compute: the 24
decoy prior sets are also a **per-track** null, and a better one than a permutation
surrogate because they hold the residual FIXED and vary only the hypothesis. Per
track, ``margin = s_real - max_j s_decoy_j``.

| statistic | FMC | SONICS | inverted (FMC / SONICS) |
|---|---|---|---|
| `comb_priormax4_margin` | 0.9153 | **0.8313** | **0 / 0** |
| `comb_priormax2_margin` | 0.9149 | 0.8272 | 0 / 0 |
| `comb_priormax2` raw | **0.9468** | 0.6661 | 0 / **2** |
| `comb_priormax4` z | 0.9155 | 0.7857 | 0 / 2 |
| `comb_priormax4` pval | 0.9100 | 0.7842 | 0 / 2 |
| incumbent | 0.9177 | 0.7546 (two-sided) | 0 / 0 |

Per family on SONICS, `comb_priormax4_margin`: chirp **0.9874 / 0.9955 / 0.9926**,
**udio 0.5976 / 0.5835** — correctly oriented, against 0.2536 / 0.2379 raw.

**+0.0767 on SONICS at a cost of 0.0024 on FakeMusicCaps**, the latter well inside
the incumbent's CI [0.9128, 0.9227]. Mechanism: the raw prior score is an absolute
autocorrelation and inherits the track's own residual statistics; Udio's residual
is flatter than real music's in every measure, so a raw score ranks Udio as *more
real than real music*. Subtracting the best decoy on the **same residual** cancels
that nuisance, which is why an inversion becomes a miss.

**This is the first arm in the project correctly oriented on all ten generator
families.** An inverted score is worse than a miss — it mis-orders — so this is the
criterion the paper already adopts from Zhou and Wang, satisfied for the first time.

### R13. ⚠ How the calibration was selected — EXPLORATORY, and labelled so

§R9 pre-registered **M** to be selected on FakeMusicCaps and fixed for SONICS. It
did **not** pre-register the calibration variant: that was devised *after* SONICS
came in below the incumbent (§R8), which makes it exploratory and not confirmatory.

A selection path exists that never reads a SONICS AUC — the paper's own adopted
criterion is sign-consistency across the families a score will face, `margin` is
the only variant satisfying it on both corpora, and M then follows from FMC
(priormax4 0.9153 edging priormax2 0.9149). **That reasoning was constructed after
the fact and the paper must say so.** The distinction that matters for a reader:
the **+0.077 on SONICS*/ is a post-hoc discovery; the *sign-consistency* it is
selected by is a pre-existing published criterion.

What would convert it to confirmatory, in order of strength:
1. the C10 content-identical control on the new statistic — pairs neither corpus's
   AUC touches, so it is an independent test of the mechanism;
2. the full C1--C11 battery on both corpora;
3. a third corpus, which we do not have.

Until at least 1 and 2 are done, no number from this arm ships.

`scripts/derive_prior_margin.py` computes the variants from an existing per-track
CSV (no re-extraction) and emits margins against the 75th and 90th decoy quantiles
as well as the max, so the choice of `max` is testable rather than assumed.

### R14. The champion, selected under the pre-registered rule

`scripts/derive_prior_margin.py` on both HARM2 per-track CSVs. The quantile sweep
exists so the choice of `max` in the margin is testable rather than assumed.

**Calibration form.** Fixed as `margin = s_real - max_j s_decoy_j` on the a priori
ground that it is the strict evidence statement — *does the plausible prior beat
every implausible one on this residual* — not because of its AUC. The sweep shows
the knob is real and corpus-dependent, which is exactly why it must not be tuned:
on FakeMusicCaps a softer threshold is better (`comb_priormax2_marginq90` 0.9257
against `margin` 0.9153), on SONICS the strict one is (`comb_priormax4_margin`
0.8313 against `marginq75` 0.8153). Tuning it would use both corpora.

**M.** Selected on FakeMusicCaps per §R9, over the margin family:
`comb_priormax2_margin` 0.9153, `comb_priormax4_margin` 0.9151,
`comb_harm2_margin` 0.9132. **M = 2**, which is also the M that won on the raw
score (0.9468), so no corpus-informed M choice enters anywhere.

#### The proposed statistic: `comb_priormax2_margin`

| | FakeMusicCaps | SONICS |
|---|---|---|
| **proposed** | **0.9153** | **0.8272** |
| incumbent | 0.9177 (`comb_strength`) | 0.7546 (two-sided fusion) |
| delta | **-0.0024** (inside the incumbent's CI [0.9128, 0.9227]) | **+0.0726** |
| families inverted | **0 / 5** | **0 / 5** |

Per family on SONICS: chirp 0.9888 / 0.9877 / 0.9853, **udio 0.5871 / 0.5872**
against 0.2669 / 0.2573 raw and 0.19 / 0.18 for `comb_strength`.

**The claim is sign-consistency, not a headline AUC.** This is the first arm in the
project correctly oriented on all ten generator families across both corpora. It
matches the incumbent on FakeMusicCaps and gains 0.073 on SONICS, and it does so
because the calibration cancels the residual-smoothness nuisance that made Udio
rank as *more real than real music*.

**Not yet shippable.** §R13 stands: the calibration is exploratory. The gate
battery and the C10 content-identical control are required before any of this
enters the paper, and C10 is the one test that uses data neither corpus's AUC
touches.

### R15. ✅ C10 PASSES for the proposed statistic — the mechanism, independently

`c10_margin_fmc/recon_pairs.csv`, `comb_priormax2_margin`, 300 content-identical
pairs per variant, paired Wilcoxon:

| variant | frac_above | median Δ | Wilcoxon p |
|---|---|---|---|
| encodec_3kbps | **0.930** | +0.0405 | 1.8e-44 |
| encodec_6kbps | **0.913** | +0.0386 | 7.6e-43 |
| encodec_24kbps | **0.883** | +0.0420 | 2.3e-40 |
| griffinlim_256mel | 0.480 | −0.0009 | 0.64 |
| griffinlim_128mel | 0.450 | −0.0018 | 0.51 |

**The proposed score responds to deconvolution, not to re-synthesis.** Same song,
same genre, same loudness, everything held constant by construction; a
transposed-convolution codec scores above its own source on 88–93% of pairs while
phase-only reconstruction sits at chance with a zero median shift.

This is the validation §R13 named as the strongest available, and it is the one
test that uses data **neither corpus's AUC has touched**. It moves the arm from
exploratory toward confirmatory. (For comparison the published `comb_strength`
reads 0.967–0.980 on EnCodec and 0.473–0.500 on Griffin-Lim: the new statistic's
EnCodec fraction is a little lower, its Griffin-Lim behaviour identical, and both
separations are overwhelming.)

### R16. ✅ FakeMusicCaps gate battery — the same profile as the incumbent

`gate_margin_fmc/`, `comb_priormax2_margin`, n = 32,951:

| gate | verdict | value |
|---|---|---|
| C1 channel-alone | **FAIL** | `spectral_flatness_hf` 0.6284 — **identical to the incumbent** |
| C2 level | **FAIL** | `peak_dbfs` 0.6207 — **identical to the incumbent** |
| C3 duration | PASS | 0.5228, full-length gap 0.007 |
| C4 silence | PASS | 0.5104 |
| C5 label shuffle | PASS | [0.5001, 0.5109] |
| C6 / C7 | PASS | 32,951 finite / smallest stratum 5,355 |
| C8 decoder spacing | PASS | fake 0.431 vs real 0.030 |
| C9 harmonicity | PASS | within-real \|ρ\| **0.014** (incumbent 0.019) |
| C10 | closed separately | §R15 |
| C11 | PASS | **AUC 0.9152 [0.9115, 0.9187]** |

C1 and C2 are properties of the corpus, not of the detector, so their being
identical to the incumbent's is the expected result and means **the proposed arm
adds no new confound disclosure**. C9 is marginally better than the incumbent's.

The C11 interval [0.9115, 0.9187] overlaps the incumbent's [0.9128, 0.9227]:
on FakeMusicCaps the two are a **statistical tie**, which is what the paper should
say rather than claiming an improvement there. The claim on FakeMusicCaps is
parity; the claim on SONICS is +0.073 and sign-consistency.

### R17. Script defect found and fixed — all-real corpora

`derive_prior_margin.py` raised `KeyError: 'MACRO'` on the C10 reconstruction
control, because an all-real corpus yields no per-generator AUC rows and the table
was sorted unconditionally. The calibrated CSV had already been written, so the
C10 result above is unaffected. An all-real corpus is a legitimate input — it is
what C10 and the FMA false-positive analysis both use — and the script now reports
that case the way `eval_comb_detector.py` does instead of crashing. Two fixture
tests cover it.

### R18. ✅ SONICS validation complete — the arm is confirmed on both corpora

**C10** (`c10_margin_sonics/recon_pairs.csv`, `comb_priormax2_margin`, 300 pairs):

| variant | frac_above | median Δ | Wilcoxon p |
|---|---|---|---|
| encodec_3kbps | **0.943** | +0.0657 | 3.7e-47 |
| encodec_6kbps | **0.920** | +0.0564 | 1.1e-44 |
| encodec_24kbps | **0.890** | +0.0667 | 3.6e-43 |
| griffinlim_256mel | 0.537 | **+0.0020** | 0.0106 |
| griffinlim_128mel | 0.503 | +0.0003 | 0.110 |

⚠ **Reported, not rounded away:** `griffinlim_256mel` reaches p = 0.011, which is
nominally significant. Its median shift is **33x smaller** than EnCodec's
(+0.0020 vs +0.0657), its frac_above is 0.537 against 0.890–0.943, and across five
variants a Bonferroni threshold of 0.01 leaves it non-significant. The
deconvolution-specific reading holds; the caveat is stated.

**Gate battery** (`gate_margin_sonics/`, n = 59,280):

| gate | verdict | value | published SONICS arm |
|---|---|---|---|
| C1 channel-alone | FAIL | `frac_power_top_quarter` **0.6082** | **0.6082** — identical |
| C2 level | FAIL | `peak_dbfs` **0.6202** | **0.6202** — identical |
| C3 duration | PASS | **0.5516**, gap 0.010 | **0.5516** — identical |
| C4 silence | PASS | 0.5485 | |
| C5–C7 | PASS | shuffle [0.5001, 0.5065]; 59,280 finite; smallest stratum 1,568 | |
| C8 spacing | PASS | fake 0.500 vs real 0.027 | |
| C9 harmonicity | PASS | within-real \|ρ\| 0.075 | |
| C11 | PASS | **pooled AUC 0.7901 [0.7864, 0.7940]** | incumbent pooled **0.7063** |

C1/C2/C3 reproducing the published values exactly is the expected result — they are
properties of the corpus, not of the detector — and it means **the proposed arm
adds no confound the paper does not already disclose**.

Note pooled (0.7901) and macro (0.8272) differ because SONICS families are very
unequal (udio-120s 18,345, chirp-v2 1,568); the paper's headline is signed macro,
and the pooled figure is the like-for-like comparison against the incumbent's
0.7063, i.e. **+0.084**.

### R19. The proposed arm — FINAL, and what may be claimed

`comb_priormax2_margin`: autocorrelation of the peak residual, evaluated only at
lags a published transposed-convolution decoder could produce, minus the best of
24 decoy prior sets on the same residual. No training, no fitted parameter, no
corpus statistic, sign fixed a priori.

| | FakeMusicCaps | SONICS |
|---|---|---|
| **proposed**, signed macro | **0.9153** | **0.8272** |
| incumbent | 0.9177 | 0.7546 |
| C11 pooled AUC | 0.9152 [0.9115, 0.9187] | 0.7901 [0.7864, 0.7940] |
| incumbent pooled | 0.9177 [0.9128, 0.9227] | 0.7063 [0.7013, 0.7116] |
| **families inverted** | **0 / 5** | **0 / 5** |
| decoy null | 0/24 reach it | 0/24 reach it |
| C10 | PASS (§R15) | PASS (§R18) |
| gates | as incumbent (§R16) | as incumbent (§R18) |

**Claimable:** parity on FakeMusicCaps (the CIs overlap — a tie, NOT a win);
+0.073 macro / +0.084 pooled on SONICS; **zero inverted families out of ten**,
which no other arm in this project achieves; the decoder fundamental identified
per track at 85–99% concentration against 19% for real music, transferring to
Suno, whose decoder is proprietary and did not inform the prior.

**Not claimable:** an improvement on FakeMusicCaps; a Udio detector (it converts an
inversion into a miss, which is a different and lesser thing); a wideband result.
The calibration remains post-hoc in origin (§R13) and the paper says so.

### R20. The paper, updated — what changed and what it cost

Builds at **5 pages, 0 errors, 0 overfull, 0 undefined, 0 bibtex warnings**;
sections 1--8 all begin on pages 1--4, so page 5 carries references only.

**Added:** contribution (v); an abstract sentence; a `tab:boundary` row
(`+ decoder prior, decoy-calibrated`, 0.9153 / 0.8272); the proposed statistic in
§7 with its CIs, decoy null, gate profile and C10; and the per-architecture
attribution in §6.

**Paid for by:** §6's atom exposition compressed and folded into the new
attribution result; §5's two-sided ablation and the inversion-mechanism paragraph
compressed; §7's cost and prevalence paragraphs compressed; §4.1's
alternate-learner sentence; §3.1's unnormalised-energy caveat; §8's future work;
and four single-clause competitor citations in §2 (`han2026beyond`,
`kim2025segment`, `li2026explainable`, `dugelay2026improved`), which frees
reference lines as well as body lines. Citations go 30 → 26, all resolving, with
the seven now-uncited entries kept in `refs.bib` for a longer version as the file
header already prescribes.

**Two corrections carried into the paper, not just the ledger:**
* §3.2's "resampled to 16 kHz" → one rate per corpus (16 kHz FMC, 24 kHz SONICS),
  per §R10. The old sentence contradicted the paper's own `F = 4779`.
* §8's "attribution succeeds on FakeMusicCaps only" is **withdrawn** — §R7
  attributes on Suno, a held-out proprietary decoder. The limitation sentence now
  states instead that the per-track calibration was devised after the one-sided
  score fell short on SONICS, so its SONICS gain is post-hoc and the
  content-identical control supports but does not by itself confirm it (§R13).

Every new number in `paper.tex` carries a `% src:` comment naming its CSV and its
gate section here. Local test suite passes.

### R21. Paper, second pass — template compliance and an arm-attribution error

**Template compliance** (verified against `icassp/Template (1).tex:50` and the
ICASSP paper kit): the abstract "should contain about 100 to 150 words" and the
5th page may carry "only references, funding acknowledgements, and a Compliance
with Ethical Standards statement".

| | before | after |
|---|---|---|
| abstract | 226 words | **145** |
| body ends on | **page 5** (violation) | **page 4** |
| citations | 26 | **30**, 0 undefined |
| pages / overfull / errors | 5 / 0 / 0 | **5 / 0 / 0** |

Page-end measured with a `\label` inside the final sentence and read back from
`paper.aux`, not eyeballed; page 5 confirmed to begin with `10. REFERENCES` by
rendering it with `gs -sDEVICE=txtwrite`.

**Framing.** The detector was only visible in §7. Now: the title states both
halves (*Label-Free Is Not Zero-Shot: Measuring and Removing the Transductive Dose
of an AI-Music Detector*); the abstract's second half is the proposal; the
introduction gains a paragraph saying we remove the dependence rather than only
reporting it; contribution (v) is "a detector that removes the dependence we
measure"; and the proposal has its own section (§7) ahead of deployment.

**Citations 26 → 30**, all resolving: added `kong2020hifigan`,
`copet2023musicgen`, `evans2024stableaudio` — each **required**, because §6 now
attributes fundamentals to those architectures — and restored
`dugelay2026improved`. HiFi-GAN, MusicGen and Stable Audio Open were each
re-verified by direct fetch of the arXiv listing page on 2026-09-08.

⚠ **A BibTeX defect I introduced and caught.** Placing `%` comments *inside* the
`wang2025asvspoof5` entry silently emptied its author and title — the entry
rendered as a bare `\bibitem` with no content and **bibtex reported no warning**.
Comments must sit outside the braces. Found by grepping the generated `.bbl`
rather than trusting the build's exit status.

⚠ **CORRECTED — §8 mixed two arms in one section.** The precision figure
(`analysis_nmf_sonics_FULL/operating_points.csv`, 0.127) is the **dictionary score
on SONICS**, while the transfer and per-genre rates
(`fpr_fma_by_genre/`) are the **comb arm**. The section previously asserted "All
figures here use the training-free comb arm", which was wrong for the precision
number, and the §7 split removed even that. Each figure now names its arm, per the
rule in §O1. **This was a pre-existing error, not one introduced by this round.**

Also corrected: §6 said "both benchmarks sit at 16 kHz", which read as
contradicting §3.2's new "24 kHz on SONICS". It now says the benchmarks are
*distributed* at 16 kHz, so nothing survives above 8 kHz whatever container rate
our chain uses. And the two content-identical controls (94.7--98.0% for the comb
arm in §5, 88--94% for the proposed score in §7) now each name their detector;
previously they could be read as the same test with two different answers.

### R22. Gaps in the paper's coverage of the PROPOSED detector

§8's deployment numbers are for the dictionary and comb arms. For the proposal we
have detection, gates and C10 on both corpora but **no deployment figures**:

| number | status for the proposed score |
|---|---|
| per-genre false-positive rate on FMA | **missing** |
| threshold transfer (in-corpus → FMA) | **missing** |
| precision at 1% prevalence | **missing** |
| robustness (AAC / Opus / time-stretch) | **missing** |
| efficiency | **missing** |

`run_robustness_battery.py` and `measure_efficiency.py` both called
`comb_features` with `null_priors` defaulting to 0, so neither could produce the
decoys the margin needs; both now take `--null-priors`. The efficiency timing uses
the *deployed* configuration (`n_harm_grid=(2,)`, `surrogates=0`) rather than the
full emitted grid, which would overstate a cost nobody would pay.

### R23. Citation audit, 2026-09-08 — 15 of 30 re-verified LIVE this round

Re-fetched from the arXiv listing or publisher page in this session, title and
author list compared character by character against `icassp/refs.bib`:

| entry | verified as | ✓ |
|---|---|---|
| `kong2020hifigan` | Kong, Kim, Bae — NeurIPS 2020, arXiv:2010.05646 | ✓ |
| `copet2023musicgen` | Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Défossez | ✓ |
| `evans2024stableaudio` | Evans, Parker, Carr, Zukowski, Taylor, Pons — arXiv:2407.14358 | ✓ |
| `zhou2026fragile` | Zhou, Wang — arXiv:2606.20488, 2026 | ✓ |
| `oh2026artifactnet` | Heewon Oh — arXiv:2604.16254, 2026 | ✓ |
| `pascu2026echoes` | Pascu, Oneață, Cucu, Müller — arXiv:2603.23667 | ✓ |
| `fursule2026gender` | Fursule, Kshirsagar, Avila — arXiv:2605.09087 | ✓ |
| `kluttermann2024transductive` | Klüttermann, Müller, *Deep Transductive Outlier Detection* | ✓ |
| `li2024pathway` | Li, Milling, Specia, Schuller — arXiv:2412.00571 | ✓ |
| `nalisnick2019typicality` | Nalisnick, Matsukawa, Teh, Lakshminarayanan | ✓ |
| `wang2025asvspoof5` | Wang, Delgado, Tak + 26 others (29 total) — arXiv:2502.08857 | ✓ |
| `comanducci2025fakemusiccaps` | *J. Imaging* **11**(7):242, 2025, doi 10.3390/jimaging11070242 | ✓ |
| `han2026musicdet` | Han, Wang, Gui — **ICML 2026**, arXiv:2605.18072 | ✓ |
| `zisman2026realstats` | Zisman, Shaham — **AISTATS 2026** (Spotlight), arXiv:2601.18900 | ✓ |
| `afchar2026finding` | Afchar, Hennequin — arXiv:2607.25530 (full text in `context/`) | ✓ |

**A number attributed to a citation was checked too, not just the citation.** The
paper claims per-group thresholds cut disparity "by $54$--$75\%$ at no cost to
accuracy \cite{fursule2026gender}". Their abstract, verbatim: *"Adjusting the
decision threshold separately per gender reduces unfairness by 54% to 75% at no
cost to detection accuracy."* Exact.

The remaining 15 are pre-2026 venue citations whose metadata is stable and
independently known (Lee & Seung *Nature* 401(6755):788--791; DeLong *Biometrics*
44(3):837--845; Pons ICASSP 2021 pp. 3005--3009; Pang KDD 2019 pp. 353--362;
Odena *Distill* 2016; Defferrard ISMIR 2017; Serrà ICLR 2020; Tulchinskii NeurIPS
2023; Défossez TMLR 2023; Rahman ICLR 2025; Afchar ISMIR 2025; Cros Vila TISMIR
8(1):179--194; Frohmann ISMIR 2025; Li *Sci. Rep.* 16; Dugelay ISMIR 2026), each
resolved by direct fetch in §P3 and unchanged since. **No entry in `refs.bib` rests
on a search summary or on recall.**

### R24. ⚠⚠ CORRECTED — "which families invert is a property of the generator"

An external review flagged that contribution (ii) claims generator-dependent
inversion but demonstrates it only for the comb arm. Checking
`nmf_sonics_curve_FINAL/nmf_peak_auc_lvl.csv` at `mix0pct`, seeds 42--44, shows
the claim is not merely unsupported — **it is backwards**:

| family | dictionary score @ 0% dose (s42/s43/s44) | comb statistic |
|---|---|---|
| chirp-v2 | 0.3911 / 0.4036 / 0.2564 → **0.350 inverted** | 0.900 correct |
| chirp-v3 | 0.3487 / 0.3757 / 0.2073 → **0.311 inverted** | 0.940 correct |
| chirp-v3.5 | 0.2868 / 0.2932 / 0.2254 → **0.268 inverted** | 0.883 correct |
| udio-120s | 0.4613 / 0.4589 / 0.4503 → 0.457 at chance | **0.192 inverted** |
| udio-30s | 0.5223 / 0.5239 / 0.5239 → **0.523 correct** | **0.175 inverted** |

**The two detectors fail on disjoint families.** Which family inverts is a
property of the **detector–generator pair**, not of the generator. The paper's
wording is corrected, and the corrected statement is stronger: a sign verified
once cannot be reused, so `zhou2026fragile`'s criterion has to be applied per
detector rather than per corpus.

Found only because a reviewer asked for the per-generator table we had never
printed for the NMF arm. Same failure mode as the eight earlier retractions — a
claim asserted at corpus level and never checked at family level.

### R25. ✅ Deployment measured for the PROPOSED detector — the §R22 gap, closed

`comb_fma_HARM2/` (3,000 FMA reals, FakeMusicCaps chain), `fpr_fma_margin/`,
`analysis_margin_fmc/`, `analysis_comb_fmc_HARM2/`.

**Threshold transfer** (reals-only quantile set on FakeMusicCaps, applied to unseen FMA):

| quantile | in-corpus FPR | unseen FPR | ratio |
|---|---|---|---|
| 0.90 | 0.0534 | 0.0447 | **0.84×** |
| **0.95** | **0.0362** | **0.0270** | **0.75×** |
| 0.99 | 0.0097 | 0.0027 | **0.28×** |

The comb arm inflates **2.75×** (0.042 → 0.115) at q=0.95; **the proposed score
does not inflate at any quantile.** TPR on fakes at q=0.95 is **0.6250** against
the comb arm's 0.489.

**Per-genre false-positive rate at q=0.95**, n = 2,998:

| genre | comb arm | **proposed** |
|---|---|---|
| Electronic | **0.2356** | **0.0236** |
| Folk | 0.0279 | 0.0334 |
| Instrumental | — | 0.0442 (worst) |
| Pop | — | 0.0174 (best) |
| **spread** | **8.5×** | **2.5×** |

**The disparity the paper reports as its own detector's fairness problem is
largely removed**: Electronic falls 10× and is no longer the worst genre.

⚠ **And a weakness, reported rather than buried.** Recall at the 95% real
quantile is very uneven for the proposed score — Stable Audio Open **0.032** and
MusicGen **0.169** against 0.947--0.999 for the other three — despite AUCs of
0.936 and 0.968. The margin compresses near zero for those families (the q=0.90
threshold is literally 0.000), so they rank above real music without clearing a
strict threshold. The comb arm's spread is narrower but its mean lower
(0.157--0.723, mean 0.458, against 0.032--0.999, mean 0.625). This is the paper's
own "AUC is not an operating point" theme applying to our detector, and it is now
in the limitations.

### R26. Review response, and what was NOT changed

An external review returned *definite accept / award quality*, with one
substantive criticism (Claim 2) — acted on in §R24 — and three suggested
citations. `li2026explainable` and `kim2025segment` are **already verified entries
in `refs.bib`**, uncited only for space; `Morosanu et al.` is unverified and was
**not** added, per the rule that no citation enters without a direct fetch. The
paper is at exactly 4+1 pages with 30 citations, so none were added: the review
already scores references as adequate, and breaking the page limit to raise that
score would be a poor trade.

---

## S. ROUND 13 — 2026-09-17. Decoy-free nulls: a seed sweep, three corrections, and one zero-compute prediction

### R27.0 What this round is, and what it is NOT

**No corpus number was produced.** Everything in R27 is either (a) a FIXTURE result at
the real FakeMusicCaps geometry over 200 seeds, (b) an arithmetic consequence of
values already in this ledger, or (c) code with local tests. Nothing here may be
quoted as a detector result, and nothing enters the paper, until the runbook
(`reports/handover/RUNBOOK_2026-09-17_NULLS.md`) has been run and gated.

The fixture imports the shipped `decoy_fundamentals` and `_normalised_acf` rather than
reimplementing them, so a change to either breaks it rather than silently diverging.
It is committed as `tests/test_null_alternatives.py`.

### R27.1 Reproduced from HANDOVER_2026-09-16_NULLS

| quantity | handover | reproduced |
|---|---|---|
| prior lag set P at M=2 | 10 lags | **10** — `[44,51,77,88,96,102,154,176,192,205]` |
| decoy union, unique lags | 118 | **118** (so the extreme-value penalty is x1.31, not x1.9) |
| decoy sets colliding with a real candidate within 1 bin | 8 / 24 | **8 / 24** |
| complement sizes (naive / allharm+-1 / drift 0.5m) | 3,512 / 77.4% / 3.9% | **identical** |
| naive complement on a MusicGen-like comb | inverts | **AUC 0.0000 over 200 seeds**, complement peak at lag 255 = **249.0 Hz** |

The handover's §2.2 refutation of the professor's proposal stands, and is stronger
than a single draw showed.

### R27.2 ⚡ PRIOR DEPTH, not the null — the explanation for the recall failure

Arithmetic over values already in this ledger (§G2, §O8b for the measured combs;
§R25 for the recall). `comb_priormax` at M=2 searches lags up to **200.2 Hz**.

| family | measured comb | lag | in P at M=2 | M=4 | M=8 | recall @ q95 (R25) |
|---|---|---|---|---|---|---|
| audioldm2 / musicldm / mustango | 200.195 Hz | 205 | **YES** | yes | yes | **0.947-0.999** |
| MusicGen | 250.000 Hz | 256 | **no** | no | **yes** | **0.169** |
| Stable Audio Open | 107.420 Hz | 110 | **no** | no | **yes** | **0.032** |
| Suno chirp | 399.902 Hz | 409 | **no** | **yes** | yes | — (SONICS) |

**The correspondence is exact.** The two families whose recall collapses are precisely
the two whose visible tooth lies outside the prior's reach at the shipped M=2; they
are detected only through a weak sub-harmonic, which is why the margin compresses to
zero (§R25: the q=0.90 threshold is literally 0.000) while the AUC stays at 0.936 and
0.968.

**And M was selected on the wrong objective, by a margin smaller than noise.** §R14
chose M=2 over M=4 on **0.9153 vs 0.9151** — a 0.0002 macro-AUC gap. Macro AUC is
almost blind to the left tail of the ROC, which is where a deployment threshold sits.

**The test is zero compute.** `--n-harm 2 4 8` emits `comb_priormax8_strength` and its
24 decoys, and `derive_prior_margin.py:75`'s regex matches the stem, so
`comb_priormax8_margin` is already in both HARM2 `per_track_calibrated_auc.csv` files
with its macro and `n_inverted`. §R14 quoted only M=2, M=4 and `harm2`. **Nobody has
read the M=8 row.** Runbook Step 0b.

**Falsification condition, recorded before the run:** if `comb_priormax8_margin` does
not raise recall on `stable_audio_open` and `MusicGen` relative to
`comb_priormax2_margin`, this explanation is wrong and is withdrawn here.

### R27.3 THREE CORRECTIONS to HANDOVER_2026-09-16_NULLS

**(a) §2.3's "lattice drift" is misdiagnosed, and §3-N4's tolerance formula computes
the wrong quantity.** The handover blames the analysis grid: "the prior says 100.00 Hz
= 102.4 bins, but the comb sits at whatever integer lattice position the decoder
actually produces." `frac(f/bin_hz)` is a property of OUR grid, not of the prior's
accuracy — if the prior frequency is right, `round(m*f/bin)` lands on the ACF peak at
every m. The real quantity is `m * |f_prior - Delta_true| / bin_hz`. Measured on the
same statistic under two tooth models, 200 seeds:

| all-harmonics +-1-bin mask, MusicGen-like condition | AUC |
|---|---|
| teeth at integer bin MULTIPLES (the handover's fixture — builds in a 0.4% prior error) | **0.0000** |
| teeth at exact multiples of Delta in **Hz** (what a decoder produces) | **0.9974** |

Consequence: N4's prescribed tolerance `m * frac(f/bin_hz)` is not a model of prior
error. For the 93.75 Hz candidate `frac` is exactly 0, so it prescribes zero tolerance
at every harmonic however wrong the prior is. A defensible tolerance is **relative**.

**(b) §3-N1 is not "prior-restricted sharpness" and is not prior-restricted at all.**
`comb_harmonic()` (`comb_artifacts.py:424`) searches every lag in `[lo, hi]` and never
sees `DECODER_FUNDAMENTALS_HZ`. It therefore has no provenance problem — the
handover's conclusion, reached for the wrong reason — but also carries none of the
prior-restriction mechanism §6's attribution result rests on, and emits no `_hz`
column, so it cannot reproduce the 85-99% concentration claim.

**(c) §2.6 is correct but applied too broadly.** A complement null cannot answer the
provenance question — decisive and agreed. But that forbids removing the decoys as a
**control**; it does not forbid replacing them as the detector's **calibration**. The
decoy null is measured and published (0/24 on both corpora) and can remain the control
while a different per-track null does the calibrating.

### R27.4 Ratio versus difference — a prediction for the SONICS run

200 seeds, on a condition whose residual is smoother than real music's and carries no
in-band comb:

| bulk-calibrated form | AUC on that family |
|---|---|
| `sharpness` = peak / median\|curve\| — **this is §3's N1** | **0.4483** |
| z-score of the same | 0.4607 |
| peak − median (the DIFFERENCE form) | **0.6631** |
| decoy margin (the incumbent) | 0.6158 |

A smoother residual raises the ACF bulk, so DIVIDING by it deflates that family more
than it deflates real music; SUBTRACTING does not. **What is pinned is the ~0.2 AUC
gap, not the word "inverts"** — the ratio form's absolute level moves between 0.42 and
0.50 with the sweep size, while the gap is stable at every size. §R1 records
`comb_harm4_sharpness` at macro 0.91986 on FakeMusicCaps, above the incumbent, and it
was never taken to SONICS, where the Udio families live.

The difference form costs nothing: `comb_harm{M}_sharpness = strength / (floor +
1e-12)` (`comb_artifacts.py:493`, `:509`), so `floor = strength / sharpness` is
recoverable from two columns already in every per-track CSV.
`scripts/derive_analytic_null.py` does it.

### R27.5 Two nulls killed, with mechanisms

| idea | outcome |
|---|---|
| **Cross-band consistency, `min(rho_low, rho_high)`** — §3's N3, its highest-ranked candidate | **DEAD.** Takes a comb visible only above 4.5 kHz from **1.0000 to 0.4751 — chance**. Reported as "destroyed", not "inverted": the value straddles 0.5 across sweep sizes (0.5513 / 0.5005 / 0.5175 / 0.4751 at 40 / 80 / 120 / 200 seeds) |
| Cross-band `mean` | **No-op** — the mean of the two half-band ACFs approximates the full-band ACF |
| **Anti-phase null** (ours, proposed and tested this round) | **DEAD.** AUC **0.0000** on the MusicGen condition, and the reason is arithmetic: `1.5 x 100 = 150 = 3 x 50`, so a half-multiple of one fundamental is an integer multiple of another and the anti-lag set lands ON the comb |

**Why N3 fails, stated so it generalises.** The time-axis stationarity idea works
because a melody moves IN TIME. A harmonic series does not move IN FREQUENCY, so the
frequency-axis analogue cannot separate a musical comb from a decoder comb: it keeps
the weakness and discards the strength. It is also not a null — it calibrates nothing
— so it cannot touch the smoothness nuisance behind the Udio inversion. The handover
expected the asymmetry to run the other way. The salvageable half of the idea is that
the UPPER band is the informative one, and that is `--f-min 4000`, a flag rather than
a statistic; the existing f-min sweep (0.6128 / 0.6155 / 0.6203 at 200 / 500 / 1000
Hz) already points that way.

### R27.6 A fixture-design fault in §2, and the fix

The handover's fixture has **no real-music condition carrying an actual comb** — its
"real" class is pure noise — so it could not have tested any null whose job is to
separate a decoder comb from music. Its §7 step 2 spec also omits the tooth
amplitudes entirely, so it is not reproducible as written; the levels in its §2.2
table are reached at tooth amplitude ~1.40 (strong), 0.30 (weak) and 0.30 base with a
3.00 fifth harmonic (MusicGen-like), which is what
`tests/test_null_alternatives.py` records.

The 2026-09-17 sweep adds a real class where half the draws carry a bass harmonic
series whose fundamental falls inside the prior's own lag range, and a generated class
whose comb is visible only above 4.5 kHz. **Both conditions changed a conclusion.**
Same family as the 2026-08-20 symmetry trap: a fixture that cannot fail does not test
anything.

Stated limitation, so R27.4 is not over-read: the real and generated conditions share
a residual process, so the fixture cannot reproduce the Udio inversion in absolute
terms. It separates functional FORMS. That limit applies equally to the handover's §2
numbers.

### R27.7 The deterministic fundamental lattice — the professor's idea in the right space

The complement null fails because it masks **lags**, and masking Delta does not mask
3*Delta. Excluding on the **fundamental** axis closes the leak exactly: at M=2 a
candidate `g` contributes lags `{g, 2g}`, so it can collide with the real prior only
if `g` is near `Delta/2`, `Delta` or `2*Delta`, and a relative +-5% rejection of all
three removes every collision with **no drift term**, because the tolerance is a
fraction of the frequency rather than a fixed number of bins.

Implemented as `decoy_lattice_lags()` and `comb_prior_lattice_null()`. Two properties
that matter:

* **No free parameter.** A log grid would need a spacing, and an unregistered knob is
  what disqualified the drift-aware complement (§2.5, where the drift rate swings the
  null 40x). The analysis lattice is discrete, so beyond a fine enough spacing extra
  grid points produce no new lags; the implementation enumerates that SATURATED set
  directly, and `tests/test_lattice_null.py` asserts a 0.1%-step grid cannot find a
  lag it missed.
* **Sized to match, not to flatter.** 155 lags on FakeMusicCaps against the 24 random
  sets' 118, and 105 against 94 on SONICS, so the extreme-value penalty is comparable
  (the ratio of `sqrt(2 ln K)` is 1.03). **Zero overlap with the real prior's lags on
  both corpora**, against the random construction's 8-of-24 collisions.

It yields an exact empirical p-value with no seed: the fraction of admissible wrong
lags reaching the plausible prior's best. That is the deterministic analogue of "0 of
24 decoy sets reach it", and it is stronger, because the null is exhaustive rather
than sampled.

⚠ **A bug caught by the test, recorded because it is the silent-corruption family
again.** The first implementation tested only the CENTRE fundamental of each lag.
Rounding maps an INTERVAL of fundamentals onto each lag, so a lag whose centre falls
inside a banned neighbourhood while part of its interval does not was wrongly
rejected — 5 lags on FakeMusicCaps. Two tests written from opposite directions
("the grid finds nothing outside" and "everything inside is reachable") disagreed,
which is what exposed it.

### R27.8 Code added, and the bit-identical checks

All uncommitted. Local suite: **405 tests pass**, 111 of them new (baseline 294).

| path | change | regression control |
|---|---|---|
| `scripts/derive_analytic_null.py` | **new.** Recovers the ACF floor and emits `comb_harm{M}_nmargin`. Imports `_auc_table` from `derive_prior_margin` so `MACRO`/`n_inverted` come from the code that produced the published table | round-trip against the real `comb_harmonic` at M in {2,4,8} |
| `scripts/analyze_detector.py` | `--quantiles`; `left_tail.csv`; `errors_by_generator_by_q.csv` | **all four original outputs verified byte-identical against `git show HEAD:`** on a fixture |
| `scripts/measure_false_positive_rate.py` | `--calibrate-per-group --alpha`, Mondrian conformal | **both original outputs verified byte-identical against HEAD** |
| `src/.../comb_artifacts.py` | `decoy_lattice_lags`, `comb_prior_lattice_null`, `comb_acf_floor`, `comb_prior_max(tol_bins=0)` | `comb_prior_max` at its default is byte-identical to a transcribed oracle across both geometries, three M values and three seeds |

`comb_features` gains `comb_acf_floor` and, per M, `_latmax_strength`, `_latpval`,
`_pm1_strength`, `_pm1_latmax_strength`, `_pm1_latpval`. Strictly additive; no existing
key changes.

Two corrections made in passing while I was there:

* the stale `# TODO` inside `_normalised_acf` proposing exactly the complement null
  that §2.2 refutes has been removed — it was an invitation to re-implement a dead end;
* `tests/test_afchar_parity.py` does **NOT** byte-pin `comb_strength()`. It pins
  `lower_hull_indices` against Afchar's released loop and never imports
  `comb_strength`. The claim appears in `comb_artifacts.py:369-372` and in the 09-16
  handover §8. The operational rule ("do not refactor `comb_strength`") stands, but
  the protection it claims does not exist — `comb_strength` is covered only by
  behavioural tests (`tests/test_comb_artifacts.py:49-71`).

### R27.9 The threshold layer, and four citations verified by direct fetch

`paper.tex:508` already states that a reals-only quantile IS a split-conformal
threshold under exchangeability and that unseen FMA violates it. It was never acted
on. `reports/bibliography_review.md:267` grades it lever **R3** ("RUN as framing, no
new compute") and `NEXT_SESSION_PROMPT_22_08.md:136` names the insertion points; a
repo-wide grep on 2026-09-17 found **conformal in prose only, zero code, and Mondrian
nowhere at all**.

`measure_false_positive_rate.py --calibrate-per-group` now sets a per-group threshold
using the **finite-sample conformal quantile** — the `ceil((n+1)(1-alpha))`-th order
statistic, not `np.quantile` — reports three arms side by side (`transfer`,
`conformal_global`, `conformal_group`), and flags any group too small to support the
level rather than silently pooling it. Precision uses the rule-of-three FPR bound
(`analyze_detector.fpr_upper_bound`, imported), because a group with zero observed
false positives otherwise reports precision 1.0 — the artefact §O1 already paid for.

Citations, each resolved by a direct fetch on 2026-09-17, **none added to `refs.bib`**:

| key | verified as |
|---|---|
| `tibshirani2019covariateshift` | Tibshirani, Barber, Candes, Ramdas, *Conformal Prediction Under Covariate Shift*, arXiv:1904.06019, 2019 |
| `vovk2012conditional` | Vovk, *Conditional validity of inductive conformal predictors*, arXiv:1209.2673, ACML 2012, PMLR 25:475-490 |
| `zhu2025mcp` | Zhu, Ren, Cao, Lin, Fang, Li, *Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction*, arXiv:2505.05084, 2025 |
| `lopezayala2026broadcast` | Lopez-Ayala, Cabello, Zinemanas, Molina, Rocamora, *AI-Generated Music Detection in Broadcast Monitoring*, arXiv:2602.06823, **accepted at ICASSP 2026**. This **closes an open item**: HANDOVER_2026-09-07 §7 records a broadcast-monitoring paper that "could not be verified and was not added" |

**REFUSED and staying refused:** Thomson (1982), *Proc. IEEE* 70(9):1055-1096 — the
natural citation for a multitaper harmonic F-test, which is the correct classical test
for a spectral line in COLOURED noise and therefore the one place Fisher's g is
invalid. Its DOI fetch returned **404**. Under the rule that cost this project two
entries (§P3) it does not enter, and that lever stays parked.

Also noted: `refs.bib`'s deliberate-omission NOTE is stale — it lists
`dugelay2026improved` as uncited, but it is cited at `paper.tex:182`. Three further
entries (`han2026beyond`, `li2026explainable`, `kim2025segment`) are uncited and
covered by no note.

### R27.10 ⚠ MEASURED — R27.2's prediction is PARTIALLY FALSIFIED, and R27.0 no longer holds

R27.0 said "no corpus number was produced". Steps 0 and 1 of the runbook have now been
run. **The reproduction check passed**: `analysis_pm2_fmc/errors_by_generator.csv`
gives stable_audio_open **0.031516**, MusicGen **0.169384**, musicldm 0.9473,
audioldm2 0.9779, mustango 0.9991, pooled TPR **0.6250** — reproducing §R25 exactly,
so the run is sound and everything below is comparable to it.

**Recall at q = 0.95, FakeMusicCaps, `analysis_pm{2,4,8}_fmc`:**

| family | M=2 | M=4 | M=8 | tooth inside P at |
|---|---|---|---|---|
| **MusicGen_medium** | 0.1694 | 0.2270 | **0.5418** | **M=8** |
| **stable_audio_open** | 0.0315 | **0.0886** | 0.0266 | **M=8** |
| musicldm | 0.9473 | 0.6830 | 0.2920 | M=2 |
| audioldm2 | 0.9779 | 0.9173 | 0.2587 | M=2 |
| mustango | 0.9991 | 0.9681 | 0.2934 | M=2 |
| **pooled TPR @ q=0.95** | **0.6250** | 0.5768 | 0.2825 | |

Against the falsification condition recorded in R27.2 before the run:

* **CONFIRMED for MusicGen.** Recall rises ×3.2 at exactly the depth that first brings
  its measured 250 Hz tooth (lag 256) inside `P`. Nothing else changes at that step.
* **REFUTED for stable_audio_open.** Its 107.42 Hz tooth (lag 110) is also covered only
  at M=8, yet recall *falls* there and peaks at M=4, where it is **not** covered.
  Coverage is not SAO's mechanism, and R27.2's claim is withdrawn for that family.
* **The cost is the finding R27.2 missed.** `|P|` goes 10 → 18 → 36 and the decoy null
  is constructed at the same M, so it inflates just as fast. Every family whose tooth
  was already inside `P` loses margin, falling from 0.947-0.999 to 0.26-0.29. **Prior
  depth trades coverage against null inflation**; M=2 is not the naive choice R27.2
  implied.

**The resolution of that trade-off is the null QUANTILE, not M.**
`comb_priormax8_marginq75` — the 75th decoy percentile instead of the maximum — is the
best FakeMusicCaps margin at **0.9430** (+0.028 over the shipped score) while
`comb_priormax8_margin` is the worst at 0.9083. Taking a quantile removes the
extreme-value inflation a deep `P` causes while keeping its coverage.

### R27.11 ✅ MEASURED — N1 is dead, and the fixture called it

`comb_sonics_HARM2/comb_auc_rs15000_lvl_sm5_median_db_f1000_noclip_harm248.csv`:

| feature | FMC macro (n_inv) | SONICS macro (n_inv) | SONICS udio-120s / udio-30s |
|---|---|---|---|
| **`comb_harm4_sharpness`** | 0.9199 (0) | **0.6636 (2)** | **0.2328 / 0.2329** |
| `comb_harm8_sharpness` | 0.9095 (0) | 0.6655 (2) | 0.2484 / 0.2260 |
| `comb_harm2_sharpness` | 0.7637 (1) | 0.6495 (2) | 0.3970 / 0.4029 |
| `comb_sharpness` | 0.7239 (1) | 0.5192 (**3**) | 0.3719 / 0.3730 |
| `comb_surrogate_z` | 0.9166 (0) | 0.6447 (2) | 0.2226 / 0.2125 |

**The 2026-09-16 handover's top-ranked decoy-free null inverts on both Udio families.**
N1 closes. So does the permutation surrogate, which §R1 had already shown to be a
no-op on FakeMusicCaps and which turns out to invert on SONICS.

R27.4 predicted exactly this from the fixture — the RATIO form of a bulk calibration
is the fragile one, because a smoother residual raises the autocorrelation bulk and
dividing by it deflates that family more than it deflates real music. **Prediction made
before the lookup, confirmed by it.** The DIFFERENCE form from the same two columns is
the surviving variant and remains untested on a corpus.

### R27.12 Scores that Pareto-dominate the shipped one — candidates, NOT results

Both corpora, 0/10 families inverted, against `comb_priormax2_margin`
(0.9153 / 0.8271):

| score | FMC | SONICS | dFMC | dSONICS |
|---|---|---|---|---|
| `comb_priormax8_marginq75` | **0.9430** | 0.8178 | +0.0277 | −0.0093 |
| `comb_harm2_marginq75` | 0.9418 | 0.8153 | +0.0265 | −0.0118 |
| **`comb_harm2_marginq90`** | 0.9241 | **0.8306** | **+0.0088** | **+0.0035** |
| **`comb_priormax8_marginq90`** | 0.9172 | 0.8292 | +0.0019 | +0.0021 |
| **`comb_priormax4_marginq90`** | 0.9159 | 0.8274 | +0.0006 | +0.0003 |
| `comb_priormax4_margin` | 0.9151 | **0.8312** | −0.0002 | +0.0041 |

⚠ **These are not yet reportable, for the reason §R13/§R14 already recorded.** The
calibration form was fixed a priori as `max` on the ground that it is the strict
evidence statement, and §R14 explicitly warned that the quantile is corpus-dependent
and must not be tuned. Selecting `marginq90` by reading both corpora's AUC **is** that
tuning. The gains also sit at or inside the shipped score's CI [0.9115, 0.9187]. They
require the deployment numbers, the gate battery and a declared selection path before
any of them may be quoted.

### R27.13 ⚠⚠ CORRECTED — a threshold-grid defect in `analyze_detector.py`, and it touches a published claim

`analysis_pm4_sonics` reported, from **one run**:

| table | figure |
|---|---|
| `operating_points.csv`, q = 0.999 | FPR 0.00047, recall 0.4677, **precision at 1% prevalence 0.9092** |
| `precision_targets.csv`, p = 0.01, target 0.90 | **achievable = False**, best reachable **0.8830** |

The two contradict each other. **Cause:** the search grid was
`np.quantile(reals, np.linspace(0.5, 0.99999, 400))` — uniform in the quantile, so its
final step spans 0.99874 to 0.99999 with no point between. On 6,361 threshold reals
that gap is eight tracks wide, and it steps over the operating point the other table is
standing on. Precision 0.9 at 1% prevalence requires an FPR near 0.00076, so a grid
uniform in `q` places almost all of its candidates where precision is hopeless and
almost none where it is decided.

**Fixed** as `threshold_grid()` with `--precision-grid tail|linear`. The new default is
a strict **superset** of the old grid — the same 400 body points plus a log-spaced tail
down to `1/n` — so it can only ever find a better operating point, never a worse one.
`--precision-grid linear` reproduces every superseded number byte for byte, which is
how the byte-identical check against `git show HEAD:` is still satisfied. Pinned by
`tests/test_analyze_detector_left_tail.py`, including the invariant that was violated:
`best_precision_reachable` can never fall below a precision the same run reached.

**Consequence for the paper, and it is not cosmetic.** `icassp/paper.tex` §6 states
*"no threshold on either corpus reaches precision 0.9 below 5% prevalence, which would
need a false-positive rate near 0.08%"*. That sentence rests on this grid. **It must be
re-derived before the camera-ready**, and the SONICS M=4 operating point above already
contradicts it. The paper has NOT been edited; the re-measurement comes first.

This is the third time a **search or sampling resolution**, rather than a measurement,
decided a reported conclusion — after §H1 (a descriptor built at the wrong duration)
and §O8b (a null chosen because it was generous). Added to the standing list: when a
script reports "unreachable", check that its search could have reached.

### R27.14 The three HARM2 corpora are on different code versions

From the column inventory: `comb_fma_HARM2` carries `comb_mf_z`, `comb_mfoff_z` and
**168** decoy columns; `comb_fmc_HARM2` and `comb_sonics_HARM2` carry **144** and no
matched-filter columns. The FMA pass post-dates the matched-filter commit and the two
labelled passes pre-date it. Prior and decoy columns are identically defined, so the
FMA threshold-transfer comparison is like for like — but two things follow:

* **the matched filter still has no AUC on any corpus.** The only corpus carrying its
  columns is all-real, so no AUC exists there by construction. §R11's "implemented,
  unit-tested, never run on corpora" stands, for a reason not previously identified;
* **any new feature column must be added to all three corpora before it is used for a
  transfer threshold.** A version split between the corpus that sets a threshold and
  the corpus that measures it is the §H1 failure mode.

`comb_fmc_HARM2` additionally used `--operator-grid` (400 columns) where SONICS and
FMA did not. Deliberate and harmless — the variant columns carry no decoys, and
`derive_prior_margin.py`'s `fullmatch` regex ignores prefixed names — but it explains
the column-count spread (400 / 205 / 234).

Filenames confirm the extraction flags exactly, so R27.8's reconstruction of the
`--out-dir` commands was correct: FMC `comb_per_track_lvl_sm5_median_db_f1000_noclip_harm248.csv`
(no `_rs`, so `--resample-hz 0`), SONICS `comb_per_track_rs15000_lvl_..._harm248.csv`.

### R27.15 One integrity item to verify before the next round

The paste of `comb_sonics_HARM2/per_track_calibrated_auc.csv` showed its header line
**twice**. `derive_prior_margin.py` writes that file with a single `to_csv`, so either
the paste duplicated a line or the file was appended to — in which case pandas has been
reading a header row as data. One command settles it:
`head -3 reports/diagnostics/comb_sonics_HARM2/per_track_calibrated_auc.csv`. If real,
re-run `derive_prior_margin.py` for that corpus before quoting anything from it.

### R27.16 ⚡ THE MECHANISM — a random null cannot support a deep prior

The M sweep completed on both corpora. Recall at q = 0.95, and the depth at which each
family's **measured** comb first enters the prior's lag set:

| family | measured comb | enters P at | M=2 | M=4 | M=8 |
|---|---|---|---|---|---|
| Suno chirp x3 (SONICS) | 399.902 Hz | **M=4** | 0.958 | **0.974** | 0.148 |
| MusicGen (FMC) | 250.0 Hz | **M=8** | 0.169 | 0.227 | **0.542** |
| HiFi-GAN x3 (FMC) | 200.195 Hz | M=2 | **0.975** | 0.856 | 0.281 |
| stable_audio_open | 107.42 Hz | M=8 | 0.032 | **0.089** | 0.027 |

Coverage is **confirmed for Suno** (its q=0.999 tail recall goes 0.2623 to **0.4677**
at M=4) and **for MusicGen** (x3.2 at M=8), and **refuted for stable_audio_open**, per
R27.10. Two of three.

**Why depth nonetheless destroys everything else — the decoy null saturates.**
Measured on FakeMusicCaps:

| M | prior lags | decoy union | **prior lags ALSO in the null** | decoy sets touching the prior |
|---|---|---|---|---|
| 2 | 10 | 118 | 4 | 4 / 24 |
| 4 | 18 | 257 | 10 | 12 / 24 |
| 8 | 36 | 540 | **21** | **19 / 24** |

`margin = real - max(null)`. A shared lag forces the margin to <= 0 and to **exactly
0** when that lag is the null's maximum too. **An atom of probability at zero caps
recall at every quantile and makes any threshold degenerate.** That is the complete
explanation for three separate observations that looked unrelated:

* `comb_priormax8_margin` pooled recall 0.283 against M=2's 0.625;
* the SONICS M=8 operating points reading identically at q=0.90 and q=0.95 (both
  thresholds are 0.0);
* `fpr_fma_pm8_mondrian/conformal_by_group.csv` returning **every threshold 0.0 with
  all three arms identical** — a conformal threshold cannot separate a constant.

**The deterministic lattice has ZERO overlap with the prior at every depth on both
corpora**, by construction. That property, not its determinism, is what makes a deep
prior usable — and a deep prior is what Suno (M=4) and MusicGen (M=8) need.
`derive_analytic_null.py` now emits a `*_zeroatom.csv` reporting
`frac_exactly_zero` per margin column, so the two nulls can be compared on it directly.

### R27.17 ⚠ CORRECTED — the lattice exclusion was incomplete for M > 2

`decoy_lattice_lags` excluded only `{Delta/2, Delta, 2*Delta}`. A candidate `g`
collides with the prior whenever `m*g ~ m'*Delta` for `m, m' <= M`, i.e.
`g ~ (m'/m)*Delta`, so that set is the `m, m' <= 2` case alone. At M = 4 it leaves
eight ratios unexcluded (1/4, 1/3, 2/3, 3/4, 4/3, 3/2, 3, 4) and at M = 8 forty.

| | overlap with the prior's lags, FMC |
|---|---|
| lattice as first shipped | M=2 **0**, M=4 **13 / 18**, M=8 **33 / 36** |
| random decoys | M=2 4 / 10, M=4 10 / 18, M=8 21 / 36 |
| **lattice, corrected** | **0 at every M, both corpora** |

So the "deterministic" null was *worse than the decoys it replaces* at M = 4 and M = 8.
**Consequence: `comb_sonics_DETNULL`'s M=4 and M=8 lattice columns are contaminated and
that pass must be repeated. M=2 is valid.** Found by measuring the overlap rather than
trusting the argument in the docstring — the same discipline that caught §O8b.

**And a real ceiling follows.** The ratios tile the candidate range as M grows (3
distinct ratios at M=2, 11 at M=4, 43 at M=8), so at the inherited `reject_within=0.05`
the admissible set is **155 lags at M=2, 74 at M=4 and EMPTY at M >= 6** on
FakeMusicCaps. **M = 4 is the deepest depth this construction supports** — exactly what
Suno needs, and not enough for MusicGen. Lowering the tolerance to 2% would admit M = 8
(197 lags), but 5% is inherited from the shipped `decoy_fundamentals` and lowering it
to reach a result would be a new free parameter of precisely the kind §2.5 rejected.
The limit is reported, not tuned away. Pinned by
`tests/test_lattice_null.py::TestTheDepthTheConstructionCanSupport`.

### R27.18 ⚡ The matched filter, measured for the first time — and its own criterion fails

`comb_sonics_DETNULL`, signed macro AUC:

| feature | chirp-v2 | chirp-v3 | chirp-v3.5 | udio-120s | udio-30s | MACRO | inverted |
|---|---|---|---|---|---|---|---|
| **`comb_mfoff_z`** | 0.9999 | 0.9999 | 0.9998 | 0.6164 | 0.5466 | **0.83252** | **0 / 5** |
| `comb_mf_z` | 0.9999 | 0.9999 | 0.9998 | 0.5523 | 0.5117 | 0.81272 | 0 / 5 |
| `comb_priormax2_margin` (shipped) | 0.9886 | 0.9876 | 0.9853 | 0.5870 | 0.5871 | 0.8271 | 0 / 5 |

This closes §R11's "implemented, unit-tested, never run on corpora" — and it is the
best signed macro of any single feature in the project on SONICS. **Two reasons no
number from it may be quoted yet.**

1. **`comb_mf_z` does NOT beat `comb_mfoff_z`.** The offset-scanning variant is the
   CONTROL, and `comb_matched_filter`'s docstring pre-registers the criterion
   verbatim: *"if the DC-aligned score does not beat it in AUC then alignment carries
   no information and this whole argument is void."* It does not, on every family.
   **The alignment argument is void** — whatever the matched filter reads, it is not
   DC alignment, and the mechanism claimed for it in §R11 is withdrawn.
2. **0.9999 on a channel-labelled corpus is the signature of the confound that read
   1.0000 before the chain control** (§B4, `frac_power_8k_11k`). The run applied
   `--resample-hz 15000`, so the control is in place, but C1 and C2 are mandatory
   before this is called anything.

Its 24 decoys (`comb_mf_decoy{NN}_z`) were also computed and its margin has never been
formed; `derive_prior_margin.py`'s regex requires a `_strength` suffix and will not see
them.

### R27.19 Status of the three corpora, and what must be re-run

| corpus | dir | code version | valid for |
|---|---|---|---|
| FakeMusicCaps | `comb_fmc_HARM2` | pre-matched-filter, 144 decoy cols, `--operator-grid` | the M sweep, the margin families |
| SONICS | `comb_sonics_HARM2` | pre-matched-filter, 144 decoy cols | the M sweep, the margin families |
| FMA (all-real) | `comb_fma_HARM2` | post-matched-filter, 168 decoy cols | transfer thresholds only |
| SONICS | `comb_sonics_DETNULL` | lattice, **pre-R27.17 fix** | **M=2 lattice columns only** |
| FakeMusicCaps | — | **never run with the lattice** | — |

Four versions across five artifacts. The next pass should put all three corpora on one
code version at `--n-harm 2 4` (M=8 has no admissible lattice null, R27.17), which is
what `RUNBOOK_2026-09-17_NULLS.md` Step B does.

### R27.20 ✅ The zero-atom mechanism, proven end to end

`analytic_null_zeroatom.csv`, fraction of tracks whose margin is **exactly** 0:

| score | FakeMusicCaps | SONICS |
|---|---|---|
| `comb_priormax2_margin` (decoy) | **0.1952** | 0.0322 |
| `comb_priormax4_margin` (decoy) | **0.2591** | 0.0436 |
| `comb_priormax2_latmargin` (lattice) | **0.0000** | **0.0000** |
| `comb_priormax4_latmargin` (lattice) | **0.0000** | **0.0000** |

And the controlled before/after, same corpus and depth, differing only by the §R27.17
exclusion fix: SONICS `comb_priormax4_latmargin` **0.4217 → 0.0000** as the prior
overlap went 13/18 → 0. §R27.16's mechanism is therefore measured, not argued.

### R27.21 ✅ Evenness on FakeMusicCaps — the criterion the brief set

| FMC, 0/5 inverted both | macro | min family AUC | recall @ q95 | worst-family recall |
|---|---|---|---|---|
| `comb_priormax2_margin` (shipped) | 0.9153 | 0.7909 | 0.6277 | **0.0331** (SAO) |
| **`comb_priormax4_latmargin`** | **0.9438** | **0.8668** | **0.8523** | **0.6359** (SAO) |

**Worst-family recall improves 19x on the exact weakness §R25 reports**, with a higher
macro and a higher minimum per-family AUC. Not yet gated; C10 is SKIP.

### R27.22 ⚠ RETRACTED — "the DIFFERENCE form survives where the RATIO form dies"

§R27.4 predicted from the fixture that a difference-form bulk calibration would keep
the sign on the Udio families where the ratio form loses it. **On the corpus both
invert, and the difference inverts harder.** SONICS, udio-120s / udio-30s:

| form | udio | macro | inverted |
|---|---|---|---|
| `comb_harm4_sharpness` (ratio) | 0.2328 / 0.2329 | 0.6636 | 2 |
| `comb_harm4_nmargin` (difference) | 0.2015 / 0.1892 | 0.6379 | 2 |
| `comb_priormax4_afmargin` (difference, exact `comb_acf_floor`) | 0.2625 / 0.2483 | 0.6718 | 2 |
| `comb_harm4_floor` (the null alone) | 0.2358 / 0.1995 | 0.6209 | 2 |
| `comb_priormax{2,4}_latconf` (p-value form) | 0.4718 / 0.4663 | 0.7488 | 2 |

**The replacement statement, which is stronger and general: the functional form is
irrelevant; only a null that varies the HYPOTHESIS fixes the sign.** Every score with
0 inverted families on SONICS is decoy- or lattice-calibrated; every statistic
estimated from the track's own autocorrelation bulk inverts in every form tried.
Handover candidates N1, N2 and N6 close together. The fixture separated forms because
its Udio-like condition (a smoother residual with no comb) is not what Udio is, and
that limitation was recorded in §R27.6 before this run.

Note the p-value form of a *hypothesis-varying* null also inverts (`latconf` 2/5)
where its margin form does not (`latmargin` 0/5): a within-track rank discards the
scale the margin needs.

⚠ `comb_priormax2_afmargin` reaches **0.9539 macro on FakeMusicCaps with 0/5
inverted** -- the best FMC margin measured in this project -- and 0.6754 with 2
inverted on SONICS. A single-corpus result, and precisely what the two-corpus rule
exists to catch. It must not be quoted alone.

### R27.23 The remaining tension, its diagnosis, and the null it implies

| score (all 0/10 inverted) | FMC macro | SONICS macro | FMC worst-family recall | SONICS best precision @ p=1% |
|---|---|---|---|---|
| `comb_priormax2_margin` (shipped) | 0.9153 | 0.8271 | 0.0331 | 0.8329 |
| `comb_priormax4_margin` (decoy) | 0.9151 | **0.8312** | 0.0920 | **0.9091** |
| `comb_priormax4_latmargin` (lattice) | **0.9438** | 0.8063 | **0.6359** | 0.7015 |

The lattice wins evenness on FakeMusicCaps; the decoys win precision on SONICS.
**Diagnosed as null SIZE:** the lattice at M=4 on SONICS holds **52** lags against the
decoy union's **209**, and a null a quarter the size cancels a quarter as much of the
residual smoothness that inverts Udio -- measured, udio margin 0.530 / 0.527 (lattice)
against 0.597 / 0.584 (decoy). Coverage and disjointness are independent properties
and no null yet had both.

**`comb_prior_union_null` added**: every lag either null proposes, minus anything
within one bin of a lag the prior searches. 242 lags at M=4 on FakeMusicCaps, 179 on
SONICS, zero overlap. Still seeded, since it contains the drawn decoys, so it is
reported beside the lattice rather than instead of it.

**PRE-REGISTERED, before the run:** `comb_priormax4_unionmargin` keeps
`frac_exactly_zero = 0.0000` and recovers the decoy Udio margin (~0.59, not the
lattice's 0.53), giving SONICS macro ~0.83 at 0/5 inverted while holding FMC
worst-family recall near 0.6. **If Udio stays at 0.53, null size is not the
explanation and this goes to `negative-results` as such.**

### R27.24 The matched filter: gated, significant, and its rationale withdrawn

`gate_mfoff_sonics/`, `comb_mfoff_z`, n = 59,280:

| gate | verdict | value |
|---|---|---|
| C1 channel-alone | FAIL | `frac_power_top_quarter` **0.6082** -- identical to the incumbent (§R18) |
| C2 level | FAIL | `peak_dbfs` **0.6202** -- identical to the incumbent |
| C3 / C4 / C5 / C6 / C7 / C9 | PASS | duration 0.5516; silence 0.5485; shuffle [0.5001, 0.5068]; 59,280 finite; smallest stratum 1,568; within-real \|rho\| 0.030 |
| C8 spacing | PASS | **but it reads `comb_spacing_hz`, not this score** -- says nothing about `comb_mfoff_z` |
| C10 | **SKIP** | no `--recon-csv`; deconvolution specificity UNTESTED |
| C11 | PASS | pooled **0.8049 [0.8013, 0.8084]**; DeLong vs `comb_priormax2_margin` z = 6.44, **p = 1.2e-10**, delta +0.0148 |

Per-generator with CIs: chirp 0.9999 [0.9997, 1.0] x3, udio-120s 0.6164 [0.610, 0.623],
udio-30s 0.5466 [0.537, 0.555] -- **0/5 inverted, confirmed by interval**. So it adds
no confound the paper does not already disclose and beats the shipped score
significantly on pooled AUC.

**Two reasons no number from it may ship.**

1. **The alignment rationale is void, by the criterion the code itself
   pre-registers.** `comb_matched_filter`'s docstring: *"if the DC-aligned score does
   not beat [the offset-scanning control] in AUC then alignment carries no information
   and this whole argument is void."* Measured: `comb_mf_z` **0.8127** against
   `comb_mfoff_z` **0.8325**, on every family. §R11's mechanism claim is withdrawn.
2. **It has no per-track null.** `comb_mf_decoy{NN}_z` were computed for the
   DC-aligned filter, not the offset-scanning one, so the statistic that scores 0.8325
   has no margin and no provenance control.

### R27.25 ✅ MEASURED — the corrected grid, and what it does to the paper's §6 sentence

`pgrid_{linear,tail}_{fmc,sonics}/precision_targets.csv`, `comb_priormax2_margin`:

| corpus | best precision at p = 1%, published grid | corrected grid | prec 0.95 at p = 5% |
|---|---|---|---|
| FakeMusicCaps | 0.6554 | **0.7303** | unreachable either way |
| SONICS | 0.6836 | **0.8328** | **unreachable -> REACHABLE** (recall 0.2397) |

**The paper's sentence survives for the score it reports** -- precision 0.9 at 1%
prevalence is still out of reach on both corpora for `comb_priormax2_margin`. **It is
false as the unqualified claim it makes**, because `comb_priormax4_margin` on SONICS
reaches **0.9091 at 1% prevalence** with recall 0.4677 at q = 0.999 (§R27.10's table).
The minimal camera-ready fix is to scope *"no threshold on either corpus"* to the
reported detector. Both values are recorded here; `icassp/paper.tex` is **not** edited
until §R27.23 settles which score is reported.

### R27.26 ⚠ C10 is blocked on a stale extraction, and is SKIP on every new score

`comb_recon_control/comb_per_track_lvl_sm5_median_db_f1000_noclip.csv` (2026-08-22)
carries **no `_harm248` suffix**, so it was extracted without `--n-harm` and has no
prior, decoy or lattice columns at all. `derive_prior_margin.py` refused it correctly.
**The content-identical control must be re-extracted with the current feature set
before C10 can be run on any score proposed since §R14.** Until then C10 is SKIP for
every one of them, and a SKIP is not a PASS.

### R27.27 ✅ The union null — pre-registered prediction confirmed on three of four checks

§R27.23 recorded, before the run: union keeps `frac_exactly_zero = 0`, recovers the
decoy Udio margin (~0.59 not 0.53), SONICS macro ~0.83 at 0/5 inverted.

| check | predicted | measured |
|---|---|---|
| zero atom | 0.0000 | **0.0000**, both corpora |
| SONICS udio-120s / udio-30s | ~0.59 | **0.5823 / 0.5709** |
| SONICS macro, inverted | ~0.83, 0/5 | **0.8257, 0/5** |
| FMC worst-family recall | ~0.6 | **not yet measured** |

FakeMusicCaps macro 0.9335, 0/5 inverted. **The null-size diagnosis of §R27.23 is
therefore confirmed.** But the union contains the drawn decoy lags, so it is seeded and
**not decoy-free**; it is a control that establishes the mechanism, not a candidate.

### R27.28 ⭐ The SONICS deficit of the decoy-free null is entirely Udio

| SONICS, M = 4 | chirp x3 mean | udio x2 mean | MACRO |
|---|---|---|---|
| `comb_priormax4_margin` (decoy) | **0.9918** | 0.5905 | 0.8312 |
| `comb_priormax4_latmargin` (lattice) | **0.9913** | 0.5287 | 0.8063 |
| `comb_priormax4_unionmargin` (union) | **0.9917** | 0.5766 | 0.8257 |

**On every family with an in-band comb the three nulls agree to three decimal places.**
The whole macro gap sits on the two Udio families, where §R7 records three independent
statistics agreeing there is no in-band comb at all -- concentration 0.290/0.217 against
real music's 0.189, `comb_strength` 0.055 below real's 0.072, and dispersion showing
Udio *less* stationary than real music. An AUC above 0.5 on Udio is therefore
residual-smoothness cancellation, not comb detection, and **the decoy-free null is
weaker only where there is nothing to detect.**

This is a MECHANISM claim and it is not yet supported: **C10 is SKIP for every score
proposed since §R14**, and C10 is the only control that would distinguish "cancels a
nuisance" from "detects a generator". Until §R27.30 runs, the reframing above is an
interpretation, not a result.

### R27.29 The decoy-free detector, and what it costs — both directions stated

`comb_priormax4_latmargin`: prior = 6 fundamentals x 4 harmonics (18 lags, reach
400 Hz); null = every lag reachable by a fundamental in [20, 150] Hz that is not within
5% of any `(m'/m)*Delta`, enumerated exhaustively. **No seed, no draws, no grid
spacing.** Disjoint from the prior by construction.

**Gains:** deterministic; `frac_exactly_zero` **0.0000** against the decoy margin's
0.1952 (M=2) / 0.2591 (M=4) on FMC; FMC macro **0.9438** against 0.9153; FMC minimum
family AUC **0.8668** against 0.7909; **FMC worst-family recall 0.6359 against 0.0331**,
a 19x improvement on the weakness §R25 reports; threshold transfer to unseen FMA
**0.94x with TPR 0.8523** against 0.75x with TPR 0.6250. 0/10 families inverted.

**Costs, not to be softened:** SONICS macro **0.8063 against 0.8271** (all of it Udio,
§R27.28); **precision is NOT improved** -- 0.7015 at 1% prevalence against the shipped
0.8329 and `comb_priormax4_margin`'s 0.9091, and at a matched ~1% in-corpus FPR its TPR
is 0.4870 against 0.5385. It trades far-left-tail precision for even recall across
families. Depth is capped at M = 4 (§R27.17).

**Not yet gated, and C10 is SKIP.** No number above may enter a report until §R27.30.

### R27.30 ⚠ C10 remains BLOCKED, and the manifest paths are now known

`comb_recon_control/comb_per_track_lvl_sm5_median_db_f1000_noclip.csv` (2026-08-22)
carries no `_harm248` suffix and therefore no prior, decoy or lattice columns;
`derive_prior_margin.py` refused it correctly. A runbook command then guessed a
manifest path that does not exist. The real ones, from `find`:

* FakeMusicCaps: `data/processed/recon_control_v3/audio/recon_audio_manifest.csv`
* SONICS: `data/processed/recon_control_sonics_audio/wav/recon_audio_manifest.csv`

The control must be re-extracted at `--n-harm 2 4 --null-priors 24` before C10 can run
on any score proposed since §R14. **This is the highest-priority outstanding control**,
because §R27.28's interpretation of the Udio gap depends on it.

### R27.31 ⚠ The per-genre conformal experiment is UNDERPOWERED, not negative

`mondrian_comb_priormax4_latmargin`: transfer spread 9.57x, conformal_group 9.03x,
Electronic FPR 0.1047 against a 0.05 target. That reads as a failure of
group-conditional calibration. It is not interpretable. Simulated with a **perfectly**
calibrated conformal procedure on **perfectly** exchangeable data, n = 191 calibration
reals per group, eight groups, 3,000 trials:

| statistic under perfect calibration | median | 90th pct | max seen |
|---|---|---|---|
| max/min FPR spread across the eight groups | **4.00x** | **8.50x** | 26x |
| worst group's realised FPR | 0.0785 | **0.1047** | 0.1518 |

The measured 9.03x sits near the 90th percentile of that null and Electronic's 0.1047
*is* its 90th percentile. A single calibration split is a point estimate, exactly as
§L1 established for the dictionary draws.

**`--n-splits` added** to `measure_false_positive_rate.py`: it averages over independent
calibration splits and emits `fpr_conformal_group_sd`, the across-split spread per
group. A single split now logs a WARNING naming the noise floor. Use `--n-splits 50`.
If the across-split SD is comparable to the across-group spread, FMA's ~380 tracks per
genre cannot support per-genre calibration, and that is itself the finding.

Transfer for the decoy-free score, measured (FMC -> unseen FMA): q = 0.90 in-corpus
0.1001 -> unseen 0.0957 (TPR 0.9056); **q = 0.95 0.0456 -> 0.0427, 0.94x, TPR 0.8523**;
q = 0.99 0.0093 -> 0.0037 (TPR 0.4870). The shipped score transfers 0.75x with TPR
0.6250 (§R25).

### R27.32 ✅ C10 PASSES for the decoy-free score, and beats the incumbent

`c10_comb_priormax4_latmargin/recon_pairs.csv`, 300 content-identical pairs per variant,
re-extracted at `--n-harm 2 4 --null-priors 24` from
`data/processed/recon_control_v3/audio/recon_audio_manifest.csv` (1,800 tracks):

| variant | `priormax4_latmargin` | `priormax4_unionmargin` | `priormax2_margin` (shipped) |
|---|---|---|---|
| encodec_3kbps | **0.9667** (+0.0712) | **0.9800** (+0.0665) | 0.9300 (+0.0405) |
| encodec_6kbps | **0.9533** (+0.0693) | **0.9700** (+0.0669) | 0.9133 (+0.0386) |
| encodec_24kbps | **0.9367** (+0.0661) | **0.9567** (+0.0642) | 0.8833 (+0.0420) |
| griffinlim_128mel | 0.5267 (p = 0.20) | 0.5433 (p = 0.22) | 0.4500 (p = 0.51) |
| griffinlim_256mel | 0.4900 (p = 0.85) | 0.5200 (p = 0.62) | 0.4800 (p = 0.64) |

frac_above with median delta in brackets; Wilcoxon p < 1.2e-47 for every EnCodec cell.
**The decoy-free score responds to deconvolution ~1.7x more strongly than the incumbent**
(median delta 0.066-0.071 against 0.039-0.042) with Griffin-Lim cleanly at chance and
both p-values non-significant -- cleaner than §R18, where `griffinlim_256mel` reached a
nominally significant p = 0.011 on SONICS.

**This is the control §R27.28 was waiting on.** The claim that the SONICS gap is Udio
"where there is nothing to detect" now rests on a measurement rather than an argument.

### R27.33 ✅ FakeMusicCaps gate battery — same exposure, significantly better

`gate_lat4_fmc/`, `comb_priormax4_latmargin`, n = 32,951:

| gate | verdict | value |
|---|---|---|
| C1 channel-alone | FAIL | `spectral_flatness_hf` **0.6284** -- identical to the incumbent (§R16) |
| C2 level | FAIL | `peak_dbfs` **0.6207** -- identical to the incumbent |
| C3 duration | PASS | 0.5228, gap 0.007 |
| C4 silence | PASS | 0.5104 |
| C5 shuffle | PASS | [0.5001, 0.5103] |
| C6 / C7 | PASS | 32,951 finite / smallest stratum 5,355 |
| C8 spacing | PASS | fake 0.431 vs real 0.030 (reads `comb_spacing_hz`, so not about this score) |
| C9 harmonicity | PASS | within-real 0.063 |
| C10 | PASS | separately, §R27.32 |
| C11 | PASS | **AUC 0.9438 [0.9408, 0.9469]**; DeLong vs `comb_priormax2_margin` **delta +0.0286, z = 20.82, p = 2.8e-96** |

Per-family CIs: MusicGen [0.8592, 0.8741], audioldm2 [0.9823, 0.9872], musicldm
[0.9704, 0.9767], mustango [0.9943, 0.9971], stable_audio_open [0.8917, 0.9043] --
**0/5 inverted, confirmed by interval.** C1/C2 reproducing the incumbent exactly is the
expected result and means the arm adds no confound the paper does not already disclose.

⚠ **The SONICS gate has NOT been run.** Nothing ships on a one-corpus battery.

### R27.34 ⭐ Group-conditional conformal WORKS — and only on a score with no tie mass

`mondrian50_*`, 50 independent calibration splits, alpha = 0.05, FMA reals by genre:

| score | transfer spread | conformal_global | **conformal_group** | tie mass at 0 |
|---|---|---|---|---|
| **`comb_priormax4_latmargin`** | 3.56x | 3.63x | **1.23x** | **0.0%** |
| `comb_priormax2_margin` (shipped) | 2.51x | 2.57x | 1.97x | 19.5% |

For the lattice score all eight genres land between **0.0384 and 0.0472** against the
0.05 target, across-split SD 0.017-0.024, worst-genre precision at 1% prevalence
**0.109 -> 0.168 (+54%)**, recall 0.858 global against 0.773-0.886 per genre.

**The shipped score cannot be equalised as well, and the reason is mechanical:** a
quantile threshold cannot be placed inside a tie mass, and 19.5% of its margins are tied
at exactly zero (§R27.20). So the chain is: **disjoint null -> no tie mass -> conformal
calibration works -> per-genre errors become fair.** This is the strongest argument for
the lattice construction that does not mention AUC at all, and it retires the
single-split reading of §R27.31.

⚠ The union arm's Mondrian could not run: `comb_fma_DETNULL2` predates the union columns.
The FMA pass must be repeated at `_DETNULL3`.

### R27.35 The verdict on `comb_priormax4_latmargin`, both directions

**Better on 8 criteria** (FMC macro +0.0286 at p = 2.8e-96; FMC worst-family recall
0.0317 -> 0.6519, 20x; transfer 0.75x -> 0.94x with TPR 0.626 -> 0.858; per-genre spread
1.97x -> 1.23x; worst-genre precision +39%; C10 delta 1.7x larger; tie mass 19.5% -> 0%;
deterministic). **Worse on 3** (SONICS macro 0.8271 -> 0.8063; SONICS worst-family
0.0401 -> 0.0053; global precision at 1% 0.738 -> 0.453). **Equal on 3** (0/10 inverted;
Griffin-Lim at chance; gate profile).

**The three losses are one loss.** Splitting SONICS by whether an in-band comb exists:
Suno x3 reads **0.9918 -> 0.9913**, identical to three decimals; Udio x2 reads 0.5905 ->
0.5287, and §R7 records three independent statistics agreeing Udio leaves no comb inside
the band these benchmarks preserve. §R27.32's C10 result supports reading the Udio
column as nuisance cancellation rather than detection.

**Not claimable:** a SONICS improvement; better precision at a single global threshold;
a Udio detector; any depth beyond M = 4 (§R27.17).

**Outstanding before anything enters the paper:** the SONICS gate battery (§R27.33), the
FMA `_DETNULL3` pass (§R27.34), and robustness + efficiency, which have never been run on
any score newer than `comb_priormax2_margin` although the paper quotes both.

A plain-language explainer of the whole methodology, English and Greek, written for a
reader with no background, is at `docs/methodology.html`.

### R27.36 ⚠ CORRECTED — the SONICS gate makes `comb_priormax4_latmargin` a two-sided result

`gate_lat4_sonics/`, n = 59,280. C1 FAIL 0.6082 / C2 FAIL 0.6202 (identical to the
incumbent, §R18), C3-C9 PASS, C10 SKIP, and:

**C11 pooled AUC 0.7646 [0.7607, 0.7686]; DeLong vs `comb_priormax2_margin`
delta = -0.0255, z = -16.29, p = 1.24e-59.**

§R27.35 reported this as "SONICS macro 0.8271 -> 0.8063", which understated it. With the
pooled AUC and a formal test the score is **significantly WORSE on SONICS**, exactly as
it is significantly better on FakeMusicCaps (+0.0286, p = 2.8e-96). Per-family CIs
confirm 0/5 inverted (chirp 0.991-0.992, udio-120s [0.5232, 0.5372], udio-30s
[0.5183, 0.5366]), but Udio is barely above chance. **The arm is a two-sided result and
must not be written as a clean improvement.**

### R27.37 With the FMA pass complete, the UNION is the only score >= shipped on both

| score | FMC | SONICS | transfer @q95 | TPR | Mondrian spread | zero atom | seed-free |
|---|---|---|---|---|---|---|---|
| `comb_priormax2_margin` | 0.9153 | **0.8271** | 0.73x | 0.626 | 1.93x | 19.5% | no |
| `comb_priormax4_latmargin` | **0.9438** | 0.8063 | 0.89x | **0.858** | **1.21x** | **0.0%** | **yes** |
| `comb_priormax4_unionmargin` | 0.9335 | **0.8257** | 0.78x | 0.836 | **1.22x** | **0.0%** | no |

Union against shipped: FMC **+0.0182**, SONICS **-0.0014** (a tie), TPR **+0.210**,
per-genre spread 1.93x -> 1.22x, zero atom 19.5% -> 0%. It is the only construction that
does not give up a corpus. **But it contains the drawn decoy lags, so it is seeded and
not decoy-free** -- it does not answer the question the professor asked.

### R27.38 ⭐ The diagnosis: the lattice protects the PRIOR but not the COMB's own series

`decoy_lattice_lags` excludes candidates near `(m'/m)*Delta` for `m, m' <= M`. That makes
the null disjoint from the prior's lag set -- §R27.20's zero atom -- but the ratio set
stops at M, so a candidate near `(5/4)*Delta` is admissible at M = 4 and its 4th multiple
lands on `5*Delta`, which is **signal, not noise**.

Measured directly: **11 true-comb harmonics sit inside the M=2 lattice null and 3 inside
the M=4 one** on FakeMusicCaps, and one of them is **stable_audio_open's own visible tooth
at 21.53 x 5 = 107.65 Hz = lag 110**. On a synthetic comb with teeth at EXACT multiples of
50 Hz -- MusicGen's spacing, whose visible tooth is the 5th harmonic:

| null, M = 2, FakeMusicCaps geometry | margin on a clean 50 Hz comb |
|---|---|
| lattice | **-0.1284 -- INVERTED** |
| union | -0.1284 |
| **harmonic-protected** | **+0.5113** |

That is the measured explanation for `comb_priormax2_latmargin` reading MusicGen
**0.6466** against the shipped score's 0.8191. It is the same failure as the lag-axis
complement (§R27.1), one level out: the detector subtracts part of its own evidence.

The ratio exclusion is simultaneously **too broad**: rejecting +-5% of eleven ratios
leaves only **74** admissible lags at M=4 on FakeMusicCaps and **52** on SONICS, against
the decoy union's 275 and 220. Null SIZE is what cancels the residual smoothness that
inverts Udio (§R27.23), so the over-exclusion is the measured cause of the SONICS deficit
in §R27.36.

### R27.39 `comb_prior_harmonic_null` — built, tested, NOT yet run on a corpus

Take every lag in `[40 Hz, M x 150 Hz]` and remove only `+-1 bin` around `k*Delta` for
every real fundamental and **every** integer k, not just `k <= M`. Exact and minimal: it
is precisely the set of lags where a real decoder could have put energy. This is the
professor's instruction taken literally -- *the whole harmonic series*, not its first two
members -- confined to the plausible-fundamental lag range so §R27.1's failure cannot
recur beyond it.

| null | overlaps the prior | contains comb harmonics | lags, M=4 FMC / SONICS | seed-free |
|---|---|---|---|---|
| 24 random decoys | yes, 4/10 | yes | 275 / 220 | no |
| lattice | **no** | yes | 74 / 52 | **yes** |
| union | **no** | yes | 242 / 179 | no |
| **harmonic-protected** | **no** | **no** | **441 / 252** (CORRECTED 2026-09-23 from 253: `n_lags` at bin 24000/16384 is 252, as R27.43 records) | **yes** |

**The first construction with both properties.** `tol_bins = 1` is forced by rounding
rather than fitted -- the analysis lattice cannot place `k*Delta` on an exact integer bin
-- and the residual prior-frequency error is negligible over this range (Suno's 49.99 Hz
against a 50.00 Hz prior drifts 0.08 bins by its 8th harmonic).

**PRE-REGISTERED before the run:** `comb_priormax4_hmargin` keeps
`frac_exactly_zero = 0.0000`, holds the FakeMusicCaps gains (macro ~0.94, worst-family
recall ~0.65), and **closes the SONICS gap** -- Udio ~0.58-0.59 rather than the lattice's
0.529, SONICS macro ~0.825 at 0/5 inverted. **If Udio stays near 0.53 the null-size
argument of §R27.23 and §R27.38 is wrong** and goes to `negative-results` as such.

### R27.40 ⚠ Robustness and efficiency CANNOT score any detector newer than 2026-09-07

`run_robustness_battery.py --training-free {comb_strength,comb_stat_strength}` and
`measure_efficiency.py --training-free {comb_strength,nmf_peak}`. Neither choice list
includes a margin score, so neither can measure `comb_priormax2_margin` -- the score the
paper currently proposes -- let alone anything from §R27. §R22 recorded that both "now
take `--null-priors`", which is true and insufficient: the flag supplies decoys to the
feature extractor but there is no way to select the calibrated score as the detector.

**The paper quotes both numbers** (robustness +0.3/+0.4 EER under AAC/Opus, +14.7 under
time-stretch; 380 audio-s/s, 0 parameters) and they are currently the COMB arm's, not the
proposed arm's. Extending the two scripts is a code change and is on the critical path
for any camera-ready claim about cost or robustness.

### R27.41 Plain-language documentation

`docs/methodology.html` -- the whole method from zero for a reader with no background, in
English and Greek, covering the comb mechanism, the prior, why a null is needed, the
defect in the 24 decoys, the three replacement nulls and the measured trade-offs
including the negative SONICS result. Published at
https://claude.ai/code/artifact/9aa7977d-5493-465c-94af-b7f54de98652

### R27.42 ⭐⭐ `comb_priormax4_hmargin` — all five pre-registered predictions CONFIRMED

`comb_{fmc,sonics}_HNULL/analytic_null_auc.csv`. §R27.39 recorded the predictions before
the run:

| predicted | measured | |
|---|---|---|
| `frac_exactly_zero` stays 0.0000 | **0.0000** (FMC) | ✅ |
| holds the FakeMusicCaps gains, macro ~0.94 | **0.9445** | ✅ |
| closes the SONICS gap, Udio ~0.58-0.59 (lattice 0.5302 / 0.5272) | **0.5862 / 0.5792** | ✅ |
| SONICS macro ~0.825 at 0/5 inverted | **0.8286, 0/5** | ✅ |
| M=2 recovers, the lattice having eaten MusicGen's 5th harmonic | MusicGen **0.6466 -> 0.8505** (+0.204) | ✅ |

**The standings, all 0/10 inverted:**

| score | FMC | SONICS | zero atom | seed-free | beats shipped on |
|---|---|---|---|---|---|
| `comb_priormax2_margin` (shipped) | 0.9153 | 0.8271 | 19.5% | no | — |
| `comb_priormax4_latmargin` | 0.9438 | 0.8063 | 0.0% | yes | FMC only |
| `comb_priormax4_unionmargin` | 0.9335 | 0.8257 | 0.0% | no | FMC only |
| **`comb_priormax4_hmargin`** | **0.9445** | **0.8286** | **0.0%** | **yes** | **BOTH** |

FakeMusicCaps **+0.0292**, SONICS **+0.0015**. **The only construction that beats the
shipped detector on both corpora, and the only one that does so with no seed.** It also
dominates the union on both corpora without randomness, so **the union is superseded**.

⚠ **NOT YET GATED.** §R27.36 records that `latmargin`'s SONICS C11 was a significant
LOSS (pooled 0.7646 against 0.7901, DeLong z = -16.29, p = 1.2e-59) even though its macro
gap looked small. **`hmargin`'s SONICS DeLong is the number that decides whether the
"beats on both corpora" claim survives**; a +0.0015 macro difference is well inside the
range that a pooled test has overturned before. No number from §R27.42 ships until
`gate_h4_sonics` returns.

### R27.43 ⛔ UDIO — the last lever, considered and rejected for no compute

`park2026probing` (arXiv:2606.08663, *Probing Token Spaces under Generator Shift in
AI-Generated Music Detection*, Park, Kim, Koh, Saito, ICML 2026 ML4Audio workshop) was
**verified by direct fetch** this session. It is in `refs.bib`, uncited, and memory
flagged it as carrying "the Udio/X-Codec lever". Its abstract states *"X-Codec tokens are
strongest when training on Udio alone"* -- which is a claim about a **tokenizer used as a
detection feature**, not about Udio's decoder. **It supplies no strides, hops, frame
rates or token rates for either.**

X-Codec-2.0 runs a 50 Hz latent rate at 16 kHz, hop 320. The same stride product at
44.1/48 kHz gives 137.8/150 Hz, neither in `DECODER_FUNDAMENTALS_HZ`. That motivates a
**cross-product prior**: published stride products {256, 320, 512, 640, 1024, 2048} x
plausible output rates {16, 22.05, 24, 32, 44.1, 48} kHz = 21 distinct fundamentals in
15-200 Hz, of which 15 are new.

**Rejected, for three reasons:**

1. **The apparent evidence is an arithmetic artefact.** Udio's measured median spacing
   600.59 Hz is "explained" to within 0.1% by 25, 37.5, 50, 75, 100 **and** 150 Hz
   simultaneously -- because every one of them divides 600. Testing "is T a multiple of
   f within 1%" accepts a window of `+-0.01*k`, which GROWS with harmonic order: at k=15
   it accepts +-0.15 of an integer (30% by chance), at k=30 +-0.30 (**60% by chance**),
   at k=41 +-0.41 (82%). The 643.07 Hz "matches" are at k = 15-41. This is exactly the
   sub-multiple hazard that retracted the SONICS half of the atom-attribution claim
   (§O8b): *the rule that makes the match count large is the rule that destroys its power
   to attribute.*
2. **Udio's architecture is unpublished.** The prior's legitimacy rests on enumerating
   PUBLISHED stride products; a tokenizer that happens to be a good feature for detecting
   Udio is not evidence about Udio's decoder.
3. **It degrades both mechanisms that make the detector work.** The prior doubles (10->20
   lags at M=2, 18->40 at M=4), raising real music's extreme-value baseline; and the
   harmonic-protected null halves (FakeMusicCaps 441->281, SONICS 252->126), when null
   SIZE is precisely what cancels the residual smoothness that inverts Udio (§R27.23,
   §R27.38). Two adverse effects against one imaginary benefit.

⚠ **Correction to a statement made in this session:** I first reported that the
cross-product null would be "almost empty" and that the idea "dies here". That was an
overstatement -- 126 to 281 lags survive, which is a usable null. The objection is that
the extension is **unmotivated and costly**, not that it is infeasible.

**Conclusion: nothing inside the band these benchmarks preserve recovers Udio.** §R7's
three independent statistics stand, §Q4's above-Nyquist prediction stands, and the only
test that could settle it is wideband Udio audio, already in the paper's future work. The
honest framing remains §R19's: the arm turns an inversion into a miss, a lesser and
different thing.

### R27.44 ⛔ Robustness and efficiency cannot score ANY calibrated score

Confirmed from the CLIs: `run_robustness_battery.py --training-free
{comb_strength,comb_stat_strength}` and `measure_efficiency.py --training-free
{comb_strength,nmf_peak}`. Neither choice list contains a margin score, so **neither can
measure `comb_priormax2_margin` -- the detector the submitted paper proposes** -- let
alone anything from §R27. §R22's note that both "now take `--null-priors`" is true and
insufficient: the flag feeds decoys to the feature extractor, but the calibrated score
cannot be selected as the detector.

The paper quotes robustness (+0.3/+0.4 EER under AAC and Opus-64k, +14.7 under
time-stretch) and cost (380 audio-s/s, 0 trainable parameters) in §6. **Those are the
COMB arm's numbers, not the proposed arm's.** Extending the two scripts is a code change
and is on the critical path for any camera-ready claim about cost or robustness.


---

## R27.45 — `comb_priormax4_hmargin`: the gate verdict on both corpora

Source: `reports/diagnostics/gate_h4_fmc/`, `reports/diagnostics/gate_h4_sonics/`,
pasted 2026-09-18. Descriptors: `channel_fmc_harm/` and
`channel_sonics_chain_120s/channel_per_track_canonical_profile_lvl.csv` (the 120 s
variant, matching the 120 s scores — H1's failure mode avoided).

| | FakeMusicCaps | SONICS |
|---|---|---|
| pooled AUC, signed | **0.9445** [0.9416, 0.9474] | **0.7925** [0.7887, 0.7963] |
| incumbent `priormax2_margin` | 0.9152 | 0.7901 |
| DeLong Δ | **+0.0292** | +0.0024 |
| DeLong z, p | **21.65, 5.6e-104** | **1.62, 0.106** |
| C11 verdict | **PASS** | FAIL *(rival criterion only)* |
| C1, C2 | FAIL, identical to incumbent | FAIL, identical to incumbent |
| C3–C9 | PASS | PASS |
| families inverted by interval | **0 / 5** | **0 / 5** |

**The SONICS C11 FAIL must be read precisely.** The gate requires DeLong significance
against the rival to award a PASS; p = 0.106 does not clear it. The score's own CI lower
bound is 0.7887, far above 0.5, so the score is valid — the gate is refusing the *claim of
improvement*, not the score. **The correct statement is parity on SONICS.** The same
vocabulary R16 used for the shipped score's FMC tie.

Contrast, so the distinction is on the record: `latmargin`'s SONICS DeLong was
z = **−16.29**, p = 1.2e-59 — significantly *worse* (R27.31, itself a correction of an
earlier understatement). `hmargin` is the first decoy-free null that is not worse on
SONICS.

**C1/C2 being identical to the incumbent's is the load-bearing negative.** Both scores
inherit the same corpus-level channel confound; the harmonic null adds no new exposure.

## R27.46 — C10 on SONICS: stronger on EnCodec, and a NEW caveat on Griffin-Lim

Source: `reports/diagnostics/c10_h4_sonics/recon_pairs.csv`.

| variant | frac_above | p | median Δ |
|---|---|---|---|
| EnCodec 6 kbps | 0.930 | ≈0 | **+0.133** |
| EnCodec 12 kbps | 0.973 | ≈0 | **+0.159** |
| Griffin-Lim 256 | **0.583** | **0.0014** | +0.0040 |
| Griffin-Lim 512 | **0.580** | **0.0018** | +0.0041 |

**The magnitude claim is stronger than the incumbent's.** Shipped `priormax2_margin` gave
EnCodec median Δ +0.056…+0.067 (R18); `hmargin` gives +0.133…+0.159, ~2.4×. The
EnCodec : Griffin-Lim median ratio is **33–40×**.

**⚠ But the sentence "Griffin-Lim sits at chance" can no longer be written for this score
on SONICS.** p = 0.0014 / 0.0018 survives a Bonferroni 0.01 across five variants. The
incumbent's worst was 0.537 at p = 0.0106 (R18) — nominally non-significant. This is a
real regression on one gate and is recorded as such, not buried under the EnCodec win.

Reading: a deeper prior (M=4) and a null that protects every harmonic make the score
sensitive to *any* resynthesis that reimposes phase structure, not only to transposed
convolution. The deconvolution-specific reading survives on **effect size**, not on a
clean null.

**FMC C10 for `hmargin` is NOT RUN.** The recon control at `_DETNULL3` predates
`comb_prior_harmonic_null`, so it carries no `hmax_strength` columns and
`analyze_recon_pairs.py` cannot find the score. Re-extraction is Step B of
`RUNBOOK_2026-09-17_NULLS.md`. **Until it runs, C10 for this detector is single-corpus.**

## R27.47 — deployment: the numbers that justify the switch

Sources: `analysis_h4_fmc/`, `analysis_h4_sonics/`, `fpr_fma_h4/`.

| | shipped `pm2_margin` | `pm4_hmargin` |
|---|---|---|
| FMC worst-family recall @ q95 | **0.0308** (Stable Audio Open) | **0.6924** (MusicGen) |
| FMC pooled recall @ q95 | 0.6246 | **0.8760** |
| FMA transfer ratio @ q95 | 0.75× | 0.72× |
| FMA TPR at the transferred threshold | 0.6250 | **0.8760** |
| per-genre FPR spread, 50-split conformal | 1.93× | **1.17×** |
| all 8 genres' conformal FPR | — | 0.0441–0.0514 vs a 0.05 target |
| margins exactly 0 | 19.5% | **0.0%** |

**Two things to note, both against my own earlier framing.**

1. **The worst family CHANGED.** Stable Audio Open was the paper's named deployment
   failure; at M=4 the prior reaches its 107.4 Hz tooth and it is no longer worst.
   MusicGen at 0.6924 is now the floor — a 22× lift on the binding number.
2. **The transfer ratio did not improve** (0.75× → 0.72×, still over-conservative). What
   improved is the *recall paid for it*: 0.625 → 0.876. Do not quote 0.72× as a win.

**Precision at 1% prevalence under a single global threshold is WORSE**: FMC
0.743 → 0.560, SONICS 0.833 → 0.782. The left tail is where the incumbent's tie mass
happened to help. The honest deployment claim is therefore **per-genre conformal**, where
`hmargin` wins on every number — not a global threshold, where it does not. Recorded so
the paper cannot quote the favourable half.

## R27.48 — `comb_calibrated_score`, and the robustness/efficiency scripts unblocked

R27.44 recorded that `run_robustness_battery.py` and `measure_efficiency.py` could score
only `comb_strength` / `comb_stat_strength` / `nmf_peak` — so **neither could measure any
margin detector**, including the one the paper already proposes. Fixed:

* `comb_artifacts.comb_calibrated_score(feats, name)` is now the single definition of
  every calibrated score, resolving `comb_priormax{M}_{margin,hmargin,latmargin,unionmargin}`
  from a feature dict, raising on a missing null with the flag that would supply it.
* Both scripts take the eight calibrated names and call it; the operator dict auto-sets
  `n_harm_grid` from the name, and sets `null_priors=0` for `*_hmargin` / `*_latmargin`,
  whose nulls are deterministic. **The decoy-free null is therefore also the cheaper one**
  — a claim now measurable rather than asserted.
* `tests/test_calibrated_score.py` (18 tests) asserts the per-buffer path agrees with
  transcribed oracles of `derive_analytic_null.py` and `derive_prior_margin.py:94`.

Suite: **475 collected, all pass.** Commands in `RUNBOOK_2026-09-17_NULLS.md` Part 2
Step A, including the incumbent through the same battery so the comparison is like for
like. **No robustness or cost number for this detector exists yet** — the paper's
+0.3/+0.4 EER and 380 audio-s/s remain the raw comb arm's.

## R27.49 — two gaps found while writing up, both zero-compute, both open

**(a) `comb_priormax4_hmargin` is the only harmonic-null score ever scored.** M=4 was
pre-registered from physics (R27.41) — the shallowest depth reaching SAO's 107.4 Hz tooth
and Suno's 399.9 Hz — but **M=2 was never measured for this null**, so the depth claim
rests on an argument, not a comparison. That is the same defect R14 introduced (M=2
selected on a 0.0002 macro gap) and this programme was opened to fix.

`comb_priormax2_hmargin` is **already in both `analytic_null.csv` files**: the extraction
ran `--n-harm 2 4` (M=8 has no admissible *lattice* null, R27.17 — that constraint does
not obviously bind the *harmonic* null, which is a separate open question),
`comb_features` emits `_hmax_strength` per M, and `derive_analytic_null.py:181` loops over
every M present. **The row has never been read** — the same oversight as the unread M=8
row that produced §1.4. Runbook Step A0.1.

**(b) Leave-one-out has never been run.** `grep -i "leave-one-out"` returns **zero hits
across 3,249 ledger lines**, though HANDOVER §0 lists it as mandatory. It did not bind
while nothing was fitted; M is now a two-way choice made with knowledge of which families
failed. Cheap form: hold out one family, choose M on the other four, score the held-out
one. Vacuous if (a) shows M=2 and M=4 tie — run (a) first. Runbook Step A0.2.

Neither gap changes any number already reported. Both are recorded before the runs so the
answers cannot be selected after the fact.

## R27.50 ⭐ Efficiency: the decoy-free detector is 2.36× faster, and it is the ONLY calibrated score that runs at the paper's published speed

`data/processed/efficiency_comb_priormax4_hmargin.json`,
`data/processed/efficiency_comb_priormax2_margin.json`. 60 tracks, 5 trials, 9 s clips,
4 CPU cores. First numbers ever produced for a calibrated detector (R27.44 recorded that
the script could not score one).

| score | tracks/s (mean ± SD, n=5) | audio-s/s | decoys |
|---|---|---|---|
| `comb_priormax4_hmargin` | **41.99 ± 0.29** | **377.9** | **0** |
| `comb_priormax2_margin` (proposed in the paper) | 17.81 ± 0.05 | **160.3** | 24 |

**2.36× (41.99/17.81 = 2.358, 377.9/160.3 = 2.357).** The mechanism is not subtle: the
decoy margin evaluates 24 extra prior sets per track and the harmonic null evaluates none.
The determinism argument and the speed argument are the same argument.

**⚠ This exposes a defect in `paper.tex:526`.** The cost sentence reads "No trainable
parameter and no training, at 380 audio-seconds per second", and its `% src:` comment
points at `robustness_comb_median_db/` — **the raw `comb_strength` arm**. In a paragraph
about the proposed system, a reader will read 380 as the proposed detector's throughput.
**It is not: `comb_priormax2_margin` runs at 160.3 audio-s/s, 2.4× slower than the quoted
figure.** `comb_priormax4_hmargin`'s 377.9 is within 0.6% of 380.

So the camera-ready has exactly two honest options: **(a)** switch the proposed detector
to `hmargin`, and 380 stands essentially unchanged; or **(b)** keep the decoy margin and
correct the sentence to ~160 audio-s/s. This is now an argument *for* the new detector
rather than a neutral improvement.

**Not yet closed:** `comb_strength` itself has not been timed through *this* script, so
the 377.9-vs-380 agreement is across two measurement paths. One 30-second run settles it
(runbook Step A).

## R27.51 ⭐ C10 on FakeMusicCaps: CLEAN PASS — the Griffin-Lim caveat is SONICS-only

`reports/diagnostics/c10_h4_fmc/recon_pairs.csv`, 300 pairs per variant, after
re-extracting the recon control at `comb_recon_fmc_HNULL` with `--n-harm 2 4`.

| variant | frac_above | median Δ | Wilcoxon p |
|---|---|---|---|
| EnCodec 3 kbps | 0.980 | +0.0702 | 3.1e-48 |
| EnCodec 6 kbps | 0.970 | +0.0698 | 9.9e-48 |
| EnCodec 24 kbps | 0.960 | +0.0656 | 2.4e-47 |
| Griffin-Lim 128 mel | 0.543 | +0.0017 | **0.154** |
| Griffin-Lim 256 mel | 0.470 | −0.0010 | **0.733** |

**Griffin-Lim sits at chance on both variants** — one of them *below* 0.5. EnCodec is
0.96–0.98 with p ≈ 1e-47. This is the textbook C10 result and it **PASSES**.

**This materially revises R27.46.** The caveat recorded there is **corpus-specific, not a
property of the score**: on FakeMusicCaps the deconvolution-specific reading is clean; on
SONICS Griffin-Lim reaches 0.583 at p = 0.0014 with a median Δ of +0.0040, 33–40× below
EnCodec's. The correct sentence is now **"deconvolution-specific on FakeMusicCaps; on
SONICS a small Griffin-Lim shift is detectable, two orders of magnitude below the
codec's"** — not the blanket caveat R27.46 stated while only SONICS had been run.

I recorded the caveat before this run and am not walking it back now that it is
convenient: it is real, it is confined to one corpus, and both corpora are reported.

**Also from the same run** (`analytic_null.csv` zero-atom diagnostic, 1800 all-real rows):
`comb_priormax4_hmargin` and `comb_priormax2_hmargin` both **0.0000** exactly-zero, while
`comb_priormax4_margin` is **0.2989** and `comb_priormax2_margin` 0.0389. The tie-mass
mechanism reproduces on a corpus neither AUC touched.

## R27.52 — my A0.1 path was wrong, and the harmonic extraction's output dir is unrecorded

Runbook A0.1 sent the M=2 check at `comb_*_DETNULL3/analytic_null.csv`. **That pass
predates `comb_prior_harmonic_null` and has no `hmax` columns** — it carries
`_latmax_strength`, `_unionmax_strength` and `_pm1_*` only. Both invocations exited with
`no column 'comb_priormax2_hmargin'` and left two empty output dirs.

**The premise survives; the path did not.** `derive_analytic_null.py:181` emits `_hmargin`
for every M with an `hmax_strength` column, and the fresh `comb_recon_fmc_HNULL` run lists
**both** `comb_priormax2_hmargin` and `comb_priormax4_hmargin` — so the M=2 score is there
to be read in whichever full-corpus directory the harmonic pass wrote.

**Which directory that is appears nowhere** — not in this ledger, not in any runbook,
not in `print_paper_numbers.sh`. Third instance of this class (the HARM2 dirs, the recon
manifests, now this). The runbook A0.1 now *discovers* the directory by grepping every
`analytic_null.csv` header for the column, instead of naming one.

## R27.53 — housekeeping closed

`head -3 reports/diagnostics/comb_sonics_HARM2/per_track_calibrated_auc.csv` returns a
single well-formed header. **The duplicated-header worry from an earlier paste was a
paste artefact, not a file defect.** Withdrawn.

Incidental: that table's top row is `comb_priormax4_margin` at SONICS MACRO **0.8312**,
above the shipped `comb_priormax2_margin`'s 0.8271 — i.e. prior *depth* alone helps SONICS
even with the decoy null. Consistent with §1.4 and not previously noted.

## R27.54 ⭐⭐ M=2 vs M=4 harmonic null: a dead heat — which REFUTES the prior-depth explanation

`reports/diagnostics/analysis_h2_fmc/`. FakeMusicCaps, recall at q=0.95:

| family | **M=2 hmargin** | M=4 hmargin | shipped `pm2_margin` |
|---|---|---|---|
| MusicGen_medium | 0.6953 | **0.6924** | 0.169 |
| stable_audio_open | **0.7711** | — | **0.0308** |
| musicldm | 0.9201 | — | — |
| audioldm2 | 0.9656 | — | — |
| mustango | 0.9989 | — | — |
| **worst family** | **0.6953** (MusicGen) | 0.6924 (MusicGen) | 0.0308 (SAO) |
| **pooled TPR @ q95** | 0.8702 | **0.8760** | 0.6246 |
| FPR at q95 | 0.0411 | — | — |

**M=2 is 0.0029 better on the worst family and 0.0058 worse pooled. That is a tie.**

**This refutes §1.4's prior-depth story for this null, and I am recording it as a
refutation of my own prediction.** The argument was that Stable Audio Open fails because
its visible tooth at 21.53 x 5 = 107.65 Hz lies outside the prior's reach, so a deeper
prior is needed. But at M=2 the prior reaches at most 2*f, and 107.65 Hz is the **fifth**
harmonic — the prior does not reach it at M=2 or at M=4 — yet SAO's recall is **0.7711**
against the shipped detector's 0.0308.

**The fix was never prior-side. It is entirely null-side.** With drawn decoys, lag 110 sat
*inside* the null, so `max(null)` was inflated by SAO's own comb and the margin collapsed
to ~0. `comb_prior_harmonic_null` excludes `k*Delta` for **every** integer k
(`comb_artifacts.py`, the `while True` loop over k — not bounded by `n_harm`), so lag 110
is excluded and the detector stops subtracting its own evidence. M changes only the prior
depth and the null's upper limit (`hi = n_harm * hi_hz`); neither matters here.

**Consequence for the paper.** This is a *free sensitivity analysis and a strong one*: the
improvement is insensitive to the one hyperparameter a reviewer would attack. **Keep M=4
as the detector** — it is the depth that carries every gate, C10, Mondrian and transfer
result — and report the M=2 tie as evidence that the gain is attributable to the null
construction rather than to prior depth. No re-gating needed, and leave-one-out over M
(R27.49b) is now **vacuous**: there is no selection to cross-validate between two tied
options. Record that LOO was considered and is not applicable, rather than silently
dropping it.

SONICS M=2 (`analysis_h2_sonics/`): udio-120s **0.0179**, udio-30s **0.0182**, chirp-v3
0.9020, chirp-v3.5 0.9331, chirp-v2-xxl 0.9579; pooled TPR@q95 0.4824, FPR 0.0473. Udio
remains at floor, as R7 predicts. **The M=4 SONICS per-family recall was never pasted** —
`analysis_h4_sonics/errors_by_generator.csv` exists and is one `cat` away; the M comparison
is FMC-only until it is read.

## R27.55 ⚠ CORRECTION to R27.50: the `comb_strength` control was run with the decoys ON

Second efficiency pass, run **while the robustness battery was occupying the box**:

| score | tracks/s | audio-s/s | flags as logged |
|---|---|---|---|
| `comb_priormax4_hmargin` | 36.13 ± 1.85 | 324.3 | `null_priors: 0`, `n_harm_grid: (4,)` |
| `comb_priormax2_margin` | 15.21 ± 0.52 | 136.8 | `null_priors: 24`, `(2,)` |
| `comb_strength` | 14.76 ± 0.43 | 132.8 | **`null_priors: 24`**, `(2,)` |

**Two things, one solid and one withdrawn.**

**Solid — the speedup is load-invariant.** 324.3/136.8 = **2.37x** against the idle run's
377.9/160.3 = **2.36x**. All three scores fell ~14% under contention and hmargin's SD rose
0.29 -> 1.85, so the absolute figures from the loaded run must not be quoted; the **ratio**
reproduces to two decimals and is the claim.

**Withdrawn — "377.9 ≈ the paper's 380, therefore hmargin restores the published speed."**
My runbook built the `--null-priors` flag with a shell conditional that emitted 0 only for
`hmargin`, so **`comb_strength` was timed with all 24 decoy prior sets still being
computed** — the log confirms `'null_priors': 24`. `comb_features` evaluates them whether
or not the requested score reads them. That is exactly why `comb_strength` (132.8) and
`comb_priormax2_margin` (136.8) came out equal: they were doing the same work. **132.8 is
not the raw comb arm's throughput and the R27.50 comparison against `paper.tex:526`'s 380
is not established.** My error, in the command I wrote.

Re-run needed: `comb_strength` at **`--null-priors 0`**, on an idle box. Prediction: it
lands at or above hmargin's ~378, since it skips the prior max and the harmonic scan. If
it does, the 380 figure is consistent with a decoy-free score and R27.50's conclusion is
restored on correct evidence. If it lands near 137 even with the decoys off, then 380 came
from a different measurement path entirely and **no claim about it may be made from this
script**.

Unchanged and still true: `comb_priormax2_margin`, the detector the paper proposes, runs
**2.37x slower than `comb_priormax4_hmargin`**.

## R27.56 — the robustness checkpoint: margin-type confirmed, exact provenance still open

`data/processed/robustness_hmargin/*checkpoint*` first rows: `comb_score` **+0.00361**
(baseline) and **-0.04413** (pitch_shift) for `fake_48336_suno_1`.

**The negative value proves these rows are a MARGIN, not `comb_strength`** — a normalised
ACF peak height cannot be negative. So the 337 resumed rows are not cross-contamination
with the raw arm. They do **not** prove the score was `hmargin` rather than another margin.

Decisive and cheap, once the battery exits: take three `track_id`s from the resumed rows,
re-score them in isolation at the same flags, and compare bit for bit. Runbook Step A.

Battery timing observed: ~106 s per 100 tracks => **~88 min for 5,000**, then the
incumbent arm at 2.4x the per-track cost.

## R27.57 ⭐ Efficiency, settled: the paper's 380 is REAL, and it is the raw arm's

`comb_strength --null-priors 0`, run twice. Loaded box (11:45, battery still running):
38.57 ± 3.2 tracks/s, **344.7** audio-s/s. **Idle box (12:55): 43.10 ± 0.70 tracks/s,
387.8 audio-s/s.** Take the idle figure; the SD collapses from 3.2 to 0.7.

| score | audio-s/s (idle) | vs raw arm | decoys |
|---|---|---|---|
| `comb_strength` — the raw arm | **387.8** | 100% | 0 |
| `comb_priormax4_hmargin` | **377.9** | **97.4%** | **0** |
| `comb_priormax2_margin` — what the paper proposes | 160.3 | **41.3%** | 24 |

**`paper.tex:526`'s 380 audio-s/s is confirmed at 387.8 through this script** (2% apart, a
different machine). **R27.55's withdrawal is lifted and R27.50's conclusion is restored on
correct evidence:**

* the published 380 belongs to the **raw comb arm**, as its `% src:` comment always said;
* the **proposed** detector runs at **160** — the printed figure overstates it **2.4x**;
* **`hmargin` costs 2.6% against the raw arm.** Per-track calibration is essentially free
  when the null is deterministic, and expensive when it is drawn.

The camera-ready therefore has a clean statement available for the first time: *the
calibrated, deterministic detector runs at 378 audio-s/s, within 3% of the uncalibrated
arm's 388, with zero trainable parameters.* The decoy margin cannot make that claim.

## R27.58 ⭐ Robustness: the LAST experiment. The paper's claims hold; pitch-shift inverts both.

`data/processed/robustness_hmargin/`, `robustness_pm2margin/`, 998 real + 4,000 fake,
39,986 scored rows each. **First robustness numbers ever produced for a calibrated
detector** (R27.44).

| manipulation | `pm4_hmargin` AUC / EER% | `pm2_margin` AUC / EER% | MusicDET T6 EER% |
|---|---|---|---|
| baseline | 0.7618 / 32.44 | **0.7737** / 32.31 | — |
| aac_64k | 0.7619 / 32.67 | **0.7753** / 31.53 | 35.85 |
| opus_64k | 0.7634 / 33.38 | **0.7729** / 31.68 | 22.15 |
| equalization | 0.7609 / 32.28 | **0.7741** / 31.78 | 6.04 |
| white_noise | 0.6632 / 39.12 | 0.6658 / 39.51 | 44.11 |
| reverberation | 0.6051 / 43.94 | **0.6288** / 42.68 | 4.04 |
| time_stretch | **0.5509** / 46.01 | 0.5165 / 48.34 | 2.44 |
| pitch_shift | **0.4665** / 52.03 | 0.4571 / 53.45 | 44.73 |

**(a) The paper's robustness sentence survives for a calibrated detector.** It claims
"+0.3/+0.4 EER under AAC and Opus-64k ... against +14.7 under time-stretch" for the **comb
arm**. Measured on the margins, against each one's own baseline:

| | AAC | Opus | time-stretch |
|---|---|---|---|
| `hmargin` | **+0.23** | **+0.94** | **+13.57** |
| `pm2_margin` | **-0.78** | **-0.63** | **+16.03** |

Both keep compression under ~1 EER point and both lose ~14 to time-stretch. **The
mechanism claim — a stride-determined periodicity survives transcoding and does not survive
resampling — reproduces on the detector the paper proposes and on the one replacing it.**

**(b) ⚠ NEW, and it is an inversion: pitch-shift takes both detectors BELOW chance.**
AUC **0.4665** and **0.4571**, EER 52-53%. Mechanism-consistent — pitch-shifting resamples,
which *moves* Delta off the six-frequency prior, so the prior systematically misses on fakes
while real music is unaffected. But the project rule is signed AUC, and this is signed below
0.5. **The paper does not currently mention pitch-shift. It must, or the manipulation set
must be stated as the one it reports.** `hmargin` is the less inverted of the two.

**(c) Head to head the two are a wash, and this subset cannot separate them.**
`pm2_margin` is ahead on 6 of 8 (by 0.003-0.024) and `hmargin` on 2 (pitch-shift +0.009,
time-stretch **+0.034**). Mean gap **+0.0038 to `pm2_margin`** — smaller than the AUC
standard error at this n, and **no DeLong was run**, so *no* ordering here is established.

**Why the subset cannot separate them, and this must be checked before anything is quoted:**
the baseline AUC is **0.76-0.77**, not the 0.94 `hmargin` reaches on FakeMusicCaps. The
resumed checkpoint's first row is `fake_48336_suno_1`, `algorithm: chirp-v3` — **a SONICS
track**. If `bitrate_sweep_subset.csv` is SONICS-weighted or mixed, then this battery runs
exactly where the two detectors are already known to be **tied** (DeLong p = 0.106), and a
0.004 mean gap is the expected result rather than an informative one. **Verify the subset's
composition before drawing any conclusion from (c).**

## R27.59 — correction: `analysis_h4_sonics/` does not exist

R27.47 listed `analysis_h4_sonics/` among its sources. `cat` returns *No such file or
directory*; the directory was never created and every row in that table is FakeMusicCaps,
FMA or Mondrian. **The source list was wrong, the numbers were not.** Corrected here.

Consequence: the M=2 vs M=4 comparison (R27.54) remains **FakeMusicCaps-only**. Completing it
on SONICS is a `analyze_detector.py` *run*, not a `cat`.

## R27.60 — WHY the decoy path is slow: 97% of it is the autocorrelation, recomputed 48 times

Local timing, 8,000-bin residual, the real geometry:

| operation | time | note |
|---|---|---|
| one `_normalised_acf` | **0.124 ms** | two FFTs of an 8,000-point array |
| `comb_prior_harmonic_null` | **0.325 ms** | one ACF + a boolean mask + one `max` |
| the 24-decoy loop | **6.181 ms** | **19.0x** the harmonic null |

`comb_features:1624-1631` calls `comb_prior_strength` **and** `comb_prior_max` per decoy
set, and **each of those calls `_normalised_acf(residual)` again**. 24 sets x 2 calls x
0.124 ms = 5.95 ms of the 6.18 ms measured — **97%**.

**So the speedup is not "fewer lags to look at".** Indexing a few hundred lags out of a
precomputed array is free. It is **48 avoided FFT pairs on a residual whose autocorrelation
never changes.** Worth stating precisely, because the obvious explanation is the wrong one.

*(This also means the decoy null could have been made ~48x cheaper by hoisting the ACF out
of the loop, without changing a single number. It was not, and that is a fair criticism of
the incumbent implementation rather than of the decoy idea. The harmonic null needs no such
fix.)*

## R27.61 — the null's 600 Hz ceiling is NOT a cost decision

Taking `max` over **all** ~7,959 lags costs **0.8 microseconds**. Cost cannot be the reason
for the range, so the reason must be statistical — and the measured effect is modest:

| null range | lags in N | max over N | margin |
|---|---|---|---|
| **M x 150 = 600 Hz (what we do)** | **441** | 0.0359 | −0.0203 |
| 1500 Hz | 1,150 | 0.0359 | −0.0203 |
| 4000 Hz | 3,130 | 0.0395 | −0.0238 |
| everything | 6,139 | 0.0395 | −0.0238 |

**Correcting my own overstatement.** I wrote that widening 441 -> 4,000 inflates the baseline
by `sqrt(ln 4000 / ln 441) = 1.17`, i.e. 17%. **Measured it is ~10%** (0.0359 -> 0.0395), and
600 -> 1500 Hz changes nothing at all. The `sqrt(ln N)` form is an asymptotic heuristic for
independent draws; adjacent ACF lags are correlated, so it over-predicts. Quote the measured
10%, not the heuristic.

**The load-bearing reason is the hypothesis space, not the extreme value.** `hi_hz = 150` is
the highest plausible decoder fundamental, so `[40, M x 150]` is exactly the lag set reachable
*if the fundamental were anything up to 150 Hz*. That makes **N the same hypothesis space as P
with the six real hypotheses removed** — which is what makes the subtraction a hypothesis test
rather than an arbitrary offset.

## R27.62 ⚡ NEW LEVER — the 40 Hz floor's stated rationale does not reproduce, and lowering it helps

`comb_prior_max`'s docstring justifies `min_spacing_hz = 40` as excluding *"the very short lags
where an autocorrelation reflects residual smoothness rather than a comb"*. **Measured on 25
comb-free tracks at the real geometry, that is true for the HULL residual and FALSE for the
MEDIAN residual we actually use:**

| residual operator | mean rho, lags 2–40 | mean rho, lags 41–614 |
|---|---|---|
| median, 5 bins — **our operator** | **−0.0036** | −0.0000 |
| median, 65 bins | −0.0076 | −0.0001 |
| hull (Afchar's, clipped at 0) | **+0.0073** | −0.0003 |

The 5-bin median already removes everything slower than 5 bins, so the residual is nearly
white and there is **no elevated short-lag floor to protect against**. The hull residual is
clipped at zero, is therefore non-negative and short-lag correlated, and the floor *is*
motivated there. **The floor was inherited from the hull path into the median path with its
rationale intact but its premise absent.**

**And it costs us.** Lowering it admits Stable Audio Open's own fundamental — 21.53 Hz, lag
22 — which is *currently excluded at every M*:

| floor | \|P\| | \|N\| | AUC, Stable-Audio-like comb |
|---|---|---|---|
| **40 Hz (current)** | 18 | 441 | **0.6350** |
| 30 Hz | 18 | 451 | 0.6381 |
| 25 Hz | 18 | 456 | 0.6381 |
| **20 Hz** | **19** | 459 | **0.7275** |
| 15 Hz | 19 | 464 | 0.7275 |

**+0.093 AUC**, and the step lands exactly where lag 22 enters P. The other families are
unharmed: MusicGen **+0.0006**, AudioLDM2 **+0.0100**, Suno chirp **+0.0081**.

⚠ **This is synthetic and must not be quoted as a corpus result.** Whether SAO's 21.53 Hz
fundamental is actually present in 16 kHz-band audio is unknown — the ledger only records its
*fifth* harmonic at 107.42 Hz as visible. **But it is the first new lever in several rounds,
it is one extraction, and it targets the family the paper names as its deployment failure.**
Recorded as an open experiment: the programme is **not** exhausted after all.

## R27.63 ⭐⭐ Making the prior EXHAUSTIVE destroys the detector — the restriction IS the method

The natural question after R27.54: MusicGen's loudest tooth is its 5th, at 250 Hz, which the
prior reaches at no M we use. Would a deeper prior catch it? Simulated at the real geometry,
MusicGen-like comb (Δ = 50 Hz, 5th tooth loudest), 40 seeds per arm against 40 comb-free:

| M | \|P\| | reach | is 250 Hz in P? | AUC vs real | `sqrt(ln\|P\|/ln 4000)` |
|---|---|---|---|---|---|
| **2** | 10 | 200 Hz | no | **0.7606** | 0.527 |
| 4 | 18 | 400 Hz | no | 0.6725 | 0.590 |
| 6 | 28 | 600 Hz | **yes** | 0.6369 | 0.634 |
| 8 | 36 | 800 Hz | yes | 0.5931 | 0.657 |
| 12 | 54 | 1200 Hz | yes | **0.5044 — chance** | 0.694 |

**AUC falls monotonically, and keeps falling after the target tooth enters the prior at M=6.**
The extreme-value penalty from a larger candidate set outweighs the gain from reaching the
tooth, every time. At M=12 the detector is at chance.

**So the answer to "if we searched exhaustively, would we still miss MusicGen?" is: we would
miss everything.** The prior's *smallness* is not a convenience — it is the mechanism. This is
the cleanest statement of the method available and it belongs in the paper.

It also independently corroborates R27.54's corpus tie: the fixture ranks M=2 >= M=4, the
corpus ranked them equal. The fixture exaggerates the penalty (it has no real musical
structure to break ties) but the direction is the same, and **no evidence anywhere supports a
deeper prior.**

## R27.64 ⚠ CORRECTION — widening the null is not worse. It is very slightly BETTER.

R27.61 measured one margin and I reasoned from it that the 600 Hz ceiling was the right
choice. **That was the wrong measurement.** A uniform shift in `max(null)` changes every
track's margin equally and cannot change the ranking, so the decisive quantity is AUC, not a
margin. Measured, 40 tracks per family against 40 comb-free:

| null ceiling | \|N\| | MusicGen | StableAudio | AudioLDM2 | Suno | **mean** |
|---|---|---|---|---|---|---|
| **M x 150 = 600 Hz (published)** | 441 | 1.0000 | 1.0000 | 0.9550 | 0.9962 | **0.9878** |
| 1500 Hz | 1,150 | 1.0000 | 1.0000 | **0.9731** | 0.9975 | **0.9927** |
| 4000 Hz | 3,130 | 1.0000 | 1.0000 | 0.9725 | 0.9975 | 0.9925 |
| everything (~7,800) | 6,128 | 1.0000 | 1.0000 | 0.9725 | 0.9975 | 0.9925 |

**+0.005 mean AUC for widening, saturating by 1500 Hz.** And rho's null distribution is close
enough to stationary for the comparison to be legitimate:

| lag block | mean rho | sd | E[max over block] |
|---|---|---|---|
| 41–200 | −0.00001 | 0.01275 | 0.0328 |
| 200–614 | −0.00004 | 0.01254 | 0.0370 |
| 614–1500 | −0.00003 | 0.01190 | 0.0383 |
| 1500–4000 | −0.00002 | 0.01033 | 0.0362 |
| 4000–7900 | −0.00001 | 0.00645 | 0.0272 |

The sd *falls* at long lags — a linear correlation has fewer overlapping terms there — so the
extreme-value penalty I predicted does not materialise; `E[max]` peaks around 614–1500 and
declines after.

**What survives of the argument for the cap is semantic, not statistical.** The null's
`pval` (`mean(vals >= real)`) is a p-value against *whatever hypothesis class the null
spans*. Capped at `M x 150` it reads "against every plausible decoder-like fundamental up to
150 Hz". Uncapped it reads "against any periodicity whatsoever", which is a different and
weaker claim. **Ranking is unaffected; the meaning of the reported p is not.**

**And it does NOT remove M.** M has two jobs — it sets the prior's depth (`m = 1..M`) and the
null's ceiling (`n_harm * hi_hz`). Widening retires the second. The first cannot be retired:
R27.63 shows an exhaustive prior falls to chance. **Dropping the cap removes one of M's jobs,
not the parameter.**

Both factors are now emitted and gated together (R27.66).

## R27.65 ⭐ HOW a track is misclassified, measured — and a retraction of my own numerology

**False positives.** A sustained note at `f0` puts harmonics every `f0` Hz: *its spectrum is
itself a comb of spacing `f0`*. If `f0` lands on a decoder fundamental the detector cannot
tell them apart. Measured, 40 seeds per condition, flagged at margin > 0:

| bass content | mean margin | flagged |
|---|---|---|
| A2 = 110 Hz (a tone away) | −0.0165 | **0.0%** |
| G#2 = 103.83 Hz | −0.0159 | 5.0% |
| G2 = 97.999 Hz — 35 cents from the 100 Hz prior | −0.0157 | **2.5%** |
| **exactly 100 Hz** | −0.0008 | **47.5%** |
| D2 = 73.416 Hz — 37 cents from the 75 Hz prior | −0.0176 | **0.0%** |
| **exactly 75 Hz** | −0.0004 | **47.5%** |
| **exactly 50 Hz** | −0.0005 | **47.5%** |
| C#3 = 138.59 Hz (between priors) | −0.0179 | 0.0% |

**⚠ RETRACTION of something I was about to claim.** I computed that all six decoder
fundamentals sit within 23–37 cents of an equal-tempered pitch and was going to present that
as a hazard. **It is numerology.** Any frequency is within 50 cents of *some* pitch by
construction; six of six within 37 cents has probability `0.74^6 = 0.16`. Worse, the
measurement says the opposite of the alarming reading: **a note 23–37 cents off a decoder
frequency sits at the 0–5% baseline.** At this resolution 98 Hz is lag 100 and 100 Hz is lag
102 — two bins apart, and with `tol_bins = 0` that is enough to separate them. **Equal
temperament misses the decoder frequencies by exactly enough.**

**The real hazard is an EXACT hit, and there is a concrete one: 50 Hz mains hum.** European
mains and its harmonic series land on the 50 Hz prior and its multiples exactly, not
approximately. Synthesised drones tuned to round numbers are the other case. Both are
plausible in real catalogues and neither is tested by anything in the battery. **Worth a
sentence in the limitations, and worth a control.**

**False negatives**, same fixture, caught at margin > 0:

| escape mode | mean margin | caught | why |
|---|---|---|---|
| MusicGen, loudest tooth the 5th | +0.0334 | **100%** | the harmonic null already fixed this |
| a quiet comb (4 dB teeth) | −0.0130 | 5.0% | buried in the residual |
| no comb at all (Udio-like) | −0.0137 | 5.0% | nothing in band — R7 |
| **pitch-shifted +3%** | −0.0282 | **0.0%** | resampling moved Δ off every prior lag |

Pitch-shift is not merely missed, it is **anti-detected** — the margin goes *more* negative
than a real track's, which is why the corpus AUC is 0.4665 (R27.58). Consistent across
fixture and corpus.

## R27.66 — the 2x2 probe is implemented, and the published columns did not move

`comb_artifacts.py`: `LO20_HZ = 20.0`, `WIDE_HI_HZ = 1000.0`, a `CALIBRATED_REAL_KEY` table
beside the existing null table, and four new columns per M
(`_lo20_strength`, `_lo20_hmax_strength`, `_wide_hmax_strength`, `_lo20wide_hmax_strength`,
plus `_hnlags` for every arm). `comb_calibrated_score` resolves
`lo20margin` / `widemargin` / `lo20widemargin`; `derive_analytic_null.py` emits them from a
CSV; both deployment scripts accept them and set `null_priors = 0` (all three arms are
decoy-free).

**The lowered floor is given to the prior AND the null.** `lo20margin` reads
`_lo20_strength`, not the published `_strength` — a floor applied to the null alone would
enlarge the null and shrink every margin, a guaranteed loss dressed as a test.

Verification: **487 tests pass** (13 new in `tests/test_floor_ceiling_probe.py`), and against
`HEAD` on identical audio **171 published columns, 16 added, zero changed value**.

Null sizes at M=4 on the FakeMusicCaps geometry: narrow **441**, lo20 **459**, wide **3130**,
lo20wide **3148** — matching the simulation exactly.

## R27.67 ⛔ RETRACTION 8 — R27.63's "an exhaustive prior falls to chance" does NOT reproduce

R27.63 swept prior depth and reported AUC **0.7606 → 0.5044** from M=2 to M=12, concluding
*"the prior's smallness IS the mechanism."* **Two defects, both mine.**

**(a) The sweep confounded two variables.** `hi = n_harm * hi_hz`, so deepening the prior also
widened the null. The collapse was attributed entirely to the prior. **Controlled** — null held
fixed at 4000 Hz while only the prior moves:

| M | \|P\| | mean AUC, null TIED to M | mean AUC, null FIXED at 4 kHz |
|---|---|---|---|
| 2 | 10 | 0.9878 | **0.9934** |
| 4 | 18 | 0.9878 | 0.9925 |
| 8 | 36 | 0.9877 | 0.9875 |
| 12 | 54 | 0.9822 | 0.9819 |
| 20 | 91 | 0.9742 | 0.9742 |
| 40 | 183 | 0.9722 | 0.9722 |

**Nothing collapses in either column.** M=2 → M=40 costs about **0.02 AUC**, not 0.46.

**(b) The fixture was weaker than the one every other table in this round used.** R27.63 ran at
`amp = 6` dB teeth; the end-to-end tables ran at `amp = 12`. I changed the fixture between
rounds and compared across them. Swept properly:

| tooth amplitude | M=2 | M=4 | M=8 | M=12 | M=40 | M=2 → M=40 |
|---|---|---|---|---|---|---|
| 3 dB | 0.6369 | 0.6969 | 0.6250 | 0.6281 | 0.6025 | −0.034 |
| 6 dB | 0.8275 | 0.8400 | 0.7863 | 0.7675 | 0.7206 | **−0.107** |
| 8 dB | 0.9437 | 0.9469 | 0.9187 | 0.8875 | 0.8544 | −0.089 |
| **12 dB** | 1.0000 | 1.0000 | 1.0000 | 0.9981 | 0.9956 | **−0.004** |
| 18 dB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000 |

**WITHDRAWN:** "an exhaustive prior falls to chance"; "the prior's smallness *is* the
mechanism"; the 0.5044 figure. Published in the ledger, in the explainer §15.5 and §16.1, and
on slide 10 of the deck. **All three are corrected in this round.**

**WHAT SURVIVES, and it is still worth saying:**

1. **A deeper prior is never better** — monotone or flat in every arm of every run.
2. **The penalty scales with how faint the comb is**: ~0.004 AUC at 12 dB teeth, ~0.107 at
   6 dB. The restriction buys noise suppression that matters *most exactly where the detector
   is weakest*, which is where MusicGen and Stable Audio Open live.
3. **So M is irreducible but not delicate.** You must choose some depth; the choice is monotone
   and the best value is the smallest that covers all six architectures — **M = 2**
   (M = 1 leaves Stable Audio unrepresented, R27.62).

This is a weaker claim than R27.63's and a more defensible one. It is also the honest answer to
"can we throw M?": **no, but you can pin it by physics instead of tuning it.**

## R27.68 — the decoupling the question was really reaching for

M currently does **two** jobs with one number: the prior's depth and the null's ceiling. They
are unrelated, and R27.64 shows the null wants to be wide while R27.67 shows the prior wants to
be shallow — **the single parameter is pulling in two directions at once.**

Decoupled: **null ceiling fixed and wide; prior depth pinned at the physical minimum.** One
parameter, one justification, instead of one parameter with two conflicting ones.

**This is testable from the extraction already running, with no re-extraction**, because
`comb_features` emits `_wide_hmax_strength` at both M=2 and M=4. Crossing them gives a prior of
depth 2 against the M=4 (4 kHz) null:

* `comb_priormax{m}_xwidemargin` = `comb_priormax{m}_strength` − `comb_priormax4_wide_hmax_strength`
* `comb_priormax{m}_xlo20widemargin` = `comb_priormax{m}_lo20_strength` − `comb_priormax4_lo20wide_hmax_strength`

At m = 4 these coincide with the existing `widemargin` / `lo20widemargin`; at m = 2 they are new
and genuinely decoupled. Together with the existing arms this makes the probe a full
**2 × 2 × 2**: floor {40, 20} × null {narrow, 4 kHz} × prior depth {2, 4}. Implemented at CSV
level in `derive_analytic_null.py` only — **no feature code changed, so the running extraction
stays valid.**

## R27.69 ⭐ The sharpest form of "can we throw M": with a 20 Hz floor, M = 1 becomes ADMISSIBLE

The floor and the depth are not independent questions. At the published 40 Hz floor, M = 1
leaves Stable Audio unrepresented (lag 22 < lag 41), which is *why* M >= 2 is forced. **Lower
the floor to 20 Hz and lag 22 clears it — so at M = 1 all six architectures have a candidate
and the prior becomes literally "the six published fundamentals", with no depth choice left to
make.** Simulated, null fixed at 4 kHz, 40 tracks per arm:

| floor | M | \|P\| | MusicGen | StableAudio | AudioLDM2 | Suno | **mean** | covers all six? |
|---|---|---|---|---|---|---|---|---|
| 40 Hz | 1 | 5 | 1.0000 | 0.9825 | 0.9444 | 1.0000 | 0.9817 | **no** |
| 40 Hz | **2** | 10 | 1.0000 | 0.9988 | 0.9769 | 0.9981 | **0.9934** | yes |
| 40 Hz | 4 | 18 | 1.0000 | 1.0000 | 0.9725 | 0.9975 | 0.9925 | yes |
| **20 Hz** | **1** | **6** | 1.0000 | **1.0000** | 0.9656 | 1.0000 | **0.9914** | **yes** |
| **20 Hz** | **2** | 11 | 1.0000 | 1.0000 | 0.9794 | 0.9981 | **0.9944** | yes |
| 20 Hz | 4 | 19 | 1.0000 | 1.0000 | 0.9756 | 0.9975 | 0.9933 | yes |
| 20 Hz | 8 | 37 | 1.0000 | 1.0000 | 0.9581 | 0.9962 | 0.9886 | yes |

**Two findings.**

1. **The lowered floor helps at every depth**, not only at M = 1: +0.0010 at M = 2, +0.0008 at
   M = 4, and **+0.0097 at M = 1** (Stable Audio 0.9825 -> 1.0000). Independent support for
   R27.62 from a different direction.
2. **M = 1 becomes viable but is not free.** 0.9914 against M = 2's 0.9944 — the 0.003 lands
   almost entirely on AudioLDM2 (0.9656 vs 0.9794), whose visible tooth is its *second*
   harmonic and therefore outside an M = 1 prior by construction.

**So the answer to "can we throw M?" is now precise and it is not simply no.** With the floor
at 20 Hz you *may* set M = 1, at which point there is no depth parameter at all — the prior is
the published physics and nothing else. It costs about **0.003 mean AUC** on this fixture, paid
by the one family whose evidence is a second harmonic. That is a real trade between
**parsimony** and **coverage**, and it is the corpus's to settle, not the fixture's.

⚠ **NOT testable from the running extraction** — that pass is `--n-harm 2 4`, so no
`comb_priormax1_*` columns exist. Settling it needs a second pass at `--n-harm 1 2 4`. **Do not
restart the running job for a 0.003 question**; queue it only if the floor arm wins.

## R27.70 ⭐⭐ THE PROBE LANDED. `comb_priormax4_widemargin` dominates on every family.

`reports/diagnostics/comb_fmc_PROBE/`, FakeMusicCaps, `--n-harm 1 2 4`, all 0/5 inverted.

| score | MusicGen | audioldm2 | musicldm | mustango | SAO | **MACRO** | worst-family recall @q95 |
|---|---|---|---|---|---|---|---|
| `pm4_hmargin` (control) | 0.8428 | 0.9854 | 0.9749 | 0.9963 | 0.9230 | 0.9445 | 0.6978 |
| **`pm4_widemargin`** | **0.8524** | **0.9871** | **0.9779** | **0.9973** | **0.9287** | **0.9487** | **0.7111** |
| `pm4_lo20margin` | 0.8207 | 0.9801 | 0.9696 | 0.9954 | 0.9259 | 0.9383 | 0.6906 |
| `pm4_lo20widemargin` | 0.8304 | 0.9819 | 0.9730 | 0.9964 | 0.9304 | 0.9424 | 0.7007 |
| `pm2_widemargin` | 0.8562 | 0.9857 | 0.9771 | 0.9975 | 0.9206 | 0.9474 | — |

**The wide null wins on all five families simultaneously** — +0.0096, +0.0017, +0.0030, +0.0010,
+0.0057 — and on the binding deployment number, **worst-family recall 0.6978 → 0.7111**. No
family pays. That is the first change in this whole programme that is a strict improvement with
no trade.

**The floor is a TRADE, not a win, and the prediction was only half right.** It does what
R27.62 said on Stable Audio — recall @q95 **0.8013 → 0.8287** with the narrow null and
**0.8165 → 0.8372** with the wide one, and per-family AUC 0.9230 → 0.9259 — but it *costs*
MusicGen (0.8428 → 0.8207), musicldm and audioldm2, so macro falls 0.9445 → 0.9383.
**The synthetic +0.09 did not transfer; the corpus gives +0.003 AUC on SAO and a net loss.**
Recorded as a partial confirmation: right mechanism, wrong magnitude, wrong sign on the total.

**Where the floor DOES earn its keep is exactly where R27.69 predicted** — at M = 1, where SAO
is otherwise unrepresented: `pm1_hmargin` SAO **0.8571 → 0.9084** with `lo20`, **+0.051**, the
largest floor effect anywhere. The mechanism is confirmed; it is simply not needed once M ≥ 2
puts a candidate near SAO's series by another route.

## R27.71 ⛔ Can we throw M? **NO — settled on the corpus, and M = 1 can INVERT a family.**

| null construction | M=1 | M=2 | M=4 |
|---|---|---|---|
| `hmargin` | 0.9298 | 0.9436 | **0.9445** |
| `widemargin` | 0.9264 | 0.9474 | **0.9487** |
| `lo20margin` | 0.9302 | 0.9375 | **0.9383** |
| `lo20widemargin` | 0.9303 | 0.9421 | **0.9424** |
| `latmargin` | **0.8560, n_inverted = 1** | 0.8869 | 0.9438 |
| `unionmargin` | **0.8560, n_inverted = 1** | 0.8883 | 0.9335 |
| `pm1_latmargin` | **0.8204, n_inverted = 1** | 0.8498 | 0.9314 |

**M = 1 is worse on every single construction**, by 0.012–0.022 macro — and on the three
lattice-family nulls it **inverts Stable Audio Open** (AUC 0.4788, and 0.3499 for
`pm1_latmargin`). Even with the lowered floor, which was the whole argument for M = 1, it
recovers only to 0.9303 against M = 2's 0.9421.

**My fixture under-predicted this badly** — R27.69 estimated the M=1 penalty at 0.003 mean AUC.
The corpus says 0.012–0.022 and an inversion. Synthetic combs are too clean: real weak combs
need the second harmonic that M = 1 cannot see.

**The settled answer: M is required, and M ∈ {2, 4} with no meaningful difference between them**
(`widemargin` 0.9474 vs 0.9487, a 0.0013 gap). The parameter is irreducible, insensitive across
its useful range, and has exactly one wrong setting — 1.

## R27.72 ⚠ `afmargin` tops the AUC table and must NOT be adopted — the scorecard's whole point

`comb_priormax2_afmargin` **MACRO 0.9539** and `comb_priormax1_afmargin` **0.9522**, both above
every margin arm. `comb_priormax1_afmargin` even passes C11 against the incumbent:
**AUC 0.9522 [0.9483, 0.9558], DeLong z = 3.63, p = 0.000287, Δ +0.0077.**

**Reject it anyway. Three reasons, in order of severity.**

1. **Deployment recall collapses.** Worst-family recall @q95 is **0.4659** (musicldm) against
   `widemargin`'s **0.7111**. The per-quantile table is the tell: at q99 every family is
   0.09–0.20, and **at q99.9 four of five families are exactly 0.0000**. The score *ranks*
   correctly and does not *separate* — precisely the failure `analyze_detector.py`'s own banner
   warns about, and precisely why this project judges on four numbers.
2. **C1 is worse than the incumbent's.** `spectral_flatness_hf` **0.6284** against the
   incumbent's 0.6082. It carries *more* channel exposure, not the same.
3. **C8 says it is not locking onto Stable Audio's spacing.** Per-generator concentration
   0.4016 / 0.6612 / 0.3997 / 0.6774 but **SAO 0.0143** — two orders below the others. Its SAO
   AUC is coming from something other than a consistent comb spacing.

**Mechanism hypothesis, not yet tested:** `afmargin` subtracts `comb_acf_floor` = `median(|acf|)`
over the band. A median over ~4,000 lags has *far lower across-track variance* than a max over a
null, so the subtraction removes almost no per-track scale — the score is close to the
**uncalibrated** `comb_priormax{m}_strength` plus an offset. That would explain the exact
pattern seen: good ranking, no separation. **One command settles it** — see the runbook.

**Also note what `afmargin` gets right**, because it is worth keeping in mind for the follow-up:
it is the only construction that reaches **MusicGen 0.9706**, against every margin arm's ~0.85.
Taking a *median* rather than a *max* over the complement sidesteps the harmonic leak that kills
the lag-axis complement (AUC 0.0000, handover §2), because a handful of comb harmonics cannot
move a median. That is a real idea and it is not exhausted; it is simply not a deployable
detector in this form.

## R27.73 ⛔ THE PROBE IS A NEGATIVE RESULT. `comb_priormax4_hmargin` remains the detector.

Both factors gated on both corpora. **Neither clears the pre-registered ship criterion**
(plan §3: *at least a tie on FakeMusicCaps, at least a tie on SONICS, 0/10 inverted, and
strictly better on recall or precision*).

| arm | FMC macro | SONICS macro | SONICS pooled vs `hmargin` | verdict |
|---|---|---|---|---|
| **`pm4_hmargin`** (incumbent) | **0.9445** | **0.8286** | — | **kept** |
| `pm4_widemargin` | **0.9487** | 0.8212 | **z = −14.70, p = 6.1e-49** | ⛔ **significantly WORSE on SONICS** |
| `pm4_lo20margin` | 0.9383 | 0.8289 | not tested (macro tie) | ⛔ worse on FMC |
| `pm4_lo20widemargin` | 0.9424 | 0.8212 | — | ⛔ worse on both |

**FakeMusicCaps, `widemargin`:** C11 PASS, AUC **0.9487 [0.9461, 0.9513]**, DeLong vs `hmargin`
**z = 14.43, p = 3.4e-47**, Δ +0.0042. 0/5 inverted, all CIs clear.
**SONICS, `widemargin`:** C11 PASS on its own CI (0.7831 [0.7794, 0.7870]) but the DeLong is
**negative and overwhelming**: Δ **−0.0094**, z = −14.70.

**This is exactly the lattice's failure pattern** (R27.36: FMC +0.0286 at p=2.8e-96, SONICS
−0.0255 at z=−16.29) and it is rejected for exactly the same reason. **Winning one corpus and
losing the other significantly is not a generalized detector.** Recorded before any temptation
to quote the FMC half alone.

**Where the SONICS loss comes from:** the chirp families are unchanged (recall @q95
0.9815/0.9836/0.9902 against `hmargin`'s 0.9809/0.9845/0.9886 — a wash). The whole gap is
**Udio**: per-family AUC 0.5666/0.5610 against `hmargin`'s 0.5862/0.5792. A wider null gives the
smoothness nuisance that suppresses Udio more room to find a large value, which is the same
mechanism R27.37 identified when the lattice's 74-lag null lost SONICS — **null size is not
monotone in the direction I assumed; there is an optimum and `M x 150` is near it.**

**The floor, finally.** FMC 0.9383 (worse), SONICS 0.8289 (a 0.0003 tie). Right mechanism —
SAO recall +0.027 on FMC and Udio +0.013 on SONICS — but it never once produces a net gain.
**R27.62's synthetic +0.09 is fully withdrawn as a corpus prediction.**

**So the programme ends where R27.42 left it.** Two well-motivated improvements, both
implemented, gated, and refuted. The detector is unchanged and now considerably better
supported: it has survived two serious attempts to beat it.

## R27.74 ⚠ CORRECTION to R27.72 — C1 is score-INDEPENDENT and I compared across corpora

R27.72 rejected `afmargin` partly because *"C1 is worse than the incumbent's — 0.6284 against
0.6082."* **That comparison is invalid.** C1 measures how well the best *channel descriptor
alone* separates real from fake; **it never sees the score.** The 0.6082 figure is **SONICS**
(R27.24); FakeMusicCaps is **0.6284**. I compared a number from one corpus to a number from the
other and read a defect into it.

Confirmed by this round: on FakeMusicCaps, `afmargin` and `widemargin` both report C1 = 0.6284
and C2 = 0.6207 — identical, as they must be. On SONICS, `widemargin` reports C1 = 0.6082,
C2 = 0.6202 — the values R27.45 recorded for `hmargin`. **C1/C2 are corpus properties and are
identical for every score gated against the same descriptor file.**

**The `afmargin` rejection is unaffected and is now overdetermined** — see R27.75.

## R27.75 ⭐ `afmargin` mechanism CONFIRMED, and it inverts Udio. Definitively dead.

The hypothesis in R27.72 was that `afmargin` is close to the *uncalibrated* prior strength,
because `comb_acf_floor` barely varies across tracks. Measured:

| M | corr(`afmargin`, raw prior) | corr(`widemargin`, raw prior) |
|---|---|---|
| 1 | **0.9620** | 0.5152 |
| 2 | **0.9625** | 0.5527 |
| 4 | **0.9621** | 0.5439 |

`sd(comb_acf_floor) = 0.0473` against `sd(afmargin) = 0.127` and `sd(widemargin) = 0.101`.
**Confirmed: subtracting the median removes almost no per-track scale.** `afmargin` is 96%
the uncalibrated score; a real margin sits at 0.52–0.55. That is the complete explanation for
"ranks correctly, does not separate" — FMC worst-family recall 0.4659 and four of five families
at exactly 0.0000 at q99.9.

**And SONICS kills it outright:** macro **0.6718 / 0.6754 / 0.6887** at M=4/2/1 with
**n_inverted = 2** — Udio at **0.2625 / 0.2483**, far below chance. A training-free score
inverted on any family is not usable zero-shot. **Closed.**

*(What remains worth remembering: `afmargin` is still the only construction reaching MusicGen
0.9706 on FMC. A **median** over the complement sidesteps the harmonic leak that makes the
**max** version score 0.0000. The idea survives; the score does not.)*

## R27.76 — M, confirmed on the second corpus

SONICS macro by prior depth, `widemargin`: M=1 **0.7775**, M=2 0.8121, M=4 **0.8212**. Same
ordering as FakeMusicCaps, and the M=1 damage is far worse here: recall @q95 for **chirp-v3.5
collapses to 0.3473** at M=1 against 0.9836 at M=4.

**M is settled on both corpora.** Required; M ∈ {2, 4} either works; **M = 1 is the one wrong
value** and it is catastrophic, not marginal.

## R27.77 — the two blocked follow-ups are now moot

`analyze_recon_pairs.py` and `measure_false_positive_rate.py` both refused `widemargin`: the
recon control (`comb_recon_fmc_HNULL`) and the FMA pass (`comb_fma_HARM2`) predate the probe
columns. **Since `widemargin` is rejected (R27.73), neither re-extraction is needed.** The
published C10 (R27.46, R27.51) and FMA transfer (R27.47) for `comb_priormax4_hmargin` stand as
the detector's numbers.

**THE MEASUREMENT PROGRAMME IS CLOSED.** Everything the detector needs has run:
gates both corpora, C10 both corpora, deployment, robustness, efficiency, prior depth, and two
refuted improvement attempts. What remains are reporting decisions, listed in
`RUNBOOK_2026-09-17_NULLS.md` PART 2b.

## R27.78 — the median window was never swept, and the fixture says 5 is too narrow

Asked directly whether a 3-bin median window would beat the published 5.
**`grep -n "smooth_bins" reports/provenance_ledger.md` returns ZERO hits across 4,058 lines** —
`--operator-grid` swept the residual operator, the average, `f_min` and `hull_clip_db`, but
never the window. It is an inherited default that has never been tested on either corpus.

Simulated at the real geometry, 3-bin teeth, 40 tracks per family:

| `smooth_bins` | width | MusicGen | StableAudio | AudioLDM2 | Suno | **mean** |
|---|---|---|---|---|---|---|
| 3 | 2.9 Hz | 0.9788 | 0.9925 | 0.8512 | 0.9469 | **0.9423** |
| **5 — published** | 4.9 Hz | 1.0000 | 1.0000 | 0.9550 | 0.9962 | **0.9878** |
| **7** | 6.8 Hz | 1.0000 | 1.0000 | **1.0000** | **1.0000** | **1.0000** |
| 9 / 13 / 21 | 8.8–20.5 Hz | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**So 3 is worse, and the answer has a clean mechanism.** A running median ignores the tooth only
if the tooth is a **minority of the window**. Measuring how much of a 10 dB, 3-bin tooth survives
into the residual:

| window | tooth height kept |
|---|---|
| 3 bins | 5.49 dB — **55%** |
| 5 bins | 5.51 dB — **55%** |
| 7 bins | 9.96 dB — **100%** |

At w = 3 and w = 5 the 3-bin tooth is a *majority* of the window, so the median lands on a tooth
bin and the envelope rides up onto the very peak it should be ignoring — **45% of the signal is
subtracted away.** At w = 7 the tooth is 3 of 7, a minority, the median lands on floor, and the
tooth survives whole. **The rule is `window > 2 × tooth width`, and the published 5 does not
satisfy it for a 3-bin tooth.**

⚠ **NOT a recommendation, and this is deliberate.** My synthetic-to-corpus transfer record this
week is **0 for 2** — the lowered floor predicted +0.09 AUC on Stable Audio and delivered a net
loss (R27.70); the exhaustive-prior collapse did not reproduce at all (R27.67). Real teeth are
not exactly 3 bins wide and the corpus has four families competing for one threshold, which is
precisely what broke the last two predictions. **Recorded as an open, untested lever with a
clean mechanism and a poor prior on transfer.** One extraction would settle it; the programme is
closed unless the user reopens it.

## R27.79 ⚠ CORRECTION to R27.71 — M = 1 does NOT invert for the detector's own null family

R27.71 said *"M = 1 is worse on every construction and inverts Stable Audio on three of them."*
The second half is true but **misleading, and I stated it as if it applied to the detector.**
Sorted by null family:

| construction | M=1 | M=2 | M=4 | family |
|---|---|---|---|---|
| `hmargin` | 0.9298 **(0 inv)** | 0.9436 (0) | 0.9445 (0) | **harmonic — the detector's** |
| `lo20margin` | 0.9302 **(0 inv)** | 0.9375 (0) | 0.9383 (0) | harmonic |
| `widemargin` | 0.9264 **(0 inv)** | 0.9474 (0) | 0.9487 (0) | harmonic |
| `lo20widemargin` | 0.9303 **(0 inv)** | 0.9421 (0) | 0.9424 (0) | harmonic |
| `latmargin` | 0.8560 (1 inv) | 0.8869 (0) | 0.9438 (0) | lattice — *already rejected* |
| `unionmargin` | 0.8560 (1 inv) | 0.8883 (0) | 0.9335 (0) | lattice |
| `pm1_latmargin` | 0.8204 (1 inv) | 0.8498 (0) | 0.9314 (0) | lattice |

**All three inversions are in the lattice/union family, which is not the detector and was
rejected on other grounds months ago. The harmonic family never inverts at any M.**

**What M = 1 actually costs the detector's family:**

| construction | FMC cost (M=1 − M=4) | SONICS cost |
|---|---|---|
| `hmargin` | −0.0147 | **−0.0289** |
| `lo20margin` | −0.0081 | **−0.0498** |
| `widemargin` | −0.0223 | −0.0437 |
| `lo20widemargin` | −0.0121 | **−0.0579** |

Plus the per-family collapse already recorded: SONICS chirp-v3.5 recall **0.3473** at M=1
against 0.9836 at M=4. **The rejection of M = 1 stands and is not close — but on cost, not on
inversion.** Corrected in the ledger, the reference guide and the talk.

## R27.80 — the combination the question asked about WAS tested, and the floor does rescue SAO at M=1

Asked whether M = 1 with the lowered floor *and* the wide null had been run. **It has:**
`comb_priormax1_lo20widemargin` — FMC **0.9303, 0 inverted**; SONICS **0.7633**. The
2 × 2 × 3 grid (floor × ceiling × depth) is complete; nothing was missed.

**And the floor does exactly what R27.62 predicted, at M = 1 where it should matter most:**
Stable Audio Open per-family AUC **0.8571 → 0.9084 (+0.0513)** with the floor lowered. The
mechanism is confirmed for the third time.

**It still loses, and the reason is worth recording.** At M=1 + lo20 against M=4 + no floor
change, every FMC family is worse — *including Stable Audio itself*:

| family | M=1 + lo20 | M=4 `hmargin` | Δ |
|---|---|---|---|
| MusicGen | 0.8257 | 0.8428 | −0.0171 |
| audioldm2 | 0.9678 | 0.9854 | −0.0176 |
| musicldm | 0.9539 | 0.9749 | −0.0210 |
| mustango | 0.9954 | 0.9963 | −0.0009 |
| **stable_audio_open** | **0.9084** | **0.9230** | **−0.0146** |

**Admitting SAO's own 21.53 Hz fundamental is worth less than keeping its second, third and
fourth harmonics.** The fundamental is the weakest tooth in the series; the prior's depth buys
more than the floor's reach gives back.

## R27.81 ⭐ The intuition "a real peak beats the null whatever the search length" — CONFIRMED, with the limit

Asked directly. Measured: MusicGen-like comb, 40 tracks per cell, null ceiling swept.

| tooth height | `max_P` @600Hz → @4kHz | `max_N` @600Hz → @4kHz | caught @600Hz | @4kHz |
|---|---|---|---|---|
| 4 dB | 0.0286 → **0.0286** | 0.0370 → 0.0399 | 12.5% | 32.5% |
| 6 dB | 0.0345 → **0.0345** | 0.0368 → 0.0396 | 32.5% | 57.5% |
| 9 dB | 0.0497 → **0.0497** | 0.0363 → 0.0390 | 87.5% | 92.5% |
| 14 dB | 0.0912 → **0.0912** | 0.0355 → 0.0379 | **100%** | **100%** |

**`max_P` is exactly flat across every ceiling — the prior peak genuinely does not care how
long the null search is.** The intuition is right. What moves is `max_N`, which rises ~+0.003
absolute (about 8–11%) from 441 to 3,130 lags and saturates by ~1500 Hz.

**And at 14 dB the catch rate is 100% at every ceiling** — a strong comb is untouchable, exactly
as the intuition says. The cost lands entirely on **faint** combs, where a ±0.003 shift in the
baseline moves tracks across the threshold. Those are precisely MusicGen, Stable Audio and Udio.

⚠ **On this fixture a wider null HELPS the faint cases** (12.5% → 32.5%, 32.5% → 57.5%), because
the threshold is recomputed with the same wider null and the real class loses more margin than
the fake class. **The corpus disagreed on SONICS.** See R27.82.

## R27.82 ⛔ My proposed mechanism for the SONICS loss is REFUTED by my own fixture

R27.73 attributed the wide null's SONICS loss to Udio, correctly (per-family AUC
0.5862 → 0.5666 while the three chirp families were a wash). I then proposed a mechanism:
*Udio's residual is unusually smooth (R7), a smoother residual has a broader autocorrelation,
so its null maximum grows faster when the null widens.* Tested:

| residual character | `max_N` @600Hz | @4kHz | rise |
|---|---|---|---|
| ordinary | 0.0371 | 0.0413 | **+0.0041 (11.2%)** |
| mildly smooth | 0.0947 | 0.0947 | **+0.0000** |
| smooth | 0.4437 | 0.4437 | **+0.0000** |
| very smooth | 0.5530 | 0.5530 | **+0.0000** |

**Exactly backwards.** A smooth residual's null max is already saturated at short lags, so
widening adds nothing; only the *ordinary* residual rises. **The mechanism is withdrawn before
it was published anywhere.**

**Standing position: the Udio loss under a wider null is a measured corpus fact with no verified
mechanism.** That is the honest state. This is the third synthetic mechanism story to fail this
month (R27.67, R27.70, and this one), and the pattern is now itself worth reporting: *this
fixture reproduces the detector's arithmetic faithfully and its failure modes badly.*

## R27.83 ⭐ The fully M-free construction, simulated: the NULL can be exhaustive, the PRIOR cannot

Asked whether M can be dropped entirely by making **both** lists exhaustive — prior = *all*
`k·Δ` to the band edge, null = everything else. That is not `widemargin` (which keeps the prior
at `m ≤ M` and only widens the null), and it had not been tested. Simulated at the real geometry,
40 tracks per family:

| arm | \|P\| | \|N\| | mean AUC, 12 dB teeth | mean AUC, **6 dB teeth** |
|---|---|---|---|---|
| **published** — prior m≤4, null to 600 Hz | **18** | 441 | 0.9878 | **0.7675** |
| wide null — prior m≤4, null to 4 kHz | **18** | 3,130 | **0.9925** | **0.8056** |
| M=1 — prior m≤1, null to 4 kHz | 5 | 3,130 | 0.9817 | — |
| FREE — prior ALL k, both to 1 kHz | 80 | 751 | 0.9742 | — |
| FREE — prior ALL k, both to 2 kHz | 162 | 1,550 | 0.9702 | — |
| **FREE — prior ALL k, both to 4 kHz** | **331** | 3,130 | **0.9675** | **0.6564** |
| FREE — prior ALL k, both to 7.8 kHz | 663 | 6,128 | 0.9675 | — |

**The two halves of the proposal separate cleanly and point opposite ways.**

* **Widening the NULL is fine** — better on this fixture at both comb strengths
  (0.9878 → 0.9925 strong, 0.7675 → **0.8056** faint). That is the arm already gated and
  rejected on SONICS (R27.73), not on any fixture objection.
* **Widening the PRIOR is not** — the prior balloons from 18 to **331** lags and the faint-comb
  AUC collapses **0.8056 → 0.6564, a loss of 0.149.** Monotone in prior size: 80 lags 0.9742,
  162 lags 0.9702, 331 lags 0.9675, and saturating once the band is exhausted.

**The mechanism is the extreme-value penalty and it is quantifiable.**
`√(ln 18 / ln 4000) = 0.59` against `√(ln 331 / ln 4000) = 0.84` — the chance-level peak rises
**42%** when the prior goes exhaustive. Every one of 331 lags gets an independent chance to win
by noise; the maximum of 331 noisy values is much larger than the maximum of 18.

**The clean statement this yields, and it is the best framing of the method found so far:**

> **The null keeps every harmonic; the prior keeps very few. Those are opposite jobs and both
> are necessary.** Harmonic protection exists so the null cannot contain the comb's own teeth —
> that is why it must be exhaustive. The prior's restriction exists so noise rarely wins — that
> is why it must be small. Making them symmetric destroys the second.

So **M is not a parameter of the null** (the null already keeps all harmonics, unbounded in k).
**M is the statement of how many harmonics count as *evidence*.** It cannot be removed because
"no restriction" is a setting, and it is a bad one.

⚠ Fixture result. Synthetic-to-corpus transfer has failed three times this month (R27.67,
R27.70, R27.82). But note the direction: this one *agrees* with the corpus M-sweep (R27.71,
R27.76), where a deeper prior was worse on both benchmarks at every setting tested.

## R27.84 — clarification: the M=1 arms never had a 4 kHz null

`widemargin`'s ceiling is `n_harm × hi_hz` with `hi_hz = 1000`, so it scales with M:

| M | `hmargin` ceiling | `widemargin` ceiling |
|---|---|---|
| 1 | 150 Hz | **1000 Hz** |
| 2 | 300 Hz | 2000 Hz |
| 4 | 600 Hz | **4000 Hz** |

So `comb_priormax1_widemargin` (SONICS chirp-v3.5 recall **0.3473**) used a **1 kHz** null, not
4 kHz. **The M=1 + 4 kHz combination is nevertheless tested** — that is exactly what the
decoupled `x` arms are (R27.68): `comb_priormax1_xwidemargin` crosses the M=1 prior against
M=4's 4 kHz null. **SONICS 0.7761**, against M=4 `widemargin`'s 0.8212 — still 0.045 worse.
Nothing in the grid is missing.

## R27.85 ⚠ DOCUMENT DEFECT — SONICS numbers were reported unlabelled as macro or pooled

Caught in review: the talk quotes **SONICS 0.8286** on one slide and **SONICS pooled 0.7925** on
another, without saying which is which. Both are correct and they are different statistics:

| corpus | macro (mean of 5 per-family AUCs) | pooled (all fakes vs all reals) |
|---|---|---|
| FakeMusicCaps | 0.9445 | 0.9445 — *coincidentally equal to 4 dp* |
| **SONICS** | **0.8286** | **0.7925** |

The gap is composition: Udio is roughly half of SONICS by track count but only two of five
families, so pooling weights it far more heavily than the macro does. **FakeMusicCaps hides the
distinction because the two agree there**, which is precisely how the inconsistency survived
review. Every SONICS figure in both documents is now explicitly labelled. **The DeLong test is
defined on the pooled quantity**, so `p = 0.106` belongs next to 0.7925 and never next to 0.8286.

## R27.86 — why the residual operator is a median and not Afchar's hull

Asked directly; the reasoning is in `_residual_from`'s docstring and the answer is that the
operator is coupled to the *readout*.

Afchar's `lower_hull` + `max_normalise` clips the residual at 5 dB and divides by its maximum.
**That is right for their readout** — a logistic regression over a 445-dimensional profile only
needs the *pattern* of which bins peak, so making the descriptor invariant to peak loudness
helps it. **It is wrong for ours.** `comb_strength` is an autocorrelation peak and needs the
residual's amplitude structure; once teeth exceed 5 dB they all saturate to 1.0, the residual
becomes a near-binary mask, and noise bins that also exceed 5 dB inflate the ACF floor.
Measured, separation on synthetic 3-bin teeth:

| tooth height | hull + clip 5 dB + max-norm | hull, no clip | **median** |
|---|---|---|---|
| 4 dB | 0.0245 | 0.0453 | **0.0594** |
| 8 dB | 0.0483 | 0.2255 | **0.2471** |
| 20 dB | 0.0493 | 0.6229 | **0.6639** |
| 40 dB | **0.0493 — saturated** | 0.6229 | **0.6639** |

The clipped hull saturates at 0.049 and stops responding to tooth height entirely. Unclipped it
recovers, and the median is slightly better again — a lower envelope leaves a noise pedestal
that a median does not.

**It was then settled on real audio, not by argument.** `eval_comb_detector.py --operator-grid`
swept residual operator × average × `f_min` × `hull_clip_db` in one pass over identical audio
(`comb_fmc_HARM2`, 400 columns), and the published column is `median_db__comb_strength` at
**0.9177** (ledger row 31). ⚠ **The per-operator grid table itself is not transcribed into the
ledger** — only the winner is. That is a provenance gap: the *decision* is recorded, the
*comparison* is not.

## R27.87 ⛔ BOTH explanations for the M penalty are REFUTED. It is purely extreme value.

Two mechanisms were on the table for why a deeper prior costs accuracy:

**(a) Tooth amplitude fades with frequency, so deep candidates carry no signal.** — proposed
in review. **Refuted by our own measured data.** Ledger G2/O8b records the *loudest* tooth per
family: audioldm2/musicldm/mustango **k=2**, MusicGen **k=5**, Stable Audio Open **k=5**, Suno
chirp **k=8**. **Never k=1.** Tooth amplitude does not fade monotonically from the fundamental —
it rises to a filter-determined peak and falls after. If it faded from k=1, M=1 would be the best
setting; it is the worst.

**(b) The ACF peak itself decays with lag, so deep candidates have lower SNR.** — my restatement,
and **also refuted.** Measured on a synthetic comb with *exactly equal* teeth at every harmonic
(so (a) is switched off by construction), 15 harmonics:

| m | exact lag | rounded | \|rounding error\| | ρ |
|---|---|---|---|---|
| 1 | 102.40 | 102 | 0.40 | 0.4344 |
| 2 | 204.80 | 205 | 0.20 | 0.5728 |
| 4 | 409.60 | 410 | 0.40 | 0.4177 |
| **5** | **512.00** | **512** | **0.00** | **0.6802** |
| 9 | 921.60 | 922 | 0.40 | 0.3755 |
| **10** | **1024.00** | **1024** | **0.00** | **0.6196** |
| **15** | **1536.00** | **1536** | **0.00** | **0.5752** |

**corr(|rounding error|, ρ) = −0.931. corr(m, ρ) = −0.173.** Mean ρ is **0.625** where the
error is < 0.15 and **0.395** where it exceeds 0.30. **There is no decay with m at all** — the
apparent decline in a short listing is the rounding pattern aliasing against it.

**So the M penalty has no signal-side story. It is the extreme-value effect alone:** more
candidates, higher chance maximum, `√(ln|P|)`. That is a *cleaner* result than either
mechanism — the restriction buys exactly what §9 says it buys and nothing else.

**And there is a genuine incidental finding.** The prior's candidates have systematically
unequal quality depending on whether `m·Δ/b` lands near an integer — the same ±1-bin straddling
that forces the null's tolerance (R27.62, §13), showing up in the **prior**. A ±1-tolerant prior
exists in the code (`comb_prior_max(tol_bins=1)`, emitted as `pm1_strength`) but was never
combined with the harmonic null.

⚠ **Not proposed as a lever.** `pm1` *was* tested against the lattice null and made it **worse**:
`pm1_latmargin` 0.9314 against `latmargin` 0.9438 at M=4 on FMC. Widening the prior's tolerance
also widens what noise can win, which is the same trade as widening the prior's depth. Recorded
as an observation with contrary evidence already on file.

## R27.88 — sensitivity summary: every remaining knob is small except removing an architecture

| knob | FMC macro Δ | SONICS macro Δ |
|---|---|---|
| lag floor 40 → 20 Hz | **−0.0062** | +0.0003 |
| null ceiling 600 Hz → 4 kHz | +0.0042 | **−0.0074** |
| prior depth M = 4 → 2 | −0.0009 | −0.0071 |
| **prior depth M = 4 → 1** | **−0.0147** | **−0.0289** |
| prior exhaustive (331 lags) | *fixture only:* −0.021 strong / **−0.149 faint** | — |

**Every knob moves macro AUC by ≤ 0.007 except the one that removes an architecture from the
prior.** That is a strong robustness statement and it belongs in the paper: the detector is
insensitive to all its remaining design choices, and the single choice that matters is forced by
physics (M ≥ 2 so Stable Audio's 21.53 Hz series is representable above the lag floor).

The corollary is the honest reading of the floor question: **the 40 Hz floor genuinely almost
does not matter** (−0.006 / +0.000). It is kept because lowering it measured slightly worse, not
because it is load-bearing.

## R27.89 — deployment: the FMA false-alarm rate and what precision is reachable

**Transfer to the unseen catalogue.** The threshold is set on the benchmark's reals and applied
unchanged to FMA, which contains no fakes, so every firing is a false alarm. Measured ratio
**0.72× intended**, i.e. it lands *under* budget:

| threshold | intended FPR | measured on FMA | catch rate |
|---|---|---|---|
| q90 | 9.94% | **7.16%** | 0.9025 |
| **q95** | 4.11% | **2.96%** | **0.8702** |
| q99 | 0.71% | **0.51%** | 0.5274 |
| q99.9 | 0.11% | **0.08%** | 0.1538 |

**Precision at prevalence** (`precision = p·TPR / (p·TPR + (1−p)·FPR)`):

| threshold | p=0.1% | p=1% | p=5% | p=10% | p=30% |
|---|---|---|---|---|---|
| q90 | 0.009 | 0.084 | 0.323 | 0.502 | 0.796 |
| q95 | 0.021 | 0.176 | 0.527 | 0.702 | 0.901 |
| q99 | 0.069 | 0.429 | 0.796 | 0.892 | 0.970 |
| **q99.9** | 0.121 | **0.581** | 0.878 | 0.938 | 0.983 |

**Inverted — the prevalence required to reach a target precision:**

| target precision | q90 | q95 | q99 | **q99.9** |
|---|---|---|---|---|
| 50% | 9.9% | 4.5% | 1.3% | **0.7%** |
| 80% | 30.6% | 15.9% | 5.1% | **2.8%** |
| **90%** | 49.8% | 29.8% | 10.8% | **6.2%** |
| 95% | 67.7% | 47.3% | 20.4% | **12.2%** |

**The answer to "what do we need for sufficient precision":** at a realistic **1% prevalence no
operating point reaches 90% precision** — the best is **0.581** at q99.9, catching 15% of fakes.
To reach 90% you need either **~6% of the stream to be fake** (q99.9) or **~30%** (q95).

**That is a base-rate property, not a detector deficiency**, and the paper should say so: at
FPR 0.11% with 99% of the stream real you raise ~1.1 false flags per 1,000 clean tracks and find
~1.5 genuine fakes per 1,000. **The correct framing is a triage filter feeding review, not an
adjudicator.** Three defensible regimes: q95 for triage (catch 87%, flag ~3% of a clean
catalogue), q99 balanced (catch 53%, 0.51%), q99.9 high-confidence (catch 15%, 0.08%).

## R27.90 — the misses are explained by REACHABILITY, not by any decay story

With the decay explanations refuted (R27.87), what remains to explain MusicGen's 0.6978 and
Stable Audio's 0.8013? Cross the measured visible tooth (G2/O8b) against the prior's reach:

| family | C8 concentration | AUC | recall @q95 | visible tooth | in the prior at M=4? |
|---|---|---|---|---|---|
| mustango | 0.6774 | 0.9963 | **0.9995** | 200.2 Hz | **yes** — 2 × 100 |
| audioldm2 | 0.6612 | 0.9854 | **0.9752** | 200.2 Hz | **yes** |
| musicldm | 0.3997 | 0.9749 | **0.9272** | 200.2 Hz | **yes** |
| MusicGen | 0.4016 | 0.8428 | **0.6978** | 250.0 Hz | **no** — 5 × 50 |
| Stable Audio Open | **0.0143** | 0.9230 | **0.8013** | 107.4 Hz | **no** — 5 × 21.53 |

**The split is reachability.** The three families whose visible tooth is 200.2 Hz — exactly
`2 × 100`, squarely inside the prior — average **0.967** recall. The two whose visible tooth
is outside it average **0.750**. `corr(concentration, recall) = +0.642`, weaker and driven
largely by the same grouping.

**This is not a re-run of the M argument.** Deepening the prior to reach 250 Hz was tested and
is worse (R27.71, R27.76, R27.83): the extra candidates cost more in chance maxima than the
reachable tooth gains. **The correct statement is that these two families are detected through
harmonics that are *not* their loudest**, so their margins are smaller — and R27.81 quantifies
what a smaller margin costs (catch rate 12.5% at 4 dB teeth, 100% at 14 dB).

**Stable Audio is the instructive anomaly and worth a sentence in the paper.** Its C8
concentration is **0.0143** — its tracks essentially never agree on which spacing won — yet 80%
are caught. **The margin can be positive without the argmax being stable.** The detector is
asserting *"there is more periodicity at machine spacings than at non-machine ones"* without
committing to which machine spacing, which is precisely what the score is defined to say and no
more. It is a detector, not an attributor — and C8 measuring 0.0143 for a family we catch at 80%
is the cleanest demonstration of that distinction in the record.

## R27.91 — the decoder chain, worked: where Δ = rate / ∏strides actually comes from

EnCodec 32 kHz (MusicGen's decoder), strides (8, 5, 4, 4), product 640:

| layer | stride | input rate | output rate | 1 s of audio |
|---|---|---|---|---|
| — | — | — | 50 Hz | 50 latent vectors |
| 1 | 8 | 50 | 400 | 50 → 400 |
| 2 | 5 | 400 | 2000 | 400 → 2000 |
| 3 | 4 | 2000 | 8000 | 2000 → 8000 |
| 4 | 4 | 8000 | 32000 | 8000 → **32,000 samples** |

**Each layer zero-stuffs at its own input rate, so each leaves a comb at that rate:** layer 1 at
**50 Hz**, layer 2 at 400 Hz, layer 3 at 2 kHz, layer 4 at 8 kHz. **The finest is the first
layer's, at `rate / ∏strides` — and that is where Δ comes from.** The coarser combs from later
layers exist too but are far sparser, so an autocorrelation over the 1–8 kHz band sees far fewer
of their teeth.

This is a better statement of the mechanism than "the decoder leaves a comb at
`rate / ∏strides`": it says *which* layer leaves it and why the product appears.

## R27.92 — worked examples added for three things previously only asserted

**(a) Why the mean must be removed.** Same six numbers, centred and offset by 10:

| | ρ(1) | ρ(2) | ρ(3) | argmax |
|---|---|---|---|---|
| `[+2,−1,−1,+2,−1,−1]` | −0.3333 | −0.4167 | **+0.5000** | **lag 3 — correct** |
| `[12, 9, 9, 12, 9, 9]` | **+0.7941** | +0.6618 | +0.5000 | **lag 1 — wrong** |

Identical shape, tooth spacing 3 in both. The offset version picks **lag 1**, where the signal
anti-correlates. With mean μ every product is `μ² + μ(a+b) + ab`; the `μ²` term is identical at
every lag and there are ~n of them, so it is a plateau of height ~`μ²n`. Here `μ² = 100`, so
~600 of the 612 total is plateau and ~12 is the pattern — a 2% ripple on a 98% pedestal, and the
pedestal is *taller at short lags* because more terms overlap. **The argmax slides to the
smallest available lag.** One line prevents it.

**(b) The extreme-value effect**, 20,000 trials of `max` over N standard normals:

| N | E[max] | √(2 ln N) | **P(max > 3σ)** |
|---|---|---|---|
| 10 | 1.542 | 2.146 | 1.4% |
| **18** | 1.821 | 2.404 | **2.4%** |
| 441 | 2.999 | 3.490 | 44.7% |
| **4000** | 3.618 | 4.073 | **99.6%** |

With 18 candidates a 3σ value appears 2.4% of the time; with 4,000 it appears **almost always**,
on pure noise. Chance peak 4.07σ → 2.40σ, ratio **0.590**, matching `√(ln18/ln4000)` exactly.

**(c) How many FMA tracks we falsely flag.** The unseen pool is **n = 2,998**, all real
(ledger row 984), so every flag is an error:

| threshold | FPR on FMA | tracks wrongly flagged | correctly left alone |
|---|---|---|---|
| q90 | 7.16% | 215 | 2,783 |
| **q95** | **2.96%** | **89** | **2,909** |
| q99 | 0.51% | 15 | 2,983 |
| q99.9 | 0.08% | 2 | 2,996 |

## R27.93 — per-generator AUC for the detector, both corpora

Recall was tabulated everywhere; the per-family **AUC** was not. From
`analytic_null_auc.csv`, `comb_priormax4_hmargin`, all 0/5 inverted on each corpus:

| FakeMusicCaps | AUC | recall @q95 | | SONICS | AUC | recall @q95 |
|---|---|---|---|---|---|---|
| mustango | **0.9963** | 0.9995 | | chirp-v3 | **0.9943** | 0.9886 |
| audioldm2 | **0.9854** | 0.9752 | | chirp-v3.5 | **0.9925** | 0.9845 |
| musicldm | **0.9749** | 0.9272 | | chirp-v2-xxl-alpha | **0.9906** | 0.9809 |
| stable_audio_open | **0.9230** | 0.8013 | | udio-120s | **0.5862** | 0.0190 |
| MusicGen_medium | **0.8428** | 0.6978 | | udio-30s | **0.5792** | 0.0171 |
| **macro** | **0.9445** | 0.8760 | | **macro** | **0.8286** | 0.4824 |
| **pooled** | **0.9445** [0.9416, 0.9474] | | | **pooled** | **0.7925** [0.7887, 0.7963] | |

**AUC and recall order the families differently, and the gap is the point.** Stable Audio has
a *higher* AUC than MusicGen (0.9230 vs 0.8428) but both sit far below the three easy families
on recall. And on SONICS the three chirp families are at **0.99 AUC** — the corpus-level 0.7925
is almost entirely Udio's two families dragging a pooled average.

⚠ **Per-family confidence intervals are transcribed for `widemargin` but not for `hmargin`** —
R27.45 recorded only "0/5 inverted by interval". `widemargin`'s intervals are a fair proxy for
width (FMC: MusicGen [0.8450, 0.8601], SAO [0.9234, 0.9339]; SONICS: chirp [0.9873, 0.9961],
udio-120s [0.5597, 0.5733]) since the two scores differ by 0.004, but they are **not** the
detector's own numbers and must not be quoted as such.

## R27.94 ⭐ The real-world prevalence is NOT 1%, and that changes the deployment verdict

Every precision table in this project used 0.1%–10% prevalence as the plausible range. **Fetched
and verified**: Deezer's newsroom, June 2026 —

> *"more than 50% of total uploads of new music"* are AI-generated, at a peak of
> **90,000 tracks per day**; AI-generated music accounts for *"between 1-3% of total streams"*;
> *"up to 85% of the streams generated by fully AI-generated tracks were in fact fraudulent in
> 2025"*.

Trajectory on the same platform: Sept 2025 **28%** (30k/day) → Nov 2025 34% (50k/day) →
Jan 2026 39% (60k/day) → Apr 2026 44% (75k/day) → **Jun 2026 >50%** (90k/day).

**So the prevalence is not one number — it depends entirely on where you measure, and the two
realistic answers differ by a factor of ~20.**

| where you measure | p | q90 | q95 | q99 | q99.9 |
|---|---|---|---|---|---|
| **streams / consumption** | **2%** | 0.156 | 0.302 | 0.603 | 0.737 |
| a back-catalogue *(interpolation, not measured)* | 5% | 0.323 | 0.527 | 0.796 | 0.878 |
| 2025 upload stream | 28% | 0.779 | **0.892** | **0.967** | 0.982 |
| **current upload stream** | **52%** | **0.908** | **0.958** | **0.988** | 0.993 |

**This flips the deployment conclusion recorded in R27.89.** That entry said *"at a realistic 1%
prevalence no operating point reaches 90% precision"*, which is true **and was answering the
wrong question**: 1% is roughly the *consumption* figure, and nobody screens consumption. **A
rights platform screens the upload stream, where the base rate is now above 50%.**

Worked, one day of that stream at q95 using the **FMA-measured** FPR (2.96%, i.e. on unseen
music, not the in-corpus 4.11%):

> ~170,000 uploads/day, 52% AI → 88,400 AI and 81,600 human.
> Flagged: **76,926 true + 2,415 false**. **Precision 0.970, recall 0.870**, 11,474 AI missed.

⚠ Note the two FPR figures: the table above uses the in-corpus 4.11% (giving 0.958) and the
worked example the FMA-measured 2.96% (giving 0.970). **Deployment claims should use the FMA
number** — it is the one measured on music the detector has never seen.

**What this does and does not license.** It licenses: *"at the base rate a rights platform
actually faces, this detector reaches 0.96 precision at 87% recall with no trainable
parameters."* It does not license dropping the low-prevalence caveat — the 1–3% consumption
figure is real, and any use that screens *streams* rather than *uploads* is back in the regime
where 0.58 is the ceiling. Both belong in the paper.

Source fetched 2026-09-21: https://newsroom-deezer.com/2026/07/ai-music-exceeds-50-percent-daily-uploads-deezer/
**Not yet added to `refs.bib`** — a press release is a weak citation for a paper; the same
figures appear in trade coverage and the underlying Deezer detector work may be a better anchor.
Flagged for the camera-ready decision, not added unilaterally.

## R27.95 ⚠ CORRECTION — time-stretch does NOT move the comb spacing. Only pitch-shift does.

R27.58 and both explainer documents say the time-stretch failure is because *"resampling moves
Δ off the prior"*, grouping it with pitch-shift. **Measured directly** — a synthetic 100 Hz comb
through `librosa.effects.*`, spacing recovered by the detector's own estimator:

| | recovered comb spacing |
|---|---|
| original | **200.20 Hz** |
| `time_stretch(rate=1.2)` | **200.20 Hz — unchanged** |
| `pitch_shift(+2 semitones)` | **112.30 Hz** (predicted 100 × 2^(2/12) = 112.25) |

**`librosa.effects.time_stretch` is a phase vocoder** — STFT, interpolate frames, ISTFT. It
changes *duration* and preserves *frequency*, so Δ does not move. **`pitch_shift` is
`time_stretch` followed by a resample**, and it is the resample that shifts the whole frequency
axis by the pitch ratio.

**So the two failures have different mechanisms, and the AUCs already said so:**

| manipulation | AUC | mechanism |
|---|---|---|
| time_stretch | **0.5509** — above chance | phase-vocoder **smearing** blurs the fine spectral structure; Δ stays put, the teeth get fuzzy |
| pitch_shift | **0.4665** — **below** chance | the resample **moves Δ** off all six prior frequencies, so fakes lose their comb while real music is unaffected |

**That is why only pitch-shift inverts.** Smearing degrades both classes toward each other;
moving Δ removes the signal from the fakes *specifically*, which pushes them below the reals.
The sign of the AUC distinguishes the two mechanisms, and I had merged them.

Corrected in the ledger, the reference guide and the talk. **The paper's limitations sentence
should name pitch-shift alone as the resampling failure**, and describe time-stretch as
smearing.

## R27.96 — how per-generator AUC and recall are computed, for the record

**AUC per generator** (`derive_prior_margin.py::_auc_table`):

```
is_real = (df["label"] == "real")
for g in generators:
    sel = is_real | (df["algorithm"] == g)   # g's fakes + ALL reals, shared pool
    y   = (~is_real)[sel]                    # 1 = fake, 0 = real
    per[g] = roc_auc_score(y, score[sel])    # sklearn, via auc_and_eer
MACRO      = mean(per.values())
n_inverted = sum(v < 0.5 for v in per.values())
```

The real pool is **shared across generators**, so each `AUC_g` asks "can this family be
separated from real music". No threshold is involved. `n_inverted` is the admissibility gate.

**Recall per generator** (`analyze_detector.py:341-355`):

```
thr_reals, eval_reals = real_idx[:n//2], real_idx[n//2:]   # a half-split of the reals
thr95 = np.quantile(score[thr_reals], 0.95)                # threshold from REALS only
df["_flagged"] = score > thr95
by_gen = df[y==1].groupby("algorithm")["_flagged"].agg(["mean", "size"])
```

**The half-split matters and is easy to miss:** the threshold is set on the *first* half of the
reals and the false-positive rate measured on the *second*, so the reported FPR is out-of-sample
with respect to the threshold. Recall is then simply the mean of a boolean per family.

## R27.97 — the robustness manipulations, exactly as implemented

`scripts/apply_audio_manipulations.py`, applied by `run_robustness_battery.py`. Parameters are
drawn **per track** from a CRC32-seeded RNG (`randomized_manipulation`) so the protocol matches
MusicDET's stated randomisation rather than a single fixed point.

| manipulation | implementation | parameter | what it does to a comb |
|---|---|---|---|
| **pitch_shift** | `librosa.effects.pitch_shift` | `n_steps ~ U(−2, +2)` semitones | resamples → **moves Δ** by 2^(n/12) |
| **time_stretch** | `librosa.effects.time_stretch` | `rate ~ U(0.8, 1.2)` | phase vocoder → **smears**, Δ unchanged |
| **equalization** | two Butterworth-2 shelves, `filtfilt` | −4 dB below 200 Hz, +4 dB above 4 kHz, fixed | changes tooth **heights**, not spacings |
| **reverberation** | convolve with exponentially-decaying white-noise IR | decay 0.6 s, 35% wet, fixed | multiplies the spectrum by a dense random envelope → smears |
| **white_noise** | Gaussian at fixed SNR, realisation drawn per track | 20 dB SNR | raises the floor → **buries faint teeth** |
| **aac_64k** | `ffmpeg` round-trip, back to original sr/mono | 64 kbps | lossy re-encode |
| **opus_64k** | same | 64 kbps | lossy re-encode |

⚠ **Two caveats already in the script's own header and worth repeating in any write-up.**
(1) **The codec rows are a CASCADE, not a clean transcode.** Manipulations are applied to
already-canonicalised MP3-64k audio, so `aac_64k` measures MP3→AAC and `opus_64k` MP3→Opus —
unlike MusicDET, whose model is clean-trained. (2) `equalization` and `reverberation` are
**deterministic** because MusicDET's paper specifies no parameters for them; our settings are
our own choice and are documented in the functions rather than matched to theirs.
