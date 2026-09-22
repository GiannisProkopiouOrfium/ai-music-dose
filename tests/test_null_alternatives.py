"""Per-track nulls that need no decoys: what survives a seed sweep, and what dies.

Why this file exists
--------------------
``reports/handover/HANDOVER_2026-09-16_NULLS.md`` proposes six replacements for the
24 decoy prior sets (its section 3) and reports a single-draw fixture for the
professor's "mask the expected fundamentals and take the max of the rest" idea (its
section 2). A single draw is not evidence, and three of that handover's conclusions
do not survive a sweep. This file pins what does.

Every result below was measured at 200 seeds; the asserts run at ``N_SEEDS`` and are
loose ORDERINGS rather than exact values, so the file is a guard against a claim
being re-adopted, not a brittle regression on a float.

What is pinned, and the claim each assert protects
--------------------------------------------------
1. **The naive complement null inverts MusicGen.** AUC 0.0000 over 200 seeds, with
   its complement peak at lag 255 = 249.0 Hz. A decoder emits peaks at *every*
   multiple of the spacing; ``P`` at M=2 holds only the first two, so everything from
   3*Delta up sits in the complement carrying the signal being measured. The detector
   subtracts its own evidence from itself.

2. **Cross-band ``min`` destroys a comb visible only in the upper band.** The raw
   prior score reads 1.0000 on that condition and cross-band ``min`` reads **0.4751**
   -- chance. Stated precisely: across sweep sizes it straddles 0.5 (0.5513 at 40
   seeds, 0.5005 at 80, 0.5175 at 120, 0.4751 at 200), so the reliable claim is that
   the family becomes UNDETECTABLE, not that it reliably inverts. Either way the
   handover ranks cross-band consistency as its highest-novelty candidate and it is
   the most dangerous one. The time-axis analogue works because a melody moves IN
   TIME. A harmonic series does not move IN FREQUENCY, so cross-band agreement cannot
   separate a musical comb from a decoder comb: it keeps the weakness and discards
   the strength. ``mean`` is a no-op, because the mean of the two half-band
   autocorrelations approximates the full-band one.

3. **An anti-phase null inverts MusicGen**, and the reason is arithmetic rather than
   statistical. Scoring ``rho(tau_hat)`` against ``rho`` at half-integer multiples of
   the winning fundamental looks perfectly lag-matched and knob-free, but the six
   fundamentals are related by small rational ratios: ``1.5 * 100 = 150 = 3 * 50``.
   A half-multiple of one fundamental is an integer multiple of another, so the
   anti-lag set lands ON the comb. There is no obvious repair.

4. **A RATIO bulk calibration sits at or below chance on a Udio-like family where
   the DIFFERENCE form is clearly above it.** At 200 seeds: sharpness (peak / median)
   **0.4483** and its z-score cousin 0.4607, against peak - median at **0.6631**. The
   absolute level of the ratio form moves with the sweep size (0.4219 / 0.5009 /
   0.4728 / 0.4483 at 40 / 80 / 120 / 200 seeds), so what is pinned is the ORDERING
   and the roughly 0.2 AUC gap, which is stable at every size -- not the word
   "inverts". A smoother residual raises the autocorrelation bulk, so dividing by it
   deflates that family more than it deflates real music; subtracting is the right
   operation when the nuisance is additive in autocorrelation units. This matters
   because ``comb_harm4_sharpness`` -- the handover's section 3 N1, its top-ranked
   candidate -- is a ratio, and was never taken to SONICS.

5. **The handover's "lattice drift" diagnosis is wrong, and it is its own fixture's
   artefact.** Its section 2.3 blames the analysis grid: "the prior says 100.00 Hz =
   102.4 bins, but the comb sits at whatever integer lattice position the decoder
   actually produces". ``frac(f/bin_hz)`` is a property of OUR grid, not of the
   prior's accuracy -- if the prior frequency is right, ``round(m*f/bin)`` lands on
   the autocorrelation peak at every m. The real quantity is
   ``m * |f_prior - Delta_true| / bin_hz``: prior ERROR accumulating linearly in
   harmonic order. Measured on the same statistic under two tooth models, the
   all-harmonics +-1-bin mask reads **0.0000** when teeth sit at integer bin
   multiples (which builds in a 0.4% prior error) and **0.9974** when they sit at
   exact multiples of Delta in Hz. The consequence for section 3's N4: its prescribed
   tolerance ``m * frac(f/bin_hz)`` computes the wrong quantity, and a defensible
   tolerance is RELATIVE (``+-delta*m`` bins for a stated prior precision delta).

6. **Prior depth, as pure arithmetic** -- see
   :class:`TestPriorDepthReachesTheMeasuredCombs`. No RNG, so it cannot drift.

RNG convention -- a deliberate departure from this file's siblings
------------------------------------------------------------------
``tests/test_comb_harmonic.py`` and ``tests/test_stationarity.py`` use the legacy
``np.random.RandomState``. This file uses ``np.random.default_rng`` because the
handover's section 7 step 2 specifies it, and reproducing that fixture bit-for-bit is
the point of the exercise.

The fixture's own limitation, stated so it is not over-read
------------------------------------------------------------
The real and generated conditions here share a residual process, so there is no
corpus-scale smoothness nuisance to cancel and the fixture CANNOT reproduce the Udio
inversion in absolute terms. It separates functional FORMS, which is what item 4
uses it for. That limitation applies equally to the handover's section 2 numbers.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from intrinsic_ai_music_detection.features.comb_artifacts import (
    DECODER_FUNDAMENTALS_HZ,
    _normalised_acf,
    decoy_fundamentals,
)

# The real FakeMusicCaps analysis geometry: 16 kHz, n_fft 16384, band [1 kHz, 8 kHz].
# Ledger R10 fixes these; handover section 7 step 2 repeats them.
BIN = 16_000 / 16_384
F = 7_167
F_START = 1_000.0
LO = int(round(40.0 / BIN))          # the 40 Hz floor comb_strength uses
HI = min(int(round(4_000.0 / BIN)), F - 1)
HALF = F // 2
M_SHIPPED = 2                        # the shipped comb_priormax2_margin

# The numbers in the docstring are at 200 seeds and the asserts run there too: the
# two quantities that straddle chance (items 2 and 4) move by up to 0.08 between 40
# and 200 seeds, which is larger than some of the gaps being pinned. Both sweeps
# together cost about four seconds.
N_SEEDS = 200


def _prior_lags(n_harm: int) -> np.ndarray:
    """The lag set ``comb_prior_max`` searches, built exactly as that function does."""
    lags = sorted({int(round(m * f / BIN)) for f in DECODER_FUNDAMENTALS_HZ for m in range(1, n_harm + 1)})
    return np.asarray([k for k in lags if LO <= k <= F - 1], dtype=np.int64)


def _decoy_lags(n_sets: int = 24, n_harm: int = M_SHIPPED) -> np.ndarray:
    """Union of the 24 decoy sets' lags: what ``max_j max_tau`` actually maximises over."""
    lags = {
        int(round(m * f / BIN))
        for d in decoy_fundamentals(n_sets)
        for f in d
        for m in range(1, n_harm + 1)
    }
    return np.asarray(sorted(k for k in lags if LO <= k <= F - 1), dtype=np.int64)


def _antiphase_lags(n_harm: int = M_SHIPPED) -> np.ndarray:
    """Half-integer multiples of every fundamental: the anti-phase null of item 3."""
    lags = {
        int(round((m + 0.5) * f / BIN)) for f in DECODER_FUNDAMENTALS_HZ for m in range(1, n_harm + 1)
    }
    return np.asarray(sorted(k for k in lags if LO <= k <= F - 1), dtype=np.int64)


def _complement(tolerance) -> np.ndarray:
    """Lags surviving after masking every harmonic of every fundamental.

    ``tolerance(m)`` is the half-width in bins at harmonic ``m``. ``tolerance=lambda
    m: 1`` is the handover's section 2.3 repair; ``ceil(0.5*m)`` is its section 2.5
    drift-aware variant.
    """
    keep = np.zeros(F, dtype=bool)
    keep[LO:HALF] = True
    for f in DECODER_FUNDAMENTALS_HZ:
        m = 1
        while True:
            k = int(round(m * f / BIN))
            if k >= HALF:
                break
            t = tolerance(m)
            keep[max(0, k - t) : k + t + 1] = False
            m += 1
    return np.flatnonzero(keep)


def _complement_of_prior_only(prior: np.ndarray) -> np.ndarray:
    """The NAIVE complement: mask only ``P`` itself, +-1 bin. Item 1's null."""
    idx = np.arange(F)
    far = np.min(np.abs(idx[:, None] - prior[None, :]), axis=1) > 1
    return np.flatnonzero((idx >= LO) & (idx < HALF) & far)


def _residual(seed: int, smooth_weight: float = 0.6, kernel: int = 9) -> np.ndarray:
    """A peak residual with no comb: the handover's section 7 step 2 recipe.

    ``smooth_weight`` and ``kernel`` are raised for the Udio-like condition, which is
    a residual that is SMOOTHER than real music's and carries no periodicity at all.
    """
    rng = np.random.default_rng(seed)
    smooth = np.abs(uniform_filter1d(rng.standard_normal(F), kernel))
    return smooth_weight * smooth + (1.0 - smooth_weight) * np.abs(rng.standard_normal(F))


def _add_teeth_on_lattice(r, spacing_hz, amp, emphasise=None, emphasis_amp=None):
    """Teeth at ``m * round(spacing/BIN)`` -- the handover's fixture.

    This places the comb at an INTEGER bin multiple, so its true spacing is
    ``round(spacing/BIN) * BIN``, which differs from ``spacing_hz`` by up to half a
    bin. That built-in prior error is what item 5 shows the handover mistook for a
    property of the analysis grid.
    """
    cb = int(round(spacing_hz / BIN))
    m = 1
    while m * cb < F:
        a = emphasis_amp if (emphasise and m % emphasise == 0) else amp
        _tooth(r, m * cb, a)
        m += 1
    return r


def _add_teeth_at_frequency(r, spacing_hz, amp, emphasise=None, emphasis_amp=None, decay=None, f_min=None):
    """Teeth at exact multiples of ``spacing_hz`` in Hz, sampled onto the bin grid.

    This is what a decoder actually produces: the spacing is ``f_s / prod(strides)``,
    an architecture constant in Hz, and our bin grid samples it. ``decay`` gives the
    1/f roll-off of a MUSICAL harmonic series; ``f_min`` restricts the comb to the
    upper band, which is the regime the published detectors work in.
    """
    f_top = F_START + (F - 1) * BIN
    m = 1
    while m * spacing_hz <= f_top:
        f = m * spacing_hz
        if f >= F_START and (f_min is None or f >= f_min):
            a = emphasis_amp if (emphasise and m % emphasise == 0) else amp
            if decay:
                a *= (F_START / f) ** decay
            _tooth(r, int(round((f - F_START) / BIN)), a)
        m += 1
    return r


def _tooth(r, i, amp):
    """One tooth: full amplitude at ``i``, half at each neighbour."""
    if 0 <= i < F:
        r[i] += amp
        if i - 1 >= 0:
            r[i - 1] += amp / 2
        if i + 1 < F:
            r[i + 1] += amp / 2


def _auc(pos, neg) -> float:
    """Mann-Whitney AUC, so the fixture needs no sklearn."""
    pos = np.asarray([v for v in pos if np.isfinite(v)], dtype=float)
    neg = np.asarray([v for v in neg if np.isfinite(v)], dtype=float)
    ranks = np.concatenate([neg, pos]).argsort().argsort() + 1
    return float((ranks[len(neg) :].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# Amplitudes calibrated so the conditions reproduce the handover's section 2.2 score
# levels at seed 3: strong +0.287, weak +0.019, MusicGen-like +0.093 with its
# complement maximum at lag 255 = 249.0 Hz, which is the value that table reports.
AMP_STRONG = 1.40
AMP_WEAK = 0.30
AMP_MG_BASE, AMP_MG_FIFTH = 0.30, 3.00
AMP_MUSICAL = 2.20
MUSICAL_DECAY = 1.6


def _conditions(seed: int, lattice: bool) -> dict[str, np.ndarray]:
    """Five generated conditions and one real one, at one seed.

    ``R_real`` is a MIXTURE: half the draws carry a bass harmonic series whose
    fundamental lands inside the prior's own lag range. A fixture whose real class is
    pure noise cannot test any null that has to separate a decoder comb from music,
    and the handover's does not have one.
    """
    add = _add_teeth_on_lattice if lattice else _add_teeth_at_frequency
    f0 = float(np.random.default_rng(50_000 + seed).uniform(45.0, 210.0))
    real = (
        _residual(9_000 + seed)
        if seed % 2
        else _add_teeth_at_frequency(_residual(9_000 + seed), f0, AMP_MUSICAL, decay=MUSICAL_DECAY)
    )
    return {
        "A_strong": add(_residual(seed), 100.0, AMP_STRONG),
        "B_weak": add(_residual(seed), 100.0, AMP_WEAK),
        "D_musicgen": add(_residual(seed), 50.0, AMP_MG_BASE, emphasise=5, emphasis_amp=AMP_MG_FIFTH),
        "G_highband": _add_teeth_at_frequency(_residual(seed), 100.0, AMP_STRONG, f_min=4_500.0),
        "U_udio": _residual(seed, smooth_weight=0.85, kernel=15),
        "R_real": real,
    }


def _scores(r: np.ndarray) -> dict[str, float]:
    """Every null under test, on one residual, from one autocorrelation."""
    acf = _normalised_acf(r)
    prior, decoys = _prior_lags(M_SHIPPED), _decoy_lags()
    vals = acf[prior]
    tau = int(prior[int(np.argmax(vals))])
    raw = float(np.max(vals))

    band = acf[LO:HI]
    abs_median = float(np.median(np.abs(band)))
    median = float(np.median(band))
    mad = float(np.median(np.abs(band - median)))

    lo_half, hi_half = _normalised_acf(r[:HALF]), _normalised_acf(r[HALF:])

    return {
        "raw_priormax": raw,
        "margin_decoy24": raw - float(np.max(acf[decoys])),
        "complement_naive": raw - float(np.max(acf[_complement_of_prior_only(prior)])),
        "complement_allharm1": raw - float(np.max(acf[_complement(lambda m: 1)])),
        "complement_drift": raw - float(np.max(acf[_complement(lambda m: int(math.ceil(0.5 * m)))])),
        "sharpness_ratio": raw / (abs_median + 1e-12),
        "bulk_difference": raw - abs_median,
        "bulk_zscore": (raw - median) / (1.4826 * mad + 1e-12),
        "antiphase": raw - float(np.max(acf[_antiphase_lags()])),
        "crossband_min": float(min(lo_half[tau], hi_half[tau])),
        "crossband_mean": float((lo_half[tau] + hi_half[tau]) / 2),
    }


@lru_cache(maxsize=2)
def _sweep(lattice: bool) -> dict[str, dict[str, tuple[float, ...]]]:
    """``{score: {condition: values-over-seeds}}``. Cached: the sweep is the slow part."""
    acc: dict[str, dict[str, list[float]]] = {}
    for seed in range(N_SEEDS):
        for condition, residual in _conditions(seed, lattice).items():
            for score, value in _scores(residual).items():
                acc.setdefault(score, {}).setdefault(condition, []).append(value)
    return {s: {c: tuple(v) for c, v in d.items()} for s, d in acc.items()}


def _auc_for(score: str, condition: str, lattice: bool = False) -> float:
    table = _sweep(lattice)[score]
    return _auc(table[condition], table["R_real"])


class TestTheGeometryIsTheRealOne:
    """If these drift, every number below is measuring something else."""

    def test_prior_lag_set_is_the_shipped_one(self):
        assert _prior_lags(M_SHIPPED).tolist() == [44, 51, 77, 88, 96, 102, 154, 176, 192, 205]

    def test_the_decoy_null_is_a_max_over_118_lags_not_10(self):
        # Handover section 2.4 corrects its own earlier x1.9 extreme-value claim to
        # x1.31 on exactly this count. Do not reinstate x1.9.
        assert len(_decoy_lags()) == 118

    def test_eight_of_twenty_four_decoy_sets_collide_with_a_real_candidate(self):
        # Handover section 2.7's untidy wrinkle: decoy FUNDAMENTALS are rejected near
        # real ones, but their HARMONICS can still collide. It biases the null
        # against us, which is the safe direction, but a reviewer may ask.
        prior = _prior_lags(M_SHIPPED)
        collisions = sum(
            any(min(abs(int(round(m * f / BIN)) - p) for p in prior) <= 1 for f in d for m in (1, 2))
            for d in decoy_fundamentals(24)
        )
        assert collisions == 8


class TestTheProfessorsComplementNullInvertsMusicGen:
    """Item 1. The complement contains the comb, so the score subtracts its own evidence."""

    def test_the_musicgen_condition_inverts(self):
        auc = _auc_for("complement_naive", "D_musicgen")
        assert auc < 0.10, (
            f"the naive complement null read {auc:.4f} on the MusicGen condition; at 200 seeds it is "
            "0.0000. If this is no longer an inversion the handover's section 2.2 refutation of the "
            "professor's proposal has to be revisited."
        )

    def test_the_decoy_null_does_not_invert_the_same_condition(self):
        # The contrast is the whole argument: same residual, same prior, different null.
        assert _auc_for("margin_decoy24", "D_musicgen") > 0.85

    def test_it_also_costs_the_strong_comb(self):
        assert _auc_for("complement_naive", "A_strong") < _auc_for("margin_decoy24", "A_strong")


class TestCrossBandConsistencyIsNotTheBestCandidate:
    """Item 2. The handover's highest-ranked idea inverts a high-band-only comb."""

    def test_min_destroys_a_comb_visible_only_in_the_upper_band(self):
        auc = _auc_for("crossband_min", "G_highband")
        raw = _auc_for("raw_priormax", "G_highband")
        assert auc < 0.60 and raw - auc > 0.35, (
            f"cross-band min read {auc:.4f} on a comb present only above 4.5 kHz against the raw "
            f"prior score's {raw:.4f}; at 200 seeds those are 0.4751 and 1.0000. Afchar et al. work "
            "at 3-15 kHz precisely because that is where the comb is visible, so this is not a "
            "corner case. The claim protected here is that min takes the family to chance, not that "
            "it reliably inverts -- the absolute value straddles 0.5 across sweep sizes."
        )

    def test_the_raw_prior_score_handles_that_condition_easily(self):
        assert _auc_for("raw_priormax", "G_highband") > 0.90

    def test_mean_is_a_no_op_rather_than_a_fix(self):
        # The mean of the two half-band autocorrelations approximates the full-band
        # one, so it cannot add information: reporting it as an alternative to min
        # would be reporting the incumbent under a new name.
        assert abs(_auc_for("crossband_mean", "A_strong") - _auc_for("raw_priormax", "A_strong")) < 0.05


class TestTheAntiPhaseNullIsDead:
    """Item 3. Killed by arithmetic, not by statistics."""

    def test_it_inverts_the_musicgen_condition(self):
        auc = _auc_for("antiphase", "D_musicgen")
        assert auc < 0.20, (
            f"the anti-phase null read {auc:.4f} on the MusicGen condition; at 200 seeds it is 0.0000."
        )

    def test_the_reason_a_half_multiple_of_one_fundamental_is_a_multiple_of_another(self):
        # 1.5 * 100 = 150 = 3 * 50. The anti-lag set is supposed to sample the comb's
        # troughs; here it samples its teeth.
        anti = set(_antiphase_lags().tolist())
        musicgen_teeth = {int(round(m * 50.0 / BIN)) for m in range(1, 40)}
        assert anti & musicgen_teeth, (
            "the anti-phase construction no longer collides with a 50 Hz comb, so the recorded "
            "mechanism for its failure is wrong and the idea deserves a fresh look."
        )


class TestRatioCalibrationInvertsWhereDifferenceDoesNot:
    """Item 4. The functional form of a per-track null decides its sign behaviour.

    This is the prediction the SONICS run tests: ``comb_harm4_sharpness`` is a RATIO.
    """

    def test_the_ratio_form_sits_at_or_below_chance_on_the_udio_like_family(self):
        # Not "inverts": the absolute value moves between 0.42 and 0.50 with the
        # sweep size. At or below chance is what is stable.
        assert _auc_for("sharpness_ratio", "U_udio") < 0.52

    def test_the_zscore_form_behaves_the_same_way(self):
        assert _auc_for("bulk_zscore", "U_udio") < 0.52

    def test_the_difference_form_is_clearly_above_the_ratio_form(self):
        ratio = _auc_for("sharpness_ratio", "U_udio")
        difference = _auc_for("bulk_difference", "U_udio")
        assert difference - ratio > 0.12 and difference > 0.55, (
            f"ratio {ratio:.4f}, difference {difference:.4f}; at 200 seeds they are 0.4483 and "
            "0.6631. A smoother residual raises the autocorrelation bulk, so dividing by it deflates "
            "that family more than real music while subtracting does not. The GAP is what is stable "
            "across sweep sizes and what is pinned here. If this ordering reverses, the SONICS "
            "prediction for comb_harm4_sharpness is void."
        )

    def test_the_difference_form_keeps_the_combs_it_should_find(self):
        for condition in ("A_strong", "D_musicgen", "G_highband"):
            assert _auc_for("bulk_difference", condition) > 0.85


class TestTheDriftDiagnosisWasTheFixturesOwnArtefact:
    """Item 5. The same mask succeeds or fails depending on where the teeth are put."""

    def test_the_mask_fails_when_the_prior_frequency_is_wrong(self):
        # Teeth at integer bin multiples give a true spacing 0.4% from the prior's.
        assert _auc_for("complement_allharm1", "D_musicgen", lattice=True) < 0.20

    def test_the_same_mask_succeeds_when_the_prior_frequency_is_right(self):
        assert _auc_for("complement_allharm1", "D_musicgen", lattice=False) > 0.85

    def test_the_prescribed_tolerance_measures_the_analysis_grid_not_the_prior_error(self):
        # Handover section 3 N4 prescribes a tolerance of m * frac(f/bin_hz). For the
        # 93.75 Hz candidate frac is exactly 0, so that formula prescribes a ZERO
        # tolerance at every harmonic however wrong the prior turns out to be --
        # which shows it is not a model of prior error at all.
        assert abs(93.75 / BIN - round(93.75 / BIN)) < 1e-9
        assert abs(100.0 / BIN - round(100.0 / BIN)) == pytest.approx(0.4, abs=0.01)


class TestPriorDepthReachesTheMeasuredCombs:
    """Item 6. Pure arithmetic over measured values -- no RNG, so it cannot drift.

    The ledger's measured decoder combs (sections G2 and O8b) against the lag set
    ``comb_prior_max`` actually searches, and against the per-family recall at the
    95% real quantile reported in section R25. The two families with catastrophic
    recall are exactly the two whose visible tooth lies outside the prior's reach at
    the shipped M=2.
    """

    # (measured comb in Hz, recall at q=0.95 from ledger R25)
    MEASURED = {
        "hifigan_family": (200.195, 0.947),
        "musicgen": (250.000, 0.169),
        "stable_audio_open": (107.420, 0.032),
    }

    def _covered(self, comb_hz: float, n_harm: int) -> bool:
        lag = int(round(comb_hz / BIN))
        return any(abs(lag - p) <= 1 for p in _prior_lags(n_harm))

    def test_recall_tracks_prior_coverage_at_the_shipped_depth(self):
        for name, (comb_hz, recall) in self.MEASURED.items():
            covered = self._covered(comb_hz, M_SHIPPED)
            assert covered == (recall > 0.5), (
                f"{name}: measured comb {comb_hz} Hz, covered at M={M_SHIPPED} is {covered}, but "
                f"recall at q=0.95 is {recall}. The correspondence between prior coverage and recall "
                "is the stated explanation for the 0.032 / 0.169 weakness; if it breaks, that "
                "explanation is withdrawn."
            )

    def test_depth_eight_reaches_every_measured_comb(self):
        for name, (comb_hz, _) in self.MEASURED.items():
            assert self._covered(comb_hz, 8), f"{name} at {comb_hz} Hz is not reached even at M=8"

    def test_suno_needs_depth_four(self):
        # 399.902 Hz is the 8th harmonic of ~49.99 Hz (ledger R7), and Suno's decoder
        # is proprietary and did not inform the candidate list.
        assert not self._covered(399.902, 2)
        assert self._covered(399.902, 4)

    def test_the_shipped_depth_stops_below_the_two_weak_families(self):
        assert max(_prior_lags(M_SHIPPED)) * BIN == pytest.approx(200.2, abs=0.1)
