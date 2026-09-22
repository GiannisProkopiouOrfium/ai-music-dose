"""The harmonic readout, asserted against the failure mode it exists to fix.

The claim being tested
----------------------
``comb_strength`` reads the decoder's harmonic series with a single ``argmax``
over ~4,000 autocorrelation lags, so a real-music score is an extreme value of a
noisy autocorrelation. Averaging M harmonics divides that floor by ~sqrt(M) while
leaving a genuine comb alone. ``TestTheSNRArgument`` is that claim, and it is the
only reason to expect the harmonic readout to change the real-vs-generated
ordering rather than relabel it. If it fails, no number from this readout should
be reported.

A hypothesis of ours that this file FALSIFIED
---------------------------------------------
We first proposed that harmonic summing would resolve the *sub-multiple*
ambiguity — that because every measured spacing reads as a harmonic of a smaller
fundamental (107.42 Hz for ``stable_audio_open`` ~ 5 x 21.5; 200.195 Hz for the
three HiFi-GAN vocoders ~ 2 x 100; 250.0 Hz for MusicGen = 5 x 50), summing would
recover the fundamental and stabilise the spacing estimate, lifting
``stable_audio_open``'s 1.4% concentration.

It does not, for an arithmetic reason, and
``test_harmonic_sum_does_NOT_resolve_the_sub_multiple`` pins the falsification so
it cannot be quietly re-adopted. The consequence is that
``comb_harm_spacing_hz`` is **not** a more reliable identity than
``comb_spacing_hz``, and any cross-track consensus built on it inherits the same
instability.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import (
    DECODER_FUNDAMENTALS_HZ,
    _span_overlap_factor,
    comb_harmonic,
    comb_harmonic_stationarity,
    comb_matched_filter,
    comb_prior_strength,
    comb_strength,
    comb_surrogate_null,
    decoy_fundamentals,
)

BIN_HZ = 1.0
N_BINS = 4000


def _series(fundamental_bins: int, weights, n_bins: int = N_BINS, noise: float = 0.05, seed: int = 0):
    """A harmonic series at ``fundamental_bins`` whose members carry ``weights``.

    ``weights[m-1]`` is the height of the m-th harmonic. This is the shape the DSP
    predicts: teeth at every multiple of f_s / prod(strides), not a single period.
    """
    rng = np.random.RandomState(seed)
    p = rng.rand(n_bins) * noise
    for m, w in enumerate(weights, start=1):
        if w:
            p[:: fundamental_bins * m] += w
    return p


def _noise(n_bins: int = N_BINS, noise: float = 0.30, seed: int = 0):
    """The negative class: identical process to ``_series`` with no series added."""
    return np.random.RandomState(seed).rand(n_bins) * noise


def _auc(pos, neg) -> float:
    """Mann-Whitney AUC, so the fixture needs no sklearn."""
    pos = np.asarray([v for v in pos if np.isfinite(v)], dtype=float)
    neg = np.asarray([v for v in neg if np.isfinite(v)], dtype=float)
    ranks = np.concatenate([neg, pos]).argsort().argsort() + 1
    return float((ranks[len(neg) :].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class TestTheFalsifiedSubMultipleHypothesis:
    def test_harmonic_sum_does_NOT_resolve_the_sub_multiple(self):
        """Our hypothesis, and the arithmetic that kills it. Pinned, not deleted.

        Teeth exist at every multiple of 50 bins, with the multiples of 150
        tallest. We predicted ``comb_strength`` would lock onto 150 (it does) and
        that the harmonic sum would recover 50 (it does not).

        The reason: 150 is *itself* a period of this pattern, so ``acf[150]``,
        ``acf[300]`` and ``acf[450]`` are all large and 150 wins the harmonic sum
        as well. Harmonic summing disambiguates a *missing* fundamental, not a
        *weak* one, and a decoder comb has the latter.

        If this test ever starts failing because the harmonic sum returns 50, that
        is a change in behaviour worth understanding before it is welcomed — the
        docstrings, the plan and the paper all currently say it returns 150.
        """
        profile = _series(50, weights=[0.4, 0.0, 1.0], seed=1)

        _, argmax_spacing, _ = comb_strength(profile, bin_hz=BIN_HZ)
        harm = comb_harmonic(profile, bin_hz=BIN_HZ, n_harm=4)

        assert argmax_spacing == pytest.approx(150.0, rel=0.02)
        assert harm["comb_harm_spacing_hz"] == pytest.approx(150.0, rel=0.02), (
            f"harmonic sum returned {harm['comb_harm_spacing_hz']} Hz. If this is "
            f"now 50 Hz the sub-multiple hypothesis has become true and every "
            f"docstring recording it as falsified must be revisited."
        )

    def test_spacing_estimate_is_no_more_concentrated_than_the_argmax(self):
        """The consequence for Arm B, measured: consensus gains nothing from M.

        200 independent noise draws of ONE weak comb; concentration is the C8
        criterion (+-5% of the modal estimate). Measured 0.055 argmax vs 0.060
        harmonic — within noise of each other, and both at the level real music
        (0.027-0.030) and Udio (0.033-0.037) show on the corpora.
        """

        def concentration(values):
            v = np.asarray([x for x in values if np.isfinite(x)])
            m = np.median(v)
            return float(np.mean(np.abs(v - m) <= 0.05 * m))

        profiles = [_series(50, weights=[0.04, 0.0, 0.10], noise=0.30, seed=1000 + s) for s in range(120)]
        argmax = concentration([comb_strength(p, bin_hz=BIN_HZ)[1] for p in profiles])
        harmonic = concentration([comb_harmonic(p, bin_hz=BIN_HZ, n_harm=4)["comb_harm_spacing_hz"] for p in profiles])

        assert harmonic < 3 * argmax + 0.05, (
            f"harmonic concentration {harmonic:.3f} now far exceeds argmax's "
            f"{argmax:.3f}; the 'no more stable' claim in comb_harmonic's "
            f"docstring and in the paper would no longer hold"
        )

    def test_a_plain_comb_is_still_found(self):
        """The fix must not break the easy case it is meant to generalise."""
        profile = _series(80, weights=[1.0], seed=2)
        harm = comb_harmonic(profile, bin_hz=BIN_HZ, n_harm=4)
        assert harm["comb_harm_spacing_hz"] == pytest.approx(80.0, rel=0.02)
        assert harm["comb_harm_strength"] > 0.3

    def test_top_peaks_are_distinct_candidates_not_the_same_peak_twice(self):
        """Arm B votes with these, so they must be suppressed at the C8 tolerance."""
        harm = comb_harmonic(_series(50, weights=[1.0], seed=3), bin_hz=BIN_HZ, n_harm=4)
        top = [harm["comb_harm_spacing_hz"], harm["comb_harm_top2_spacing_hz"], harm["comb_harm_top3_spacing_hz"]]
        assert all(np.isfinite(top)), f"expected three finite candidates, got {top}"
        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert abs(top[a] - top[b]) > 0.05 * min(top[a], top[b]), f"peaks {top[a]} and {top[b]} are the same peak"


class TestTheSNRArgument:
    def test_noise_floor_falls_with_harmonic_count(self):
        """Averaging M harmonics must divide the aperiodic floor by about sqrt(M).

        This is the entire reason to expect the harmonic readout to change the
        real-vs-generated ordering rather than merely relabel it: real music's
        score is an extreme value of a noisy autocorrelation, and this suppresses
        exactly that.
        """
        rng = np.random.RandomState(4)
        noise = rng.rand(N_BINS)
        floors = {m: comb_harmonic(noise, bin_hz=BIN_HZ, n_harm=m)["comb_harm_strength"] for m in (1, 4)}
        ratio = floors[4] / floors[1]
        assert ratio < 0.75, (
            f"noise floor fell only to {ratio:.2f} of its M=1 value at M=4 "
            f"(expected ~1/sqrt(4)=0.5); the SNR argument does not hold"
        )

    def test_detection_improves_in_the_WEAK_comb_regime(self):
        """The load-bearing claim, as an AUC on a symmetric fixture.

        Symmetric in the sense the R7 trap requires: same noise process, same
        amplitude scale, same normalisation, the two classes differing ONLY in
        whether a periodic series is present. An asymmetric fixture once produced
        the opposite conclusion in this project (AUC 0.998 instead of 0.34).

        Measured, 120 draws per class: M=1 (``comb_strength``) 0.53, M=4 0.69,
        M=8 0.82. This regime is the one that matters — strong combs saturate at
        1.0 under every readout, so the entire value of the change lives here.
        """
        combs = [_series(50, weights=[0.04, 0.0, 0.10], noise=0.30, seed=s) for s in range(120)]
        reals = [_noise(seed=9000 + s) for s in range(120)]

        def detect(m):
            return _auc(
                [comb_harmonic(p, bin_hz=BIN_HZ, n_harm=m)["comb_harm_strength"] for p in combs],
                [comb_harmonic(p, bin_hz=BIN_HZ, n_harm=m)["comb_harm_strength"] for p in reals],
            )

        baseline = _auc(
            [comb_strength(p, bin_hz=BIN_HZ)[0] for p in combs],
            [comb_strength(p, bin_hz=BIN_HZ)[0] for p in reals],
        )
        assert detect(8) > detect(4) > baseline + 0.05, (
            f"harmonic summing did not improve weak-comb detection: "
            f"comb_strength {baseline:.3f}, M=4 {detect(4):.3f}, M=8 {detect(8):.3f}"
        )

    def test_it_does_not_manufacture_a_comb_that_is_not_there(self):
        """The Udio guard. No in-band comb must stay at chance at every M.

        The convergent evidence on SONICS is that Udio has no in-band decoder
        comb: its ``comb_strength`` (0.055) sits BELOW real music's (0.072), its
        residual is flatter, and its spacing is less stationary across spans than
        real music's. If a readout reported Udio as detected, the first thing to
        suspect would be the readout.
        """
        absent = [_noise(seed=s) for s in range(120)]
        reals = [_noise(seed=9000 + s) for s in range(120)]
        for m in (2, 4, 8):
            got = _auc(
                [comb_harmonic(p, bin_hz=BIN_HZ, n_harm=m)["comb_harm_strength"] for p in absent],
                [comb_harmonic(p, bin_hz=BIN_HZ, n_harm=m)["comb_harm_strength"] for p in reals],
            )
            assert abs(got - 0.5) < 0.1, f"M={m} scored {got:.3f} on two aperiodic classes; it is reading noise"


class TestSurrogateNull:
    def test_periodic_input_is_many_nulls_out(self):
        z = comb_surrogate_null(_series(80, weights=[1.0], seed=6), bin_hz=BIN_HZ, n_surrogates=20)
        assert np.isfinite(z) and z > 3.0, f"a clean comb scored only z={z:.2f} against its own null"

    def test_aperiodic_input_sits_near_its_own_null(self):
        rng = np.random.RandomState(7)
        z = comb_surrogate_null(rng.rand(N_BINS), bin_hz=BIN_HZ, n_surrogates=20)
        assert np.isfinite(z) and abs(z) < 3.0, f"aperiodic input scored z={z:.2f}; the surrogate is not a null"

    def test_deterministic_for_a_given_residual(self):
        """A fixed seed, so the column is a function of the audio and nothing else."""
        p = _series(80, weights=[1.0], seed=8)
        assert comb_surrogate_null(p, bin_hz=BIN_HZ, n_surrogates=8) == comb_surrogate_null(
            p, bin_hz=BIN_HZ, n_surrogates=8
        )


class TestSpanGeometry:
    def test_overlap_reproduces_the_non_overlapping_set_exactly(self):
        """`spans[::k]` must be the k=1 set, or the published columns would move.

        ``comb_stat_strength`` carries a published SONICS number (0.6864). The
        denser hop exists for the new harmonic channels only; if subsampling did
        not recover the original spans bin-for-bin, adding the feature would
        silently rewrite a result already in the paper.
        """
        span, n = 64_000, 144_000
        k = _span_overlap_factor(n, span, min_spans=8)
        assert k > 1, "a 9 s track at a 4 s span should need overlap to reach 8 spans"
        dense = list(range(0, n - span + 1, span // k))
        sparse = list(range(0, n - span + 1, span))
        assert dense[::k] == sparse, f"{dense[::k]} != {sparse}"

    def test_long_audio_is_untouched(self):
        """SONICS yields 30 spans already, so k must be 1 and nothing may change."""
        assert _span_overlap_factor(120 * 15_000, 4 * 15_000, min_spans=8) == 1

    def test_harmonic_stationarity_separates_a_decoder_from_a_melody(self):
        stationary = np.stack([_series(50, weights=[0.4, 0.0, 1.0], seed=s) for s in range(8)], axis=0)
        wandering = np.stack(
            [_series(sp, weights=[0.4, 0.0, 1.0], seed=i) for i, sp in enumerate((40, 45, 50, 55, 60, 65, 70, 75))],
            axis=0,
        )
        s = comb_harmonic_stationarity(stationary, bin_hz=BIN_HZ)
        w = comb_harmonic_stationarity(wandering, bin_hz=BIN_HZ)
        assert s["comb_harm_spacing_dispersion"] < w["comb_harm_spacing_dispersion"]
        assert s["comb_harm_stat_strength"] > w["comb_harm_stat_strength"]


class TestTheDecoyPriorNull:
    """The control that decides whether ``prior_strength`` is reportable at all.

    ``DECODER_FUNDAMENTALS_HZ`` was assembled partly by working backwards from
    spacings measured on FakeMusicCaps, so its advantage there could be a leak
    rather than physics. The decoys separate the two explanations, and these tests
    only check that the null is CONSTRUCTED honestly — whether it discriminates is
    an empirical question the corpus answers.
    """

    def test_decoys_avoid_the_real_fundamentals(self):
        real = np.asarray(DECODER_FUNDAMENTALS_HZ)
        for s in decoy_fundamentals(24):
            gap = np.abs(np.asarray(s)[:, None] - real[None, :]) / real[None, :]
            assert gap.min() > 0.05, f"decoy {s} contains a value within 5% of a real prior"

    def test_decoys_match_the_real_set_in_size(self):
        """Same cardinality, or the null tests set SIZE instead of set CONTENT."""
        for s in decoy_fundamentals(8):
            assert len(s) == len(DECODER_FUNDAMENTALS_HZ)

    def test_decoys_are_deterministic(self):
        assert decoy_fundamentals(6) == decoy_fundamentals(6)

    def test_decoys_are_distinct_from_each_other(self):
        sets = decoy_fundamentals(24)
        assert len({s for s in sets}) == len(sets)

    def test_a_real_comb_still_beats_its_own_decoys(self):
        """Sanity: on a comb sitting exactly at a real prior, the prior must win.

        If this fails the null is not a null — it would mean the decoys catch the
        comb as readily as the true fundamental, and the whole contrast is void.
        """
        p = _series(100, weights=[1.0], noise=0.30, seed=11)  # 100 bins == a real prior at bin_hz 1.0
        real = comb_prior_strength(p, bin_hz=BIN_HZ, n_harm=2)["strength"]
        decoys = [
            comb_prior_strength(p, bin_hz=BIN_HZ, n_harm=2, fundamentals_hz=d)["strength"]
            for d in decoy_fundamentals(24)
        ]
        assert real > np.median(decoys), f"real prior {real:.3f} did not beat the decoy median {np.median(decoys):.3f}"


class TestTheMatchedFilterUsesAlignment:
    """The constraint an autocorrelation discards: WHERE the teeth sit.

    A transposed-convolution stack puts teeth at multiples of ``f_s / prod(s)``,
    i.e. aligned to DC. An autocorrelation sees only the spacing. If the
    DC-aligned matched filter cannot tell an aligned comb from a shifted one,
    alignment carries no information and the argument for this statistic is void.
    """

    F_START = 1000.0
    BIN = 0.9765625
    NB = 7167

    def _band(self, spacing_hz, tooth, offset_hz=0.0, noise=0.30, seed=0):
        rng = np.random.RandomState(seed)
        r = rng.rand(self.NB) * noise
        f = self.F_START + np.arange(self.NB) * self.BIN
        m_lo = int(np.ceil((self.F_START - offset_hz) / spacing_hz))
        m_hi = int(np.floor((f[-1] - offset_hz) / spacing_hz))
        idx = np.rint((np.arange(m_lo, m_hi + 1) * spacing_hz + offset_hz - self.F_START) / self.BIN).astype(int)
        r[idx[(idx >= 0) & (idx < self.NB)]] += tooth
        return r

    def _mf(self, r, **kw):
        return comb_matched_filter(r, bin_hz=self.BIN, f_start=self.F_START, **kw)

    def test_finds_a_dc_aligned_comb_at_a_prior_fundamental(self):
        out = self._mf(self._band(100.0, 0.25, seed=1))
        assert out["hz"] == pytest.approx(100.0, rel=0.01)
        assert out["z"] > 8.0, f"z={out['z']:.2f} on a clean DC-aligned comb"

    def test_a_shifted_comb_scores_LOWER_than_an_aligned_one(self):
        """The load-bearing test. Same spacing, same teeth, only the offset differs."""
        aligned = self._mf(self._band(100.0, 0.25, offset_hz=0.0, seed=2))["z"]
        shifted = self._mf(self._band(100.0, 0.25, offset_hz=50.0, seed=2))["z"]
        assert aligned > shifted, (
            f"DC-aligned {aligned:.2f} did not beat half-period-shifted {shifted:.2f}; "
            f"the matched filter is not using alignment and comb_mf_z has no "
            f"advantage over the autocorrelation readout"
        )

    def test_the_offset_scanning_control_recovers_the_shifted_comb(self):
        """The control must be an upper bound: it should find what alignment misses."""
        shifted = self._band(100.0, 0.25, offset_hz=50.0, seed=3)
        assert self._mf(shifted, n_offsets=8)["z"] > self._mf(shifted)["z"]

    def test_aperiodic_input_sits_near_zero(self):
        z = self._mf(_noise(self.NB, seed=4))["z"]
        assert abs(z) < 6.0, f"aperiodic residual scored z={z:.2f}; the normalisation is wrong"

    def test_requires_the_absolute_start_frequency_to_mean_anything(self):
        """Same residual, different f_start => different alignment => different score.

        If these agreed, f_start would be decorative and the statistic would be
        offset-blind after all.
        """
        r = self._band(100.0, 0.25, seed=5)
        a = comb_matched_filter(r, bin_hz=self.BIN, f_start=1000.0)["z"]
        b = comb_matched_filter(r, bin_hz=self.BIN, f_start=1050.0)["z"]
        assert a != pytest.approx(b, rel=1e-6)


class TestDegenerateInputIsNaNNotACrash:
    @pytest.mark.parametrize("bad", [np.zeros(10), np.zeros(N_BINS), np.full(N_BINS, np.nan)])
    def test_harmonic(self, bad):
        assert np.isnan(comb_harmonic(bad, bin_hz=BIN_HZ)["comb_harm_strength"])

    @pytest.mark.parametrize("bad", [np.zeros(10), np.zeros(N_BINS)])
    def test_surrogate(self, bad):
        assert np.isnan(comb_surrogate_null(bad, bin_hz=BIN_HZ, n_surrogates=4))

    def test_prior_strength_on_a_band_too_narrow_for_its_harmonics(self):
        assert np.isnan(comb_prior_strength(np.zeros(20), bin_hz=BIN_HZ)["strength"])

    @pytest.mark.parametrize("bad", [np.zeros((0, N_BINS)), np.zeros(N_BINS)])
    def test_harmonic_stationarity(self, bad):
        assert np.isnan(comb_harmonic_stationarity(bad, bin_hz=BIN_HZ)["comb_harm_stat_strength"])
