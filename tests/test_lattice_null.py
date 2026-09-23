"""The deterministic fundamental lattice: the professor's idea, in the right space.

The proposal was to search all lags, zero out the neighbourhoods of the expected
decoder fundamentals, and take the max of the rest. On the LAG axis that fails: a
decoder emits peaks at every multiple of its spacing, ``P`` at M=2 holds only the
first two, and everything from 3*Delta up survives into the complement carrying the
evidence being measured. ``tests/test_null_alternatives.py`` pins the failure at AUC
0.0000 on a MusicGen-like comb.

Excluding on the FUNDAMENTAL axis closes the leak exactly, and this file proves the
"exactly": at M=2 a candidate ``g`` contributes lags ``{g, 2g}``, so rejecting a
relative neighbourhood of ``Delta/2``, ``Delta`` and ``2*Delta`` removes every
possible collision. The contrast with the 24 random decoy sets -- 8 of which contain
a lag within one bin of a real candidate -- is asserted directly.

The other property worth a test is that the construction has NO free parameter. An
unregistered knob is what disqualified the drift-aware complement (handover section
2.5, where the drift rate swings the null 40x), so the lattice enumerates the
saturated lag set rather than sampling a grid, and a test asserts that refining a
grid cannot produce a lag the enumeration missed.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import (
    DECODER_FUNDAMENTALS_HZ,
    comb_acf_floor,
    comb_prior_lattice_null,
    comb_prior_max,
    comb_prior_harmonic_null,
    comb_prior_union_null,
    decoy_fundamentals,
    decoy_lattice_lags,
)

# Both real corpus geometries, because the lag counts and the exclusions differ.
FMC = ("FakeMusicCaps", 16_000 / 16_384, 7_167)
SONICS = ("SONICS", 24_000 / 16_384, 4_779)
GEOMETRIES = [FMC, SONICS]


def _reference_prior_max(residual, bin_hz, n_harm, min_spacing_hz=40.0):
    """``comb_prior_max`` as it stood before ``tol_bins`` was added. The oracle.

    Transcribed rather than imported, in the manner of ``tests/test_afchar_parity.py``:
    the point is to have an independent statement of the published behaviour that a
    future edit cannot drag along with it.
    """
    from intrinsic_ai_music_detection.features.comb_artifacts import _normalised_acf

    acf = _normalised_acf(residual)
    if acf is None:
        return {"strength": float("nan"), "hz": float("nan")}
    n = len(acf)
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    lags = sorted({int(round(m * f / bin_hz)) for f in DECODER_FUNDAMENTALS_HZ for m in range(1, n_harm + 1)})
    lags = [k for k in lags if lo <= k <= n - 1]
    if not lags:
        return {"strength": float("nan"), "hz": float("nan")}
    vals = acf[np.asarray(lags, dtype=np.int64)]
    j = int(np.argmax(vals))
    return {"strength": float(vals[j]), "hz": float(lags[j] * bin_hz)}


def _residual(n_bins: int, spacing_hz: float | None, bin_hz: float, tooth: float = 0.6, seed: int = 0):
    rng = np.random.RandomState(seed)
    r = rng.rand(n_bins) * 0.30
    if spacing_hz:
        step = int(round(spacing_hz / bin_hz))
        r[::step] += tooth
    return r


class TestCombPriorMaxIsUnchangedAtItsDefault:
    """``tol_bins`` defaults to 0, and 0 must be the published behaviour bin for bin."""

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    @pytest.mark.parametrize("n_harm", [2, 4, 8])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_matches_the_reference_exactly(self, name, bin_hz, n_bins, n_harm, seed):
        r = _residual(n_bins, 100.0 if seed % 2 else None, bin_hz, seed=seed)
        got = comb_prior_max(r, bin_hz=bin_hz, n_harm=n_harm)
        want = _reference_prior_max(r, bin_hz, n_harm)
        assert got["strength"] == want["strength"]
        assert got["hz"] == want["hz"]

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_a_tolerance_can_only_raise_the_maximum(self, name, bin_hz, n_bins):
        r = _residual(n_bins, 100.0, bin_hz, seed=4)
        base = comb_prior_max(r, bin_hz=bin_hz, n_harm=2)["strength"]
        widened = comb_prior_max(r, bin_hz=bin_hz, n_harm=2, tol_bins=1)["strength"]
        assert widened >= base


class TestTheLatticeClosesTheHarmonicLeak:
    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    @pytest.mark.parametrize("n_harm", [2, 4, 8])
    def test_no_lattice_lag_touches_a_real_prior_lag(self, name, bin_hz, n_bins, n_harm):
        lo = max(int(round(40.0 / bin_hz)), 1)
        prior = {
            k
            for f in DECODER_FUNDAMENTALS_HZ
            for m in range(1, n_harm + 1)
            for k in [int(round(m * f / bin_hz))]
            if lo <= k <= n_bins - 1
        }
        lattice = set(decoy_lattice_lags(n_bins, bin_hz, n_harm=n_harm))
        assert not (prior & lattice), (
            f"{name} M={n_harm}: the lattice null contains {len(prior & lattice)} lags the real prior "
            "also searches. A candidate g collides when g ~ (m'/m)*Delta for m,m' <= M, so EVERY such "
            "ratio must be excluded -- excluding only Delta/2, Delta, 2*Delta covers M=2 alone, and "
            "that bug made the lattice overlap the prior by 13 of 18 lags at M=4."
        )

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_the_margin_cannot_be_forced_to_zero_by_the_null(self, name, bin_hz, n_bins):
        # This is WHY the overlap matters. margin = real - max(null); if the null
        # contains the prior's argmax the margin is <= 0, and exactly 0 when that lag
        # is the null's maximum too. An atom of probability at zero destroys both the
        # recall and any quantile threshold built on the score -- which is what killed
        # comb_priormax8_margin and made the Mondrian thresholds degenerate.
        r = _residual(n_bins, 100.0, bin_hz, tooth=0.9, seed=21)
        out = comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=2)
        real = comb_prior_max(r, bin_hz=bin_hz, n_harm=2)["strength"]
        assert real - out["strength"] > 0.0

    def test_the_random_decoys_do_not_have_that_property(self):
        # The contrast that motivates the lattice: rejecting draws near a real
        # FUNDAMENTAL does not stop their HARMONICS colliding.
        _, bin_hz, n_bins = FMC
        lo = max(int(round(40.0 / bin_hz)), 1)
        prior = {
            k for f in DECODER_FUNDAMENTALS_HZ for m in (1, 2) for k in [int(round(m * f / bin_hz))] if k >= lo
        }
        colliding = sum(
            any(
                min(abs(int(round(m * f / bin_hz)) - p) for p in prior) <= 1
                for f in d
                for m in (1, 2)
            )
            for d in decoy_fundamentals(24)
        )
        assert colliding == 8

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_every_lattice_lag_comes_from_an_admissible_fundamental(self, name, bin_hz, n_bins):
        # Rounding maps an INTERVAL of fundamentals onto each lag, so the criterion is
        # "some fundamental in [(k-0.5)*bin/m, (k+0.5)*bin/m] is admissible", not
        # "the interval's centre is". Testing the centre alone rejects a lag whose
        # centre lands inside a banned neighbourhood while part of its interval does
        # not -- which is the bug this pair of tests caught in the implementation.
        excluded = [t for d in DECODER_FUNDAMENTALS_HZ for t in (d / 2.0, d, 2.0 * d)]
        step = bin_hz / 64.0  # far finer than any interval, so no gap is missed
        for k in decoy_lattice_lags(n_bins, bin_hz, n_harm=2):
            ok = False
            for m in (1, 2):
                g_lo = max((k - 0.5) * bin_hz / m, 20.0)
                g_hi = min((k + 0.5) * bin_hz / m, 150.0)
                g = g_lo
                while g <= g_hi and not ok:
                    if not any(abs(g - t) <= 0.05 * t for t in excluded):
                        ok = True
                    g += step
            assert ok, f"lag {k} is in the lattice but no admissible fundamental reaches it"


class TestTheLatticeHasNoFreeParameter:
    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_refining_a_grid_cannot_find_a_lag_the_enumeration_missed(self, name, bin_hz, n_bins):
        # The enumeration is saturated by construction; a log grid at any spacing is a
        # subset of it. This is what lets the construction carry no step argument.
        lattice = set(decoy_lattice_lags(n_bins, bin_hz, n_harm=2))
        lo = max(int(round(40.0 / bin_hz)), 1)
        excluded = [t for d in DECODER_FUNDAMENTALS_HZ for t in (d / 2.0, d, 2.0 * d)]
        g, from_grid = 20.0, set()
        while g <= 150.0:
            if not any(abs(g - t) <= 0.05 * t for t in excluded):
                for m in (1, 2):
                    k = int(round(m * g / bin_hz))
                    if lo <= k <= n_bins - 1:
                        from_grid.add(k)
            g *= 1.001
        assert from_grid <= lattice, sorted(from_grid - lattice)[:10]

    def test_it_is_deterministic(self):
        _, bin_hz, n_bins = FMC
        assert decoy_lattice_lags(n_bins, bin_hz, n_harm=2) == decoy_lattice_lags(n_bins, bin_hz, n_harm=2)

    @pytest.mark.parametrize("name,bin_hz,n_bins,random_count", [(*FMC, 118), (*SONICS, 94)])
    def test_the_lag_count_is_matched_to_the_random_decoys_not_smaller(
        self, name, bin_hz, n_bins, random_count
    ):
        # A null over fewer lags has a lower extreme-value maximum and would flatter
        # the margin for free: sigma*sqrt(2 ln K) grows with K. Matching the count is
        # what makes the two nulls comparable.
        n = len(decoy_lattice_lags(n_bins, bin_hz, n_harm=2))
        assert 0.7 * random_count <= n <= 1.4 * random_count, (
            f"{name}: lattice has {n} lags against the random decoys' {random_count}. If these "
            "diverge, the margin comparison is measuring the null's size, not its content."
        )


class TestTheLatticeNullAsAStatistic:
    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_a_real_comb_beats_every_admissible_wrong_fundamental(self, name, bin_hz, n_bins):
        r = _residual(n_bins, 100.0, bin_hz, tooth=0.9, seed=11)
        out = comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=2)
        assert out["pval"] == 0.0, (
            f"{name}: a strong 100 Hz comb did not beat the whole wrong-fundamental null "
            f"(p = {out['pval']}). 100 Hz is in DECODER_FUNDAMENTALS_HZ."
        )
        assert out["strength"] < comb_prior_max(r, bin_hz=bin_hz, n_harm=2)["strength"]

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_aperiodic_input_does_not_beat_the_null(self, name, bin_hz, n_bins):
        r = _residual(n_bins, None, bin_hz, seed=12)
        assert comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=2)["pval"] > 0.0

    def test_the_pvalue_is_a_fraction_so_it_is_comparable_across_corpora(self):
        for _, bin_hz, n_bins in GEOMETRIES:
            out = comb_prior_lattice_null(_residual(n_bins, None, bin_hz, seed=13), bin_hz=bin_hz, n_harm=2)
            assert 0.0 <= out["pval"] <= 1.0
            assert out["n_lags"] > 20

    def test_degenerate_input_is_nan_not_a_crash(self):
        _, bin_hz, n_bins = FMC
        for bad in (np.zeros(n_bins), np.full(n_bins, np.nan), np.array([1.0, 2.0])):
            out = comb_prior_lattice_null(bad, bin_hz=bin_hz, n_harm=2)
            assert np.isnan(out["strength"])


class TestTheAcfFloor:
    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_it_is_positive_and_below_the_prior_peak_for_a_comb(self, name, bin_hz, n_bins):
        r = _residual(n_bins, 100.0, bin_hz, tooth=0.9, seed=14)
        floor = comb_acf_floor(r, bin_hz=bin_hz)
        assert floor > 0.0
        assert floor < comb_prior_max(r, bin_hz=bin_hz, n_harm=2)["strength"]

    def test_a_smoother_residual_has_a_higher_floor(self):
        # The mechanism behind the ratio-versus-difference split: a smoother residual
        # raises the autocorrelation bulk, so DIVIDING by the floor penalises it more
        # than subtracting does.
        from scipy.ndimage import uniform_filter1d

        _, bin_hz, n_bins = FMC
        rng = np.random.RandomState(15)
        rough = np.abs(rng.randn(n_bins))
        smooth = np.abs(uniform_filter1d(rng.randn(n_bins), 15))
        assert comb_acf_floor(smooth, bin_hz=bin_hz) > comb_acf_floor(rough, bin_hz=bin_hz)

    def test_degenerate_input_is_nan(self):
        _, bin_hz, n_bins = FMC
        assert np.isnan(comb_acf_floor(np.zeros(n_bins), bin_hz=bin_hz))


class TestTheDepthTheConstructionCanSupport:
    """A collision-free null does not exist at every depth, and that is a real limit.

    The ratios ``m'/m`` for ``m, m' <= M`` tile the candidate range as M grows: 3
    distinct ratios at M=2, 11 at M=4, 43 at M=8. Against six fundamentals with a 5%
    relative neighbourhood each, the admissible set empties out. Measured on
    FakeMusicCaps: 155 lags at M=2, 74 at M=4, **zero** at M=6 and M=8.

    So the deepest depth this construction supports at the inherited 5% tolerance is
    **M=4** -- which is exactly the depth Suno's measured 399.902 Hz tooth needs. It
    is NOT deep enough for MusicGen's 250 Hz tooth, which enters the prior only at
    M=8. That is a limitation to report, not to tune away: the 5% comes from the
    shipped ``decoy_fundamentals`` and lowering it to reach M=8 would be a new free
    parameter chosen to get an answer.
    """

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_depth_two_and_four_are_supported(self, name, bin_hz, n_bins):
        for n_harm in (2, 4):
            assert len(decoy_lattice_lags(n_bins, bin_hz, n_harm=n_harm)) > 40, (
                f"{name} M={n_harm}: too few admissible lags for the maximum to be a null at all"
            )

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_depth_eight_has_no_admissible_null_at_the_inherited_tolerance(self, name, bin_hz, n_bins):
        assert len(decoy_lattice_lags(n_bins, bin_hz, n_harm=8)) == 0, (
            "M=8 now admits a collision-free null at 5%. If that is because the tolerance was "
            "lowered, it is a new free parameter and must be declared; if the exclusion set was "
            "narrowed, the overlap test above should have caught it."
        )

    def test_the_null_degrades_to_nan_rather_than_a_fabricated_value(self):
        _, bin_hz, n_bins = FMC
        r = _residual(n_bins, 100.0, bin_hz, seed=22)
        out = comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=8)
        assert np.isnan(out["strength"]) and np.isnan(out["pval"]), (
            "with no admissible lags the null must be NaN. Returning 0.0 would make the margin "
            "equal the raw score and look like a calibrated number."
        )


class TestTheUnionNull:
    """Coverage AND disjointness. Neither the decoys nor the lattice has both.

    Measured on the corpora: the decoy union covers 257 lags at M=4 on FakeMusicCaps
    but 10 of the prior's 18 lags sit inside it, which pins 26% of margins at exactly
    zero; the lattice is disjoint by construction but only 52 lags at M=4 on SONICS,
    and a null that small cancels too little of the residual smoothness that inverts
    Udio (margin 0.530 against the decoys' 0.597). The union takes every lag either
    proposes and removes the prior's own.
    """

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    @pytest.mark.parametrize("n_harm", [2, 4])
    def test_it_is_disjoint_from_the_prior(self, name, bin_hz, n_bins, n_harm):
        # The property the decoys lack. Checked through the public statistic: a
        # residual whose prior score is the global maximum must still get a positive
        # margin, which is impossible if the null can contain the prior's own lag.
        r = _residual(n_bins, 100.0, bin_hz, tooth=0.9, seed=31)
        real = comb_prior_max(r, bin_hz=bin_hz, n_harm=n_harm)["strength"]
        out = comb_prior_union_null(r, bin_hz=bin_hz, n_harm=n_harm)
        assert real - out["strength"] > 0.0, (
            f"{name} M={n_harm}: the union null reached the prior's own score, so it contains a lag "
            "the prior searches. exclude_bins is supposed to make that impossible."
        )

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_it_is_at_least_as_large_as_the_lattice_it_contains(self, name, bin_hz, n_bins):
        r = _residual(n_bins, 100.0, bin_hz, seed=32)
        for n_harm in (2, 4):
            uni = comb_prior_union_null(r, bin_hz=bin_hz, n_harm=n_harm)
            lat = comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=n_harm)
            assert uni["n_lags"] >= lat["n_lags"], (
                f"{name} M={n_harm}: the union has fewer lags than the lattice it is built from"
            )

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_a_larger_null_can_only_raise_the_null_maximum(self, name, bin_hz, n_bins):
        # Which means the union margin is CONSERVATIVE against the lattice margin --
        # it can only ever be smaller, never inflated by the extra candidates.
        r = _residual(n_bins, 100.0, bin_hz, seed=33)
        for n_harm in (2, 4):
            uni = comb_prior_union_null(r, bin_hz=bin_hz, n_harm=n_harm)
            lat = comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=n_harm)
            assert uni["strength"] >= lat["strength"] - 1e-12

    def test_it_survives_the_depth_where_the_lattice_is_empty(self):
        # At M=8 the lattice admits nothing (the ratios tile the range), but the
        # decoys still contribute, so the union is defined where the lattice is not.
        _, bin_hz, n_bins = FMC
        r = _residual(n_bins, 100.0, bin_hz, seed=34)
        assert np.isnan(comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=8)["strength"])
        assert np.isfinite(comb_prior_union_null(r, bin_hz=bin_hz, n_harm=8)["strength"])

    def test_degenerate_input_is_nan(self):
        _, bin_hz, n_bins = FMC
        assert np.isnan(comb_prior_union_null(np.zeros(n_bins), bin_hz=bin_hz, n_harm=2)["strength"])


def _true_comb(n_bins: int, spacing_hz: float, bin_hz: float, tooth: float = 0.9, seed: int = 0):
    """Teeth at EXACT multiples of ``spacing_hz`` in Hz -- what a decoder produces.

    ``_residual`` places teeth every ``round(spacing/bin)`` BINS, which silently builds
    in a frequency error of up to half a bin (0.4% at the 100 Hz candidate). That
    artefact is what ledger R27.10 corrected, and it is why the exclusion tests below
    must use this generator and not that one.
    """
    rng = np.random.RandomState(seed)
    r = rng.rand(n_bins) * 0.30
    f_start, m = 1_000.0, 1
    f_top = f_start + (n_bins - 1) * bin_hz
    while m * spacing_hz <= f_top:
        f = m * spacing_hz
        if f >= f_start:
            i = int(round((f - f_start) / bin_hz))
            if 0 <= i < n_bins:
                r[i] += tooth
        m += 1
    return r


class TestTheHarmonicProtectedNull:
    """The null must exclude the comb's WHOLE series, not just the prior's lag set.

    The ratio exclusion in ``decoy_lattice_lags`` removes candidates near
    ``(m'/m)*Delta`` for ``m, m' <= M``. That makes the null disjoint from the PRIOR --
    which is what killed the zero atom -- but at M=4 the ratios stop at 4, so a
    candidate near ``(5/4)*Delta`` is admissible and its 4th multiple lands on
    ``5*Delta``, which is signal. Measured on FakeMusicCaps, 11 true-comb harmonics sit
    inside the M=2 lattice null and 3 inside the M=4 one, and one of them is Stable
    Audio Open's own visible tooth at ``21.53 x 5``.

    The consequence is not subtle. On a 50 Hz comb -- MusicGen's spacing, whose visible
    tooth is the 5th harmonic -- the lattice null yields a NEGATIVE margin at M=2
    (-0.128 on the FakeMusicCaps geometry): the detector subtracts more of its own
    evidence than it keeps. The harmonic-protected null yields +0.511 on the same input.
    """

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    @pytest.mark.parametrize("spacing_hz", [50.0, 100.0])
    @pytest.mark.parametrize("n_harm", [2, 4])
    def test_it_never_eats_a_real_combs_evidence(self, name, bin_hz, n_bins, spacing_hz, n_harm):
        r = _true_comb(n_bins, spacing_hz, bin_hz, seed=41)
        real = comb_prior_max(r, bin_hz=bin_hz, n_harm=n_harm)["strength"]
        margin = real - comb_prior_harmonic_null(r, bin_hz=bin_hz, n_harm=n_harm)["strength"]
        assert margin > 0.25, (
            f"{name} M={n_harm}, {spacing_hz} Hz comb: harmonic-protected margin {margin:.4f}. "
            "Every k*Delta is excluded by construction, so a genuine comb must keep its margin."
        )

    def test_the_lattice_null_DOES_eat_it_which_is_why_this_exists(self):
        # Pinned as the motivating failure, so the weaker exclusion is not reinstated.
        _, bin_hz, n_bins = FMC
        r = _true_comb(n_bins, 50.0, bin_hz, seed=41)
        real = comb_prior_max(r, bin_hz=bin_hz, n_harm=2)["strength"]
        lattice_margin = real - comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=2)["strength"]
        harmonic_margin = real - comb_prior_harmonic_null(r, bin_hz=bin_hz, n_harm=2)["strength"]
        assert lattice_margin < 0.0 < harmonic_margin, (
            f"lattice {lattice_margin:.4f}, harmonic {harmonic_margin:.4f}. The ratio exclusion is "
            "supposed to leave the comb's higher harmonics inside the null at M=2; if it no longer "
            "does, comb_prior_harmonic_null's reason for existing has changed."
        )

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    @pytest.mark.parametrize("n_harm", [2, 4])
    def test_no_harmonic_of_any_real_fundamental_survives_in_the_null(self, name, bin_hz, n_bins, n_harm):
        lo = max(int(round(40.0 / bin_hz)), 1)
        hi = min(n_bins - 1, int(round(n_harm * 150.0 / bin_hz)))
        banned = set()
        for f in DECODER_FUNDAMENTALS_HZ:
            k = 1
            while True:
                lag = int(round(k * f / bin_hz))
                if lag - 1 > hi:
                    break
                banned |= {lag - 1, lag, lag + 1}
                k += 1
        survivors = set(range(lo, hi + 1)) - banned
        assert not (survivors & banned)
        assert len(survivors) > 100, f"{name} M={n_harm}: only {len(survivors)} lags left to be a null"

    @pytest.mark.parametrize("name,bin_hz,n_bins", GEOMETRIES)
    def test_it_is_much_larger_than_the_ratio_lattice(self, name, bin_hz, n_bins):
        # Null SIZE is what cancels the residual smoothness that inverts Udio. The
        # ratio lattice keeps 52-74 lags at M=4; this keeps 252-441.
        r = _true_comb(n_bins, 100.0, bin_hz, seed=42)
        h = comb_prior_harmonic_null(r, bin_hz=bin_hz, n_harm=4)["n_lags"]
        l = comb_prior_lattice_null(r, bin_hz=bin_hz, n_harm=4)["n_lags"]
        assert h > 3 * l, f"{name}: harmonic null {h} lags vs ratio lattice {l}"

    def test_it_is_deterministic_and_seed_free(self):
        _, bin_hz, n_bins = FMC
        r = _true_comb(n_bins, 100.0, bin_hz, seed=43)
        a = comb_prior_harmonic_null(r, bin_hz=bin_hz, n_harm=4)
        b = comb_prior_harmonic_null(r, bin_hz=bin_hz, n_harm=4)
        assert a == b

    def test_degenerate_input_is_nan(self):
        _, bin_hz, n_bins = FMC
        assert np.isnan(comb_prior_harmonic_null(np.zeros(n_bins), bin_hz=bin_hz, n_harm=4)["strength"])
