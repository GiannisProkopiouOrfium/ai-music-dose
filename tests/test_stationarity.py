"""The F4 claim, asserted: stationarity separates a decoder comb from a harmonic one.

Our measured "real music" comb spacings are 401.9 Hz (FakeMusicCaps) and 295.2 Hz
(SONICS) — squarely inside the range of musical f0 — so a time-averaged spectrum
alone cannot distinguish a decoder comb from the harmonic series of a sustained
note. The physical difference is that a decoder's spacing is fixed by its strides
and therefore identical in every span, while a melody's is not.

If ``test_stationary_comb_beats_wandering_comb`` ever fails, ``comb_stat_strength``
does not measure what its docstring claims and no number from it should be
reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import comb_stationarity, comb_strength

BIN_HZ = 1.0
N_BINS = 3000
N_SPANS = 8


def _comb_profile(spacing_bins: int, n_bins: int = N_BINS, noise: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    p = rng.rand(n_bins) * noise
    p[::spacing_bins] += 1.0
    return p


def _stationary(spacing_bins: int = 200):
    return np.stack([_comb_profile(spacing_bins, seed=s) for s in range(N_SPANS)], axis=0)


def _wandering(spacings=(150, 170, 190, 210, 230, 250, 270, 290)):
    return np.stack([_comb_profile(sp, seed=i) for i, sp in enumerate(spacings)], axis=0)


class TestCombStationarity:
    def test_stationary_comb_beats_wandering_comb(self):
        stat = comb_stationarity(_stationary(), bin_hz=BIN_HZ)
        wand = comb_stationarity(_wandering(), bin_hz=BIN_HZ)
        assert stat["comb_stat_strength"] > wand["comb_stat_strength"], (
            f"stationary {stat['comb_stat_strength']:.3f} must exceed " f"wandering {wand['comb_stat_strength']:.3f}"
        )

    def test_both_look_equally_comb_like_within_a_single_span(self):
        """The discriminator must come from ACROSS spans, not from within one.

        If a single span already separated them, stationarity would be redundant
        with `comb_strength` and F4 would not be a new feature.
        """
        s_one, _, _ = comb_strength(_stationary()[0], bin_hz=BIN_HZ)
        w_one, _, _ = comb_strength(_wandering()[0], bin_hz=BIN_HZ)
        assert abs(s_one - w_one) < 0.25, (
            f"single-span strengths differ too much ({s_one:.3f} vs {w_one:.3f}); "
            "the fixture does not isolate stationarity"
        )

    def test_spacing_dispersion_is_near_zero_for_a_decoder(self):
        stat = comb_stationarity(_stationary(), bin_hz=BIN_HZ)
        wand = comb_stationarity(_wandering(), bin_hz=BIN_HZ)
        assert stat["comb_spacing_dispersion"] < wand["comb_spacing_dispersion"]

    def test_single_span_yields_nan_not_a_fabricated_one(self):
        """One span carries no stationarity information; say so rather than invent it."""
        out = comb_stationarity(_stationary()[:1], bin_hz=BIN_HZ)
        assert np.isnan(out["comb_stat_strength"])

    @pytest.mark.parametrize("bad", [np.zeros((0, N_BINS)), np.zeros(N_BINS)])
    def test_degenerate_input_is_nan_not_a_crash(self, bad):
        out = comb_stationarity(bad, bin_hz=BIN_HZ)
        assert np.isnan(out["comb_stat_strength"])
