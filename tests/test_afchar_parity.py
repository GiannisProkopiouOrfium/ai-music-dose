"""Parity with Afchar et al.'s released operator, and the orientation it implies.

Why this file exists
--------------------
The project independently reimplemented ``lower_hull`` as a median filter and a
running minimum, reported numbers from it for weeks, and recorded a negative
result that was really an operator mismatch. ``features/fakeprints.py`` had their
exact loop the whole time.

The rule these tests encode: *before recording a result from a bibliography
method, assert byte-level agreement with the source implementation on a fixture.*

Reference implementation, transcribed from
``deezer/ismir25-ai-music-detector/compute_fakeprints.py`` (CC BY-NC 4.0)::

    def lower_hull(x, area=10):
        idx, hull = [], []
        for i in range(len(x)-area+1):
            patch = x[i:i+area]
            rel_idx = np.argmin(patch); abs_idx = rel_idx + i
            if abs_idx not in idx:
                idx.append(abs_idx); hull.append(patch[rel_idx])
        if idx[0] != 0:      idx.insert(0, 0);        hull.insert(0, x[0])
        if idx[-1] != len(x)-1: idx.append(len(x)-1); hull.append(x[-1])
        return np.array(idx), np.array(hull)
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import afchar_hull_curve, lower_hull_indices


def _reference_lower_hull(x, area=10):
    """Their loop, verbatim. O(n * area) — used only as the oracle."""
    idx: list[int] = []
    hull: list[float] = []
    for i in range(len(x) - area + 1):
        patch = x[i : i + area]
        rel_idx = int(np.argmin(patch))
        abs_idx = rel_idx + i
        if abs_idx not in idx:
            idx.append(abs_idx)
            hull.append(float(patch[rel_idx]))
    if idx[0] != 0:
        idx.insert(0, 0)
        hull.insert(0, float(x[0]))
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1)
        hull.append(float(x[-1]))
    return np.array(idx), np.array(hull)


class TestLowerHullParity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    @pytest.mark.parametrize("area", [5, 10, 33])
    def test_indices_match_reference_exactly(self, seed, area):
        rng = np.random.RandomState(seed)
        x = rng.randn(2000).astype(np.float64) * 5.0
        ref_idx, _ = _reference_lower_hull(x, area=area)
        got = lower_hull_indices(x, area=area)
        np.testing.assert_array_equal(got, ref_idx)

    def test_ties_keep_the_earlier_index_like_argmin(self):
        """np.argmin returns the FIRST minimum; the deque must not drift to the last."""
        x = np.array([5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        ref_idx, _ = _reference_lower_hull(x, area=4)
        np.testing.assert_array_equal(lower_hull_indices(x, area=4), ref_idx)

    def test_on_a_comb_signal(self):
        """The realistic case: a smooth envelope with narrow periodic teeth."""
        n = 4096
        base = -40.0 + 10.0 * np.exp(-np.arange(n) / 900.0)
        base[::64] += 6.0  # teeth every 64 bins
        ref_idx, _ = _reference_lower_hull(base, area=10)
        np.testing.assert_array_equal(lower_hull_indices(base, area=10), ref_idx)


class TestHullCurve:
    def test_hull_sits_below_the_spectrum_between_teeth(self):
        n = 4096
        freqs = np.linspace(1000.0, 8000.0, n)
        spec = -40.0 + 10.0 * np.exp(-np.arange(n) / 900.0)
        spec[::64] += 6.0
        hull = afchar_hull_curve(freqs, spec, area=10)
        teeth = np.zeros(n, dtype=bool)
        teeth[::64] = True
        # The residual must be concentrated ON the teeth, which is the whole
        # point of a lower envelope rather than a median.
        assert (spec[teeth] - hull[teeth]).mean() > (spec[~teeth] - hull[~teeth]).mean()

    def test_clipped_at_min_db(self):
        freqs = np.linspace(1000.0, 8000.0, 512)
        spec = np.full(512, -200.0)
        hull = afchar_hull_curve(freqs, spec, area=10, min_db=-45.0)
        assert np.all(hull >= -45.0 - 1e-9)


class TestMeanDomainMatters:
    """dB-domain averaging vs power-domain averaging are not interchangeable.

    This is §0.2's claim, asserted rather than argued: on a signal that is mostly
    loud music with a faint stationary comb, the geometric (dB) mean exposes the
    comb and the arithmetic (power) mean buries it.
    """

    def test_db_mean_recovers_a_faint_stationary_comb_better(self):
        rng = np.random.RandomState(0)
        n_frames, n_bins = 200, 512
        # Loud, wandering "music": a few strong bins that move frame to frame.
        power = rng.rand(n_frames, n_bins) * 1e-4
        for t in range(n_frames):
            power[t, rng.randint(0, n_bins, size=8)] += 1.0
        # A faint but perfectly stationary comb.
        power[:, ::32] += 1e-3

        db_mean = (10.0 * np.log10(np.clip(power, 1e-10, 1e6))).mean(axis=0)
        power_mean = 10.0 * np.log10(np.maximum(power.mean(axis=0), 1e-20))

        teeth = np.zeros(n_bins, dtype=bool)
        teeth[::32] = True

        def contrast(v):
            return (v[teeth].mean() - v[~teeth].mean()) / (v.std() + 1e-12)

        assert contrast(db_mean) > contrast(power_mean), (
            "dB-domain averaging must expose a stationary comb more than "
            "power-domain averaging; if this fails, §0.2's rationale is wrong."
        )
