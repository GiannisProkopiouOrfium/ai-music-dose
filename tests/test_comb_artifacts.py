"""Tests for the deconvolution-comb detector (Afchar et al., arXiv:2506.19108).

This is the one signal in the bibliography with a mechanistic derivation — the
comb spacing is set by a decoder's transposed-conv strides, not by the training
data — so it is the only artifact feature with a principled reason to transfer
across corpora. These tests pin that it actually detects a comb, recovers the
right spacing, and does not fire on clean music.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import comb_features, comb_strength, peak_residual

SR = 24_000


def _tone_bed(sr: int = SR, seconds: float = 4.0) -> np.ndarray:
    """A musical-ish signal: a few harmonics plus noise, no comb."""
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * seconds)) / sr
    x = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 220 * (k + 1) * t) for k in range(5))
    return (x + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _add_comb(audio: np.ndarray, spacing_hz: float, sr: int = SR, amp: float = 0.02) -> np.ndarray:
    """Superimpose equally spaced partials — the deconvolution signature."""
    t = np.arange(len(audio)) / sr
    comb = np.zeros_like(t)
    f = spacing_hz
    while f < 0.45 * sr:
        comb += np.sin(2 * np.pi * f * t)
        f += spacing_hz
    return (audio + amp * comb / max(np.abs(comb).max(), 1e-9)).astype(np.float32)


def test_peak_residual_removes_the_envelope_and_keeps_narrow_peaks():
    freqs = np.linspace(0, 1, 4096)
    envelope = -30.0 * freqs  # smooth tilt
    spikes = np.zeros_like(freqs)
    spikes[::64] = 6.0  # narrow, periodic
    residual = peak_residual(envelope + spikes, smooth_bins=33)
    assert abs(residual.mean()) < 1.0, "the smooth tilt must be removed"
    assert residual[::64].mean() > residual.mean() + 1.0, "the peaks must survive"


def test_comb_strength_recovers_a_known_spacing():
    """A synthetic comb in the residual must be found at the right lag."""
    n, bin_hz, period_bins = 4096, 1.5, 40
    residual = np.zeros(n)
    residual[::period_bins] = 1.0
    strength, spacing, sharpness = comb_strength(residual, bin_hz=bin_hz)
    assert spacing == pytest.approx(period_bins * bin_hz, rel=0.05)
    assert strength > 0.5
    assert sharpness > 2.0


def test_comb_strength_is_low_on_noise():
    rng = np.random.default_rng(1)
    strength, _, sharpness = comb_strength(rng.standard_normal(4096), bin_hz=1.5)
    assert strength < 0.2, "white noise must not read as a comb"
    assert sharpness < 10.0


def test_degenerate_input_returns_nan_not_a_number():
    """A constant residual has no comb. Returning a plausible float here would
    put a meaningless value into a fusion."""
    strength, spacing, _ = comb_strength(np.ones(4096), bin_hz=1.5)
    assert np.isnan(strength) and np.isnan(spacing)


def test_combed_audio_scores_higher_than_clean_audio():
    pytest.importorskip("librosa")
    clean = comb_features(_tone_bed(), SR)
    combed = comb_features(_add_comb(_tone_bed(), spacing_hz=750.0), SR)
    assert clean and combed
    assert combed["comb_strength"] > clean["comb_strength"], "a decoder comb must raise the score above clean music"


def test_band_is_taken_from_nyquist_not_a_hard_coded_constant():
    """The published 3-15 kHz band does not exist after our bandwidth control
    (Nyquist 7.5 kHz). The descriptor must adapt to whatever band survives, or it
    silently returns nothing on controlled audio."""
    pytest.importorskip("librosa")
    feats = comb_features(_add_comb(_tone_bed(), spacing_hz=600.0), SR, f_max=None)
    assert feats, "must produce features without an explicit f_max"
    assert np.isfinite(feats["comb_strength"])


def test_log_axis_variant_is_invariant_to_frequency_scaling():
    """A pitch/speed change multiplies every frequency by a constant, which is a
    translation on a log axis (Dugelay et al. arXiv:2607.27454). The log-axis
    comb strength should therefore move far less than the linear one."""
    pytest.importorskip("librosa")
    base = _add_comb(_tone_bed(), spacing_hz=600.0)
    scaled = _add_comb(_tone_bed(), spacing_hz=600.0 * 1.12)  # ~+2 semitones

    lin = [comb_features(x, SR)["comb_strength"] for x in (base, scaled)]
    log = [comb_features(x, SR, log_axis=True)["comb_log_strength"] for x in (base, scaled)]
    assert all(np.isfinite(v) for v in lin + log)
    assert abs(log[0] - log[1]) <= abs(lin[0] - lin[1]) + 0.15
