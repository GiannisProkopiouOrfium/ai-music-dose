"""End-to-end behaviour of the corrected comb operator.

These need librosa (present on EC2, absent locally) and so skip on the dev box —
same pattern as the other 20 local skips. They are the smoke test that must pass
before the runbook's measurement battery is worth starting: if a planted comb at
a known spacing is not recovered here, no corpus number from this detector means
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

librosa = pytest.importorskip("librosa", reason="audio stack not installed locally")

from intrinsic_ai_music_detection.features.comb_artifacts import afchar_fakeprint, comb_features  # noqa: E402

SR = 16_000
DUR = 9.0
TOOTH_HZ = 200.195  # the spacing three FakeMusicCaps generators actually cluster at


def _music(seed: int = 0, dur: float = DUR) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    rng = np.random.RandomState(seed)
    x = rng.randn(len(t)).astype(np.float32) * 0.01
    for f0 in (220.0, 330.0, 440.0):
        x += (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    return x


def _add_comb(x: np.ndarray, spacing_hz: float = TOOTH_HZ, amp: float = 0.004) -> np.ndarray:
    t = np.arange(len(x)) / SR
    y = x.copy()
    for k in range(1, int(0.98 * (SR / 2) / spacing_hz)):
        y += (amp * np.sin(2 * np.pi * spacing_hz * k * t)).astype(np.float32)
    return y


class TestCorrectedOperatorRecoversAPlantedComb:
    def test_comb_raises_strength_above_music_alone(self):
        clean = comb_features(_music(), SR)
        combed = comb_features(_add_comb(_music()), SR)
        assert combed["comb_strength"] > clean["comb_strength"], (
            f"planted comb {combed['comb_strength']:.4f} must beat music-only " f"{clean['comb_strength']:.4f}"
        )

    def test_recovered_spacing_matches_the_planted_one(self):
        f = comb_features(_add_comb(_music()), SR)
        got = f["comb_spacing_hz"]
        # Accept any harmonic of the planted spacing: the autocorrelation peak may
        # land on a multiple, which is why the runbook reports spacing WITH the
        # smoothing width rather than alone.
        ratio = got / TOOTH_HZ
        assert (
            abs(ratio - round(ratio)) < 0.05 and round(ratio) >= 1
        ), f"recovered {got:.1f} Hz is not a harmonic of the planted {TOOTH_HZ} Hz"

    def test_the_5db_clip_is_what_costs_the_scalar_readout_its_separation(self):
        """CORRECTED 2026-08-20. The original assertion here was wrong.

        It claimed "hull + dB-mean must beat median + power-mean" and FAILED on
        EC2: corrected separation 0.0769 against legacy 0.8076. The diagnosis is
        that the operator and the READOUT are coupled.

        Afchar's ``max_normalise`` clips the residual at 5 dB and divides by the
        maximum. That is right for THEIR readout — a logistic regression over 445
        bins needs only the pattern of which bins peak, so loudness-invariance
        helps. It is wrong for OURS: ``comb_strength`` is an autocorrelation peak
        and needs amplitude structure. Above 5 dB every tooth saturates to 1.0,
        the residual becomes a near-binary mask, and noise bins that also clear
        5 dB inflate the autocorrelation floor.

        So the assertion is now the one that is actually true and actually
        useful: **unclipping recovers the separation.** Which envelope and which
        averaging domain win is left to real audio via
        ``eval_comb_detector.py --operator-grid``, because a synthetic fixture has
        already answered that question two different ways.
        """
        clean, combed = _music(), _add_comb(_music())

        def gap(**kw):
            return comb_features(combed, SR, **kw)["comb_strength"] - comb_features(clean, SR, **kw)["comb_strength"]

        clipped = gap(residual_op="hull", average="db", f_min=1000.0, hull_clip_db=5.0)
        unclipped = gap(residual_op="hull", average="db", f_min=1000.0, hull_clip_db=0.0)
        assert unclipped > clipped, (
            f"removing the 5 dB clip must restore separation for the scalar readout: "
            f"unclipped {unclipped:.4f} vs clipped {clipped:.4f}"
        )

    def test_clipping_is_still_correct_for_the_profile_readout(self):
        """The clip is not a bug — it is right for the path it was designed for.

        ``afchar_fakeprint`` (the NMF / logistic-regression / combprint path) keeps
        it, and must: a max-normalised profile is what their published dimension
        and their classifier expect.
        """
        _, fp = afchar_fakeprint(_add_comb(_music()), SR, n_bins=445)
        assert abs(fp.max() - 1.0) < 1e-6, "the profile path must stay max-normalised"


class TestStationarityDistinguishesHarmonicsFromDecoder:
    def test_a_moving_melody_is_less_stationary_than_a_fixed_comb(self):
        """The F4 confound check, on audio rather than on synthetic profiles."""
        t = np.arange(int(SR * DUR)) / SR
        # A "melody": harmonic comb whose f0 steps between spans.
        melody = np.zeros(len(t), dtype=np.float32)
        span = int(SR * 3.0)
        for i, f0 in enumerate((196.0, 261.6, 329.6)):
            sl = slice(i * span, (i + 1) * span)
            tt = t[sl]
            for k in range(1, 40):
                melody[sl] += (0.05 * np.sin(2 * np.pi * f0 * k * tt)).astype(np.float32)

        fixed = _add_comb(_music(), amp=0.02)

        m = comb_features(melody, SR)["comb_stat_strength"]
        d = comb_features(fixed, SR)["comb_stat_strength"]
        assert d > m, (
            f"a stride-fixed comb ({d:.4f}) must be more stationary than a moving "
            f"harmonic series ({m:.4f}) — this is the discriminator F4 rests on"
        )


class TestAfcharFakeprint:
    def test_returns_the_published_dimension_when_asked(self):
        _, fp = afchar_fakeprint(_add_comb(_music()), SR, n_bins=445)
        assert fp.shape == (445,)

    def test_defaults_to_full_resolution_not_445(self):
        """445 bins would smear a 1-3 bin tooth by ~16x on the autocorrelation path."""
        _, fp = afchar_fakeprint(_add_comb(_music()), SR)
        assert len(fp) > 3000, f"expected full-resolution residual, got {len(fp)} bins"

    def test_max_normalised_to_unit_peak(self):
        _, fp = afchar_fakeprint(_add_comb(_music()), SR)
        assert 0.0 <= fp.min() and abs(fp.max() - 1.0) < 1e-6

    def test_empty_band_is_a_hard_error(self):
        with pytest.raises(ValueError, match="empty analysis band"):
            afchar_fakeprint(_music(), SR, f_min=7900.0, f_max=7950.0)
