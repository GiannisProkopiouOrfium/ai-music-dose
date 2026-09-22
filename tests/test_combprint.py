"""Tests for the comb-profile front end.

This arm is the project's unification: a mechanistic artifact feature
(Afchar et al., arXiv:2506.19108) fed to the one-class flow trained on real
music only. The scalar `comb_strength` reaches 0.899 / 0.730 from a single
autocorrelation peak; this keeps the whole profile, which is what their >99%
result actually uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.config import CombPrintConfig

SR = 16_000


def _music(seconds: float = 6.0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * seconds)) / sr
    x = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 220 * (k + 1) * t) for k in range(5))
    return (x + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _with_comb(x: np.ndarray, spacing_hz: float, sr: int = SR, amp: float = 0.02) -> np.ndarray:
    t = np.arange(len(x)) / sr
    comb = np.zeros_like(t)
    f = spacing_hz
    while f < 0.45 * sr:
        comb += np.sin(2 * np.pi * f * t)
        f += spacing_hz
    return (x + amp * comb / max(np.abs(comb).max(), 1e-9)).astype(np.float32)


def test_extractor_is_registered_by_name():
    # get_extractor pulls in the audio stack, which is present on EC2 and not on
    # the local dev machine — same reason the other extractor tests skip here.
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    ex = get_extractor("combprint", device="cpu")
    # CORRECTED 2026-08-20: this asserted `n_lags` (128), the dimension of the
    # `autocorr` profile. The default `profile_kind` has been "hull" (445-d) since
    # the 2026-08-18 runbook. The assertion was stale, not the code — and it went
    # unnoticed for weeks because this file SKIPS locally (no librosa) and only
    # runs on EC2, where nobody ran the full suite. Assert against the config's
    # actual mode so the two cannot drift again.
    cfg = CombPrintConfig()
    expected = {"hull": cfg.n_bins, "autocorr": cfg.n_lags, "both": cfg.n_bins + cfg.n_lags}
    assert ex.embedding_dim == expected[cfg.profile_kind]
    assert ex.sample_rate == 16_000


def test_output_is_a_frame_matrix_the_flow_can_consume():
    """Must look like every other front end: [T, d], finite, float32 — otherwise
    the windowing, pooling and caching machinery cannot be reused unchanged."""
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

    cfg = CombPrintConfig()
    feats = CombPrintExtractor().extract(_music(6.0), SR)
    assert feats.ndim == 2
    expected = {"hull": cfg.n_bins, "autocorr": cfg.n_lags, "both": cfg.n_bins + cfg.n_lags}
    assert feats.shape[1] == expected[cfg.profile_kind]
    assert feats.shape[0] >= 2, "a 6 s clip must yield several spans"
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_profile_separates_combed_from_clean_audio():
    """CORRECTED 2026-08-20 — and this failure was a real gap, not a stale assertion.

    The old assertion was `combed.max() > clean.max()` on the extractor's output.
    In the DEFAULT `hull` mode the profile is max-normalised, so every profile's
    maximum is exactly 1.0 by construction and the assertion can never be true.

    That means the combprint front end has never passed a does-it-carry-signal
    check in the mode it actually runs in — and the combprint flow rows
    (FMC 0.450 / SONICS 0.242) were produced by it. Recorded in the runbook as a
    further reason those rows must be re-measured.

    The correct check for a max-normalised profile is about the SHAPE: a comb must
    make the profile more periodic, which is what `comb_strength` measures.
    """
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.comb_artifacts import comb_strength
    from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

    cfg = CombPrintConfig()
    ex = CombPrintExtractor(cfg)
    # `_music` is a 220 Hz tone WITH five harmonics, so 660/880/1100 Hz land inside
    # the analysis band and form their own comb at 220 Hz spacing — exactly the
    # harmonicity confound the F4 work is about. For a does-it-carry-signal check
    # the clean side must be tonally empty in-band, or the test measures the
    # fixture's own harmonics.
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 6.0)) / SR
    base = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    clean = ex.extract(base, SR).mean(axis=0).astype(float)
    combed = ex.extract(_with_comb(base, 300.0), SR).mean(axis=0).astype(float)

    if cfg.profile_kind == "hull":
        # CORRECTED: each ROW is max-normalised, but the mean ACROSS spans is not —
        # the maxima sit at different bins in different spans, so the average peaks
        # below 1.0. Assert the per-row property, which is the one that holds.
        per_row = ex.extract(base, SR)
        assert np.allclose(
            per_row.max(axis=1), 1.0, atol=1e-4
        ), "hull profiles are max-normalised per row; if this changes, revisit below"
        span_hz = (0.98 * SR / 2 - cfg.f_min) / len(clean)
        s_clean, _, _ = comb_strength(clean, bin_hz=span_hz)
        s_combed, _, _ = comb_strength(combed, bin_hz=span_hz)
        assert s_combed > s_clean, (
            f"a decoder comb must make the profile more periodic: " f"combed {s_combed:.4f} vs clean {s_clean:.4f}"
        )
    else:
        assert combed.max() > clean.max()


def test_smooth_bins_is_configurable_and_changes_the_profile():
    """The one free parameter. It must be selectable, and it must be selected on
    a DIFFERENT corpus from the one reported — sweeping it by AUC on the reported
    corpus is selection on the test set."""
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

    audio = _with_comb(_music(6.0), 300.0)
    # CORRECTED 2026-08-20: `smooth_bins` is the MEDIAN filter's width and has no
    # effect in the default `hull` mode, so this compared two identical arrays.
    # Pin it to the mode the parameter actually belongs to.
    kind = "autocorr" if CombPrintConfig().profile_kind == "hull" else CombPrintConfig().profile_kind
    narrow = CombPrintExtractor(CombPrintConfig(smooth_bins=5, profile_kind=kind)).extract(audio, SR)
    wide = CombPrintExtractor(CombPrintConfig(smooth_bins=65, profile_kind=kind)).extract(audio, SR)
    assert narrow.shape == wide.shape
    assert not np.allclose(narrow, wide)


def test_registered_in_the_runner_config():
    """A front end the runner does not know about cannot be selected with
    --embeddings, and the failure would be a confusing KeyError at run time."""
    import pathlib
    import re

    src = pathlib.Path("scripts/run_balanced_ablation.py").read_text()
    assert re.search(r'"combprint":\s*\{"target_sr":\s*16_000', src)


def test_hull_is_the_default_and_matches_afchars_dimension():
    """Their descriptor is 445-d. Ours collapsed it to one autocorrelation peak,
    which is the measured gap: 0.899 for us against their reported >99%."""
    cfg = CombPrintConfig()
    assert cfg.profile_kind == "hull"
    assert cfg.n_bins == 445
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

    ex = CombPrintExtractor()
    assert ex.embedding_dim == 445
    feats = ex.extract(_music(6.0), SR)
    assert feats.shape[1] == 445
    # max_normalise: clipped to [0, max_db] then divided by the peak
    assert feats.min() >= 0.0 and feats.max() <= 1.0 + 1e-6


def test_profile_kinds_have_the_right_dimensions():
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

    assert CombPrintExtractor(CombPrintConfig(profile_kind="autocorr")).embedding_dim == 128
    assert CombPrintExtractor(CombPrintConfig(profile_kind="both")).embedding_dim == 445 + 128
    with pytest.raises(ValueError, match="unknown profile_kind"):
        CombPrintExtractor(CombPrintConfig(profile_kind="median"))


def test_lower_envelope_sits_below_the_spectrum_unlike_a_median():
    """The substantive correction. A median tracks the middle; Afchar subtracts a
    LOWER envelope, so the residual is 'height above the noise floor' rather than
    'distance from typical'."""
    from intrinsic_ai_music_detection.features.comb_artifacts import lower_envelope, peak_residual

    rng = np.random.default_rng(0)
    spec = -0.005 * np.arange(4000) + rng.normal(0, 0.3, 4000)
    spec[::40] += 8.0
    env = lower_envelope(spec, area=10)
    assert (env <= spec + 1e-9).mean() > 0.95, "a lower envelope must sit under the spectrum"
    median_env = spec - peak_residual(spec, smooth_bins=11)
    assert env.mean() < median_env.mean(), "the lower envelope must sit below the median envelope"
