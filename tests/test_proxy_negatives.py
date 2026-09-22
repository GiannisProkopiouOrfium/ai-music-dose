"""Tests for the synthesised-negative path (Afchar arXiv:2501.10111).

The negatives are the training signal, so a bug here does not crash — it
produces a detector that learned the wrong thing and reports a plausible AUC.
These pin the properties the method depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.proxy_negatives import RECON_TYPES, _match_length, synthesise_negative


def test_reconstruction_length_is_matched_to_the_source():
    """Codec framing changes the sample count. Left unmatched, duration becomes
    a feature correlated with the label — a channel cue of exactly the kind the
    whole proxy-negative design exists to avoid.
    """
    src_len = 24_000
    assert len(_match_length(np.zeros(src_len + 512, dtype=np.float32), src_len)) == src_len
    assert len(_match_length(np.zeros(src_len - 512, dtype=np.float32), src_len)) == src_len
    assert len(_match_length(np.zeros(src_len, dtype=np.float32), src_len)) == src_len


def test_short_reconstruction_is_padded_not_looped():
    """Padding must be silence, not a repeat: a looped tail would introduce a
    periodicity the flow could key on."""
    out = _match_length(np.ones(100, dtype=np.float32), 150)
    assert np.all(out[:100] == 1.0)
    assert np.all(out[100:] == 0.0)


def test_unknown_recon_type_is_rejected_loudly():
    """A typo in --recon-types must fail, not silently yield zero negatives."""
    audio = np.zeros(24_000, dtype=np.float32)
    with pytest.raises(ValueError, match="unknown recon_type"):
        synthesise_negative(audio, 24_000, "hifigan_v3", post_mp3_kbps=None)


def test_recon_types_are_multiple_decoder_families():
    """Training on one decoder teaches 'this is EnCodec', not 'this is a neural
    decoder'. The available set must span families."""
    assert any(t.startswith("encodec_") for t in RECON_TYPES)
    assert any(t.startswith("griffinlim_") for t in RECON_TYPES)


def test_griffinlim_negative_preserves_length_and_bandwidth_control():
    """End-to-end on the one decoder that needs no model download."""
    librosa = pytest.importorskip("librosa")
    sr = 24_000
    t = np.arange(sr) / sr
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    out = synthesise_negative(audio, sr, "griffinlim_128mel", post_mp3_kbps=None, resample_hz=15_000)
    assert len(out) == len(audio), "the negative must not differ from its source in duration"
    assert np.isfinite(out).all()

    spec = np.abs(librosa.stft(out, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    above = spec.mean(axis=1)[freqs > 9_000].sum()
    total = spec.mean(axis=1).sum()
    assert above / max(total, 1e-20) < 1e-3, (
        "the bandwidth control must apply to negatives too, or the detector learns "
        "'negatives are band-limited' instead of 'negatives are decoded'"
    )
