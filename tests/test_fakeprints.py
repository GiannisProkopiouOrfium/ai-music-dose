"""Tests for the fakeprint feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.fakeprints import (
    compute_fakeprint,
    compute_spectrogram,
    lower_hull,
    max_normalise,
)


class TestLowerHull:
    def test_basic_shape(self) -> None:
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        idx, hull = lower_hull(x, area=10)
        assert len(idx) == len(hull)
        assert idx[0] == 0
        assert idx[-1] == 99

    def test_constant_signal(self) -> None:
        x = np.ones(50)
        idx, hull = lower_hull(x, area=5)
        assert len(idx) > 0


class TestMaxNormalise:
    def test_output_range(self) -> None:
        x = np.array([0.0, 1.0, 3.0, 5.0, 7.0])
        result = max_normalise(x, max_db=5.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)

    def test_zero_input(self) -> None:
        x = np.zeros(10)
        result = max_normalise(x)
        assert np.allclose(result, 0.0)


class TestComputeSpectrogram:
    def test_mono_signal(self) -> None:
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        stft = compute_spectrogram(audio, sr)
        assert stft.ndim == 3  # [channels, freq, time]
        assert stft.shape[0] == 1  # mono


class TestComputeFakeprint:
    def test_returns_1d(self) -> None:
        sr = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        fp = compute_fakeprint(audio, sr)
        assert fp.ndim == 1
        assert len(fp) > 0
        assert np.all(fp >= 0)
        assert np.all(fp <= 1.0)
