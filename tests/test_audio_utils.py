"""Tests for audio utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.data.audio_utils import normalize_audio, sliding_window


class TestNormalizeAudio:
    def test_peak_normalize(self) -> None:
        audio = np.array([0.0, 0.5, -1.0, 0.25], dtype=np.float32)
        result = normalize_audio(audio)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_silent_audio(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = normalize_audio(audio)
        assert np.allclose(result, 0.0)


class TestSlidingWindow:
    def test_basic_windowing(self) -> None:
        sr = 100
        audio = np.arange(100, dtype=np.float32)  # 1 second at sr=100
        chunks = sliding_window(audio, sr=sr, window_size=0.3, hop_size=0.1)
        assert len(chunks) > 0
        assert all(len(c) == 30 for c in chunks)

    def test_short_audio(self) -> None:
        sr = 100
        audio = np.arange(10, dtype=np.float32)  # 0.1 seconds at sr=100
        chunks = sliding_window(audio, sr=sr, window_size=0.2, hop_size=0.1)
        # Audio shorter than window — should return empty
        assert len(chunks) == 0
