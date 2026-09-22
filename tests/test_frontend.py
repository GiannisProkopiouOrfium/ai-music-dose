"""Tests for the shared front-end helper.

These pin the invariant that ten of the twelve silent-corruption bugs violated:
a window derived for EnCodec's 75 fps must not become a data-destroying filter
for a front end at ~1 row/s. Every assertion here is "rows survive" or "an empty
result is loud", per the standing rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.frontend import (
    MIN_WINDOW_ROWS,
    EmptyFrontEndOutput,
    FrontEndMismatch,
    frames_for_window,
    resolve_frontend,
    windows_or_raise,
)


class TestFramesForWindow:
    def test_encodec_rate(self):
        # 75 rows/s over 9 s; a 4 s window is 300 rows, a 2 s hop is 150.
        win, hop = frames_for_window(n_rows=675, total_seconds=9.0, window_duration=4.0, hop_duration=2.0)
        assert win == 300
        assert hop == 150

    def test_combprint_rate_does_not_collapse(self):
        # ~1 profile/s: the exact case that returned zero rows for every track.
        win, hop = frames_for_window(n_rows=9, total_seconds=9.0, window_duration=4.0, hop_duration=2.0)
        assert win == 4
        assert hop == 2
        assert win <= 9

    def test_spec_musicdet_rate(self):
        # 100 rows/s (hop 160 @ 16 kHz).
        win, hop = frames_for_window(n_rows=900, total_seconds=9.0, window_duration=4.0, hop_duration=2.0)
        assert win == 400
        assert hop == 200

    @pytest.mark.parametrize("n_rows,seconds", [(675, 9.0), (9, 9.0), (900, 9.0), (7, 7.0), (3, 25.0)])
    def test_window_never_exceeds_available_rows(self, n_rows, seconds):
        win, hop = frames_for_window(n_rows, seconds, window_duration=4.0, hop_duration=2.0)
        assert MIN_WINDOW_ROWS <= win <= max(n_rows, MIN_WINDOW_ROWS)
        assert hop >= 1

    def test_zero_rows_is_a_hard_error(self):
        with pytest.raises(EmptyFrontEndOutput):
            frames_for_window(0, 9.0, 4.0, 2.0)

    def test_nonpositive_duration_is_a_hard_error(self):
        with pytest.raises(ValueError):
            frames_for_window(675, 0.0, 4.0, 2.0)


class TestWindowsOrRaise:
    @pytest.mark.parametrize(
        "n_rows,seconds",
        [(675, 9.0), (9, 9.0), (900, 9.0), (25, 25.0), (7, 7.0)],
    )
    def test_rows_survive_for_every_front_end_rate(self, n_rows, seconds):
        """The assertion the standing rule demands for every new flag: rows survive."""
        emb = np.random.RandomState(0).randn(n_rows, 16).astype(np.float32)
        pooled = windows_or_raise(emb, seconds, window_duration=4.0, hop_duration=2.0)
        assert len(pooled) >= 1, f"{n_rows} rows over {seconds}s produced no windows"

    def test_empty_input_raises_rather_than_returning_empty(self):
        emb = np.zeros((0, 16), dtype=np.float32)
        with pytest.raises(EmptyFrontEndOutput):
            windows_or_raise(emb, 9.0, 4.0, 2.0)


class _FakeFlow:
    def __init__(self, embedding_name):
        self.embedding_name = embedding_name


class TestResolveFrontend:
    def test_mismatch_is_fatal(self):
        """Several front ends are 128-d, so a mismatch is invisible downstream."""
        with pytest.raises(FrontEndMismatch):
            resolve_frontend(embedding="encodec", flow=_FakeFlow("combprint"))

    def test_no_name_anywhere_is_fatal_rather_than_defaulting_to_encodec(self):
        with pytest.raises(FrontEndMismatch):
            resolve_frontend(embedding=None, flow=_FakeFlow(None))

    def test_agreement_does_not_raise_before_instantiation(self):
        """A matching request must get past the guard (extractor load may still fail
        without optional deps, which is a different failure and not our concern)."""
        try:
            fe = resolve_frontend(embedding="combprint", flow=_FakeFlow("combprint"))
        except FrontEndMismatch:
            pytest.fail("matching front end must not raise FrontEndMismatch")
        except Exception:
            pytest.skip("extractor dependencies unavailable locally")
        else:
            assert fe.name == "combprint"
