"""Tests for sub-window block pooling.

Mean pooling averages ~300 frames of a 4 s window into one vector — a ~300x
information bottleneck — and synthesis artifacts are frame-local. This mode
recovers the temporal axis, which is the largest remaining architectural gap
against MusicDET's 2-D convolutional couplings (§9.7.19, item 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.pooling import pool_windows

FRAMES, DIM = 300, 4
RAMP = np.arange(FRAMES * DIM, dtype=float).reshape(FRAMES, DIM)


def test_output_dimension_is_n_blocks_times_input():
    assert pool_windows(RAMP, FRAMES, 150, mode="mean").shape == (1, DIM)
    assert pool_windows(RAMP, FRAMES, 150, mode="blocks", n_blocks=8).shape == (1, 8 * DIM)
    assert pool_windows(RAMP, FRAMES, 150, mode="blocks", n_blocks=4).shape == (1, 4 * DIM)


def test_first_block_is_the_mean_of_the_first_sub_span():
    blocks = pool_windows(RAMP, FRAMES, 150, mode="blocks", n_blocks=8)
    edges = np.linspace(0, FRAMES, 9).astype(int)
    assert np.allclose(blocks[0, :DIM], RAMP[edges[0] : edges[1]].mean(axis=0))


def test_blocks_retain_time_order_where_mean_does_not():
    """The whole point. Mean pooling is invariant to time reversal, so it cannot
    represent a temporal artifact signature at all; blocks can."""
    forward = RAMP
    reversed_ = RAMP[::-1].copy()

    assert np.allclose(
        pool_windows(forward, FRAMES, 150, mode="mean"),
        pool_windows(reversed_, FRAMES, 150, mode="mean"),
    ), "mean pooling is time-blind — this is the bottleneck being addressed"

    assert not np.allclose(
        pool_windows(forward, FRAMES, 150, mode="blocks", n_blocks=8),
        pool_windows(reversed_, FRAMES, 150, mode="blocks", n_blocks=8),
    )


def test_blocks_average_back_to_approximately_the_window_mean():
    """Sub-spans are near-equal (300/8 = 37.5 frames), so block means average
    back to the window mean up to the rounding of the split points. A large
    deviation would mean the spans do not tile the window."""
    blocks = pool_windows(RAMP, FRAMES, 150, mode="blocks", n_blocks=8)
    mean = pool_windows(RAMP, FRAMES, 150, mode="mean")
    assert np.allclose(blocks[0].reshape(8, DIM).mean(axis=0), mean[0], rtol=5e-3)


def test_windows_with_fewer_finite_frames_than_blocks_are_dropped():
    """Zero-padding a short window would inject a constant block, which the flow
    could read as 'this clip was short' — and clip length correlates with class
    on both corpora. Dropping is the safe behaviour."""
    short = np.ones((3, DIM))
    assert pool_windows(short, 3, 3, mode="blocks", n_blocks=8).shape == (0, 8 * DIM)
    # ...but a window with exactly n_blocks frames is fine
    assert pool_windows(np.ones((8, DIM)), 8, 8, mode="blocks", n_blocks=8).shape == (1, 8 * DIM)


def test_non_finite_frames_are_still_dropped():
    dirty = RAMP.copy()
    dirty[10] = np.nan
    out = pool_windows(dirty, FRAMES, 150, mode="blocks", n_blocks=8)
    assert out.shape == (1, 8 * DIM)
    assert np.isfinite(out).all()


def test_invalid_block_count_is_rejected():
    with pytest.raises(ValueError, match="n_blocks"):
        pool_windows(RAMP, FRAMES, 150, mode="blocks", n_blocks=0)


def test_unknown_mode_still_rejected():
    with pytest.raises(ValueError, match="unknown pooling mode"):
        pool_windows(RAMP, FRAMES, 150, mode="attention")
