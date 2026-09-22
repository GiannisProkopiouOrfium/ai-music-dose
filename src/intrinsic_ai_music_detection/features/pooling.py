"""Canonical window pooling for embedding frame matrices.

This is THE single implementation of the 4s/2s (or any) sliding-window pooling
used across the pipeline. Historically ~6 scripts carried near-duplicate private
copies of ``_pool_windows`` with silently divergent edge-case behaviour (empty
vs whole-track fallback when a clip is shorter than one window, with/without
non-finite filtering). All consumers should import from here.

Canonical semantics (matching ``run_balanced_ablation.py``'s historical
implementation, which produced every reported number):

- Slide a ``window_frames``-long window with ``hop_frames`` hop over the frame
  axis; the trailing partial window is dropped.
- Within each window, rows (frames) containing any non-finite value are
  dropped; the window is kept if at least one finite frame remains.
- A matrix shorter than one window yields an EMPTY ``(0, d)`` result — callers
  decide how to handle short clips explicitly (no silent whole-track fallback).

The EnCodec 24 kHz encoder produces frames at 75 Hz (24_000 / 320); the
constant lives here so scripts stop re-declaring it.
"""

from __future__ import annotations

import numpy as np

# EnCodec 24 kHz: hop 320 samples -> 75 latent frames per second.
ENCODEC_FPS = 75.0


def pool_windows(
    emb_matrix: np.ndarray,
    window_frames: int,
    hop_frames: int,
    mode: str = "mean",
    n_blocks: int = 8,
) -> np.ndarray:
    """Pool embedding frames within each sliding window.

    Parameters
    ----------
    emb_matrix : ``[n_frames, d]`` frame embeddings.
    window_frames / hop_frames : window length and hop, in frames.
    mode : ``"mean"`` — per-window frame centroid, output ``[n_windows, d]``
           (the headline representation);
           ``"meanstd"`` — per-window mean concatenated with per-window std,
           output ``[n_windows, 2*d]`` (keeps within-window texture that the
           mean discards);
           ``"blocks"`` — split each window into ``n_blocks`` equal sub-spans,
           mean-pool each, concatenate, output ``[n_windows, n_blocks*d]``.
    n_blocks : number of sub-spans for ``mode="blocks"``.

    Returns an empty ``(0, out_dim)`` array when the input is shorter than one
    window or 0-length.

    Why ``"blocks"`` exists
    -----------------------
    Mean pooling averages ~300 frames of a 4 s window into a single ``d``-vector
    — roughly a 300x information bottleneck — and synthesis artifacts are
    frame-local. MusicDET (arXiv:2605.18072) instead models a 64x100
    time-frequency map with convolutional coupling nets, and that is the largest
    remaining architectural gap between their method and ours (§9.7.19, item 3:
    "the honest remaining gap"). Splitting the window into sub-spans recovers the
    temporal axis at ``n_blocks`` x the dimensionality while keeping the existing
    MLP-coupling flow, so it costs no new machinery.

    It also costs no re-extraction: pooling is applied to cached FRAME matrices,
    so a pooling sweep is flow-training only on caches that already exist.
    """
    if mode not in ("mean", "meanstd", "delta", "blocks"):
        raise ValueError(f"unknown pooling mode: {mode!r}")
    if mode == "blocks" and n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    d = emb_matrix.shape[1] if emb_matrix.ndim == 2 else 0
    if mode == "meanstd":
        out_dim = 2 * d
    elif mode == "blocks":
        out_dim = n_blocks * d
    else:
        out_dim = d
    pooled: list[np.ndarray] = []
    n = len(emb_matrix)
    start = 0
    while start + window_frames <= n:
        sub = emb_matrix[start : start + window_frames]
        finite = sub[np.isfinite(sub).all(axis=1)]
        if len(finite) >= 1:
            if mode == "meanstd":
                pooled.append(np.concatenate([finite.mean(axis=0), finite.std(axis=0)]))
            elif mode == "blocks":
                # Drop the window rather than pad when there are fewer finite
                # frames than blocks: zero-padding would inject a constant block
                # that the flow could read as "this window was short", which
                # correlates with clip length and therefore with class.
                if len(finite) < n_blocks:
                    start += hop_frames
                    continue
                edges = np.linspace(0, len(finite), n_blocks + 1).astype(int)
                pooled.append(np.concatenate([finite[edges[b] : edges[b + 1]].mean(axis=0) for b in range(n_blocks)]))
            else:  # "mean", and "delta" (which post-processes the mean sequence below)
                pooled.append(finite.mean(axis=0))
        start += hop_frames
    if not pooled:
        return np.empty((0, out_dim), dtype=np.float64)
    out = np.array(pooled, dtype=np.float64)

    if mode == "delta":
        # TRANSITION modelling: each row becomes [window, window - previous window],
        # so the flow learns the joint density of "what this window looks like" AND
        # "how the music got here from the previous window". The density model can
        # then flag an implausible *transition* even when both endpoints are
        # individually plausible — "a real song would never move like that".
        # The first window has no predecessor and is dropped (never zero-padded,
        # which would inject a systematic fake transition at every track start).
        if len(out) < 2:
            return np.empty((0, out_dim), dtype=np.float64)
        out = np.concatenate([out[1:], np.diff(out, axis=0)], axis=1)
    return out


def frame_features(
    emb_matrix: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """Return per-frame features (no pooling), non-finite frames dropped.

    The frame-level ablation arm: at 75 Hz a 10 s clip yields ~750 samples
    instead of ~4 pooled windows — a ~200x increase in effective training
    samples per track for corpora with short clips (e.g. MusicCaps).

    ``stride`` keeps every ``stride``-th frame (after non-finite filtering)
    to bound memory for long corpora; 1 keeps everything.
    """
    if emb_matrix.ndim != 2 or len(emb_matrix) == 0:
        d = emb_matrix.shape[1] if emb_matrix.ndim == 2 else 0
        return np.empty((0, d), dtype=np.float64)
    finite = emb_matrix[np.isfinite(emb_matrix).all(axis=1)]
    if stride > 1:
        finite = finite[::stride]
    return np.asarray(finite, dtype=np.float64)
