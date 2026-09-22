"""Audio loading, resampling, and windowing utilities."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
import numpy.typing as npt


def load_audio(
    path: str | Path,
    target_sr: int,
    mono: bool = True,
    max_duration: float | None = None,
    preprocess: Literal["raw", "canonical"] | None = None,
    preprocess_kwargs: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.float32], int]:
    """Load an audio file, resample to *target_sr*, and optionally preprocess.

    Parameters
    ----------
    path
        Audio file path.
    target_sr
        Target sample rate.
    mono
        Convert to mono (ignored when ``preprocess`` is set).
    max_duration
        Maximum duration in seconds to load.
    preprocess
        When ``"raw"`` or ``"canonical"``, delegates to
        :func:`~intrinsic_ai_music_detection.data.audio_preprocessing.preprocess_audio`
        for unified processing.  ``None`` uses the original librosa-only path.
    preprocess_kwargs
        Additional keyword arguments forwarded to ``preprocess_audio``
        (e.g. ``mp3_bitrate_kbps``, ``target_lufs``).

    Returns
    -------
    audio : ndarray of shape ``(n_samples,)``
    sr : int
    """
    if preprocess is not None:
        from intrinsic_ai_music_detection.data.audio_preprocessing import preprocess_audio

        kw: dict[str, Any] = {"target_sr": target_sr, "mode": preprocess}
        if max_duration is not None:
            kw["max_duration"] = max_duration
        if preprocess_kwargs:
            kw.update(preprocess_kwargs)
        audio, sr, _ = preprocess_audio(path, **kw)
        return audio, sr

    duration = max_duration if max_duration else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        audio, sr = librosa.load(path, sr=target_sr, mono=mono, duration=duration)
    return audio.astype(np.float32), sr


def get_duration(path: str | Path) -> float:
    """Return the duration of an audio file in seconds without loading it fully."""
    return librosa.get_duration(path=path)


def normalize_audio(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio


def region_crop(
    audio: npt.NDArray[np.float32],
    sr: int,
    region: str,
    duration: float,
    rng: "np.random.Generator | None" = None,
) -> npt.NDArray[np.float32]:
    """Extract a contiguous segment of *duration* seconds from *audio*.

    Parameters
    ----------
    region : {'full', 'intro', 'middle', 'outro', 'random'}
        full   — return audio unchanged (caller enforces the duration cap).
        intro  — first *duration* seconds.
        middle — centre-crop to *duration* seconds (same as center_crop).
        outro  — last *duration* seconds.
        random — uniformly random start position.
    duration : float
        Target region length in seconds.
    rng : optional NumPy Generator for reproducible random crops.
    """
    target = int(round(duration * sr))
    n = len(audio)
    if region == "full" or n <= target:
        return audio
    if region == "intro":
        return audio[:target]
    if region == "middle":
        start = (n - target) // 2
        return audio[start : start + target]
    if region == "outro":
        return audio[n - target :]
    if region == "random":
        _rng = rng if rng is not None else np.random.default_rng(seed=0)
        start = int(_rng.integers(0, n - target + 1))
        return audio[start : start + target]
    raise ValueError(f"Unknown region '{region}'. Choose from: full, intro, middle, outro, random")


def center_crop(
    audio: npt.NDArray[np.float32],
    sr: int,
    duration: float,
    pad: bool = True,
) -> npt.NDArray[np.float32]:
    """Center-crop (or pad) *audio* to exactly ``duration`` seconds.

    Cropping from the centre yields a fixed number of samples for every track,
    which in turn produces a fixed number of embedding frames and therefore a
    constant temporal-window count.  This structurally removes the duration
    confound (e.g. SONICS ``udio-30s`` tracks being shorter than reals, which
    makes window-count a trivial separator).

    Parameters
    ----------
    audio : 1-D float array
    sr : sample rate
    duration : target length in seconds
    pad : when the track is shorter than ``duration``, zero-pad symmetrically
        so the output is always exactly ``int(duration * sr)`` samples.  When
        ``False``, short tracks are returned unchanged.

    Returns
    -------
    1-D float array of length ``int(duration * sr)`` (when ``pad`` or the track
    is long enough), otherwise the original short array.
    """
    target = int(round(duration * sr))
    if target <= 0:
        return audio
    n = len(audio)
    if n == target:
        return audio
    if n > target:
        start = (n - target) // 2
        return audio[start : start + target]
    if not pad:
        return audio
    total_pad = target - n
    left = total_pad // 2
    right = total_pad - left
    return np.pad(audio, (left, right), mode="constant")


def sliding_window(
    audio: npt.NDArray[np.float32],
    sr: int,
    window_size: float = 1.0,
    hop_size: float = 0.5,
) -> list[npt.NDArray[np.float32]]:
    """Slice *audio* into overlapping frames.

    Parameters
    ----------
    audio : 1-D float array
    sr : sample rate
    window_size : frame length in seconds
    hop_size : hop between frames in seconds

    Returns
    -------
    list of 1-D arrays, each of length ``int(window_size * sr)``
    """
    win_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    frames: list[npt.NDArray[np.float32]] = []
    start = 0
    while start + win_samples <= len(audio):
        frames.append(audio[start : start + win_samples])
        start += hop_samples

    return frames


def sliding_window_chunked(
    audio: npt.NDArray[np.float32],
    sr: int,
    chunk_duration: float = 10.0,
    hop_duration: float = 5.0,
) -> list[npt.NDArray[np.float32]]:
    """Slice audio into larger chunks (e.g. for CLAP which expects ~10 s)."""
    return sliding_window(audio, sr, window_size=chunk_duration, hop_size=hop_duration)
