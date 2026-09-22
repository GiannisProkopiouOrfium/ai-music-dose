"""Fourier fakeprint baseline for AI music detection.

Adapted from deezer/ismir25-ai-music-detector (CC BY-NC 4.0, research only):
https://github.com/deezer/ismir25-ai-music-detector

Reference: "A Fourier Explanation of AI-music Artifacts"
           Afchar, Meseguer Brocal, Akesbi, Hennequin — ISMIR 2025 Best Paper.

The method detects spectral artifacts from strided deconvolution layers
(transposed convolutions) used in neural audio codecs/vocoders. These
artifacts manifest as periodic peaks in the mean spectrogram, forming
a fractal-like pattern across frequencies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soxr
import torch
import torchaudio

from intrinsic_ai_music_detection.config import FakeprintConfig

logger = logging.getLogger(__name__)


def open_audio(path: str | Path) -> tuple[npt.NDArray[np.float32], int]:
    """Load audio file and return as numpy array + sample rate."""
    audio, sr = torchaudio.load(str(path), channels_first=False)
    return audio.numpy().astype(np.float32), int(sr)


def compute_spectrogram(
    audio: npt.NDArray[np.float32],
    sr: int,
    cfg: FakeprintConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Compute power spectrogram in dB.

    Parameters
    ----------
    audio : array of shape ``[samples]`` or ``[samples, channels]``
    sr : sample rate of the input audio
    cfg : fakeprint configuration

    Returns
    -------
    spectrogram : array of shape ``[channels, freq_bins, time_frames]`` in dB
    """
    if cfg is None:
        cfg = FakeprintConfig()

    target_sr = cfg.sample_rate

    # Resample if needed
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr).astype(np.float32)

    # Truncate to max_duration
    max_samples = target_sr * cfg.max_duration
    audio = audio[:max_samples]

    # Ensure 2D: [samples, channels]
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    stft_transform = torchaudio.transforms.Spectrogram(n_fft=cfg.n_fft, power=2)
    stft = stft_transform(torch.from_numpy(audio.T)).numpy()
    stft = 10.0 * np.log10(np.clip(stft, 1e-10, 1e6))
    return stft


def lower_hull(x: npt.NDArray[np.float64], area: int = 10) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Compute the lower convex hull of a 1-D signal.

    Slides a window of size *area* and picks the local minima.
    Returns indices and values of hull points.
    """
    idx: list[int] = []
    hull: list[float] = []

    for i in range(len(x) - area + 1):
        patch = x[i : i + area]
        rel_idx = int(np.argmin(patch))
        abs_idx = rel_idx + i
        if abs_idx not in idx:
            idx.append(abs_idx)
            hull.append(float(patch[rel_idx]))

    # Ensure endpoints are included
    if idx[0] != 0:
        idx.insert(0, 0)
        hull.insert(0, float(x[0]))
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1)
        hull.append(float(x[-1]))

    return np.array(idx), np.array(hull)


def curve_profile(
    x_freq: npt.NDArray[np.float64],
    curve: npt.NDArray[np.float64],
    f_range: tuple[int, int] = (5000, 16000),
    min_db: float = -45.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract the residual peak profile above the lower hull.

    Parameters
    ----------
    x_freq : frequency axis values
    curve : mean spectral magnitude values (dB)
    f_range : (min_freq, max_freq) to analyse
    min_db : clamp floor for the hull

    Returns
    -------
    x_cut : frequency values in range
    residual : spectral residual above the hull (peaks only)
    """
    from scipy import interpolate

    mask = (f_range[0] < x_freq) & (x_freq < f_range[1])
    x_cut = x_freq[mask]
    c_cut = curve[mask]

    hull_idx, hull_vals = lower_hull(c_cut, area=10)
    hull_curve = interpolate.interp1d(x_cut[hull_idx], hull_vals, kind="quadratic")(x_cut)
    hull_curve = np.clip(hull_curve, min_db, None)

    return x_cut, np.clip(c_cut - hull_curve, 0, None)


def max_normalise(x: npt.NDArray[np.float64], max_db: float = 5.0) -> npt.NDArray[np.float64]:
    """Clip and normalise a spectral residual to [0, 1]."""
    x = np.clip(x, 0, max_db)
    return x / (1e-6 + np.max(x))


def compute_fakeprint(
    audio: npt.NDArray[np.float32],
    sr: int,
    cfg: FakeprintConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Compute the fakeprint feature vector for an audio signal.

    Parameters
    ----------
    audio : 1-D or 2-D float audio
    sr : sample rate
    cfg : configuration

    Returns
    -------
    fakeprint : 1-D array — the normalised spectral residual vector
    """
    if cfg is None:
        cfg = FakeprintConfig()

    stft = compute_spectrogram(audio, sr, cfg)

    # Mean across channels and time → 1-D spectral profile
    mean_spec = np.mean(stft, axis=(0, 2))

    # Frequency axis
    freq_axis = np.linspace(0, cfg.sample_rate / 2, num=len(mean_spec))

    _, residual = curve_profile(freq_axis, mean_spec, f_range=(cfg.f_min, cfg.f_max))
    return max_normalise(residual)


def compute_fakeprint_from_file(
    path: str | Path,
    cfg: FakeprintConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Load an audio file and compute its fakeprint."""
    audio, sr = open_audio(path)
    return compute_fakeprint(audio, sr, cfg)
