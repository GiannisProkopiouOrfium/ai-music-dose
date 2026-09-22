"""Universal-compressor complexity estimates for audio.

Serrà et al. (ICLR 2020, arXiv:1909.11480) show that likelihoods from deep
generative models are dominated by input COMPLEXITY, and propose a
parameter-free correction: a likelihood ratio against a universal lossless
compressor (they use PNG/JPEG2000/FLIF for images). The audio analog used
throughout this project is **FLAC-compressed bits per second**.

Two entry points share one implementation so every call site measures the same
quantity:
  - ``flac_bits_per_sec_file``  — for canonicalized files on disk.
  - ``flac_bits_per_sec_array`` — for in-memory waveforms (e.g. the
    reconstruction-control pairs, which are never written to disk).

Both return NaN rather than raising when ffmpeg is missing or fails, so callers
can treat complexity as an optional covariate.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _flac_bits(path: str, timeout: float = 120.0) -> float:
    """Size in bits of the FLAC-compressed audio stream."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-c:a", "flac", "-f", "flac", "-"],
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or not proc.stdout:
        return float("nan")
    return float(len(proc.stdout) * 8)


def flac_bits_per_sec_file(path: str, timeout: float = 120.0) -> float:
    """FLAC-compressed bits per second of audio for a file on disk."""
    try:
        n_bits = _flac_bits(path, timeout=timeout)
        if not np.isfinite(n_bits):
            return float("nan")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        dur = float(probe.stdout.strip())
        return n_bits / max(dur, 0.1)
    except Exception:
        return float("nan")


def flac_bits_per_sec_array(audio: np.ndarray, sr: int, timeout: float = 120.0) -> float:
    """FLAC-compressed bits per second for an in-memory mono waveform.

    The waveform is written to a temporary 16-bit PCM WAV (the same
    quantization the canonical corpus is stored at, so file-based and
    array-based measurements are comparable) and compressed with FLAC.
    """
    a = np.asarray(audio, dtype=np.float64).ravel()
    if a.size == 0 or not np.isfinite(a).any():
        return float("nan")
    peak = np.max(np.abs(a))
    if peak > 1.0:  # avoid clipping on write; FLAC size is scale-insensitive enough
        a = a / peak
    pcm = (np.nan_to_num(a) * 32767.0).astype(np.int16)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "clip.wav")
            with wave.open(p, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(sr))
                w.writeframes(pcm.tobytes())
            n_bits = _flac_bits(p, timeout=timeout)
        if not np.isfinite(n_bits):
            return float("nan")
        return n_bits / max(len(pcm) / float(sr), 0.1)
    except Exception:
        return float("nan")
