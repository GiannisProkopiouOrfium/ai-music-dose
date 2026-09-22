"""Audio manipulation functions matching MusicDET's Table 6 robustness protocol.

MusicDET (arXiv 2605.18072) reports EER degradation under: pitch shifting
(±2 semitones), time stretching (80-120%), equalization, reverberation, white
noise, and codec re-encoding at 64 kbps (MP3/AAC/Opus). We already have the
MP3 64kbps case (our canonical pipeline IS a 64kbps MP3 round-trip, and
run_bitrate_sweep.sh covers 32/128kbps too) — this module implements the
remaining five so we can build a directly comparable robustness table.

These are intentionally simple, standard implementations (librosa/scipy), not
tuned to any specific reference implementation, since MusicDET's paper does not
specify exact filter parameters either.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal


def pitch_shift(audio: npt.NDArray[np.float32], sr: int, n_steps: float = 2.0) -> npt.NDArray[np.float32]:
    """Shift pitch by n_steps semitones (MusicDET: random ±2 semitones)."""
    import librosa

    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(audio: npt.NDArray[np.float32], rate: float = 1.1) -> npt.NDArray[np.float32]:
    """Stretch/compress tempo by `rate` (MusicDET: random 80-120%, i.e. rate in [0.8, 1.2])."""
    import librosa

    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate).astype(np.float32)


def equalize(
    audio: npt.NDArray[np.float32], sr: int, low_shelf_db: float = -4.0, high_shelf_db: float = 4.0
) -> npt.NDArray[np.float32]:
    """Simple two-band shelving EQ: cut below 200 Hz, boost above 4 kHz.

    Implemented as two Butterworth shelf approximations (low-pass boosted/cut
    signal mixed back in), which is a standard cheap way to approximate a
    shelving filter without needing a dedicated parametric-EQ library.
    """
    nyq = sr / 2.0
    low_cut = min(200.0 / nyq, 0.99)
    high_cut = min(4000.0 / nyq, 0.99)

    b_low, a_low = scipy.signal.butter(2, low_cut, btype="low")
    low_band = scipy.signal.filtfilt(b_low, a_low, audio)
    low_gain = 10 ** (low_shelf_db / 20.0) - 1.0

    b_high, a_high = scipy.signal.butter(2, high_cut, btype="high")
    high_band = scipy.signal.filtfilt(b_high, a_high, audio)
    high_gain = 10 ** (high_shelf_db / 20.0) - 1.0

    out = audio + low_gain * low_band + high_gain * high_band
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def add_reverb(
    audio: npt.NDArray[np.float32], sr: int, decay_s: float = 0.6, wet: float = 0.35
) -> npt.NDArray[np.float32]:
    """Synthetic reverb via convolution with an exponentially-decaying noise
    impulse response (a standard cheap synthetic-reverb technique that needs
    no external impulse-response files).
    """
    n_ir = int(decay_s * sr)
    t = np.arange(n_ir) / sr
    rng = np.random.default_rng(42)
    ir = rng.standard_normal(n_ir).astype(np.float32) * np.exp(-t / (decay_s / 5.0))
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9

    wet_signal = scipy.signal.fftconvolve(audio, ir, mode="full")[: len(audio)]
    wet_signal = wet_signal / (np.max(np.abs(wet_signal)) + 1e-9) * (np.max(np.abs(audio)) + 1e-9)
    out = (1 - wet) * audio + wet * wet_signal
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def add_white_noise(audio: npt.NDArray[np.float32], snr_db: float = 20.0, seed: int = 42) -> npt.NDArray[np.float32]:
    """Add Gaussian white noise at the target SNR (dB)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(audio**2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    out = audio + noise
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


# LEGACY fixed-parameter table (kept so pre-correction runs stay reproducible).
# NOTE: MusicDET randomizes pitch over U(-2,+2) semitones and stretch over
# U(0.8, 1.2) per input; these fixed single points (+2, 1.1x) are NOT
# comparable to their Table 6 rows — use `randomized_manipulation` instead.
MANIPULATIONS = {
    "pitch_shift": lambda audio, sr: pitch_shift(audio, sr, n_steps=2.0),
    "time_stretch": lambda audio, sr: time_stretch(audio, rate=1.1),
    "equalization": lambda audio, sr: equalize(audio, sr),
    "reverberation": lambda audio, sr: add_reverb(audio, sr),
    "white_noise": lambda audio, sr: add_white_noise(audio, snr_db=20.0),
}

RANDOMIZED_MANIPULATION_NAMES = tuple(MANIPULATIONS)


def randomized_manipulation(
    name: str,
    audio: npt.NDArray[np.float32],
    sr: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], dict]:
    """Apply one manipulation with MusicDET-Table-6-style randomized parameters.

    Per-input draws (pass a per-track seeded ``rng`` for reproducibility):
      - pitch_shift : n_steps ~ U(-2, +2) semitones  (their "±2 semitones")
      - time_stretch: rate ~ U(0.8, 1.2)             (their "80–120%")
      - white_noise : fixed 20 dB SNR (unspecified in their paper), but the
        noise REALIZATION is drawn per track instead of one global seed.
      - equalization / reverberation: deterministic (their paper specifies no
        parameters; our fixed settings are documented in the functions above).

    Returns ``(manipulated_audio, params_dict)`` so the drawn parameters can be
    logged alongside each score.
    """
    if name == "pitch_shift":
        n_steps = float(rng.uniform(-2.0, 2.0))
        return pitch_shift(audio, sr, n_steps=n_steps), {"n_steps": round(n_steps, 3)}
    if name == "time_stretch":
        rate = float(rng.uniform(0.8, 1.2))
        return time_stretch(audio, rate=rate), {"rate": round(rate, 4)}
    if name == "white_noise":
        noise_seed = int(rng.integers(0, 2**31 - 1))
        return add_white_noise(audio, snr_db=20.0, seed=noise_seed), {
            "snr_db": 20.0,
            "noise_seed": noise_seed,
        }
    if name == "equalization":
        return equalize(audio, sr), {}
    if name == "reverberation":
        return add_reverb(audio, sr), {}
    raise KeyError(f"unknown manipulation: {name!r}")
