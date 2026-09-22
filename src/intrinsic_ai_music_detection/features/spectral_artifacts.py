"""Low-level spectral-artifact features for AI-music detection.

Motivation
----------
The embedding-geometry axis (intrinsic dimension + trajectory descriptors) is
strong *within* a generator family (Suno/chirp) but hits a wall on udio-120s,
which sits on the real manifold in every embedding we tried. The one orthogonal
lever left is the **low-level signal**: neural codecs and vocoders synthesise
audio with *strided transposed convolutions* (deconvolution). Zero-insertion in
upsampling is mathematically a multiplication by a Dirac comb, which in the
frequency domain clones the spectrum at multiples of ``sr / stride`` — a
periodic "comb" of peaks ("fakeprints", Afchar et al., ISMIR 2025 Best Paper:
*A Fourier Explanation of AI-music Artifacts*).

Confound discipline (why this module is *not* a naive fakeprint)
----------------------------------------------------------------
The canonical preprocessing in this repo (24 kHz, 64 kbps MP3 round-trip,
LUFS normalisation) and the 8 kHz low-pass used for embeddings were *designed*
to equalise bandwidth / bitrate / loudness so a detector cannot cheat on those.
Classic fakeprints live in the high band (5-16 kHz) and are largely destroyed by
exactly that normalisation. Two consequences shape the feature design:

1. **Band-limited, structure-based features.** We analyse only a *shared in-band*
   region (default 1.0-7.8 kHz) that both real and band-limited fakes possess,
   and we measure the *periodic comb structure* (autocorrelation of the spectral
   residual, residual kurtosis/peak count) — NOT absolute high-frequency energy,
   which would re-introduce the trivial 16 kHz-fake vs 44 kHz-real confound.

2. **Generator-invariant by construction.** Comb *presence* (``spec_comb_peak``,
   ``spec_residual_kurtosis``, ``spec_comb_peak_count``) fires for *any* strided
   codec/vocoder regardless of stride, so it should pull Suno and Udio closer for
   a single detector. Comb *period* (``spec_comb_lag_hz``) is generator-specific
   and kept only as a diagnostic / family covariate.

Each function returns plain scalar features (``spec_*``) so they merge into the
ablation feature table keyed by ``track_id`` and fuse with the geometry bundle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal, stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class SpectralArtifactConfig:
    """Configuration for spectral-artifact feature extraction.

    Defaults are tuned for the canonical 24 kHz corpus and confound safety.
    """

    analysis_sr: int = 24_000
    n_fft: int = 8_192  # fine freq resolution (~2.9 Hz/bin) for comb detection
    hop_length: int = 4_096
    band_lo_hz: float = 1_000.0  # shared in-band region (both classes have content)
    band_hi_hz: float = 7_800.0  # below 8 kHz Nyquist of band-limited fakes
    max_duration: float = 120.0  # seconds analysed
    hull_area: int = 12  # lower-hull window (bins) for residual extraction
    min_db: float = -90.0  # floor for log-power
    comb_min_lag_hz: float = 60.0  # ignore autocorr lags below this (broadband tilt)
    peak_prominence_db: float = 1.0  # residual-peak prominence threshold


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if audio.ndim == 2:
        # average channels regardless of orientation
        axis = 0 if audio.shape[0] < audio.shape[1] else 1
        audio = audio.mean(axis=axis)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _resample(audio: npt.NDArray[np.float32], sr: int, target_sr: int) -> npt.NDArray[np.float32]:
    if sr == target_sr:
        return audio
    try:
        import soxr

        return soxr.resample(audio, sr, target_sr).astype(np.float32)
    except Exception:  # pragma: no cover - fallback path
        n_out = int(round(len(audio) * target_sr / sr))
        return sp_signal.resample(audio, n_out).astype(np.float32)


def _power_spectrogram(
    audio: npt.NDArray[np.float32], cfg: SpectralArtifactConfig
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (power[freq, time], freqs) using a Hann-windowed STFT."""
    n_fft = cfg.n_fft
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    freqs, _, zxx = sp_signal.stft(
        audio,
        fs=cfg.analysis_sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - cfg.hop_length,
        boundary=None,
        padded=False,
    )
    power = (np.abs(zxx) ** 2).astype(np.float64)  # [freq, time]
    return power, freqs.astype(np.float64)


def _lower_hull(x: npt.NDArray[np.float64], area: int) -> npt.NDArray[np.float64]:
    """Lower-envelope of a 1-D curve via sliding-window minima + interpolation."""
    n = len(x)
    if n <= area:
        return np.full_like(x, float(np.min(x)))
    idx: list[int] = []
    val: list[float] = []
    for i in range(n - area + 1):
        j = i + int(np.argmin(x[i : i + area]))
        if not idx or j != idx[-1]:
            idx.append(j)
            val.append(float(x[j]))
    if idx[0] != 0:
        idx.insert(0, 0)
        val.insert(0, float(x[0]))
    if idx[-1] != n - 1:
        idx.append(n - 1)
        val.append(float(x[n - 1]))
    return np.interp(np.arange(n), np.asarray(idx), np.asarray(val))


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------


def _comb_features(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    cfg: SpectralArtifactConfig,
) -> dict[str, float]:
    """Periodic-comb signature of the in-band spectral residual.

    The autocorrelation of the residual peaks strongly when the residual is a
    regular comb (codec/vocoder deconvolution). Real recordings give a low,
    rapidly-decaying autocorrelation.
    """
    r = residual - residual.mean()
    denom = float(np.dot(r, r))
    if denom <= 1e-12 or len(r) < 8:
        return {
            "spec_comb_peak": 0.0,
            "spec_comb_lag_hz": float("nan"),
            "spec_comb_peak_count": 0.0,
            "spec_comb_ac_decay": float("nan"),
        }
    ac = np.correlate(r, r, mode="full")[len(r) - 1 :]
    ac = ac / denom  # ac[0] == 1
    min_lag = max(2, int(round(cfg.comb_min_lag_hz / bin_hz)))
    if min_lag >= len(ac) - 1:
        comb_peak, comb_lag = 0.0, float("nan")
    else:
        tail = ac[min_lag:]
        k = int(np.argmax(tail))
        comb_peak = float(tail[k])
        comb_lag = float((min_lag + k) * bin_hz)

    # count prominent peaks in the residual itself
    prom = cfg.peak_prominence_db
    peaks, _ = sp_signal.find_peaks(residual, prominence=prom)
    # autocorrelation decay: mean |ac| over the searched lag range (low = noise-like)
    ac_decay = float(np.mean(np.abs(ac[min_lag:]))) if min_lag < len(ac) else float("nan")

    return {
        "spec_comb_peak": _safe(comb_peak),
        "spec_comb_lag_hz": _safe(comb_lag),
        "spec_comb_peak_count": float(len(peaks)),
        "spec_comb_ac_decay": _safe(ac_decay),
    }


def _residual_features(residual: npt.NDArray[np.float64]) -> dict[str, float]:
    if residual.size == 0:
        return {
            k: float("nan")
            for k in (
                "spec_residual_mean",
                "spec_residual_std",
                "spec_residual_max",
                "spec_residual_kurtosis",
                "spec_residual_energy_frac",
            )
        }
    total = float(np.sum(np.abs(residual))) + 1e-12
    return {
        "spec_residual_mean": _safe(float(np.mean(residual))),
        "spec_residual_std": _safe(float(np.std(residual))),
        "spec_residual_max": _safe(float(np.max(residual))),
        "spec_residual_kurtosis": _safe(float(sp_stats.kurtosis(residual, fisher=True))),
        "spec_residual_energy_frac": _safe(float(np.max(residual) / total)),
    }


def _shape_features(
    profile_lin: npt.NDArray[np.float64],
    profile_db: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Static spectral shape within the analysed band (time-averaged)."""
    p = np.clip(profile_lin, 1e-12, None)
    total = float(np.sum(p))
    centroid = float(np.sum(freqs * p) / total)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p) / total))
    cum = np.cumsum(p) / total
    rolloff85 = float(freqs[int(np.searchsorted(cum, 0.85))]) if cum[-1] >= 0.85 else float(freqs[-1])
    rolloff95 = float(freqs[int(np.searchsorted(cum, 0.95))]) if cum[-1] >= 0.95 else float(freqs[-1])
    geo = float(np.exp(np.mean(np.log(p))))
    flatness = float(geo / (np.mean(p) + 1e-12))
    # log-spectral slope (dB per kHz) via least squares
    slope = float(np.polyfit(freqs / 1000.0, profile_db, 1)[0])
    mid = len(p) // 2
    hf_lf_ratio = float(np.sum(p[mid:]) / (np.sum(p[:mid]) + 1e-12))
    return {
        "spec_centroid_hz": _safe(centroid),
        "spec_bandwidth_hz": _safe(spread),
        "spec_rolloff85_hz": _safe(rolloff85),
        "spec_rolloff95_hz": _safe(rolloff95),
        "spec_flatness": _safe(flatness),
        "spec_slope_db_khz": _safe(slope),
        "spec_hf_lf_ratio": _safe(hf_lf_ratio),
    }


def _dynamics_features(
    band_power: npt.NDArray[np.float64],
    freqs_band: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Temporal variability of the in-band spectrum across frames.

    Our leak-free analysis found the shared AI signature is *unsteadiness* of the
    trajectory (traj_vel_std / traj_acc_std). These features are the spectral
    analogue: how much the in-band spectral shape and energy fluctuate over time.
    """
    n_freq, n_time = band_power.shape
    if n_time < 3:
        return {
            k: float("nan")
            for k in (
                "spec_centroid_std",
                "spec_flatness_std",
                "spec_rolloff_std",
                "spec_flux_mean",
                "spec_flux_std",
            )
        }
    p = np.clip(band_power, 1e-12, None)
    fcol = freqs_band[:, None]
    colsum = np.sum(p, axis=0) + 1e-12
    centroid_t = np.sum(fcol * p, axis=0) / colsum
    geo_t = np.exp(np.mean(np.log(p), axis=0))
    flat_t = geo_t / (np.mean(p, axis=0) + 1e-12)
    cum = np.cumsum(p, axis=0) / colsum[None, :]
    roll_idx = np.argmax(cum >= 0.85, axis=0)
    rolloff_t = freqs_band[np.clip(roll_idx, 0, n_freq - 1)]
    # spectral flux: L2 norm of frame-to-frame magnitude change (normalised)
    mag = np.sqrt(p)
    mag = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + 1e-12)
    flux = np.linalg.norm(np.diff(mag, axis=1), axis=0)
    return {
        "spec_centroid_std": _safe(float(np.std(centroid_t))),
        "spec_flatness_std": _safe(float(np.std(flat_t))),
        "spec_rolloff_std": _safe(float(np.std(rolloff_t))),
        "spec_flux_mean": _safe(float(np.mean(flux))),
        "spec_flux_std": _safe(float(np.std(flux))),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spectral_artifact_features(
    audio: npt.NDArray[np.float32],
    sr: int,
    cfg: SpectralArtifactConfig | None = None,
) -> dict[str, float]:
    """Compute scalar ``spec_*`` features for a mono audio signal.

    Parameters
    ----------
    audio : float array, mono or multi-channel
    sr : sample rate of ``audio``
    cfg : configuration (defaults tuned for the 24 kHz canonical corpus)
    """
    if cfg is None:
        cfg = SpectralArtifactConfig()

    audio = _to_mono(np.asarray(audio, dtype=np.float32))
    audio = _resample(audio, sr, cfg.analysis_sr)
    max_samples = int(cfg.analysis_sr * cfg.max_duration)
    if max_samples and len(audio) > max_samples:
        audio = audio[:max_samples]
    if len(audio) < cfg.n_fft:
        audio = np.pad(audio, (0, cfg.n_fft - len(audio)))

    power, freqs = _power_spectrogram(audio, cfg)  # [freq, time], [freq]
    band = (freqs >= cfg.band_lo_hz) & (freqs <= cfg.band_hi_hz)
    if band.sum() < 8:
        logger.warning("spectral_artifact_features: band too narrow (%d bins)", int(band.sum()))
        return {}

    band_power = power[band]  # [band_freq, time]
    freqs_band = freqs[band]
    bin_hz = float(freqs[1] - freqs[0])

    profile_lin = band_power.mean(axis=1)  # time-averaged power, in-band
    profile_db = 10.0 * np.log10(np.clip(profile_lin, 10 ** (cfg.min_db / 10.0), None))
    hull_db = _lower_hull(profile_db, cfg.hull_area)
    residual = np.clip(profile_db - hull_db, 0.0, None)  # peaks above local floor

    feats: dict[str, float] = {}
    feats.update(_comb_features(residual, bin_hz, cfg))
    feats.update(_residual_features(residual))
    feats.update(_shape_features(profile_lin, profile_db, freqs_band))
    feats.update(_dynamics_features(band_power, freqs_band))
    return feats


def spectral_artifact_features_from_file(
    path: str,
    cfg: SpectralArtifactConfig | None = None,
) -> dict[str, float]:
    """Load an audio file (any format) and compute its ``spec_*`` features."""
    import librosa

    if cfg is None:
        cfg = SpectralArtifactConfig()
    audio, sr = librosa.load(
        path,
        sr=cfg.analysis_sr,
        mono=True,
        duration=cfg.max_duration if cfg.max_duration else None,
    )
    return spectral_artifact_features(audio.astype(np.float32), int(sr), cfg)
