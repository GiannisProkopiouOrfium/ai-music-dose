"""Unified audio preprocessing pipeline.

Two modes
---------
raw
    Legacy baseline: decode → mono → resample → peak-normalise.
    Used for confound-quantification runs only.

canonical
    Removes the format-confound between real YouTube audio (high-quality
    m4a → mp3) and SONICS fake audio (MP3 ~37 kbps, 8 kHz band-limited):

      1. Decode to float32 PCM (librosa; handles m4a / mp3 / wav / ogg)
      2. Convert to mono, resample to ``target_sr``
      3. Apply a consistent duration cap (``max_duration``)
      4. MP3 round-trip via ``ffmpeg`` at ``mp3_bitrate_kbps`` — makes both
         sides share the same codec bandwidth and quantisation noise floor
         without adding neural fakeprint artefacts (plain MP3 uses
         sub-band coding, not strided deconvolution)
      5. LUFS-normalise to ``target_lufs`` dB (ITU-R BS.1770-3) so that
         loudness is not a confound variable

Measurements taken *before* normalisation are stored in the returned
``PreprocessMetadata`` dict for downstream analysis and balance auditing.

Usage
-----
from intrinsic_ai_music_detection.data.audio_preprocessing import preprocess_audio

audio, sr, meta = preprocess_audio(path, mode="canonical")
audio, sr, meta = preprocess_audio(path, mode="raw")
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.io.wavfile
import scipy.signal

logger = logging.getLogger(__name__)

PreprocessMode = Literal["raw", "canonical"]

# --- defaults ---
CANONICAL_TARGET_SR: int = 24_000  # MERT / EnCodec native → no extra resample
CANONICAL_MP3_KBPS: int = 64  # ≈ fakes' bitrate ceiling
CANONICAL_TARGET_LUFS: float = -23.0  # EBU R128 broadcast standard
CANONICAL_MAX_DURATION: float = 120.0  # seconds


# ---------------------------------------------------------------------------
# LUFS measurement + normalisation (requires pyloudnorm)
# ---------------------------------------------------------------------------


def measure_loudness_lufs(audio: npt.NDArray[np.float32], sr: int) -> float:
    """Measure integrated loudness in LUFS (ITU-R BS.1770-3).

    Returns ``float('nan')`` when ``pyloudnorm`` is unavailable or the
    signal is too quiet to measure reliably.
    """
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        return float(loudness)
    except ImportError:
        logger.debug("pyloudnorm not installed; LUFS measurement unavailable")
        return float("nan")
    except Exception as exc:  # noqa: BLE001
        logger.debug("LUFS measurement failed: %s", exc)
        return float("nan")


def normalise_loudness(
    audio: npt.NDArray[np.float32],
    sr: int,
    target_lufs: float = CANONICAL_TARGET_LUFS,
) -> npt.NDArray[np.float32]:
    """Normalise integrated loudness to ``target_lufs`` dB.

    Falls back to peak-normalisation when ``pyloudnorm`` is unavailable
    or the signal is silent / too quiet (< -70 LUFS).
    """
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isfinite(loudness) or loudness < -70.0:
            logger.debug("Signal too quiet for LUFS normalisation (%.1f); peak-normalising", loudness)
            peak = float(np.abs(audio).max())
            return (audio / peak).astype(np.float32) if peak > 1e-8 else audio
        normalised = pyln.normalize.loudness(audio.astype(np.float64), loudness, target_lufs)
        # Hard-clamp after potentially large gain
        return np.clip(normalised, -1.0, 1.0).astype(np.float32)
    except ImportError:
        logger.warning(
            "pyloudnorm not installed. Install with: pip install pyloudnorm. " "Falling back to peak normalisation."
        )
        peak = float(np.abs(audio).max())
        return (audio / peak).astype(np.float32) if peak > 1e-8 else audio
    except Exception as exc:  # noqa: BLE001
        logger.warning("LUFS normalisation failed (%s); peak-normalising", exc)
        peak = float(np.abs(audio).max())
        return (audio / peak).astype(np.float32) if peak > 1e-8 else audio


# ---------------------------------------------------------------------------
# MP3 round-trip via ffmpeg
# ---------------------------------------------------------------------------


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def mp3_roundtrip(
    audio: npt.NDArray[np.float32],
    sr: int,
    bitrate_kbps: int = CANONICAL_MP3_KBPS,
) -> npt.NDArray[np.float32]:
    """Encode audio to MP3 then decode back via ffmpeg.

    This equalises codec bandwidth and quantisation noise between the
    real (high-quality) and fake (low-bitrate) classes.

    Both sides pass through identical compression so that the resulting
    audios share the same ~8 kHz bandwidth ceiling and quantisation floor
    as the SONICS fake songs — removing the format advantage without
    adding neural-vocoder fakeprint artefacts.

    Requires ``ffmpeg`` on PATH.  If ffmpeg is unavailable the original
    audio is returned unchanged with a warning.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_wav = Path(tmp_dir) / "input.wav"
            tmp_mp3 = Path(tmp_dir) / "output.mp3"
            tmp_out = Path(tmp_dir) / "decoded.wav"

            # Write PCM-16 WAV
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            scipy.io.wavfile.write(str(tmp_wav), sr, audio_int16)

            # Encode to MP3
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_wav), "-b:a", f"{bitrate_kbps}k", str(tmp_mp3)],
                capture_output=True,
                check=True,
                timeout=120,
            )

            # Decode back to mono WAV at target SR
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tmp_mp3),
                    "-ar",
                    str(sr),
                    "-ac",
                    "1",
                    str(tmp_out),
                ],
                capture_output=True,
                check=True,
                timeout=120,
            )

            out_sr, out_data = scipy.io.wavfile.read(str(tmp_out))
            if out_data.dtype == np.int16:
                out_f32 = out_data.astype(np.float32) / 32767.0
            elif out_data.dtype == np.int32:
                out_f32 = out_data.astype(np.float32) / 2147483647.0
            else:
                out_f32 = out_data.astype(np.float32)

            # Trim to match original length (codec may add a few padding samples)
            expected = len(audio)
            if len(out_f32) > expected:
                out_f32 = out_f32[:expected]

            return out_f32

    except FileNotFoundError:
        logger.warning("ffmpeg not found — MP3 round-trip skipped. Install ffmpeg for canonical mode.")
        return audio
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg error during MP3 round-trip: %s — skipping", exc.stderr[:200] if exc.stderr else exc)
        return audio
    except Exception as exc:  # noqa: BLE001
        logger.warning("MP3 round-trip failed unexpectedly (%s) — skipping", exc)
        return audio


# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------


def lowpass_filter(
    audio: npt.NDArray[np.float32],
    sr: int,
    cutoff_hz: float,
    order: int = 8,
) -> npt.NDArray[np.float32]:
    """Apply a Butterworth low-pass filter at ``cutoff_hz`` Hz.

    Used to equalise spectral bandwidth between classes that differ in their
    original recording/generation bandwidths (e.g. YouTube reals vs 8 kHz-limited
    SONICS fakes).  Applying the same filter to both sides removes the bandwidth
    shortcut so detectors must rely on other cues.

    Parameters
    ----------
    audio   float32 mono array
    sr      sample rate in Hz
    cutoff_hz  -3 dB frequency of the low-pass
    order   filter order (default 8 — steep roll-off, minimal passband ripple)
    """
    nyq = sr / 2.0
    if cutoff_hz >= nyq:
        logger.debug("lowpass_filter: cutoff %.0f Hz >= Nyquist %.0f Hz — skipping", cutoff_hz, nyq)
        return audio
    sos = scipy.signal.butter(order, cutoff_hz / nyq, btype="low", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, audio.astype(np.float64))
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)


def level_match(
    audio: npt.NDArray[np.float32],
    sr: int,
    remove_dc: bool = True,
    trim_silence_dbfs: float | None = -60.0,
    peak_normalise: bool = False,
    hp_hz: float = 20.0,
) -> npt.NDArray[np.float32]:
    """Remove level and silence cues that no bandwidth control can touch.

    MEASURED 2026-08-13 under ``resample_hz=15000`` (i.e. with the bandwidth
    confound already closed), best macro |AUC| of a single descriptor used alone:

    | descriptor | SONICS | FakeMusicCaps |
    |---|---|---|
    | ``silence_frac``   | **0.7773** | 0.5965 |
    | ``dc_offset``      | 0.7709 (**0.9976** on chirp-v2) | **0.7470** |
    | ``lead_silence_s`` | 0.7708 | 0.5470 |
    | ``peak_dbfs``      | 0.6820 | 0.6470 |

    None of these are synthesis artifacts:

    * **DC offset** at 0.9976 on one generator is a preprocessing defect, not a
      finding — canonical audio should not carry DC at all. An MP3 decode and a
      loudness gain can both introduce it, and it survives every spectral
      control because it lives at 0 Hz.
    * **Leading/trailing silence** is a decoder-buffer tell: generators emit a
      fixed-length buffer, mastered music does not. It is trivially learnable and
      says nothing about how the audio was made.
    * **Peak / crest** differ because generated audio never passed a mastering
      limiter.

    Applied BEFORE loudness normalisation and bandwidth matching, since trimming
    silence changes the integrated loudness of the remainder.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return a

    if remove_dc:
        # High-pass rather than mean subtraction: DC in decoded audio is often a
        # slow drift, not a constant, and subtracting the mean leaves the drift.
        nyq = sr / 2.0
        if 0.0 < hp_hz < nyq:
            sos = scipy.signal.butter(2, hp_hz / nyq, btype="high", output="sos")
            a = scipy.signal.sosfiltfilt(sos, a.astype(np.float64)).astype(np.float32)
        else:
            a = (a - float(np.mean(a))).astype(np.float32)

    if trim_silence_dbfs is not None:
        frame = 1024
        n_frames = len(a) // frame
        if n_frames >= 2:
            frames = a[: n_frames * frame].reshape(n_frames, frame)
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            loud = rms >= (10.0 ** (trim_silence_dbfs / 20.0))
            if loud.any():
                first, last = int(np.argmax(loud)), n_frames - int(np.argmax(loud[::-1]))
                trimmed = a[first * frame : last * frame]
                # A track that is silent by this threshold is left alone rather
                # than reduced to nothing: an empty array downstream would be
                # dropped, and a class-skewed drop rate silently rebalances the
                # comparison (§8.4).
                if len(trimmed) >= frame:
                    a = trimmed

    if peak_normalise:
        peak = float(np.abs(a).max())
        if peak > 1e-8:
            a = (a / peak).astype(np.float32)

    return np.asarray(a, dtype=np.float32)


def bandwidth_match(
    audio: npt.NDArray[np.float32],
    sr: int,
    resample_hz: float | None = None,
    lowpass_hz: float | None = None,
) -> npt.NDArray[np.float32]:
    """Force both classes onto one shared bandwidth ceiling.

    MEASURED 2026-08-13, and this is why ``resample_hz`` exists:

    | control on canonical SONICS | best channel-alone AUC |
    |---|---|
    | none                        | **1.0000** (`frac_power_8k_11k`) |
    | ``lowpass_hz=7000``         | **0.9287** (`spectral_flatness_hf`) |
    | ``resample_hz=14000``       | **0.5766** |

    A Butterworth low-pass does **not** equalise bandwidth, it only attenuates.
    Pushing a real track 20 dB down leaves a *structured* residual where a
    band-limited fake has a *flat noise floor*, and spectral flatness reads that
    difference almost as well as the raw energy ratio did — the leak moves from
    one descriptor to another rather than closing. Every historical
    ``--lowpass-hz 8000`` "bandwidth recheck" in this project is therefore not a
    bandwidth control, and conclusions drawn from those runs do not follow.

    Decimating to ``resample_hz`` and returning to ``sr`` removes the band
    outright: above ``resample_hz/2`` there is nothing for either class, so there
    is nothing left to differ in.

    The cost is real and must be stated wherever this is used: the discarded band
    is also where the artifact literature places its evidence (Afchar &
    Hennequin's fakeprints at 3-15 kHz). On these corpora that band carries the
    channel label, so it cannot be used as evidence without controlling it first.

    Applied in this order: decimate, then optionally low-pass — so the low-pass
    acts on already-band-limited audio and cannot re-introduce a transition-band
    difference.
    """
    if resample_hz:
        import librosa

        target = int(resample_hz)
        if target < sr:
            low = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target, res_type="soxr_vhq")
            audio = librosa.resample(low, orig_sr=target, target_sr=sr, res_type="soxr_vhq")
        else:
            logger.debug(
                "bandwidth_match: resample_hz %d >= sr %d — no band to remove, skipping",
                target,
                sr,
            )
    if lowpass_hz:
        audio = lowpass_filter(np.asarray(audio, dtype=np.float32), sr, cutoff_hz=float(lowpass_hz))
    return np.asarray(audio, dtype=np.float32)


def preprocess_audio(
    path: str | Path,
    target_sr: int = CANONICAL_TARGET_SR,
    mode: PreprocessMode = "canonical",
    max_duration: float = CANONICAL_MAX_DURATION,
    mp3_bitrate_kbps: int = CANONICAL_MP3_KBPS,
    target_lufs: float = CANONICAL_TARGET_LUFS,
    lowpass_hz: float | None = None,
    resample_hz: float | None = None,
) -> tuple[npt.NDArray[np.float32], int, dict[str, Any]]:
    """Load and preprocess one audio file.

    Parameters
    ----------
    path
        Audio file path (any format supported by librosa / ffmpeg).
    target_sr
        Output sample rate.
    mode
        ``"canonical"`` applies MP3 round-trip + LUFS normalisation.
        ``"raw"`` applies only decode → mono → resample → peak-normalise.
    max_duration
        Truncate to this many seconds (applied during decode).
    mp3_bitrate_kbps
        Canonical-mode only. Target MP3 bitrate in kbps.
    target_lufs
        Canonical-mode only. Target integrated loudness in LUFS.
    lowpass_hz
        If set, apply a Butterworth low-pass filter at this frequency (Hz) as the
        **last** step, applied to **both** modes.  ⚠️ This is NOT an adequate
        bandwidth control on its own: measured on canonical SONICS it leaves a
        channel-alone AUC of 0.9287 (down from 1.0000, but the leak simply moves
        to spectral flatness). Use ``resample_hz``.
    resample_hz
        If set, decimate to this rate and return to ``target_sr``, so the band
        above ``resample_hz/2`` does not exist for either class. This IS an
        adequate control (0.5766 on canonical SONICS at 14000). Applied before
        ``lowpass_hz``. See ``bandwidth_match``.

    Returns
    -------
    audio : float32 mono ndarray at ``target_sr``
    sr : int — always equal to ``target_sr``
    metadata : dict with diagnostic measurements
    """
    import librosa  # local import to keep module importable without heavy deps

    path = Path(path)
    meta: dict[str, Any] = {
        "mode": mode,
        "original_path": str(path),
        "target_sr": target_sr,
        "max_duration": max_duration,
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        audio, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=True,
            duration=max_duration if max_duration else None,
        )

    audio = audio.astype(np.float32)
    meta["original_sr"] = sr
    meta["actual_duration"] = float(len(audio) / sr)

    # Measure LUFS / peak before any processing
    meta["measured_lufs"] = measure_loudness_lufs(audio, sr)
    meta["measured_peak"] = float(np.abs(audio).max())

    if mode == "raw":
        peak = meta["measured_peak"]
        if peak > 1e-8:
            audio = audio / peak
        if lowpass_hz or resample_hz:
            audio = bandwidth_match(audio, sr, resample_hz=resample_hz, lowpass_hz=lowpass_hz)
            meta["lowpass_hz"] = lowpass_hz
            meta["resample_hz"] = resample_hz
        return audio, sr, meta

    # ---- canonical mode ----
    meta["mp3_bitrate_kbps"] = mp3_bitrate_kbps
    meta["target_lufs"] = target_lufs

    # MP3 round-trip
    audio = mp3_roundtrip(audio, sr, bitrate_kbps=mp3_bitrate_kbps)

    # LUFS normalisation
    audio = normalise_loudness(audio, sr, target_lufs=target_lufs)

    # Optional bandwidth equalisation between classes. NOTE: a low-pass alone is
    # NOT sufficient — see bandwidth_match's measured table.
    if lowpass_hz or resample_hz:
        audio = bandwidth_match(audio, sr, resample_hz=resample_hz, lowpass_hz=lowpass_hz)
        meta["lowpass_hz"] = lowpass_hz
        meta["resample_hz"] = resample_hz

    meta["actual_duration_after"] = float(len(audio) / sr)
    return audio, sr, meta
