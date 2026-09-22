"""GATE: quantify how much of a corpus's real/fake separability is CHANNEL, not synthesis.

Why this exists
---------------
Every reported number in this project is measured on audio that reached us through
a delivery chain, and the two classes did not travel the same chain:

* FakeMusicCaps ships **every** generated clip resampled to 16 kHz mono
  (arXiv:2409.10684), while its real side is a YouTube pull at 44.1/48 kHz.
* This repo's own canonicalisation docstring records SONICS fakes as
  "MP3 ~37 kbps, 8 kHz band-limited" against high-quality real m4a.

A detector that notices "this file has no energy above 8 kHz" is measuring the
*channel*, not the generator. Our canonicalisation (24 kHz + MP3-64k) does NOT
close that gap: LAME at 64 kbps / 24 kHz keeps content to roughly 10-11 kHz, i.e.
comfortably ABOVE an 8 kHz ceiling, so the discriminative band survives intact.

Worse, the log-mel front end maps an empty band to a single exact constant
(``np.log(mel + 1e-10)`` = -23.03 for zero energy), which makes "band is empty"
a near-deterministic feature rather than a noisy one.

This script measures the confound directly, BEFORE any flow is trained on the
corpus, and answers one question per generator:

    What AUC does a single channel descriptor reach on its own?

If that number is high, every AUC measured on that corpus without a shared
low-pass is partly a channel measurement and must be reported alongside its
bandwidth-matched twin.

Three modes
-----------
``--mode inventory``  (cheap, no DSP)
    Which path columns exist in the manifest, how many resolve on disk, and the
    container-level sample rate / codec / bitrate per class via ffprobe. This is
    the gate for the fakeprint artifact branch: Afchar & Hennequin's 3-15 kHz
    descriptor is undefined on 16 kHz audio, so if the raw sources are already
    band-limited the branch cannot be built as published.

``--mode profile``  (default; decodes audio)
    Per-track channel descriptors + per-generator AUC of each one used ALONE.

``--mode both``

Descriptors
-----------
``cutoff_hz``
    Estimated low-pass cutoff: the highest frequency whose local mean spectrum is
    still within ``--cutoff-drop-db`` of an in-band reference level. Unlike the
    90 %-cumulative-energy ``effective_bandwidth_hz`` already in
    ``profile_covariates.py`` (which is dominated by bass energy and barely moves
    when a 10 kHz cliff appears), this detects the cliff itself.
``frac_power_above_8k`` / ``frac_power_8k_11k``
    Energy fractions in exactly the band the hypothesis is about. ``8k_11k`` is
    the band that survives MP3-64k @ 24 kHz but is absent from 16 kHz-native
    audio -- the specific leak.
``hf_floor_frac``
    Fraction of STFT bins above 8 kHz sitting at the numerical floor. This is the
    quantity the log-mel front end turns into a constant.
``spectral_flatness_hf``
    Flatness above 8 kHz: separates "band-limited" (flat, at floor) from
    "genuinely quiet up there" (structured).

Usage (EC2)
-----------
    # Gate G2 -- raw-audio inventory, seconds, no decoding
    python scripts/diagnose_channel_confound.py \
        --manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
        --mode inventory --out-dir reports/diagnostics/channel_sonics

    # Gate G1 -- channel profile of the audio the flow actually sees
    python scripts/diagnose_channel_confound.py \
        --manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
        --audio-column canonical_path --per-stratum 400 --workers 6 \
        --out-dir reports/diagnostics/channel_sonics
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("diagnose_channel_confound")

# Candidate columns holding a path to the ORIGINAL (pre-canonicalisation) file,
# in priority order. Manifests in this repo have used several names; `src_path`
# is what build_canonical_corpus.py actually writes. Note that the SONICS
# manifest also carries `delete_src_after`, so a resolvable column does not
# guarantee the files still exist — that is exactly what the inventory measures.
RAW_PATH_COLUMNS = (
    "src_path",
    "original_path",
    "source_path",
    "raw_path",
    "audio_path",
    "filepath",
    "path",
)
CANON_PATH_COLUMNS = ("canonical_path",)

# Descriptors that are, on their own, pure channel measurements. Reported with
# BOTH orientations because which class is band-limited differs per corpus:
# in SONICS the fakes are, in a YouTube-vs-16 kHz corpus so are the fakes, but a
# future corpus could invert it and a one-sided AUC would hide that.
DESCRIPTORS = (
    # --- bandwidth / delivery chain ---
    "cutoff_hz",
    "frac_power_above_8k",
    "frac_power_8k_11k",
    "frac_power_top_quarter",
    "hf_floor_frac",
    "spectral_flatness_hf",
    # --- level and silence structure ---
    # Generated audio very often carries digital silence at its head or tail
    # (the decoder emits a fixed-length buffer), exact-zero samples, a DC offset
    # from an untrimmed decoder, or a different crest factor because it never
    # went through a mastering chain. Any one of these can reach a high AUC on
    # its own while having nothing to do with synthesis artifacts, and none of
    # them are removed by a bandwidth control -- so they have to be measured
    # separately or a "bandwidth-matched" number can still be a corpus artifact.
    "silence_frac",
    "lead_silence_s",
    "tail_silence_s",
    "exact_zero_frac",
    "dc_offset",
    "peak_dbfs",
    "crest_factor_db",
    "duration_s",
    # --- harmonicity (gate C9) ---
    # A sustained note's harmonic series IS a comb. Without these, C9 can only
    # report SKIP, and "untested" is not "passed".
    "harmonicity",
    "pitch_salience",
    "spectral_peak_count",
)

ANALYSIS_SR = 48_000  # analyse at the native-ish rate; never upsample past the source
N_FFT = 4096


# ---------------------------------------------------------------------------
# Container-level probe (no decoding)
# ---------------------------------------------------------------------------


def ffprobe_stream(path: Path) -> dict:
    """Return container/stream metadata via ffprobe, or ``{}`` when unavailable.

    Deliberately tolerant: ffprobe missing, file missing, or a stream without a
    declared bitrate must all degrade to a partial row rather than kill the run.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,codec_name,bit_rate,channels",
                "-show_entries",
                "format=bit_rate,format_name",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
    except FileNotFoundError:
        return {"probe_error": "ffprobe not on PATH"}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": str(exc)[:120]}

    try:
        payload = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return {"probe_error": "unparseable ffprobe output"}

    streams = payload.get("streams") or [{}]
    stream, fmt = streams[0], payload.get("format", {})

    def _int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "native_sr": _int(stream.get("sample_rate")),
        "codec": stream.get("codec_name"),
        "channels": _int(stream.get("channels")),
        "stream_bitrate": _int(stream.get("bit_rate")),
        "format_bitrate": _int(fmt.get("bit_rate")),
        "container": fmt.get("format_name"),
    }


# ---------------------------------------------------------------------------
# Spectral channel descriptors (decodes audio)
# ---------------------------------------------------------------------------


def _apply_control(
    audio: np.ndarray,
    sr: int,
    lowpass_hz: float | None,
    resample_hz: float | None,
    level_match: bool = False,
    level_peak: bool = False,
) -> tuple[np.ndarray, int]:
    """Apply a candidate bandwidth control, exactly as the pipeline would.

    Two candidates, because they are not equivalent:

    ``lowpass_hz``
        The 8th-order zero-phase Butterworth from ``audio_preprocessing`` — what
        ``run_balanced_ablation.py --lowpass-hz`` actually applies. It ATTENUATES
        the band rather than removing it, and attenuation is not enough here: a
        real track pushed 20 dB down still has energy where a band-limited fake
        has exactly none, and a log-spectrum feature separates those perfectly.

    ``resample_hz``
        Decimate and return to the original rate. The band then does not exist
        for either class, so there is nothing left to differ in. Strictly
        stronger, at the cost of also discarding any genuine artifact up there.

    Running the diagnostic under each is how we find out which control is
    actually required, instead of assuming the cheap one works.
    """
    from intrinsic_ai_music_detection.data.audio_preprocessing import level_match as _level_match, lowpass_filter

    if level_match:
        # First: trimming silence changes the loudness of what remains.
        audio = _level_match(audio, sr, peak_normalise=level_peak)
    if resample_hz:
        import librosa

        low = librosa.resample(audio, orig_sr=sr, target_sr=int(resample_hz), res_type="soxr_vhq")
        audio = librosa.resample(low, orig_sr=int(resample_hz), target_sr=sr, res_type="soxr_vhq")
    if lowpass_hz:
        audio = lowpass_filter(audio.astype(np.float32), sr, cutoff_hz=float(lowpass_hz))
    return np.asarray(audio, dtype=np.float32), sr


def _harmonicity_descriptors(audio: np.ndarray, sr: int) -> dict:
    """How tonal/harmonic the music is — gate C9's missing measurement.

    Why this belongs in the channel diagnostic
    ------------------------------------------
    Our detector reads a COMB: periodic peaks in the average spectrum. A sustained
    musical note also produces a comb — its harmonic series at f0. Our measured
    "real music" comb spacings are 265-490 Hz, squarely inside the range of
    musical f0, so `comb_strength` could be partly measuring how tonal a track is
    rather than whether a decoder touched it.

    No control in the protocol addressed this, and gate C9 has reported SKIP on
    every arm because no harmonicity column existed. These three do the job:

    * `harmonicity` — fraction of spectral energy explained by a harmonic model,
      via librosa's HPSS harmonic/percussive split.
    * `pitch_salience` — mean strength of the dominant pitch track (piptrack).
    * `spectral_peak_count` — how many prominent peaks the average spectrum has,
      which is the closest non-periodic analogue of the comb statistic.

    A high |corr(score, harmonicity)| within a class means the detector is partly
    a tonality detector, and the paper must say so.
    """
    import librosa

    out: dict = {}
    try:
        y = np.asarray(audio, dtype=np.float32)
        if len(y) < sr // 2:
            return out
        harm, perc = librosa.effects.hpss(y)
        e_h, e_p = float(np.sum(harm**2)), float(np.sum(perc**2))
        out["harmonicity"] = e_h / (e_h + e_p + 1e-12)

        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        pitches, mags = librosa.piptrack(S=S, sr=sr)
        strong = mags[mags > 0]
        out["pitch_salience"] = float(np.mean(strong)) if strong.size else float("nan")

        from scipy.signal import find_peaks

        mean_spec = 20.0 * np.log10(np.maximum(S.mean(axis=1), 1e-10))
        peaks, _ = find_peaks(mean_spec, prominence=3.0)
        out["spectral_peak_count"] = float(len(peaks))
    except Exception as exc:  # noqa: BLE001
        logger.debug("harmonicity descriptors failed: %s", exc)
    return out


def _level_descriptors(audio: np.ndarray, sr: int, silence_dbfs: float = -60.0, frame: int = 1024) -> dict:
    """Level and silence structure — confounds a bandwidth control cannot remove.

    Neural decoders emit fixed-length buffers, so generated audio frequently
    carries digital silence at the head or tail that mastered real music does
    not. Exact-zero samples, an untrimmed DC offset, and an unmastered crest
    factor are the same class of artifact: strong, trivially learnable, and
    nothing to do with synthesis.

    These are reported alongside the bandwidth descriptors precisely because a
    number can be "bandwidth-matched" and still be a corpus artifact.
    """
    out: dict = {}
    a = np.asarray(audio, dtype=np.float64).ravel()
    n = len(a)
    if n == 0:
        return out

    out["duration_s"] = round(n / float(sr), 3)
    peak = float(np.abs(a).max())
    rms = float(np.sqrt(np.mean(a**2)))
    out["peak_dbfs"] = round(20.0 * np.log10(max(peak, 1e-12)), 2)
    # Crest factor separates mastered (heavily limited, low crest) from raw
    # decoder output (higher crest) without touching frequency content at all.
    out["crest_factor_db"] = round(20.0 * np.log10(max(peak, 1e-12) / max(rms, 1e-12)), 2)
    out["dc_offset"] = round(float(np.mean(a)), 8)
    # Exact equality is deliberate: this counts DIGITAL silence (samples written
    # as literal zero by a decoder or a padding step), which is categorically
    # different from "very quiet". A tolerance here would measure quietness and
    # miss the artifact.
    out["exact_zero_frac"] = round(float(np.mean(a == 0.0)), 6)

    n_frames = n // frame
    if n_frames < 1:
        return out
    frames = a[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    quiet = frame_rms < (10.0 ** (silence_dbfs / 20.0))
    out["silence_frac"] = round(float(np.mean(quiet)), 6)

    # Leading/trailing silence, in seconds — the specific decoder-buffer tell.
    lead = int(np.argmax(~quiet)) if (~quiet).any() else n_frames
    tail = int(np.argmax(~quiet[::-1])) if (~quiet).any() else n_frames
    out["lead_silence_s"] = round(lead * frame / float(sr), 3)
    out["tail_silence_s"] = round(tail * frame / float(sr), 3)
    return out


def channel_descriptors(
    path: Path,
    max_duration: float = 30.0,
    cutoff_drop_db: float = 50.0,
    floor_margin_db: float = 6.0,
    lowpass_hz: float | None = None,
    resample_hz: float | None = None,
    level_match: bool = False,
    level_peak: bool = False,
    fixed_duration: float | None = None,
) -> dict:
    """Estimate low-pass cutoff and high-band energy structure for one file.

    The cutoff estimate walks DOWN from Nyquist and returns the first frequency
    whose smoothed spectrum rises above ``reference - cutoff_drop_db``, where the
    reference is the median level over 300 Hz - 1 kHz. Walking down (rather than
    up) is what makes it a cliff detector: a track that is merely dull in the
    highs still has a gradual slope and no cliff, whereas a resampled or
    band-limited file drops tens of dB within a few bins.
    """
    import librosa

    out: dict = {"path": str(path)}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            # sr=None preserves the NATIVE rate: resampling to a fixed analysis
            # rate would either invent or destroy the very band-edge we measure.
            audio, sr = librosa.load(str(path), sr=None, mono=True, duration=max_duration, res_type="soxr_hq")
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:200]
        return out

    audio = np.asarray(audio, dtype=np.float32)
    out["native_sr_decoded"] = int(sr)
    out["nyquist_hz"] = float(sr) / 2.0
    if audio.size < N_FFT:
        out["error"] = f"too short: {audio.size} samples < n_fft {N_FFT}"
        return out

    if lowpass_hz or resample_hz or level_match:
        try:
            audio, sr = _apply_control(audio, sr, lowpass_hz, resample_hz, level_match, level_peak)
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"control failed: {exc}"[:200]
            return out
        out["control_lowpass_hz"] = lowpass_hz
        out["control_resample_hz"] = resample_hz

    if fixed_duration:
        # Centre-crop to an EXACT length, after trimming. Trimming removes more
        # from generated audio (it carries more silence), which by itself creates
        # a duration difference where the raw cap had hidden one -- so the crop
        # has to come last. Tracks shorter than the target are left alone rather
        # than zero-padded, since padding would re-introduce the silence cue for
        # exactly the tracks that had the most of it.
        want = int(round(fixed_duration * sr))
        if len(audio) > want:
            start = (len(audio) - want) // 2
            audio = audio[start : start + want]
        out["control_fixed_duration"] = fixed_duration

    out.update(_level_descriptors(audio, sr))
    out.update(_harmonicity_descriptors(audio, sr))

    spec = np.abs(librosa.stft(audio, n_fft=N_FFT)) ** 2
    mean_power = spec.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = float(mean_power.sum())
    if not np.isfinite(total) or total <= 0:
        out["error"] = "silent or non-finite spectrum"
        return out

    db = 10.0 * np.log10(np.maximum(mean_power, 1e-20))

    # In-band reference: median over 300 Hz - 1 kHz. Median, not mean, so a
    # single tonal peak cannot set the reference.
    ref_band = (freqs >= 300.0) & (freqs <= 1000.0)
    reference_db = float(np.median(db[ref_band])) if ref_band.any() else float(np.median(db))
    out["reference_db"] = reference_db

    # Smooth over ~5 bins so one noisy bin cannot terminate the walk early.
    kernel = np.ones(5) / 5.0
    db_smooth = np.convolve(db, kernel, mode="same")

    threshold = reference_db - cutoff_drop_db
    above = np.nonzero(db_smooth > threshold)[0]
    # No bin above threshold => degenerate; report 0 rather than silently NaN,
    # since a genuinely empty spectrum is itself a finding.
    out["cutoff_hz"] = float(freqs[above[-1]]) if above.size else 0.0

    def _band_fraction(lo_hz: float, hi_hz: float) -> float:
        band = (freqs >= lo_hz) & (freqs < hi_hz)
        return float(mean_power[band].sum() / total) if band.any() else float("nan")

    nyquist = float(sr) / 2.0

    # The 8 kHz descriptors are only meaningful when there IS a band above
    # 8 kHz. On 16 kHz audio Nyquist *is* 8 kHz, so they collapse onto the top
    # one or two bins and report a large, confident, meaningless AUC — which is
    # exactly what happened on the raw-16 kHz FakeMusicCaps build
    # (hf_floor_frac 0.7038, above the gate, from a single degenerate bin).
    # Report NaN instead: an undefined descriptor must not look like a confound.
    if nyquist > 9_000.0:
        out["frac_power_above_8k"] = _band_fraction(8_000.0, nyquist)
        out["frac_power_8k_11k"] = _band_fraction(8_000.0, 11_000.0)
    else:
        out["frac_power_above_8k"] = float("nan")
        out["frac_power_8k_11k"] = float("nan")

    # Sample-rate-relative high band, so there is always a meaningful HF
    # measurement whatever the rate: the top quarter of the usable spectrum.
    hf_lo = 0.75 * nyquist
    out["frac_power_top_quarter"] = _band_fraction(hf_lo, nyquist)

    hf = freqs >= hf_lo
    if hf.any():
        hf_db = db[hf]
        floor_db = float(np.min(db))
        out["hf_floor_frac"] = float(np.mean(hf_db <= floor_db + floor_margin_db))
        hf_power = np.maximum(mean_power[hf], 1e-20)
        # Geometric/arithmetic mean ratio: 1.0 = flat (noise or floor), ->0 = tonal.
        out["spectral_flatness_hf"] = float(np.exp(np.mean(np.log(hf_power))) / np.mean(hf_power))
    else:
        out["hf_floor_frac"] = float("nan")
        out["spectral_flatness_hf"] = float("nan")

    return out


def _worker(job: dict) -> dict:
    row = dict(job["meta"])
    if job["probe"]:
        row.update(ffprobe_stream(Path(job["path"])))
    if job["decode"]:
        row.update(
            channel_descriptors(
                Path(job["path"]),
                max_duration=job["max_duration"],
                cutoff_drop_db=job["cutoff_drop_db"],
                lowpass_hz=job.get("lowpass_hz"),
                resample_hz=job.get("resample_hz"),
                level_match=job.get("level_match", False),
                level_peak=job.get("level_peak", False),
                fixed_duration=job.get("fixed_duration"),
            )
        )
    return row


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------


def pick_audio_column(df: pd.DataFrame, requested: str | None, which: str) -> str:
    """Resolve which manifest column holds the audio path we should analyse.

    Hard-fails rather than guessing: analysing the canonical WAV when you meant
    the raw source (or vice versa) yields a plausible table that answers the
    wrong question -- the exact failure class this repo has hit four times.
    """
    if requested:
        if requested not in df.columns:
            raise SystemExit(f"--audio-column {requested!r} not in manifest (columns: {list(df.columns)})")
        return requested

    candidates = CANON_PATH_COLUMNS if which == "canonical" else RAW_PATH_COLUMNS
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    raise SystemExit(
        f"no {which} path column found in manifest. Tried {list(candidates)}; "
        f"available: {list(df.columns)}. Pass --audio-column explicitly."
    )


def load_manifest(path: Path, per_stratum: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].astype(str) == "ok"]
        logger.info("status=='ok' filter: %d -> %d rows", before, len(df))
    if "label" not in df.columns:
        raise SystemExit(f"manifest {path} has no 'label' column (columns: {list(df.columns)})")
    df["label"] = df["label"].astype(str)
    df["algorithm"] = df.get("algorithm", pd.Series([""] * len(df))).fillna("").astype(str)
    if "track_id" in df.columns:
        df["track_id"] = df["track_id"].astype(str)

    if per_stratum:
        rng = np.random.default_rng(seed)
        parts = []
        for (label, alg), grp in df.groupby(["label", "algorithm"], sort=True):
            n = min(per_stratum, len(grp))
            idx = rng.choice(len(grp), size=n, replace=False)
            parts.append(grp.iloc[np.sort(idx)])
        df = pd.concat(parts, ignore_index=True)
        logger.info("sampled %d tracks (<=%d per label x algorithm stratum)", len(df), per_stratum)

    # A sampler that returns nothing must fail loudly, not produce an empty table.
    if df.empty:
        raise SystemExit(f"no usable rows in {path} after filtering — refusing to continue")
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def channel_alone_auc(df: pd.DataFrame, descriptors: list[str]) -> pd.DataFrame:
    """AUC of each descriptor used ALONE as a detector, per generator.

    Reported as ``max(auc, 1 - auc)`` alongside the signed value: a descriptor
    that separates the classes perfectly in the inverted direction is just as
    much of a confound, and the sign tells you which class is band-limited.
    """
    real = df[df["label"] == "real"]
    fake = df[df["label"] != "real"]
    rows: list[dict] = []
    for alg in sorted(set(fake["algorithm"])):
        sub = fake[fake["algorithm"] == alg]
        for desc in descriptors:
            if desc not in df.columns:
                continue
            r = real[desc].to_numpy(float)
            f = sub[desc].to_numpy(float)
            r, f = r[np.isfinite(r)], f[np.isfinite(f)]
            if len(r) < 5 or len(f) < 5:
                continue
            y = np.r_[np.zeros(len(r)), np.ones(len(f))]
            auc, eer = auc_and_eer(y, np.r_[r, f])
            rows.append(
                {
                    "algorithm": alg,
                    "descriptor": desc,
                    "auc_signed": round(auc, 4),
                    "auc_abs": round(max(auc, 1.0 - auc), 4),
                    "eer_pct": round(eer * 100, 2),
                    "real_median": round(float(np.median(r)), 4),
                    "fake_median": round(float(np.median(f)), 4),
                    "n_real": len(r),
                    "n_fake": len(f),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify the channel (bandwidth / sample-rate) confound in a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="canonical_manifest.csv / combined_manifest.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["inventory", "profile", "both"],
        default="profile",
        help="inventory = ffprobe metadata only (no decoding); profile = spectral descriptors.",
    )
    parser.add_argument(
        "--audio-column",
        default=None,
        help="Manifest column holding the path to analyse. Auto-detected from --which if omitted.",
    )
    parser.add_argument(
        "--which",
        choices=["canonical", "raw"],
        default="canonical",
        help="Which side of the pipeline to profile when --audio-column is not given.",
    )
    parser.add_argument("--per-stratum", type=int, default=400, help="tracks per label x algorithm; 0 = all")
    parser.add_argument("--max-duration", type=float, default=30.0, help="seconds decoded per track")
    parser.add_argument("--cutoff-drop-db", type=float, default=50.0)
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=None,
        help=(
            "Re-measure the channel confound AFTER applying the same Butterworth low-pass that "
            "run_balanced_ablation.py --lowpass-hz applies. Use this to CHECK the control works "
            "before spending GPU time on it: a Butterworth attenuates the band rather than "
            "removing it, and a real track 20 dB down still has energy where a band-limited fake "
            "has exactly none."
        ),
    )
    parser.add_argument(
        "--fixed-duration",
        type=float,
        default=None,
        help=(
            "Centre-crop every track to EXACTLY this many seconds, applied after trimming. "
            "Silence trimming removes more from generated audio than from mastered music, so it "
            "creates a duration difference at any analysis cap (measured: duration_s 0.7696 macro "
            "on SONICS at a 25 s cap, 0.9229 on chirp-v2). A fixed crop removes it outright and "
            "equalises the window count as well."
        ),
    )
    parser.add_argument(
        "--level-peak-normalise",
        action="store_true",
        default=False,
        help="With --level-match, also peak-normalise (removes the residual peak_dbfs cue).",
    )
    parser.add_argument(
        "--level-match",
        action="store_true",
        default=False,
        help=(
            "Also apply the LEVEL control (DC removal + silence trimming) before measuring. "
            "Measured 2026-08-13 with the bandwidth confound already closed, level descriptors "
            "still reach 0.7773 macro on SONICS and 0.7470 on FakeMusicCaps — no bandwidth "
            "control removes them, so this validates the second control the same way."
        ),
    )
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=None,
        help=(
            "Stronger alternative control: decimate to this rate and back, so the band does not "
            "exist for either class. Combine with --lowpass-hz to test both together."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(Path(args.manifest), args.per_stratum or None, args.seed)
    audio_col = pick_audio_column(df, args.audio_column, args.which)
    logger.info("analysing column %r (%s side)", audio_col, args.which)

    meta_cols = [c for c in ("track_id", "label", "algorithm", "fake_label") if c in df.columns]
    jobs: list[dict] = []
    missing = 0
    for row in df.itertuples(index=False):
        path = str(getattr(row, audio_col, "") or "").strip()
        if not path or not Path(path).exists():
            missing += 1
            continue
        jobs.append(
            {
                "path": path,
                "meta": {c: getattr(row, c) for c in meta_cols},
                "probe": args.mode in ("inventory", "both"),
                "decode": args.mode in ("profile", "both"),
                "max_duration": args.max_duration,
                "cutoff_drop_db": args.cutoff_drop_db,
                "lowpass_hz": args.lowpass_hz,
                "resample_hz": args.resample_hz,
                "level_match": args.level_match,
                "level_peak": args.level_peak_normalise,
                "fixed_duration": args.fixed_duration,
            }
        )

    logger.info("%d files resolve on disk, %d missing (column %r)", len(jobs), missing, audio_col)
    if not jobs:
        raise SystemExit(
            f"NO files resolved from column {audio_col!r}. If you are probing raw sources, they "
            f"may not be retained on this box — that is itself the answer to gate G2."
        )
    if missing:
        logger.warning(
            "%d/%d rows had no resolvable file. Report this ratio: a class-skewed miss rate "
            "silently rebalances the comparison.",
            missing,
            missing + len(jobs),
        )

    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 200 == 0 or i == len(futures):
                    logger.info("  %d/%d", i, len(futures))
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % 200 == 0 or i == len(jobs):
                logger.info("  %d/%d", i, len(jobs))

    per_track = pd.DataFrame(rows)
    ctrl = ""
    if args.lowpass_hz:
        ctrl += f"_lp{int(args.lowpass_hz)}"
    if args.resample_hz:
        ctrl += f"_rs{int(args.resample_hz)}"
    if args.level_match:
        ctrl += "_lvlpk" if args.level_peak_normalise else "_lvl"
    if args.fixed_duration:
        ctrl += f"_fd{args.fixed_duration:g}"
    tag = f"{args.which}_{args.mode}{ctrl}"
    per_track_path = out_dir / f"channel_per_track_{tag}.csv"
    per_track.to_csv(per_track_path, index=False)
    logger.info("per-track -> %s (%d rows)", per_track_path, len(per_track))

    if "error" in per_track.columns:
        n_err = int(per_track["error"].notna().sum())
        if n_err:
            logger.warning("%d/%d tracks failed analysis; see the 'error' column", n_err, len(per_track))

    # --- container-level summary (inventory) ---
    if args.mode in ("inventory", "both") and "native_sr" in per_track.columns:
        inv = (
            per_track.groupby(["label", "algorithm"])
            .agg(
                n=("native_sr", "size"),
                sr_modal=("native_sr", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
                sr_min=("native_sr", "min"),
                sr_max=("native_sr", "max"),
                codec_modal=("codec", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            )
            .reset_index()
        )
        inv.to_csv(out_dir / f"channel_inventory_{args.which}.csv", index=False)
        logger.info("\nCONTAINER INVENTORY (%s side):\n%s", args.which, inv.to_string(index=False))
        logger.info(
            "READ THIS FIRST: if the modal sample rate differs between label=real and the fake "
            "rows, the corpus carries a sample-rate label. Any detector with access to the band "
            "above min(Nyquist) can read it, and a 3-15 kHz fakeprint is undefined below 30 kHz."
        )

    # --- channel-alone AUC (the gate) ---
    if args.mode in ("profile", "both"):
        present = [d for d in DESCRIPTORS if d in per_track.columns]
        summary = channel_alone_auc(per_track, present)
        if summary.empty:
            logger.warning("no descriptor produced a comparable real/fake pair — check the manifest")
        else:
            summary_path = out_dir / f"channel_alone_auc_{args.which}{ctrl}.csv"
            summary.to_csv(summary_path, index=False)
            pivot = summary.pivot_table(index="descriptor", columns="algorithm", values="auc_abs")
            pivot["MACRO"] = pivot.mean(axis=1)
            pivot = pivot.sort_values("MACRO", ascending=False)
            logger.info(
                "\nCHANNEL-ALONE AUC (|AUC|, higher = more of the separation is delivery chain, "
                "not synthesis) on the %s audio%s:\n%s",
                args.which,
                f" [control{ctrl}]" if ctrl else " [NO control]",
                pivot.to_string(),
            )
            # Split the verdict by family: a bandwidth control fixes the first
            # group and does nothing at all for the second, so a single "best
            # descriptor" number would let a level confound hide behind a
            # successful bandwidth control.
            band_rows = [
                d
                for d in pivot.index
                if d
                in {
                    "cutoff_hz",
                    "frac_power_above_8k",
                    "frac_power_8k_11k",
                    "frac_power_top_quarter",
                    "hf_floor_frac",
                    "spectral_flatness_hf",
                }
            ]
            level_rows = [d for d in pivot.index if d not in band_rows]
            if band_rows:
                logger.info(
                    "  bandwidth family : best macro |AUC| %.4f (%s)",
                    pivot.loc[band_rows, "MACRO"].max(),
                    pivot.loc[band_rows, "MACRO"].idxmax(),
                )
            if level_rows:
                logger.info(
                    "  level/silence family : best macro |AUC| %.4f (%s)  " "— NOT removed by any bandwidth control",
                    pivot.loc[level_rows, "MACRO"].max(),
                    pivot.loc[level_rows, "MACRO"].idxmax(),
                )

            worst = float(pivot["MACRO"].max())
            logger.info(
                "\nVERDICT: best single channel descriptor reaches macro |AUC| %.4f.\n"
                "  >= 0.90  the corpus is channel-labelled; every AUC on it without a shared "
                "low-pass is partly a channel measurement and MUST ship with its "
                "bandwidth-matched twin (--lowpass-hz).\n"
                "  0.70-0.90 a material confound; report the matched twin.\n"
                "  <  0.70  the channel is not, on its own, a strong detector here.",
                worst,
            )

    logger.info(
        "\nNOTE: this measures what a descriptor can do ALONE. It does not prove the flow uses "
        "the channel — that is what the --lowpass-hz twin run and the reconstruction control "
        "(build_reconstruction_control.py, content-identical pairs) establish."
    )


if __name__ == "__main__":
    main()
