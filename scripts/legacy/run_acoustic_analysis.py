"""Acoustic feature extraction and analysis for AI music detection.

Extracts handcrafted acoustic features designed to expose AI generation artifacts
across three signal channels:

  1. Codec boundary artifacts
       phase_disc_frame  — instantaneous phase discontinuities at codec frame
                           boundaries (~20 ms at 24 kHz / 75 Hz frame rate).
       codec_periodicity — autocorrelation of the RMS envelope at the codec
                           frame lag; a strong peak indicates regular stitching.
       hf_energy_ratio   — fraction of energy above 8 kHz.

  2. Over-smoothed dynamics  (diffusion / language-model vocoders)
       mfcc_delta2_energy  — mean energy of MFCC acceleration (2nd derivative).
       micro_dynamics      — variance of 5 ms RMS frames (sub-beat roughness).
       rms_cv              — coefficient of variation of RMS across 100 ms frames.
       spectral_flux_cv    — CV of spectral flux.

  3. Tonal / structural over-regularity
       chroma_entropy    — entropy of the mean chroma vector.
       chroma_stability  — mean within-frame chroma std.
       onset_ioi_cv      — CV of inter-onset intervals.
       hpss_ratio        — harmonic-to-total energy from HPSS.

Confound audit
--------------
Run with --preprocess-mode canonical to apply the MP3 round-trip + LUFS
normalisation from audio_preprocessing.py before feature extraction.
Then use --compare-raw-csv <path-to-raw-acoustic_features.csv> to produce
a side-by-side AUC comparison (raw vs canonical) that reveals which features
survive format matching (confound-resistant) vs. which collapse (format-driven).

Outputs (all in --output-dir)
------------------------------
  acoustic_features.csv             per-track feature matrix (+ duration, lufs)
  acoustic_analysis.csv             per-feature AUC, Cohen d, p-value, per-algo means
  acoustic_classification.csv       logistic regression AUC summary
  acoustic_combined.csv             joined with --ablation-csv (if supplied)
  confound_audit.csv                raw-vs-canonical AUC comparison (if requested)
  confound_resistant_shortlist.csv  features that survive format matching

Usage
-----
# Raw baseline (no format matching):
poetry run python scripts/run_acoustic_analysis.py \\
  --per-stratum 200 --preprocess-mode raw \\
  --output-dir data/processed/acoustic_analysis_raw

# Canonical (format-matched):
poetry run python scripts/run_acoustic_analysis.py \\
  --per-stratum 200 --preprocess-mode canonical \\
  --output-dir data/processed/acoustic_analysis_canonical

# Confound audit (compare raw vs canonical):
poetry run python scripts/run_acoustic_analysis.py \\
  --per-stratum 200 --preprocess-mode canonical \\
  --compare-raw-csv data/processed/acoustic_analysis_raw/acoustic_features.csv \\
  --output-dir data/processed/acoustic_analysis_canonical

# Restrict to hard algorithms only:
poetry run python scripts/run_acoustic_analysis.py \\
  --per-stratum 200 --algorithms chirp-v3.5 udio-120s chirp-v3 \\
  --output-dir data/processed/acoustic_analysis

# Combine with existing MERT ablation to test joint classifier:
poetry run python scripts/run_acoustic_analysis.py \\
  --per-stratum 200 \\
  --ablation-csv data/processed/sonics_balanced_ablation/ablation_mert.csv \\
  --output-dir data/processed/acoustic_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_preprocessing import (
    CANONICAL_MAX_DURATION,
    CANONICAL_TARGET_SR,
    measure_loudness_lufs,
    preprocess_audio,
)
from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio
from intrinsic_ai_music_detection.data.balanced_sampling import load_balanced_tracks as _shared_load_balanced_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")
ANALYSIS_SR = CANONICAL_TARGET_SR
MAX_DURATION = CANONICAL_MAX_DURATION
# EnCodec 24 kHz encoder stride = 320 samples  →  75 Hz frame rate
CODEC_FRAME_SAMPLES = 320
MICRO_WIN_SAMPLES = int(ANALYSIS_SR * 0.005)  # 5 ms
RMS_HOP_SAMPLES = int(ANALYSIS_SR * 0.100)  # 100 ms


# ---------------------------------------------------------------------------
# Track loading (delegates to shared balanced_sampling module)
# ---------------------------------------------------------------------------


def load_balanced_tracks(
    per_stratum: int,
    fake_types: list[str] | None = None,
    algorithms: list[str] | None = None,
    max_real: int | None = None,
    seed: int = 42,
    real_genres_csv: str | None = None,
    covariate_profiles_csv: str | None = None,
    canonical_manifest: str | None = None,
) -> tuple[list[dict], list[dict]]:
    return _shared_load_balanced_tracks(
        per_stratum=per_stratum,
        fake_types=fake_types,
        algorithms=algorithms,
        max_real=max_real,
        seed=seed,
        real_genres_csv=real_genres_csv,
        covariate_profiles_csv=covariate_profiles_csv,
        match_covariates=True,
        canonical_manifest=canonical_manifest,
    )
    return sampled_real, sampled_fake


# ---------------------------------------------------------------------------
# Acoustic feature extraction
# ---------------------------------------------------------------------------


def _safe_kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    return float(scipy.stats.kurtosis(x, fisher=True))


def _entropy(x: np.ndarray, bins: int = 32) -> float:
    """Normalised Shannon entropy (binned)."""
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))


def extract_acoustic_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute all acoustic features for one track. Returns NaN on failure."""
    feat: dict[str, float] = {}
    n = len(audio)
    dur = n / sr

    # ------------------------------------------------------------------
    # 1. MFCC-based dynamics
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)

    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat["mfcc_delta_energy"] = float(np.mean(np.linalg.norm(delta1, axis=0)))
    feat["mfcc_delta2_energy"] = float(np.mean(np.linalg.norm(delta2, axis=0)))
    feat["mfcc_delta_kurtosis"] = _safe_kurtosis(delta1.ravel())
    d1 = feat["mfcc_delta_energy"]
    d2 = feat["mfcc_delta2_energy"]
    feat["mfcc_jerk_ratio"] = (d2 / d1) if d1 > 1e-9 else np.nan

    # ------------------------------------------------------------------
    # 2. Spectral features
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512).ravel()
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512).ravel()
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=512).ravel()

    feat["spectral_flatness_mean"] = float(np.mean(flatness))
    feat["spectral_flatness_std"] = float(np.std(flatness))
    feat["spectral_flatness_kurtosis"] = _safe_kurtosis(flatness)
    feat["spectral_centroid_mean"] = float(np.mean(centroid))
    feat["spectral_centroid_std"] = float(np.std(centroid))
    feat["spectral_rolloff_mean"] = float(np.mean(rolloff))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        S = np.abs(librosa.stft(audio, hop_length=512))

    flux = np.mean(np.abs(np.diff(S, axis=1)), axis=0)
    feat["spectral_flux_mean"] = float(np.mean(flux))
    feat["spectral_flux_cv"] = float(np.std(flux) / (np.mean(flux) + 1e-9))

    # ------------------------------------------------------------------
    # 3. Mel-band kurtosis
    #    Real music: heavy-tailed per-band distributions (transients).
    #    Diffusion generation: more Gaussian → lower kurtosis.
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mel_S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=512)

    mel_kurt = np.array([_safe_kurtosis(mel_S[i]) for i in range(mel_S.shape[0])])
    mel_kurt = mel_kurt[np.isfinite(mel_kurt)]
    feat["mel_kurtosis_mean"] = float(np.mean(mel_kurt)) if len(mel_kurt) else np.nan
    feat["mel_kurtosis_std"] = float(np.std(mel_kurt)) if len(mel_kurt) else np.nan

    # ------------------------------------------------------------------
    # 4. RMS dynamics
    # ------------------------------------------------------------------
    rms = librosa.feature.rms(y=audio, hop_length=RMS_HOP_SAMPLES).ravel()
    rms = rms[rms > 1e-9]
    if len(rms) > 3:
        feat["rms_cv"] = float(np.std(rms) / np.mean(rms))
        feat["rms_kurtosis"] = _safe_kurtosis(rms)
        feat["rms_entropy"] = _entropy(rms)
    else:
        feat["rms_cv"] = feat["rms_kurtosis"] = feat["rms_entropy"] = np.nan

    # Micro-dynamics: variance of 5 ms RMS frames
    n_micro = n // MICRO_WIN_SAMPLES
    if n_micro > 4:
        micro = np.array(
            [np.sqrt(np.mean(audio[i * MICRO_WIN_SAMPLES : (i + 1) * MICRO_WIN_SAMPLES] ** 2)) for i in range(n_micro)]
        )
        feat["micro_dynamics"] = float(np.var(micro))
    else:
        feat["micro_dynamics"] = np.nan

    # ------------------------------------------------------------------
    # 5. Onset / rhythm
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512)

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    feat["onset_rate"] = float(len(onset_times) / dur) if dur > 0 else np.nan

    if len(onset_times) >= 4:
        ioi = np.diff(onset_times)
        ioi = ioi[ioi > 0.05]
        if len(ioi) >= 3:
            feat["onset_ioi_std"] = float(np.std(ioi))
            feat["onset_ioi_cv"] = float(np.std(ioi) / np.mean(ioi))
        else:
            feat["onset_ioi_std"] = feat["onset_ioi_cv"] = np.nan
    else:
        feat["onset_ioi_std"] = feat["onset_ioi_cv"] = np.nan

    # ------------------------------------------------------------------
    # 6. Tonal / chroma
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)  # [12, T]

    mean_chroma = chroma.mean(axis=1)
    p = mean_chroma / (mean_chroma.sum() + 1e-9)
    p = p[p > 0]
    feat["chroma_entropy"] = float(-np.sum(p * np.log2(p)) / np.log2(12))
    feat["chroma_stability"] = float(np.mean(np.std(chroma, axis=0)))

    # ------------------------------------------------------------------
    # 7. Harmonic / Percussive separation
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        D_harm, D_perc = librosa.decompose.hpss(S)

    harm_e = float(np.sum(D_harm**2))
    total_e = harm_e + float(np.sum(D_perc**2))
    feat["hpss_ratio"] = (harm_e / total_e) if total_e > 0 else np.nan

    # ------------------------------------------------------------------
    # 8. Codec boundary artifact detection
    #
    #    EnCodec 24 kHz stride = 320 samples → 75 Hz frame rate.
    #    The decoder stitches independently quantised 30 s chunks;
    #    residual phase seams at every 320-sample boundary are a
    #    reliable codec fingerprint.
    # ------------------------------------------------------------------

    # (a) Phase discontinuity ratio: boundaries vs mid-frame controls
    try:
        analytic = scipy.signal.hilbert(audio)
        inst_phase = np.unwrap(np.angle(analytic))
        boundary_idx = np.arange(CODEC_FRAME_SAMPLES, n - CODEC_FRAME_SAMPLES, CODEC_FRAME_SAMPLES)
        if len(boundary_idx) > 10:
            phase_diff = np.abs(np.diff(inst_phase))
            at_boundary = phase_diff[boundary_idx]
            control_idx = boundary_idx + CODEC_FRAME_SAMPLES // 2
            control_idx = control_idx[control_idx < len(phase_diff)]
            at_control = phase_diff[control_idx[: len(boundary_idx)]]
            feat["phase_disc_frame"] = float(np.mean(at_boundary) / (np.mean(at_control) + 1e-9))
        else:
            feat["phase_disc_frame"] = np.nan
    except Exception:
        feat["phase_disc_frame"] = np.nan

    # (b) Codec periodicity: normalised ACF at lag 1 (= 1 codec frame)
    try:
        rms_fine = librosa.feature.rms(y=audio, hop_length=CODEC_FRAME_SAMPLES).ravel()
        if len(rms_fine) > 40:
            rms_z = (rms_fine - rms_fine.mean()) / (rms_fine.std() + 1e-9)
            acf = np.correlate(rms_z, rms_z, mode="full")[len(rms_z) - 1 :]
            acf /= acf[0] + 1e-9
            feat["codec_periodicity"] = float(acf[1]) if len(acf) > 1 else np.nan
        else:
            feat["codec_periodicity"] = np.nan
    except Exception:
        feat["codec_periodicity"] = np.nan

    # (c) High-frequency energy ratio (>8 kHz)
    try:
        freq_res = sr / (2 * S.shape[0])
        hf_bin = int(8000 / freq_res)
        hf_e = float(np.sum(S[hf_bin:] ** 2))
        tot_S_e = float(np.sum(S**2))
        feat["hf_energy_ratio"] = (hf_e / tot_S_e) if tot_S_e > 0 else np.nan
    except Exception:
        feat["hf_energy_ratio"] = np.nan

    # ------------------------------------------------------------------
    # 9. Zero-crossing rate
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512).ravel()

    feat["zcr_mean"] = float(np.mean(zcr))
    feat["zcr_cv"] = float(np.std(zcr) / (np.mean(zcr) + 1e-9))

    return feat


FEATURE_NAMES: list[str] = [
    "mfcc_delta_energy",
    "mfcc_delta2_energy",
    "mfcc_delta_kurtosis",
    "mfcc_jerk_ratio",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_flatness_kurtosis",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_flux_mean",
    "spectral_flux_cv",
    "mel_kurtosis_mean",
    "mel_kurtosis_std",
    "rms_cv",
    "rms_kurtosis",
    "rms_entropy",
    "micro_dynamics",
    "onset_rate",
    "onset_ioi_std",
    "onset_ioi_cv",
    "chroma_entropy",
    "chroma_stability",
    "hpss_ratio",
    "phase_disc_frame",
    "codec_periodicity",
    "hf_energy_ratio",
    "zcr_mean",
    "zcr_cv",
    # Covariates (not discriminative per se; used for confound auditing)
    "duration",
    "loudness_lufs",
]

# Features expected to survive format-matching (confound-resistant candidates).
# These do not rely on bandwidth or bitrate differences.
CONFOUND_RESISTANT_CANDIDATES: list[str] = [
    "mel_kurtosis_std",
    "mel_kurtosis_mean",
    "spectral_centroid_std",
    "mfcc_delta2_energy",
    "micro_dynamics",
    "rms_cv",
    "spectral_flux_cv",
    "onset_ioi_cv",
    "chroma_stability",
    "codec_periodicity",
    "phase_disc_frame",
]


def process_track(
    track: dict,
    preprocess_mode: str = "raw",
    lowpass_hz: float | None = None,
    analysis_duration: float | None = None,
) -> dict | None:
    try:
        dur = analysis_duration if analysis_duration else MAX_DURATION
        if preprocess_mode == "canonical":
            audio, sr, meta = preprocess_audio(
                track["path"],
                target_sr=ANALYSIS_SR,
                mode="canonical",
                max_duration=dur,
                lowpass_hz=lowpass_hz,
            )
            measured_lufs = meta.get("measured_lufs", float("nan"))
            actual_duration = meta.get("actual_duration", float("nan"))
        elif preprocess_mode == "preprocessed":
            # Pre-built canonical WAV — load as-is, then optionally lowpass + trim.
            audio, sr = load_audio(track["path"], target_sr=ANALYSIS_SR, max_duration=dur)
            if lowpass_hz:
                from intrinsic_ai_music_detection.data.audio_preprocessing import lowpass_filter

                audio = lowpass_filter(audio, sr, cutoff_hz=lowpass_hz)
            measured_lufs = measure_loudness_lufs(audio, sr)
            actual_duration = float(len(audio) / sr)
        else:
            audio, sr = load_audio(track["path"], target_sr=ANALYSIS_SR, max_duration=dur)
            if lowpass_hz:
                from intrinsic_ai_music_detection.data.audio_preprocessing import lowpass_filter

                audio = lowpass_filter(audio, sr, cutoff_hz=lowpass_hz)
            measured_lufs = measure_loudness_lufs(audio, sr)
            actual_duration = float(len(audio) / sr)
            audio = normalize_audio(audio)

        feat = extract_acoustic_features(audio, sr)
        del audio
        feat["duration"] = actual_duration
        feat["loudness_lufs"] = measured_lufs
        return {**track, "preprocess_mode": preprocess_mode, **feat}
    except Exception as e:
        logger.warning("Failed %s: %s: %s", track["track_id"], type(e).__name__, e)
        return None


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _auc_for_feature(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(x)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return np.nan
    auc = roc_auc_score(y[mask], x[mask])
    return float(max(auc, 1 - auc))  # polarity-invariant


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-12))


def _mwu_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, p = scipy.stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def analyse_features(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute per-feature discriminability statistics and per-algorithm group means."""
    y = (df["label"] == "fake").astype(int).to_numpy()
    real_mask = y == 0
    fake_mask = y == 1

    rows: list[dict] = []
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        x_real = x[real_mask & np.isfinite(x)]
        x_fake = x[fake_mask & np.isfinite(x)]

        row: dict = {
            "feature": feat,
            "auc_global": _auc_for_feature(y, x),
            "cohens_d": _cohens_d(x_real, x_fake),
            "mwu_p": _mwu_p(x_real, x_fake),
            "mean_real": float(np.mean(x_real)) if len(x_real) else np.nan,
            "mean_fake": float(np.mean(x_fake)) if len(x_fake) else np.nan,
        }

        # Per-algorithm: mean value and AUC (real vs that algorithm's fake tracks)
        for algo in sorted(df["algorithm"].dropna().unique()):
            if not str(algo).strip():
                continue
            sub = x[df["algorithm"].values == algo]
            sub = sub[np.isfinite(sub)]
            row[f"mean_{algo}"] = float(np.mean(sub)) if len(sub) else np.nan

            algo_mask = (df["algorithm"].values == algo) & np.isfinite(x)
            y_sub = np.concatenate([y[real_mask & np.isfinite(x)], y[algo_mask]])
            x_sub = np.concatenate([x[real_mask & np.isfinite(x)], x[algo_mask]])
            row[f"auc_{algo}"] = _auc_for_feature(y_sub, x_sub)

        rows.append(row)

    analysis = pd.DataFrame(rows).sort_values("auc_global", ascending=False)
    analysis.to_csv(output_dir / "acoustic_analysis.csv", index=False)

    logger.info("=" * 75)
    logger.info("%-32s  %6s  %8s  %9s  %9s", "feature", "AUC", "Cohen-d", "mean_real", "mean_fake")
    logger.info("-" * 75)
    for _, r in analysis.iterrows():
        logger.info(
            "%-32s  %.3f   %+.3f    %9.5f  %9.5f",
            r["feature"],
            r.get("auc_global", np.nan),
            r.get("cohens_d", np.nan),
            r.get("mean_real", np.nan),
            r.get("mean_fake", np.nan),
        )
    logger.info("=" * 75)
    return analysis


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )


def _cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[0] < 20 or len(np.unique(y)) < 2:
        return np.nan
    pipe = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def run_classification(df: pd.DataFrame, output_dir: Path, ablation_csv: str | None) -> None:
    """Logistic regression: acoustic-only vs. ID-only vs. combined."""
    y = (df["label"] == "fake").astype(int).to_numpy()
    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X_all = df[feat_cols].to_numpy(dtype=float)
    finite = np.isfinite(X_all).all(axis=1)

    df_clean = df[finite].copy()
    X_ac = X_all[finite]
    y_ac = y[finite]

    logger.info("Acoustic-only classifier: %d tracks, %d features", len(y_ac), len(feat_cols))
    auc_acoustic = _cv_auc(X_ac, y_ac)
    logger.info("  Acoustic-only  AUC = %.3f", auc_acoustic)
    clf_rows: list[dict] = [{"bundle": "acoustic_only", "n": int(len(y_ac)), "auc": auc_acoustic}]

    # Per-algorithm acoustic AUC
    for algo in sorted(df_clean["algorithm"].dropna().unique()):
        if not str(algo).strip():
            continue
        sub = df_clean[
            (df_clean["label"] == "real") | ((df_clean["label"] == "fake") & (df_clean["algorithm"] == algo))
        ]
        xs = sub[feat_cols].to_numpy(dtype=float)
        ys = (sub["label"] == "fake").astype(int).to_numpy()
        fm = np.isfinite(xs).all(axis=1)
        xs, ys = xs[fm], ys[fm]
        auc_a = _cv_auc(xs, ys)
        logger.info("  Acoustic-only  [%-20s]  AUC = %.3f  (n=%d)", algo, auc_a, len(ys))
        clf_rows.append({"bundle": f"acoustic_only:{algo}", "n": int(len(ys)), "auc": auc_a})

    # Combined with ID features from existing ablation CSV
    if ablation_csv:
        abl_path = Path(ablation_csv)
        if not abl_path.exists():
            logger.warning("--ablation-csv not found: %s", ablation_csv)
        else:
            df_abl = pd.read_csv(abl_path, low_memory=False)
            id_feat_cols = [
                c
                for c in df_abl.columns
                if any(c.startswith(p) for p in ("id_", "mert_id_", "encodec_id_", "temporal_"))
                and pd.to_numeric(df_abl[c], errors="coerce").notna().any()
            ]
            if id_feat_cols and "track_id" in df_abl.columns:
                df_merged = df_clean.merge(df_abl[["track_id"] + id_feat_cols], on="track_id", how="inner")
                if len(df_merged) > 20:
                    df_merged.to_csv(output_dir / "acoustic_combined.csv", index=False)
                    logger.info(
                        "Saved acoustic_combined.csv: %d tracks x %d cols",
                        len(df_merged),
                        len(df_merged.columns),
                    )
                    y_m = (df_merged["label"] == "fake").astype(int).to_numpy()
                    X_id = df_merged[id_feat_cols].to_numpy(dtype=float)
                    X_comb = np.hstack([df_merged[feat_cols].to_numpy(dtype=float), X_id])
                    m = np.isfinite(X_comb).all(axis=1)
                    auc_id = _cv_auc(X_id[m], y_m[m])
                    auc_comb = _cv_auc(X_comb[m], y_m[m])
                    logger.info("  ID-only        AUC = %.3f", auc_id)
                    logger.info(
                        "  Combined       AUC = %.3f  (delta %+.3f vs ID-only)",
                        auc_comb,
                        auc_comb - auc_id,
                    )
                    clf_rows.extend(
                        [
                            {"bundle": "id_only", "n": int(m.sum()), "auc": auc_id},
                            {"bundle": "combined", "n": int(m.sum()), "auc": auc_comb},
                        ]
                    )

    pd.DataFrame(clf_rows).to_csv(output_dir / "acoustic_classification.csv", index=False)
    logger.info("Saved acoustic_classification.csv")


# ---------------------------------------------------------------------------
# Confound audit: raw-vs-canonical AUC comparison
# ---------------------------------------------------------------------------


def run_confound_audit(
    df_canonical: pd.DataFrame,
    raw_csv: str | Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare per-feature AUC on raw vs canonical preprocessed audio.

    Features whose AUC drops substantially when format is matched are
    format-driven confounds.  Features whose AUC persists are genuine
    discriminative signals.

    Parameters
    ----------
    df_canonical
        Acoustic features extracted from the canonical (format-matched) corpus.
    raw_csv
        Path to ``acoustic_features.csv`` produced with ``--preprocess-mode raw``.
    output_dir
        Where to save the output files.

    Outputs
    -------
    confound_audit.csv
        Per-feature: auc_raw, auc_canonical, auc_drop, is_confound, is_survivor
    confound_resistant_shortlist.csv
        Subset with auc_canonical >= 0.6 and auc_drop <= 0.10
    """
    raw_path = Path(raw_csv)
    if not raw_path.exists():
        logger.warning("--compare-raw-csv not found: %s — skipping confound audit", raw_csv)
        return pd.DataFrame()

    df_raw = pd.read_csv(raw_path, low_memory=False)

    # Use only features in both runs (exclude covariates duration/loudness_lufs from discriminability)
    analysis_features = [f for f in FEATURE_NAMES if f not in ("duration", "loudness_lufs")]

    rows = []
    for feat in analysis_features:
        auc_raw = np.nan
        auc_can = np.nan

        if feat in df_raw.columns:
            y_r = (df_raw["label"] == "fake").astype(int).to_numpy()
            x_r = pd.to_numeric(df_raw[feat], errors="coerce").to_numpy(dtype=float)
            auc_raw = _auc_for_feature(y_r, x_r)

        if feat in df_canonical.columns:
            y_c = (df_canonical["label"] == "fake").astype(int).to_numpy()
            x_c = pd.to_numeric(df_canonical[feat], errors="coerce").to_numpy(dtype=float)
            auc_can = _auc_for_feature(y_c, x_c)

        auc_drop = (auc_raw - auc_can) if (np.isfinite(auc_raw) and np.isfinite(auc_can)) else np.nan
        is_confound = bool(np.isfinite(auc_drop) and auc_drop > 0.10)
        is_survivor = bool(np.isfinite(auc_can) and auc_can >= 0.60 and (not np.isfinite(auc_drop) or auc_drop <= 0.10))

        rows.append(
            {
                "feature": feat,
                "auc_raw": auc_raw,
                "auc_canonical": auc_can,
                "auc_drop": auc_drop,
                "is_format_confound": is_confound,
                "is_confound_resistant": is_survivor,
                "expected_resistant": feat in CONFOUND_RESISTANT_CANDIDATES,
            }
        )

    audit_df = pd.DataFrame(rows).sort_values("auc_canonical", ascending=False)
    audit_df.to_csv(output_dir / "confound_audit.csv", index=False)
    logger.info("Saved confound_audit.csv  (%d features)", len(audit_df))

    # Log summary
    n_confounds = audit_df["is_format_confound"].sum()
    n_survivors = audit_df["is_confound_resistant"].sum()
    logger.info("  Format confounds (auc_drop > 0.10): %d", n_confounds)
    logger.info("  Confound-resistant survivors (auc_canonical >= 0.60, drop <= 0.10): %d", n_survivors)

    logger.info("=== Confound Audit (sorted by canonical AUC) ===")
    logger.info("%-35s  %6s  %6s  %7s  %s", "feature", "raw", "canon", "drop", "status")
    for _, r in audit_df.iterrows():
        status = "CONFOUND" if r["is_format_confound"] else ("SURVIVOR" if r["is_confound_resistant"] else "neutral")
        logger.info(
            "%-35s  %.3f  %.3f  %+.3f   %s",
            r["feature"],
            r["auc_raw"] if np.isfinite(r["auc_raw"]) else -1,
            r["auc_canonical"] if np.isfinite(r["auc_canonical"]) else -1,
            r["auc_drop"] if np.isfinite(r["auc_drop"]) else 0.0,
            status,
        )

    # Shortlist
    shortlist = audit_df[audit_df["is_confound_resistant"]].copy()
    shortlist.to_csv(output_dir / "confound_resistant_shortlist.csv", index=False)
    logger.info("Saved confound_resistant_shortlist.csv  (%d features)", len(shortlist))

    return audit_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic feature extraction and analysis for AI music detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--per-stratum", type=int, default=200, help="Max tracks per (algorithm x fake_label) stratum")
    parser.add_argument("--max-real", type=int, default=None, help="Max real tracks (default: match total fake count)")
    parser.add_argument(
        "--fake-types", nargs="+", default=None, help="Restrict fake_label values, e.g. 'full fake' 'half fake'"
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None, help="Restrict algorithm versions, e.g. chirp-v3.5 udio-120s"
    )
    parser.add_argument("--output-dir", type=str, default="data/processed/acoustic_analysis")
    parser.add_argument(
        "--ablation-csv", type=str, default=None, help="Path to existing ablation CSV for combined classifier eval"
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["raw", "canonical", "preprocessed"],
        default="raw",
        help=(
            "raw: legacy decode+peak-normalise. "
            "canonical: MP3 round-trip + LUFS on the fly (use without --canonical-manifest). "
            "preprocessed: load pre-built canonical WAV as-is (use WITH --canonical-manifest)."
        ),
    )
    parser.add_argument(
        "--compare-raw-csv",
        type=str,
        default=None,
        help=(
            "Path to acoustic_features.csv from a raw-mode run. "
            "When supplied together with --preprocess-mode canonical, "
            "produces confound_audit.csv comparing raw vs canonical AUC."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for feature extraction. "
            "On g4dn.xlarge (4 vCPUs, CPU-bound librosa): 2-4 workers recommended. "
            "Keep at 1 if GPU memory is tight (no GPU used here, but avoids GIL contention)."
        ),
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=None,
        help=(
            "Apply a Butterworth low-pass filter at this frequency (Hz) to BOTH classes before "
            "feature extraction. Use 8000 to equalise the spectral bandwidth confound in SONICS "
            "(fakes are band-limited to ~8 kHz). Applied after loading / canonical preprocessing."
        ),
    )
    parser.add_argument(
        "--analysis-duration",
        type=float,
        default=None,
        help=(
            "Maximum audio duration (seconds) to analyse per track. "
            "Use e.g. 60 to equalise the real (~119 s) vs fake (~97 s) duration confound and "
            "also speed up the run. Default: use full canonical max duration (120 s)."
        ),
    )
    # --- new: fair sampling ---
    parser.add_argument(
        "--real-genres-csv",
        default=None,
        help="Path to real_genres.csv from tag_real_genres.py (for genre-matching).",
    )
    parser.add_argument(
        "--covariate-profiles-csv",
        default=None,
        help="Path to covariate_profiles.csv from profile_covariates.py.",
    )
    parser.add_argument(
        "--canonical-manifest",
        default=None,
        help="Path to canonical_manifest.csv (uses canonical audio paths).",
    )
    # --- workers + checkpoint ---
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]  # prevent CLAP from hijacking our args

    if args.canonical_manifest and args.preprocess_mode == "canonical":
        logger.warning(
            "--canonical-manifest points to pre-built WAVs; switching --preprocess-mode "
            "from 'canonical' to 'preprocessed' to avoid double preprocessing."
        )
        args.preprocess_mode = "preprocessed"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    workers = getattr(args, "workers", 1)
    # Embed lowpass/duration in checkpoint name so different experiment configs don't share state
    lowpass_tag = f"_lp{int(getattr(args, 'lowpass_hz', 0) or 0)}" if getattr(args, "lowpass_hz", None) else ""
    dur_tag = f"_d{int(getattr(args, 'analysis_duration', 0) or 0)}" if getattr(args, "analysis_duration", None) else ""
    checkpoint_path = output_dir / f"acoustic_checkpoint{lowpass_tag}{dur_tag}.jsonl"

    # Resume: load any already-processed track IDs from the checkpoint
    done_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    done_ids.add(str(row.get("track_id", "")))
                except Exception:
                    pass
        logger.info("Resuming: %d tracks already in checkpoint", len(done_ids))

    logger.info("Loading balanced tracks (per-stratum=%d, mode=%s)...", args.per_stratum, args.preprocess_mode)
    real_tracks, fake_tracks = load_balanced_tracks(
        per_stratum=args.per_stratum,
        fake_types=args.fake_types,
        algorithms=args.algorithms,
        max_real=args.max_real,
        seed=args.seed,
        real_genres_csv=getattr(args, "real_genres_csv", None),
        covariate_profiles_csv=getattr(args, "covariate_profiles_csv", None),
        canonical_manifest=getattr(args, "canonical_manifest", None),
    )
    all_tracks = real_tracks + fake_tracks
    todo_tracks = [t for t in all_tracks if str(t.get("track_id", "")) not in done_ids]
    logger.info("Total tracks: %d  (skipping %d already done)", len(all_tracks), len(all_tracks) - len(todo_tracks))

    results: list[dict] = []
    t0 = time.time()

    lowpass_hz = getattr(args, "lowpass_hz", None)
    analysis_duration = getattr(args, "analysis_duration", None)
    if lowpass_hz:
        logger.info("Low-pass filter: %.0f Hz applied to both classes (bandwidth equalisation)", lowpass_hz)
    if analysis_duration:
        logger.info("Analysis duration cap: %.0f s (duration confound reduction)", analysis_duration)

    worker_fn = partial(
        process_track,
        preprocess_mode=args.preprocess_mode,
        lowpass_hz=lowpass_hz,
        analysis_duration=analysis_duration,
    )

    def _append_checkpoint(row: dict) -> None:
        with open(checkpoint_path, "a") as fh:
            fh.write(json.dumps({k: v for k, v in row.items() if not isinstance(v, float) or not (v != v)}) + "\n")

    if workers > 1:
        logger.info("Using %d parallel workers for feature extraction", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker_fn, t): t for t in todo_tracks}
            done_count = 0
            for future in as_completed(futures):
                row = future.result()
                done_count += 1
                if row is not None:
                    results.append(row)
                    _append_checkpoint(row)
                if done_count % 50 == 0 or done_count == len(todo_tracks):
                    elapsed = time.time() - t0
                    eta = elapsed / done_count * max(len(todo_tracks) - done_count, 0)
                    logger.info("[%d/%d] elapsed=%.0fm eta=%.0fm", done_count, len(todo_tracks), elapsed / 60, eta / 60)
    else:
        for i, track in enumerate(todo_tracks, start=1):
            row = worker_fn(track)
            if row is not None:
                results.append(row)
                _append_checkpoint(row)
            if i % 50 == 0 or i == len(todo_tracks):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(todo_tracks) - i)
                logger.info("[%d/%d] elapsed=%.0fm eta=%.0fm", i, len(todo_tracks), elapsed / 60, eta / 60)

    # Merge checkpoint rows from prior runs with current results
    if checkpoint_path.exists() and done_ids:
        with open(checkpoint_path) as fh:
            prior_rows = [json.loads(line) for line in fh if line.strip()]
        prior_ids = {str(r.get("track_id", "")) for r in results}
        results = [r for r in prior_rows if str(r.get("track_id", "")) not in prior_ids] + results

    if not results:
        logger.error("No tracks processed successfully.")
        sys.exit(1)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "acoustic_features.csv", index=False)
    logger.info("Saved acoustic_features.csv: %d rows x %d cols", len(df), len(df.columns))

    logger.info("Running per-feature analysis...")
    analyse_features(df, output_dir)

    logger.info("Running classification...")
    run_classification(df, output_dir, args.ablation_csv)

    if args.compare_raw_csv and args.preprocess_mode in ("canonical", "preprocessed"):
        logger.info("Running confound audit (raw vs canonical/preprocessed)...")
        run_confound_audit(df, args.compare_raw_csv, output_dir)
    elif args.compare_raw_csv:
        logger.warning("--compare-raw-csv only used with canonical/preprocessed mode; skipping audit.")

    logger.info("=" * 60)
    logger.info("Outputs saved to %s", output_dir)
    logger.info("  acoustic_features.csv        per-track feature matrix (+ duration, lufs)")
    logger.info("  acoustic_analysis.csv        AUC / Cohen-d / p-value + per-algo means")
    logger.info("  acoustic_classification.csv  logistic regression AUC summary")
    if args.ablation_csv:
        logger.info("  acoustic_combined.csv        joined with ablation CSV")
    if args.compare_raw_csv and args.preprocess_mode in ("canonical", "preprocessed"):
        logger.info("  confound_audit.csv           raw-vs-canonical AUC comparison")
        logger.info("  confound_resistant_shortlist.csv  features surviving format matching")


if __name__ == "__main__":
    main()
