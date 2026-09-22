"""Profile acoustic covariates for real and fake SONICS audio.

Extracts duration, tempo, key/chroma, LUFS, and spectral centroid for every
track, then produces distribution-gap diagnostics (standardised mean difference,
KS test, effect size) to identify which covariates are imbalanced between the
real and fake classes.

This is a pre-requisite for the balanced sampler: the diagnostics table shows
which variables the sampler must match on to ensure the subsample is not biased.

Outputs (all in --output-dir)
------------------------------
  covariate_profiles.csv       per-track covariate values
  covariate_diagnostics.csv    per-covariate SMD, KS-stat, p-value, imbalance flag
  covariate_distributions.png  distribution comparison plots

Usage
-----
poetry run python scripts/profile_covariates.py \
  --output-dir data/processed/covariate_profile \
  --per-class 500

# From sampler canonical manifest (recommended after build_canonical_for_sampler.py):
poetry run python scripts/profile_covariates.py \
  --canonical-manifest data/processed/canonical_sampler/canonical_manifest.csv \
  --workers 4 \
  --output-dir data/processed/covariate_profile
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.stats

# NOTE: we used to set NUMBA_DISABLE_JIT=1 here to avoid chroma_cqt/beat_track
# segfaults in worker subprocesses. Both of those calls have since been replaced
# (chroma_stft instead of chroma_cqt; librosa.feature.tempo() instead of
# beat_track's numba-jitted DP step), so that workaround is no longer needed —
# and it actively broke things: NUMBA_DISABLE_JIT=1 is incompatible with numba's
# @guvectorize-decorated functions (used internally by resampy, librosa's
# fallback resampler), which is exactly what produced
# "'function' object has no attribute 'get_call_template'" on any track whose
# native sample rate differed from ANALYSIS_SR and therefore needed resampling.
# We force res_type="soxr_hq" below instead (see librosa.load call) so resampy
# is never invoked at all — sidesteps both the original segfault risk and this bug.


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_preprocessing import lowpass_filter, measure_loudness_lufs
from intrinsic_ai_music_detection.data.audio_utils import load_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")
ANALYSIS_SR = 24_000
MAX_DURATION = 120.0

COVARIATE_NAMES = [
    "duration",
    "tempo",
    "key",
    "loudness_lufs",
    "spectral_centroid_mean",
    # Effective bandwidth: frequency at which 90% of cumulative spectral energy is reached.
    # A key diagnostic for the bandwidth confound: AI generators often have a lower
    # effective bandwidth due to vooder upsampling artifacts / delivery-chain band-limiting.
    "effective_bandwidth_hz",
    # Spectral rolloff (sklearn convention: 85% energy) — complementary to eff. bandwidth.
    "spectral_rolloff_85_hz",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_covariates(
    path: str | Path,
    sr: int = ANALYSIS_SR,
    max_dur: float = MAX_DURATION,
    lowpass_hz: float | None = None,
) -> dict:
    """Extract duration, tempo, key, LUFS, and spectral centroid for one track.

    When ``lowpass_hz`` is set, the same low-pass filter applied before embedding
    extraction is applied here, so covariate balance is measured under the
    bandwidth-equalised conditions the classifier actually sees (e.g. 8 kHz to
    neutralise the fake-class bandwidth ceiling).
    """
    path = Path(path)
    result: dict = {"path": str(path)}

    try:
        # Duration (without full load)
        try:
            result["duration"] = librosa.get_duration(path=str(path))
        except Exception:
            result["duration"] = float("nan")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            # res_type="soxr_hq" forces the soxr backend (no numba) instead of
            # librosa's resampy fallback — see module-level note above.
            audio, sr_loaded = librosa.load(str(path), sr=sr, mono=True, duration=max_dur, res_type="soxr_hq")

        audio = audio.astype(np.float32)
        if lowpass_hz:
            audio = lowpass_filter(audio, sr_loaded, cutoff_hz=lowpass_hz)

        # Tempo.
        # NOTE: librosa.beat.beat_track()'s internal DP tempo-estimation step is
        # numba-jitted. With NUMBA_DISABLE_JIT=1 (set above to avoid chroma_cqt
        # segfaults in forked/spawned workers), that jitted function either raises
        # or becomes catastrophically slow, which is why tempo came back 100% NaN
        # in every profiling run to date despite librosa.beat.beat_track working
        # fine in a plain, non-multiprocessing script. librosa.feature.tempo()
        # (autocorrelation/tempogram-based, no numba DP) avoids this entirely and
        # is the librosa-recommended replacement for librosa.beat.tempo() anyway.
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr_loaded)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr_loaded)
            result["tempo"] = float(np.atleast_1d(tempo)[0])
        except Exception as exc:
            result["tempo"] = float("nan")
            result["tempo_error"] = str(exc)[:200]

        # Key (chromagram → argmax of mean chroma).
        # Use chroma_stft (STFT-based) instead of chroma_cqt (CQT/numba-based)
        # to avoid numba JIT segfaults in forked processes.
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr_loaded)
            mean_chroma = chroma.mean(axis=1)
            result["key"] = int(np.argmax(mean_chroma))
        except Exception as exc:
            result["key"] = float("nan")
            result["key_error"] = str(exc)[:200]

        # LUFS
        result["loudness_lufs"] = measure_loudness_lufs(audio, sr_loaded)

        # Spectral centroid
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr_loaded).ravel()
            result["spectral_centroid_mean"] = float(np.nanmean(centroid))
        except Exception as exc:
            result["spectral_centroid_mean"] = float("nan")
            result["spectral_centroid_error"] = str(exc)[:200]

        # Effective bandwidth (frequency at which 90% cumulative spectral energy is reached)
        # and spectral rolloff at 85% — both robust bandwidth confound diagnostics.
        try:
            D = np.abs(librosa.stft(audio))  # [freq_bins, time_frames]
            power = D**2
            mean_power = power.mean(axis=1)  # avg over time -> [freq_bins]
            cum_power = np.cumsum(mean_power)
            total_power = cum_power[-1]
            if total_power > 0:
                freqs = librosa.fft_frequencies(sr=sr_loaded, n_fft=D.shape[0] * 2 - 2)
                thresh_90 = np.searchsorted(cum_power, 0.90 * total_power)
                result["effective_bandwidth_hz"] = float(freqs[min(thresh_90, len(freqs) - 1)])
                thresh_85 = np.searchsorted(cum_power, 0.85 * total_power)
                result["spectral_rolloff_85_hz"] = float(freqs[min(thresh_85, len(freqs) - 1)])
            else:
                result["effective_bandwidth_hz"] = float("nan")
                result["spectral_rolloff_85_hz"] = float("nan")
        except Exception:
            result["effective_bandwidth_hz"] = float("nan")
            result["spectral_rolloff_85_hz"] = float("nan")

    except Exception as exc:
        logger.warning("Failed %s: %s", path.name, exc)

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def compute_covariate_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-covariate imbalance diagnostics between real and fake classes.

    Returns a DataFrame with one row per covariate containing:
      - mean_real, mean_fake, std_real, std_fake
      - smd: standardised mean difference (absolute)
      - ks_stat, ks_pvalue: Kolmogorov-Smirnov two-sample test
      - imbalanced: True when |SMD| > 0.1 (threshold from propensity-score literature)
    """
    real = df[df["label"] == "real"]
    fake = df[df["label"] == "fake"]

    rows = []
    for cov in COVARIATE_NAMES:
        if cov not in df.columns:
            continue
        x_r = pd.to_numeric(real[cov], errors="coerce").dropna().to_numpy(dtype=float)
        x_f = pd.to_numeric(fake[cov], errors="coerce").dropna().to_numpy(dtype=float)

        if len(x_r) < 5 or len(x_f) < 5:
            continue

        pooled_std = np.sqrt((np.var(x_r, ddof=1) + np.var(x_f, ddof=1)) / 2)
        smd = abs(np.mean(x_r) - np.mean(x_f)) / (pooled_std + 1e-12)
        ks_stat, ks_pval = scipy.stats.ks_2samp(x_r, x_f)

        rows.append(
            {
                "covariate": cov,
                "n_real": len(x_r),
                "n_fake": len(x_f),
                "mean_real": float(np.mean(x_r)),
                "mean_fake": float(np.mean(x_f)),
                "std_real": float(np.std(x_r, ddof=1)),
                "std_fake": float(np.std(x_f, ddof=1)),
                "smd": float(smd),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "imbalanced": bool(smd > 0.10 or ks_pval < 0.05),
            }
        )

    return pd.DataFrame(rows)


def _print_diagnostics(diag_df: pd.DataFrame) -> None:
    logger.info("=" * 80)
    logger.info("Covariate balance diagnostics (real vs fake)")
    logger.info("SMD threshold: 0.10 | KS p-value threshold: 0.05")
    logger.info("-" * 80)
    logger.info("%-30s %6s %7s %7s %6s  %5s", "covariate", "n_real", "n_fake", "SMD", "KS_p", "flag")
    for _, r in diag_df.iterrows():
        flag = "IMBALANCED" if r["imbalanced"] else "ok"
        logger.info(
            "%-30s %6d %7d %7.3f %6.4f  %s",
            r["covariate"],
            int(r["n_real"]),
            int(r["n_fake"]),
            r["smd"],
            r["ks_pvalue"],
            flag,
        )
    n_imb = diag_df["imbalanced"].sum()
    if n_imb:
        logger.warning("%d covariate(s) are imbalanced — must be matched in sampler!", n_imb)
    else:
        logger.info("All covariates within balance threshold.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Track loaders
# ---------------------------------------------------------------------------


def _load_track_list(manifest_csv: str | None, per_class: int | None, seed: int) -> list[dict]:
    if manifest_csv and Path(manifest_csv).exists():
        df = pd.read_csv(manifest_csv, low_memory=False)
        ok_mask = df.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        df = df[ok_mask]
        real = df[df["label"] == "real"]
        fake = df[df["label"] == "fake"]
        rng = np.random.default_rng(seed)
        if per_class:
            real = real.sample(min(per_class, len(real)), random_state=seed)
            fake = fake.sample(min(per_class, len(fake)), random_state=seed)

        tracks = []
        for _, r in real.iterrows():
            path = r.get("canonical_path", r.get("src_path", ""))
            if path and Path(path).exists():
                tracks.append(
                    {
                        "track_id": str(r.get("track_id", "")),
                        "path": str(path),
                        "label": "real",
                        "algorithm": "",
                        "genre": "",
                    }
                )
        for _, r in fake.iterrows():
            path = r.get("canonical_path", r.get("src_path", ""))
            if path and Path(path).exists():
                tracks.append(
                    {
                        "track_id": str(r.get("track_id", "")),
                        "path": str(path),
                        "label": "fake",
                        "algorithm": str(r.get("algorithm", "")),
                        "genre": str(r.get("genre", "")),
                    }
                )
        return tracks

    # Fallback: raw SONICS metadata
    metadata_dir = SONICS_DIR / "metadata"
    real_csv = metadata_dir / "real_songs_ready.csv"
    if not real_csv.exists():
        real_csv = metadata_dir / "real_songs.csv"
    fake_csv = metadata_dir / "fake_songs_ready.csv"
    if not fake_csv.exists():
        fake_csv = metadata_dir / "fake_songs.csv"

    if not real_csv.exists() or not fake_csv.exists():
        logger.error("Metadata CSVs not found.")
        sys.exit(1)

    real_dir = SONICS_DIR / "real_songs"
    fake_dir = SONICS_DIR / "fake_songs"

    df_real = pd.read_csv(real_csv, low_memory=False)
    df_fake = pd.read_csv(fake_csv, low_memory=False)

    if per_class:
        df_real = df_real.sample(min(per_class, len(df_real)), random_state=seed)
        df_fake = df_fake.sample(min(per_class, len(df_fake)), random_state=seed)

    tracks = []
    for _, row in df_real.iterrows():
        yt_id = str(row.get("youtube_id", "")).strip()
        cands = list(real_dir.glob(f"{yt_id}.*")) if real_dir.exists() else []
        if cands:
            tracks.append({"track_id": yt_id, "path": str(cands[0]), "label": "real", "algorithm": "", "genre": ""})

    for _, row in df_fake.iterrows():
        fname = str(row.get("filename", row.get("id", ""))).strip()
        stem = Path(fname).stem
        cands = list(fake_dir.glob(f"{stem}.*")) if fake_dir.exists() else []
        if cands:
            tracks.append(
                {
                    "track_id": stem,
                    "path": str(cands[0]),
                    "label": "fake",
                    "algorithm": str(row.get("algorithm", "")),
                    "genre": str(row.get("genre", "")),
                }
            )

    return tracks


# ---------------------------------------------------------------------------
# Worker (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _process_covariate_track(track: dict) -> dict:
    """Extract covariates for one track. Must be top-level for multiprocessing."""
    try:
        covs = extract_covariates(
            track["path"],
            max_dur=float(track.get("max_dur", MAX_DURATION)),
            lowpass_hz=track.get("lowpass_hz"),
        )
        return {
            "track_id": track["track_id"],
            "label": track["label"],
            "algorithm": track.get("algorithm", ""),
            "genre": track.get("genre", ""),
            **covs,
        }
    except Exception:
        # Return a minimal row on any unhandled error so processing continues
        return {
            "track_id": track["track_id"],
            "label": track["label"],
            "algorithm": track.get("algorithm", ""),
            "genre": track.get("genre", ""),
            "path": str(track.get("path", "")),
            "error": traceback.format_exc()[-200:],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile acoustic covariates for real and fake SONICS audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/processed/covariate_profile")
    parser.add_argument("--canonical-manifest", default=None, help="Use canonical manifest for track paths")
    parser.add_argument("--per-class", type=int, default=None, help="Max tracks per class")
    parser.add_argument("--plot", action="store_true", default=True, help="Save distribution plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Parallel worker processes. "
            "librosa is CPU-bound; on g4dn.xlarge (4 vCPUs) use 2-4. "
            "Each worker loads ~50-100 MB of audio; 4 workers ≈ 400 MB peak RAM."
        ),
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=None,
        help=(
            "Apply this low-pass cutoff (Hz) before measuring covariates so balance "
            "is profiled under the same bandwidth-equalised conditions used for embedding "
            "extraction. Use 8000 to neutralise the SONICS fake-class 8 kHz ceiling."
        ),
    )
    parser.add_argument(
        "--analysis-duration",
        type=float,
        default=None,
        help=(
            "Cap audio to this many seconds before measuring covariates (matches "
            "run_balanced_ablation --analysis-duration). Default: full 120 s."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "covariate_checkpoint.jsonl"

    # Resume: load already-processed track IDs from checkpoint.
    # IMPORTANT: a track_id is only considered "done" if its checkpointed row
    # has valid (non-NaN) values for the covariates that are known to have
    # failed silently in past runs (tempo / key / spectral_centroid_mean —
    # see the NUMBA_DISABLE_JIT note in extract_covariates). Otherwise a stale
    # row from a pre-fix run would permanently block reprocessing even after
    # the extraction bug is fixed, which is exactly what happened before
    # (tempo ended up 100% NaN, key/centroid ~71% NaN across the whole corpus).
    STALE_CHECK_COLS = ["tempo", "key", "spectral_centroid_mean"]
    done_ids: set[str] = set()
    n_stale = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                tid = str(row.get("track_id", ""))
                is_stale = all(
                    row.get(c) is None or (isinstance(row.get(c), float) and np.isnan(row.get(c)))
                    for c in STALE_CHECK_COLS
                )
                if is_stale:
                    n_stale += 1
                    continue
                done_ids.add(tid)
        logger.info(
            "Resuming: %d tracks already in checkpoint with valid tempo/key/centroid "
            "(%d stale rows will be reprocessed)",
            len(done_ids),
            n_stale,
        )

    tracks = _load_track_list(args.canonical_manifest, args.per_class, args.seed)
    # Propagate analysis conditions into each track so the multiprocessing
    # workers profile covariates under the same bandwidth / duration the
    # classifier sees.
    max_dur_eff = args.analysis_duration if args.analysis_duration else MAX_DURATION
    for t in tracks:
        t["max_dur"] = max_dur_eff
        t["lowpass_hz"] = args.lowpass_hz
    if args.lowpass_hz:
        logger.info("Low-pass   : %.0f Hz (bandwidth equalisation before profiling)", args.lowpass_hz)
    if args.analysis_duration:
        logger.info("Duration   : capped at %.0f s", args.analysis_duration)
    todo = [t for t in tracks if str(t.get("track_id", "")) not in done_ids]
    logger.info(
        "Loaded %d tracks (%d real, %d fake)  |  %d todo",
        len(tracks),
        sum(1 for t in tracks if t["label"] == "real"),
        sum(1 for t in tracks if t["label"] == "fake"),
        len(todo),
    )

    def _checkpoint_append(row: dict) -> None:
        with open(checkpoint_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    results: list[dict] = []
    done_count = 0

    if args.workers > 1:
        logger.info("Using %d parallel workers (spawn context)", args.workers)
        # Use 'spawn' instead of 'fork' to avoid inheriting potentially broken
        # state from the parent process (numba JIT cache, OpenBLAS threads, etc.)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_process_covariate_track, t): t for t in todo}
            for future in as_completed(futures):
                try:
                    row = future.result(timeout=120)
                except Exception as exc:
                    logger.warning("Worker failed: %s", exc)
                    continue
                done_count += 1
                results.append(row)
                _checkpoint_append(row)
                if done_count % 50 == 0 or done_count == len(todo):
                    logger.info("[%d/%d] (errors in row['error'] if any)", done_count, len(todo))
    else:
        logger.info("Serial processing (1 worker)")
        for i, track in enumerate(todo, start=1):
            try:
                row = _process_covariate_track(track)
            except Exception as exc:
                logger.warning("Track %s failed: %s", track.get("track_id"), exc)
                continue
            results.append(row)
            _checkpoint_append(row)
            done_count += 1
            if i % 10 == 0 or i == len(todo):
                logger.info("[%d/%d]", i, len(todo))

    # Merge with prior checkpoint rows
    if checkpoint_path.exists() and done_ids:
        with open(checkpoint_path) as fh:
            prior_rows = [json.loads(line) for line in fh if line.strip()]
        current_ids = {str(r.get("track_id", "")) for r in results}
        results = [r for r in prior_rows if str(r.get("track_id", "")) not in current_ids] + results

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "covariate_profiles.csv", index=False)
    logger.info("Saved covariate_profiles.csv (%d rows)", len(df))

    diag_df = compute_covariate_diagnostics(df)
    diag_df.to_csv(output_dir / "covariate_diagnostics.csv", index=False)
    logger.info("Saved covariate_diagnostics.csv")
    _print_diagnostics(diag_df)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            n_covs = len(COVARIATE_NAMES)
            fig, axes = plt.subplots(1, n_covs, figsize=(4 * n_covs, 4))
            if n_covs == 1:
                axes = [axes]
            for ax, cov in zip(axes, COVARIATE_NAMES):
                if cov not in df.columns:
                    continue
                real_vals = pd.to_numeric(df[df["label"] == "real"][cov], errors="coerce").dropna()
                fake_vals = pd.to_numeric(df[df["label"] == "fake"][cov], errors="coerce").dropna()
                ax.hist(real_vals, bins=40, alpha=0.5, label="real", density=True)
                ax.hist(fake_vals, bins=40, alpha=0.5, label="fake", density=True)
                ax.set_title(cov)
                ax.legend(fontsize=7)
            plt.tight_layout()
            fig.savefig(output_dir / "covariate_distributions.png", dpi=120)
            plt.close(fig)
            logger.info("Saved covariate_distributions.png")
        except Exception as exc:
            logger.warning("Could not save plots: %s", exc)


if __name__ == "__main__":
    main()
