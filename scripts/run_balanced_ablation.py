"""Balanced multi-model ablation experiment.

Samples tracks balanced across (algorithm × fake_label) strata, then runs:
  1. Static ID at three scales (full / half / quarter embedding subsampling)
  2. Temporal windowed ID trajectory statistics  (opt-in via --with-temporal)
  3. Classification for every feature bundle
  4. Stratified evaluation by source, fake_label, algorithm

Outputs (all in --output-dir)
------------------------------
  ablation_features.csv          wide-format per-track feature table
  ablation_features_{emb}.csv    per-embedding intermediate CSV
  classification_ablation.csv    CV metrics per bundle, sorted by AUC
  interpretability_ablation.json logistic coef mean / std / sign_consistency
  stratified_eval_ablation.csv   per-source / fake_label / algorithm AUC
  stratum_coverage.csv           tracks sampled per (algorithm × fake_label)
  checkpoint_{emb}.json          per-embedding resume file

Timing (approx, g4dn.xlarge GPU)
----------------------------------
  --per-stratum 200 --embeddings encodec mert                    ~22 h
  --per-stratum 200 --embeddings encodec mert --with-temporal    ~75 h (3 days)
  --per-stratum 200 --embeddings all                             ~48 h (2 days)
  --per-stratum 50  --embeddings encodec          # smoke test    ~2 h

Usage
-----
# Quick validation — encodec + mert, static only (~22 h):
poetry run python scripts/run_balanced_ablation.py \\
  --embeddings encodec mert --per-stratum 200 \\
  --device cuda --output-dir data/processed/sonics_balanced_ablation

# Full ablation with temporal (encodec + mert, ~3 days):
poetry run python scripts/run_balanced_ablation.py \\
  --embeddings encodec mert --per-stratum 200 --with-temporal \\
  --device cuda --output-dir data/processed/sonics_balanced_ablation

# Full 4-embedding static ablation (~2 days):
poetry run python scripts/run_balanced_ablation.py \\
  --embeddings all --per-stratum 200 \\
  --device cuda --output-dir data/processed/sonics_balanced_ablation

# Smoke test (~2 h):
poetry run python scripts/run_balanced_ablation.py \\
  --embeddings encodec --per-stratum 50 \\
  --device cuda --output-dir data/processed/sonics_ablation_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_preprocessing import preprocess_audio
from intrinsic_ai_music_detection.data.audio_utils import (
    center_crop,
    load_audio,
    normalize_audio,
    region_crop,
    sliding_window,
)
from intrinsic_ai_music_detection.data.balanced_sampling import load_balanced_tracks as _load_balanced_tracks
from intrinsic_ai_music_detection.features.cache_keys import (
    band_match_tag,
    embedding_cache_key,
    level_match_tag,
    spec_floor_tag,
)
from intrinsic_ai_music_detection.features.embeddings import get_extractor
from intrinsic_ai_music_detection.features.id_estimators import estimate_all
from intrinsic_ai_music_detection.models.evaluate import equal_error_rate
from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")

DEFAULT_ESTIMATORS: list[str] = ["phd", "twonn"]
ALL_ESTIMATORS: list[str] = ["phd", "twonn", "mle"]

EMBEDDING_CONFIGS: dict[str, dict] = {
    "encodec": {"target_sr": 24_000, "metric": "euclidean"},
    # Classical log-mel front end: 24 kHz, 75 fps, 128-d — rate- and
    # dimension-matched to encodec so `--embeddings spec` is a controlled swap
    # of the representation only (MusicDET's spectral domain, our pipeline).
    "spec": {"target_sr": 24_000, "metric": "euclidean"},
    "logmel": {"target_sr": 24_000, "metric": "euclidean"},
    # MusicDET's ACTUAL front end, read from their source (model.py:611-654):
    # 16 kHz, n_fft 512, hop 160 (=100 fps), power=2 LINEAR STFT cropped to
    # bins [0:256] over (0, 8 kHz). Linear (not mel) is the point: mel
    # compresses high frequencies logarithmically, averaging away the
    # fine band-edge structure where codec/vocoder artifacts live. 256 is
    # divisible by 2/4/8, so --wf-n-bands works on this arm (257 would not).
    "spec-musicdet": {"target_sr": 16_000, "metric": "euclidean"},
    # The one-class flow, trained on reals only, INSIDE the deconvolution-comb
    # space rather than a generic embedding. Every other front end hands the flow
    # a representation whose likelihood is dominated by corpus identity; this one
    # hands it a mechanistic artifact signature (Afchar et al. arXiv:2506.19108),
    # so "trained only on real music" and "measures synthesis" can hold at once.
    "combprint": {"target_sr": 16_000, "metric": "euclidean"},
    "clap": {"target_sr": 48_000, "metric": "cosine"},
    "mert": {"target_sr": 24_000, "metric": "cosine"},
    "mert-95m": {"target_sr": 24_000, "metric": "cosine"},
    "muq": {"target_sr": 24_000, "metric": "cosine"},
    "xls-r": {"target_sr": 16_000, "metric": "cosine"},
}

# ---------------------------------------------------------------------------
# Balanced sampling (delegates to shared module in balanced_sampling.py)
# ---------------------------------------------------------------------------


def load_balanced_tracks(
    per_stratum: int,
    fake_types: list[str] | None = None,
    algorithms: list[str] | None = None,
    max_real: int | None = None,
    seed: int = 42,
    real_genres_csv: str | None = None,
    covariate_profiles_csv: str | None = None,
    match_covariates: bool = True,
    canonical_manifest: str | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Sample balanced fake tracks, genre-matched + covariate-matched reals.

    Returns (real_tracks, fake_tracks, coverage_df).
    Uses the shared :mod:`intrinsic_ai_music_detection.data.balanced_sampling`
    module; this wrapper also computes stratum coverage.
    """
    real_tracks, fake_tracks = _load_balanced_tracks(
        per_stratum=per_stratum,
        fake_types=fake_types,
        algorithms=algorithms,
        max_real=max_real,
        seed=seed,
        real_genres_csv=real_genres_csv,
        covariate_profiles_csv=covariate_profiles_csv,
        match_covariates=match_covariates,
        canonical_manifest=canonical_manifest,
    )

    # Build coverage summary
    fake_df = pd.DataFrame(fake_tracks)
    coverage_rows: list[dict] = []
    if not fake_df.empty:
        for (algo, fl), grp in fake_df.groupby(["algorithm", "fake_label"], dropna=False, sort=True):
            coverage_rows.append(
                {
                    "algorithm": algo,
                    "fake_label": fl,
                    "n_sampled": len(grp),
                }
            )

    coverage_df = pd.DataFrame(coverage_rows)
    return real_tracks, fake_tracks, coverage_df


# ---------------------------------------------------------------------------
# Per-track feature computation
# ---------------------------------------------------------------------------


def _multiscale_id(
    emb_matrix: np.ndarray,
    metric: str,
    estimators: list[str],
    rng: np.random.Generator,
    max_points: int = 2000,
) -> dict[str, float]:
    """Compute ID at full / half / quarter scale by subsampling rows of emb_matrix.

    Scale semantics
    ---------------
    full    — all N embedding vectors (capped at max_points for speed)
    half    — random N//2 vectors  (local neighbourhood, ~60s audio equivalent)
    quarter — random N//4 vectors  (very local,          ~30s audio equivalent)

    The quarter-scale PHD separation being stronger than full-scale would indicate
    that AI artifacts are concentrated at the phrase level, not the track level.
    """
    result: dict[str, float] = {}
    finite = emb_matrix[np.isfinite(emb_matrix).all(axis=1)]
    n = len(finite)

    for scale_name, k in [("full", n), ("half", max(n // 2, 0)), ("quarter", max(n // 4, 0))]:
        if k < 10:
            continue
        sub = finite if k == n else finite[np.sort(rng.choice(n, size=k, replace=False))]
        ids = estimate_all(sub, metric=metric, methods=estimators, max_points=max_points)
        for est, val in ids.items():
            result[f"id_{est}_{scale_name}"] = val

    result["n_embeddings_full"] = n
    return result


def _temporal_id_from_matrix(
    emb_matrix: np.ndarray,
    audio_duration: float,
    metric: str,
    estimators: list[str],
    window_duration: float,
    hop_duration: float,
    temporal_max_points: int = 300,
) -> dict[str, float]:
    """Compute temporal ID by slicing the pre-extracted embedding matrix.

    Zero additional model forward passes — the matrix is already computed
    for static multiscale ID, so we just split it temporally.

    ``temporal_max_points`` caps the point cloud inside each window before ID
    estimation.  PHD distance-matrix cost is O(n²), so halving n cuts wall-time
    by ~4×.  300 points is sufficient for relative trajectory comparisons; use
    the full static ``max_points`` for absolute ID values.
    """
    n_frames = len(emb_matrix)
    if n_frames < 2 or audio_duration <= 0:
        return {}

    fps = n_frames / audio_duration  # e.g. ~75 Hz for MERT/EnCodec
    window_frames = max(int(window_duration * fps), 10)
    hop_frames = max(int(hop_duration * fps), 1)

    if window_frames >= n_frames:
        # Adaptive fallback for short tracks (e.g. udio-30s with 30s window):
        # use 1/3 of the track as window, 1/6 as hop → at least 3 windows.
        window_frames = max(n_frames // 3, 10)
        hop_frames = max(window_frames // 2, 1)
        if window_frames < 10:
            return {}

    # Lightweight PHD config for per-window estimates: fewer reruns and
    # intermediate points — we care about the *shape* of the trajectory, not
    # a maximally accurate absolute value at each window.
    from intrinsic_ai_music_detection.config import PHDConfig

    fast_phd_cfg = PHDConfig(n_reruns=1, intermediate_points=5)

    win_ids: dict[str, list[float]] = {est: [] for est in estimators}
    times: list[float] = []

    start = 0
    while start + window_frames <= n_frames:
        sub = emb_matrix[start : start + window_frames]
        sub = sub[np.isfinite(sub).all(axis=1)]
        times.append((start + window_frames / 2) / fps)
        if len(sub) < 5:
            for est in estimators:
                win_ids[est].append(np.nan)
        else:
            try:
                ids = estimate_all(
                    sub,
                    metric=metric,
                    methods=estimators,
                    max_points=temporal_max_points,
                    phd_cfg=fast_phd_cfg,
                )
                for est in estimators:
                    win_ids[est].append(ids.get(est, np.nan))
            except Exception:
                for est in estimators:
                    win_ids[est].append(np.nan)
        start += hop_frames

    if not times:
        return {}

    result: dict[str, float] = {}
    times_arr = np.array(times)
    for est, values in win_ids.items():
        arr = np.array(values, dtype=float)
        finite_mask = np.isfinite(arr)
        finite = arr[finite_mask]
        if len(finite) < 2:
            continue
        t_finite = times_arr[finite_mask]
        # --- summary statistics ---
        result[f"temporal_{est}_mean"] = float(np.mean(finite))
        result[f"temporal_{est}_std"] = float(np.std(finite))
        result[f"temporal_{est}_range"] = float(np.max(finite) - np.min(finite))
        result[f"temporal_{est}_slope"] = float(np.polyfit(t_finite, finite, 1)[0])
        result[f"temporal_{est}_first_last"] = float(finite[-1] - finite[0])
        result[f"temporal_{est}_n_windows"] = int(len(finite))
        result[f"temporal_{est}_window_ids"] = json.dumps([round(v, 4) if np.isfinite(v) else None for v in values])

        # --- transition entropy / HMM / change-point extensions ---
        try:
            from intrinsic_ai_music_detection.features.temporal_features import all_temporal_features

            ext = all_temporal_features(values, run_hmm=True, run_changepoint=True)
            for feat_name, feat_val in ext.items():
                result[f"temporal_{est}_{feat_name}"] = feat_val
        except Exception as exc:
            logger.debug("Temporal extension features failed for %s: %s", est, exc)

    return result


def _cached_audio_duration(
    cache_path: Path,
    emb_matrix: np.ndarray,
    fixed_duration: float | None,
) -> float:
    """Reconstruct a track's audio duration for a cached embedding matrix.

    Resolution order:
      1. ``fixed_duration`` when set (exact — every cached track shares it);
      2. a ``.json`` sidecar written alongside the ``.npy`` cache;
      3. a ~75 Hz frame-rate approximation (legacy caches without sidecars).
    """
    if fixed_duration:
        return float(fixed_duration)
    sidecar = cache_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return float(json.loads(sidecar.read_text())["audio_duration"])
        except Exception:  # noqa: BLE001 — fall through to approximation
            pass
    return float(emb_matrix.shape[0] / 75.0)


def _process_track(
    track: dict,
    extractor,
    target_sr: int,
    metric: str,
    estimators: list[str],
    max_duration: float,
    with_temporal: bool,
    window_duration: float,
    hop_duration: float,
    rng: np.random.Generator,
    preprocess_mode: str = "raw",
    embedding_cache_dir: str | None = None,
    max_points: int = 2000,
    lowpass_hz: float | None = None,
    temporal_max_points: int = 300,
    fixed_duration: float | None = None,
    analysis_region: str = "full",
    cache_dtype: str = "float32",
    cache_extra_tags: tuple[str, ...] = (),
    resample_hz: float | None = None,
    level_match: bool = False,
    level_peak: bool = False,
) -> dict | None:
    """Load audio once, compute static multiscale + temporal ID.

    Zero additional extractor calls for temporal — we slice the embedding
    matrix that was already computed for the static multiscale features.

    When ``fixed_duration`` is set, the audio is centre-cropped (or padded) to
    exactly that length before embedding extraction so that every track yields
    the same number of windows (removes the duration / window-count confound).
    """
    try:
        # Check embedding cache — the key must name every knob that changes the
        # stored array (lowpass, fixed duration, region, extractor overrides), or
        # two configurations silently read each other's vectors. Single source of
        # truth in features/cache_keys.py; historical keys are unchanged.
        if embedding_cache_dir:
            cache_key = embedding_cache_key(
                str(track["track_id"]),
                target_sr,
                max_duration,
                preprocess_mode,
                lowpass_hz=lowpass_hz,
                fixed_duration=fixed_duration,
                analysis_region=analysis_region,
                extra_tags=cache_extra_tags,
            )
            cache_path = Path(embedding_cache_dir) / f"{cache_key}.npy"
            if cache_path.exists():
                emb_matrix = np.load(str(cache_path)).astype(np.float32, copy=False)
                audio_duration = _cached_audio_duration(cache_path, emb_matrix, fixed_duration)
                result: dict = {**track, "n_embeddings": len(emb_matrix)}
                # Skip to ID computation
                return _compute_ids_from_emb(
                    result,
                    emb_matrix,
                    audio_duration,
                    metric,
                    estimators,
                    with_temporal,
                    window_duration,
                    hop_duration,
                    rng,
                    max_points=max_points,
                    temporal_max_points=temporal_max_points,
                )

        if preprocess_mode == "canonical":
            audio, sr, _ = preprocess_audio(
                track["path"],
                target_sr=target_sr,
                mode="canonical",
                max_duration=max_duration,
                lowpass_hz=lowpass_hz,
                resample_hz=resample_hz,
            )
        elif preprocess_mode == "preprocessed":
            audio, sr = load_audio(track["path"], target_sr=target_sr, max_duration=max_duration)
            if level_match:
                from intrinsic_ai_music_detection.data.audio_preprocessing import level_match as _level_match

                # Before the bandwidth control: trimming silence changes the
                # loudness of what remains, so it must come first.
                audio = _level_match(audio, sr, peak_normalise=level_peak)
            if lowpass_hz or resample_hz:
                from intrinsic_ai_music_detection.data.audio_preprocessing import bandwidth_match

                audio = bandwidth_match(audio, sr, resample_hz=resample_hz, lowpass_hz=lowpass_hz)
        else:
            audio, sr = load_audio(track["path"], target_sr=target_sr, max_duration=max_duration)
            if level_match:
                from intrinsic_ai_music_detection.data.audio_preprocessing import level_match as _level_match

                # Before the bandwidth control: trimming silence changes the
                # loudness of what remains, so it must come first.
                audio = _level_match(audio, sr, peak_normalise=level_peak)
            if lowpass_hz or resample_hz:
                from intrinsic_ai_music_detection.data.audio_preprocessing import bandwidth_match

                audio = bandwidth_match(audio, sr, resample_hz=resample_hz, lowpass_hz=lowpass_hz)
            audio = normalize_audio(audio)

        # Region selection: extract the specified portion of the track.
        # For 'middle' this is equivalent to centre_crop (legacy default).
        # For 'intro'/'outro'/'random' we select a different region before
        # embedding, enabling the Phase-1 temporal sweep.
        if fixed_duration:
            if analysis_region == "full" or analysis_region == "middle":
                audio = center_crop(audio, sr, fixed_duration)
            else:
                audio = region_crop(audio, sr, analysis_region, fixed_duration, rng=rng)
        elif analysis_region != "full" and max_duration:
            # No fixed_duration but region requested: crop to max_duration from region.
            audio = region_crop(audio, sr, analysis_region, max_duration, rng=rng)

        audio_duration = len(audio) / sr
        emb_matrix = extractor.extract(audio, sr)  # ONE forward pass for the entire track
        del audio

        # Save to cache (+ sidecar storing the exact audio duration so the
        # temporal frame-rate is reconstructed correctly on cache hits, across
        # representations with different frame rates).
        if embedding_cache_dir:
            Path(embedding_cache_dir).mkdir(parents=True, exist_ok=True)
            # float16 halves cache size at negligible cost here: features are
            # standardised before the flow sees them, and float16 still carries
            # ~3 decimal digits. Matters because a 256-d @ 100 fps cache is
            # ~2.7x the 128-d @ 75 fps one and does not otherwise fit on disk.
            np.save(str(cache_path), emb_matrix.astype(cache_dtype, copy=False))
            try:
                cache_path.with_suffix(".json").write_text(json.dumps({"audio_duration": float(audio_duration)}))
            except Exception:  # noqa: BLE001 — sidecar is a best-effort optimisation
                pass

        result: dict = {**track, "n_embeddings": len(emb_matrix)}
        return _compute_ids_from_emb(
            result,
            emb_matrix,
            audio_duration,
            metric,
            estimators,
            with_temporal,
            window_duration,
            hop_duration,
            rng,
            max_points=max_points,
            temporal_max_points=temporal_max_points,
        )
    except Exception as e:
        logger.warning("Failed %s: %s: %s", track["track_id"], type(e).__name__, e)
        return None


def _compute_ids_from_emb(
    result: dict,
    emb_matrix: np.ndarray,
    audio_duration: float,
    metric: str,
    estimators: list[str],
    with_temporal: bool,
    window_duration: float,
    hop_duration: float,
    rng: np.random.Generator,
    max_points: int = 2000,
    temporal_max_points: int = 300,
) -> dict | None:
    """Compute static + temporal IDs from an already-extracted embedding matrix."""
    # --flow-only passes estimators=[] so the expensive intrinsic-dimension work
    # never runs. The gate below assumes at least one estimator produced a
    # value, so with an empty list it returned None for EVERY track and the run
    # ended with "No results produced" before the window-flow eval could start.
    # The window-flow eval needs only the manifest row plus the embedding cache,
    # so the bare result is the correct return here.
    if estimators:
        ms = _multiscale_id(emb_matrix, metric, estimators, rng, max_points=max_points)
        if not ms or "id_phd_full" not in ms and "id_twonn_full" not in ms:
            return None
        result.update(ms)

    if with_temporal:
        temporal = _temporal_id_from_matrix(
            emb_matrix,
            audio_duration,
            metric,
            estimators,
            window_duration,
            hop_duration,
            temporal_max_points=temporal_max_points,
        )
        result.update(temporal)

        # Model-agnostic geometric window descriptors (non-ID): same sliding
        # window, cheap linear-algebra + trajectory dynamics. Recomputable from
        # the embedding cache via --recompute-features.
        try:
            from intrinsic_ai_music_detection.features.window_descriptors import window_descriptor_features

            result.update(window_descriptor_features(emb_matrix, audio_duration, window_duration, hop_duration))
        except Exception as exc:  # noqa: BLE001 — descriptors are additive features
            logger.debug("Window descriptors failed: %s", exc)

    del emb_matrix
    return result


def _recompute_descriptors_for_df(
    df: pd.DataFrame,
    emb_name: str,
    *,
    embedding_cache_dir: str,
    target_sr: int,
    max_duration: float,
    preprocess_mode: str,
    lowpass_hz: float | None,
    fixed_duration: float | None,
    window_duration: float,
    hop_duration: float,
    cache_extra_tags: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Recompute ONLY window descriptors from the embedding cache and merge them
    into an existing per-embedding feature table.

    Reuses every saved PHD / TwoNN / temporal column unchanged (no model forward
    pass, no ID recomputation). Existing ``{emb}_wd_*`` columns are dropped and
    cleanly replaced. Tracks whose embedding is not in the cache keep NaN
    descriptors. Makes descriptor iteration near-instant on a fixed cache.
    """
    from intrinsic_ai_music_detection.features.window_descriptors import window_descriptor_features

    wd_cols = [c for c in df.columns if c.startswith(f"{emb_name}_wd_")]
    if wd_cols:
        df = df.drop(columns=wd_cols)

    # Per-model cache namespace (see run_embedding) — must match the extraction path.
    cache_dir = Path(embedding_cache_dir) / emb_name
    new_rows: list[dict] = []
    n_ok = 0
    for track_id in df["track_id"].astype(str):
        # NOTE: analysis_region is deliberately left at "full" here to reproduce
        # this path's historical key exactly (it never carried the region tag).
        # Descriptor recompute on a region-cropped cache was therefore already a
        # miss, not a collision — it reads nothing rather than the wrong thing.
        cache_key = embedding_cache_key(
            track_id,
            target_sr,
            max_duration,
            preprocess_mode,
            lowpass_hz=lowpass_hz,
            fixed_duration=fixed_duration,
            extra_tags=cache_extra_tags,
        )
        cache_path = cache_dir / f"{cache_key}.npy"
        feats: dict = {}
        if cache_path.exists():
            try:
                emb_matrix = np.load(str(cache_path)).astype(np.float32, copy=False)
                audio_duration = _cached_audio_duration(cache_path, emb_matrix, fixed_duration)
                raw = window_descriptor_features(emb_matrix, audio_duration, window_duration, hop_duration)
                feats = {f"{emb_name}_{k}": v for k, v in raw.items()}
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — keep NaN row, keep going
                logger.debug("descriptors-only failed for %s: %s", track_id, exc)
        new_rows.append(feats)

    wd_df = pd.DataFrame(new_rows, index=df.index)
    logger.info("  %s: descriptors recomputed for %d/%d tracks (cache hits)", emb_name, n_ok, len(df))
    if n_ok < len(df) * 0.99:
        logger.warning(
            "  !!! %s: cache is INCOMPLETE (%d/%d hits) — %d tracks now have NaN descriptors. "
            "This usually means the embedding cache did not fully fit on disk. Point "
            "--embedding-cache-dir at the COMPLETE cache (or re-extract) before trusting these features.",
            emb_name,
            n_ok,
            len(df),
            len(df) - n_ok,
        )
    return pd.concat([df, wd_df], axis=1)


# ---------------------------------------------------------------------------
# Embedding-level orchestration
# ---------------------------------------------------------------------------


class _LazyExtractor:
    """Defer loading the embedding model until the first genuine cache miss.

    Delegates ``extract`` to the real extractor, which is built on first use.
    With a fully-populated embedding cache, the GPU model is never loaded.
    """

    def __init__(self, emb_name: str, device: str, extractor_kwargs: dict | None = None) -> None:
        self._emb_name = emb_name
        self._device = device
        # Config overrides for the underlying extractor (e.g. the spectrogram
        # dynamic-range floor). These change the stored vectors, so whatever is
        # passed here MUST also be reflected in the cache key — see
        # features/cache_keys.py.
        self._kwargs = dict(extractor_kwargs or {})
        self._impl = None

    def extract(self, audio, sr):
        if self._impl is None:
            logger.info(
                "Loading extractor %s on %s (first cache miss)%s",
                self._emb_name,
                self._device,
                f" with {self._kwargs}" if self._kwargs else "",
            )
            self._impl = get_extractor(self._emb_name, device=self._device, **self._kwargs)
        return self._impl.extract(audio, sr)


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON atomically: serialise to a temp file in the same directory,
    fsync, then os.replace. A crash or disk-full mid-write leaves the temp file
    truncated but the real checkpoint stays the last good version (os.replace is
    atomic on a single filesystem)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_embedding(
    tracks: list[dict],
    emb_name: str,
    device: str,
    estimators: list[str],
    max_duration: float,
    with_temporal: bool,
    window_duration: float,
    hop_duration: float,
    output_dir: Path,
    seed: int = 42,
    preprocess_mode: str = "raw",
    embedding_cache_dir: str | None = None,
    max_points: int = 2000,
    lowpass_hz: float | None = None,
    temporal_max_points: int = 300,
    fixed_duration: float | None = None,
    recompute_features: bool = False,
    analysis_region: str = "full",
    cache_dtype: str = "float32",
    extractor_kwargs: dict | None = None,
    cache_extra_tags: tuple[str, ...] = (),
    resample_hz: float | None = None,
    level_match: bool = False,
    level_peak: bool = False,
) -> pd.DataFrame | None:
    """Run full ablation for one embedding model. Supports checkpoint / resume."""
    ckpt_file = output_dir / f"checkpoint_{emb_name}.json"
    results: list[dict] = []
    processed_ids: set[str] = set()

    if ckpt_file.exists() and not recompute_features:
        with open(ckpt_file) as f:
            results = json.load(f)
        processed_ids = {r["track_id"] for r in results}
        logger.info("Resuming %s: %d tracks already done", emb_name, len(results))
    elif recompute_features:
        logger.info("Recompute mode: ignoring checkpoint for %s (features from cache)", emb_name)

    cfg = EMBEDDING_CONFIGS[emb_name]
    rng = np.random.default_rng(seed)

    logger.info("=" * 60)
    logger.info(
        "Embedding: %s | device=%s | estimators=%s | temporal=%s | max_duration=%.0fs | preprocess=%s",
        emb_name,
        device,
        estimators,
        with_temporal,
        max_duration,
        preprocess_mode,
    )
    # Lazy extractor: the embedding model (GPU weights) is only loaded on the
    # first genuine cache miss. With a fully-populated --embedding-cache-dir
    # (e.g. --recompute-features iteration) no forward pass ever runs, so the
    # GPU model is never touched and the loop stays CPU-only.
    extractor = _LazyExtractor(emb_name, device, extractor_kwargs)
    # Namespace the embedding cache per model. mert-95m / encodec / muq all use
    # target_sr=24000, so without per-model namespacing they share a cache key and
    # would read each other's cached matrices (silent corruption in multi-embedding
    # runs). Each model now caches under <cache_dir>/<emb_name>/.
    emb_cache_dir = str(Path(embedding_cache_dir) / emb_name) if embedding_cache_dir else None
    n_total = len(tracks)
    start = time.time()

    for i, track in enumerate(tracks, start=1):
        if track["track_id"] in processed_ids:
            continue

        result = _process_track(
            track,
            extractor,
            target_sr=cfg["target_sr"],
            metric=cfg["metric"],
            estimators=estimators,
            max_duration=max_duration,
            with_temporal=with_temporal,
            window_duration=window_duration,
            hop_duration=hop_duration,
            rng=rng,
            preprocess_mode=preprocess_mode,
            embedding_cache_dir=emb_cache_dir,
            max_points=max_points,
            lowpass_hz=lowpass_hz,
            temporal_max_points=temporal_max_points,
            fixed_duration=fixed_duration,
            analysis_region=analysis_region,
            cache_dtype=cache_dtype,
            cache_extra_tags=cache_extra_tags,
            resample_hz=resample_hz,
            level_match=level_match,
            level_peak=level_peak,
        )
        if result is None:
            continue
        results.append(result)

        if i % 10 == 0 or i == n_total:
            elapsed = time.time() - start
            eta = elapsed / max(i, 1) * max(n_total - i, 0)
            ids_str = "  ".join(f"{est}={result.get(f'id_{est}_full', float('nan')):.2f}" for est in estimators)
            logger.info(
                "[%d/%d] %s %s | %s | elapsed=%.0fm eta=%.0fm",
                i,
                n_total,
                track["label"],
                track["track_id"],
                ids_str,
                elapsed / 60,
                eta / 60,
            )

        if len(results) % 25 == 0:
            _atomic_write_json(ckpt_file, results)
            logger.info("Checkpoint saved: %d tracks", len(results))

    # final checkpoint
    _atomic_write_json(ckpt_file, results)

    del extractor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if not results:
        logger.warning("No results produced for %s", emb_name)
        return None

    df = pd.DataFrame(results)
    # prefix all feature columns with embedding name
    id_cols = [
        c
        for c in df.columns
        if c.startswith("id_") or c.startswith("temporal_") or c.startswith("wd_") or c == "n_embeddings"
    ]
    df = df.rename(columns={c: f"{emb_name}_{c}" for c in id_cols})

    out_path = output_dir / f"ablation_{emb_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s: %d rows × %d cols", out_path.name, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Classification & interpretability helpers
# ---------------------------------------------------------------------------


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ],
        memory=None,
    )


def _usable_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(arr).any():
            out.append(c)
    return out


def _eval_bundle(name: str, df: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    """Stratified 5-fold CV + per-fold coefficient extraction."""
    feature_cols = _usable_cols(df, feature_cols)
    if not feature_cols:
        return None

    valid = df.dropna(subset=feature_cols).copy()
    X = valid[feature_cols].to_numpy(dtype=np.float32)
    y = (valid["label"] == "fake").astype(int).to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 20 or len(np.unique(y)) < 2:
        return None

    pipe = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tp = int(((y == 1) & (y_pred == 1)).sum())

    fold_coefs: list[list[float]] = []
    for tr, _ in cv.split(X, y):
        pf = _build_pipeline()
        pf.fit(X[tr], y[tr])
        fold_coefs.append(pf.named_steps["clf"].coef_.ravel().tolist())

    coef_arr = np.array(fold_coefs)
    return {
        "name": name,
        "n_samples": int(len(y)),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "auc": float(roc_auc_score(y, y_prob)),
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp": tp,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "coef_mean": coef_arr.mean(axis=0).tolist(),
        "coef_std": coef_arr.std(axis=0).tolist(),
        "coef_sign_consistency": (np.sign(coef_arr) == np.sign(coef_arr.mean(axis=0))).mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Bundle definition, classification, stratified eval
# ---------------------------------------------------------------------------


def run_classification_and_eval(
    merged: pd.DataFrame,
    emb_names: list[str],
    estimators: list[str],
    output_dir: Path,
) -> None:
    """Define all feature bundles, evaluate each with 5-fold CV, save results."""
    scales = ["full", "half", "quarter"]
    temporal_stats = ["mean", "std", "slope", "range", "first_last"]
    # Transition-entropy features computed inline during ablation (always available
    # when --with-temporal is used; column names match temporal_features.py registry)
    trans_stats = ["trans_entropy_flat", "trans_entropy_conditional", "trans_self_prob", "trans_n_distinct"]
    # HMM / changepoint — may be absent if hmmlearn / ruptures not installed
    hmm_stats = ["hmm_n_states_bic", "hmm_log_likelihood", "hmm_viterbi_entropy", "hmm_dwell_mean", "hmm_dwell_cv"]
    cp_stats = ["cp_n_changepoints", "cp_mean_segment_len", "cp_first_change_rel", "cp_last_change_rel", "cp_density"]

    bundles: dict[str, list[str]] = {}

    for emb in emb_names:
        # --- static bundles ---
        static_full = _usable_cols(merged, [f"{emb}_id_{est}_full" for est in estimators])
        if static_full:
            bundles[f"static:{emb}:full"] = static_full

        static_multi = _usable_cols(merged, [f"{emb}_id_{est}_{sc}" for est in estimators for sc in scales])
        if len(static_multi) > len(static_full):
            bundles[f"static:{emb}:multiscale"] = static_multi

        # --- temporal summary ---
        temporal = _usable_cols(
            merged, [f"{emb}_temporal_{est}_{stat}" for est in estimators for stat in temporal_stats]
        )
        if temporal:
            bundles[f"temporal:{emb}"] = temporal

        # --- transition-entropy features (always available with --with-temporal) ---
        trans = _usable_cols(merged, [f"{emb}_temporal_{est}_{s}" for est in estimators for s in trans_stats])
        if trans:
            bundles[f"temporal:{emb}:transition"] = trans
            # Combined: summary + transition
            temporal_plus_trans = _usable_cols(merged, temporal + trans)
            if len(temporal_plus_trans) > len(temporal):
                bundles[f"temporal:{emb}:extended"] = temporal_plus_trans

        # --- HMM and changepoint (optional — only if columns present) ---
        hmm = _usable_cols(merged, [f"{emb}_temporal_{est}_{s}" for est in estimators for s in hmm_stats])
        cp = _usable_cols(merged, [f"{emb}_temporal_{est}_{s}" for est in estimators for s in cp_stats])
        all_trajectory = _usable_cols(merged, temporal + trans + hmm + cp)
        if len(all_trajectory) > len(temporal):
            bundles[f"temporal:{emb}:all_trajectory"] = all_trajectory

        # --- model-agnostic geometric window descriptors (non-ID) ---
        # *_n_windows is a DURATION PROXY (track-length leak): excluded from every
        # predictive bundle, isolated below as an explicit confound diagnostic.
        wd_cols = [c for c in merged.columns if c.startswith(f"{emb}_wd_")]
        wd_predictive = [c for c in wd_cols if not c.endswith("_n_windows")]
        wd_all = _usable_cols(merged, wd_predictive)
        wd_geom = _usable_cols(merged, [c for c in wd_predictive if "_traj_" not in c])
        wd_traj = _usable_cols(merged, [c for c in wd_predictive if "_traj_" in c])
        wd_confound = _usable_cols(merged, [c for c in wd_cols if c.endswith("_n_windows")])
        if wd_geom:
            bundles[f"descriptors:{emb}:geometry"] = wd_geom
        if wd_traj:
            bundles[f"descriptors:{emb}:trajectory"] = wd_traj
        if wd_all:
            bundles[f"descriptors:{emb}"] = wd_all
            # descriptors combined with temporal ID — does geometry add over ID?
            id_plus_desc = _usable_cols(merged, temporal + wd_all)
            if len(id_plus_desc) > len(wd_all):
                bundles[f"temporal+descriptors:{emb}"] = id_plus_desc
        if wd_confound:
            # Diagnostic ONLY: how much does raw track length alone separate real
            # vs fake? A high AUC here signals a duration shortcut to remove via
            # --fixed-duration, NOT a generalisable AI-music signature.
            bundles[f"confound:duration:{emb}"] = wd_confound

        # --- combined (static full + best temporal set) ---
        best_temporal = all_trajectory if all_trajectory else (temporal_plus_trans if trans else temporal)
        combined = _usable_cols(merged, static_full + best_temporal)
        if combined and best_temporal:
            bundles[f"combined:{emb}"] = combined

    # --- cross-embedding fusions ---
    fusion_full = _usable_cols(merged, [f"{emb}_id_{est}_full" for emb in emb_names for est in estimators])
    if fusion_full:
        bundles["fusion:static:all_emb"] = fusion_full

    fusion_multi = _usable_cols(
        merged, [f"{emb}_id_{est}_{sc}" for emb in emb_names for est in estimators for sc in scales]
    )
    if len(fusion_multi) > len(fusion_full):
        bundles["fusion:multiscale:all_emb"] = fusion_multi

    all_temporal = _usable_cols(
        merged, [f"{emb}_temporal_{est}_{stat}" for emb in emb_names for est in estimators for stat in temporal_stats]
    )
    if all_temporal:
        bundles["fusion:temporal:all_emb"] = all_temporal

    # Fusion including transition entropy across all embeddings
    all_trans = _usable_cols(
        merged, [f"{emb}_temporal_{est}_{s}" for emb in emb_names for est in estimators for s in trans_stats]
    )
    fusion_extended = _usable_cols(merged, all_temporal + all_trans)
    if len(fusion_extended) > len(all_temporal):
        bundles["fusion:temporal:extended"] = fusion_extended

    # Cross-embedding fusion of geometric window descriptors
    all_descriptors = _usable_cols(merged, [c for c in merged.columns if "_wd_" in c and not c.endswith("_n_windows")])
    if all_descriptors:
        bundles["fusion:descriptors:all_emb"] = all_descriptors

    all_feat = _usable_cols(
        merged, list({c for name, cols in bundles.items() if not name.startswith("confound:") for c in cols})
    )
    if all_feat:
        bundles["fusion:all"] = all_feat

    # --- evaluate ---
    rows: list[dict] = []
    interpretability: dict[str, dict] = {}

    for bundle_name, cols in bundles.items():
        res = _eval_bundle(bundle_name, merged, cols)
        if res is None:
            continue
        interpretability[bundle_name] = {
            k: res[k] for k in ("feature_names", "coef_mean", "coef_std", "coef_sign_consistency")
        }
        rows.append(
            {
                k: v
                for k, v in res.items()
                if k not in ("coef_mean", "coef_std", "coef_sign_consistency", "feature_names")
            }
        )
        logger.info(
            "  [bundle] %-48s  AUC=%.3f  F1=%.3f  FPR=%.3f  FNR=%.3f",
            bundle_name,
            res["auc"],
            res["f1"],
            res["fpr"],
            res["fnr"],
        )

    if rows:
        clf_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        clf_df.to_csv(output_dir / "classification_ablation.csv", index=False)
        # Wide bundles (e.g. fusion:all incl. NaN-heavy HMM/changepoint cols) drop
        # many rows to dropna -> smaller n -> AUC inflated on an easier subset. Pick
        # the headline/stratified bundle only among those evaluated on a COMPARABLE
        # sample (>=95% of the max n) so the reported best is honest.
        max_n = int(clf_df["n_samples"].max())
        eligible = clf_df[(clf_df["n_samples"] >= 0.95 * max_n) & (~clf_df["name"].str.startswith("confound:"))]
        if eligible.empty:
            eligible = clf_df
        best = eligible.sort_values("auc", ascending=False).iloc[0]
        logger.info(
            "Best comparable bundle: %s  AUC=%.3f  (n=%d / max n=%d)",
            best["name"],
            best["auc"],
            int(best["n_samples"]),
            max_n,
        )
        raw_best = clf_df.iloc[0]
        if raw_best["name"] != best["name"]:
            logger.info(
                "  (raw top AUC %s=%.3f excluded: n=%d only)",
                raw_best["name"],
                raw_best["auc"],
                int(raw_best["n_samples"]),
            )

    with open(output_dir / "interpretability_ablation.json", "w") as f:
        json.dump(interpretability, f, indent=2)
    logger.info("Interpretability saved → interpretability_ablation.json")

    # --- stratified eval with best comparable bundle ---
    if not rows:
        return
    best_name = best["name"]
    best_cols = bundles.get(best_name, [])
    if not best_cols:
        return

    strat_rows: list[dict] = []
    for dim, col in [("source", "source"), ("fake_type", "fake_label"), ("algorithm", "algorithm")]:
        if col not in merged.columns:
            continue
        for val in sorted(v for v in merged[col].dropna().unique() if str(v).strip()):
            sub = merged[(merged["label"] == "real") | ((merged["label"] == "fake") & (merged[col] == val))]
            res = _eval_bundle(f"{dim}:{val}", sub, best_cols)
            if res:
                strat_rows.append(
                    {
                        "dimension": dim,
                        "value": val,
                        "n_real": int((sub["label"] == "real").sum()),
                        "n_fake": int((sub["label"] == "fake").sum()),
                        "best_bundle": best_name,
                        "auc": res["auc"],
                        "f1": res["f1"],
                        "fpr": res["fpr"],
                        "fnr": res["fnr"],
                        "accuracy": res["accuracy"],
                    }
                )
                logger.info(
                    "  strat [%s=%s]  n_real=%d n_fake=%d  AUC=%.3f  F1=%.3f",
                    dim,
                    val,
                    strat_rows[-1]["n_real"],
                    strat_rows[-1]["n_fake"],
                    res["auc"],
                    res["f1"],
                )

    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(output_dir / "stratified_eval_ablation.csv", index=False)
        logger.info("Stratified eval saved → stratified_eval_ablation.csv (%d rows)", len(strat_rows))


# ---------------------------------------------------------------------------
# Generator-shift: cross-generator zero-shot evaluation
# ---------------------------------------------------------------------------


def _run_generator_shift_eval(
    merged: pd.DataFrame,
    emb_names: list[str],
    estimators: list[str],
    output_dir: Path,
) -> None:
    """Cross-generator zero-shot evaluation.

    For each held-out generator ``G``:
      - Train on all fakes from all other generators + all reals
      - Test on fakes from ``G`` + all reals
    This measures how well the ID signal generalises across generator versions
    — the core claim motivating the intrinsic-dimension approach.

    Aligns with the "Probing Token Spaces under Generator Shift" framing
    (arXiv:2606.08663).
    """
    if "algorithm" not in merged.columns:
        logger.warning("No 'algorithm' column — skipping generator-shift eval")
        return

    # Collect per-embedding feature bundles
    all_bundles: dict[str, list[str]] = {}
    for emb_name in emb_names:
        prefix = f"{emb_name}_"
        for est in estimators:
            # Static features
            static_cols = [c for c in merged.columns if c.startswith(f"{prefix}id_{est}_")]
            if static_cols:
                all_bundles[f"{emb_name}_{est}_static"] = static_cols
            # Temporal summary stats
            temporal_cols = [
                c
                for c in merged.columns
                if c.startswith(f"{prefix}temporal_{est}_")
                and "window_ids" not in c
                and "trans_" not in c
                and "hmm_" not in c
                and "cp_" not in c
            ]
            if temporal_cols:
                all_bundles[f"{emb_name}_{est}_temporal"] = static_cols + temporal_cols

        # Model-agnostic window descriptors (non-ID geometry). The key robustness
        # question: does the strong in-distribution descriptor signal generalise
        # across held-out generators, or just overfit one generator's artefacts?
        # *_n_windows excluded — it is a duration proxy, not geometry.
        wd_cols = [c for c in merged.columns if c.startswith(f"{prefix}wd_") and not c.endswith("_n_windows")]
        if wd_cols:
            all_bundles[f"{emb_name}_descriptors"] = wd_cols
            temporal_clean = [
                c
                for c in merged.columns
                if c.startswith(f"{prefix}temporal_")
                and "window_ids" not in c
                and "trans_" not in c
                and "hmm_" not in c
                and "cp_" not in c
            ]
            if temporal_clean:
                all_bundles[f"{emb_name}_temporal+descriptors"] = wd_cols + temporal_clean

    algorithms = sorted(str(a) for a in merged["algorithm"].dropna().unique() if str(a).strip())
    shift_rows: list[dict] = []

    for held_out in algorithms:
        test_mask = (merged["label"] == "fake") & (merged["algorithm"] == held_out)
        train_mask = (merged["label"] == "real") | ((merged["label"] == "fake") & (merged["algorithm"] != held_out))

        if test_mask.sum() < 5:
            continue

        test_df = pd.concat([merged[test_mask], merged[merged["label"] == "real"]], ignore_index=True)
        train_df = merged[train_mask]

        for bundle_name, feat_cols in all_bundles.items():
            available = [c for c in feat_cols if c in merged.columns]
            if not available:
                continue

            X_train = train_df[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            y_train = (train_df["label"] == "fake").astype(int).to_numpy()
            X_test = test_df[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            y_test = (test_df["label"] == "fake").astype(int).to_numpy()

            train_mask_finite = np.isfinite(X_train).all(axis=1)
            test_mask_finite = np.isfinite(X_test).all(axis=1)

            if train_mask_finite.sum() < 20 or test_mask_finite.sum() < 5:
                continue
            if len(np.unique(y_train[train_mask_finite])) < 2:
                continue

            try:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, C=0.1)),
                    ]
                )
                pipe.fit(X_train[train_mask_finite], y_train[train_mask_finite])
                probs = pipe.predict_proba(X_test[test_mask_finite])[:, 1]
                y_t = y_test[test_mask_finite]
                if len(np.unique(y_t)) < 2:
                    continue
                auc = roc_auc_score(y_t, probs)
                preds = (probs >= 0.5).astype(int)
                f1 = f1_score(y_t, preds, zero_division=0)
                shift_rows.append(
                    {
                        "held_out_generator": held_out,
                        "bundle": bundle_name,
                        "n_train": int(train_mask_finite.sum()),
                        "n_test": int(test_mask_finite.sum()),
                        "auc": auc,
                        "f1": f1,
                    }
                )
            except Exception as exc:
                logger.debug("Generator-shift eval failed for %s / %s: %s", held_out, bundle_name, exc)

    if shift_rows:
        shift_df = pd.DataFrame(shift_rows)
        shift_df.to_csv(output_dir / "generator_shift_eval.csv", index=False)
        logger.info("Generator-shift eval saved → generator_shift_eval.csv (%d rows)", len(shift_rows))

        # Summary: mean AUC by held-out generator
        logger.info("=== Cross-generator zero-shot AUC summary ===")
        for gen, grp in shift_df.groupby("held_out_generator"):
            best_row = grp.loc[grp["auc"].idxmax()]
            logger.info("  held-out %-20s  best_AUC=%.3f  bundle=%s", gen, best_row["auc"], best_row["bundle"])


def _real_manifold_scores(
    x_all: np.ndarray, train_rows: np.ndarray, score_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a real-only standardised Ledoit-Wolf model on ``train_rows`` and score
    ``score_rows`` by Mahalanobis distance and mean-|z| deviation."""
    from sklearn.covariance import LedoitWolf

    mu = x_all[train_rows].mean(axis=0)
    sd = x_all[train_rows].std(axis=0) + 1e-9
    model = LedoitWolf().fit((x_all[train_rows] - mu) / sd)
    z = (x_all[score_rows] - mu) / sd
    return model.mahalanobis(z), np.abs(z).mean(axis=1)


def _real_manifold_scores_flow(
    x_all: np.ndarray, train_rows: np.ndarray, score_rows: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Fit a real-only RealNVP flow on ``train_rows`` and score ``score_rows`` by
    negative log-likelihood (higher = more anomalous = more AI-like).

    A normalizing flow is a strictly more expressive density estimator than the
    single-Gaussian Mahalanobis ball, able to capture the multimodal (genre /
    production) structure of real music — the same motivation as MusicDET, but
    placed on frozen embeddings so it stays orthogonal to the bandwidth confound.
    """
    cfg = RealNVPConfig(device=device, n_epochs=200, patience=20, seed=42)
    det = RealNVPOneClass(cfg).fit(x_all[train_rows])
    return det.score_samples(x_all[score_rows])


# ---------------------------------------------------------------------------
# Per-window flow log-likelihood trajectory eval (Phase 2 novel core)
# ---------------------------------------------------------------------------


def _pool_windows(
    emb_matrix: np.ndarray,
    window_frames: int,
    hop_frames: int,
    mode: str = "mean",
    frame_stride: int = 1,
    n_blocks: int = 8,
) -> np.ndarray:
    """Window-pool embedding frames — delegates to the canonical library
    implementation (``intrinsic_ai_music_detection.features.pooling``) so every
    consumer shares identical semantics.

    ``mode``: "mean" (headline, per-window centroid), "meanstd" (mean⊕std,
    2×dim), "blocks" (``n_blocks`` sub-span means concatenated, n_blocks×dim —
    recovers the temporal axis the mean discards), or "frame" (no pooling —
    per-frame features, ``window_frames`` / ``hop_frames`` ignored, every
    ``frame_stride``-th finite frame kept).
    """
    from intrinsic_ai_music_detection.features.pooling import frame_features, pool_windows

    if mode == "frame":
        return frame_features(emb_matrix, stride=frame_stride)
    return pool_windows(emb_matrix, window_frames, hop_frames, mode=mode, n_blocks=n_blocks)


def _build_genre_inverse_frequency_weights(
    genre_csv: str,
    merged: pd.DataFrame,
    weight_cap: float = 20.0,
) -> dict[str, float]:
    """Compute per-real-track inverse-genre-frequency weights for balanced sampling.

    weight(track) = (max genre count among real tracks) / (count of track's genre
    among real tracks), capped at ``weight_cap`` so ultra-rare genres (e.g.
    baroque, n=22) don't collapse the effective training set into a handful
    of tracks repeated hundreds of times per epoch. Tracks with an unknown /
    missing genre get weight 1.0 (treated as an "average" genre).

    This directly tests balancing (equal genre representation per epoch) as
    distinct from B2's broadening (simply adding more real tracks from other
    corpora) using the exact same 12,722-track SONICS-real corpus.
    """
    genre_df = pd.read_csv(genre_csv, low_memory=False)
    id_col = "track_id" if "track_id" in genre_df.columns else genre_df.columns[0]
    genre_col = next(
        (c for c in ["genre_top1", "clap_genre_top1", "top1_genre", "genre"] if c in genre_df.columns), None
    )
    if genre_col is None:
        raise ValueError(f"Could not find a genre column in {genre_csv} (columns: {list(genre_df.columns)})")

    real_track_ids = set(merged.loc[merged["label"] == "real", "track_id"].astype(str))
    genre_df = genre_df[genre_df[id_col].astype(str).isin(real_track_ids)].copy()
    genre_df[id_col] = genre_df[id_col].astype(str)
    genre_df = genre_df.dropna(subset=[genre_col])

    genre_counts = genre_df[genre_col].value_counts()
    max_count = float(genre_counts.max())
    logger.info(
        "Genre-balanced sampling: %d real tracks tagged across %d genres (min=%d, max=%d)",
        len(genre_df),
        len(genre_counts),
        int(genre_counts.min()),
        int(genre_counts.max()),
    )

    weight_by_track: dict[str, float] = {}
    for _, row in genre_df.iterrows():
        g = row[genre_col]
        raw_weight = max_count / float(genre_counts[g])
        weight_by_track[str(row[id_col])] = float(min(raw_weight, weight_cap))

    n_capped = sum(1 for w in weight_by_track.values() if w >= weight_cap)
    if n_capped:
        logger.info(
            "Genre-balanced sampling: %d tracks hit the %.0fx weight cap (ultra-rare genres)",
            n_capped,
            weight_cap,
        )
    return weight_by_track


def _make_flow(
    cfg,
    n_bands: int,
    global_prior: bool = False,
    global_layers: int | None = None,
    background: str = "none",
    background_sigma: float = 0.5,
):
    """Build the density model: plain flow, banded flow, and/or likelihood ratio.

    ``background != "none"`` wraps whichever flow is selected in a
    foreground/background PAIR scored by their log-likelihood ratio (Ren et al.
    1906.02845). The background flow is trained on the SAME real windows after a
    corruption, so no labels and no fakes are involved — the detector stays
    fully label-free while subtracting corpus/production identity from the score.
    """
    if background and background != "none":
        from intrinsic_ai_music_detection.models.background import LikelihoodRatioOneClass

        return LikelihoodRatioOneClass(
            cfg,
            corruption=background,
            noise_sigma=background_sigma,
            n_bands=n_bands,
            global_prior=global_prior,
            global_n_coupling_layers=global_layers,
        )
    if n_bands and n_bands > 1:
        from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass

        return BandedRealNVPOneClass(
            cfg,
            n_bands=n_bands,
            global_prior=global_prior,
            global_n_coupling_layers=global_layers,
        )
    return RealNVPOneClass(cfg)


def _run_window_flow_eval(
    merged: pd.DataFrame,
    emb_names: list[str],
    output_dir: Path,
    embedding_cache_dir: str,
    window_duration: float = 4.0,
    hop_duration: float = 2.0,
    max_duration: float = 120.0,
    preprocess_mode: str = "raw",
    lowpass_hz: float | None = None,
    fixed_duration: float | None = None,
    analysis_region: str = "full",
    device: str = "cpu",
    pca_components: int = 64,
    flow_epochs: int = 200,
    seed: int = 42,
    save_flow_path: str | None = None,
    max_windows: int | None = None,
    genre_weight_by_track: dict[str, float] | None = None,
    prior_mean: float = 0.0,
    n_coupling_layers: int = 8,
    hidden_dim: int = 128,
    pooling_mode: str = "mean",
    frame_stride: int = 1,
    pooling_blocks: int = 8,
    symmetric_fold_scoring: bool = False,
    save_trajectories_dir: str | None = None,
    save_fold_flows: bool = False,
    max_train_windows: int | None = None,
    n_bands: int = 1,
    global_prior: bool = False,
    global_layers: int | None = None,
    background: str = "none",
    background_sigma: float = 0.5,
    augment_p: float = 0.0,
    augment_n_masks: int = 1,
    augment_width: tuple[int, int] = (6, 20),
    cache_extra_tags: tuple[str, ...] = (),
) -> pd.DataFrame | None:
    """Fit a per-window normalizing-flow density model and compute likelihood
    trajectory features. This is the Phase-2 novel core:

        per-window mean-pooled embedding  →  RealNVP (trained real-only)
        →  log-likelihood per window  →  trajectory features
        (mean / std / slope / range / transition entropy / changepoints)

    Uses 5-fold cross-validation on real tracks so every real track is scored
    by a flow that *never saw it* during training.  Fake tracks are scored
    against the full model trained on all real windows.  This removes the
    train/test overlap that inflates AUC when real tracks appear in both
    training and evaluation, giving honest per-generator AUC and EER aligned
    with the MusicDET one-class evaluation standard.

    ``pca_components`` reduces the embedding dimension before fitting (e.g.
    MERT 768-d → 64-d) to keep the flow tractable on T4 16 GB.

    Outputs ``window_flow_eval.csv`` in ``output_dir`` with per-track
    trajectory features prefixed ``{emb_name}_wf_``.

    ``genre_weight_by_track``, if given, maps real track_id -> inverse-genre-
    frequency weight. When set, every flow fit (each K-fold's training split
    AND the final full-corpus flow) uses weighted-with-replacement sampling
    instead of uniform sampling, so rare genres are seen roughly as often as
    common ones per epoch. This isolates BALANCING from B2's broadening: the
    same 12,722 real tracks are used, only the training distribution changes.
    """
    from intrinsic_ai_music_detection.features.temporal_features import all_temporal_features

    rows: list[dict] = []

    for emb_name in emb_names:
        cfg_emb = EMBEDDING_CONFIGS.get(emb_name, {})
        target_sr = cfg_emb.get("target_sr", 24_000)
        cache_dir = Path(embedding_cache_dir) / emb_name

        def _cache_path(track_id: str, _sr: int = target_sr) -> Path:
            # _sr is bound as a default so the closure cannot capture a later
            # loop iteration's target_sr when emb_names has more than one entry.
            key = embedding_cache_key(
                track_id,
                _sr,
                max_duration,
                preprocess_mode,
                lowpass_hz=lowpass_hz,
                fixed_duration=fixed_duration,
                analysis_region=analysis_region,
                extra_tags=cache_extra_tags,
            )
            return (
                cache_dir / f"{key}.npy"
            )  # noqa: B023  (superseded flow arm; closure is consumed inside the same iteration)

        # --- collect per-track window arrays for real tracks (K-fold LOGO) ---
        is_real = (merged["label"] == "real").to_numpy()
        real_ids: list[str] = []
        real_wins: list[np.ndarray] = []

        for row in merged[is_real].itertuples(index=False):
            cp = _cache_path(str(row.track_id))
            if not cp.exists():
                continue
            try:
                mat = np.load(str(cp)).astype(np.float32, copy=False)
                audio_dur = _cached_audio_duration(cp, mat, fixed_duration)
                fps = mat.shape[0] / max(audio_dur, 1.0)
                wf = max(int(window_duration * fps), 5)
                hf = max(int(hop_duration * fps), 1)
                pw = _pool_windows(
                    mat,
                    wf,
                    hf,
                    mode=pooling_mode,
                    frame_stride=frame_stride,
                    n_blocks=pooling_blocks,
                )
                if len(pw) > 0:
                    real_ids.append(str(row.track_id))
                    real_wins.append(pw)
            except Exception as exc:
                logger.debug("window-flow real pool failed for %s: %s", row.track_id, exc)

        if not real_ids:
            logger.warning("[%s] no real-track embeddings found in cache — skipping window-flow eval", emb_name)
            continue

        x_real_all = np.concatenate(real_wins, axis=0)
        logger.info(
            "[%s] window-flow: %d real windows from %d tracks, d=%d",
            emb_name,
            len(x_real_all),
            len(real_ids),
            x_real_all.shape[1],
        )

        # --- Likelihood-ratio sanity: the 'shuffle' background is the product of
        # the feature marginals, so the ratio measures CROSS-DIMENSIONAL
        # structure. PCA produces uncorrelated components, i.e. it has already
        # removed the linear part of exactly that structure, leaving the ratio
        # only higher-order dependence to work with. Not wrong, but much weaker
        # -- run the likelihood-ratio arms with --wf-pca-components 0. ---
        if background == "shuffle" and pca_components:
            logger.warning(
                "[%s] --wf-background shuffle with --wf-pca-components %d: PCA decorrelates the "
                "features, so the marginal-product background is close to the joint model and the "
                "ratio loses most of its signal. Strongly prefer --wf-pca-components 0 here.",
                emb_name,
                pca_components,
            )

        # --- Banding sanity: bands are CONTIGUOUS COLUMN BLOCKS, so they are
        # only *frequency* bands if the columns are frequency-ordered, and only
        # if nothing has rotated them. PCA mixes all frequencies into every
        # component, which would silently turn "frequency-guided" into
        # "arbitrary-subspace-guided" while every shape check still passes.
        # Fail loudly rather than report a meaningless banding result. ---
        if n_bands and n_bands > 1:
            if pca_components:
                raise SystemExit(
                    f"--wf-n-bands {n_bands} is incompatible with --wf-pca-components "
                    f"{pca_components}: PCA rotates the axes, so contiguous column blocks are no "
                    f"longer frequency bands. Re-run with --wf-pca-components 0."
                )
            if emb_name not in ("spec", "logmel", "spec-musicdet"):
                logger.warning(
                    "[%s] --wf-n-bands %d on a NON-spectral embedding: its dimensions are not "
                    "frequency-ordered, so these are arbitrary contiguous blocks, NOT frequency "
                    "bands. Valid only as a deliberate control — do not report it as "
                    "frequency-guided banding.",
                    emb_name,
                    n_bands,
                )
            if x_real_all.shape[1] % n_bands:
                logger.warning(
                    "[%s] feature dim %d is not divisible by n_bands=%d; bands will be unequal "
                    "(MusicDET asserts divisibility).",
                    emb_name,
                    x_real_all.shape[1],
                    n_bands,
                )

        # --- PCA on all real windows (fitted once; PCA itself does not label-leak) ---
        pca = None
        x_for_flow = x_real_all
        if pca_components and x_real_all.shape[1] > pca_components and len(x_real_all) > pca_components:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=pca_components, random_state=seed)
            x_for_flow = pca.fit_transform(x_real_all)
            logger.info(
                "[%s] PCA: %d→%d dims (%.1f%% var)",
                emb_name,
                x_real_all.shape[1],
                pca_components,
                100 * pca.explained_variance_ratio_.sum(),
            )

        # Pre-transform per-track arrays through shared PCA
        real_wins_pca: list[np.ndarray] = []
        if pca is not None:
            start = 0
            for tw in real_wins:
                end = start + len(tw)
                real_wins_pca.append(x_for_flow[start:end])
                start = end
        else:
            real_wins_pca = real_wins

        # --- per-window genre weights (broadcast per-track weight to its windows) ---
        window_weights_by_track: np.ndarray | None = None
        if genre_weight_by_track:
            per_track_w = np.array([genre_weight_by_track.get(str(tid), 1.0) for tid in real_ids], dtype=np.float64)
            n_untagged = sum(1 for tid in real_ids if str(tid) not in genre_weight_by_track)
            logger.info(
                "[%s] genre-balanced sampling ENABLED: weight range [%.2f, %.2f] across %d tracks "
                "(%d without a genre tag default to weight 1.0)",
                emb_name,
                per_track_w.min(),
                per_track_w.max(),
                len(real_ids),
                n_untagged,
            )
            window_weights_by_track = per_track_w

        def _window_weights_for(indices: list[int]) -> np.ndarray | None:
            if (
                window_weights_by_track is None
            ):  # noqa: B023  (superseded flow arm; closure is consumed inside the same iteration)
                return None
            return np.concatenate(
                [np.full(len(real_wins_pca[i]), window_weights_by_track[i]) for i in indices]
            )  # noqa: B023  (superseded flow arm; closure is consumed inside the same iteration)

        # --- K-fold cross-validation on real tracks: honest held-out log-likelihoods ---
        # Each real track is scored by a flow that never saw it during training.
        # Removes the train/test overlap that inflated AUC in the original single-model eval.
        from sklearn.model_selection import KFold

        N_FOLDS = 5
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        real_heldout: dict[str, list[float]] = {}  # track_id → per-window lls
        fold_of_track: dict[str, int] = {}  # real track_id → fold index

        # --- optional pre-pooling of FAKE tracks for symmetric per-fold scoring ---
        # (fold flows exist only inside the loop below, so fakes must be pooled first)
        fake_pw_by_track: dict[str, tuple[str, np.ndarray]] | None = None
        if symmetric_fold_scoring:
            fake_pw_by_track = {}
            for row in merged[~is_real].itertuples(index=False):
                cp = _cache_path(str(row.track_id))
                if not cp.exists():
                    continue
                try:
                    mat = np.load(str(cp)).astype(np.float32, copy=False)
                    audio_dur = _cached_audio_duration(cp, mat, fixed_duration)
                    fps = mat.shape[0] / max(audio_dur, 1.0)
                    wf = max(int(window_duration * fps), 5)
                    hf = max(int(hop_duration * fps), 1)
                    pw = _pool_windows(
                        mat,
                        wf,
                        hf,
                        mode=pooling_mode,
                        frame_stride=frame_stride,
                        n_blocks=pooling_blocks,
                    )
                    if len(pw) < 2:
                        continue
                    if pca is not None:
                        pw = pca.transform(pw)
                    if max_windows is not None:
                        pw = pw[:max_windows]
                    fake_pw_by_track[str(row.track_id)] = (str(getattr(row, "algorithm", "")), pw)
                except Exception as exc:
                    logger.debug("[%s] symmetric fake pool failed for %s: %s", emb_name, row.track_id, exc)
            logger.info("[%s] symmetric-fold scoring: pre-pooled %d fake tracks", emb_name, len(fake_pw_by_track))

        sym_rows: list[dict] = []  # per-fold per-generator AUC/EER (symmetric protocol)
        val_rows: list[dict] = []  # per-fold held-out real validation NLL

        def _cap_training_windows(x: np.ndarray, w: np.ndarray | None, tag: str):
            """Uniformly subsample the flow's TRAINING windows (scoring is unaffected)."""
            if max_train_windows is None or len(x) <= max_train_windows:
                return x, w
            rs = np.random.default_rng(seed)
            sel = rs.choice(len(x), size=max_train_windows, replace=False)
            sel.sort()
            logger.info(
                "[%s] %s: capping training windows %d → %d (--wf-max-train-windows)",
                emb_name,  # noqa: B023  (superseded flow arm; closure is consumed inside the same iteration)
                tag,
                len(x),
                max_train_windows,
            )
            return x[sel], (w[sel] if w is not None else None)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(real_ids)))):
            x_train_fold = np.concatenate([real_wins_pca[i] for i in train_idx], axis=0)
            w_train_fold = _window_weights_for(list(train_idx))
            x_train_fold, w_train_fold = _cap_training_windows(x_train_fold, w_train_fold, f"fold{fold_idx}")
            flow_cfg_fold = RealNVPConfig(
                device=device,
                n_epochs=flow_epochs,
                patience=20,
                seed=seed + fold_idx,
                verbose=False,
                prior_mean=prior_mean,
                n_coupling_layers=n_coupling_layers,
                hidden_dim=hidden_dim,
                augment_p=augment_p,
                augment_n_masks=augment_n_masks,
                augment_width=augment_width,
            )
            flow_fold = _make_flow(
                flow_cfg_fold, n_bands, global_prior, global_layers, background, background_sigma
            ).fit(x_train_fold, sample_weights=w_train_fold)

            fold_real_track_nll: dict[str, float] = {}  # this fold's held-out real per-track mean NLL
            for i in test_idx:
                pw = real_wins_pca[i]
                if len(pw) < 2:
                    continue
                lls = -flow_fold.score_samples(pw)
                real_heldout[real_ids[i]] = list(lls)
                fold_of_track[real_ids[i]] = fold_idx
                fold_real_track_nll[real_ids[i]] = float(-lls.mean())  # anomaly = mean NLL

            # --- held-out real validation NLL: the one-class-consistent model-
            # selection criterion (no fakes involved). Lower = better density fit. ---
            if fold_real_track_nll:
                val_rows.append(
                    {
                        "fold": fold_idx,
                        "n_heldout_real_tracks": len(fold_real_track_nll),
                        "heldout_real_nll_mean": float(np.mean(list(fold_real_track_nll.values()))),
                        "heldout_real_nll_std": float(np.std(list(fold_real_track_nll.values()))),
                    }
                )

            # --- optional fold-flow checkpoint (reproducibility of held-out scores) ---
            if save_fold_flows and save_flow_path:
                _fp_fold = (
                    Path(save_flow_path).parent
                    / f"{Path(save_flow_path).stem}_{emb_name}_fold{fold_idx}{Path(save_flow_path).suffix}"
                )
                if pca is not None:
                    flow_fold.attach_pca(pca)
                flow_fold.save(_fp_fold)

            # --- symmetric per-fold scoring: SAME flow scores this fold's held-out
            # reals AND all fakes, so real/fake scores share calibration (removes
            # the fold-flow-vs-full-flow offset inside the AUC). ---
            if symmetric_fold_scoring and fake_pw_by_track and fold_real_track_nll:
                real_anoms = np.array(list(fold_real_track_nll.values()))
                fake_anom_by_alg: dict[str, list[float]] = {}
                for _tid, (alg, pw_f) in fake_pw_by_track.items():
                    anom = float(flow_fold.score_samples(pw_f).mean())
                    fake_anom_by_alg.setdefault(alg, []).append(anom)
                for alg in sorted(fake_anom_by_alg):
                    fake_anoms = np.array(fake_anom_by_alg[alg])
                    if len(fake_anoms) < 5 or len(real_anoms) < 5:
                        continue
                    y = np.r_[np.zeros(len(real_anoms)), np.ones(len(fake_anoms))]
                    s = np.r_[real_anoms, fake_anoms]
                    try:
                        sym_rows.append(
                            {
                                "fold": fold_idx,
                                "algorithm": alg,
                                "auc": float(roc_auc_score(y, s)),
                                "eer": float(equal_error_rate(y, s)),
                                "n_real_heldout": len(real_anoms),
                                "n_fake": len(fake_anoms),
                            }
                        )
                    except Exception:
                        pass

        logger.info(
            "[%s] K-fold (n=%d) window-flow: %d/%d real tracks scored (held-out)",
            emb_name,
            N_FOLDS,
            len(real_heldout),
            len(real_ids),
        )

        if val_rows:
            df_val = pd.DataFrame(val_rows)
            df_val.to_csv(output_dir / f"wf_validation_nll_{emb_name}.csv", index=False)
            logger.info(
                "[%s] held-out real validation NLL: %.4f ± %.4f (per-fold means; K=%d, "
                "the label-free model-selection criterion) → wf_validation_nll_%s.csv",
                emb_name,
                df_val["heldout_real_nll_mean"].mean(),
                df_val["heldout_real_nll_mean"].std(),
                n_coupling_layers,
                emb_name,
            )

        if sym_rows:
            df_sym = pd.DataFrame(sym_rows)
            df_sym.to_csv(output_dir / f"wf_symmetric_fold_auc_{emb_name}.csv", index=False)
            pooled = df_sym.groupby("algorithm")[["auc", "eer"]].agg(["mean", "std"])
            logger.info(
                "[%s] SYMMETRIC per-fold AUC/EER (same flow scores reals+fakes) → " "wf_symmetric_fold_auc_%s.csv\n%s",
                emb_name,
                emb_name,
                pooled.to_string(),
            )

        # --- full flow on ALL real windows — used only for scoring fake tracks ---
        w_full = _window_weights_for(list(range(len(real_ids))))
        flow_cfg = RealNVPConfig(
            device=device,
            n_epochs=flow_epochs,
            patience=20,
            seed=seed,
            verbose=True,
            prior_mean=prior_mean,
            n_coupling_layers=n_coupling_layers,
            hidden_dim=hidden_dim,
            augment_p=augment_p,
            augment_n_masks=augment_n_masks,
            augment_width=augment_width,
        )
        x_full_fit, w_full = _cap_training_windows(x_for_flow, w_full, "full")
        flow = _make_flow(flow_cfg, n_bands, global_prior, global_layers, background, background_sigma).fit(
            x_full_fit, sample_weights=w_full
        )
        logger.info("[%s] full flow fitted on %d real windows (fake scoring)", emb_name, len(x_for_flow))
        if pca is not None:
            # Serialize the reduction with the checkpoint so --flow-path consumers
            # can feed raw pooled windows (see RealNVPOneClass.attach_pca).
            flow.attach_pca(pca)
        # Tag the checkpoint with its front end so consumers cannot silently
        # score a different representation of the same dimensionality.
        flow.embedding_name = emb_name
        if save_flow_path:
            _fp = Path(save_flow_path).parent / f"{Path(save_flow_path).stem}_{emb_name}{Path(save_flow_path).suffix}"
            flow.save(_fp)
            logger.info("[%s] full flow saved → %s", emb_name, _fp)

        # --- score each track ---
        traj_store: dict[str, np.ndarray] = {}  # track_id → ANOMALY trajectory (-ll)
        traj_meta: list[dict] = []
        for row in merged.itertuples(index=False):
            cp = _cache_path(str(row.track_id))
            feat: dict = {"track_id": row.track_id, "label": row.label, "algorithm": getattr(row, "algorithm", None)}

            if row.label == "real":
                # Use held-out K-fold likelihoods (flow never saw this track)
                traj = real_heldout.get(str(row.track_id))
                if traj is None:
                    rows.append(feat)
                    continue
            else:
                # Fake track: score against full real-trained flow
                if not cp.exists():
                    rows.append(feat)
                    continue
                try:
                    mat = np.load(str(cp)).astype(np.float32, copy=False)
                    audio_dur = _cached_audio_duration(cp, mat, fixed_duration)
                    fps = mat.shape[0] / max(audio_dur, 1.0)
                    wf = max(int(window_duration * fps), 5)
                    hf = max(int(hop_duration * fps), 1)
                    pw = _pool_windows(
                        mat,
                        wf,
                        hf,
                        mode=pooling_mode,
                        frame_stride=frame_stride,
                        n_blocks=pooling_blocks,
                    )
                    if len(pw) < 2:
                        rows.append(feat)
                        continue
                    if pca is not None:
                        pw = pca.transform(pw)
                    lls = -flow.score_samples(pw)  # negate: score_samples = -loglik
                    traj = list(lls)
                except Exception as exc:
                    logger.debug("[%s] window-flow scoring failed for %s: %s", emb_name, row.track_id, exc)
                    rows.append(feat)
                    continue

            # --- optional evidence-budget cap (confound check: are shorter-native
            # generators like udio-30s harder because of genuine acoustic realism,
            # or simply because they yield fewer windows / less evidence per track
            # under the same window/hop scheme?). Applied symmetrically to real and
            # fake so the comparison stays fair. ---
            if max_windows is not None:
                traj = traj[:max_windows]

            # --- optional per-window trajectory persistence (aggregation ablations
            # become CPU-only post-processing; stored ANOMALY-oriented = -loglik). ---
            if save_trajectories_dir is not None:
                tid = str(row.track_id)
                # UNIQUE key: track_id alone COLLIDES on corpora where the same
                # source id yields one real and N generator variants (e.g.
                # FakeMusicCaps reuses the MusicCaps YouTube id for all 5 TTM
                # systems). A track_id-keyed store silently kept only the last
                # writer, corrupting any downstream trajectory analysis.
                key = f"{row.label}|{getattr(row, 'algorithm', None) or ''}|{tid}"
                traj_store[key] = -np.asarray(traj, dtype=np.float64)
                traj_meta.append(
                    {
                        "key": key,
                        "track_id": tid,
                        "label": row.label,
                        "algorithm": getattr(row, "algorithm", None),
                        "fold": fold_of_track.get(tid, -1),
                        "n_windows": len(traj),
                    }
                )

            # --- summary statistics (shared for held-out real and fake) ---
            finite = np.array([v for v in traj if np.isfinite(v)])
            if len(finite) < 2:
                rows.append(feat)
                continue
            feat[f"{emb_name}_wf_mean"] = float(finite.mean())
            feat[f"{emb_name}_wf_std"] = float(finite.std())
            feat[f"{emb_name}_wf_range"] = float(finite.max() - finite.min())
            feat[f"{emb_name}_wf_slope"] = float(np.polyfit(np.arange(len(finite)), finite, 1)[0])
            feat[f"{emb_name}_wf_first_last"] = float(finite[-1] - finite[0])
            feat[f"{emb_name}_wf_n_windows"] = int(len(finite))

            # --- transition entropy / HMM / changepoints on likelihood series ---
            ext = all_temporal_features(traj, run_hmm=True, run_changepoint=True)
            for k, v in ext.items():
                feat[f"{emb_name}_wf_{k}"] = v
            rows.append(feat)

        if save_trajectories_dir is not None and traj_store:
            traj_dir = Path(save_trajectories_dir)
            traj_dir.mkdir(parents=True, exist_ok=True)
            if len(traj_store) != len(traj_meta):
                logger.error(
                    "[%s] trajectory key collision: %d unique keys for %d scored tracks — "
                    "downstream aggregation would be WRONG. Investigate before using.",
                    emb_name,
                    len(traj_store),
                    len(traj_meta),
                )
            np.savez_compressed(traj_dir / f"wf_trajectories_{emb_name}.npz", **traj_store)
            pd.DataFrame(traj_meta).to_csv(traj_dir / f"wf_trajectories_{emb_name}_meta.csv", index=False)
            logger.info(
                "[%s] saved %d per-window ANOMALY trajectories (%d meta rows) → %s "
                "(use scripts/ablate_aggregation.py)",
                emb_name,
                len(traj_store),
                len(traj_meta),
                traj_dir / f"wf_trajectories_{emb_name}.npz",
            )

    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_dir / "window_flow_eval.csv", index=False)
        wf_cols = [c for c in df_out.columns if "_wf_" in c]
        logger.info(
            "Window-flow trajectory eval → window_flow_eval.csv (%d rows, %d features)", len(df_out), len(wf_cols)
        )
        # Quick per-generator AUC summary on wf_mean (should be lowest / most real-like for real)
        for emb_name in emb_names:
            col = f"{emb_name}_wf_mean"
            if col not in df_out.columns:
                continue
            is_real_mask = (df_out["label"] == "real").to_numpy()
            is_fake_mask = (df_out["label"] == "fake").to_numpy()
            real_s = df_out.loc[is_real_mask, col].dropna().to_numpy(float)
            if len(real_s) < 5:
                continue
            algos = sorted(str(g) for g in df_out.loc[is_fake_mask, "algorithm"].dropna().unique() if str(g).strip())
            parts = []
            for g in algos:
                g_mask = is_fake_mask & (df_out["algorithm"] == g).to_numpy()
                fake_s = df_out.loc[g_mask, col].dropna().to_numpy(float)
                if len(fake_s) < 5:
                    continue
                y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
                # wf_mean is log-likelihood (higher = more real) → negate for AUC
                s = np.r_[-real_s, -fake_s]
                try:
                    auc = roc_auc_score(y, s)
                    eer = equal_error_rate(y, s)
                    parts.append(f"{g}={auc:.3f}/{eer:.3f}")
                except Exception:
                    pass
            if parts:
                logger.info("  [%s/wf_mean LOGO-%d-fold] (auc/eer) %s", emb_name, N_FOLDS, "  ".join(parts))

        return df_out
    return None


def _run_phase5_fusion_eval(
    wf_df: pd.DataFrame,
    merged: pd.DataFrame,
    emb_names: list[str],
    output_dir: Path,
) -> None:
    """Phase 5: supervised classification including window-flow trajectory features.

    Merges ``wf_df`` (from ``window_flow_eval.csv``) into ``merged`` and evaluates
    bundles combining geometric window descriptors with normalizing-flow trajectory
    features (mean / std / slope + HMM / changepoint on the per-window
    log-likelihood series).  Key hypothesis: ``descriptors+flow`` > descriptors
    alone because they probe orthogonal manifold axes (geometry vs density shape).

    Explainability note: ``wf_cp_first_change_rel`` shows WHERE in the track
    AI-like density shifts first appear; ``wf_slope`` shows whether AI-likeness
    increases or decreases over time — interpretable localisation of AI artefacts.

    Outputs
    -------
    ablation_features_with_flow.csv : full merged feature table incl. wf_ cols
    classification_phase5.csv       : Phase-5 bundle AUCs sorted descending
    """
    merge_keys = [c for c in ["track_id", "label", "algorithm"] if c in merged.columns and c in wf_df.columns]
    wf_feat_cols = [c for c in wf_df.columns if "_wf_" in c and not c.endswith("_n_windows")]
    if not wf_feat_cols:
        logger.warning("Phase 5: no usable wf_ feature columns in wf_df — skipping.")
        return

    merged_aug = merged.merge(wf_df[merge_keys + wf_feat_cols], on=merge_keys, how="left")
    merged_aug.to_csv(output_dir / "ablation_features_with_flow.csv", index=False)
    logger.info(
        "Phase-5 feature table: %d rows × %d cols (+%d wf_ features) → ablation_features_with_flow.csv",
        len(merged_aug),
        len(merged_aug.columns),
        len(wf_feat_cols),
    )

    bundles: dict[str, list[str]] = {}

    for emb in emb_names:
        wf_cols = _usable_cols(merged_aug, [c for c in merged_aug.columns if c.startswith(f"{emb}_wf_")])
        wd_cols = _usable_cols(
            merged_aug,
            [c for c in merged_aug.columns if c.startswith(f"{emb}_wd_") and not c.endswith("_n_windows")],
        )
        temp_cols = _usable_cols(
            merged_aug,
            [
                c
                for c in merged_aug.columns
                if c.startswith(f"{emb}_temporal_") or (c.startswith(f"{emb}_id_") and "_full" in c)
            ],
        )
        if wf_cols:
            bundles[f"flow_only:{emb}"] = wf_cols
        if wf_cols and wd_cols:
            bundles[f"descriptors+flow:{emb}"] = wd_cols + wf_cols
        if wf_cols and wd_cols and temp_cols:
            bundles[f"temporal+descriptors+flow:{emb}"] = temp_cols + wd_cols + wf_cols

    # Cross-embedding fusions
    all_wf = _usable_cols(merged_aug, [c for c in merged_aug.columns if "_wf_" in c and not c.endswith("_n_windows")])
    all_wd = _usable_cols(merged_aug, [c for c in merged_aug.columns if "_wd_" in c and not c.endswith("_n_windows")])
    all_temp = _usable_cols(
        merged_aug,
        [c for c in merged_aug.columns if "_temporal_" in c or ("_id_" in c and "_full" in c)],
    )
    if all_wf and all_wd:
        bundles["fusion:descriptors+flow:all_emb"] = all_wd + all_wf
    if all_wf and all_wd and all_temp:
        bundles["fusion:all_features:all_emb"] = all_temp + all_wd + all_wf

    rows_out: list[dict] = []
    for bundle_name, cols in bundles.items():
        res = _eval_bundle(bundle_name, merged_aug, cols)
        if res is None:
            continue
        rows_out.append(
            {
                k: v
                for k, v in res.items()
                if k not in ("coef_mean", "coef_std", "coef_sign_consistency", "feature_names")
            }
        )
        logger.info(
            "  [phase5 bundle] %-52s  AUC=%.3f  F1=%.3f  FPR=%.3f  FNR=%.3f",
            bundle_name,
            res["auc"],
            res["f1"],
            res["fpr"],
            res["fnr"],
        )

    if rows_out:
        phase5_df = pd.DataFrame(rows_out).sort_values("auc", ascending=False)
        phase5_df.to_csv(output_dir / "classification_phase5.csv", index=False)
        best = phase5_df.iloc[0]
        logger.info(
            "Phase-5 best bundle: %s  AUC=%.3f  (n=%d)",
            best["name"],
            best["auc"],
            int(best["n_samples"]),
        )
        logger.info("Phase-5 classification → classification_phase5.csv (%d bundles)", len(rows_out))


def _run_real_anomaly_eval(
    merged: pd.DataFrame,
    emb_names: list[str],
    output_dir: Path,
    one_class_method: str = "mahalanobis",
    device: str = "cpu",
) -> None:
    """One-class (real-only) anomaly detection, evaluated per generator.

    The 2-class linear classifier learns a *signed* decision direction from the
    training generators, so it fails when a held-out generator deviates from real
    in the OPPOSITE direction (in this corpus chirp has LOW effrank_range, udio
    HIGH). A one-class detector instead models ONLY the real-music manifold and
    flags deviation in ANY direction — the generator-agnostic formulation.

    Each track is scored by its distance from the real descriptor distribution:
      - Mahalanobis distance (Ledoit-Wolf shrinkage covariance), and
      - a robust mean-|z| fallback.
    Real scores are out-of-fold (5-fold over reals) so reals are never
    self-scored; fakes are scored by a model fit on all reals. We then report
    ROC-AUC of held-out-real vs each generator — a true zero-shot, LABEL-FREE
    generalisation test (no fake ever seen in training).
    """
    if "algorithm" not in merged.columns:
        logger.warning("No 'algorithm' column — skipping real-anomaly eval")
        return

    rows: list[dict] = []
    for emb in emb_names:
        cols = [c for c in merged.columns if c.startswith(f"{emb}_wd_") and not c.endswith("_n_windows")]
        if not cols:
            continue
        x_all = merged[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x_all).all(axis=1)
        is_real = (merged["label"] == "real").to_numpy() & finite
        is_fake = (merged["label"] == "fake").to_numpy() & finite
        if is_real.sum() < 50 or is_fake.sum() < 5:
            continue

        maha = np.full(len(merged), np.nan)
        absz = np.full(len(merged), np.nan)
        flow = np.full(len(merged), np.nan)
        real_idx = np.nonzero(is_real)[0]
        use_flow = one_class_method in ("flow", "both")

        # out-of-fold scores for reals (never self-scored)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in kf.split(real_idx, np.zeros(len(real_idx))):
            m, a = _real_manifold_scores(x_all, real_idx[tr], real_idx[te])
            maha[real_idx[te]] = m
            absz[real_idx[te]] = a
            if use_flow:
                flow[real_idx[te]] = _real_manifold_scores_flow(x_all, real_idx[tr], real_idx[te], device)

        # fakes scored by a model fit on ALL reals
        fake_idx = np.nonzero(is_fake)[0]
        m, a = _real_manifold_scores(x_all, real_idx, fake_idx)
        maha[fake_idx] = m
        absz[fake_idx] = a
        if use_flow:
            flow[fake_idx] = _real_manifold_scores_flow(x_all, real_idx, fake_idx, device)

        scorers: list[tuple[str, np.ndarray]] = []
        if one_class_method in ("mahalanobis", "both"):
            scorers += [("mahalanobis", maha), ("abs_z", absz)]
        if use_flow:
            scorers.append(("flow", flow))

        algos = sorted(str(g) for g in merged.loc[is_fake, "algorithm"].dropna().unique() if str(g).strip())
        for scorer, score in scorers:
            rs = score[real_idx]
            rs = rs[np.isfinite(rs)]
            for g in algos:
                g_idx = np.nonzero(is_fake & (merged["algorithm"] == g).to_numpy())[0]
                gs = score[g_idx]
                gs = gs[np.isfinite(gs)]
                if len(gs) < 5:
                    continue
                y = np.r_[np.zeros(len(rs)), np.ones(len(gs))]
                s = np.r_[rs, gs]
                auc = roc_auc_score(y, s)
                rows.append(
                    {
                        "embedding": emb,
                        "scorer": scorer,
                        "held_out_generator": g,
                        "n_real": int(len(rs)),
                        "n_fake": int(len(gs)),
                        "auc": float(auc),
                        "eer": float(equal_error_rate(y, s)),
                    }
                )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "real_anomaly_eval.csv", index=False)
        logger.info("Real-anomaly (one-class) eval saved → real_anomaly_eval.csv (%d rows)", len(rows))
        logger.info("=== One-class real-manifold AUC / EER (LABEL-FREE, per held-out generator) ===")
        for (emb, scorer), grp in df.groupby(["embedding", "scorer"]):
            summary = "  ".join(f"{r.held_out_generator}={r.auc:.3f}/{r.eer:.3f}" for r in grp.itertuples())
            logger.info("  [%s/%s] (auc/eer) %s", emb, scorer, summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanced multi-model ablation: static multiscale + temporal ID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        # NOTE: this list and EMBEDDING_CONFIGS must stay in sync — adding a
        # front end to one and not the other fails at argument-parse time with a
        # message that does not say which list is missing it.
        choices=sorted(set(EMBEDDING_CONFIGS) | {"all"}),
        default=["encodec"],
        help=(
            "Embedding models to run. " "mert-95m is ~3.5× faster than mert (330M) — recommended for subsample pilots."
        ),
    )
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=200,
        help="Max tracks per (algorithm × fake_label) stratum",
    )
    parser.add_argument("--max-real", type=int, default=None, help="Max real tracks (default: match total fake count)")
    parser.add_argument(
        "--fake-types", nargs="+", default=None, help="Restrict fake_label values, e.g. 'full fake' 'half fake'"
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None, help="Restrict algorithm versions, e.g. chirp-v3.5 udio-120s"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-duration", type=float, default=120.0, help="Max audio duration per track (seconds)")
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=DEFAULT_ESTIMATORS,
        choices=["phd", "twonn", "mle"],
        help="ID estimators to run",
    )
    parser.add_argument("--include-mle", action="store_true", help="Add MLE estimator (noisy; off by default)")
    parser.add_argument("--with-temporal", action="store_true", help="Also compute temporal windowed ID features")
    parser.add_argument("--window-duration", type=float, default=30.0, help="Temporal window size (seconds)")
    parser.add_argument("--hop-duration", type=float, default=15.0, help="Temporal window hop (seconds)")
    parser.add_argument("--output-dir", type=str, default="data/processed/sonics_balanced_ablation")
    parser.add_argument("--seed", type=int, default=42)
    # --- new: canonical preprocessing + fair sampling ---
    parser.add_argument(
        "--preprocess-mode",
        choices=["raw", "canonical", "preprocessed"],
        default="raw",
        help=(
            "raw: legacy peak-normalise. "
            "canonical: MP3 round-trip + LUFS on the fly. "
            "preprocessed: load pre-built canonical WAV as-is (use WITH --canonical-manifest)."
        ),
    )
    parser.add_argument(
        "--canonical-manifest",
        default=None,
        help="Path to canonical_manifest.csv (uses pre-built canonical audio).",
    )
    parser.add_argument(
        "--real-genres-csv",
        default=None,
        help="Path to real_genres.csv from tag_real_genres.py (for genre-matching).",
    )
    parser.add_argument(
        "--covariate-profiles-csv",
        default=None,
        help="Path to covariate_profiles.csv from profile_covariates.py (for covariate matching).",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="Directory to cache embedding arrays to disk (avoids recomputation across runs).",
    )
    # --- full-scale optimizations ---
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help=(
            "Max points for ID estimation (subsamples large embeddings). "
            "2000 gives nearly identical results to full cloud at ~1/20th the time. "
            "Use 500 for fastest ablations, 5000 for maximum accuracy."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 for embedding extraction (faster on CUDA, slight quality reduction).",
    )
    parser.add_argument(
        "--generator-shift-eval",
        action="store_true",
        default=False,
        help=(
            "Run cross-generator zero-shot evaluation: train on all algorithms except "
            "one held-out generator, test on it. Repeats for each algorithm. "
            "Requires --with-temporal."
        ),
    )
    parser.add_argument(
        "--real-anomaly-eval",
        action="store_true",
        default=False,
        help=(
            "Run one-class real-manifold anomaly evaluation: model the REAL descriptor "
            "distribution only (Ledoit-Wolf Mahalanobis + mean-|z|) and score deviation in "
            "ANY direction. Reports label-free per-generator AUC. This is the "
            "generator-agnostic formulation — it does not learn a signed boundary from "
            "specific generators, so it can catch families that bracket real on opposite "
            "sides (e.g. chirp low vs udio high effrank_range)."
        ),
    )
    parser.add_argument(
        "--one-class-method",
        choices=["mahalanobis", "flow", "both"],
        default="mahalanobis",
        help=(
            "Density model for the one-class real-manifold eval. 'mahalanobis' = "
            "Ledoit-Wolf Gaussian (fast, default). 'flow' = RealNVP normalizing flow "
            "(more expressive, captures multimodal real-music structure; the MusicDET-"
            "style density upgrade, but on frozen embeddings so it stays orthogonal to "
            "the bandwidth confound). 'both' runs and reports each for comparison."
        ),
    )
    parser.add_argument(
        "--anomaly-from-features",
        default=None,
        help=(
            "Path to an existing merged ablation_features.csv. Runs ONLY the one-class "
            "real-manifold eval (flow/mahalanobis + EER) on it, inferring embeddings from "
            "the *_wd_* columns, then exits. Use to re-score a cached multi-embedding "
            "descriptor table without any re-extraction."
        ),
    )
    parser.add_argument(
        "--window-flow-eval",
        action="store_true",
        default=False,
        help=(
            "After extraction, run the per-window normalizing-flow trajectory eval "
            "(Phase-2 core). Requires --embedding-cache-dir to be set. Fits a RealNVP "
            "flow on mean-pooled real-track window vectors, then scores every track's "
            "per-window likelihood trajectory and extracts transition entropy / HMM / "
            "changepoint features on it. Window size controlled by --wf-window-duration "
            "and --wf-hop-duration (default 4s/2s, matching MusicDET's 4-s clips). "
            "Output: window_flow_eval.csv."
        ),
    )
    parser.add_argument(
        "--wf-window-duration",
        type=float,
        default=4.0,
        help="Window size for per-window flow eval (seconds; default 4 = MusicDET).",
    )
    parser.add_argument(
        "--wf-hop-duration", type=float, default=2.0, help="Hop size for per-window flow eval (seconds; default 2)."
    )
    parser.add_argument(
        "--wf-pca-components", type=int, default=64, help="PCA reduction before flow fitting (0 = no PCA; default 64)."
    )
    parser.add_argument(
        "--wf-flow-epochs", type=int, default=200, help="Training epochs for per-window RealNVP flow (default 200)."
    )
    parser.add_argument(
        "--wf-max-windows",
        type=int,
        default=None,
        help=(
            "Confound check: cap each track's window trajectory to the first N windows "
            "(applied symmetrically to real held-out and fake scoring) before computing "
            "summary stats. Use this to test whether a generator's weak AUC (e.g. udio-30s, "
            "which is natively ~30s and yields only ~15 windows vs ~26 for 55s-capped "
            "generators) is a genuine acoustic-difficulty effect or an artifact of having "
            "less evidence per track. Default: no cap (use all available windows)."
        ),
    )
    parser.add_argument(
        "--wf-n-coupling-layers",
        type=int,
        default=8,
        help=(
            "Number of RealNVP coupling layers (our analog of MusicDET's flow depth K per "
            "band; arXiv 2605.18072 Sec 4.3 'Effect of Flow Steps K'). Default 8."
        ),
    )
    parser.add_argument(
        "--wf-hidden-dim",
        type=int,
        default=128,
        help="Hidden width of each coupling-layer MLP. Default 128.",
    )
    parser.add_argument(
        "--wf-prior-mean",
        type=float,
        default=0.0,
        help=(
            "Shift the RealNVP base-distribution mean N(prior_mean, I) away from the "
            "standard-normal default (0.0). Reproduces MusicDET's (arXiv 2605.18072, "
            "Sec 4.3) 'Effect of the Prior Mean mu' ablation on our flow: they report "
            "EER monotonically decreasing as mu_real increases from 0. Try e.g. 1, 2, 3, 5, 8."
        ),
    )
    parser.add_argument(
        "--wf-save-flow-path",
        default=None,
        help=(
            "If set, save the full RealNVP flow (trained on all real windows) to this path "
            "after training. Embedding name is appended before the extension "
            "(e.g. 'flow.pt' → 'flow_encodec.pt'). Use with score_external_corpus.py "
            "to score FMA/FakeMusicCaps without re-training."
        ),
    )
    parser.add_argument(
        "--wf-pooling",
        choices=["mean", "meanstd", "frame", "delta", "blocks"],
        default="mean",
        help=(
            "Window representation for the flow. 'mean' = per-window frame centroid "
            "(headline). 'blocks' = split the window into --wf-pooling-blocks sub-spans and "
            "concatenate their means (n_blocks x dim) — recovers the temporal axis that the "
            "mean averages away, which is the largest remaining architectural gap against "
            "MusicDET's 2-D convolutional couplings (§9.7.19 item 3). Costs NO re-extraction: "
            "pooling runs on cached frame matrices, so a sweep is flow-training only. "
            "'meanstd' = mean concatenated with per-window std (2x dim; keeps "
            "within-window texture). 'frame' = NO pooling — the flow is trained/scored on "
            "raw 75 Hz frames (window/hop ignored; ~200x more samples per track, the "
            "short-clip/data-starvation ablation arm; combine with --wf-frame-stride to "
            "bound memory on large corpora)."
        ),
    )
    parser.add_argument(
        "--wf-pooling-blocks",
        type=int,
        default=8,
        help=(
            "Sub-spans per window for --wf-pooling blocks. At 4 s / 75 fps, 8 blocks = 0.5 s "
            "each and a 1024-d vector for a 128-d front end. Higher keeps more temporal detail "
            "at the cost of flow input dimension."
        ),
    )
    parser.add_argument(
        "--wf-frame-stride",
        type=int,
        default=1,
        help=(
            "With --wf-pooling frame: keep every Nth finite frame (default 1 = all). "
            "Use ~5-10 for the full SONICS corpus to keep the training tensor within "
            "T4 GPU memory; 1 is fine for 10s-clip corpora like MusicCaps."
        ),
    )
    parser.add_argument(
        "--wf-background",
        choices=["none", "shuffle", "noise", "mask"],
        default="none",
        help=(
            "LIKELIHOOD-RATIO scoring (Ren et al. 1906.02845): also fit a BACKGROUND flow on the "
            "same real windows after a corruption, and score log p_fg - log p_bg. This subtracts "
            "corpus/production identity -- the (B) component that makes the raw likelihood "
            "corpus-specific and sometimes inverted -- while staying fully label-free. "
            "'shuffle' (recommended) permutes each feature dimension independently across "
            "samples, so the background is the product of marginals and the ratio measures "
            "cross-dimensional structure: production identity is largely marginal, synthesis "
            "artifacts are relational. 'noise'/'mask' are the perturbation variants. "
            "Doubles flow training time; scoring cost is ~2x a single flow."
        ),
    )
    parser.add_argument(
        "--wf-background-sigma",
        type=float,
        default=0.5,
        help="noise level for --wf-background noise, in units of per-dimension std",
    )
    parser.add_argument(
        "--emb-cache-dtype",
        choices=["float32", "float16"],
        default="float32",
        help=(
            "dtype for NEW embedding-cache entries. float16 halves disk usage and is promoted "
            "back to float32 on read, so it is transparent downstream. Use it for the 256-d @ "
            "100 fps spec-musicdet caches, which are ~2.7x the size of an EnCodec cache."
        ),
    )
    parser.add_argument(
        "--level-match",
        action="store_true",
        default=False,
        help=(
            "LEVEL CONFOUND CONTROL. Remove DC (20 Hz high-pass) and trim leading/trailing "
            "digital silence before embedding. Measured 2026-08-13 WITH the bandwidth confound "
            "already closed (--resample-hz 15000), single descriptors still reach macro |AUC| "
            "0.7773 on SONICS (silence_frac; dc_offset hits 0.9976 on chirp-v2) and 0.7470 on "
            "FakeMusicCaps (dc_offset). Neither is a synthesis artifact: DC is a preprocessing "
            "defect and leading silence is a decoder-buffer tell. No bandwidth control removes "
            "them. Part of the cache key."
        ),
    )
    parser.add_argument(
        "--level-peak-normalise",
        action="store_true",
        default=False,
        help=(
            "With --level-match, also peak-normalise. Removes the residual peak_dbfs cue "
            "(macro 0.693 on SONICS, 0.866 on udio-30s, after DC+silence control). NOT on by "
            "default because it partly undoes the -23 LUFS canonicalisation: you can hold peak "
            "OR loudness constant, not both. crest_factor_db (0.616) survives either way and is "
            "a genuine production property, not a channel one. Part of the cache key."
        ),
    )
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=None,
        help=(
            "BANDWIDTH CONTROL. Decimate to this rate and return to the target rate, so the band "
            "above resample_hz/2 does not exist for EITHER class. Use this, not --lowpass-hz: "
            "measured on canonical SONICS 2026-08-13, the best channel-alone AUC is 1.0000 with "
            "no control, still 0.9287 under --lowpass-hz 7000 (the leak just moves from the "
            "energy ratio to spectral flatness), and 0.5766 under --resample-hz 14000. A "
            "Butterworth attenuates the band; it does not equalise it. Part of the cache key."
        ),
    )
    parser.add_argument(
        "--spec-log-floor-db",
        type=float,
        default=None,
        help=(
            "SPECTROGRAM ARMS ONLY (--embeddings spec/logmel/spec-musicdet). Clamp the power "
            "spectrum to this many dB below each track's peak before the log. Default (unset) "
            "keeps the historical log(power + 1e-10), under which a band with NO energy maps to "
            "one exact constant in every frame — so 'this file is band-limited' becomes a "
            "near-deterministic feature. Both benchmarks are band-asymmetric (FakeMusicCaps "
            "resamples every fake to 16 kHz; SONICS fakes arrive band-limited), and MP3-64k @ "
            "24 kHz cuts ABOVE 8 kHz, so canonicalisation does not close the gap. Try 80. "
            "The value is part of the embedding-cache key, so runs with different floors cannot "
            "collide — but they DO need separate extraction time."
        ),
    )
    parser.add_argument(
        "--wf-augment-p",
        type=float,
        default=0.0,
        help=(
            "Probability of applying feature-block masking to each TRAINING minibatch, ported from "
            "MusicDET's SpecAugmentFT (they train their flow with p=0.5; we have always trained on "
            "clean features). Masked dims are set to the training mean, so the density must be "
            "supported by evidence spread across the feature axis instead of a narrow "
            "corpus-specific band. 0.0 (default) reproduces every pre-2026-08-12 result exactly."
        ),
    )
    parser.add_argument("--wf-augment-n-masks", type=int, default=1, help="masks per batch (see --wf-augment-p)")
    parser.add_argument(
        "--wf-augment-width", default="6,20", help="min,max contiguous dims per mask (MusicDET uses 6,20)"
    )
    parser.add_argument(
        "--wf-global-prior-layers",
        type=int,
        default=None,
        help=(
            "Coupling-layer depth of the global-prior stage. MusicDET uses global_K=1 (a single "
            "layer) while each band gets K, so pass 1 for an exact match; default reuses the band "
            "depth, which gives the global stage MORE capacity than theirs."
        ),
    )
    parser.add_argument(
        "--wf-global-prior",
        action="store_true",
        help=(
            "With --wf-n-bands > 1, add MusicDET's SECOND stage: instead of summing "
            "independent per-band log-likelihoods, concatenate the band latents and fit a "
            "joint flow over them (their glow_model.py: 'band-wise flows followed by a global "
            "prior'). Plain summation assumes bands are independent, which discards exactly the "
            "cross-band structure — harmonic/formant relations, and the band-edge correlations a "
            "codec or vocoder imprints. Costs one extra small flow; no effect without banding."
        ),
    )
    parser.add_argument(
        "--wf-n-bands",
        type=int,
        default=1,
        help=(
            "FREQUENCY-GUIDED flow (MusicDET's core architecture): split the feature vector into "
            "N contiguous bands and fit an INDEPENDENT flow per band, summing log-likelihoods. "
            "Replaces one D-dim density estimate with N D/N-dim ones — far better determined from "
            "limited data, which is exactly FakeMusicCaps' regime (~21k windows for 128 dims). "
            "Only meaningful for frequency-ordered features (--embeddings spec/logmel/spec-musicdet); do NOT "
            "combine with PCA (--wf-pca-components 0), which mixes frequencies and destroys the "
            "band semantics. 1 = the current single joint flow."
        ),
    )
    parser.add_argument(
        "--wf-max-train-windows",
        type=int,
        default=None,
        help=(
            "Cap the number of REAL windows used to FIT each flow by uniform random "
            "subsampling (seeded). Scoring still uses every window of every track, so only "
            "the density estimate's training set is capped. Required for --wf-pooling frame "
            "on large corpora: full-frame SONICS is ~52M frames (5.25M even at stride 10), "
            "which exhausts host RAM during the concatenate step. A few million windows is "
            "already far past the saturation point measured in the coverage curve."
        ),
    )
    parser.add_argument(
        "--wf-symmetric-fold-scoring",
        action="store_true",
        default=False,
        help=(
            "Protocol validation: in addition to the standard eval, score ALL fake tracks "
            "with EACH K-fold flow and report per-fold AUC/EER where the SAME flow scores "
            "that fold's held-out reals and the fakes (removes the fold-flow-vs-full-flow "
            "calibration asymmetry from the AUC). Writes wf_symmetric_fold_auc_<emb>.csv. "
            "Extra cost: 5x fake scoring passes (no extra training)."
        ),
    )
    parser.add_argument(
        "--wf-save-trajectories",
        default=None,
        help=(
            "Directory to persist per-track per-window ANOMALY (-loglik) trajectories "
            "(wf_trajectories_<emb>.npz + _meta.csv). Enables scripts/ablate_aggregation.py "
            "to sweep track-score aggregators (mean/median/top-k%%/max/...) as pure CPU "
            "post-processing without any re-scoring."
        ),
    )
    parser.add_argument(
        "--wf-save-fold-flows",
        action="store_true",
        default=False,
        help=(
            "Also save each of the 5 K-fold flows (suffix _fold<i>) next to "
            "--wf-save-flow-path, so held-out real scores are reproducible from "
            "artifacts without retraining. Requires --wf-save-flow-path."
        ),
    )
    parser.add_argument(
        "--genre-csv",
        default=None,
        help=(
            "CSV with columns track_id, genre_top1 (e.g. reports/confound_analysis or "
            "genre_tags_all/all_genres.csv). Required by --wf-genre-balanced-sampling."
        ),
    )
    parser.add_argument(
        "--wf-genre-balanced-sampling",
        action="store_true",
        default=False,
        help=(
            "Train the window-flow with inverse-genre-frequency WEIGHTED sampling "
            "(WeightedRandomSampler-style, with replacement) instead of uniform "
            "sampling, so rare genres get roughly equal representation per epoch. "
            "This tests BALANCING as distinct from B2's broadening (adding more real "
            "data): same 12,722-track SONICS-real corpus, only the sampling distribution "
            "changes. Requires --genre-csv. Window count and total training data are "
            "unchanged — no windows are duplicated or dropped, only reweighted."
        ),
    )
    parser.add_argument(
        "--genre-weight-cap",
        type=float,
        default=20.0,
        help=(
            "Cap on inverse-frequency genre weight (relative to the largest genre) to "
            "avoid extreme over-sampling of ultra-rare genres (e.g. baroque, n=22) "
            "collapsing the effective training set. Default 20x."
        ),
    )
    parser.add_argument(
        "--temporal-max-points",
        type=int,
        default=300,
        help=(
            "Max points per temporal window for ID estimation. "
            "PHD distance-matrix cost is O(n²) so this is the most impactful speed knob. "
            "300 is good for pilots (trajectory shape is preserved); "
            "use 500-1000 for final publication runs. "
            "Independent of --max-points (which controls static full/half/quarter scales)."
        ),
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=None,
        help=(
            "Apply a Butterworth low-pass filter at this frequency (Hz) to BOTH classes before "
            "embedding extraction. Recommended: 8000 Hz to remove the spectral bandwidth confound "
            "in SONICS (fakes are band-limited to ~8 kHz). Applied after canonical/preprocessed load."
        ),
    )
    parser.add_argument(
        "--analysis-duration",
        type=float,
        default=None,
        help=(
            "Maximum audio duration (seconds) to analyse per track. Use e.g. 60 to equalise "
            "the real (~119 s) vs fake (~97 s) duration confound and reduce MERT runtime. "
            "Also controls temporal window coverage. Default: full canonical max (120 s)."
        ),
    )
    parser.add_argument(
        "--fixed-duration",
        type=float,
        default=None,
        help=(
            "Centre-crop (or zero-pad) every track to EXACTLY this many seconds before "
            "embedding extraction. Unlike --analysis-duration (an upper cap), this forces a "
            "constant length so all tracks yield the same number of temporal windows — "
            "structurally removing the duration / window-count confound (e.g. SONICS udio-30s). "
            "Recommended ~50 for the fair fixed-duration subsample."
        ),
    )
    parser.add_argument(
        "--analysis-region",
        choices=["full", "intro", "middle", "outro", "random"],
        default="full",
        help=(
            "Which region of the audio to extract for embedding. "
            "full=entire track (default); intro=first N seconds; middle=center N seconds "
            "(same as centre-crop / legacy default when --fixed-duration is set); "
            "outro=last N seconds; random=random N-second crop. "
            "N is determined by --fixed-duration when set, otherwise --analysis-duration / "
            "--max-duration. This is the Phase-1 sweep variable: test the 'AI forgets late' "
            "hypothesis (outro vs intro) and find the optimal analysis region. "
            "Cache keys are region-tagged so runs with different regions do not collide."
        ),
    )
    parser.add_argument(
        "--flow-only",
        action="store_true",
        default=False,
        help=(
            "Skip the per-track DESCRIPTOR/intrinsic-dimension stage (twonn/PHD, window "
            "descriptors, spectral features) and the supervised classification bundles, and run "
            "ONLY the window-flow density eval. Those descriptors are a legacy research track "
            "that the flow pipeline never consumes, yet they dominate wall-clock: ~56 min of a "
            "~3 h run at 9.5k tracks, and ~110 min at 33k. Embeddings are still extracted (and "
            "cached), so the flow sees identical input. Requires --window-flow-eval."
        ),
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help=(
            "Skip balanced sampling AND embedding extraction entirely; reload the existing "
            "ablation_{emb}.csv feature tables from --output-dir and re-run only the "
            "classification + stratified + generator-shift evaluation. Seconds on CPU — use "
            "when iterating on classifier bundles without changing per-track features."
        ),
    )
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help=(
            "Ignore the per-embedding checkpoint_{emb}.json and recompute every per-track "
            "feature from the cached embedding .npy files (no GPU forward pass when "
            "--embedding-cache-dir is populated). Use when adding new window descriptors / "
            "ID estimators on a fixed embedding cache."
        ),
    )
    parser.add_argument(
        "--descriptors-only",
        action="store_true",
        help=(
            "Reload the existing ablation_{emb}.csv from --output-dir and recompute ONLY the "
            "window descriptors (wd_*) from the embedding cache (--embedding-cache-dir), "
            "reusing every saved PHD/TwoNN/temporal column unchanged. Then re-run classification "
            "+ stratified + generator-shift. Near-instant — use to iterate on descriptor windowing "
            "without recomputing the O(n²) PHD features. Requires --embedding-cache-dir."
        ),
    )
    args = parser.parse_args()

    if args.canonical_manifest and args.preprocess_mode == "canonical":
        logger.warning(
            "--canonical-manifest points to pre-built WAVs; switching --preprocess-mode "
            "from 'canonical' to 'preprocessed' to avoid double preprocessing."
        )
        args.preprocess_mode = "preprocessed"

    # prevent CLAP from hijacking our args
    sys.argv = [sys.argv[0]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    emb_names = ["encodec", "clap", "mert", "muq"] if "all" in args.embeddings else list(args.embeddings)
    estimators = list(args.estimators)
    if args.include_mle and "mle" not in estimators:
        estimators.append("mle")

    lowpass_hz = getattr(args, "lowpass_hz", None)
    analysis_duration = getattr(args, "analysis_duration", None)
    fixed_duration = getattr(args, "fixed_duration", None)
    max_duration_eff = analysis_duration if analysis_duration else args.max_duration
    temporal_max_points = getattr(args, "temporal_max_points", 300)

    # --- spectrogram dynamic-range floor -----------------------------------
    # Only the spectrogram extractors accept it; passing it to EnCodec/MERT/etc.
    # would be a TypeError on the config constructor, so refuse loudly rather
    # than silently drop it (a silently-dropped knob produces a run that looks
    # like the requested configuration and is not).
    spec_log_floor_db = getattr(args, "spec_log_floor_db", None)
    _SPEC_ARMS = {"spec", "logmel", "spec-musicdet"}
    if spec_log_floor_db is not None:
        non_spec = [e for e in emb_names if e not in _SPEC_ARMS]
        if non_spec:
            raise SystemExit(
                f"--spec-log-floor-db applies only to {sorted(_SPEC_ARMS)}; "
                f"got --embeddings {non_spec}. Split the run."
            )
    extractor_kwargs = {} if spec_log_floor_db is None else {"log_floor_db": spec_log_floor_db}

    level_match = bool(getattr(args, "level_match", False))
    level_peak = bool(getattr(args, "level_peak_normalise", False))
    if level_peak and not level_match:
        raise SystemExit("--level-peak-normalise requires --level-match")
    resample_hz = getattr(args, "resample_hz", None)
    if lowpass_hz and not resample_hz:
        logger.warning(
            "--lowpass-hz WITHOUT --resample-hz is not a bandwidth control. Measured on "
            "canonical SONICS: no control -> channel-alone AUC 1.0000; --lowpass-hz 7000 -> "
            "0.9287 (the leak moves to spectral flatness); --resample-hz 14000 -> 0.5766. "
            "Results from this run must not be described as bandwidth-matched."
        )

    # Any knob that changes the stored vectors must be in the cache key.
    cache_extra_tags: tuple[str, ...] = tuple(
        t
        for t in (
            band_match_tag(resample_hz),
            level_match_tag(level_match, level_peak),
            spec_floor_tag(spec_log_floor_db),
        )
        if t
    )

    logger.info("Command    : %s", " ".join(sys.argv))
    logger.info("Embeddings : %s", emb_names)
    logger.info("Estimators : %s", estimators)
    logger.info("Temporal   : %s (window=%.0fs hop=%.0fs)", args.with_temporal, args.window_duration, args.hop_duration)
    logger.info("Per-stratum: %d", args.per_stratum)
    if lowpass_hz:
        logger.info(
            "Low-pass   : %.0f Hz  ⚠️ NOT a bandwidth control on its own (leaves 0.9287 "
            "channel-alone AUC on SONICS) — pair it with --resample-hz",
            lowpass_hz,
        )
    if resample_hz:
        logger.info(
            "Band-match : decimate to %.0f Hz and back (Nyquist %.0f Hz shared by both classes)",
            resample_hz,
            resample_hz / 2,
        )
    if level_match:
        logger.info("Level-match: DC removed (20 Hz HP) + leading/trailing silence trimmed")
    if spec_log_floor_db is not None:
        logger.info(
            "Spec floor : %.0f dB below per-track peak (empty bands become quiet, not a constant); " "cache tag %s",
            spec_log_floor_db,
            cache_extra_tags,
        )
    if analysis_duration:
        logger.info("Duration   : capped at %.0f s (duration confound reduction)", analysis_duration)
    if fixed_duration:
        logger.info("Fixed-dur  : centre-crop to %.1f s (constant window count)", fixed_duration)
    logger.info(
        "Max-points : static=%d  temporal=%d (PHD O(n²) — main speed knob)",
        getattr(args, "max_points", 2000),
        temporal_max_points,
    )

    embedding_dfs: dict[str, pd.DataFrame] = {}

    # --- fast path: re-score the one-class eval on an existing merged table ---
    from_features = getattr(args, "anomaly_from_features", None)
    if from_features:
        feat_path = Path(from_features)
        if not feat_path.exists():
            logger.error("--anomaly-from-features: file not found: %s", feat_path)
            sys.exit(1)
        merged = pd.read_csv(feat_path)
        inferred = sorted({c.split("_wd_")[0] for c in merged.columns if "_wd_" in c})
        if not inferred:
            logger.error("--anomaly-from-features: no *_wd_* descriptor columns in %s", feat_path)
            sys.exit(1)
        logger.info("ANOMALY-FROM-FEATURES: %s (%d rows), embeddings=%s", feat_path, len(merged), inferred)
        _run_real_anomaly_eval(
            merged,
            inferred,
            output_dir,
            one_class_method=args.one_class_method,
            device=args.device,
        )
        return

    # --- classify-only fast path: reload cached per-embedding feature tables ---
    if getattr(args, "classify_only", False):
        logger.info("CLASSIFY-ONLY: reloading ablation_{emb}.csv from %s (no extraction)", output_dir)
        for emb_name in emb_names:
            csv_path = output_dir / f"ablation_{emb_name}.csv"
            if csv_path.exists():
                embedding_dfs[emb_name] = pd.read_csv(csv_path)
                logger.info("  loaded %s (%d rows)", csv_path.name, len(embedding_dfs[emb_name]))
            else:
                logger.warning("  missing %s — skipping", csv_path.name)
        if not embedding_dfs:
            logger.error("classify-only: no ablation_{emb}.csv found in %s", output_dir)
            sys.exit(1)
    elif getattr(args, "descriptors_only", False):
        cache_dir = getattr(args, "embedding_cache_dir", None)
        if not cache_dir:
            logger.error("--descriptors-only requires --embedding-cache-dir")
            sys.exit(1)
        logger.info(
            "DESCRIPTORS-ONLY: recomputing wd_* from %s, reusing saved ID features (window=%.0fs hop=%.0fs)",
            cache_dir,
            args.window_duration,
            args.hop_duration,
        )
        for emb_name in emb_names:
            csv_path = output_dir / f"ablation_{emb_name}.csv"
            if not csv_path.exists():
                logger.warning("  missing %s — skipping", csv_path.name)
                continue
            cfg = EMBEDDING_CONFIGS[emb_name]
            df = _recompute_descriptors_for_df(
                pd.read_csv(csv_path),
                emb_name,
                embedding_cache_dir=cache_dir,
                target_sr=cfg["target_sr"],
                max_duration=max_duration_eff,
                preprocess_mode=getattr(args, "preprocess_mode", "raw"),
                lowpass_hz=lowpass_hz,
                fixed_duration=fixed_duration,
                window_duration=args.window_duration,
                hop_duration=args.hop_duration,
                cache_extra_tags=cache_extra_tags,
            )
            df.to_csv(csv_path, index=False)
            embedding_dfs[emb_name] = df
            logger.info("  updated %s (%d rows)", csv_path.name, len(df))
        if not embedding_dfs:
            logger.error("descriptors-only: no ablation_{emb}.csv found in %s", output_dir)
            sys.exit(1)
    else:
        # --- balanced sampling ---
        real_tracks, fake_tracks, coverage_df = load_balanced_tracks(
            per_stratum=args.per_stratum,
            fake_types=args.fake_types,
            algorithms=args.algorithms,
            max_real=args.max_real,
            seed=args.seed,
            real_genres_csv=getattr(args, "real_genres_csv", None),
            covariate_profiles_csv=getattr(args, "covariate_profiles_csv", None),
            match_covariates=True,
            canonical_manifest=getattr(args, "canonical_manifest", None),
        )
        coverage_df.to_csv(output_dir / "stratum_coverage.csv", index=False)
        all_tracks = real_tracks + fake_tracks
        logger.info("Total tracks to process per embedding: %d", len(all_tracks))

        # --flow-only: reuse the SAME extraction/caching path (run_embedding) but
        # with an empty estimator list, so estimate_all() is a no-op and the
        # expensive per-track intrinsic-dimension work (twonn, the dominant
        # wall-clock cost) never runs. Reusing the real path — rather than
        # reimplementing extraction — guarantees the embedding cache is
        # byte-identical to a normal run.
        _flow_only = bool(getattr(args, "flow_only", False))
        if _flow_only:
            if not getattr(args, "window_flow_eval", False):
                logger.error("--flow-only requires --window-flow-eval")
                sys.exit(1)
            if not getattr(args, "embedding_cache_dir", None):
                logger.error("--flow-only requires --embedding-cache-dir")
                sys.exit(1)
            logger.info(
                "FLOW-ONLY: skipping intrinsic-dimension/descriptor computation and the "
                "supervised classification bundles — the window-flow eval consumes neither."
            )
            estimators = []

        # --- run each embedding ---
        for emb_name in emb_names:
            df = run_embedding(
                all_tracks,
                emb_name=emb_name,
                device=args.device,
                estimators=estimators,
                max_duration=max_duration_eff,
                with_temporal=args.with_temporal,
                window_duration=args.window_duration,
                hop_duration=args.hop_duration,
                output_dir=output_dir,
                seed=args.seed,
                preprocess_mode=getattr(args, "preprocess_mode", "raw"),
                embedding_cache_dir=getattr(args, "embedding_cache_dir", None),
                max_points=getattr(args, "max_points", 2000),
                lowpass_hz=lowpass_hz,
                temporal_max_points=temporal_max_points,
                fixed_duration=fixed_duration,
                recompute_features=getattr(args, "recompute_features", False),
                analysis_region=getattr(args, "analysis_region", "full"),
                cache_dtype=getattr(args, "emb_cache_dtype", "float32"),
                extractor_kwargs=extractor_kwargs,
                cache_extra_tags=cache_extra_tags,
                resample_hz=resample_hz,
                level_match=level_match,
                level_peak=level_peak,
            )
            if df is not None:
                embedding_dfs[emb_name] = df

    if not embedding_dfs:
        logger.error("No embeddings produced results.")
        sys.exit(1)

    # --- build wide fusion table ---
    logger.info("Building wide ablation feature table...")
    meta_cols = ["track_id", "label", "split", "source", "fake_label", "algorithm"]
    merged: pd.DataFrame | None = None

    for emb_name, df in embedding_dfs.items():
        keep_meta = [c for c in meta_cols if c in df.columns]
        feat_cols = [c for c in df.columns if c.startswith(f"{emb_name}_")]
        temp = df[keep_meta + feat_cols].copy()
        if merged is None:
            merged = temp
        else:
            merge_on = [c for c in meta_cols if c in merged.columns and c in temp.columns]
            merged = merged.merge(temp[merge_on + feat_cols], on=merge_on, how="inner")

    if merged is None:
        logger.error("Failed to build fusion table.")
        sys.exit(1)

    merged["y"] = (merged["label"] == "fake").astype(int)
    merged.to_csv(output_dir / "ablation_features.csv", index=False)
    logger.info(
        "Wide table: %d rows × %d cols → ablation_features.csv",
        len(merged),
        len(merged.columns),
    )

    # --- classification + stratified eval ---
    if getattr(args, "flow_only", False):
        logger.info("FLOW-ONLY: skipping classification bundles and stratified eval.")
    else:
        logger.info("Running classification bundles and stratified evaluation...")
        run_classification_and_eval(merged, list(embedding_dfs.keys()), estimators, output_dir)

    # --- generator-shift zero-shot eval ---
    if getattr(args, "generator_shift_eval", False):
        logger.info("Running cross-generator zero-shot evaluation...")
        _run_generator_shift_eval(merged, list(embedding_dfs.keys()), estimators, output_dir)

    # --- one-class real-manifold (label-free) eval ---
    if getattr(args, "real_anomaly_eval", False):
        logger.info("Running one-class real-manifold anomaly evaluation...")
        _run_real_anomaly_eval(
            merged,
            list(embedding_dfs.keys()),
            output_dir,
            one_class_method=getattr(args, "one_class_method", "mahalanobis"),
            device=getattr(args, "device", "cpu"),
        )

    # --- per-window flow likelihood trajectory eval (Phase 2 core) ---
    if getattr(args, "window_flow_eval", False):
        emb_cache = getattr(args, "embedding_cache_dir", None)
        if not emb_cache:
            logger.error("--window-flow-eval requires --embedding-cache-dir")
        else:
            genre_weight_by_track = None
            if getattr(args, "wf_genre_balanced_sampling", False):
                genre_csv = getattr(args, "genre_csv", None)
                if not genre_csv:
                    logger.error("--wf-genre-balanced-sampling requires --genre-csv")
                else:
                    genre_weight_by_track = _build_genre_inverse_frequency_weights(
                        genre_csv, merged, weight_cap=getattr(args, "genre_weight_cap", 20.0)
                    )
            logger.info("Running per-window flow likelihood trajectory eval...")
            wf_df = _run_window_flow_eval(
                merged,
                list(embedding_dfs.keys()),
                output_dir,
                embedding_cache_dir=emb_cache,
                window_duration=getattr(args, "wf_window_duration", 4.0),
                hop_duration=getattr(args, "wf_hop_duration", 2.0),
                max_duration=max_duration_eff,
                preprocess_mode=getattr(args, "preprocess_mode", "raw"),
                lowpass_hz=lowpass_hz,
                fixed_duration=fixed_duration,
                analysis_region=getattr(args, "analysis_region", "full"),
                device=getattr(args, "device", "cpu"),
                pca_components=getattr(args, "wf_pca_components", 64),
                flow_epochs=getattr(args, "wf_flow_epochs", 200),
                seed=args.seed,
                save_flow_path=getattr(args, "wf_save_flow_path", None),
                max_windows=getattr(args, "wf_max_windows", None),
                genre_weight_by_track=genre_weight_by_track,
                prior_mean=getattr(args, "wf_prior_mean", 0.0),
                n_coupling_layers=getattr(args, "wf_n_coupling_layers", 8),
                hidden_dim=getattr(args, "wf_hidden_dim", 128),
                pooling_mode=getattr(args, "wf_pooling", "mean"),
                frame_stride=getattr(args, "wf_frame_stride", 1),
                pooling_blocks=getattr(args, "wf_pooling_blocks", 8),
                symmetric_fold_scoring=getattr(args, "wf_symmetric_fold_scoring", False),
                max_train_windows=getattr(args, "wf_max_train_windows", None),
                n_bands=getattr(args, "wf_n_bands", 1),
                global_prior=getattr(args, "wf_global_prior", False),
                global_layers=getattr(args, "wf_global_prior_layers", None),
                background=getattr(args, "wf_background", "none"),
                background_sigma=getattr(args, "wf_background_sigma", 0.5),
                augment_p=getattr(args, "wf_augment_p", 0.0),
                augment_n_masks=getattr(args, "wf_augment_n_masks", 1),
                augment_width=tuple(int(v) for v in str(getattr(args, "wf_augment_width", "6,20")).split(",")),
                save_trajectories_dir=getattr(args, "wf_save_trajectories", None),
                save_fold_flows=getattr(args, "wf_save_fold_flows", False),
                cache_extra_tags=cache_extra_tags,
            )
            # Phase 5: supervised fusion of descriptors + flow trajectory features
            if wf_df is not None and not wf_df.empty:
                logger.info("Running Phase-5 supervised fusion (descriptors + flow trajectories)...")
                _run_phase5_fusion_eval(wf_df, merged, list(embedding_dfs.keys()), output_dir)

    logger.info("=" * 60)
    logger.info("All outputs saved to %s", output_dir)
    logger.info("  stratum_coverage.csv           sampling coverage per stratum")
    logger.info("  ablation_features.csv          wide-format per-track features")
    logger.info("  ablation_{emb}.csv             per-embedding CSV")
    logger.info("  classification_ablation.csv    CV metrics per bundle, sorted by AUC")
    logger.info("  interpretability_ablation.json coef mean / std / sign_consistency")
    logger.info("  stratified_eval_ablation.csv   per-source / fake_label / algorithm AUC")
    if getattr(args, "generator_shift_eval", False):
        logger.info("  generator_shift_eval.csv       cross-generator zero-shot AUC")
    if getattr(args, "real_anomaly_eval", False):
        logger.info("  real_anomaly_eval.csv          one-class real-manifold AUC (label-free)")
    if getattr(args, "window_flow_eval", False):
        logger.info("  window_flow_eval.csv           per-window flow likelihood trajectory features")
        logger.info("  ablation_features_with_flow.csv  merged features incl. wf_ (Phase 5)")
        logger.info("  classification_phase5.csv        Phase-5 supervised fusion (descriptors + flow)")


if __name__ == "__main__":
    main()
