"""Balanced and genre-covariate-stratified sampling for SONICS experiments.

Provides a shared :func:`load_balanced_tracks` that replaces the duplicated
per-script implementations in ``run_balanced_ablation.py`` and
``run_acoustic_analysis.py``.

Sampling strategy
-----------------
1. **Fake stratification**: sample ``per_stratum`` fake tracks per
   ``algorithm × fake_label`` cell to ensure all generator variants are
   equally represented.
2. **Genre matching**: if ``real_genres_csv`` is provided, select real tracks
   to mirror the genre distribution of the sampled fake set.  Otherwise, fall
   back to random sampling (with a warning).
3. **Covariate matching**: optionally match real tracks to fake tracks on
   continuous covariates (duration, tempo, loudness_lufs, spectral_centroid)
   via nearest-neighbour matching in standardised covariate space.  Requires
   ``covariate_profiles_csv``.

All three steps can be combined for maximum fairness, or used individually
depending on what data is available at call time.

Public API
----------
load_balanced_tracks(
    per_stratum=200,
    fake_types=None,
    algorithms=None,
    max_real=None,
    seed=42,
    real_genres_csv=None,       # from tag_real_genres.py
    covariate_profiles_csv=None, # from profile_covariates.py
    match_covariates=True,
    canonical_manifest=None,    # prefer canonical paths when available
) -> tuple[list[dict], list[dict]]   # (real_tracks, fake_tracks)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")

# Covariates used for nearest-neighbour matching (when available)
MATCH_COVARIATES = ["duration", "tempo", "loudness_lufs", "spectral_centroid_mean"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_audio_path(
    row: pd.Series,
    audio_dir: Path,
    id_col: str,
    filename_col: str = "filename",
) -> str | None:
    """Resolve the audio file path from metadata row, checking several candidates."""
    # Explicit canonical / audio_path column
    for col in ("canonical_path", "audio_path"):
        val = row.get(col, "")
        if val and pd.notna(val) and Path(str(val)).exists():
            return str(val)

    track_id = str(row.get(id_col, "")).strip()
    if not track_id or not audio_dir.exists():
        return None

    for pat in (f"{track_id}.*", f"{Path(track_id).stem}.*"):
        cands = list(audio_dir.glob(pat))
        if cands:
            return str(cands[0])

    # Fallback: recursive search handles algorithm-subdirectory structures such as
    # fake_songs/chirp-v3.5/abc.mp3 that flat glob misses.
    stem = Path(track_id).stem
    cands = list(audio_dir.rglob(f"{stem}.*"))
    if cands:
        return str(cands[0])

    return None


def _load_fake_tracks(
    fake_csv: Path,
    per_stratum: int,
    fake_types: list[str] | None,
    algorithms: list[str] | None,
    seed: int,
    audio_dir: Path,
) -> pd.DataFrame:
    """Load and stratify-sample fake track metadata."""
    df = pd.read_csv(fake_csv, low_memory=False)

    # Normalise column names (SONICS CSV uses 'label' for fake_label)
    if "label" in df.columns and "fake_label" not in df.columns:
        df = df.rename(columns={"label": "fake_label"})

    if fake_types:
        df = df[df["fake_label"].astype(str).isin(fake_types)]
    if algorithms:
        df = df[df["algorithm"].astype(str).isin(algorithms)]

    # Stratified sample per algorithm × fake_label cell
    strata = df.groupby(["algorithm", "fake_label"], sort=True)
    sampled: list[pd.DataFrame] = []
    for name, grp in strata:
        n = min(per_stratum, len(grp))
        sampled.append(grp.sample(n, random_state=seed))

    if not sampled:
        return pd.DataFrame()

    sampled_df = pd.concat(sampled, ignore_index=True)

    # Build stem→path index once for O(1) per-track lookups.
    # SONICS downloads organise fakes in algorithm subdirectories
    # (e.g. fake_songs/chirp-v3.5/abc.mp3); a flat glob misses these.
    audio_index: dict[str, str] = {}
    if audio_dir.exists():
        for _af in audio_dir.rglob("*"):
            if _af.is_file() and _af.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
                audio_index[_af.stem.lower()] = str(_af)
        logger.info("Built fake audio index: %d files under %s", len(audio_index), audio_dir)

    # Resolve audio paths
    rows = []
    for _, row in sampled_df.iterrows():
        stem_key = Path(str(row.get("filename", ""))).stem.lower()
        path = audio_index.get(stem_key) or _resolve_audio_path(row, audio_dir, id_col="filename")
        if path:
            track_id = str(row.get("filename", row.get("id", ""))).strip()
            rows.append(
                {
                    "track_id": Path(track_id).stem,
                    "path": path,
                    "label": "fake",
                    "algorithm": str(row.get("algorithm", "")),
                    "fake_label": str(row.get("fake_label", "")),
                    "genre": str(row.get("genre", "")),
                    "style": str(row.get("style", "")),
                    "mood": str(row.get("mood", "")),
                }
            )

    return pd.DataFrame(rows)


def _load_real_tracks_with_genres(
    real_csv: Path,
    real_genres_csv: str | None,
    fake_genre_counts: pd.Series,
    n_target: int,
    seed: int,
    audio_dir: Path,
    covariate_profiles_csv: str | None = None,
    match_covariates: bool = True,
    fake_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load real tracks, optionally genre-matching and covariate-matching to fakes."""
    df_real = pd.read_csv(real_csv, low_memory=False)

    # Merge inferred genres if available
    genre_col = "inferred_genre"
    if real_genres_csv and Path(real_genres_csv).exists():
        df_genres = pd.read_csv(real_genres_csv, low_memory=False)
        df_genres = df_genres.rename(columns={"genre_top1": "inferred_genre"})
        merge_col = "youtube_id" if "youtube_id" in df_real.columns else "track_id"
        df_real = df_real.merge(
            df_genres[["track_id", "inferred_genre"]].rename(columns={"track_id": merge_col}),
            on=merge_col,
            how="left",
        )
        logger.info(
            "Merged genre tags: %d/%d reals have a genre", df_real["inferred_genre"].notna().sum(), len(df_real)
        )
    else:
        df_real["inferred_genre"] = None
        if real_genres_csv:
            logger.warning("real_genres_csv not found (%s) — genre matching disabled", real_genres_csv)
        else:
            logger.info("No real_genres_csv provided — real tracks sampled without genre matching")

    # Resolve audio paths
    real_with_paths = []
    id_col = "youtube_id" if "youtube_id" in df_real.columns else "track_id"
    for _, row in df_real.iterrows():
        path = _resolve_audio_path(row, audio_dir, id_col=id_col)
        if path:
            yt_id = str(row.get(id_col, "")).strip()
            real_with_paths.append(
                {
                    "track_id": yt_id,
                    "path": path,
                    "label": "real",
                    "algorithm": "",
                    "fake_label": "",
                    "genre": str(row.get(genre_col, "")),
                    "artist": str(row.get("artist", "")),
                    "year": str(row.get("year", "")),
                }
            )

    df_pool = pd.DataFrame(real_with_paths)
    if df_pool.empty:
        return df_pool

    # Genre-stratified selection
    if fake_genre_counts is not None and df_real["inferred_genre"].notna().any():
        selected_rows: list[pd.DataFrame] = []
        remaining = df_pool.copy()

        for genre, target_n in fake_genre_counts.items():
            genre_pool = remaining[remaining["genre"].astype(str).str.lower() == genre.lower()]
            n_take = min(target_n, len(genre_pool))
            if n_take > 0:
                taken = genre_pool.sample(n_take, random_state=seed)
                selected_rows.append(taken)
                remaining = remaining.drop(taken.index)

        # Fill remaining quota from unmatched tracks (genre not in vocabulary or not tagged)
        if selected_rows:
            selected_df = pd.concat(selected_rows, ignore_index=True)
        else:
            selected_df = pd.DataFrame()

        n_taken = len(selected_df)
        n_fill = max(0, n_target - n_taken)
        if n_fill > 0:
            fill_pool = remaining.head(n_fill) if len(remaining) >= n_fill else remaining
            filler = fill_pool.sample(min(n_fill, len(fill_pool)), random_state=seed)
            selected_df = pd.concat([selected_df, filler], ignore_index=True)

        df_pool = selected_df.head(n_target)
        logger.info("Genre-matched reals: %d selected (target=%d)", len(df_pool), n_target)
    else:
        df_pool = df_pool.sample(min(n_target, len(df_pool)), random_state=seed)
        logger.info("Random real sampling (no genre matching): %d selected", len(df_pool))

    # Covariate matching (nearest-neighbour in standardised covariate space)
    if match_covariates and covariate_profiles_csv and Path(covariate_profiles_csv).exists() and fake_df is not None:
        df_pool = _covariate_match(df_pool, fake_df, covariate_profiles_csv, seed)

    return df_pool


def _covariate_match(
    real_pool: pd.DataFrame,
    fake_df: pd.DataFrame,
    covariate_profiles_csv: str,
    seed: int,
) -> pd.DataFrame:
    """Re-select real tracks to minimise covariate distance to the fake distribution.

    Uses nearest-neighbour matching: for each fake track, find the closest real
    track (by Euclidean distance in standardised covariate space).  Real tracks
    that are never matched are dropped.
    """
    try:
        covs_df = pd.read_csv(covariate_profiles_csv, low_memory=False)
    except Exception as exc:
        logger.warning("Could not load covariate_profiles_csv (%s) — skipping matching", exc)
        return real_pool

    available_covs = [c for c in MATCH_COVARIATES if c in covs_df.columns]
    if not available_covs:
        logger.warning("No matching covariates found in profile CSV — skipping covariate matching")
        return real_pool

    # Add covariates to real_pool and fake_df from profile CSV
    covs_real = covs_df[covs_df["label"] == "real"][["track_id"] + available_covs]
    covs_fake = covs_df[covs_df["label"] == "fake"][["track_id"] + available_covs]

    real_pool = real_pool.merge(
        covs_real.rename(columns={c: f"cov_{c}" for c in available_covs}), on="track_id", how="left"
    )
    fake_df = fake_df.copy()
    fake_df = fake_df.merge(
        covs_fake.rename(columns={c: f"cov_{c}" for c in available_covs}), on="track_id", how="left"
    )

    cov_cols_real = [f"cov_{c}" for c in available_covs]

    X_real = real_pool[cov_cols_real].to_numpy(dtype=float)
    X_fake = fake_df[cov_cols_real].to_numpy(dtype=float)

    # Standardise
    means = np.nanmean(np.vstack([X_real, X_fake]), axis=0)
    stds = np.nanstd(np.vstack([X_real, X_fake]), axis=0) + 1e-8
    X_real_std = (X_real - means) / stds
    X_fake_std = (X_fake - means) / stds

    # Replace NaN with 0 (mean in standardised space)
    X_real_std = np.where(np.isfinite(X_real_std), X_real_std, 0.0)
    X_fake_std = np.where(np.isfinite(X_fake_std), X_fake_std, 0.0)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(X_real_std)
    _, indices = nn.kneighbors(X_fake_std)

    matched_indices = np.unique(indices.ravel())
    matched_real = real_pool.iloc[matched_indices].copy()
    # Drop the temporary covariate columns
    matched_real = matched_real[[c for c in matched_real.columns if not c.startswith("cov_")]]

    logger.info(
        "Covariate matching: %d reals → %d matched (using %s)",
        len(real_pool),
        len(matched_real),
        available_covs,
    )
    return matched_real


# ---------------------------------------------------------------------------
# Canonical manifest helper
# ---------------------------------------------------------------------------


def _override_with_canonical_paths(tracks: list[dict], canonical_manifest: str) -> list[dict]:
    """Replace audio paths with canonical WAV paths when available."""
    manifest_path = Path(canonical_manifest)
    if not manifest_path.exists():
        logger.warning("canonical_manifest not found: %s", canonical_manifest)
        return tracks

    df_can = pd.read_csv(manifest_path, low_memory=False)
    ok_df = df_can[df_can.get("status", "ok") == "ok"]
    canon_map: dict[str, str] = {}
    for _, row in ok_df.iterrows():
        tid = str(row.get("track_id", "")).strip()
        cpath = str(row.get("canonical_path", "")).strip()
        if tid and cpath and Path(cpath).exists():
            canon_map[tid] = cpath

    updated = []
    n_replaced = 0
    for t in tracks:
        cp = canon_map.get(t["track_id"])
        if cp:
            updated.append({**t, "path": cp})
            n_replaced += 1
        else:
            updated.append(t)

    logger.info("Canonical paths substituted for %d/%d tracks", n_replaced, len(tracks))
    return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_balanced_tracks(
    per_stratum: int = 200,
    fake_types: list[str] | None = None,
    algorithms: list[str] | None = None,
    max_real: int | None = None,
    seed: int = 42,
    real_genres_csv: str | None = None,
    covariate_profiles_csv: str | None = None,
    match_covariates: bool = True,
    canonical_manifest: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load a balanced set of real and fake SONICS tracks.

    Parameters
    ----------
    per_stratum
        Maximum tracks per ``algorithm × fake_label`` stratum.
    fake_types
        If provided, restrict to these ``fake_label`` values.
    algorithms
        If provided, restrict to these ``algorithm`` values.
    max_real
        Override the total number of real tracks.
        Default: match the total fake count.
    seed
        Random seed for reproducibility.
    real_genres_csv
        Path to ``real_genres.csv`` from ``tag_real_genres.py``.
        When provided, real tracks are genre-matched to the fake distribution.
    covariate_profiles_csv
        Path to ``covariate_profiles.csv`` from ``profile_covariates.py``.
        When provided and ``match_covariates=True``, uses nearest-neighbour
        matching in standardised covariate space.
    match_covariates
        Whether to perform covariate matching (requires ``covariate_profiles_csv``).
    canonical_manifest
        Path to ``canonical_manifest.csv`` from ``build_canonical_corpus.py``.
        When provided, canonical WAV paths are substituted for original paths.

    Returns
    -------
    real_tracks : list[dict]
        Each dict has keys: track_id, path, label, algorithm, fake_label, genre.
    fake_tracks : list[dict]
        Same schema.
    """
    metadata_dir = SONICS_DIR / "metadata"

    real_csv = metadata_dir / "real_songs_ready.csv"
    if not real_csv.exists():
        real_csv = metadata_dir / "real_songs.csv"
    fake_csv = metadata_dir / "fake_songs_ready.csv"
    if not fake_csv.exists():
        fake_csv = metadata_dir / "fake_songs.csv"

    # The SONICS metadata CSVs are only needed for the SONICS fallback path. A
    # caller supplying an explicit canonical_manifest is working with a
    # different corpus (FakeMusicCaps, FMA, ...) and must not be forced to have
    # SONICS on disk — nor silently routed through it.
    if not canonical_manifest and (not real_csv.exists() or not fake_csv.exists()):
        raise FileNotFoundError(
            "SONICS metadata CSVs not found. "
            f"Expected: {real_csv} and {fake_csv}. "
            "Run: python scripts/download_sonics.py --prepare"
        )

    real_audio_dir = SONICS_DIR / "real_songs"
    fake_audio_dir = SONICS_DIR / "fake_songs"

    # If canonical manifest provided, it has canonical_path already
    if canonical_manifest and Path(canonical_manifest).exists():
        df_can = pd.read_csv(canonical_manifest, low_memory=False)
        ok_df = df_can[df_can.get("status", "ok") == "ok"]
        # Update fake_csv / real_csv logic to use canonical manifest
        # Build interim lookup from manifest
        real_manifest = ok_df[ok_df["label"] == "real"]
        fake_manifest = ok_df[ok_df["label"] == "fake"]

        real_rows = []
        for _, row in real_manifest.iterrows():
            path = str(row.get("canonical_path", row.get("src_path", "")))
            if path and Path(path).exists():
                real_rows.append(
                    {
                        "track_id": str(row.get("track_id", "")),
                        "path": path,
                        "label": "real",
                        "algorithm": "",
                        "fake_label": "",
                        "genre": "",
                        "youtube_id": str(row.get("track_id", "")),
                    }
                )
        fake_rows = []
        for _, row in fake_manifest.iterrows():
            path = str(row.get("canonical_path", row.get("src_path", "")))
            if path and Path(path).exists():
                fake_rows.append(
                    {
                        "track_id": str(row.get("track_id", "")),
                        "path": path,
                        "label": "fake",
                        "algorithm": str(row.get("algorithm", "")),
                        "fake_label": str(row.get("fake_label", "")),
                        "genre": str(row.get("genre", "")),
                    }
                )

        df_real_pool = pd.DataFrame(real_rows)
        df_fake_raw = pd.DataFrame(fake_rows)

        # HARD FAIL — never silently substitute a different corpus.
        # When a canonical manifest is passed explicitly, the caller means "use
        # THIS corpus". If it yields no usable rows (missing/renamed audio, or a
        # manifest without a canonical_path column), the code below would fall
        # through to the hard-coded SONICS CSVs and train on a completely
        # different dataset while logging nothing. That has now happened twice
        # in this project's history (once via a stale canonical_path after a
        # `mv`, once via a reconstructed manifest that omitted the column) and
        # each time it silently produced hours of results for the wrong corpus.
        if df_real_pool.empty or df_fake_raw.empty:
            has_path_col = "canonical_path" in ok_df.columns or "src_path" in ok_df.columns
            raise ValueError(
                f"canonical_manifest {canonical_manifest!r} produced "
                f"{len(df_real_pool)} usable real and {len(df_fake_raw)} usable fake rows "
                f"out of {len(ok_df)} status==ok rows"
                + (
                    " — the manifest has NO 'canonical_path'/'src_path' column."
                    if not has_path_col
                    else " — the referenced audio files do not exist on disk."
                )
                + " Refusing to continue, because the fallback path would silently train on the "
                "hard-coded SONICS corpus instead. Rebuild the canonical corpus (or fix the "
                "paths in the manifest) and retry."
            )
    else:
        if canonical_manifest:
            raise FileNotFoundError(
                f"canonical_manifest {canonical_manifest!r} does not exist. Refusing to fall back "
                "to the hard-coded SONICS corpus — that would train on a different dataset."
            )
        df_real_pool = None  # will be loaded inside _load_real_tracks_with_genres
        df_fake_raw = None

    # Sample fakes
    if df_fake_raw is not None and not df_fake_raw.empty:
        # Stratify from pre-loaded manifest
        fake_label_col = "fake_label"
        if fake_types:
            df_fake_raw = df_fake_raw[df_fake_raw[fake_label_col].isin(fake_types)]
        if algorithms:
            df_fake_raw = df_fake_raw[df_fake_raw["algorithm"].isin(algorithms)]

        strata = df_fake_raw.groupby(["algorithm", "fake_label"], sort=True)
        sampled: list[pd.DataFrame] = []
        for _, grp in strata:
            n = min(per_stratum, len(grp))
            sampled.append(grp.sample(n, random_state=seed))
        fake_df = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    else:
        fake_df = _load_fake_tracks(fake_csv, per_stratum, fake_types, algorithms, seed, fake_audio_dir)

    if fake_df.empty:
        logger.error("No fake tracks found after filtering.")
        return [], []

    n_real_target = max_real if max_real is not None else len(fake_df)
    logger.info("Fakes sampled: %d  |  Real target: %d", len(fake_df), n_real_target)

    # Genre distribution of sampled fakes
    fake_genre_counts: pd.Series | None = None
    if "genre" in fake_df.columns and fake_df["genre"].notna().any():
        fake_genre_counts = (
            fake_df["genre"]
            .astype(str)
            .str.lower()
            .value_counts()
            .apply(lambda cnt: max(1, int(cnt / len(fake_df) * n_real_target)))
        )

    # Sample reals
    if df_real_pool is not None and not df_real_pool.empty:
        # Merge genre tags if available
        if real_genres_csv and Path(real_genres_csv).exists():
            df_genres = pd.read_csv(real_genres_csv, low_memory=False)
            df_genres = df_genres.rename(columns={"genre_top1": "inferred_genre"})
            df_real_pool = df_real_pool.merge(df_genres[["track_id", "inferred_genre"]], on="track_id", how="left")
            df_real_pool["genre"] = df_real_pool["inferred_genre"].fillna("")

        if fake_genre_counts is not None and "genre" in df_real_pool.columns:
            selected: list[pd.DataFrame] = []
            remaining = df_real_pool.copy()
            for genre, tgt in fake_genre_counts.items():
                grp = remaining[remaining["genre"].astype(str).str.lower() == genre.lower()]
                take = min(int(tgt), len(grp))
                if take > 0:
                    taken = grp.sample(take, random_state=seed)
                    selected.append(taken)
                    remaining = remaining.drop(taken.index)
            real_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
            n_fill = max(0, n_real_target - len(real_df))
            if n_fill > 0:
                fill = remaining.sample(min(n_fill, len(remaining)), random_state=seed)
                real_df = pd.concat([real_df, fill], ignore_index=True)
        else:
            real_df = df_real_pool.sample(min(n_real_target, len(df_real_pool)), random_state=seed)
    else:
        real_df = _load_real_tracks_with_genres(
            real_csv=real_csv,
            real_genres_csv=real_genres_csv,
            fake_genre_counts=fake_genre_counts,
            n_target=n_real_target,
            seed=seed,
            audio_dir=real_audio_dir,
            covariate_profiles_csv=covariate_profiles_csv,
            match_covariates=match_covariates,
            fake_df=fake_df,
        )

    if real_df.empty:
        logger.error("No real tracks found.")
        return [], []

    real_tracks = real_df.to_dict("records")
    fake_tracks = fake_df.to_dict("records")

    logger.info("Final sample: %d real tracks, %d fake tracks", len(real_tracks), len(fake_tracks))
    return real_tracks, fake_tracks
