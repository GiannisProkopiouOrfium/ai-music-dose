"""Diagnose why load_balanced_tracks returns zero fake tracks.

Run from the repo root on EC2:

    python scripts/debug_balanced_sampling.py

This script introspects the *actual* balanced_sampling module API and the
SONICS metadata, isolating the two possible failure modes:

  (A) zero strata  -> algorithm / fake_label column is all-NaN, so
      groupby(["algorithm", "fake_label"]) yields no groups.
  (B) zero resolved -> strata are fine, but _resolve_audio_path() cannot
      locate any audio file on disk for the sampled rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.balanced_sampling import (  # noqa: E402
    SONICS_DIR,
    _load_fake_tracks,
    _resolve_audio_path,
    load_balanced_tracks,
)

METADATA_DIR = SONICS_DIR / "metadata"
FAKE_DIR = SONICS_DIR / "fake_songs"
REAL_DIR = SONICS_DIR / "real_songs"


def _pick_fake_csv() -> Path:
    ready = METADATA_DIR / "fake_songs_ready.csv"
    return ready if ready.exists() else METADATA_DIR / "fake_songs.csv"


def _inspect_csv(fake_csv: Path) -> pd.DataFrame:
    print(f"1. fake CSV: {fake_csv}")
    raw = pd.read_csv(fake_csv, low_memory=False)
    print(f"   rows: {len(raw)}")
    print(f"   columns: {list(raw.columns)}")

    dupes = raw.columns[raw.columns.duplicated()].unique().tolist()
    if dupes:
        print(f"   !! DUPLICATE column names: {dupes}")

    # _load_fake_tracks renames label -> fake_label only if fake_label absent.
    for col in ("label", "fake_label", "algorithm", "source", "filename", "id"):
        if col in raw.columns:
            n = raw[col].notna().sum()
            sample = raw[col].dropna().head(3).tolist()
            print(f"   {col:10s} non-null={n}/{len(raw)}  sample={sample}")

    # Mirror the sampler's normalisation: label -> fake_label if needed.
    df = raw.copy()
    if "label" in df.columns and "fake_label" not in df.columns:
        df = df.rename(columns={"label": "fake_label"})

    if "algorithm" not in df.columns:
        print("   >>> ROOT CAUSE (A): no 'algorithm' column -> groupby fails.")
    elif df["algorithm"].isna().all():
        print("   >>> ROOT CAUSE (A): 'algorithm' is ALL NaN -> zero strata.")
        if "source" in df.columns and df["source"].notna().any():
            print("       (source IS populated -> fill algorithm from source.)")
    if "fake_label" not in df.columns:
        print("   >>> ROOT CAUSE (A): no 'fake_label'/'label' column -> groupby fails.")
    elif df["fake_label"].isna().all():
        print("   >>> ROOT CAUSE (A): 'fake_label' is ALL NaN -> zero strata.")

    if {"algorithm", "fake_label"}.issubset(df.columns):
        n_drop = df.groupby(["algorithm", "fake_label"], sort=True).ngroups
        n_keep = df.groupby(["algorithm", "fake_label"], sort=True, dropna=False).ngroups
        print(f"2. strata: dropna=True -> {n_drop} groups | dropna=False -> {n_keep} groups")
        if n_drop == 0:
            print("   >>> Confirms ROOT CAUSE (A): sampler uses dropna=True -> 0 fakes.")
    return df


def _test_resolution(df: pd.DataFrame) -> None:
    print(f"3. audio dir: {FAKE_DIR}  exists={FAKE_DIR.exists()}")
    if not FAKE_DIR.exists():
        print("   >>> ROOT CAUSE (B): fake audio directory is missing.")
        return

    n_files = sum(1 for _ in FAKE_DIR.iterdir())
    print(f"   files on disk: {n_files}")

    # Which column does the sampler treat as the id? It hardcodes id_col='filename'.
    id_col = "filename" if "filename" in df.columns else ("id" if "id" in df.columns else None)
    print(f"   sampler id_col='filename' present={'filename' in df.columns} (fallback id_col={id_col})")

    sample = df.head(8)
    n_resolved = 0
    for _, row in sample.iterrows():
        path = _resolve_audio_path(row, FAKE_DIR, id_col="filename")
        fname = row.get("filename", row.get("id", ""))
        marker = "OK " if path else "XX "
        if path:
            n_resolved += 1
        print(f"   {marker} filename={fname!r:40s} -> {path}")
    print(f"   resolved {n_resolved}/{len(sample)} sampled rows")
    if n_resolved == 0:
        print("   >>> ROOT CAUSE (B): filename values do not match files on disk.")
        print(f"       example on-disk names: {[p.name for p in list(FAKE_DIR.iterdir())[:3]]}")


def _trace_load_fake_tracks(fake_csv: Path) -> None:
    """Replicate _load_fake_tracks internals with per-step diagnostics."""
    df = pd.read_csv(fake_csv, low_memory=False)
    if "label" in df.columns and "fake_label" not in df.columns:
        df = df.rename(columns={"label": "fake_label"})

    strata = df.groupby(["algorithm", "fake_label"], sort=True)
    sampled = [grp.sample(min(5, len(grp)), random_state=42) for _, grp in strata]
    sampled_df = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    print(f"   trace: sampled_df rows = {len(sampled_df)}")
    if sampled_df.empty:
        print("   >>> sampled_df empty -> ROOT CAUSE (A) strata.")
        return

    n_via_col = n_via_glob = n_fail = 0
    failures: list[str] = []
    for _, row in sampled_df.iterrows():
        col_hit = None
        for col in ("canonical_path", "audio_path"):
            val = row.get(col, "")
            if val and pd.notna(val) and Path(str(val)).exists():
                col_hit = (col, str(val))
                break
        if col_hit:
            n_via_col += 1
            continue
        track_id = str(row.get("filename", "")).strip()
        glob_hit = None
        if track_id and FAKE_DIR.exists():
            for pat in (f"{track_id}.*", f"{Path(track_id).stem}.*"):
                cands = list(FAKE_DIR.glob(pat))
                if cands:
                    glob_hit = str(cands[0])
                    break
        if glob_hit:
            n_via_glob += 1
        else:
            n_fail += 1
            if len(failures) < 6:
                failures.append(f"filename={track_id!r} audio_path={row.get('audio_path', '')!r}")
    print(f"   trace: resolved via column={n_via_col}  via glob={n_via_glob}  failed={n_fail}")
    if n_via_col == 0 and n_via_glob == 0:
        print("   >>> ROOT CAUSE (B): nothing resolves. Sample failures:")
        for f in failures:
            print(f"        {f}")
        ap_series = sampled_df.get("audio_path")
        if ap_series is not None:
            n_ap_exist = ap_series.dropna().map(lambda p: Path(str(p)).exists()).sum()
            print(f"   trace: audio_path values that exist on disk: {int(n_ap_exist)}/{len(sampled_df)}")
            print(f"   trace: sample audio_path values: {ap_series.dropna().head(3).tolist()}")


def main() -> None:
    print("=== debug_balanced_sampling ===\n")
    fake_csv = _pick_fake_csv()
    if not fake_csv.exists():
        print(f"MISSING: {fake_csv}")
        return

    df = _inspect_csv(fake_csv)
    _test_resolution(df)

    print("\n4. direct _load_fake_tracks(per_stratum=5):")
    fake_df = _load_fake_tracks(fake_csv, per_stratum=5, fake_types=None, algorithms=None, seed=42, audio_dir=FAKE_DIR)
    print(f"   -> {len(fake_df)} fake tracks")
    if fake_df.empty:
        _trace_load_fake_tracks(fake_csv)

    print("\n5. load_balanced_tracks(per_stratum=5):")
    real, fake = load_balanced_tracks(per_stratum=5)
    print(f"   -> real={len(real)} fake={len(fake)}")
    if fake:
        print(f"   sample fake path: {fake[0]['path']}")


if __name__ == "__main__":
    main()
