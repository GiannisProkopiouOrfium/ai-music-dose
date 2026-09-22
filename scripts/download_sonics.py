"""Download and prepare the SONICS dataset for experiments.

Fake songs are downloaded from HuggingFace (awsaf49/sonics).
Real songs are either downloaded from S3 (s3://ai-plagiarism-data/ai-downloads/)
or fetched via youtube_id from the metadata CSV.

Usage
-----
    # Download fake songs from HuggingFace
    python scripts/download_sonics.py --download-fakes

    # Download real songs from S3 (requires aws-vault)
    aws-vault exec admin@innovation -- python scripts/download_sonics.py --download-reals-s3

    # Both
    aws-vault exec admin@innovation -- python scripts/download_sonics.py --download-fakes --download-reals-s3

    # Prepare metadata (always needed after downloads)
    python scripts/download_sonics.py --prepare
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SONICS_DIR = Path("data/raw/sonics")
FAKE_SONGS_DIR = SONICS_DIR / "fake_songs"
REAL_SONGS_DIR = SONICS_DIR / "real_songs"
METADATA_DIR = SONICS_DIR / "metadata"

S3_REAL_PREFIX = "s3://ai-plagiarism-data/ai-downloads"


def download_fakes_huggingface() -> None:
    """Download fake songs from HuggingFace.

    The HF repo stores fake songs as zip archives (part_01.zip to part_10.zip)
    under fake_songs/. CSVs are at root level.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    SONICS_DIR.mkdir(parents=True, exist_ok=True)
    hf_dir = SONICS_DIR / "hf_download"

    logger.info("Downloading SONICS dataset from HuggingFace (CSVs + zip archives)...")
    snapshot_download(
        repo_id="awsaf49/sonics",
        repo_type="dataset",
        local_dir=str(hf_dir),
    )

    # Copy CSVs to metadata dir
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for csv_name in ["real_songs.csv", "fake_songs.csv", "train.csv", "test.csv", "valid.csv"]:
        src = hf_dir / csv_name
        if src.exists():
            shutil.copy2(str(src), str(METADATA_DIR / csv_name))
            logger.info("Copied %s", csv_name)

    # Extract zip archives into fake_songs/
    FAKE_SONGS_DIR.mkdir(parents=True, exist_ok=True)
    zip_dir = hf_dir / "fake_songs"
    if zip_dir.exists():
        import zipfile

        for zip_path in sorted(zip_dir.glob("*.zip")):
            logger.info("Extracting %s...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(".mp3"):
                        fname = os.path.basename(member)
                        dest = FAKE_SONGS_DIR / fname
                        if not dest.exists():
                            with zf.open(member) as src_file, open(dest, "wb") as dst_file:
                                shutil.copyfileobj(src_file, dst_file)
            logger.info("Extracted %s", zip_path.name)

    logger.info("HuggingFace download complete")


def download_reals_from_s3(region: str = "eu-west-1", max_tracks: int | None = None) -> None:
    """Download real songs from S3 using youtube_id from real_songs.csv."""
    real_csv = METADATA_DIR / "real_songs.csv"
    if not real_csv.exists():
        logger.error("real_songs.csv not found at %s. Run --download-fakes first.", real_csv)
        sys.exit(1)

    df = pd.read_csv(real_csv)
    if "youtube_id" not in df.columns:
        logger.error("real_songs.csv does not have 'youtube_id' column")
        sys.exit(1)

    REAL_SONGS_DIR.mkdir(parents=True, exist_ok=True)

    # Try common audio extensions in S3
    extensions = [".m4a", ".mp3", ".wav", ".ogg"]
    downloaded = 0
    skipped = 0
    failed = 0

    youtube_ids = df["youtube_id"].dropna().unique()
    if max_tracks:
        youtube_ids = youtube_ids[:max_tracks]

    logger.info("Attempting to download %d real songs from S3...", len(youtube_ids))

    for yt_id in youtube_ids:
        yt_id = str(yt_id).strip()
        if not yt_id:
            continue

        # Check if already downloaded (any extension)
        existing = list(REAL_SONGS_DIR.glob(f"{yt_id}.*"))
        if existing:
            skipped += 1
            continue

        success = False
        for ext in extensions:
            s3_key = f"{S3_REAL_PREFIX}/{yt_id}{ext}"
            local_path = REAL_SONGS_DIR / f"{yt_id}{ext}"
            try:
                result = subprocess.run(
                    ["aws", "s3", "cp", s3_key, str(local_path), "--region", region],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    downloaded += 1
                    success = True
                    break
            except subprocess.TimeoutExpired:
                continue

        if not success:
            failed += 1
            if failed <= 5:
                logger.debug("Could not find %s in S3 with any extension", yt_id)

    logger.info("Download complete: %d downloaded, %d skipped (exist), %d not found", downloaded, skipped, failed)


def convert_m4a_to_mp3(directory: Path) -> None:
    """Convert any .m4a files to .mp3 using ffmpeg."""
    m4a_files = list(directory.glob("*.m4a"))
    if not m4a_files:
        logger.info("No .m4a files to convert in %s", directory)
        return

    logger.info("Converting %d .m4a files to .mp3...", len(m4a_files))
    converted = 0
    for f in m4a_files:
        mp3_path = f.with_suffix(".mp3")
        if mp3_path.exists():
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(f), "-q:a", "2", "-y", str(mp3_path)],
                capture_output=True,
                timeout=120,
                check=True,
            )
            converted += 1
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Failed to convert %s: %s", f.name, e)

    logger.info("Converted %d files to mp3", converted)


def prepare_metadata() -> None:
    """Create a unified metadata CSV for experiments."""
    fake_csv = METADATA_DIR / "fake_songs.csv"
    real_csv = METADATA_DIR / "real_songs.csv"

    if not fake_csv.exists() or not real_csv.exists():
        logger.error("Metadata CSVs not found. Run --download-fakes first.")
        sys.exit(1)

    df_fake = pd.read_csv(fake_csv)
    df_real = pd.read_csv(real_csv)

    # Check what real songs we actually have on disk
    real_files = set()
    if REAL_SONGS_DIR.exists():
        for f in REAL_SONGS_DIR.iterdir():
            if f.suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                real_files.add(f.stem)

    fake_files = set()
    if FAKE_SONGS_DIR.exists():
        for f in FAKE_SONGS_DIR.iterdir():
            if f.suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                fake_files.add(f.stem)

    # Map real songs to their files
    df_real["has_audio"] = df_real["youtube_id"].apply(lambda x: str(x) in real_files)
    df_real["audio_path"] = df_real["youtube_id"].apply(
        lambda x: str(next(REAL_SONGS_DIR.glob(f"{x}.*"), "")) if str(x) in real_files else ""
    )

    # Map fake songs to their files
    df_fake["has_audio"] = df_fake["filename"].apply(lambda x: Path(x).stem in fake_files if pd.notna(x) else False)
    df_fake["audio_path"] = df_fake["filename"].apply(
        lambda x: (
            str(next(FAKE_SONGS_DIR.glob(f"{Path(x).stem}.*"), ""))
            if pd.notna(x) and Path(x).stem in fake_files
            else ""
        )
    )

    # Summary
    n_real_with_audio = df_real["has_audio"].sum()
    n_fake_with_audio = df_fake["has_audio"].sum()

    print(f"\nSONICS Dataset Summary:")
    print(f"  Real songs: {len(df_real)} total, {n_real_with_audio} with audio")
    print(f"  Fake songs: {len(df_fake)} total, {n_fake_with_audio} with audio")

    if "split" in df_real.columns:
        print(f"\n  Real splits: {df_real[df_real['has_audio']]['split'].value_counts().to_dict()}")
    if "split" in df_fake.columns:
        print(f"  Fake splits: {df_fake[df_fake['has_audio']]['split'].value_counts().to_dict()}")
    if "source" in df_fake.columns:
        print(f"  Fake sources: {df_fake[df_fake['has_audio']]['source'].value_counts().to_dict()}")
    if "label" in df_fake.columns:
        print(f"  Fake labels: {df_fake[df_fake['has_audio']]['label'].value_counts().to_dict()}")

    # Save filtered experiment-ready CSVs
    df_real_ready = df_real[df_real["has_audio"]].copy()
    df_fake_ready = df_fake[df_fake["has_audio"]].copy()

    df_real_ready.to_csv(METADATA_DIR / "real_songs_ready.csv", index=False)
    df_fake_ready.to_csv(METADATA_DIR / "fake_songs_ready.csv", index=False)
    logger.info("Saved experiment-ready CSVs to %s", METADATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare SONICS dataset")
    parser.add_argument("--download-fakes", action="store_true", help="Download fake songs from HuggingFace")
    parser.add_argument("--download-reals-s3", action="store_true", help="Download real songs from S3")
    parser.add_argument("--convert-m4a", action="store_true", help="Convert .m4a files to .mp3")
    parser.add_argument("--prepare", action="store_true", help="Prepare metadata CSV")
    parser.add_argument("--max-tracks", type=int, default=None, help="Max real tracks to download from S3")
    parser.add_argument("--region", type=str, default="eu-west-1", help="AWS region")
    args = parser.parse_args()

    if not any([args.download_fakes, args.download_reals_s3, args.convert_m4a, args.prepare]):
        parser.print_help()
        return

    if args.download_fakes:
        download_fakes_huggingface()

    if args.download_reals_s3:
        download_reals_from_s3(region=args.region, max_tracks=args.max_tracks)

    if args.convert_m4a:
        convert_m4a_to_mp3(REAL_SONGS_DIR)

    if args.prepare:
        prepare_metadata()


if __name__ == "__main__":
    main()
