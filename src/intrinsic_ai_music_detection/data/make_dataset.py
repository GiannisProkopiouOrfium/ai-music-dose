"""Dataset loaders for the AI Audio Plagiarism corpus, YouTube AI covers, and SONICS."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from intrinsic_ai_music_detection.config import DataConfig, S3Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrackPair:
    """A matched pair of real and AI-generated audio."""

    track_id: str
    real_path: Path
    ai_path: Path
    source_model: str  # e.g. "musicgen", "mgeldm", "vevo2", "bark", "acestep"
    stem_type: str  # "instrumental", "vocal", "full_mix"
    tier: str = ""  # Gold / Silver / Bronze (for YouTube AI covers)
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _get_s3_client(region: str = "eu-west-1"):  # type: ignore[no-untyped-def]
    """Lazy-import boto3 and return an S3 client."""
    import boto3

    return boto3.client("s3", region_name=region)


def download_from_s3(
    bucket: str,
    key: str,
    local_path: Path,
    region: str = "eu-west-1",
) -> Path:
    """Download an S3 object to *local_path* if it does not already exist."""
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
    client = _get_s3_client(region)
    client.download_file(bucket, key, str(local_path))
    return local_path


def read_s3_csv(bucket: str, key: str, region: str = "eu-west-1") -> pd.DataFrame:
    """Read a CSV directly from S3 into a DataFrame."""
    client = _get_s3_client(region)
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ---------------------------------------------------------------------------
# AI Audio Plagiarism Dataset
# ---------------------------------------------------------------------------


def load_plagiarism_registry(
    s3_cfg: S3Config | None = None,
    local_csv: Path | None = None,
) -> pd.DataFrame:
    """Load the master CSV ledger from S3 or local disk."""
    if local_csv and local_csv.exists():
        return pd.read_csv(local_csv)

    if s3_cfg is None:
        s3_cfg = S3Config()
    return read_s3_csv(s3_cfg.bucket, s3_cfg.registry_key, s3_cfg.region)


def build_plagiarism_pairs(
    registry: pd.DataFrame,
    s3_cfg: S3Config | None = None,
    cache_dir: Path | None = None,
) -> list[TrackPair]:
    """Build matched TrackPair list from the plagiarism registry.

    The registry has columns: original_video_id, ai_generated_s3_path, lyrics.
    """
    if s3_cfg is None:
        s3_cfg = S3Config()
    if cache_dir is None:
        cache_dir = Path("data/raw")

    pairs: list[TrackPair] = []
    prefix = s3_cfg.prefix

    stem_map = {
        "instrumental": {
            "original_key": f"{prefix}/inputs/no_vocals",
            "ai_keys": {
                "musicgen": f"{prefix}/outputs/musicgen",
                "mgeldm": f"{prefix}/outputs/mgeldm",
            },
        },
        "full_mix": {
            "ai_keys": {
                "vevo2": f"{prefix}/outputs/mixed_vevo2",
                "bark": f"{prefix}/outputs/mixed_bark",
                "acestep": f"{prefix}/outputs/mixed_acestep",
            },
        },
    }

    for _, row in registry.iterrows():
        vid_id = str(row["original_video_id"])

        # Instrumental pairs
        inst_cfg = stem_map["instrumental"]
        orig_inst_key = f"{inst_cfg['original_key']}/{vid_id}.wav"
        orig_inst_local = cache_dir / "inputs" / "no_vocals" / f"{vid_id}.wav"

        for model_name, ai_prefix_path in inst_cfg["ai_keys"].items():
            ai_key = f"{ai_prefix_path}/{vid_id}.wav"
            ai_local = cache_dir / "outputs" / model_name / f"{vid_id}.wav"
            pairs.append(
                TrackPair(
                    track_id=vid_id,
                    real_path=orig_inst_local,
                    ai_path=ai_local,
                    source_model=model_name,
                    stem_type="instrumental",
                    metadata={"s3_real_key": orig_inst_key, "s3_ai_key": ai_key},
                )
            )

        # Full mix pairs
        mix_cfg = stem_map["full_mix"]
        for model_name, ai_prefix_path in mix_cfg["ai_keys"].items():
            ai_key = f"{ai_prefix_path}/{vid_id}.wav"
            ai_local = cache_dir / "outputs" / model_name / f"{vid_id}.wav"
            pairs.append(
                TrackPair(
                    track_id=vid_id,
                    real_path=orig_inst_local,
                    ai_path=ai_local,
                    source_model=model_name,
                    stem_type="full_mix",
                    metadata={"s3_ai_key": ai_key},
                )
            )

    logger.info("Built %d track pairs from plagiarism registry", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# YouTube AI Covers dataset
# ---------------------------------------------------------------------------


def load_youtube_ai_covers(
    csv_path: Path,
    audio_dir: Path,
    min_tier: str = "Bronze",
) -> list[TrackPair]:
    """Load YouTube AI cover pairs from a pre-exported CSV.

    Expected columns: video_id, tier, precision_score, ai_audio_path, original_audio_path
    """
    tier_order = {"Gold": 0, "Silver": 1, "Bronze": 2}
    min_rank = tier_order.get(min_tier, 2)

    df = pd.read_csv(csv_path)
    pairs: list[TrackPair] = []

    for _, row in df.iterrows():
        tier = str(row.get("tier", "Bronze"))
        if tier_order.get(tier, 3) > min_rank:
            continue

        pairs.append(
            TrackPair(
                track_id=str(row["video_id"]),
                real_path=audio_dir / str(row["original_audio_path"]),
                ai_path=audio_dir / str(row["ai_audio_path"]),
                source_model="youtube_ai_cover",
                stem_type="full_mix",
                tier=tier,
            )
        )

    logger.info("Loaded %d YouTube AI cover pairs (min_tier=%s)", len(pairs), min_tier)
    return pairs


# ---------------------------------------------------------------------------
# SONICS dataset
# ---------------------------------------------------------------------------


def load_sonics_dataset(
    sonics_dir: Path,
    split_file: Path | None = None,
    real_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """Load SONICS dataset file lists.

    Returns a dict with keys like 'train', 'test', 'valid',
    'suno_v3.5', 'udio_v120', etc. mapped to lists of file paths.
    """
    import numpy as np

    result: dict[str, list[Path]] = {}

    if split_file and split_file.exists():
        splits = np.load(split_file, allow_pickle=True).item()
        for key, filenames in splits.items():
            result[key] = [sonics_dir / fn for fn in filenames]

    if real_dir and real_dir.exists():
        result["real"] = sorted(real_dir.glob("*.mp3")) + sorted(real_dir.glob("*.wav"))

    logger.info("SONICS dataset: %s", {k: len(v) for k, v in result.items()})
    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def ensure_audio_available(
    pair: TrackPair,
    s3_cfg: S3Config | None = None,
) -> bool:
    """Download audio files for a pair if needed. Returns True if both exist."""
    if s3_cfg is None:
        s3_cfg = S3Config()

    for attr, meta_key in [("real_path", "s3_real_key"), ("ai_path", "s3_ai_key")]:
        local = getattr(pair, attr)
        if local.exists():
            continue
        meta = pair.metadata or {}
        s3_key = meta.get(meta_key)
        if s3_key:
            try:
                download_from_s3(s3_cfg.bucket, s3_key, local, s3_cfg.region)
            except Exception:
                logger.warning("Failed to download %s for %s", s3_key, pair.track_id)
                return False
        else:
            return False
    return True
