"""Extract embeddings from audio files in batch.

Reads audio files, computes embeddings (CLAP, MERT, or EnCodec),
and stores them as HDF5 for downstream ID estimation.

Usage
-----
    python -m scripts.extract_embeddings \
        --config configs/experiment.yaml \
        --embedding clap \
        --data-dir data/raw/plagiarism \
        --output-dir data/interim/embeddings
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from intrinsic_ai_music_detection.config import ExperimentConfig, load_config
from intrinsic_ai_music_detection.data.audio_utils import load_audio
from intrinsic_ai_music_detection.features.embeddings import get_extractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def find_audio_files(data_dir: Path) -> list[Path]:
    """Recursively find all audio files in a directory."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings from audio files")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config")
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["clap", "mert", "encodec"],
        required=True,
        help="Embedding model to use",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for HDF5 files")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = ExperimentConfig()

    # Discover audio files
    data_dir = Path(args.data_dir)
    audio_files = find_audio_files(data_dir)
    logger.info("Found %d audio files in %s", len(audio_files), data_dir)

    if not audio_files:
        logger.warning("No audio files found. Exiting.")
        return

    # Get the appropriate embedding config
    if args.embedding == "clap":
        emb_cfg = cfg.clap
        target_sr = 48000
    elif args.embedding == "mert":
        emb_cfg = cfg.mert
        target_sr = 24000
    elif args.embedding == "encodec":
        emb_cfg = cfg.encodec
        target_sr = 24000
    else:
        raise ValueError(f"Unknown embedding: {args.embedding}")

    # Initialise extractor
    logger.info("Loading %s model...", args.embedding)
    extractor = get_extractor(args.embedding, device=args.device)

    # Output HDF5 file
    out_path = output_dir / f"{args.embedding}_embeddings.h5"
    logger.info("Writing embeddings to %s", out_path)

    with h5py.File(out_path, "w") as hf:
        for audio_path in tqdm(audio_files, desc=f"Extracting {args.embedding}"):
            track_id = str(audio_path.relative_to(data_dir))
            try:
                audio, _ = load_audio(str(audio_path), target_sr=target_sr)
                embeddings = extractor.extract(audio, target_sr)
                hf.create_dataset(track_id, data=embeddings, compression="gzip")
            except Exception:
                logger.warning("Failed to process %s", audio_path, exc_info=True)
                continue

    logger.info("Done. Wrote %s", out_path)


if __name__ == "__main__":
    main()
