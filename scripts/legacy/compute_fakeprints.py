"""Compute fakeprint features from audio files.

Processes audio files and computes the Fourier fakeprint spectral
residual features, saving results as a NumPy archive.

Usage
-----
    python -m scripts.compute_fakeprints \
        --data-dir data/raw/plagiarism \
        --output data/processed/fakeprints.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from intrinsic_ai_music_detection.config import FakeprintConfig
from intrinsic_ai_music_detection.features.fakeprints import compute_fakeprint_from_file

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
    parser = argparse.ArgumentParser(description="Compute fakeprint features from audio")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--n-fft", type=int, default=16384, help="FFT size")
    parser.add_argument("--f-min", type=int, default=5000, help="Minimum frequency")
    parser.add_argument("--f-max", type=int, default=16000, help="Maximum frequency")
    args = parser.parse_args()

    cfg = FakeprintConfig(n_fft=args.n_fft, f_min=args.f_min, f_max=args.f_max)

    data_dir = Path(args.data_dir)
    audio_files = find_audio_files(data_dir)
    logger.info("Found %d audio files in %s", len(audio_files), data_dir)

    if not audio_files:
        logger.warning("No audio files found. Exiting.")
        return

    track_ids: list[str] = []
    features: list[np.ndarray] = []

    for audio_path in tqdm(audio_files, desc="Computing fakeprints"):
        track_id = str(audio_path.relative_to(data_dir))
        try:
            fp = compute_fakeprint_from_file(audio_path, cfg)
            track_ids.append(track_id)
            features.append(fp)
        except Exception:
            logger.warning("Failed to process %s", audio_path, exc_info=True)
            continue

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pad to uniform length for array stacking
    max_len = max(len(f) for f in features) if features else 0
    padded = np.zeros((len(features), max_len))
    for i, f in enumerate(features):
        padded[i, : len(f)] = f

    np.savez(output_path, track_ids=np.array(track_ids), features=padded)
    logger.info("Saved fakeprints to %s (%d tracks, %d features)", output_path, len(track_ids), max_len)


if __name__ == "__main__":
    main()
