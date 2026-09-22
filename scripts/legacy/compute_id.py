"""Compute intrinsic dimension from pre-extracted embeddings.

Reads HDF5 embedding files produced by extract_embeddings.py,
runs ID estimators (PHD, TwoNN, MLE), and saves results as CSV.

Usage
-----
    python -m scripts.compute_id \
        --embeddings data/interim/embeddings/clap_embeddings.h5 \
        --labels data/processed/labels.csv \
        --output data/processed/id_results.csv \
        --methods phd twonn mle
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from intrinsic_ai_music_detection.config import PHDConfig
from intrinsic_ai_music_detection.features.id_estimators import EstimatorName, estimate_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute intrinsic dimension from embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to HDF5 embeddings file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["phd", "twonn", "mle"],
        help="ID estimation methods to run",
    )
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric for PHD")
    parser.add_argument("--alpha", type=float, default=1.0, help="PHD alpha parameter")
    parser.add_argument("--n-reruns", type=int, default=3, help="PHD number of reruns")
    args = parser.parse_args()

    phd_cfg = PHDConfig(alpha=args.alpha, n_reruns=args.n_reruns)

    emb_path = Path(args.embeddings)
    logger.info("Reading embeddings from %s", emb_path)

    rows: list[dict] = []

    with h5py.File(emb_path, "r") as hf:
        track_ids = list(hf.keys())
        logger.info("Found %d tracks", len(track_ids))

        for track_id in track_ids:
            embeddings = np.array(hf[track_id], dtype=np.float32)
            logger.debug("Track %s: shape %s", track_id, embeddings.shape)

            row: dict = {"track_id": track_id, "n_embeddings": embeddings.shape[0]}

            for method in args.methods:
                id_val = estimate_id(
                    embeddings,
                    method=method,  # type: ignore[arg-type]
                    metric=args.metric,
                    phd_cfg=phd_cfg,
                )
                row[f"id_{method}"] = id_val
                logger.info("%-40s | %s = %.3f", track_id, method, id_val)

            rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s (%d tracks)", output_path, len(df))


if __name__ == "__main__":
    main()
