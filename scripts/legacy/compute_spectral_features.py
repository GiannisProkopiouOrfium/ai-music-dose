"""Compute low-level spectral-artifact features over the canonical corpus.

Produces a CSV keyed by ``track_id`` (one row per track) with ``spec_*`` columns
that merge directly into the ablation feature table
(``data/processed/multiembed_descriptors/ablation_features.csv``) so the
spectral bundle can be evaluated against / fused with the geometry bundle using
the same leak-free LOGO protocol as ``diagnose_descriptors.py``.

Audio source
------------
``--audio-source canonical`` (default): the 24 kHz / 64 kbps-MP3 / LUFS canonical
    WAVs — confound-matched to the embeddings, so spec+geometry fusion is fair.
    This is the *deployable* regime: features that survive content-preserving
    normalisation.
``--audio-source raw``: the original SONICS sources (full bandwidth, pre-MP3) —
    a diagnostic probe answering "do deconvolution-comb artifacts exist *before*
    our normalisation destroys them?". NOT confound-safe above ~8 kHz; treat as
    an upper bound on artifact availability, not a deployable detector.

Examples
--------
    # confound-matched (run on EC2 where canonical WAVs live)
    python scripts/compute_spectral_features.py \
        --manifest data/processed/canonical_sampler/canonical_manifest.csv \
        --audio-source canonical \
        --output data/processed/multiembed_descriptors/spectral_features.csv \
        --workers 6

    # artifact-availability probe on raw audio
    python scripts/compute_spectral_features.py \
        --manifest data/processed/canonical_sampler/canonical_manifest.csv \
        --audio-source raw \
        --output data/processed/multiembed_descriptors/spectral_features_raw.csv \
        --workers 6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.spectral_artifacts import (  # noqa: E402
    SpectralArtifactConfig,
    spectral_artifact_features_from_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compute_spectral_features")


def _process_one(job: dict) -> dict:
    """Top-level worker (picklable). Returns a feature row for one track."""
    cfg: SpectralArtifactConfig = job["cfg"]
    base = {
        "track_id": job["track_id"],
        "label": job["label"],
        "fake_label": job["fake_label"],
        "algorithm": job["algorithm"],
        "y": 1 if job["label"] == "fake" else 0,
    }
    try:
        feats = spectral_artifact_features_from_file(job["path"], cfg)
        if not feats:
            return {**base, "status": "error: empty features"}
        return {**base, **feats, "status": "ok"}
    except Exception as exc:  # pragma: no cover - per-track robustness
        return {**base, "status": f"error: {exc}"}


def _build_jobs(args: argparse.Namespace, cfg: SpectralArtifactConfig) -> list[dict]:
    manifest = pd.read_csv(args.manifest, low_memory=False)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"].astype(str) == "ok"]
    if "label" in manifest.columns and "fake_label" not in manifest.columns:
        manifest = manifest.rename(columns={"label": "fake_label_tmp"})

    # Build track_id -> path map for the requested source
    paths: dict[str, str] = {}
    if args.audio_source == "canonical":
        for _, r in manifest.iterrows():
            p = str(r.get("canonical_path", ""))
            if p and Path(p).exists():
                paths[str(r["track_id"])] = p
    else:  # raw
        from intrinsic_ai_music_detection.data.balanced_sampling import load_balanced_tracks

        real_tracks, fake_tracks = load_balanced_tracks(
            per_stratum=args.per_stratum, seed=args.seed, max_real=args.max_real
        )
        for t in [*real_tracks, *fake_tracks]:
            paths[str(t["track_id"])] = str(t["path"])

    jobs: list[dict] = []
    wanted = manifest["track_id"].astype(str).tolist() if "track_id" in manifest.columns else []
    meta = manifest.set_index(manifest["track_id"].astype(str)) if "track_id" in manifest.columns else manifest
    for tid in wanted:
        path = paths.get(tid)
        if not path:
            continue
        row = meta.loc[tid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        jobs.append(
            {
                "track_id": tid,
                "path": path,
                "label": str(row.get("label", "")),
                "algorithm": str(row.get("algorithm", "")),
                "fake_label": str(row.get("fake_label", row.get("fake_label_tmp", ""))),
                "cfg": cfg,
            }
        )
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute spectral-artifact (spec_*) features over the canonical corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default="data/processed/canonical_sampler/canonical_manifest.csv")
    parser.add_argument("--audio-source", choices=["canonical", "raw"], default="canonical")
    parser.add_argument("--output", default="data/processed/multiembed_descriptors/spectral_features.csv")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="Debug: process only the first N tracks.")
    # raw-source sampler params (must match the canonical build)
    parser.add_argument("--per-stratum", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-real", type=int, default=None)
    # feature config
    parser.add_argument("--analysis-sr", type=int, default=SpectralArtifactConfig.analysis_sr)
    parser.add_argument("--n-fft", type=int, default=SpectralArtifactConfig.n_fft)
    parser.add_argument("--band-lo-hz", type=float, default=SpectralArtifactConfig.band_lo_hz)
    parser.add_argument("--band-hi-hz", type=float, default=SpectralArtifactConfig.band_hi_hz)
    parser.add_argument("--max-duration", type=float, default=SpectralArtifactConfig.max_duration)
    args = parser.parse_args()

    cfg = SpectralArtifactConfig(
        analysis_sr=args.analysis_sr,
        n_fft=args.n_fft,
        hop_length=args.n_fft // 2,
        band_lo_hz=args.band_lo_hz,
        band_hi_hz=args.band_hi_hz,
        max_duration=args.max_duration,
    )

    jobs = _build_jobs(args, cfg)
    logger.info(
        "Spectral features: %d tracks (source=%s, band=%.0f-%.0f Hz, n_fft=%d)",
        len(jobs),
        args.audio_source,
        cfg.band_lo_hz,
        cfg.band_hi_hz,
        cfg.n_fft,
    )
    if not jobs:
        logger.error("No tracks to process — check --manifest / --audio-source paths.")
        sys.exit(1)

    rows: list[dict] = []
    t0 = time.time()
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, j): j["track_id"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            if not str(row.get("status", "")).startswith("ok"):
                n_err += 1
                if n_err <= 10:
                    logger.warning("  %s -> %s", row["track_id"], row.get("status"))
            if i % 200 == 0 or i == len(jobs):
                rate = i / max(time.time() - t0, 1e-6)
                logger.info("  %d/%d (%.1f tracks/s, %d errors)", i, len(jobs), rate, n_err)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    ok = int((df["status"].astype(str) == "ok").sum()) if "status" in df.columns else len(df)
    logger.info("Wrote %s — %d/%d ok, %d errors, %.1fs", out, ok, len(df), n_err, time.time() - t0)
    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    logger.info("spec_* columns (%d): %s", len(spec_cols), ", ".join(spec_cols))


if __name__ == "__main__":
    main()
