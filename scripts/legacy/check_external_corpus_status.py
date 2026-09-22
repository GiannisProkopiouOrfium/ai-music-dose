"""Audit exactly what FMA / FakeMusicCaps data and results actually exist on disk.

This repo has THREE different, partially-overlapping paths that touch FMA/FakeMusicCaps,
which have drifted out of sync with each other:

  1. download_external_data.sh   -> data/external/fma_sample_1000/, data/external/fakemusiccaps/
                                     (FakeMusicCaps from Zenodo — FAKE-ONLY, 5 generators,
                                      NO real MusicCaps counterpart in this mirror)
  2. run_external_corpus_eval.sh -> data/raw/fma_small/, data/raw/fakemusiccaps/
                                     (FakeMusicCaps from HuggingFace lucacoma/FakeMusicCaps —
                                      DOES include a real/ subdir, i.e. the true MusicCaps
                                      reference clips)
  3. score_external_corpus.py    -> scores whatever --audio-dir you point it at against the
                                     frozen SONICS flow; writes to data/processed/external_scores/*

Because of this drift, it's easy to end up with FMA/FMC partially downloaded under one path,
canonicalized under another, and scored results that quietly don't correspond to what you
think they do. This script reports, for every known location, whether it exists, how many
files/rows it has, and whether the *real* MusicCaps counterpart is present (needed for a
true apples-to-apples comparison against MusicDET's own FakeMusicCaps numbers).

Usage:
    python scripts/check_external_corpus_status.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent


def _count_audio(d: Path) -> int:
    if not d.exists():
        return -1
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in exts)


def _report_dir(label: str, path: str) -> None:
    p = REPO / path
    n = _count_audio(p)
    if n < 0:
        logger.info("  [MISSING]  %-45s %s", label, p)
    else:
        logger.info("  [n=%6d]  %-45s %s", n, label, p)


def _report_manifest(label: str, path: str) -> None:
    p = REPO / path
    if not p.exists():
        logger.info("  [MISSING]  %-45s %s", label, p)
        return
    try:
        df = pd.read_csv(p, low_memory=False)
        ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
        n_real = int((ok.get("label") == "real").sum()) if "label" in ok.columns else -1
        n_fake = int((ok.get("label") == "fake").sum()) if "label" in ok.columns else -1
        logger.info("  [rows=%6d ok=%6d real=%6d fake=%6d]  %-30s %s", len(df), len(ok), n_real, n_fake, label, p)
        if "algorithm" in ok.columns and n_fake > 0:
            for alg, n in ok.loc[ok["label"] == "fake", "algorithm"].value_counts().items():
                logger.info("       %-30s %6d", alg, n)
    except Exception as exc:
        logger.warning("  [ERROR reading %s]  %s: %s", label, p, exc)


def main() -> None:
    logger.info("=" * 78)
    logger.info("RAW DOWNLOADS")
    logger.info("=" * 78)
    logger.info("-- FMA (path A: download_external_data.sh) --")
    _report_dir("data/external/fma_small", "data/external/fma_small")
    _report_dir("data/external/fma_sample_1000", "data/external/fma_sample_1000")
    logger.info("-- FMA (path B: run_external_corpus_eval.sh) --")
    _report_dir("data/raw/fma_small", "data/raw/fma_small")
    logger.info("-- FakeMusicCaps (path A: Zenodo zip, FAKE-ONLY) --")
    _report_dir("data/external/fakemusiccaps", "data/external/fakemusiccaps")
    fmc_a = REPO / "data/external/fakemusiccaps"
    if fmc_a.exists():
        gens = sorted(d.name for d in fmc_a.iterdir() if d.is_dir())
        logger.info("     generator subdirs: %s", gens)
        logger.info("     has real/ counterpart: %s", "real" in gens)
    logger.info("-- FakeMusicCaps (path B: HuggingFace lucacoma/FakeMusicCaps, has real/) --")
    _report_dir("data/raw/fakemusiccaps", "data/raw/fakemusiccaps")
    fmc_b = REPO / "data/raw/fakemusiccaps"
    if fmc_b.exists():
        gens = sorted(d.name for d in fmc_b.iterdir() if d.is_dir())
        logger.info("     generator subdirs: %s", gens)
        logger.info("     has real/ (true MusicCaps) counterpart: %s", "real" in gens)

    logger.info("")
    logger.info("=" * 78)
    logger.info("CANONICALIZED MANIFESTS")
    logger.info("=" * 78)
    _report_manifest("canonical_fma", "data/processed/canonical_fma/canonical_manifest.csv")
    _report_manifest(
        "canonical_fmc (merged, if built per-generator)", "data/processed/canonical_fmc/canonical_manifest.csv"
    )
    for gen in ["musicgen", "musicldm", "audioldm2", "mustango", "stable_audio_open", "real"]:
        _report_manifest(f"canonical_fmc/{gen}", f"data/processed/canonical_fmc/{gen}/canonical_manifest.csv")
    _report_manifest("combined_external_manifest", "data/processed/combined_external_manifest.csv")

    logger.info("")
    logger.info("=" * 78)
    logger.info("SCORED RESULTS")
    logger.info("=" * 78)
    _report_manifest("external_scores/fma_fpr", "data/processed/external_scores/fma_fpr/external_scores.csv")
    _report_manifest(
        "external_scores/fakemusiccaps", "data/processed/external_scores/fakemusiccaps/external_scores.csv"
    )
    _report_manifest(
        "external_eval/window_flow_eval.csv (run_external_corpus_eval.sh)",
        "data/processed/external_eval/window_flow_eval.csv",
    )
    _report_manifest("external_eval/per_generator_auc.csv", "data/processed/external_eval/per_generator_auc.csv")
    _report_manifest("external_eval/fpr_fma.csv", "data/processed/external_eval/fpr_fma.csv")

    frozen_flow = REPO / "data/processed/full_sonics_all/sonics_real_flow_encodec.pt"
    logger.info("")
    logger.info("=" * 78)
    logger.info("FROZEN FLOW CHECKPOINT (required by score_external_corpus.py)")
    logger.info("=" * 78)
    logger.info("  %-45s exists=%s", str(frozen_flow), frozen_flow.exists())
    if not frozen_flow.exists():
        logger.warning(
            "  Not found. score_external_corpus.py will silently RE-TRAIN a flow from "
            "data/emb_cache_encodec_full and save it here (~8h on T4) the first time it's run. "
            "If a headline flow checkpoint already exists elsewhere (e.g. from "
            "run_balanced_ablation.py --wf-save-flow-path), copy/symlink it here first to avoid "
            "an unnecessary multi-hour retrain."
        )

    logger.info("")
    logger.info("=" * 78)
    logger.info("VERDICT")
    logger.info("=" * 78)
    logger.info(
        "For a fair, MusicDET-comparable FakeMusicCaps result you need the HuggingFace mirror "
        "(path B, includes real/) — the Zenodo mirror (path A) is fake-only and cannot support "
        "a real-vs-fake AUC entirely within FakeMusicCaps' own domain. "
        "Recommended clean path forward: score_external_corpus.py against the FROZEN flow "
        "(never retrains it on external data), NOT run_external_corpus_eval.py's combined-manifest "
        "retrain (which leaks FMA/FMC real tracks into flow TRAINING via its K-fold split and also "
        "uses 2s/1s windows instead of the canonical 4s/2s protocol)."
    )


if __name__ == "__main__":
    main()
