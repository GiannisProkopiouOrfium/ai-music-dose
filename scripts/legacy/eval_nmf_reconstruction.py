"""RETRACTED (2026-08-20) — superseded by scripts/eval_nmf_peak_detector.py.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ THE RESULT THIS SCRIPT PRODUCED IS WITHDRAWN.                        │
    │                                                                      │
    │   reported : "NMF reconstruction error INVERTS"                      │
    │              FakeMusicCaps 0.4998, SONICS-chain 0.2564               │
    │   status   : RETRACTED — misimplementation, not a finding            │
    │   replaced : scripts/eval_nmf_peak_detector.py                       │
    └──────────────────────────────────────────────────────────────────────┘

Two independent defects, both measured (see tests/test_nmf_peak.py):

1. **Wrong quantity and wrong fit set.** Afchar & Hennequin (arXiv:2607.25530)
   do not score a reconstruction error against the input, and do not fit the
   basis on reals. They factorise the POOLED, UNLABELLED matrix and score
   ``r_i = ‖H_i W − H_i (W∗G)‖₂`` — the difference between the reconstruction
   through sharp atoms and through Gaussian-BLURRED atoms, i.e. how much PEAK
   STRUCTURE the sample carries. This script computes ``‖x_i − H_i W‖`` with
   ``W`` fitted on reals only, which is a novelty score. On a symmetric fixture
   that novelty score is **inverted in 32/32 capacity cells (AUC 0.28–0.44)**
   while the paper's is correct in 8/8 seeds (0.94–1.00): a k-atom basis
   reconstructs a clean periodic comb *well* and irregular micro-texture
   *badly*, so the comb gets the lower score.

2. **Degraded input descriptor.** It consumed ``CombPrintExtractor`` profiles
   (median-filter residual, resampled to 445 bins). That resample alone costs
   AUC 0.904 → 0.665 on synthetic spectra with realistic 2-bin-wide teeth.

The module docstring below still states the original — and wrong — hypothesis
("a decoder comb is a pattern that basis does not contain, so it reconstructs
badly"). It is left visible rather than edited, because the retraction is part
of the record. Do not cite anything from this file.

Kept runnable so the retracted number can be reproduced on demand.

---

One-class detection by NMF reconstruction error on fakeprint profiles.

The last untried method in the bibliography, and the one designed for exactly the
failure we keep hitting
-----------------------------------------------------------------------------
Afchar & Hennequin, "Finding the noise: Zero-shot AI Music Detection"
(arXiv:2607.25530) fit a small NMF basis (~20 atoms) to fakeprints and score by
**reconstruction error**. We have implemented every other idea in that literature
and not this one.

Why it should work where the flow does not. A normalizing flow fitted to real
music assigns *higher* likelihood to generated music, consistently, in every
representation we have tried — EnCodec latents, spectrograms, MusicDET's front
end, and now the comb profile itself (SONICS macro 0.242, i.e. inverted so hard
that flipping the sign gives 0.758). The mechanism is settled: generated audio is
smoother and more regular than real audio, so it sits *nearer the mode* of a
real-music density. A likelihood cannot separate "typical" from "real" and no
amount of flow capacity changes that sign.

Reconstruction error inverts the question. A basis fitted to real profiles spans
the patterns real music produces; a decoder comb is a pattern that basis does not
contain, so it reconstructs *badly*. High error means "not expressible as real
music", which is the orientation we want and which a density model does not give.
Both are one-class and label-free; only one is correctly oriented.

Protocol
--------
Reals are split disjointly: one half fits the basis, the other is scored, so no
scored real contributed an atom. Fakes are only scored. Nothing about any
generator is used at any point.

Usage (EC2)
-----------
    python scripts/eval_nmf_reconstruction.py \
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \
        --per-stratum 600 --workers 6 --max-duration 9 --level-match \
        --n-atoms 20 --out-dir reports/diagnostics/nmf_fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_nmf_reconstruction")


def _profile(job: dict) -> dict:
    """Mean fakeprint profile for one track."""
    import librosa

    from intrinsic_ai_music_detection.config import CombPrintConfig
    from intrinsic_ai_music_detection.data.audio_preprocessing import level_match
    from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

    row = dict(job["meta"])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            audio, sr = librosa.load(
                job["path"],
                sr=None,
                mono=True,
                duration=job["max_duration"],
                res_type="soxr_hq",
            )
        audio = np.asarray(audio, dtype=np.float32)
        if job["level_match"]:
            audio = level_match(audio, sr)

        cfg = CombPrintConfig(profile_kind="hull", hull_area=job["hull_area"], n_bins=job["n_bins"])
        feats = CombPrintExtractor(cfg).extract(audio, sr)
        # Average the per-span profiles: the comb is stationary (it is a property
        # of the decoder), so averaging suppresses everything that is not.
        row["profile"] = feats.mean(axis=0).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:200]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-class NMF reconstruction error on fakeprint profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-column", default="canonical_path")
    parser.add_argument("--per-stratum", type=int, default=600)
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument("--hull-area", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=445)
    parser.add_argument(
        "--n-atoms",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help="NMF components. Their paper uses 20. Too many and the basis can "
        "express a comb too, which removes the very error we are measuring.",
    )
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Score a DIFFERENT corpus with the basis fitted here — the cross-corpus "
        "test, and the strongest form of the zero-shot claim.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_rows(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        if "status" in df.columns:
            df = df[df["status"].astype(str) == "ok"]
        df["label"] = df["label"].astype(str)
        df["algorithm"] = df.get("algorithm", pd.Series([""] * len(df))).fillna("").astype(str)
        rng = np.random.default_rng(args.seed)
        parts = []
        for _, grp in df.groupby(["label", "algorithm"], sort=True):
            n = min(args.per_stratum, len(grp))
            parts.append(grp.iloc[np.sort(rng.choice(len(grp), size=n, replace=False))])
        return pd.concat(parts, ignore_index=True)

    def profiles_for(df: pd.DataFrame) -> pd.DataFrame:
        meta_cols = [c for c in ("track_id", "label", "algorithm") if c in df.columns]
        jobs = [
            {
                "path": str(getattr(r, args.audio_column)),
                "meta": {c: getattr(r, c) for c in meta_cols},
                "max_duration": args.max_duration,
                "level_match": args.level_match,
                "hull_area": args.hull_area,
                "n_bins": args.n_bins,
            }
            for r in df.itertuples(index=False)
            if Path(str(getattr(r, args.audio_column))).exists()
        ]
        if not jobs:
            raise SystemExit(f"no files resolved from column {args.audio_column!r}")
        rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_profile, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 300 == 0 or i == len(futures):
                    logger.info("  profiles %d/%d", i, len(futures))
        out = pd.DataFrame(rows)
        n_err = int(out.get("error", pd.Series(dtype=object)).notna().sum())
        if n_err:
            logger.warning("%d/%d tracks failed to profile", n_err, len(out))
        out = out[out.get("profile").notna()] if "profile" in out.columns else out
        if out.empty:
            raise SystemExit("no profiles produced — refusing to continue")
        return out

    fit_df = profiles_for(load_rows(args.manifest))
    eval_df = profiles_for(load_rows(args.eval_manifest)) if args.eval_manifest else fit_df

    # Split reals disjointly: half fit the basis, half are scored. A real that
    # contributed an atom would be reconstructed too well and its score would be
    # optimistic — the same leak as scoring a track with a flow that trained on it.
    rng = np.random.default_rng(args.seed)
    fit_reals = fit_df[fit_df["label"] == "real"]
    if len(fit_reals) < 40:
        raise SystemExit(f"only {len(fit_reals)} real profiles; need >= 40 to fit a basis")
    perm = rng.permutation(len(fit_reals))
    basis_ids = set(fit_reals.iloc[np.sort(perm[: len(perm) // 2])]["track_id"].astype(str))

    x_basis = np.stack(fit_reals[fit_reals["track_id"].astype(str).isin(basis_ids)]["profile"].to_numpy())
    same_corpus = args.eval_manifest is None
    scored = (
        eval_df[~((eval_df["label"] == "real") & (eval_df["track_id"].astype(str).isin(basis_ids)))]
        if same_corpus
        else eval_df
    )
    logger.info(
        "basis fitted on %d real profiles; scoring %d tracks (%s corpus)",
        len(x_basis),
        len(scored),
        "same" if same_corpus else "cross",
    )

    x_eval = np.stack(scored["profile"].to_numpy())
    is_real = (scored["label"] == "real").to_numpy()
    algos = scored["algorithm"].to_numpy()

    from sklearn.decomposition import NMF

    report: list[dict] = []
    per_track = scored[[c for c in ("track_id", "label", "algorithm") if c in scored.columns]].copy()

    for k in args.n_atoms:
        # NMF needs non-negative input; hull_residual is already in [0, 1].
        model = NMF(n_components=k, init="nndsvda", max_iter=600, random_state=args.seed)
        model.fit(np.maximum(x_basis, 0.0))
        w = model.transform(np.maximum(x_eval, 0.0))
        err = np.linalg.norm(x_eval - w @ model.components_, axis=1)
        col = f"nmf_err_k{k}"
        per_track[col] = err

        real_s = err[is_real]
        real_s = real_s[np.isfinite(real_s)]
        aucs = []
        for alg in sorted(set(algos[~is_real]) - {""}):
            fake_s = err[(~is_real) & (algos == alg)]
            fake_s = fake_s[np.isfinite(fake_s)]
            if len(fake_s) < 5 or len(real_s) < 5:
                continue
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
            report.append(
                {
                    "n_atoms": k,
                    "algorithm": alg,
                    "auc": round(auc, 4),
                    "eer_pct": round(eer * 100, 2),
                    "n_real": len(real_s),
                    "n_fake": len(fake_s),
                }
            )
            aucs.append(auc)
        if aucs:
            report.append(
                {
                    "n_atoms": k,
                    "algorithm": "MACRO",
                    "auc": round(float(np.mean(aucs)), 4),
                    "eer_pct": float("nan"),
                    "n_real": len(real_s),
                    "n_fake": -1,
                }
            )

    if not report:
        raise SystemExit("no comparable real/fake pairs — check the manifest")

    rep = pd.DataFrame(report)
    tag = "_cross" if args.eval_manifest else ""
    rep.to_csv(out_dir / f"nmf_auc{tag}.csv", index=False)
    per_track.to_csv(out_dir / f"nmf_per_track{tag}.csv", index=False)

    pivot = rep.pivot_table(index="n_atoms", columns="algorithm", values="auc")
    logger.info("\nNMF RECONSTRUCTION-ERROR AUC by atom count:\n%s", pivot.to_string())
    logger.info(
        "\nORIENTATION: higher error = harder to express in the real-music basis = more "
        "likely generated. Unlike a likelihood, this is correctly oriented BY CONSTRUCTION "
        "— which matters because the one-class flow inverts in every representation we have "
        "tried (comb profiles included: SONICS macro 0.242)."
    )
    logger.info(
        "Both the basis and the scoring use REAL MUSIC ONLY. No generator output and no fake "
        "label at any stage, so this is a zero-shot detector in the strict sense."
    )


if __name__ == "__main__":
    main()
