"""Intrinsic dimension as a detector (GPTID, arXiv:2306.04723), under channel control.

Why this arm exists, and why it is not a repeat of Phase 1
----------------------------------------------------------
Tulchinskii et al.'s claim is exactly this project's central mechanism, arrived at
independently: **AI-generated content has LOWER intrinsic dimension than human
content** (human text ~9, generated ~1.5 lower). Their detector is a threshold on
the Persistent Homology Dimension of one sample's point cloud.

That matters here because PHD is a **topological statistic, not a density**. The
inversion that kills every one-class likelihood in this project comes from a
likelihood asking "is this typical of real music?", to which generated music
answers "yes, very". PHD does not ask that. It measures the manifold's degrees of
freedom directly, and its sign is fixed a priori by the hypothesis: *fake < real*.
So it is the one formulation in the bibliography that is simultaneously
training-free, label-free, and correctly oriented by construction.

Phase 1 of this project already measured TwoNN and PHD on MERT and EnCodec and
got AUC 0.608–0.684. Three things differ now, and all three are why this is a
pilot rather than a claim:

  1. Those runs predate the channel-control protocol. They were measured on
     uncontrolled corpora, where a two-line energy ratio scores AUC 1.0000 on
     SONICS — so 0.6–0.68 may have been channel, not content. Every number here
     goes through scripts/run_confound_gate.py before it is reported.
  2. They used music-SSL and codec embeddings — the same family that produced the
     corpus-identity confound. This script's primary cloud is the **comb-residual
     profile matrix**, where the artifact provably lives.
  3. They pooled whole-track clouds. GPTID's protocol is per-sample: one cloud per
     item, built from that item's own frame embeddings.

Honest prior: this may land at chance. If it does, it is a *correctly oriented*
negative, which is more informative than the six mis-oriented ones, and it costs
about an hour of CPU.

GPTID's published constants, used verbatim
------------------------------------------
alpha = 1.0 · minimal cloud n̂ = 40 · k = 8 subset sizes uniform in [n̂, n] ·
J = 7 resamples per size · 3 independent runs averaged · d = 1/(1 − κ).

Usage (EC2)
-----------
    python scripts/eval_phd_detector.py \\
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \\
        --per-stratum 600 --workers 6 --max-duration 9 --level-match \\
        --cloud combprint --out-dir reports/diagnostics/phd_fmc
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
logger = logging.getLogger("eval_phd")

# The floor is a property of the PHD implementation, not of this script:
# features/phd.py refuses any cloud below its own MINIMAL_CLOUD and returns NaN.
# Hard-coding 40 here meant clouds of 53 points passed OUR check, then silently
# produced NaN inside PHD — which is why the first combprint run reported only
# `id_twonn` and no `id_phd` at all, with no error. Import the real value.
from intrinsic_ai_music_detection.features.phd import MINIMAL_CLOUD as PHD_MINIMAL_CLOUD  # noqa: E402

MINIMAL_CLOUD = PHD_MINIMAL_CLOUD


def _point_cloud(job: dict) -> dict:
    """One track's point cloud → PHD and TwoNN.

    The cloud is the track's own frame/span matrix, GPTID-style: one point per
    span, never pooled across tracks.
    """
    import librosa

    from intrinsic_ai_music_detection.config import PHDConfig
    from intrinsic_ai_music_detection.data.audio_preprocessing import level_match

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

        cloud, metric = _build_cloud(audio, sr, job)
        row["n_points"] = int(len(cloud))
        if len(cloud) < MINIMAL_CLOUD:
            row["error"] = (
                f"cloud too small for PHD: {len(cloud)} points < {MINIMAL_CLOUD} "
                f"(need a longer --max-duration or a shorter --span-hop)"
            )
            return row

        from intrinsic_ai_music_detection.features.id_estimators import estimate_phd, estimate_twonn

        cfg = PHDConfig(
            alpha=1.0,
            n_reruns=job["n_reruns"],
            n_points=7,
            n_points_min=3,
            min_subsample=MINIMAL_CLOUD,
            intermediate_points=8,
        )
        row["id_phd"] = float(estimate_phd(cloud.astype(np.float32), metric=metric, cfg=cfg))
        row["id_twonn"] = float(estimate_twonn(cloud.astype(np.float32)))
        if not np.isfinite(row["id_phd"]):
            # Never let a NaN estimator silently vanish from the report table.
            row["error"] = f"PHD returned NaN for a {len(cloud)}-point cloud (floor {MINIMAL_CLOUD})"
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:200]
    return row


def _build_cloud(audio: np.ndarray, sr: int, job: dict) -> tuple[np.ndarray, str]:
    """Return ``([N, D] cloud, distance metric)`` for the requested representation."""
    import librosa

    kind = job["cloud"]
    if kind == "combprint":
        from intrinsic_ai_music_detection.config import CombPrintConfig
        from intrinsic_ai_music_detection.features.embeddings import CombPrintExtractor

        cfg0 = CombPrintConfig()
        # A span shorter than n_fft produces NO profile at all — the extractor
        # breaks out of its loop and returns nothing, which surfaced as
        # "no comb profiles produced" for every one of 1,800 tracks. The floor is
        # a property of the front end, so derive it rather than trusting the flag.
        min_span = cfg0.n_fft / cfg0.sample_rate
        span = max(job["span_duration"], min_span * 1.05)
        cfg = CombPrintConfig(span_duration=span, span_hop=job["span_hop"])
        if sr != cfg.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.sample_rate, res_type="soxr_hq")
            sr = cfg.sample_rate
        return np.asarray(CombPrintExtractor(cfg).extract(audio, sr), dtype=np.float64), "euclidean"

    from intrinsic_ai_music_detection.features.frontend import resolve_frontend

    fe = resolve_frontend(embedding=kind)
    # Every neural front end asserts its own rate (EnCodec refuses anything but
    # 24 kHz), and the canonical corpora are 16 kHz. Resample here rather than
    # letting 1,800 tracks fail one by one.
    if fe.sample_rate and sr != fe.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=fe.sample_rate, res_type="soxr_hq")
        sr = fe.sample_rate
    emb = fe.extract(audio, sr)
    # EnCodec latents are euclidean by convention here; SSL embeddings are cosine,
    # matching how each was treated in Phase 1 so the comparison is like-for-like.
    metric = "euclidean" if kind in ("encodec", "spec", "logmel", "spec-musicdet") else "cosine"
    return np.asarray(emb, dtype=np.float64), metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-column", default="canonical_path")
    parser.add_argument("--per-stratum", type=int, default=600)
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument(
        "--cloud",
        default="combprint",
        help=(
            "Representation the point cloud is built from. 'combprint' is the artifact space "
            "(the new arm); 'encodec' reproduces Phase 1's setting under the current controls, "
            "which is how we tell whether the old 0.608-0.684 was channel."
        ),
    )
    parser.add_argument(
        "--span-duration",
        type=float,
        default=1.1,
        help="Span length for combprint clouds. Shorter spans give MORE points, and PHD "
        "needs at least 40 — this is the knob to turn if tracks are being rejected.",
    )
    parser.add_argument("--span-hop", type=float, default=0.2)
    parser.add_argument("--n-reruns", type=int, default=3, help="GPTID average 3 runs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "label" not in df.columns:
        raise SystemExit("manifest needs a 'label' column")
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")

    rng = np.random.RandomState(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], dropna=False):
        n = min(len(grp), args.per_stratum)
        parts.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
    df = pd.concat(parts, ignore_index=True)
    logger.info("sampled %d tracks", len(df))

    meta_cols = [c for c in ("track_id", "label", "algorithm") if c in df.columns]
    jobs = [
        {
            "path": str(getattr(r, args.audio_column)),
            "meta": {c: getattr(r, c) for c in meta_cols},
            "max_duration": args.max_duration,
            "level_match": args.level_match,
            "cloud": args.cloud,
            "span_duration": args.span_duration,
            "span_hop": args.span_hop,
            "n_reruns": args.n_reruns,
        }
        for r in df.itertuples(index=False)
        if Path(str(getattr(r, args.audio_column))).exists()
    ]
    if not jobs:
        raise SystemExit(f"no files resolved from column {args.audio_column!r}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_point_cloud, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 100 == 0 or i == len(futures):
                logger.info("  %d/%d", i, len(futures))

    per_track = pd.DataFrame(rows)
    ctrl = "_lvl" if args.level_match else ""
    per_track.to_csv(out_dir / f"phd_per_track_{args.cloud}{ctrl}.csv", index=False)

    n_err = int(per_track.get("error", pd.Series(dtype=object)).notna().sum())
    if n_err == len(per_track):
        raise SystemExit(
            "every track failed — refusing to write an empty result. First error: "
            f"{per_track['error'].dropna().head(1).tolist()}"
        )
    if n_err:
        logger.warning(
            "%d/%d tracks failed (median cloud size %s). If most are 'cloud too small', "
            "lower --span-hop or raise --max-duration; do NOT report a number computed "
            "from the survivors without saying which tracks were excluded and why.",
            n_err,
            len(per_track),
            per_track["n_points"].median(),
        )

    is_real = (per_track["label"] == "real").to_numpy()
    report: list[dict] = []
    for col in ("id_phd", "id_twonn"):
        if col not in per_track.columns:
            continue
        s = per_track[col].to_numpy(float)
        real_s = s[is_real]
        real_s = real_s[np.isfinite(real_s)]
        for alg in sorted(set(per_track.loc[~is_real, "algorithm"]) - {""}):
            fake_s = s[(~is_real) & (per_track["algorithm"] == alg).to_numpy()]
            fake_s = fake_s[np.isfinite(fake_s)]
            if len(fake_s) < 5 or len(real_s) < 5:
                continue
            # GPTID's hypothesis is fake < real, so the DETECTOR score is -ID.
            # auc_signed > 0.5 therefore means "the published direction holds".
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            auc, eer = auc_and_eer(y, -np.r_[real_s, fake_s])
            report.append(
                {
                    "feature": col,
                    "algorithm": alg,
                    "auc_signed_fake_lower": round(auc, 4),
                    "auc_abs": round(max(auc, 1 - auc), 4),
                    "eer_pct": round(eer * 100, 2),
                    "real_median_id": round(float(np.median(real_s)), 4),
                    "fake_median_id": round(float(np.median(fake_s)), 4),
                    "delta_id": round(float(np.median(fake_s) - np.median(real_s)), 4),
                }
            )

    if not report:
        raise SystemExit("no comparable real/fake pairs — check the manifest")
    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / f"phd_auc_{args.cloud}{ctrl}.csv", index=False)

    pivot = rep.pivot_table(index="feature", columns="algorithm", values="auc_signed_fake_lower")
    pivot["MACRO"] = pivot.mean(axis=1)
    logger.info(
        "\nINTRINSIC-DIMENSION AUC on %r clouds (signed; >0.5 means GPTID's direction, "
        "fake ID < real ID, holds):\n%s",
        args.cloud,
        pivot.sort_values("MACRO", ascending=False).to_string(),
    )
    logger.info(
        "\nMedian ID per generator — the mechanistic check. GPTID report generated text "
        "sitting ~1.5 dimensions BELOW human text; the sign of delta_id is the claim:\n%s",
        rep[["feature", "algorithm", "real_median_id", "fake_median_id", "delta_id"]].to_string(index=False),
    )
    logger.info(
        "Gate this before reporting: python scripts/run_confound_gate.py "
        "--score-csv %s --score-column id_phd --manifest %s",
        out_dir / f"phd_per_track_{args.cloud}{ctrl}.csv",
        args.manifest,
    )
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
