"""Complexity-compensated likelihood scoring (Serrà et al., ICLR 2020, arXiv:1909.11480).

The FakeMusicCaps failure is a likelihood-OOD inversion: flow likelihood is
dominated by input complexity (noisy AudioSet reals -> low likelihood; smooth
TTM fakes -> high likelihood). Serrà et al.'s parameter-free remedy is a
likelihood-RATIO against a universal lossless compressor:

    S(x) = NLL_model(x) - lambda * L(x)

where L(x) is the input's complexity estimate — here **FLAC-compressed bits per
second** of the canonical audio (the audio-domain analog of their PNG/FLIF
image compressors). We implement it as a label-free RESIDUALIZATION (mirroring
this repo's covariate-residualization methodology): fit NLL ~ a + b*L on a
REFERENCE half of the real tracks, score by the residual — the part of the
likelihood NOT explained by complexity. Variants:

    nll             : baseline (current headline scorer)
    complexity_only : -L(x)  (are fakes simply smoother? diagnostic baseline)
    nll_resid       : NLL - (a + b*L)   one-sided residual
    abs_resid       : |NLL - (a + b*L)| two-sided residual (typicality x complexity)
    resid_mahal_2d  : Mahalanobis of (resid, wf_std) vs reference reals

All reference statistics are fit on a split-half of the reals (disjoint from
evaluated reals). CPU-only; FLAC extraction is cached to a sidecar CSV so
re-runs are instant.

Usage (EC2):
    python scripts/eval_complexity_compensated.py \\
        --window-flow-csv data/processed/fmc_native_flow_k2/window_flow_eval.csv \\
        --manifest data/processed/canonical_fmc_native/combined_manifest.csv \\
        --output-csv data/processed/fmc_native_flow_k2/complexity_compensated.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

SCORERS = ("nll", "complexity_only", "nll_resid", "abs_resid", "resid_mahal_2d")


from intrinsic_ai_music_detection.features.complexity import flac_bits_per_sec_file as _flac_bits_per_sec  # noqa: E402


def _compute_complexities(paths: dict[str, str], cache_csv: Path, n_workers: int) -> pd.DataFrame:
    """track_id -> flac bits/sec, cached to ``cache_csv``."""
    done: dict[str, float] = {}
    if cache_csv.exists():
        prev = pd.read_csv(cache_csv)
        done = dict(zip(prev["track_id"].astype(str), prev["flac_bits_per_sec"]))
        logger.info("complexity cache: %d entries loaded from %s", len(done), cache_csv)
    todo = {tid: p for tid, p in paths.items() if tid not in done}
    if todo:
        logger.info("computing FLAC complexity for %d tracks (%d workers)...", len(todo), n_workers)
        tids = list(todo)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, (tid, val) in enumerate(
                zip(tids, ex.map(_flac_bits_per_sec, [todo[t] for t in tids], chunksize=16))
            ):
                done[tid] = val
                if (i + 1) % 2000 == 0:
                    logger.info("  [%d/%d]", i + 1, len(tids))
                    pd.DataFrame({"track_id": list(done), "flac_bits_per_sec": list(done.values())}).to_csv(
                        cache_csv, index=False
                    )
        pd.DataFrame({"track_id": list(done), "flac_bits_per_sec": list(done.values())}).to_csv(cache_csv, index=False)
    return pd.DataFrame({"track_id": list(done), "flac_bits_per_sec": list(done.values())})


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--window-flow-csv", required=True)
    parser.add_argument("--manifest", required=True, help="CSV with track_id + canonical_path")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--per-track-csv",
        default="",
        help="also write the PER-TRACK table (track_id, label, algorithm, nll, flac_bits_per_sec). "
        "This is the file fuse_labelfree_scores.py --complexity-csv expects; --output-csv is the "
        "summary AUC table and cannot be joined onto per-track scores.",
    )
    parser.add_argument(
        "--complexity-csv", default=None, help="Cache/sidecar for FLAC complexities (default: <output>_complexity.csv)"
    )
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-group", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.window_flow_csv, low_memory=False)
    df["track_id"] = df["track_id"].astype(str)
    mean_col = next((c for c in df.columns if c.endswith("_wf_mean") or c == "wf_mean"), None)
    if mean_col is None:
        raise SystemExit(f"no *_wf_mean column in {args.window_flow_csv}")
    std_col = mean_col.replace("_wf_mean", "_wf_std")
    if std_col not in df.columns:
        std_col = None

    manifest = pd.read_csv(args.manifest, low_memory=False)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"]
    manifest["track_id"] = manifest["track_id"].astype(str)
    paths = dict(zip(manifest["track_id"], manifest["canonical_path"].astype(str)))

    cache_csv = Path(args.complexity_csv or str(Path(args.output_csv).with_suffix("")) + "_complexity.csv")
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    comp = _compute_complexities(paths, cache_csv, args.n_workers)

    df = df.merge(comp, on="track_id", how="left")
    df["nll"] = -df[mean_col]  # anomaly-oriented
    valid = df[np.isfinite(df["nll"]) & np.isfinite(df["flac_bits_per_sec"])]
    logger.info("%d/%d tracks with finite NLL + complexity", len(valid), len(df))

    real = valid[valid["label"] == "real"]
    fake = valid[valid["label"] == "fake"]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(real))
    ref = real.iloc[idx[: len(real) // 2]]
    ev_real = real.iloc[idx[len(real) // 2 :]]
    logger.info("reals: %d reference + %d evaluated; fakes: %d", len(ref), len(ev_real), len(fake))
    logger.info(
        "complexity (flac bits/s): real median %.0f  fake median %.0f  — a gap here confirms the "
        "smoothness asymmetry driving the inversion",
        real["flac_bits_per_sec"].median(),
        fake["flac_bits_per_sec"].median(),
    )

    # Persist the PER-TRACK frame. --output-csv holds the summary AUC table, which
    # is NOT what fuse_labelfree_scores.py consumes: it needs track_id +
    # flac_bits_per_sec to join the complexity component onto the flow scores.
    # Without this, fusion fails with KeyError: 'track_id'.
    if getattr(args, "per_track_csv", ""):
        keep = [c for c in ("track_id", "label", "algorithm", "nll", "flac_bits_per_sec") if c in valid.columns]
        Path(args.per_track_csv).parent.mkdir(parents=True, exist_ok=True)
        valid[keep].to_csv(args.per_track_csv, index=False)
        logger.info("per-track complexity → %s (%d rows, cols=%s)", args.per_track_csv, len(valid), keep)

    # Fit NLL ~ a + b*L on reference reals (the complexity-explained likelihood component).
    b, a = np.polyfit(ref["flac_bits_per_sec"], ref["nll"], 1)
    logger.info(
        "reference fit: NLL = %.4g + %.4g * L   (r=%.3f on reference reals)",
        a,
        b,
        np.corrcoef(ref["flac_bits_per_sec"], ref["nll"])[0, 1],
    )

    def _resid(sub: pd.DataFrame) -> np.ndarray:
        return (sub["nll"] - (a + b * sub["flac_bits_per_sec"])).to_numpy(float)

    ref_resid = _resid(ref)
    ref_std = ref[std_col].to_numpy(float) if std_col else None

    def _scores(sub: pd.DataFrame, scorer: str) -> np.ndarray:
        if scorer == "nll":
            return sub["nll"].to_numpy(float)
        if scorer == "complexity_only":
            return -sub["flac_bits_per_sec"].to_numpy(float)  # low complexity = suspicious
        r = _resid(sub)
        if scorer == "nll_resid":
            return r
        if scorer == "abs_resid":
            return np.abs(r - np.median(ref_resid))
        if scorer == "resid_mahal_2d":
            if std_col is None:
                return np.full(len(sub), np.nan)
            ref2 = np.stack([ref_resid, ref_std], axis=1)
            mu = ref2.mean(axis=0)
            cov = np.cov(ref2.T) + 1e-9 * np.eye(2)
            inv = np.linalg.inv(cov)
            x = np.stack([r, sub[std_col].to_numpy(float)], axis=1) - mu
            return np.sqrt(np.einsum("ij,jk,ik->i", x, inv, x))
        raise ValueError(scorer)

    rows = []
    for scorer in SCORERS:
        r_s = _scores(ev_real, scorer)
        r_fin = r_s[np.isfinite(r_s)]
        if len(r_fin) < args.min_per_group:
            continue
        aucs = []
        for alg in sorted(str(x) for x in fake["algorithm"].dropna().unique() if str(x).strip()):
            f = fake[fake["algorithm"] == alg]
            f_s = _scores(f, scorer)
            f_fin = f_s[np.isfinite(f_s)]
            if len(f_fin) < args.min_per_group:
                continue
            y = np.r_[np.zeros(len(r_fin)), np.ones(len(f_fin))]
            s = np.r_[r_fin, f_fin]
            auc, eer = auc_and_eer(y, s)
            rows.append(
                {
                    "scorer": scorer,
                    "algorithm": alg,
                    "auc": round(auc, 4),
                    "eer_pct": round(eer * 100, 2),
                    "n_real": len(r_fin),
                    "n_fake": len(f_fin),
                }
            )
            aucs.append(auc)
        if aucs:
            rows.append(
                {
                    "scorer": scorer,
                    "algorithm": "MACRO_AVG",
                    "auc": round(float(np.mean(aucs)), 4),
                    "eer_pct": float("nan"),
                    "n_real": len(r_fin),
                    "n_fake": len(fake),
                }
            )

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.info("Saved %s", args.output_csv)
    pivot = out[out["algorithm"] != "MACRO_AVG"].pivot_table(index="scorer", columns="algorithm", values="auc")
    pivot["MACRO_AVG"] = out[out["algorithm"] == "MACRO_AVG"].set_index("scorer")["auc"]
    logger.info("\nAUC by scorer (rows) x generator (cols):\n%s", pivot.to_string())


if __name__ == "__main__":
    main()
