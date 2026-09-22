"""Decompose the anomaly score into (A) artifact signal and (B) production/corpus signal.

Motivation (report §9.7.12). Our flow's score responds to two separable things:

  (A) genuine reconstruction/codec artifacts — proven to exist and to be
      content-independent by the reconstruction control (content-identical pairs
      separate 0.980 -> 0.727 with a monotonic dose-response);
  (B) corpus / production-chain identity — mastering, noise floor, delivery
      chain. This does NOT transfer to a new corpus, and it inflates any
      benchmark whose real and fake populations come from different pipelines.

Almost every AI-music benchmark pairs real music from one pipeline with
generated audio from another, so (B) is available to every detector evaluated on
them. This script measures how much of OUR number is (B), by residualising the
anomaly score against measurable production proxies and re-computing AUC — the
same residualisation logic already used for the genre confound, applied to
production covariates.

Outputs, per generator and pooled:
  auc_raw            AUC of the raw anomaly score (what the paper currently reports)
  auc_resid          AUC after regressing out the production proxies  = the (A)-only estimate
  delta              auc_raw - auc_resid                              = the (B) contribution
  auc_proxy_only     AUC of the production proxy alone                = (B) upper bound

The regression is fit on REAL tracks only (label-free), so no fake information
leaks into the correction.

Usage:
    python scripts/analyze_production_confound.py \\
        --window-flow-csv data/processed/full_sonics_k12_symmetric/window_flow_eval_symmetric.csv \\
        --complexity-csv data/processed/full_sonics_k12_symmetric/complexity_compensated_complexity.csv \\
        --covariate-csv data/processed/covariate_profile_canonical/covariate_profiles.csv \\
        --output-csv data/processed/full_sonics_k12_symmetric/production_confound.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Production-chain proxies: how hard the signal is to compress, how loud it was
# mastered, and how much high-frequency content survived the delivery chain.
# None of these is a synthesis artifact; all differ between recording pipelines.
PROXY_CANDIDATES = [
    "flac_bits_per_sec",
    "loudness_lufs",
    "effective_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "spectral_centroid_mean",
]


def _residualise(score: np.ndarray, x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Remove the component of `score` linearly predictable from `x`.

    Coefficients are fit on `fit_mask` rows (REAL tracks only) so the correction
    never sees a label; the residual is then evaluated on all rows.
    """
    xf = np.c_[np.ones(len(x)), x]
    coef, *_ = np.linalg.lstsq(xf[fit_mask], score[fit_mask], rcond=None)
    return score - xf @ coef


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--window-flow-csv", required=True)
    ap.add_argument("--complexity-csv", default=None, help="track_id,flac_bits_per_sec")
    ap.add_argument("--covariate-csv", default=None, help="covariate_profiles.csv (loudness, bandwidth, ...)")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--min-per-group", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.window_flow_csv, low_memory=False)
    df["track_id"] = df["track_id"].astype(str)
    mean_col = next((c for c in df.columns if "wf_mean" in c), None)
    if mean_col is None:
        raise SystemExit(f"no *_wf_mean column in {args.window_flow_csv}")
    df["anomaly"] = -df[mean_col]

    for path in (args.complexity_csv, args.covariate_csv):
        if path and Path(path).exists():
            extra = pd.read_csv(path, low_memory=False)
            extra["track_id"] = extra["track_id"].astype(str)
            keep = ["track_id"] + [c for c in extra.columns if c in PROXY_CANDIDATES]
            df = df.merge(extra[keep].drop_duplicates("track_id"), on="track_id", how="left")

    proxies = [c for c in PROXY_CANDIDATES if c in df.columns and df[c].notna().sum() > 100]
    if not proxies:
        raise SystemExit(
            "No production proxies available. Pass --complexity-csv (FLAC bits/s) and/or "
            "--covariate-csv (loudness/bandwidth/rolloff)."
        )
    logger.info("production proxies in use: %s", proxies)

    ok = np.isfinite(df["anomaly"].to_numpy(float))
    for c in proxies:
        ok &= np.isfinite(df[c].to_numpy(float))
    df = df[ok].copy()
    logger.info(
        "%d tracks with complete anomaly + proxy data (%d real, %d fake)",
        len(df),
        int((df["label"] == "real").sum()),
        int((df["label"] == "fake").sum()),
    )

    x = df[proxies].to_numpy(float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    score = df["anomaly"].to_numpy(float)
    is_real = (df["label"] == "real").to_numpy()

    df["anomaly_resid"] = _residualise(score, x, is_real)
    # A single-number proxy score for the (B) upper bound: the real-fit linear
    # predictor itself (i.e. what production alone would predict).
    df["proxy_pred"] = score - df["anomaly_resid"]

    rows = []
    for alg in sorted(str(a) for a in df.loc[df["label"] == "fake", "algorithm"].dropna().unique() if str(a).strip()):
        sub_r = df[df["label"] == "real"]
        sub_f = df[(df["label"] == "fake") & (df["algorithm"] == alg)]
        if len(sub_f) < args.min_per_group:
            continue
        y = np.r_[np.zeros(len(sub_r)), np.ones(len(sub_f))]
        out = {"algorithm": alg, "n_real": len(sub_r), "n_fake": len(sub_f)}
        for tag, col in (("raw", "anomaly"), ("resid", "anomaly_resid"), ("proxy_only", "proxy_pred")):
            auc, eer = auc_and_eer(y, np.r_[sub_r[col].to_numpy(float), sub_f[col].to_numpy(float)])
            out[f"auc_{tag}"] = round(auc, 4)
            out[f"eer_{tag}_pct"] = round(eer * 100, 2)
        out["delta_B"] = round(out["auc_raw"] - out["auc_resid"], 4)
        rows.append(out)

    res = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_csv, index=False)
    logger.info("Saved %s\n%s", args.output_csv, res.to_string(index=False))
    if len(res):
        logger.info(
            "\nMACRO: raw %.4f | residual (A-only estimate) %.4f | production-proxy alone %.4f | "
            "(B) contribution %.4f",
            res["auc_raw"].mean(),
            res["auc_resid"].mean(),
            res["auc_proxy_only"].mean(),
            res["delta_B"].mean(),
        )
        logger.info(
            "READ: 'auc_resid' is the honest estimate of artifact-driven detection once "
            "measurable production differences are regressed out (fit on REAL tracks only, so "
            "label-free). A large 'delta_B' means the benchmark — not just our model — is "
            "separable by production chain, which applies to any detector evaluated on it."
        )


if __name__ == "__main__":
    main()
