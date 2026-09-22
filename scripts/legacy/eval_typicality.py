"""Typicality / two-sided scoring ablation — the theory-grounded response to the
MusicCaps likelihood-inversion finding. CPU-only, reads existing window_flow_eval.csv.

Background (verified citations): normalizing flows systematically assign HIGHER
likelihood to smoother/lower-complexity OOD inputs than to their own training
data (Nalisnick et al., ICLR 2019, arXiv:1810.09136; Serrà et al., ICLR 2020,
arXiv:1909.11480). The proposed fix is to test whether an input lies in the
model's TYPICAL SET rather than in its high-density region (Nalisnick et al.,
arXiv:1906.02994): score by |NLL - typical NLL| (two-sided) instead of raw NLL.

Our data shows exactly this failure on FakeMusicCaps: held-out MusicCaps reals
score MORE anomalous than TTM fakes (median LL 8.92 vs 19.54 — fakes are
smooth, cleanly-decoded audio; reals are noisy AudioSet clips), and the
coverage curve DECLINES with more training data. A two-sided score flags
"suspiciously typical/high-likelihood" tracks as anomalous too, at the risk of
hurting the standard (fake = low likelihood) SONICS setting — this script
measures both sides of that trade.

Scorers (all label-free; reference statistics fit on REAL rows only, with a
split-half protocol so reference reals are disjoint from evaluated reals):
  nll            : -wf_mean (the current headline; baseline)
  typicality     : |wf_mean - median(real wf_mean)|
  typicality_z   : |wf_mean - mean(real)| / std(real)
  mahalanobis_2d : Mahalanobis distance of (wf_mean, wf_std) to the real cloud
                   (captures the dispersion difference: real tracks' window-LL
                   spread differs from fakes')

Usage:
    python scripts/eval_typicality.py \\
        --window-flow-csv data/processed/fmc_native_flow_k2/window_flow_eval.csv \\
        --output-csv data/processed/fmc_native_flow_k2/typicality_ablation.csv
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

SCORERS = ("nll", "typicality", "typicality_z", "mahalanobis_2d")


def _scores(
    wf_mean: np.ndarray,
    wf_std: np.ndarray | None,
    ref_mean: np.ndarray,
    ref_std_col: np.ndarray | None,
    scorer: str,
) -> np.ndarray:
    """Anomaly scores (higher = more anomalous). ref_* are REAL reference rows."""
    if scorer == "nll":
        return -wf_mean
    if scorer == "typicality":
        return np.abs(wf_mean - np.median(ref_mean))
    if scorer == "typicality_z":
        return np.abs(wf_mean - ref_mean.mean()) / (ref_mean.std() + 1e-9)
    if scorer == "mahalanobis_2d":
        if wf_std is None or ref_std_col is None:
            return np.full_like(wf_mean, np.nan)
        ref = np.stack([ref_mean, ref_std_col], axis=1)
        mu = ref.mean(axis=0)
        cov = np.cov(ref.T) + 1e-9 * np.eye(2)
        inv = np.linalg.inv(cov)
        x = np.stack([wf_mean, wf_std], axis=1) - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", x, inv, x))
    raise ValueError(scorer)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--window-flow-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-group", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.window_flow_csv, low_memory=False)
    mean_col = next((c for c in df.columns if c.endswith("_wf_mean") or c == "wf_mean"), None)
    if mean_col is None:
        raise SystemExit(f"no *_wf_mean column in {args.window_flow_csv}")
    std_col = mean_col.replace("_wf_mean", "_wf_std") if "_wf_mean" in mean_col else "wf_std"
    if std_col not in df.columns:
        std_col = None
    logger.info("score column: %s (std: %s)", mean_col, std_col)

    real = df[(df["label"] == "real") & np.isfinite(df[mean_col])]
    fake = df[(df["label"] == "fake") & np.isfinite(df[mean_col])]

    # Split-half: reference statistics from one half of the reals, AUC evaluated
    # on the other half (reference and evaluated reals disjoint).
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(real))
    ref_real = real.iloc[idx[: len(real) // 2]]
    eval_real = real.iloc[idx[len(real) // 2 :]]
    logger.info("reals: %d reference + %d evaluated (split-half); fakes: %d", len(ref_real), len(eval_real), len(fake))

    ref_mean = ref_real[mean_col].to_numpy(float)
    ref_stdc = ref_real[std_col].to_numpy(float) if std_col else None

    rows = []
    for scorer in SCORERS:
        r_s = _scores(
            eval_real[mean_col].to_numpy(float),
            eval_real[std_col].to_numpy(float) if std_col else None,
            ref_mean,
            ref_stdc,
            scorer,
        )
        r_fin = r_s[np.isfinite(r_s)]
        if len(r_fin) < args.min_per_group:
            continue
        aucs = []
        for alg in sorted(str(a) for a in fake["algorithm"].dropna().unique() if str(a).strip()):
            f = fake[fake["algorithm"] == alg]
            f_s = _scores(
                f[mean_col].to_numpy(float), f[std_col].to_numpy(float) if std_col else None, ref_mean, ref_stdc, scorer
            )
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
    logger.info(
        "Read: 'nll' is the current headline scorer. If 'typicality*' or 'mahalanobis_2d' "
        "beats it on an inverted corpus (FakeMusicCaps) while roughly holding on SONICS, "
        "that is a theory-grounded (typical-set) improvement worth adopting/reporting."
    )


if __name__ == "__main__":
    main()
