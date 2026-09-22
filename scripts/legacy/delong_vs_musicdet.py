"""Paired DeLong test: our flow vs. MusicDET, on the tracks BOTH scored.

DeLong (1988) compares two ROC-AUCs computed on the SAME samples, accounting for
the correlation induced by both scorers ranking the same items. That pairing is
mandatory: our eval covers all 12,722 held-out reals while MusicDET's uses its
own ~1,273-track real test split, so a naive comparison of the two AUCs is not a
paired test and cannot support a significance claim.

This script therefore INTERSECTS on track identifier first and reports the test
only on the common subset, printing exactly how many tracks that is. If the
intersection is too small it says so and refuses rather than emitting a
misleading p-value.

Sign conventions (both verified against their sources):
  ours     : ``*_wf_mean`` is +log-likelihood (higher = more real) -> negate
  MusicDET : ``score`` is -nll (higher = more real, per their test.py)   -> negate

Usage:
    python scripts/delong_vs_musicdet.py \\
        --window-flow-csv data/processed/full_sonics_k12_symmetric/window_flow_eval_symmetric.csv \\
        --musicdet-result-dir ~/MusicDET/output_sonics_matched/result \\
        --output-csv data/processed/full_sonics_k12_symmetric/delong_vs_musicdet.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, delong_test  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _load_mdet(txt_path: Path) -> pd.DataFrame | None:
    if not txt_path.exists():
        return None
    rows = []
    with open(txt_path) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) < 3:
                continue
            rows.append({"id": p[0], "score": float(p[1]), "label": p[2]})
    return pd.DataFrame(rows) if rows else None


def _norm_id(x: str) -> str:
    """Normalise a track identifier: drop directories, extension, and common
    canonicalisation prefixes/suffixes so the two pipelines' ids can be matched."""
    s = str(x).strip()
    s = Path(s).name
    s = re.sub(r"\.(wav|mp3|flac|m4a|ogg)$", "", s, flags=re.IGNORECASE)
    for pref in ("real__", "fake__"):
        if s.startswith(pref):
            s = s[len(pref) :]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--window-flow-csv", required=True)
    ap.add_argument("--musicdet-result-dir", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument(
        "--min-common", type=int, default=100, help="Minimum matched tracks per class before a test is reported."
    )
    args = ap.parse_args()

    wf = pd.read_csv(args.window_flow_csv, low_memory=False)
    mean_col = next((c for c in wf.columns if "wf_mean" in c), None)
    if mean_col is None:
        raise SystemExit(f"no *_wf_mean column in {args.window_flow_csv}")
    wf["_id"] = wf["track_id"].map(_norm_id)
    wf["_ours"] = -wf[mean_col]  # anomaly-oriented
    result_dir = Path(args.musicdet_result_dir).expanduser()

    rows = []
    for alg in sorted(wf.loc[wf["label"] == "fake", "algorithm"].dropna().unique()):
        mdet = _load_mdet(result_dir / f"eval_{str(alg).replace('/', '_')}.txt")
        if mdet is None or mdet["label"].nunique() < 2:
            logger.warning("%s: no usable MusicDET result file — skipped", alg)
            continue
        mdet["_id"] = mdet["id"].map(_norm_id)
        mdet["_mdet"] = -mdet["score"]  # anomaly-oriented
        mdet["_y"] = (mdet["label"].astype(str).str.lower() == "fake").astype(int)

        ours = wf[(wf["label"] == "real") | (wf["algorithm"] == alg)].copy()
        ours["_y"] = (ours["label"] == "fake").astype(int)

        merged = ours.merge(mdet[["_id", "_mdet", "_y"]], on="_id", how="inner", suffixes=("", "_m"))
        # labels must agree for a matched id, otherwise the join is wrong
        disagree = int((merged["_y"] != merged["_y_m"]).sum())
        if disagree:
            logger.warning("%s: %d matched ids disagree on label — dropping them", alg, disagree)
            merged = merged[merged["_y"] == merged["_y_m"]]
        merged = merged[np.isfinite(merged["_ours"]) & np.isfinite(merged["_mdet"])]

        n_pos = int((merged["_y"] == 1).sum())
        n_neg = int((merged["_y"] == 0).sum())
        logger.info("%s: %d matched tracks (%d fake / %d real)", alg, len(merged), n_pos, n_neg)
        if min(n_pos, n_neg) < args.min_common:
            logger.warning(
                "%s: only %d/%d matched per class (< --min-common %d). The two pipelines' "
                "evaluation splits barely overlap, so a paired test here would be noise. "
                "Reporting counts only.",
                alg,
                n_pos,
                n_neg,
                args.min_common,
            )
            rows.append(
                {
                    "algorithm": alg,
                    "n_matched": len(merged),
                    "n_fake": n_pos,
                    "n_real": n_neg,
                    "status": "insufficient_overlap",
                }
            )
            continue

        y = merged["_y"].to_numpy(int)
        a = merged["_ours"].to_numpy(float)
        b = merged["_mdet"].to_numpy(float)
        res = delong_test(y, a, b)
        auc_a, eer_a = auc_and_eer(y, a)
        auc_b, eer_b = auc_and_eer(y, b)
        rows.append(
            {
                "algorithm": alg,
                "n_matched": len(merged),
                "n_fake": n_pos,
                "n_real": n_neg,
                "ours_auc": round(auc_a, 4),
                "ours_eer_pct": round(eer_a * 100, 2),
                "musicdet_auc": round(auc_b, 4),
                "musicdet_eer_pct": round(eer_b * 100, 2),
                "delta_auc": round(auc_a - auc_b, 4),
                "z": round(res["z"], 2) if np.isfinite(res["z"]) else None,
                "p_value": res["p_value"],
                "status": "ok",
            }
        )

    if not rows:
        raise SystemExit("No comparable generators found — check --musicdet-result-dir.")
    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.info("Saved %s\n%s", args.output_csv, out.to_string(index=False))
    ok = out[out["status"] == "ok"]
    if len(ok):
        logger.info(
            "Significant at p<0.05: %d/%d generators (ours better in %d).",
            int((ok["p_value"] < 0.05).sum()),
            len(ok),
            int((ok["delta_auc"] > 0).sum()),
        )
        logger.info(
            "NOTE: this is a PAIRED test on the intersection of the two evaluation splits, "
            "which is smaller than either paper's full eval set — state the matched n in the paper."
        )


if __name__ == "__main__":
    main()
