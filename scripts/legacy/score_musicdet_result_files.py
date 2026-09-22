"""Score MusicDET's own result .txt files into per-generator AUC / EER / precision.

Their ``test.py`` writes ``<basename> <score> <label_str>`` per line into
``<model_path>/result/TTM0k_eval.txt``, where ``score = -nll`` — i.e. **higher =
more real** (``compute_scores``). We therefore negate to get an anomaly score
before computing AUC/EER, exactly as ``recompute_musicdet_comparison_report.py``
does, so numbers here are directly comparable to our own tables.

Reports EER (for comparability with their Table 1) *and* the precision-oriented
operating points we care about for deployment.

Usage:
    python scripts/score_musicdet_result_files.py \\
        --result-dir data/processed/musicdet_ours_fmc/result \\
        --mapping    data/processed/musicdet_protocol_fmc/mapping.csv \\
        --label      "their model / our canonical corpus" \\
        --output-csv data/processed/musicdet_ours_fmc/scored.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def load_scores(txt_path: Path) -> pd.DataFrame:
    rows = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({"id": parts[0], "score": float(parts[1]), "label": parts[-1]})
            except ValueError:
                continue
    return pd.DataFrame(rows)


def eer_from(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> tuple[float, float]:
    """Return (TPR, precision) at the largest threshold with FPR <= target."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    if len(ok) == 0:
        return float("nan"), float("nan")
    i = int(ok[-1])
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp, fp = tpr[i] * n_pos, fpr[i] * n_neg
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return float(tpr[i]), float(prec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--mapping", default="", help="mapping.csv from export_musicdet_protocol.py")
    ap.add_argument("--label", default="", help="free-text tag for this run, copied into every row")
    ap.add_argument("--output-csv", default="")
    args = ap.parse_args()

    rdir = Path(args.result_dir)
    files = sorted(rdir.glob("TTM0*_eval.txt"))
    if not files:
        raise SystemExit(f"no TTM0*_eval.txt in {rdir}")

    names: dict[str, str] = {}
    if args.mapping and Path(args.mapping).exists():
        m = pd.read_csv(args.mapping)
        names = dict(zip(m["file"], m["algorithm"]))

    rows = []
    for f in files:
        df = load_scores(f)
        if df.empty or df["label"].nunique() < 2:
            logger.warning("%s: needs both labels, got %s — skipped", f.name, df["label"].unique().tolist())
            continue
        y = (df["label"] == "fake").astype(int).to_numpy()
        s = -df["score"].to_numpy()  # their score is higher=more real -> negate for anomaly
        auc = float(roc_auc_score(y, s))
        t1, p1 = tpr_at_fpr(y, s, 0.01)
        t01, p01 = tpr_at_fpr(y, s, 0.001)
        rows.append(
            {
                "file": f.name,
                "algorithm": names.get(f.name, f.stem),
                "run": args.label,
                "n_real": int((1 - y).sum()),
                "n_fake": int(y.sum()),
                "auc": round(auc, 4),
                "eer_pct": round(100 * eer_from(y, s), 2),
                "tpr_at_fpr1pct": round(t1, 4),
                "prec_at_fpr1pct": round(p1, 4),
                "tpr_at_fpr0.1pct": round(t01, 4),
                "prec_at_fpr0.1pct": round(p01, 4),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("nothing scored")
    print(out.to_string(index=False))
    print(f"\nMACRO  auc={out['auc'].mean():.4f}  eer={out['eer_pct'].mean():.2f}%")
    if abs(out["auc"].mean() - 0.5) < 0.05:
        print(
            "\n>>> Macro AUC is at chance. Before concluding anything about the METHOD, check the\n"
            ">>> confound: if this ran on MP3-round-tripped canonical audio, MusicDET's own paper\n"
            ">>> reports 35.85% EER under MP3. Re-run on RAW audio to separate the two."
        )
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        logger.info("wrote %s", args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
