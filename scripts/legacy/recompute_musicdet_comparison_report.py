"""Recompute the window-flow vs. MusicDET comparison report (Step 5 of
run_musicdet_comparison_corrected.sh) without re-running MusicDET training or
testing. Reads the already-cached MusicDET result .txt files from
~/MusicDET/output_sonics_matched/result/ and a window_flow_eval.csv, and
regenerates wf_vs_musicdet_auc_ci.csv.

Use this any time the window-flow side needs to be recomputed against a
DIFFERENT window_flow_eval.csv (e.g. after fixing a bug or restoring the
correct uncapped file) — the MusicDET side is untouched and free to reuse.

Usage
-----
    python scripts/recompute_musicdet_comparison_report.py \\
        --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
        --output-dir data/processed/musicdet_comparison_corrected \\
        --musicdet-result-dir ~/MusicDET/output_sonics_matched/result
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer  # noqa: E402


def load_mdet_scores(txt_path: Path) -> pd.DataFrame | None:
    """MusicDET result .txt: 'basename score label_str' per line, space-separated.
    score = -nll (higher = MORE real, per test.py's compute_scores)."""
    if not txt_path.exists():
        return None
    rows = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append({"id": parts[0], "score": float(parts[1]), "label": parts[2]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-flow-csv", required=True)
    ap.add_argument("--output-dir", default="data/processed/musicdet_comparison_corrected")
    ap.add_argument(
        "--musicdet-result-dir",
        default=str(Path.home() / "MusicDET/output_sonics_matched/result"),
    )
    args = ap.parse_args()

    wf_csv = Path(args.window_flow_csv)
    out_dir = Path(args.output_dir)
    result_dir = Path(args.musicdet_result_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    wf = pd.read_csv(wf_csv, low_memory=False)
    mean_col = next((c for c in wf.columns if "wf_mean" in c), None)
    wf_real = wf[wf["label"] == "real"][mean_col].dropna().values
    print(f"Reading window-flow scores from: {wf_csv}  (n_real={len(wf_real)}, col={mean_col})")
    print(f"Reading MusicDET results from:   {result_dir}")

    rows = []
    for alg in sorted(wf[wf["label"] == "fake"]["algorithm"].dropna().unique()):
        wf_fake = wf[(wf["label"] == "fake") & (wf["algorithm"] == alg)][mean_col].dropna().values
        if len(wf_fake) < 5:
            continue
        y_wf = np.r_[np.zeros(len(wf_real)), np.ones(len(wf_fake))]
        s_wf = np.r_[-wf_real, -wf_fake]  # negate: higher anomaly score = more fake
        wf_boot = bootstrap_auc(y_wf, s_wf)
        wf_eer = bootstrap_eer(y_wf, s_wf)

        row = {
            "algorithm": alg,
            "wf_auc": round(wf_boot["auc"], 4),
            "wf_auc_ci": f"[{wf_boot['ci_lo']:.4f},{wf_boot['ci_hi']:.4f}]",
            "wf_eer_pct": round(wf_eer["eer"] * 100, 2),
        }

        safe_name = alg.replace("/", "_")
        for tag, suffix in [("clean", ""), ("mp3_64k", "_mp3")]:
            mdet = load_mdet_scores(result_dir / f"eval_{safe_name}{suffix}.txt")
            if mdet is None or mdet["label"].nunique() < 2:
                continue
            y_m = (mdet["label"] == "fake").astype(int).values
            s_m = -mdet["score"].values  # negate: their score is higher=more real too
            m_boot = bootstrap_auc(y_m, s_m)
            m_eer = bootstrap_eer(y_m, s_m)
            row[f"mdet_{tag}_auc"] = round(m_boot["auc"], 4)
            row[f"mdet_{tag}_auc_ci"] = f"[{m_boot['ci_lo']:.4f},{m_boot['ci_hi']:.4f}]"
            row[f"mdet_{tag}_eer_pct"] = round(m_eer["eer"] * 100, 2)

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "wf_vs_musicdet_auc_ci.csv", index=False)
    print(f"\nComparison table:\n{df_out.to_string(index=False)}")
    print(f"\nSaved to {out_dir / 'wf_vs_musicdet_auc_ci.csv'}")


if __name__ == "__main__":
    main()
