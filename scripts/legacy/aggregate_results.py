"""Aggregate stratified_eval_ablation.csv files from multiple run directories.

Produces a pivot table (rows = algorithm/fake_type, columns = run name)
so all AUC numbers are visible at a glance for data-driven decisions.

Usage
-----
python scripts/aggregate_results.py \
    --dirs data/processed/pilot_fair \
           data/processed/pilot_fixed30_temporal \
           data/processed/pilot_muq \
           data/processed/region_sweep/intro \
           data/processed/region_sweep/middle \
           data/processed/region_sweep/outro \
           data/processed/sonics_balanced_ablation_mert_temporal_all \
           data/processed/mert_subsample_lp8k_d55 \
    --output data/processed/results_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_dir(d: str) -> pd.DataFrame | None:
    p = Path(d) / "stratified_eval_ablation.csv"
    if not p.exists():
        print(f"[SKIP] {p} not found", file=sys.stderr)
        return None
    df = pd.read_csv(p)
    df["run"] = Path(d).name
    df["run_path"] = str(d)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate stratified AUC tables from multiple run directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dirs", nargs="+", required=True, help="Run directories to aggregate")
    parser.add_argument("--output", default="data/processed/results_summary.csv", help="Output CSV path")
    parser.add_argument(
        "--metric",
        default="auc",
        choices=["auc", "f1", "accuracy", "fpr", "fnr"],
        help="Metric column to pivot",
    )
    args = parser.parse_args()

    frames = [f for d in args.dirs if (f := load_dir(d)) is not None]
    if not frames:
        print("No results found — nothing to aggregate.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"Saved combined table → {args.output}  ({len(combined)} rows)")

    # ---- Pivot: algorithm-level ----
    alg_df = combined[combined["dimension"] == "algorithm"].copy()
    if not alg_df.empty:
        pivot_alg = alg_df.pivot_table(
            index="value",
            columns="run",
            values=args.metric,
            aggfunc="first",
        ).round(3)
        print("\n=== Per-generator AUC ===")
        print(pivot_alg.to_string())

    # ---- Pivot: fake_type-level ----
    ft_df = combined[combined["dimension"] == "fake_type"].copy()
    if not ft_df.empty:
        pivot_ft = ft_df.pivot_table(
            index="value",
            columns="run",
            values=args.metric,
            aggfunc="first",
        ).round(3)
        print("\n=== Per-fake_type AUC ===")
        print(pivot_ft.to_string())

    # ---- Best bundle per run ----
    if "best_bundle" in combined.columns:
        print("\n=== Best bundles per run ===")
        for run_name in combined["run"].unique():
            sub = combined[combined["run"] == run_name]
            bundles = sub["best_bundle"].dropna().unique()
            b_str = ", ".join(str(b) for b in bundles[:3])
            avg_auc = sub[args.metric].mean()
            print(f"  {run_name:<45} avg_{args.metric}={avg_auc:.3f}  bundles={b_str}")

    print(f"\nFull table saved to: {args.output}")


if __name__ == "__main__":
    main()
