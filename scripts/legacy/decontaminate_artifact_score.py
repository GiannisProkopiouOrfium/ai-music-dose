"""Arm F3 — remove the "how simple is this music" axis from the artifact score.

The idea, and why it is not the fusion that already failed
-----------------------------------------------------------
Serrà's complexity compensation, and every other correction in the OOD
literature, runs in one direction: correct the LIKELIHOOD by a complexity term.
We tried that; it was a wash (FMC 0.604 vs 0.620).

The reverse has never been run. ``comb_strength`` is not a pure artifact measure:
it also responds to how tonal and regular the music is, because a sustained note
puts a harmonic comb in the spectrum. The flow's log-likelihood is, by this
project's own seven-way result, an excellent measure of exactly that nuisance —
"how typical/simple is this audio". So:

    residual = comb_strength − f(log p(x)),   f fitted on REALS ONLY

projects the nuisance direction out of the artifact score. This differs from the
fusion that failed (0.8308 vs 0.9067 for strength alone) in kind, not degree:
that fusion AVERAGED two detectors, which drags a strong one toward a weak one.
This subtracts a nuisance regressor, which cannot dilute the signal — at worst
the coefficient comes out near zero and the residual equals the original score.

Zero-shot, precisely: the regression sees only real tracks and only two numbers
per track. No generator output, no fake label, at any stage.

Robust regression (Huber) rather than OLS, because the reals include outliers
whose leverage would otherwise set the slope.

Usage (EC2)
-----------
    python scripts/decontaminate_artifact_score.py \\
        --comb-csv reports/diagnostics/comb_fmc_hull_db/comb_per_track_lvl_hull_db_f1000.csv \\
        --nuisance-csv reports/diagnostics/flow_terms_fmc/flow_terms_per_track_lvl.csv \\
        --nuisance-column neg_log_prob \\
        --out-dir reports/diagnostics/decontaminated_fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, bootstrap_auc, delong_test  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("decontaminate")


def fit_nuisance_on_reals(score: np.ndarray, nuisance: np.ndarray, is_real: np.ndarray, degree: int = 1) -> np.ndarray:
    """Residual of ``score`` after regressing out ``nuisance``, fitted on reals only.

    Returns the residual for EVERY row; only the real rows contributed to the fit.
    """
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import PolynomialFeatures

    ok = np.isfinite(score) & np.isfinite(nuisance)
    fit_mask = ok & is_real
    if fit_mask.sum() < 30:
        raise SystemExit(
            f"only {int(fit_mask.sum())} real tracks with both columns finite — "
            "not enough to fit the nuisance regression"
        )

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_fit = poly.fit_transform(nuisance[fit_mask].reshape(-1, 1))
    reg = HuberRegressor().fit(x_fit, score[fit_mask])

    out = np.full(len(score), np.nan)
    x_all = poly.transform(nuisance[ok].reshape(-1, 1))
    out[ok] = score[ok] - reg.predict(x_all)
    logger.info(
        "nuisance fit on %d reals (degree %d): coef %s, intercept %.4f",
        int(fit_mask.sum()),
        degree,
        np.round(reg.coef_, 5).tolist(),
        float(reg.intercept_),
    )
    return out


def _per_generator(df: pd.DataFrame, y: np.ndarray, col: str) -> dict[str, float]:
    s = df[col].to_numpy(float)
    real = s[(y == 0) & np.isfinite(s)]
    out = {}
    for alg in sorted(set(df.loc[y == 1, "algorithm"]) - {""}):
        fake = s[(y == 1) & (df["algorithm"] == alg).to_numpy() & np.isfinite(s)]
        if len(fake) < 5 or len(real) < 5:
            continue
        auc, _ = auc_and_eer(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
        out[alg] = round(auc, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comb-csv", required=True)
    parser.add_argument("--comb-column", default="comb_strength")
    parser.add_argument("--nuisance-csv", required=True)
    parser.add_argument(
        "--nuisance-column",
        default="neg_log_prob",
        help="The 'how simple is this audio' regressor. neg_log_prob is the full flow "
        "score; neg_log_pz isolates the base-density term (see score_flow_terms.py).",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--degree", type=int, default=1, help="1 = linear; 2 tests curvature.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comb = pd.read_csv(args.comb_csv, low_memory=False)
    nuis = pd.read_csv(args.nuisance_csv, low_memory=False)
    for name, df, col in (("--comb-csv", comb, args.comb_column), ("--nuisance-csv", nuis, args.nuisance_column)):
        if col not in df.columns:
            raise SystemExit(f"{name} has no column {col!r}")
        if "track_id" not in df.columns:
            raise SystemExit(f"{name} has no track_id column")

    merged = comb.merge(nuis[["track_id", args.nuisance_column]], on="track_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise SystemExit("no overlapping track_ids between the two CSVs — refusing to write an empty result")
    logger.info("joined %d tracks (comb %d, nuisance %d)", len(merged), len(comb), len(nuis))
    if "algorithm" not in merged.columns:
        merged["algorithm"] = ""
    merged["algorithm"] = merged["algorithm"].fillna("")

    y = (merged["label"] != "real").astype(int).to_numpy()
    score = merged[args.comb_column].to_numpy(float)
    nuisance = merged[args.nuisance_column].to_numpy(float)

    merged["comb_decontaminated"] = fit_nuisance_on_reals(score, nuisance, is_real=(y == 0), degree=args.degree)
    merged.to_csv(out_dir / "decontaminated_per_track.csv", index=False)

    rows = []
    for col in (args.comb_column, args.nuisance_column, "comb_decontaminated"):
        s = merged[col].to_numpy(float)
        ok = np.isfinite(s)
        if len(set(y[ok])) < 2:
            continue
        b = bootstrap_auc(y[ok], s[ok], seed=args.seed)
        rows.append(
            {
                "score": col,
                "auc": round(float(b["auc"]), 4),
                "ci_lo": round(float(b["ci_lo"]), 4),
                "ci_hi": round(float(b["ci_hi"]), 4),
                **{f"auc_{k}": v for k, v in _per_generator(merged, y, col).items()},
            }
        )
    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / "decontaminated_auc.csv", index=False)
    logger.info("\nDECONTAMINATION (F3):\n%s", rep.to_string(index=False))

    both = np.isfinite(score) & np.isfinite(merged["comb_decontaminated"].to_numpy(float))
    if len(set(y[both])) == 2:
        d = delong_test(y[both], merged["comb_decontaminated"].to_numpy(float)[both], score[both])
        logger.info(
            "\nDeLong, decontaminated vs raw %s: AUC %.4f vs %.4f, delta %+.4f, z=%.2f, p=%.3g",
            args.comb_column,
            d["auc_a"],
            d["auc_b"],
            d["delta_auc"],
            d["z"],
            d["p_value"],
        )
        if d["delta_auc"] > 0 and d["p_value"] < 0.05:
            logger.info(
                "Significant improvement. Report it WITH the DeLong statistic, and gate it "
                "(run_confound_gate.py) before it goes in a table."
            )
        elif d["delta_auc"] <= 0:
            logger.info(
                "No improvement. Record F3 as a negative with its mechanism — the flow score "
                "is not capturing the nuisance direction that contaminates the comb — rather "
                "than dropping the arm silently."
            )
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
