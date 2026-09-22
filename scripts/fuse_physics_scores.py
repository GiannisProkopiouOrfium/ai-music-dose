"""Reals-only fusion of physics-oriented comb features. Zero-shot by construction.

Why a fusion, and why THIS one
------------------------------
The 2026-08-20 operator grid showed a clean complementarity on SONICS that no
single feature captures. At 120 s:

| feature | chirp-v2 | chirp-v3 | chirp-v3.5 | udio-120s | udio-30s | MACRO |
|---|---|---|---|---|---|---|
| `median_db__comb_strength` | 0.886 | 0.927 | 0.855 | 0.741 | 0.825 | **0.847** |
| `comb_log_strength` | 0.945 | 0.988 | 0.931 | 0.532 | 0.643 | 0.808 |
| `comb_residual_std` | 0.607 | 0.741 | 0.542 | **0.937** | 0.577 | 0.681 |
| `comb_stat_strength` | 0.603 | 0.755 | 0.518 | **0.882** | 0.764 | 0.704 |

Suno (chirp) is caught by periodicity; **Udio is caught by peak amount and
stationarity and NOT by periodicity**. A per-generator oracle over these four
reaches ~0.93 against the best single feature's 0.847, so roughly 0.08 of headroom
exists. The mechanism is interpretable: Udio's decoder does not put a comb at a
spacing our 16 kHz band resolves (its measured spacing, 83–105 Hz, sits on top of
real music's 85–97 Hz), but it still leaves excess peak structure. Afchar et al.
hit the same wall — their supervised model scores 39.83% on unseen Udio v32.

Why the previous fusion failed and this one should not
------------------------------------------------------
`fuse_labelfree_scores.py` averaged `comb_strength` and `comb_sharpness` and got
0.8308 against 0.9067 for strength alone. Averaging drags a strong detector toward
a weak one whenever the weak one is weak *everywhere*. Here the weak features are
weak on Suno and **strong exactly where the strong one fails**, so the correct
combiner is a **maximum over standardised scores**, not a mean: "any branch
fires" rather than "branches agree".

What makes it zero-shot
-----------------------
1. **Calibration is on REALS ONLY** — median and IQR of each feature over the real
   set. No fake ever touches the normalisation.
2. **Orientation is fixed a priori by physics**, never fitted. Every feature here
   is "higher = more decoder-like" by construction:
   `comb_strength`/`comb_log_strength` (more periodic residual),
   `comb_residual_std`/`comb_residual_kurtosis` (more peak structure),
   `comb_stat_strength` (peaks more time-invariant).
   That is the paper's central design principle — the sign comes from the
   mechanism, so fitting cannot mis-orient it, which is precisely what happens to
   every density arm.
3. No generator output, no fake label, at any stage.

Robust standardisation (median/IQR) rather than mean/SD, because the real set
contains outliers whose leverage would otherwise set the scale.

Usage (EC2)
-----------
    python scripts/fuse_physics_scores.py \\
        --score-csv reports/diagnostics/comb_sonics_grid_120s/comb_per_track_lvl_sm5_hull_db_f1000_noclip.csv \\
        --features median_db__comb_strength comb_log_strength comb_stat_strength comb_residual_std \\
        --out-dir reports/diagnostics/fusion_sonics_120s
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

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, bootstrap_auc, delong_test  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fuse_physics")

# Every feature whose sign is fixed by the mechanism: higher = more decoder-like.
# A feature NOT on this list must not enter the fusion, because its orientation
# would have to be learned, and learning it needs labels.
PHYSICS_ORIENTED = {
    "comb_strength",
    "comb_log_strength",
    "comb_sharpness",
    "comb_log_sharpness",
    "comb_stat_strength",
    "comb_mean_profile_strength",
    "comb_residual_std",
    "comb_residual_kurtosis",
    "comb_log_residual_std",
    "comb_log_residual_kurtosis",
}


def _is_harmonic_comb_score(col: str) -> bool:
    """`comb_harm4_prior_strength`, `comb_priormax2_strength`, `comb_mf_z` and kin.

    Admissible for the same reason `comb_strength` is, and the direction is if
    anything *more* firmly fixed. Each of these measures how much of the residual
    sits at a spacing a transposed-convolution stack could produce — either the
    autocorrelation averaged over the first M multiples of a candidate
    `f_s / prod(strides)` (`harm<M>_prior`), the peak of the autocorrelation over
    the union of those multiples (`priormax<M>`), or the residual summed at the
    tooth positions themselves (`mf`). **More energy at a physically realisable
    decoder spacing means more decoder**, in every case, and nothing in any of
    them is fitted.

    The `_hz` and `_offset_hz` companions are deliberately EXCLUDED: those are
    identities, not magnitudes, and ledger C1 established that a spacing's sign is
    a property of which decoders and which reals a corpus happens to contain
    (`comb_spacing_hz` is 0.64 on SONICS with all five families correct and 0.30
    on FakeMusicCaps with all five inverted). Sign-consistency inside one corpus is
    necessary and not sufficient, so they may never enter a zero-shot fusion.

    `comb_mfoff_z` is also excluded: it is the offset-scanning CONTROL for
    `comb_mf_z`, an upper bound that discards the DC-alignment constraint. Fusing a
    control would defeat its purpose.
    """
    base = col.split("__")[-1]
    if base in {"comb_mf_z", "comb_surrogate_z"}:
        return True
    return bool(re.fullmatch(r"comb_(harm\d+_(prior_)?|priormax\d+_)(strength|sharpness|stat_strength)", base))


def _is_nmf_peak_score(col: str) -> bool:
    """`r_k20_sigma5_bins4779_pooled` and friends.

    Afchar & Hennequin's peak-energy score is physics-oriented in exactly the same
    sense as the comb statistics: it measures how much of a track's reconstruction
    is narrow peak structure, and MORE peaks means MORE decoder. Its sign is fixed
    by the mechanism, not fitted, so it is admissible in a zero-shot fusion — and
    it is the one combination of this project's two novelties (the full-resolution
    NMF readout and two-sided scoring) that has not been tried.
    """
    return col.startswith("r_k") and "_sigma" in col


def _is_physics_oriented(col: str) -> bool:
    """Variant-prefixed names (`median_db__comb_strength`) inherit the base sign."""
    return col.split("__")[-1] in PHYSICS_ORIENTED or _is_nmf_peak_score(col) or _is_harmonic_comb_score(col)


def robust_z_on_reals(values: np.ndarray, is_real: np.ndarray, two_sided: bool = False) -> np.ndarray:
    """(x − median_real) / IQR_real, calibrated on reals only. No fake label used.

    ``two_sided`` returns the ABSOLUTE deviation instead — "how far from typical
    real music, in either direction".

    Why two-sided is not a hack, and why it is still zero-shot
    ---------------------------------------------------------
    Measured on the bandwidth-controlled SONICS corpus, every comb feature puts
    the two generator families on OPPOSITE SIDES of real music:

    | feature | chirp (signed AUC) | udio (signed AUC) |
    |---|---|---|
    | `comb_strength` | 0.908 | **0.183** |
    | `comb_stat_strength` | 0.959 | **0.259** |
    | `comb_residual_kurtosis` | 0.933 | **0.249** |

    udio at 0.18 is not noise — it is a consistent, reproducible inversion across
    every feature: udio sits BELOW real music where chirp sits far above. Real
    music is in the middle. A one-sided score must pick a direction and therefore
    must fail on one family; **an absolute deviation catches both**.

    The physical reading: a transposed-convolution vocoder ADDS periodic peaks
    (chirp), while a decoder that oversmooths REMOVES the spectral micro-texture
    real music has (udio). Both are departures from real music's spectral
    statistics, in opposite directions.

    Crucially the centre and the scale come from REALS ONLY, so no label is
    needed to decide which side a track is on — which is exactly what a
    per-generator sign flip would have required (§ sign-consistency).
    """
    ref = values[is_real & np.isfinite(values)]
    if len(ref) < 20:
        raise SystemExit(f"only {len(ref)} finite real values — too few to calibrate")
    med = float(np.median(ref))
    iqr = float(np.percentile(ref, 75) - np.percentile(ref, 25))
    if iqr <= 0:
        iqr = float(np.std(ref)) or 1.0
    z = (values - med) / iqr
    return np.abs(z) if two_sided else z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Columns to fuse. Each must be physics-oriented (higher = more fake).",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--combiner",
        nargs="+",
        default=["max", "mean"],
        help="max = 'any branch fires' (recommended; the complementarity is "
        "per-generator). mean = the combiner that FAILED before, kept as the ablation.",
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        default=False,
        help=(
            "Score |x - median_real| / IQR_real instead of the signed deviation. Catches "
            "generator families that sit on OPPOSITE sides of real music — on SONICS, "
            "chirp is far above and udio consistently below. Still zero-shot: centre and "
            "scale come from reals only."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.score_csv, low_memory=False)
    if "label" not in df.columns:
        raise SystemExit(f"{args.score_csv} has no 'label' column")
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")

    missing = [f for f in args.features if f not in df.columns]
    if missing:
        raise SystemExit(
            f"columns not in {args.score_csv}: {missing}\n"
            f"available comb columns: {sorted(c for c in df.columns if 'comb_' in c)[:30]}"
        )
    unoriented = [f for f in args.features if not _is_physics_oriented(f)]
    if unoriented:
        raise SystemExit(
            f"{unoriented} are not in PHYSICS_ORIENTED. A feature whose sign must be "
            "LEARNED cannot enter a zero-shot fusion — that is the whole point of the "
            "design. Add it to PHYSICS_ORIENTED only with a mechanistic argument for "
            "which direction means 'more decoder-like'."
        )

    y = (df["label"] != "real").astype(int).to_numpy()
    is_real = y == 0
    logger.info(
        "calibrating on %d reals; fusing %d features: %s",
        int(is_real.sum()),
        len(args.features),
        ", ".join(args.features),
    )

    z = np.column_stack(
        [robust_z_on_reals(df[f].to_numpy(float), is_real, two_sided=args.two_sided) for f in args.features]
    )
    if args.two_sided:
        logger.info(
            "TWO-SIDED: scoring |x - median_real| / IQR_real. Use this when generator "
            "families sit on OPPOSITE sides of real music (SONICS: chirp above, udio "
            "below). Report the one-sided variant alongside it as the ablation."
        )
    # A NaN in one branch must not sink the whole track: treat it as "this branch
    # did not fire" rather than propagating.
    z_max = np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=1)
    z_max[~np.isfinite(z_max)] = np.nan
    z_mean = np.nanmean(np.where(np.isfinite(z), z, np.nan), axis=1)

    if "max" in args.combiner:
        df["fuse_zmax"] = z_max
    if "mean" in args.combiner:
        df["fuse_zmean"] = z_mean

    # Also score each feature's own two-sided version, so the table shows whether
    # the gain comes from the fusion or from two-sidedness alone.
    if args.two_sided:
        for f in args.features:
            df[f"{f}__abs"] = robust_z_on_reals(df[f].to_numpy(float), is_real, two_sided=True)
    cols = list(args.features)
    cols += [f"{f}__abs" for f in args.features if f"{f}__abs" in df.columns]
    cols += [c for c in ("fuse_zmax", "fuse_zmean") if c in df.columns]
    rows: list[dict] = []
    for col in cols:
        s = df[col].to_numpy(float)
        ok = np.isfinite(s)
        if len(set(y[ok])) < 2:
            continue
        b = bootstrap_auc(y[ok], s[ok], seed=args.seed)
        row = {
            "score": col,
            "auc_pooled": round(float(b["auc"]), 4),
            "ci_lo": round(float(b["ci_lo"]), 4),
            "ci_hi": round(float(b["ci_hi"]), 4),
        }
        per_gen = {}
        for alg in sorted(set(df.loc[y == 1, "algorithm"]) - {""}):
            sel = ((df["algorithm"] == alg).to_numpy() | is_real) & ok
            if len(set(y[sel])) < 2:
                continue
            auc, _ = auc_and_eer(y[sel], s[sel])
            per_gen[alg] = round(auc, 4)
        row.update(per_gen)
        row["MACRO"] = round(float(np.mean(list(per_gen.values()))), 4) if per_gen else np.nan
        rows.append(row)

    rep = pd.DataFrame(rows).sort_values("MACRO", ascending=False)
    df.to_csv(out_dir / "fused_per_track.csv", index=False)
    rep.to_csv(out_dir / "fusion_auc.csv", index=False)
    logger.info("\nPHYSICS FUSION (reals-only calibration, signs fixed a priori):\n%s", rep.to_string(index=False))

    # Is the fusion actually better than its best single branch? DeLong, not eyeball.
    best_single = rep[rep["score"].isin(args.features)].iloc[0]
    for fused in ("fuse_zmax", "fuse_zmean"):
        if fused not in df.columns:
            continue
        a = df[fused].to_numpy(float)
        b_ = df[best_single["score"]].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b_)
        if len(set(y[both])) < 2:
            continue
        d = delong_test(y[both], a[both], b_[both])
        verdict = (
            "IMPROVEMENT"
            if d["delta_auc"] > 0 and d["p_value"] < 0.05
            else ("no significant change" if d["p_value"] >= 0.05 else "REGRESSION")
        )
        logger.info(
            "DeLong %s vs best single (%s): %.4f vs %.4f, delta %+.4f, z=%.2f, p=%.3g → %s",
            fused,
            best_single["score"],
            d["auc_a"],
            d["auc_b"],
            d["delta_auc"],
            d["z"],
            d["p_value"],
            verdict,
        )
    logger.info(
        "\nGate before reporting:\n  python scripts/run_confound_gate.py --score-csv %s "
        "--score-column fuse_zmax --descriptor-csv <channel csv> --out-dir <gate dir>",
        out_dir / "fused_per_track.csv",
    )
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
