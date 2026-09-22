"""Characterize the udio wall: why is udio-120s indistinguishable but udio-30s not?

Both come from Udio, yet udio-30s is catchable (label-free one-class AUC ~0.92)
while udio-120s sits on the real manifold everywhere (~0.40-0.49). This script
quantifies the contrast so we can explain it in the paper rather than just
reporting it.

Analyses (geometry features from ablation_features.csv, optional spec_* merged):
  1. Distance-to-real-manifold per group -- LedoitWolf Mahalanobis fit on REALS
     only, mean anomaly score per generator. Tests "udio-120s ~ real".
  2. udio-120s vs udio-30s direct contrast -- Cohen's d per feature, ranked.
  3. Per-group real-vs-fake separability -- univariate AUC summary per embedding.
  4. Duration profile per group -- is the wall a duration effect?

Run (local or EC2; CPU-only):
    python scripts/characterize_udio_wall.py \
        --features data/processed/multiembed_descriptors/ablation_features.csv \
        --spectral data/processed/multiembed_descriptors/spectral_features.csv \
        --out-dir reports/diagnostics/udio_wall
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("characterize_udio_wall")

_META = {"track_id", "label", "fake_label", "algorithm", "y", "status"}
_EXCLUDE = ("_n_windows", "_window_ids", "_n_embeddings")


def _numeric_feats(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if c not in _META and not c.endswith(_EXCLUDE) and pd.api.types.is_numeric_dtype(df[c])
    ]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/processed/multiembed_descriptors/ablation_features.csv")
    ap.add_argument("--spectral", default=None)
    ap.add_argument("--out-dir", default="reports/diagnostics/udio_wall")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    df["track_id"] = df["track_id"].astype(str)
    if "y" not in df.columns and "label" in df.columns:
        df["y"] = (df["label"].astype(str) == "fake").astype(int)
    if args.spectral:
        sp = pd.read_csv(args.spectral)
        sp["track_id"] = sp["track_id"].astype(str)
        spec_cols = [c for c in sp.columns if c.startswith("spec_")]
        df = df.merge(sp[["track_id"] + spec_cols], on="track_id", how="left", validate="m:1")

    feats = _numeric_feats(df)
    # restrict to fully-present features to keep the covariance well-conditioned
    present = [c for c in feats if df[c].notna().mean() > 0.98]
    logger.info("Using %d fully-present numeric features (of %d).", len(present), len(feats))

    groups = sorted(df.loc[df["y"] == 1, "algorithm"].dropna().unique().tolist())
    reals = df[df["y"] == 0]
    med = reals[present].median()

    # --- 1. distance-to-real-manifold per group -----------------------------
    scaler = StandardScaler().fit(reals[present].fillna(med))
    cov = LedoitWolf().fit(scaler.transform(reals[present].fillna(med)))
    logger.info("\n=== 1. MAHALANOBIS DISTANCE TO REAL MANIFOLD (higher = more anomalous) ===")
    dist_rows = []
    real_d = cov.mahalanobis(scaler.transform(reals[present].fillna(med)))
    logger.info("  %-22s mean=%8.1f  median=%8.1f  (n=%d)", "REAL", real_d.mean(), np.median(real_d), len(real_d))
    dist_rows.append(
        {"group": "REAL", "mean_maha": float(real_d.mean()), "median_maha": float(np.median(real_d)), "n": len(real_d)}
    )
    for g in groups:
        gr = df[(df["y"] == 1) & (df["algorithm"] == g)]
        d = cov.mahalanobis(scaler.transform(gr[present].fillna(med)))
        logger.info("  %-22s mean=%8.1f  median=%8.1f  (n=%d)", g, d.mean(), np.median(d), len(d))
        dist_rows.append({"group": g, "mean_maha": float(d.mean()), "median_maha": float(np.median(d)), "n": len(d)})
    pd.DataFrame(dist_rows).to_csv(out / "distance_to_real.csv", index=False)

    # --- 2. udio-120s vs udio-30s direct contrast ---------------------------
    u120 = df[df["algorithm"] == "udio-120s"]
    u30 = df[df["algorithm"] == "udio-30s"]
    if len(u120) and len(u30):
        logger.info("\n=== 2. udio-120s vs udio-30s: top |Cohen's d| features ===")
        ds = [
            {"feature": c, "cohens_d_120_minus_30": _cohens_d(u120[c].to_numpy(float), u30[c].to_numpy(float))}
            for c in present
        ]
        cd = pd.DataFrame(ds).dropna()
        cd = cd.sort_values("cohens_d_120_minus_30", key=lambda s: s.abs(), ascending=False)
        for _, r in cd.head(15).iterrows():
            logger.info("  %-34s d=%+.3f", r["feature"], r["cohens_d_120_minus_30"])
        cd.to_csv(out / "udio120_vs_udio30_cohensd.csv", index=False)

    # --- 3. per-group real-vs-fake univariate separability ------------------
    logger.info("\n=== 3. best single-feature AUC (real vs each group) ===")
    sep_rows = []
    for g in groups:
        gr = df[(df["y"] == 1) & (df["algorithm"] == g)]
        sub = pd.concat([reals, gr])
        ylab = sub["y"].to_numpy()
        best_auc, best_f = 0.5, None
        for c in present:
            x = sub[c].to_numpy(float)
            m = np.isfinite(x)
            if m.sum() < 50 or len(np.unique(ylab[m])) < 2:
                continue
            auc = roc_auc_score(ylab[m], x[m])
            auc = max(auc, 1 - auc)
            if auc > best_auc:
                best_auc, best_f = auc, c
        logger.info("  %-22s best=%.3f  via %s", g, best_auc, best_f)
        sep_rows.append({"group": g, "best_single_auc": best_auc, "best_feature": best_f})
    pd.DataFrame(sep_rows).to_csv(out / "best_single_feature_auc.csv", index=False)

    # --- 4. duration profile per group --------------------------------------
    dur_col = next(
        (
            c
            for c in ("encodec_wd_n_windows", "mert-95m_wd_n_windows", "xls-r_wd_n_windows", "encodec_n_embeddings")
            if c in df.columns
        ),
        None,
    )
    if dur_col:
        logger.info("\n=== 4. duration proxy (%s) per group ===", dur_col)
        durs = []
        for g in ["REAL", *groups]:
            sel = reals if g == "REAL" else df[df["algorithm"] == g]
            v = sel[dur_col].to_numpy(float)
            v = v[np.isfinite(v)]
            logger.info("  %-22s mean=%8.1f  median=%8.1f", g, v.mean(), np.median(v))
            durs.append({"group": g, "mean_dur_proxy": float(v.mean()), "median_dur_proxy": float(np.median(v))})
        pd.DataFrame(durs).to_csv(out / "duration_profile.csv", index=False)

    logger.info("\nWrote characterization to %s/", out)


if __name__ == "__main__":
    main()
