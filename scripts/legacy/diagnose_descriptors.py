#!/usr/bin/env python
"""Descriptor health + cross-generator sign-stability diagnostic.

Two questions, answered from the wide ablation feature table (no GPU, no cache):

  1. NaN audit  -- per embedding, what fraction of tracks actually have window
     descriptors, and is the missingness random or structured (by generator /
     by clip length)? Confirms the MERT descriptors-only/cache wipe.

  2. Sign stability -- for every geometric feature, does its real-vs-fake
     direction stay the SAME across all generators, or does it flip (the udio
     inversion)? Features whose sign is consistent across all held-out
     generators form a "robust subset" -- the honest cross-family detector.
     A leave-one-generator-out (LOGO) logistic-regression sanity check then
     compares full-feature vs robust-subset zero-shot AUC.

Usage:
    python scripts/diagnose_descriptors.py \
        --features data/processed/multiembed_descriptors/ablation_features.csv \
        --out-dir reports/diagnostics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose")

# columns that are diagnostics / confounds / metadata, never predictive features
_EXCLUDE_SUFFIXES = ("_n_windows", "_window_ids", "_n_embeddings")
_META_COLS = {"track_id", "label", "fake_label", "algorithm", "y"}
SIGN_EFFECT_THRESHOLD = 0.10  # min |z-mean-diff| (in SD) to count a direction as real


def _detect_embeddings(df: pd.DataFrame) -> list[str]:
    embs = []
    for emb in ("encodec", "mert-95m", "mert", "clap", "muq", "xls-r"):
        if f"{emb}_id_twonn_full" in df.columns:
            embs.append(emb)
    return embs


def _feature_cols(df: pd.DataFrame, emb: str, *, descriptors_only: bool) -> list[str]:
    """Numeric predictive columns for one embedding.

    ``emb == "spectral"`` is a pseudo-embedding: its columns carry the ``spec_``
    prefix and are treated as just another specialist by the leak-free LOGO
    machinery. The per-fold robust-subset selection then automatically keeps the
    generator-invariant comb-*presence* features and drops the generator-specific
    comb-*period* feature (``spec_comb_lag_hz``) without manual curation.
    """
    prefix = "spec_" if emb == "spectral" else f"{emb}_"
    cols = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        if any(c.endswith(s) for s in _EXCLUDE_SUFFIXES):
            continue
        if c in _META_COLS:
            continue
        if descriptors_only and "_wd_" not in c:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


# --------------------------------------------------------------------------- #
# 1. NaN audit
# --------------------------------------------------------------------------- #
def nan_audit(df: pd.DataFrame, embs: list[str], out_dir: Path) -> None:
    logger.info("\n=== 1. DESCRIPTOR NaN AUDIT ===")
    rows = []
    for emb in embs:
        wd_cols = [c for c in df.columns if c.startswith(f"{emb}_wd_") and not c.endswith(_EXCLUDE_SUFFIXES)]
        if not wd_cols:
            continue
        # a track "has descriptors" if its core descriptor is present
        probe = f"{emb}_wd_effrank_mean"
        probe = probe if probe in df.columns else wd_cols[0]
        present = df[probe].notna()
        frac = present.mean()
        logger.info("  %-10s descriptors present: %4d / %4d  (%.1f%%)", emb, present.sum(), len(df), 100 * frac)
        rows.append({"embedding": emb, "n_present": int(present.sum()), "n_total": len(df), "frac_present": frac})

        # is the missingness structured? break down by generator and by clip length
        if frac < 0.99:
            miss = ~present
            if "algorithm" in df.columns:
                by_alg = df.assign(_miss=miss).groupby(df["algorithm"].fillna("REAL"))["_miss"].mean()
                logger.info("    missing-rate by generator:")
                for alg, m in by_alg.sort_values(ascending=False).items():
                    logger.info("      %-22s %.1f%% missing", alg, 100 * m)
            nfld = f"{emb}_n_embeddings"
            if nfld in df.columns:
                q = pd.qcut(df[nfld], 4, duplicates="drop")
                by_len = df.assign(_miss=miss).groupby(q, observed=True)["_miss"].mean()
                logger.info("    missing-rate by frame-count quartile (random => flat ~equal):")
                for rng, m in by_len.items():
                    logger.info("      %-22s %.1f%% missing", str(rng), 100 * m)
    pd.DataFrame(rows).to_csv(out_dir / "descriptor_nan_audit.csv", index=False)


# --------------------------------------------------------------------------- #
# 2. Cross-generator sign stability
# --------------------------------------------------------------------------- #
def sign_stability(df: pd.DataFrame, embs: list[str], out_dir: Path) -> dict[str, list[str]]:
    logger.info("\n=== 2. CROSS-GENERATOR SIGN STABILITY ===")
    robust_subsets: dict[str, list[str]] = {}
    generators = sorted(df.loc[df["y"] == 1, "algorithm"].dropna().unique().tolist())
    logger.info("  generators: %s", generators)

    for emb in embs:
        feats = _feature_cols(df, emb, descriptors_only=False)
        if not feats:
            continue
        sub = df[["y", "algorithm", *feats]].copy()
        # median-impute + z-score over the whole set so effects are in SD units
        X = sub[feats].apply(lambda c: c.fillna(c.median()))
        Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=feats, index=sub.index)
        real_mask = sub["y"] == 0

        eff = {}  # feature -> {generator: signed effect}
        for g in generators:
            g_mask = (sub["y"] == 1) & (sub["algorithm"] == g)
            if g_mask.sum() < 20:
                continue
            eff[g] = Z[g_mask].mean() - Z[real_mask].mean()
        eff_df = pd.DataFrame(eff)
        if eff_df.empty:
            continue

        eff_arr = eff_df.to_numpy()
        signs = np.sign(eff_arr)
        strong = np.abs(eff_arr) >= SIGN_EFFECT_THRESHOLD
        # robust: same sign across ALL generators AND a real effect everywhere
        consistent = (np.all(signs == signs[:, [0]], axis=1)) & np.all(strong, axis=1)
        robust = eff_df.index[consistent].tolist()
        robust_subsets[emb] = robust

        # which generator is the odd-one-out (most sign flips vs majority)?
        # majority direction is PER FEATURE (across generators) -> shape (n_features,)
        majority = np.sign(np.sum(signs * strong, axis=1))
        flips = {g: int(np.sum((signs[:, j] != majority) & strong[:, j])) for j, g in enumerate(eff_df.columns)}

        logger.info(
            "\n  [%s] %d/%d features keep a consistent strong sign across all generators",
            emb,
            len(robust),
            len(feats),
        )
        logger.info("    sign-flip count vs majority (higher = more anomalous generator):")
        for g, n in sorted(flips.items(), key=lambda kv: -kv[1]):
            logger.info("      %-22s %3d flipped", g, n)

        report = eff_df.copy()
        report["consistent_sign"] = consistent
        report.to_csv(out_dir / f"sign_stability_{emb}.csv")
        Path(out_dir / f"robust_subset_{emb}.txt").write_text("\n".join(robust))
    return robust_subsets


# --------------------------------------------------------------------------- #
# LEAK-FREE helpers: robust subset is re-derived from TRAINING generators only
# --------------------------------------------------------------------------- #
def _robust_subset_for(
    frame: pd.DataFrame, emb: str, generators: list[str], *, feats: list[str] | None = None
) -> list[str]:
    """Features whose real-vs-fake sign is consistent AND strong across the GIVEN
    generators only. Pass *training* generators (held-out excluded) so feature
    selection never peeks at the test generator -- this is the leak-free subset.
    """
    feats = feats if feats is not None else _feature_cols(frame, emb, descriptors_only=False)
    if not feats:
        return []
    real_mask = frame["y"] == 0
    if real_mask.sum() < 20:
        return []
    z_arr = StandardScaler().fit_transform(frame[feats].apply(lambda c: c.fillna(c.median())))
    z = pd.DataFrame(z_arr, columns=feats, index=frame.index)
    eff = {}
    for g in generators:
        g_mask = (frame["y"] == 1) & (frame["algorithm"] == g)
        if g_mask.sum() < 20:
            continue
        eff[g] = z[g_mask].mean() - z[real_mask].mean()
    eff_df = pd.DataFrame(eff)
    if eff_df.shape[1] < 2:  # need >=2 train generators to judge consistency
        return []
    arr = eff_df.to_numpy()
    signs = np.sign(arr)
    strong = np.abs(arr) >= SIGN_EFFECT_THRESHOLD
    consistent = np.all(signs == signs[:, [0]], axis=1) & np.all(strong, axis=1)
    return eff_df.index[consistent].tolist()


def _fit_auc(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> float:
    """Standardised logistic regression trained on tr, AUC on te."""
    if not feats:
        return float("nan")
    med = tr[feats].median()
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(tr[feats].fillna(med))
    x_te = scaler.transform(te[feats].fillna(med))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, tr["y"])
    return roc_auc_score(te["y"], clf.predict_proba(x_te)[:, 1])


def _real_split(frame: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint train/test halves of the REAL rows (avoids real leakage)."""
    rng = np.random.default_rng(seed)
    real_idx = np.array(frame[frame["y"] == 0].index)
    rng.shuffle(real_idx)
    half = len(real_idx) // 2
    return real_idx[:half], real_idx[half:]


# --------------------------------------------------------------------------- #
# 3. LOGO: full features vs LEAK-FREE per-fold robust subset (per embedding)
# --------------------------------------------------------------------------- #
def logo_eval(df: pd.DataFrame, embs: list[str], out_dir: Path) -> None:
    logger.info("\n=== 3. LEAVE-ONE-GENERATOR-OUT AUC (full vs leak-free robust) ===")
    generators = sorted(df.loc[df["y"] == 1, "algorithm"].dropna().unique().tolist())
    results = []
    for emb in embs:
        full = _feature_cols(df, emb, descriptors_only=False)
        if not full:
            continue
        probe = f"{emb}_wd_effrank_mean"
        probe = probe if probe in df.columns else full[0]
        complete = df[df[probe].notna()].copy()
        if len(complete) < 200:
            logger.info("  [%s] skipped (only %d rows with descriptors)", emb, len(complete))
            continue
        real_train, real_test = _real_split(complete)
        for g in generators:
            g_rows = complete[(complete["y"] == 1) & (complete["algorithm"] == g)]
            other = complete[(complete["y"] == 1) & (complete["algorithm"] != g)]
            if len(g_rows) < 20 or len(other) < 20:
                continue
            tr = pd.concat([complete.loc[real_train], other])
            te = pd.concat([complete.loc[real_test], g_rows])
            train_gens = sorted(other["algorithm"].dropna().unique().tolist())
            robust = _robust_subset_for(tr, emb, train_gens, feats=full)  # held-out excluded
            row = {
                "embedding": emb,
                "held_out": g,
                "n_test_fake": len(g_rows),
                "n_robust": len(robust),
                "auc_full": _fit_auc(tr, te, full),
                "auc_robust": _fit_auc(tr, te, robust),
            }
            results.append(row)
            logger.info(
                "  [%s] held-out %-22s full=%.3f  robust(%d)=%.3f",
                emb,
                g,
                row["auc_full"],
                len(robust),
                row["auc_robust"],
            )
    if results:
        pd.DataFrame(results).to_csv(out_dir / "logo_full_vs_robust.csv", index=False)


# --------------------------------------------------------------------------- #
# 4. Cross-embedding pooled robust fusion (leak-free per fold)
# --------------------------------------------------------------------------- #
def logo_cross_embedding(df: pd.DataFrame, embs: list[str], out_dir: Path) -> None:
    logger.info("\n=== 4. CROSS-EMBEDDING ROBUST FUSION (pooled, leak-free per fold) ===")
    generators = sorted(df.loc[df["y"] == 1, "algorithm"].dropna().unique().tolist())
    real_train, real_test = _real_split(df)
    results = []
    for g in generators:
        g_rows = df[(df["y"] == 1) & (df["algorithm"] == g)]
        other = df[(df["y"] == 1) & (df["algorithm"] != g)]
        if len(g_rows) < 20 or len(other) < 20:
            continue
        tr = pd.concat([df.loc[real_train], other])
        te = pd.concat([df.loc[real_test], g_rows])
        train_gens = sorted(other["algorithm"].dropna().unique().tolist())
        pooled = [f for emb in embs for f in _robust_subset_for(tr, emb, train_gens)]
        if len(pooled) < 2:
            continue
        tr2 = tr[tr[pooled].notna().all(axis=1)]
        te2 = te[te[pooled].notna().all(axis=1)]
        if len(tr2) < 50 or te2["y"].nunique() < 2:
            continue
        auc = _fit_auc(tr2, te2, pooled)
        results.append({"held_out": g, "n_robust": len(pooled), "auc_fused_robust": auc})
        logger.info("  held-out %-22s fused-robust(%d)=%.3f", g, len(pooled), auc)
    if results:
        pd.DataFrame(results).to_csv(out_dir / "logo_cross_embedding_robust.csv", index=False)


# --------------------------------------------------------------------------- #
# 5. Score-level ensemble of per-embedding robust specialists (leak-free)
# --------------------------------------------------------------------------- #
def logo_score_ensemble(df: pd.DataFrame, embs: list[str], out_dir: Path) -> None:
    logger.info("\n=== 5. SCORE-LEVEL ENSEMBLE of per-embedding robust specialists (leak-free) ===")
    generators = sorted(df.loc[df["y"] == 1, "algorithm"].dropna().unique().tolist())
    real_train, real_test = _real_split(df)
    results = []
    for g in generators:
        g_rows = df[(df["y"] == 1) & (df["algorithm"] == g)]
        other = df[(df["y"] == 1) & (df["algorithm"] != g)]
        if len(g_rows) < 20 or len(other) < 20:
            continue
        tr = pd.concat([df.loc[real_train], other])
        te = pd.concat([df.loc[real_test], g_rows])
        train_gens = sorted(other["algorithm"].dropna().unique().tolist())
        probs = np.zeros(len(te))
        used = []
        for emb in embs:
            robust = _robust_subset_for(tr, emb, train_gens)
            if not robust:
                continue
            med = tr[robust].median()
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(tr[robust].fillna(med))
            x_te = scaler.transform(te[robust].fillna(med))
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(x_tr, tr["y"])
            probs += clf.predict_proba(x_te)[:, 1]
            used.append(emb)
        if not used:
            continue
        auc = roc_auc_score(te["y"], probs / len(used))
        results.append({"held_out": g, "n_embeddings": len(used), "auc_ensemble": auc})
        logger.info("  held-out %-22s ensemble(%s)=%.3f", g, "+".join(used), auc)
    if results:
        pd.DataFrame(results).to_csv(out_dir / "logo_score_ensemble.csv", index=False)


# --------------------------------------------------------------------------- #
# 6. LABEL-FREE one-class real-manifold detector on invariant features
#    (Ojha/SONICS-aligned: model the REAL distribution, flag deviations; never
#     sees a fake label at fit time -> the deployment-realistic setting)
# --------------------------------------------------------------------------- #
def logo_oneclass_invariant(df: pd.DataFrame, embs: list[str], out_dir: Path) -> None:
    from sklearn.covariance import LedoitWolf

    logger.info("\n=== 6. ONE-CLASS REAL-MANIFOLD on invariant features (LABEL-FREE) ===")
    generators = sorted(df.loc[df["y"] == 1, "algorithm"].dropna().unique().tolist())
    real_train, real_test = _real_split(df)
    results = []
    for g in generators:
        g_rows = df[(df["y"] == 1) & (df["algorithm"] == g)]
        other = df[(df["y"] == 1) & (df["algorithm"] != g)]
        if len(g_rows) < 20 or len(other) < 20:
            continue
        # invariant subset still selected from training generators only (uses labels
        # for SELECTION only; the detector itself is fit on reals alone).
        tr = pd.concat([df.loc[real_train], other])
        train_gens = sorted(other["algorithm"].dropna().unique().tolist())
        pooled = [f for emb in embs for f in _robust_subset_for(tr, emb, train_gens)]
        if len(pooled) < 2:
            continue
        real_tr = df.loc[real_train]
        real_tr = real_tr[real_tr[pooled].notna().all(axis=1)]
        te = pd.concat([df.loc[real_test], g_rows])
        te = te[te[pooled].notna().all(axis=1)]
        if len(real_tr) < 50 or te["y"].nunique() < 2:
            continue
        med = real_tr[pooled].median()
        scaler = StandardScaler().fit(real_tr[pooled].fillna(med))
        cov = LedoitWolf().fit(scaler.transform(real_tr[pooled].fillna(med)))
        scores = cov.mahalanobis(scaler.transform(te[pooled].fillna(med)))  # high = anomalous = fake
        auc = roc_auc_score(te["y"], scores)
        results.append({"held_out": g, "n_robust": len(pooled), "auc_oneclass": auc})
        logger.info("  held-out %-22s oneclass-maha(%d)=%.3f", g, len(pooled), auc)
    if results:
        pd.DataFrame(results).to_csv(out_dir / "logo_oneclass_invariant.csv", index=False)


def spectral_confound_check(df: pd.DataFrame, out_dir: Path) -> None:
    """Section 7: is any spec_ feature merely tracking clip duration?

    A deconvolution-comb artifact is architectural and must NOT scale with how
    long the clip is. We correlate every spec_ feature with a duration proxy
    (per-track window/embedding count). |rho| >= 0.5 is flagged as a possible
    duration confound rather than a genuine generation artifact.
    """
    from scipy.stats import spearmanr

    logger.info("\n=== 7. SPECTRAL CONFOUND CHECK (spec_* vs duration proxy) ===")
    spec_cols = _feature_cols(df, "spectral", descriptors_only=False)
    if not spec_cols:
        logger.info("  no spec_ columns found; skipping")
        return
    proxy = next(
        (
            c
            for c in (
                "encodec_wd_n_windows",
                "mert-95m_wd_n_windows",
                "mert_wd_n_windows",
                "xls-r_wd_n_windows",
                "encodec_n_embeddings",
                "mert-95m_n_embeddings",
            )
            if c in df.columns
        ),
        None,
    )
    if proxy is None:
        logger.info("  no duration proxy column found; skipping")
        return
    sub = df[df[spec_cols + [proxy]].notna().all(axis=1)]
    if len(sub) < 50:
        logger.info("  too few complete rows (%d); skipping", len(sub))
        return
    rows = []
    for c in spec_cols:
        rho, _ = spearmanr(sub[c], sub[proxy])
        rows.append({"feature": c, "spearman_vs_duration": float(rho)})
    rep = pd.DataFrame(rows).sort_values("spearman_vs_duration", key=lambda s: s.abs(), ascending=False)
    logger.info("  duration proxy = %s  (n=%d complete rows)", proxy, len(sub))
    for _, r in rep.iterrows():
        flag = "  <-- duration-confounded" if abs(r["spearman_vs_duration"]) >= 0.5 else ""
        logger.info("  %-28s rho=%+.3f%s", r["feature"], r["spearman_vs_duration"], flag)
    rep.to_csv(out_dir / "spectral_confound_vs_duration.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/processed/multiembed_descriptors/ablation_features.csv")
    ap.add_argument(
        "--spectral",
        default=None,
        help="Optional spectral_features.csv (spec_* per track_id) to merge and "
        "evaluate as a 'spectral' pseudo-embedding alongside the geometry bundles.",
    )
    ap.add_argument("--out-dir", default="reports/diagnostics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    if "y" not in df.columns and "label" in df.columns:
        df["y"] = (df["label"].astype(str) == "fake").astype(int)
    embs = _detect_embeddings(df)

    has_spectral = False
    if args.spectral:
        spec_df = pd.read_csv(args.spectral)
        spec_cols = [c for c in spec_df.columns if c.startswith("spec_")]
        if "track_id" not in spec_df.columns or not spec_cols:
            logger.warning("  --spectral file lacks track_id or spec_ columns; ignoring")
        else:
            spec_df = spec_df[["track_id"] + spec_cols].copy()
            spec_df["track_id"] = spec_df["track_id"].astype(str)
            df["track_id"] = df["track_id"].astype(str)
            df = df.merge(spec_df, on="track_id", how="left", validate="m:1")
            present = df[spec_cols[0]].notna().sum()
            logger.info("  merged %d spec_ features; present on %d/%d rows", len(spec_cols), present, len(df))
            has_spectral = present >= 50

    df = df.copy()  # de-fragment after read/merge to avoid PerformanceWarning
    embs_all = embs + (["spectral"] if has_spectral else [])
    logger.info("Loaded %d rows, embeddings: %s", len(df), embs_all)

    nan_audit(df, embs, out_dir)
    # Section 2 stays GLOBAL: it is the descriptive mechanism story (which features
    # are invariant across all generators), not a performance claim.
    sign_stability(df, embs_all, out_dir)
    # Sections 3-6 re-derive the robust subset PER FOLD from training generators
    # only, so the held-out generator never informs feature selection (leak-free).
    logo_eval(df, embs_all, out_dir)
    logo_cross_embedding(df, embs_all, out_dir)
    logo_score_ensemble(df, embs_all, out_dir)
    logo_oneclass_invariant(df, embs_all, out_dir)
    if has_spectral:
        spectral_confound_check(df, out_dir)
    logger.info("\nDiagnostics written to %s/", out_dir)


if __name__ == "__main__":
    main()
