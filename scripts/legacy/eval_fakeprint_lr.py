"""Afchar et al.'s SUPERVISED baseline, and the transfer collapse it hides.

Why this script is the paper's central comparison
-------------------------------------------------
Their ISMIR 2025 headline (>99%) is a **supervised logistic regression on the
full fakeprint vector**, trained with fake labels on a random 80/20 split inside
one corpus. From ``train_test_regressor.py``::

    reg = LogisticRegression(class_weight="balanced")
    reg.fit(X, Y)                       # Y = 0 real, 1 synthetic

That is a strong detector and we reproduce it here rather than paraphrase it.
But the only unseen-generator number either Deezer paper reports collapses:

    ISMIR 2025 (this method)   Udio v32, unseen : 39.83%
    arXiv:2607.25530 (NMF)     Mubert / Mureka  : 5 – 48.5%

Our claim is not "we beat their AUC". It is that a **reals-only readout of the
same feature transfers where the supervised readout does not**, and that the
published zero-shot numbers on these benchmarks are measured on channel-labelled
corpora. This script produces the left-hand side of that comparison, under three
protocols:

  in_distribution   random 80/20 within a corpus — reproduces their setting.
                    If this does NOT land near their numbers, our fakeprint is
                    still wrong and no other row here means anything. It is the
                    diagnostic, not the result.
  logo              leave-one-generator-out: train on every generator but one,
                    test on the held-out one. Their Udio-v32 row, generalised.
  cross_corpus      fit on one corpus, score the other. The hardest and the
                    most honest.

Usage (EC2)
-----------
    python scripts/eval_fakeprint_lr.py \\
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \\
        --eval-manifest data/processed/canonical_sonics_chain/canonical_manifest.csv \\
        --per-stratum 600 --workers 6 --max-duration 9 --level-match \\
        --out-dir reports/diagnostics/fakeprint_lr_fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_fakeprint_lr")


def _profile(job: dict) -> dict:
    import librosa

    from intrinsic_ai_music_detection.data.audio_preprocessing import level_match
    from intrinsic_ai_music_detection.features.comb_artifacts import afchar_fakeprint

    row = dict(job["meta"])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            audio, sr = librosa.load(
                job["path"],
                sr=None,
                mono=True,
                duration=job["max_duration"],
                res_type="soxr_hq",
            )
        audio = np.asarray(audio, dtype=np.float32)
        if job["level_match"]:
            audio = level_match(audio, sr)
        _, fp = afchar_fakeprint(audio, sr, f_min=job["f_min"], f_max=job["f_max"], n_bins=job["n_bins"])
        row["profile"] = fp.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:200]
    return row


def _collect(manifest: str, args, tag: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(manifest, low_memory=False)
    if "label" not in df.columns:
        raise SystemExit(f"{manifest}: needs a 'label' column")
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")

    rng = np.random.RandomState(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], dropna=False):
        n = min(len(grp), args.per_stratum)
        parts.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
    df = pd.concat(parts, ignore_index=True)
    logger.info("[%s] sampled %d tracks", tag, len(df))

    meta_cols = [c for c in ("track_id", "label", "algorithm") if c in df.columns]
    jobs = [
        {
            "path": str(getattr(r, args.audio_column)),
            "meta": {c: getattr(r, c) for c in meta_cols},
            "max_duration": args.max_duration,
            "level_match": args.level_match,
            "f_min": args.f_min,
            "f_max": args.f_max,
            "n_bins": args.n_bins,
        }
        for r in df.itertuples(index=False)
        if Path(str(getattr(r, args.audio_column))).exists()
    ]
    if not jobs:
        raise SystemExit(f"[{tag}] no files resolved from column {args.audio_column!r}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_profile, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(futures):
                logger.info("  [%s] %d/%d", tag, i, len(futures))

    out = pd.DataFrame(rows)
    ok = out["profile"].notna() if "profile" in out.columns else pd.Series(False, index=out.index)
    if not ok.any():
        raise SystemExit(f"[{tag}] no fakeprints extracted — refusing to write an empty result")
    if int(ok.sum()) < len(out):
        logger.warning("[%s] %d/%d tracks failed", tag, len(out) - int(ok.sum()), len(out))
    out = out[ok].reset_index(drop=True)
    return out.drop(columns=["profile"]), np.stack(out["profile"].to_numpy(), axis=0).astype(np.float64)


def _fit_score(X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    """Their regressor, their settings, plus the metrics they do not report."""
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        reg.fit(X_tr, y_tr)

    prob = reg.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        # Their reported unit is per-class accuracy at the 0.5 threshold.
        "acc_real_pct": round(float((pred[y_te == 0] == 0).mean() * 100), 2) if (y_te == 0).any() else float("nan"),
        "acc_synth_pct": round(float((pred[y_te == 1] == 1).mean() * 100), 2) if (y_te == 1).any() else float("nan"),
    }
    if (y_te == 0).any() and (y_te == 1).any():
        auc, eer = auc_and_eer(y_te, prob)
        out["auc"] = round(auc, 4)
        out["eer_pct"] = round(eer * 100, 2)
    else:
        out["auc"] = float("nan")
        out["eer_pct"] = float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Corpus the regressor is FIT on.")
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Second corpus for the cross_corpus protocol. Omit to skip that protocol.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-column", default="canonical_path")
    parser.add_argument("--per-stratum", type=int, default=600)
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument("--f-min", type=float, default=1000.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=445,
        help="Their published fakeprint dimension. The LR path is where 445 is correct.",
    )
    parser.add_argument("--split-frac", type=float, default=0.8, help="Their split_fact.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, X = _collect(args.manifest, args, "fit")
    y = (meta["label"] != "real").astype(int).to_numpy()
    rng = np.random.RandomState(args.seed)
    rows: list[dict] = []

    # --- protocol 1: in-distribution, their random split -------------------
    # Split WITHIN each class, as create_train_test_split does per database.
    tr_mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        tr_mask[idx[: int(len(idx) * args.split_frac)]] = True
    rows.append(
        {
            "protocol": "in_distribution",
            "held_out": "(random 20%)",
            **_fit_score(X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask], args.seed),
        }
    )

    # --- protocol 2: leave-one-generator-out -------------------------------
    algs = sorted(set(meta.loc[y == 1, "algorithm"]) - {""})
    for held in algs:
        held_mask = (meta["algorithm"] == held).to_numpy()
        # Reals are split too, so the held-out set has both classes and the
        # real half is disjoint from training. Without that the LOGO "accuracy"
        # would be a one-class number and not comparable to the row above.
        real_idx = np.flatnonzero(y == 0)
        rng.shuffle(real_idx)
        real_test = np.zeros(len(y), dtype=bool)
        real_test[real_idx[: len(real_idx) // 5]] = True

        test = held_mask | real_test
        train = ~test & ~held_mask
        if train.sum() < 20 or test.sum() < 20:
            continue
        rows.append(
            {
                "protocol": "logo",
                "held_out": held,
                **_fit_score(X[train], y[train], X[test], y[test], args.seed),
            }
        )

    # --- protocol 3: cross-corpus ------------------------------------------
    if args.eval_manifest:
        meta2, X2 = _collect(args.eval_manifest, args, "eval")
        if X2.shape[1] != X.shape[1]:
            raise SystemExit(
                f"fakeprint dimension mismatch: fit corpus {X.shape[1]}, eval corpus "
                f"{X2.shape[1]}. Both must use the same --n-bins and band."
            )
        y2 = (meta2["label"] != "real").astype(int).to_numpy()
        rows.append(
            {
                "protocol": "cross_corpus",
                "held_out": Path(args.eval_manifest).parent.name,
                **_fit_score(X, y, X2, y2, args.seed),
            }
        )
        for held in sorted(set(meta2.loc[y2 == 1, "algorithm"]) - {""}):
            sel = ((meta2["algorithm"] == held) | (y2 == 0)).to_numpy()
            rows.append(
                {
                    "protocol": "cross_corpus_per_generator",
                    "held_out": held,
                    **_fit_score(X, y, X2[sel], y2[sel], args.seed),
                }
            )

    rep = pd.DataFrame(rows)
    ctrl = "_lvl" if args.level_match else ""
    rep.to_csv(out_dir / f"fakeprint_lr{ctrl}.csv", index=False)
    logger.info("\nAFCHAR SUPERVISED LOGISTIC REGRESSION ON FAKEPRINTS:\n%s", rep.to_string(index=False))

    ind = rep[rep["protocol"] == "in_distribution"]["acc_synth_pct"]
    if len(ind) and float(ind.iloc[0]) < 90.0:
        logger.warning(
            "In-distribution synthetic accuracy is %.1f%%, far below their reported >99%%. "
            "Treat every other row here as UNINTERPRETABLE until the descriptor is fixed: "
            "this protocol is the diagnostic for our fakeprint, not a result about theirs.",
            float(ind.iloc[0]),
        )

    logo = rep[rep["protocol"] == "logo"]
    if len(logo):
        logger.info(
            "\nTRANSFER COLLAPSE — the comparison the paper rests on.\n"
            "  in-distribution synthetic acc : %s%%\n"
            "  leave-one-generator-out       : min %.2f%% (%s), median %.2f%%\n"
            "Their own reported unseen-generator numbers: Udio v32 39.83%% (ISMIR 2025), "
            "Mubert/Mureka 5-48.5%% (arXiv:2607.25530).",
            ind.iloc[0] if len(ind) else "n/a",
            float(logo["acc_synth_pct"].min()),
            logo.loc[logo["acc_synth_pct"].idxmin(), "held_out"],
            float(logo["acc_synth_pct"].median()),
        )
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
