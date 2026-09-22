"""Afchar & Hennequin's zero-shot detector (arXiv:2607.25530), implemented faithfully.

This replaces `scripts/eval_nmf_reconstruction.py`, which implemented a DIFFERENT
quantity and whose result (FMC 0.4998 / SONICS 0.2564, "NMF reconstruction error
inverts") is retracted. See §0.1 of RUNBOOK_2026-08-20_corrections.md.

What the paper actually does — quoted, because we got this wrong once
--------------------------------------------------------------------
    "The matrix X ∈ [0,1]^(n×d) a preprocessed dataset of n fakeprints of
     length d containing real and synthetic examples."

    "We propose to destroy the learned peak structures in W by convolving the
     atoms with a Gaussian kernel G ... we compare the reconstructions X̃ = HW
     and X̃' = H(W∗G), and define for all sample of index i, the error
     r_i = ‖X̃_i − X̃'_i‖₂"

    "given a small training collection of real examples, we take a quantile of a
     training set of errors (r_i), for instance, 95%"

Three things follow, and all three differ from what we had:

1. ``r_i`` is **not** a reconstruction error against the input. It is the norm of
   the difference between TWO RECONSTRUCTIONS — the same activations rendered
   through sharp atoms and through blurred atoms. It therefore measures **how
   much peak structure sample i's reconstruction contains**. A track whose
   activations load onto comb-like atoms loses a lot under blurring and scores
   high; a smooth track barely changes and scores low.

2. The basis is fitted on the **pooled, unlabelled** set — real and synthetic
   together. Not on reals. NMF is unsupervised, so this uses no labels; it is
   dictionary learning over whatever is in the corpus, and the comb atoms it
   discovers are what the score reads out.

3. Only **reals** set the operating threshold (a 95% quantile), which is why the
   method is zero-shot: no fake label is used at any point.

Our earlier version computed ‖x_i − H_i W‖ with W fitted on reals only.

Why the old result is retracted — two independent defects, both measured
------------------------------------------------------------------------
1. **Wrong quantity, wrong fit set.** Ours is a novelty score against the input
   with a reals-only basis; theirs is a peak-energy score from a pooled basis.
   On a symmetric synthetic fixture (both classes max-normalised, equal peak
   amplitudes, differing only in whether the peaks are evenly spaced), across
   8 seeds and a capacity grid of k ∈ {5,10,20,40} × n_fit ∈ {25…200}:

   | quantity | AUC |
   |---|---|
   | retracted ‖x − HW‖, reals-only basis | **0.28 – 0.44 — inverted in 32/32 cells** |
   | paper's ‖HW − H(W∗G)‖, pooled basis | **0.94 – 1.00 in 8/8 seeds** |

   The inversion is not a capacity artifact and does not depend on the atom
   count. A k-atom basis reconstructs a clean periodic comb *well* and irregular
   micro-texture *badly*, so the comb gets the LOWER novelty score. The quantity
   we scored is inverted by construction, and the reported 0.2564 is a property
   of that choice rather than of AI music.

2. **Degraded input descriptor.** The old arm consumed ``CombPrintExtractor``
   profiles: median-filter residual, resampled to 445 bins. On synthetic spectra
   with realistic 2-bin-wide teeth, that resample alone drops the peak score
   from **AUC 0.904 to 0.665** and ``comb_strength`` from **1.000 to 0.689**.

Both are pinned in ``tests/test_nmf_peak.py``. The fixture must be **symmetric**
between the classes — an early version gave the two classes different peak
amplitudes and produced the opposite conclusion, which is exactly the trap this
whole round exists to close.

On ``--n-bins``
---------------
445 is OURS, not theirs. RETRACTION 8 (2026-09-06): we recorded the published
fakeprint dimension as 445 where arXiv:2607.25530 sec 4.2 states 4458 --- a dropped
digit, consistent with their own settings (44100/16384 = 2.6917 Hz/bin over the
[3k, 15k] band = 4458.1). Everything built on "the published 445-bin descriptor"
was withdrawn. 445 is retained only as the low end of a resolution sweep, and the
measured curve says the published setting transfers to 16 kHz essentially intact
(-0.003 FMC, -0.000 SONICS at 4458); loss appears only below ~1500 bins.

Usage (EC2)
-----------
    python scripts/eval_nmf_peak_detector.py \\
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \\
        --per-stratum 600 --workers 6 --max-duration 9 --level-match \\
        --n-atoms 20 --blur-sigma 1 2 3 5 8 \\
        --out-dir reports/diagnostics/nmf_peak_fmc
"""

from __future__ import annotations

import argparse
import itertools
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
logger = logging.getLogger("eval_nmf_peak")

# NOT their published dimension. arXiv:2607.25530 sec 4.2 says "fakeprint vectors
# of length 4458"; we had transcribed 445. See RETRACTION 8 in the module
# docstring. This constant is now just the coarse end of the resolution sweep and
# must not be described in any write-up as the published descriptor.
FAKEPRINT_BINS = 445


def peak_energy_scores(
    X: np.ndarray,
    n_atoms: int = 20,
    blur_sigma: float = 3.0,
    seed: int = 42,
    fit_matrix: np.ndarray | None = None,
    dictionary: str = "nmf",
) -> np.ndarray:
    """r_i = ‖H_i W − H_i (W∗G)‖₂ for every row of ``X``.

    ``fit_matrix`` is the data the DICTIONARY is learned from. Default (None) is
    ``X`` itself — the paper's transductive protocol, where the pooled evaluation
    set is factorised without labels.

    Why the distinction matters for the zero-shot claim
    ---------------------------------------------------
    The paper factorises a matrix "containing real and synthetic examples". No
    labels are used, but the fakes ARE present when the dictionary is learned, so
    the protocol is *transductive*. A reviewer will ask whether the atoms encode
    the specific generators being scored. Passing ``fit_matrix`` lets us answer
    with measurements instead of an argument:

      * ``fit_on=pooled``   the paper's setting (fit_matrix = X)
      * ``fit_on=reals``    dictionary from REAL tracks only — no fake is ever
                            seen, at any stage. The strictest possible claim.
      * ``fit_on=external`` dictionary from another corpus entirely — inductive
                            transfer, and the hardest test.

    On ``dictionary`` — why a second learner exists
    -----------------------------------------------
    The transductive-boundary result is a claim about **what the dictionary is
    learned from**, not about NMF. If the inversion under a reals-only fit were
    peculiar to ``sklearn.NMF``'s objective, the claim would be about one
    implementation; a reviewer is entitled to ask. Repeating it under a second,
    genuinely different dictionary-learning objective answers that:

      * ``nmf``     Frobenius loss with non-negativity — the paper's setting.
                    **Bit-identical to the original code path.**
      * ``sparse``  L1-penalised sparse coding (``MiniBatchDictionaryLearning``,
                    non-negative dict and codes). A different objective and a
                    different sparsity regime, same atoms-and-activations form.
      * ``kmeans``  centroids as atoms — the clustering step K-SVD is built on —
                    with non-negative least-squares codes. The degenerate,
                    maximally-different case.

    Every branch returns the SAME statistic ``‖H(W − W∗G)‖₂``; only how ``W`` and
    ``H`` are obtained changes.
    """
    from scipy.ndimage import gaussian_filter1d

    if X.ndim != 2 or len(X) < n_atoms:
        raise ValueError(f"need at least {n_atoms} pooled samples, got shape {X.shape}")
    fit_on = X if fit_matrix is None else fit_matrix
    if fit_on.shape[1] != X.shape[1]:
        raise ValueError(
            f"dictionary was fitted in {fit_on.shape[1]} dimensions but the scored matrix "
            f"has {X.shape[1]}. Use a fixed --n-bins so both corpora share a grid."
        )
    if len(fit_on) < n_atoms:
        raise ValueError(f"need at least {n_atoms} samples to fit, got {len(fit_on)}")

    # A fakeprint is `clip(curve - hull, 0, None) / max`, i.e. already non-negative,
    # so the defensive clip below is almost always a full extra copy of the matrix
    # for nothing — 1.9 GB on the 32,960 x 7,167 FMC corpus, on a 15 GB box that
    # also holds X and the collected profiles. Copy only when it would change something.
    def _nonneg(mat: np.ndarray) -> np.ndarray:
        return mat if mat.min() >= 0.0 else np.clip(mat, 0.0, None)

    fit_pos = _nonneg(fit_on)
    x_pos = _nonneg(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dictionary == "nmf":
            from sklearn.decomposition import NMF

            model = NMF(n_components=n_atoms, init="nndsvda", random_state=seed, max_iter=600)
            model.fit(fit_pos)
            H = model.transform(x_pos)
            W = model.components_  # [k, d]
        elif dictionary == "sparse":
            from sklearn.decomposition import MiniBatchDictionaryLearning

            # `fit_algorithm` must be 'cd' and `transform_algorithm` 'lasso_cd':
            # sklearn refuses positive_code with any LARS-based coder
            # ("Positive constraint not supported for 'lars' coding method").
            # Non-negativity is not optional here — a fakeprint is a non-negative
            # residual and the blurred-atom statistic assumes additive atoms.
            model = MiniBatchDictionaryLearning(
                n_components=n_atoms,
                random_state=seed,
                max_iter=200,
                fit_algorithm="cd",
                positive_dict=True,
                positive_code=True,
                transform_algorithm="lasso_cd",
                transform_alpha=0.1,
            )
            model.fit(fit_pos)
            H = model.transform(x_pos)
            W = model.components_
        elif dictionary == "kmeans":
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_atoms, random_state=seed, n_init=4).fit(fit_pos)
            W = np.clip(km.cluster_centers_, 0.0, None)
            # Non-negative codes without a per-row NNLS solve: least squares onto
            # the atom basis, clipped. Exact enough for a peak-energy readout and
            # O(n) rather than O(n) QP solves — this matrix has 59k rows.
            H = np.clip(x_pos @ np.linalg.pinv(W), 0.0, None)
        else:
            raise ValueError(f"unknown dictionary learner {dictionary!r}")

    W_blur = gaussian_filter1d(W, sigma=blur_sigma, axis=1)  # atoms with peaks destroyed

    # The difference of the two reconstructions, per sample. Equivalent to
    # H @ (W - W_blur), computed that way because it is one matmul instead of two.
    return np.linalg.norm(H @ (W - W_blur), axis=1)


def _profile(job: dict) -> dict:
    """One track's fakeprint, via Afchar's operator."""
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
        if len(audio) < 16_384:
            row["error"] = f"clip too short for n_fft=16384: {len(audio)} samples"
            return row
        if job["level_match"]:
            audio = level_match(audio, sr)
        # Full resolution is kept; the 445-bin variant is derived from it below,
        # so both come from ONE decode of the same audio and the comparison
        # between them cannot be contaminated by anything else.
        _, fp = afchar_fakeprint(audio, sr, f_min=job["f_min"], f_max=job["f_max"], n_bins=None)
        row["profile"] = fp.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:200]
    return row


def _collect(df: pd.DataFrame, args, audio_column: str) -> pd.DataFrame:
    # Carry ANY extra manifest column through (genre, source, licence...).
    # Without this the FMA per-genre false-positive table cannot be built:
    # the score CSV simply has no genre column to group by.
    meta_cols = [c for c in ("track_id", "label", "algorithm") if c in df.columns]
    meta_cols += [c for c in args.carry_columns if c in df.columns and c not in meta_cols]
    jobs = [
        {
            "path": str(getattr(r, audio_column)),
            "meta": {c: getattr(r, c) for c in meta_cols},
            "max_duration": args.max_duration,
            "level_match": args.level_match,
            "f_min": args.f_min,
            "f_max": args.f_max,
        }
        for r in df.itertuples(index=False)
        if Path(str(getattr(r, audio_column))).exists()
    ]
    if not jobs:
        raise SystemExit(f"no files resolved from column {audio_column!r}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_profile, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(futures):
                logger.info("  %d/%d", i, len(futures))

    out = pd.DataFrame(rows)
    ok = out["profile"].notna() if "profile" in out.columns else pd.Series(False, index=out.index)
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise SystemExit(
            "no fakeprints extracted — refusing to write an empty result. "
            f"First error: {out.get('error', pd.Series(['(none)'])).dropna().head(1).tolist()}"
        )
    if n_ok < len(out):
        logger.warning("%d/%d tracks failed fakeprint extraction", len(out) - n_ok, len(out))

    # ⚠ `as_completed` yields futures in WORKER-COMPLETION order, which is
    # nondeterministic. Row order therefore changed between runs, and every
    # sampling that indexes into X — `--n-reals`, `--fit-real-frac` — drew a
    # DIFFERENT set of tracks each time despite a fixed --seed. Measured cost:
    # the 0%-contamination cell of the SONICS curve, an identical config, read
    # 0.3417 in one run and 0.3892 in the next.
    #
    # Sorting by track_id makes row order a deterministic function of the
    # manifest. Full-corpus results (pooled, external) were never affected — AUC
    # is a rank statistic over the same set of tracks — but every budgeted fit was.
    out = out[ok].reset_index(drop=True)
    if "track_id" in out.columns:
        out = out.sort_values("track_id", kind="mergesort").reset_index(drop=True)
    else:
        logger.warning(
            "no track_id column — row order is worker-completion order and any "
            "budgeted --n-reals/--fit-real-frac sampling will NOT be reproducible"
        )
    return out


def _evaluate(per_track: pd.DataFrame, scores: np.ndarray, tag: str) -> list[dict]:
    is_real = (per_track["label"] == "real").to_numpy()
    real_s = scores[is_real]
    report: list[dict] = []
    for alg in sorted(set(per_track.loc[~is_real, "algorithm"]) - {""}):
        fake_s = scores[(~is_real) & (per_track["algorithm"] == alg).to_numpy()]
        if len(fake_s) < 5 or len(real_s) < 5:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
        report.append(
            {
                "config": tag,
                "algorithm": alg,
                "auc_signed": round(auc, 4),
                "auc_abs": round(max(auc, 1 - auc), 4),
                "eer_pct": round(eer * 100, 2),
                "real_median": round(float(np.median(real_s)), 5),
                "fake_median": round(float(np.median(fake_s)), 5),
            }
        )
    return report


def _operating_point(per_track: pd.DataFrame, scores: np.ndarray, fpr: float, seed: int) -> dict:
    """Their decision rule: threshold at a quantile of HELD-OUT real errors.

    The reals are split so that no real used to set the threshold is also scored
    against it — otherwise the reported FPR is optimistic by construction.
    """
    rng = np.random.RandomState(seed)
    is_real = (per_track["label"] == "real").to_numpy()
    real_idx = np.flatnonzero(is_real)
    rng.shuffle(real_idx)
    half = len(real_idx) // 2
    thr_idx, eval_idx = real_idx[:half], real_idx[half:]
    if len(thr_idx) < 20 or len(eval_idx) < 20:
        return {"tpr_at_fpr": float("nan"), "threshold": float("nan"), "measured_fpr": float("nan")}

    thr = float(np.quantile(scores[thr_idx], 1.0 - fpr))
    fake = scores[~is_real]
    return {
        "target_fpr": fpr,
        "threshold": round(thr, 6),
        "measured_fpr": round(float((scores[eval_idx] > thr).mean()), 4),
        "tpr_at_fpr": round(float((fake > thr).mean()), 4),
        "n_threshold_reals": len(thr_idx),
        "n_eval_reals": len(eval_idx),
    }


def _fit_matrix_for(args, X, is_real, ext, budget, seed) -> tuple[np.ndarray | None, str, np.ndarray | None]:
    """The matrix the dictionary is learned from, per --fit-on. Never uses a label
    of a FAKE; `is_real` only ever selects the real subset, which is information a
    deployer has by construction (they know their own catalogue is real).

    Returns ``(fit_matrix, tag, fit_rows)``. ``fit_rows`` are the indices *into X*
    that went into the dictionary, or None when the dictionary came from elsewhere.

    Why ``fit_rows`` exists — the leak it lets us close
    ---------------------------------------------------
    ``r_i`` is read off a reconstruction through atoms the dictionary learned. A
    track that was IN the fit set is reconstructed through atoms it helped create,
    so its ``r_i`` is inflated relative to a track that was not. When the fit reals
    are also the scored reals, real music gets a systematically higher score and the
    AUC **inverts** — for reasons that have nothing to do with AI music.

    Measured on FakeMusicCaps, dictionary of 600 reals, everything else identical:

    | scored set | fit reals as a share of scored reals | macro AUC |
    |---|---|---|
    | 3,600 (600/stratum) | **100%** | **0.1679 — "inverted"** |
    | 32,960 (full corpus) | 11% | **0.5690** |

    That is the same train-on-test defect caught in ``score_flow_terms.py`` (R2.3),
    one level down. ``--holdout-reals`` excludes the fit rows from evaluation, which
    is also the only setting a deployer can actually be in: you fit on the catalogue
    you have and score tracks you have not seen.

    ``mixture`` is the exception, and deliberately so: it builds a *simulated
    catalogue* holding a known fraction of unlabelled generated audio. Labels are
    used to CONSTRUCT the corpus, never by the method — the dictionary is still
    learned from an unlabelled mixture, exactly as the paper's protocol does. It
    answers the deployer's question that pooled-vs-reals-only cannot: *how much AI
    music must already be sitting unlabelled in your catalogue before this works?*
    """
    if args.fit_on == "pooled":
        # The paper's transductive protocol: the dictionary sees everything it
        # scores. The leak IS the finding here, so fit_rows is deliberately None.
        return None, "", None
    if args.fit_on == "external":
        return ext, "_ext", None

    real_rows = np.flatnonzero(is_real)
    if args.fit_on == "mixture":
        rng = np.random.RandomState(seed)
        fake_rows = np.flatnonzero(~is_real)
        # --fit-real-frac exists because the contamination curve had the SAME leak
        # this module's docstring describes: the mixture fit set held every real,
        # and every real was then scored, so all ten points of the curve sat at 100%
        # overlap. At 0.5 the dictionary sees half the catalogue and --holdout-reals
        # scores the other half, which is the deployer's actual position.
        frac = float(getattr(args, "fit_real_frac", 1.0))
        n_real_fit = int(round(frac * len(real_rows)))
        picked_reals = (
            real_rows if n_real_fit >= len(real_rows) else rng.choice(real_rows, size=max(n_real_fit, 1), replace=False)
        )
        n_fake = int(round(float(budget) * len(fake_rows)))
        picked_fakes = rng.choice(fake_rows, size=n_fake, replace=False) if n_fake else np.empty(0, dtype=int)
        fit_rows = np.concatenate([picked_reals, picked_fakes]).astype(int)
        # The tag records the achieved contamination of the FIT SET, which is what
        # the curve's x-axis must be — not the requested fraction of the fake pool.
        achieved = n_fake / len(fit_rows)
        frac_tag = "" if frac >= 1.0 else f"_rf{frac:g}"
        return X[fit_rows], f"_mix{achieved * 100:.3g}pct{frac_tag}", fit_rows

    fit_rows = real_rows
    if budget is not None and budget < len(fit_rows):
        fit_rows = np.random.RandomState(seed).choice(fit_rows, size=int(budget), replace=False)
    return X[fit_rows], f"_n{len(fit_rows)}", fit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-column", default="canonical_path")
    parser.add_argument(
        "--carry-columns",
        nargs="*",
        default=["genre"],
        help="Manifest columns to copy into the per-track output, for grouped "
        "analyses such as the per-genre false-positive table.",
    )
    parser.add_argument(
        "--per-stratum", type=int, default=600, help="Tracks per label x algorithm. 0 = ALL (full-corpus)."
    )
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument(
        "--f-min",
        type=float,
        default=1000.0,
        help="Their 16 kHz band is [1k, 8k]; their 44.1 kHz band is [3k, 15k] (this paper) "
        "or [5k, 16k] (the ISMIR one).",
    )
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=20,
        help="Their published K ('the number of NMF components are set to 20').",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0, 8.0],
        help=(
            "Gaussian width used to destroy the atoms' peak structure. Swept here, but the "
            "value REPORTED for a corpus must be the one selected on the OTHER corpus — same "
            "discipline as --smooth-bins. Selecting it on the corpus you report is "
            "selection-on-test."
        ),
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        nargs="+",
        default=[445, 0],
        help=(
            "Fakeprint dimensions to evaluate. 4458 is their published length (chosen for "
            "44.1 kHz, where a bin is ~27 Hz); 0 means FULL resolution. In our 16 kHz band "
            "the 445-bin resample measurably smears 2-bin-wide teeth, so both are reported "
            "and the difference is itself a result about the published descriptor."
        ),
    )
    parser.add_argument(
        "--fake-frac",
        type=float,
        nargs="+",
        default=None,
        help=(
            "With --fit-on mixture: the fractions of the corpus's FAKE pool to mix into "
            "the dictionary's fit set (0 = reals only, 1 = every fake). Sweeping this "
            "traces the transductive boundary along the axis a deployer actually faces — "
            "how much unlabelled generated audio their catalogue already contains. "
            "Labels build the simulated corpus; the method never sees one."
        ),
    )
    parser.add_argument(
        "--fit-on",
        default="pooled",
        choices=["pooled", "reals", "external", "mixture"],
        help=(
            "What the NMF DICTIONARY is learned from. 'pooled' is the paper's "
            "transductive protocol (unlabelled real+synthetic). 'reals' never sees a fake "
            "at any stage — the strictest zero-shot claim. 'external' fits on "
            "--external-manifest and scores here (inductive transfer; needs a fixed "
            "--n-bins so the two corpora share a grid)."
        ),
    )
    parser.add_argument(
        "--external-per-stratum",
        type=int,
        default=None,
        help="Tracks per stratum for the EXTERNAL dictionary corpus (defaults to --per-stratum).",
    )
    parser.add_argument(
        "--external-manifest",
        default=None,
        help="Corpus to learn the dictionary from when --fit-on external.",
    )
    parser.add_argument(
        "--n-reals",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sweep the number of REAL tracks the dictionary/threshold get, to answer "
            "'how many reals are needed'. Only meaningful with --fit-on reals."
        ),
    )
    parser.add_argument(
        "--fit-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sweep the seed that draws the dictionary's fit set, reusing ONE extraction "
            "pass. Any budgeted fit (--n-reals, --fit-real-frac) depends on WHICH tracks "
            "are drawn, so a single seed reports a point estimate with unknown spread — "
            "on a synthetic fixture, changing only the draw moved the held-out AUC from "
            "0.343 to 0.667. Pass three or more seeds and report mean +/- SD. Costs one "
            "NMF fit per seed, not one corpus extraction per seed."
        ),
    )
    parser.add_argument(
        "--fit-real-frac",
        type=float,
        default=1.0,
        help=(
            "With --fit-on mixture: the fraction of the corpus's REAL tracks that go into "
            "the dictionary. Default 1.0 reproduces the paper's protocol, where every real "
            "is in the factorisation. Use 0.5 together with --holdout-reals to measure the "
            "contamination curve WITHOUT the dictionary-level train-on-test leak — half the "
            "catalogue fits, the other half is scored. Measured cost of the leak: 0.15 AUC "
            "at 45%% overlap on FakeMusicCaps."
        ),
    )
    parser.add_argument(
        "--holdout-reals",
        action="store_true",
        default=False,
        help=(
            "Exclude the tracks the DICTIONARY was fitted on from the evaluation. "
            "Without this, a real that was in the fit set is reconstructed through atoms "
            "it helped create, its r_i is inflated, and the AUC inverts for reasons that "
            "have nothing to do with AI music: FMC reals-only reads 0.1679 when the fit "
            "reals are 100%% of the scored reals and 0.5690 when they are 11%%. It is also "
            "the only protocol a deployer can be in — fit on the catalogue you have, score "
            "tracks you have not seen. No effect on --fit-on pooled (where the dictionary "
            "seeing the test set IS the finding) or external (disjoint by construction)."
        ),
    )
    parser.add_argument(
        "--dictionary",
        default="nmf",
        choices=["nmf", "sparse", "kmeans"],
        help=(
            "Which dictionary learner produces the atoms. 'nmf' is the paper's setting and "
            "is bit-identical to every published run. 'sparse' is L1-penalised non-negative "
            "sparse coding; 'kmeans' uses centroids as atoms (K-SVD's clustering step). The "
            "transductive-boundary result is a claim about what the dictionary is LEARNED "
            "FROM — repeating it under a second objective shows it is not an artefact of "
            "sklearn.NMF. The column name gains a suffix only for the non-default learners."
        ),
    )
    parser.add_argument("--fpr", type=float, default=0.05, help="Their example operating point (95% quantile).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "label" not in df.columns:
        raise SystemExit("manifest needs a 'label' column")
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")

    rng = np.random.RandomState(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], dropna=False):
        # 0 = use every track in the stratum (full-corpus run for the paper).
        n = len(grp) if args.per_stratum <= 0 else min(len(grp), args.per_stratum)
        parts.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
    df = pd.concat(parts, ignore_index=True)
    logger.info("sampled %d tracks", len(df))

    per_track = _collect(df, args, args.audio_column)
    X_full = np.stack(per_track["profile"].to_numpy(), axis=0).astype(np.float64)
    logger.info(
        "pooled fakeprint matrix: %s (real+synthetic together, UNLABELLED — this is the "
        "matrix the paper factorises)",
        X_full.shape,
    )

    is_real = (per_track["label"] == "real").to_numpy()

    ext_full = None
    if args.fit_on == "external":
        if not args.external_manifest:
            raise SystemExit("--fit-on external requires --external-manifest")
        ext_df = pd.read_csv(args.external_manifest, low_memory=False)
        if "algorithm" not in ext_df.columns:
            ext_df["algorithm"] = ""
        ext_df["algorithm"] = ext_df["algorithm"].fillna("")
        rng2 = np.random.RandomState(args.seed)
        parts2 = []
        # The external corpus gets its OWN budget. Reusing --per-stratum meant the
        # FMA run (per-stratum 3000) built an 18,000-track dictionary corpus.
        ext_budget = args.external_per_stratum or args.per_stratum
        for _, grp in ext_df.groupby(["label", "algorithm"], dropna=False):
            n = min(len(grp), ext_budget)
            parts2.append(grp.iloc[rng2.choice(len(grp), size=n, replace=False)])
        ext_pt = _collect(pd.concat(parts2, ignore_index=True), args, args.audio_column)
        ext_full = np.stack(ext_pt["profile"].to_numpy(), axis=0).astype(np.float64)
        logger.info("external dictionary corpus: %s", ext_full.shape)

    def _regrid(mat, n_bins):
        if not n_bins or n_bins == mat.shape[1]:
            return mat
        grid = np.linspace(0, mat.shape[1] - 1, n_bins)
        return np.stack([np.interp(grid, np.arange(mat.shape[1]), row) for row in mat], axis=0)

    report: list[dict] = []
    op_rows: list[dict] = []
    score_cols: dict[str, np.ndarray] = {}
    # `--n-reals` sweeps how many REAL tracks the dictionary is allowed, which is
    # the "how many reals do we need?" curve. None = use them all.
    real_budgets = args.n_reals or [None]
    # One extraction pass, N dictionary draws. A budgeted fit depends on WHICH
    # tracks are drawn, so a single seed is a point estimate with unknown spread.
    fit_seeds = args.fit_seeds or [args.seed]
    if args.fit_on == "mixture":
        if not args.fake_frac:
            raise SystemExit("--fit-on mixture requires --fake-frac (e.g. --fake-frac 0 0.01 0.05 0.1 0.25 0.5 1)")
        real_budgets = args.fake_frac  # the swept quantity is a fraction, not a count
    for n_bins in args.n_bins:
        X = _regrid(X_full, n_bins)
        for sigma in args.blur_sigma:
            for budget, fit_seed in itertools.product(real_budgets, fit_seeds):
                fit_matrix, budget_tag, fit_rows = _fit_matrix_for(
                    args,
                    X,
                    is_real,
                    _regrid(ext_full, n_bins) if ext_full is not None else None,
                    budget,
                    fit_seed,
                )
                scores = peak_energy_scores(
                    X,
                    n_atoms=args.n_atoms,
                    blur_sigma=sigma,
                    seed=fit_seed,
                    fit_matrix=fit_matrix,
                    dictionary=args.dictionary,
                )
                # The learner is in the tag ONLY when it is not the default, so
                # every column name from every previous run is unchanged and the
                # published numbers stay addressable by their existing names.
                dict_tag = "" if args.dictionary == "nmf" else f"_{args.dictionary}"
                hold_tag = "_heldout" if (args.holdout_reals and fit_rows is not None) else ""
                seed_tag = f"_s{fit_seed}" if len(fit_seeds) > 1 else ""
                tag = (
                    f"k{args.n_atoms}_sigma{sigma:g}_bins{X.shape[1]}"
                    f"_{args.fit_on}{budget_tag}{dict_tag}{hold_tag}{seed_tag}"
                )
                # Scores stay full-length in the CSV; only EVALUATION drops the rows
                # the dictionary was fitted on. A track reconstructed through atoms
                # it helped create has an inflated r_i, which inverted the FMC
                # reals-only AUC for reasons unrelated to AI music. See
                # _fit_matrix_for's docstring for the measurement.
                ev_track, ev_scores = per_track, scores
                if hold_tag:
                    keep = np.ones(len(scores), dtype=bool)
                    keep[fit_rows] = False
                    ev_track = per_track[keep].reset_index(drop=True)
                    ev_scores = scores[keep]
                    n_real_kept = int((ev_track["label"] == "real").sum())
                    n_fake_kept = int((ev_track["label"] != "real").sum())
                    if n_real_kept < 50:
                        raise SystemExit(
                            f"--holdout-reals left only {n_real_kept} real tracks to score "
                            f"for config {tag}. Lower --n-reals or drop the flag; an AUC on "
                            "this few reals is not worth reporting."
                        )
                    if n_fake_kept < 50:
                        # --fake-frac 1.0 under --holdout-reals puts EVERY fake in the
                        # dictionary and then holds them all out, so there is nothing left
                        # to detect. It silently produced an empty AUC row and a NaN TPR.
                        # Skip it loudly instead: the curve simply has no top point in the
                        # held-out protocol, and that is a property of the design.
                        logger.warning(
                            "%s: only %d fake tracks left after hold-out — this cell is "
                            "degenerate (every fake went into the dictionary) and is SKIPPED. "
                            "The held-out curve cannot have a 100%%-of-fakes point.",
                            tag,
                            n_fake_kept,
                        )
                        continue
                    logger.info(
                        "%s: held out %d fit rows; evaluating on %d tracks (%d real)",
                        tag,
                        len(fit_rows),
                        len(ev_track),
                        n_real_kept,
                    )
                score_cols[f"r_{tag}"] = scores
                new_rows = _evaluate(ev_track, ev_scores, tag)
                report.extend(new_rows)
                op_rows.append({"config": tag, **_operating_point(ev_track, ev_scores, args.fpr, args.seed)})

                # Persist after EVERY config. Three separate long runs — the FMC
                # held-out curve twice and the SONICS high half once — computed
                # every configuration and then wrote nothing, because the results
                # were only saved after the whole sweep finished. A sweep that
                # dies on its last cell must not cost the eleven before it.
                ctrl_i = "_lvl" if args.level_match else ""
                pd.DataFrame(report).to_csv(out_dir / f"nmf_peak_auc{ctrl_i}.csv", index=False)
                pd.DataFrame(op_rows).to_csv(out_dir / f"nmf_peak_operating_point{ctrl_i}.csv", index=False)

    if not report:
        if set(per_track["label"]) == {"real"}:
            # All-real catalogue (FMA): no AUC exists. Write the scores so
            # measure_false_positive_rate.py can apply a threshold set elsewhere.
            scored = per_track.drop(columns=["profile"]).copy()
            for k, v in score_cols.items():
                scored[k] = v
            ctrl = "_lvl" if args.level_match else ""
            path = out_dir / f"nmf_peak_per_track{ctrl}.csv"
            scored.to_csv(path, index=False)
            logger.info(
                "\nALL-REAL CORPUS (%d tracks): no AUC is defined. Scores written to %s — "
                "compute the false-positive rate with measure_false_positive_rate.py using a "
                "threshold set on the labelled corpus.",
                len(scored),
                path,
            )
            for k, v in score_cols.items():
                f = v[np.isfinite(v)]
                if len(f):
                    logger.info(
                        "  %-34s median %.4f  p90 %.4f  p99 %.4f",
                        k,
                        np.median(f),
                        np.percentile(f, 90),
                        np.percentile(f, 99),
                    )
            return
        raise SystemExit("no comparable real/fake pairs — check the manifest")

    scored = per_track.drop(columns=["profile"]).copy()
    for k, v in score_cols.items():
        scored[k] = v
    ctrl = "_lvl" if args.level_match else ""
    scored.to_csv(out_dir / f"nmf_peak_per_track{ctrl}.csv", index=False)

    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / f"nmf_peak_auc{ctrl}.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out_dir / f"nmf_peak_operating_point{ctrl}.csv", index=False)

    # auc_signed, not auc_abs: the ORIENTATION is the whole point of this rerun.
    pivot = rep.pivot_table(index="config", columns="algorithm", values="auc_signed")
    pivot["MACRO"] = pivot.mean(axis=1)
    logger.info(
        "\nNMF PEAK-ENERGY AUC (SIGNED — >0.5 means correctly oriented):\n%s",
        pivot.sort_values("MACRO", ascending=False).to_string(),
    )
    logger.info("\nOperating point at %.0f%% FPR:\n%s", args.fpr * 100, pd.DataFrame(op_rows).to_string(index=False))

    best = pivot["MACRO"].max()
    if best < 0.5:
        logger.warning(
            "Every blur width still gives macro AUC < 0.5 (best %.4f). The faithful "
            "implementation ALSO inverts, so the retraction in §0.1 concerns the mechanism "
            "and not just the code — report that explicitly rather than quietly.",
            best,
        )
    else:
        logger.info(
            "Correctly oriented (best macro %.4f). The earlier 'NMF inverts' result was a "
            "misimplementation and must be marked RETRACTED with both values.",
            best,
        )
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
