"""Zero-training detector from the deconvolution physics (arXiv:2506.19108).

Afchar et al. prove that transposed convolutions — the upsampling primitive in
every neural vocoder and codec decoder — imprint periodic spectral peaks, and
that the pattern follows from the *architecture's* strides, not from the training
data or the weights. That makes it the only artifact signal in the bibliography
with a principled reason to transfer across corpora.

Everything else we have measured is statistical and inherits the corpus: FLAC
complexity turned out to be mastering (0.52 on content-identical pairs), the
one-class likelihood is dominated by production identity, and a two-line energy
ratio reached AUC 1.0000 on SONICS purely because we collected the reals at
48 kHz.

This script fits nothing. It computes comb descriptors per track and reports
per-generator AUC, under the same controls as every other arm, so the number is
directly comparable to the flow's — and because there is no training, there is no
corpus to overfit.

Two axes are reported:
  * linear frequency — localises the comb spacing, which should cluster by
    decoder architecture rather than by corpus (that clustering is itself the
    evidence the mechanism is real);
  * log frequency — invariant to frequency scaling, per Dugelay et al.
    (arXiv:2607.27454), so it should survive pitch/speed manipulation.

Usage (EC2)
-----------
    python scripts/eval_comb_detector.py \
        --manifest data/processed/canonical_fmc_native/combined_manifest.csv \
        --per-stratum 600 --workers 6 --resample-hz 15000 --level-match \
        --max-duration 10 \
        --out-dir reports/diagnostics/comb_fmc
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

from intrinsic_ai_music_detection.features.comb_artifacts import comb_features  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_comb_detector")


def _one(job: dict) -> dict:
    import librosa

    from intrinsic_ai_music_detection.data.audio_preprocessing import bandwidth_match, level_match

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
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:200]
        return row

    audio = np.asarray(audio, dtype=np.float32)
    if job["level_match"]:
        audio = level_match(audio, sr)
    if job["resample_hz"] or job["lowpass_hz"]:
        audio = bandwidth_match(audio, sr, resample_hz=job["resample_hz"], lowpass_hz=job["lowpass_hz"])

    kw = {
        "smooth_bins": job["smooth_bins"],
        "residual_op": job["residual_op"],
        "average": job["average"],
        "f_min": job["f_min"],
        "hull_area": job["hull_area"],
        "hull_clip_db": job["hull_clip_db"],
        "span_duration": job["span_duration"],
        "n_harm_grid": job["n_harm_grid"],
        "surrogates": job["surrogates"],
        "min_spans": job["min_spans"],
        "null_priors": job["null_priors"],
    }
    try:
        row.update(comb_features(audio, sr, log_axis=False, **kw))
        row.update(comb_features(audio, sr, log_axis=True, **kw))
        # Every operator variant on IDENTICAL audio in one pass. Which envelope,
        # which averaging domain and whether to clip are empirical questions that
        # a synthetic fixture has now answered two different ways; this settles
        # them on the corpus instead. See OPERATOR_GRID.
        for name, over in job["grid"]:
            v = comb_features(audio, sr, log_axis=False, **{**kw, **over})
            row.update({f"{name}__{k}": val for k, val in v.items()})
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:200]
    return row


# (name, overrides). `legacy` is the operator that produced the 0.9067 headline.
#
# `null_priors: 0` on every variant: the decoy-prior null is emitted for the BASE
# operator only. It is 24 extra columns per harmonic order, and running it on all
# five variants as well would multiply the feature table by six for no gain — the
# question the null answers ("is the prior's advantage its size or its content?")
# is a property of the prior, not of the envelope operator.
OPERATOR_GRID = [
    ("legacy", {"residual_op": "median", "average": "power", "f_min": 500.0, "span_duration": 0.0, "null_priors": 0}),
    ("median_db", {"residual_op": "median", "average": "db", "span_duration": 0.0, "null_priors": 0}),
    (
        "hull_db_clip5",
        {"residual_op": "hull", "average": "db", "hull_clip_db": 5.0, "span_duration": 0.0, "null_priors": 0},
    ),
    (
        "hull_db_noclip",
        {"residual_op": "hull", "average": "db", "hull_clip_db": 0.0, "span_duration": 0.0, "null_priors": 0},
    ),
    (
        "hull_power_noclip",
        {"residual_op": "hull", "average": "power", "hull_clip_db": 0.0, "span_duration": 0.0, "null_priors": 0},
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-free deconvolution-comb detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=15000.0,
        help="Shared bandwidth ceiling. Pass 0 to disable (then the number is partly channel).",
    )
    parser.add_argument("--lowpass-hz", type=float, default=None)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument(
        "--smooth-bins",
        type=int,
        default=65,
        help=(
            "Width of the median filter that estimates the spectral envelope before the comb "
            "residual is taken. Must be much wider than a comb tooth and much narrower than the "
            "musical envelope; it is the one free parameter of the detector and it was NOT "
            "adjustable until now, which is why an earlier sweep produced three identical runs."
        ),
    )
    parser.add_argument(
        "--residual",
        dest="residual_op",
        default="hull",
        choices=["hull", "median"],
        help=(
            "Envelope operator. 'hull' is Afchar et al.'s lower hull (their actual operator, "
            "verified byte-for-byte in tests/test_afchar_parity.py). 'median' is the earlier "
            "local approximation, kept ONLY to reproduce pre-correction numbers — a median "
            "tracks the middle of the distribution, a lower hull tracks the noise floor "
            "between the teeth, and only the latter measures 'how far above the floor does "
            "this peak stand'."
        ),
    )
    parser.add_argument(
        "--average",
        default="db",
        choices=["db", "power"],
        help=(
            "Domain of the time average. 'db' matches their np.mean(10*log10(|STFT|^2)) — a "
            "geometric mean that suppresses loud transients and exposes the stationary floor "
            "where the comb lives. 'power' is the earlier arithmetic mean of power, dominated "
            "by the loudest frames. Kept only for the CORRECTED table."
        ),
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=1000.0,
        help=(
            "Low cutoff. Their 16 kHz setting is 1 kHz (44.1 kHz setting: 5 kHz). The earlier "
            "500 Hz floor admitted the band where musical energy dominates and no decoder "
            "comb lives."
        ),
    )
    parser.add_argument("--hull-area", type=int, default=10, help="Lower-hull window; theirs is 10.")
    parser.add_argument(
        "--hull-clip-db",
        type=float,
        default=0.0,
        help=(
            "Clip the hull residual at this many dB and max-normalise (Afchar's "
            "max_normalise, theirs is 5). 0 disables it. MEASURED: the clip is correct for "
            "their PROFILE readout (a linear classifier over 445 bins needs only the "
            "pattern) and WRONG for our SCALAR autocorrelation readout, which needs "
            "amplitude structure — with clipping, separation saturates at 0.049 however "
            "strong the comb. Default 0 here; the profile paths (NMF, LR, combprint) keep 5."
        ),
    )
    parser.add_argument(
        "--span-duration",
        type=float,
        default=4.0,
        help=(
            "Span length for the stationarity features (comb_stat_strength, "
            "comb_spacing_dispersion). 0 disables them. A decoder comb is stationary across "
            "spans; a musical harmonic comb is not — this is the only control that "
            "distinguishes them."
        ),
    )
    parser.add_argument(
        "--n-harm",
        type=int,
        nargs="*",
        default=[2, 4, 8],
        help=(
            "Harmonic-sum orders M to emit as comb_harm<M>_* columns. The readout "
            "averages the autocorrelation over the first M multiples of each candidate "
            "spacing, which divides the aperiodic floor by ~sqrt(M) while leaving a "
            "genuine comb alone. MEASURED on a symmetric fixture, weak-comb regime: "
            "comb_strength 0.53, M=2 0.59, M=4 0.69, M=8 0.82; strong combs saturate at "
            "1.0 under every M and an absent comb stays at chance under every M. "
            "DISCIPLINE: M is selected ONCE on FakeMusicCaps and then fixed for SONICS, "
            "as sigma and the bin count already are. Pass nothing to disable."
        ),
    )
    parser.add_argument(
        "--surrogates",
        type=int,
        default=20,
        help=(
            "Draws for the per-track surrogate null (comb_surrogate_z): comb_strength "
            "minus the mean of block-permuted versions of the track's OWN residual, over "
            "their SD. Calibrates each track against its own peak-height statistics "
            "rather than against the corpus, which is the asymmetry that makes udio's "
            "raw comb_strength (0.055) sit below real music's (0.072). 0 disables it; "
            "it costs ~4.7 ms per track per operator variant."
        ),
    )
    parser.add_argument(
        "--null-priors",
        type=int,
        default=24,
        help=(
            "Decoy prior sets for the comb_harm<M>_prior_strength null, emitted as "
            "comb_harm<M>_decoy<NN>_strength on the BASE operator only. Restricting the "
            "lag search to a handful of candidates lowers real music's score whatever "
            "the candidates are, because the max of a noisy autocorrelation over K lags "
            "grows like sqrt(2 ln K) -- and separately, our candidate list was built "
            "partly by working backwards from FakeMusicCaps' own measured spacings, "
            "which would be a leak. The decoys are the same cardinality at deliberately "
            "wrong frequencies: if they match the real prior's macro AUC the gain is "
            "candidate-set SIZE (content-free and reportable); if they collapse, the "
            "gain is the list's CONTENT and the FMC number is contaminated. 0 disables."
        ),
    )
    parser.add_argument(
        "--min-spans",
        type=int,
        default=8,
        help=(
            "Overlap the stationarity spans until a track yields at least this many. "
            "FMC is capped at 9 s, so a 4 s non-overlapping span gives TWO spans against "
            "SONICS's thirty and the stationarity ratio is starved. The hop is span/k for "
            "an integer k dividing the span, so spans[::k] reproduces the non-overlapping "
            "set bin-for-bin and comb_stat_strength / comb_spacing_dispersion keep the "
            "values they had. On SONICS k is 1 and nothing changes."
        ),
    )
    parser.add_argument(
        "--operator-grid",
        action="store_true",
        default=False,
        help=(
            "Also emit <variant>__comb_* columns for EVERY operator variant "
            "(legacy, median_db, hull_db_clip5, hull_db_noclip, hull_power_noclip) on "
            "IDENTICAL audio in one pass. This is how the envelope/averaging/clipping "
            "question gets settled on the corpus rather than on a synthetic fixture."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"]
    df["label"] = df["label"].astype(str)
    df["algorithm"] = df.get("algorithm", pd.Series([""] * len(df))).fillna("").astype(str)

    rng = np.random.default_rng(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], sort=True):
        # 0 = use every track in the stratum (full-corpus run for the paper).
        n = len(grp) if args.per_stratum <= 0 else min(args.per_stratum, len(grp))
        parts.append(grp.iloc[np.sort(rng.choice(len(grp), size=n, replace=False))])
    df = pd.concat(parts, ignore_index=True)
    logger.info("sampled %d tracks", len(df))

    resample_hz = args.resample_hz if args.resample_hz and args.resample_hz > 0 else None
    # Carry ANY extra manifest column through (genre, source, licence...).
    # Without this the FMA per-genre false-positive table cannot be built:
    # the score CSV simply has no genre column to group by.
    meta_cols = [c for c in ("track_id", "label", "algorithm") if c in df.columns]
    meta_cols += [c for c in args.carry_columns if c in df.columns and c not in meta_cols]
    jobs = [
        {
            "path": str(getattr(r, args.audio_column)),
            "meta": {c: getattr(r, c) for c in meta_cols},
            "max_duration": args.max_duration,
            "resample_hz": resample_hz,
            "lowpass_hz": args.lowpass_hz,
            "level_match": args.level_match,
            "smooth_bins": args.smooth_bins,
            "residual_op": args.residual_op,
            "average": args.average,
            "f_min": args.f_min,
            "hull_area": args.hull_area,
            "hull_clip_db": args.hull_clip_db,
            "span_duration": args.span_duration,
            "n_harm_grid": tuple(args.n_harm or ()),
            "surrogates": args.surrogates,
            "min_spans": args.min_spans,
            "null_priors": args.null_priors,
            "grid": OPERATOR_GRID if args.operator_grid else [],
        }
        for r in df.itertuples(index=False)
        if Path(str(getattr(r, args.audio_column))).exists()
    ]
    if not jobs:
        raise SystemExit(f"no files resolved from column {args.audio_column!r}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(futures):
                logger.info("  %d/%d", i, len(futures))

    per_track = pd.DataFrame(rows)
    ctrl = f"_rs{int(resample_hz)}" if resample_hz else ""
    ctrl += "_lvl" if args.level_match else ""
    ctrl += f"_sm{args.smooth_bins}" if args.smooth_bins != 65 else ""
    # The OPERATOR must appear in the filename. A corrected and an uncorrected run
    # differ only in flags, and writing both to comb_per_track_lvl_sm5.csv would
    # silently overwrite one with the other — the same failure mode the cache-key
    # module exists to prevent, one directory up.
    ctrl += f"_{args.residual_op}_{args.average}_f{int(args.f_min)}"
    ctrl += f"_clip{args.hull_clip_db:g}" if args.hull_clip_db else "_noclip"
    # The harmonic channels are strictly ADDITIVE — every pre-existing column keeps
    # the value it had — so the operator part of the name is honestly unchanged.
    # The suffix is still required: without it this run's CSV has the same name as
    # the published comb_fmc_FULL / comb_sonics_FULL_rs15k files, and pointing
    # --out-dir at one of those would overwrite a number that is in the paper.
    # Ledger L1/L2 are both instances of exactly this class of accident.
    if args.n_harm:
        ctrl += "_harm" + "".join(str(m) for m in args.n_harm)
    per_track.to_csv(out_dir / f"comb_per_track{ctrl}.csv", index=False)

    n_err = int(per_track.get("error", pd.Series(dtype=object)).notna().sum())
    if n_err:
        logger.warning("%d/%d tracks failed", n_err, len(per_track))

    feature_cols = [c for c in per_track.columns if ("comb_" in c) and per_track[c].notna().any()]
    is_real = (per_track["label"] == "real").to_numpy()
    report: list[dict] = []
    for col in feature_cols:
        s = per_track[col].to_numpy(float)
        real_s = s[is_real]
        real_s = real_s[np.isfinite(real_s)]
        for alg in sorted(set(per_track.loc[~is_real, "algorithm"]) - {""}):
            fake_s = s[(~is_real) & (per_track["algorithm"] == alg).to_numpy()]
            fake_s = fake_s[np.isfinite(fake_s)]
            if len(fake_s) < 5 or len(real_s) < 5:
                continue
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
            report.append(
                {
                    "feature": col,
                    "algorithm": alg,
                    # Both orientations: which class carries the comb is an
                    # empirical question, and a one-sided AUC would hide an
                    # inverted-but-strong feature.
                    "auc_signed": round(auc, 4),
                    "auc_abs": round(max(auc, 1 - auc), 4),
                    "eer_pct": round(eer * 100, 2),
                    "real_median": round(float(np.median(real_s)), 4),
                    "fake_median": round(float(np.median(fake_s)), 4),
                }
            )

    if not report:
        if set(per_track["label"]) == {"real"}:
            # An all-real corpus (FMA) has no AUC by construction — the number of
            # interest is the FALSE-POSITIVE RATE at a threshold set on another
            # corpus. Write the scores and say so, rather than exiting as if the
            # manifest were broken.
            logger.info(
                "\nALL-REAL CORPUS (%d tracks): no AUC is defined. The per-track scores are "
                "written above; compute the false-positive rate by applying the operating "
                "threshold chosen on the labelled corpus. Per-generator/genre breakdown is in "
                "the CSV.",
                len(per_track),
            )
            for col in ("comb_strength", "median_db__comb_strength"):
                if col in per_track.columns:
                    v = per_track[col].to_numpy(float)
                    v = v[np.isfinite(v)]
                    if len(v):
                        logger.info(
                            "  %-28s median %.4f  p90 %.4f  p99 %.4f",
                            col,
                            np.median(v),
                            np.percentile(v, 90),
                            np.percentile(v, 99),
                        )
            return
        raise SystemExit("no comparable real/fake pairs — check the manifest")
    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / f"comb_auc{ctrl}.csv", index=False)
    # ⚠ SIGNED is the headline. |AUC| lets a feature that is inverted on some
    # generators and correct on others look like a detector — but a zero-shot
    # detector cannot flip its sign per generator, because that would need labels.
    # This exact trap made SONICS `median_db__comb_strength` read 0.8468 (|AUC|)
    # when its honest zero-shot macro is 0.6203: udio inverts (0.2588 / 0.1751)
    # while chirp does not. Report signed; keep |AUC| only as a diagnostic of how
    # much information is present regardless of orientation.
    signed = rep.pivot_table(index="feature", columns="algorithm", values="auc_signed")
    signed["MACRO"] = signed.mean(axis=1)
    absolute = rep.pivot_table(index="feature", columns="algorithm", values="auc_abs")
    absolute["MACRO_ABS"] = absolute.mean(axis=1)

    logger.info(
        "\nCOMB-FEATURE AUC — SIGNED (this is the detector number), control%s:\n%s",
        ctrl or " NONE",
        signed.sort_values("MACRO", ascending=False).to_string(),
    )

    gen_cols = [c for c in signed.columns if c != "MACRO"]
    flips = {feat: [c for c in gen_cols if signed.loc[feat, c] < 0.5] for feat in signed.index}
    inconsistent = {f: g for f, g in flips.items() if 0 < len(g) < len(gen_cols)}
    if inconsistent:
        logger.warning(
            "\nSIGN-INCONSISTENT FEATURES — these are NOT usable zero-shot. A feature "
            "that is correct on some generators and inverted on others needs a per-generator "
            "sign, and choosing that sign requires labels:\n%s",
            "\n".join(
                f"  {f:42s} inverted on {', '.join(g)}"
                f"   (signed macro {signed.loc[f, 'MACRO']:.4f} vs |AUC| {absolute.loc[f, 'MACRO_ABS']:.4f})"
                for f, g in sorted(inconsistent.items(), key=lambda kv: -len(kv[1]))
            ),
        )
    logger.info(
        "\n|AUC| (diagnostic only — how much information is present regardless of "
        "orientation; do NOT report as detector performance):\n%s",
        absolute.sort_values("MACRO_ABS", ascending=False).to_string(),
    )
    pivot = signed

    # The interpretable check: spacing should cluster by DECODER, not by corpus.
    spacing = per_track[per_track["label"] != "real"].groupby("algorithm")["comb_spacing_hz"].median()
    if len(spacing):
        logger.info(
            "\nMedian comb spacing per generator (Hz) — the mechanistic check. If these "
            "cluster by decoder architecture rather than sitting at one corpus-wide value, "
            "the feature is measuring the generator and not the delivery chain:\n%s",
            spacing.to_string(),
        )
    real_spacing = per_track[per_track["label"] == "real"]["comb_spacing_hz"].median()
    logger.info("Real-music median spacing: %.1f Hz (expect no consistent comb)", real_spacing)


if __name__ == "__main__":
    main()
