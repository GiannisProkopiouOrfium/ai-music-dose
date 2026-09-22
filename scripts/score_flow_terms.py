"""Arm F2 — split the flow's log-likelihood into its two terms and score each alone.

The question
------------
Every one-class density in this project inverts: generated music gets a HIGHER
likelihood under a real-music flow than real music does. The stated mechanism is
that generated audio is simpler and more regular, so it sits nearer the mode.

But "nearer the mode" is a statement about ONE of the two terms:

    log p(x) = log p_Z(f(x))  +  log|det J_f(x)|
               \\____________/     \\______________/
                base density        local volume

* ``log p_Z(f(x))`` is the term that says "this latent is typical". That is
  exactly the quantity a simpler input should inflate.
* ``log|det J|`` is a **local volume** term — how much the flow has to contract
  around x. It is a local intrinsic-dimension statistic, and it is not obviously
  mis-oriented at all.

The project has only ever scored their SUM. Nobody looked at the terms
separately. If ``log|det J|`` alone is correctly oriented, there is a reals-only
flow detector hiding inside checkpoints we already have, and it connects directly
to the intrinsic-dimension arm (scripts/eval_phd_detector.py) — both would then be
measuring the same geometric property by different routes.

Cost: minutes. ``RealNVPOneClass.latents()`` already returns ``(z, logdet)``, and
the checkpoints exist.

Prediction, recorded in advance
-------------------------------
``log_pz`` inverts (AUC < 0.5) and carries the whole inversion; ``log_det_jac``
is closer to 0.5 or above. If BOTH invert equally, the log-det is not acting as a
volume/ID proxy here, and F2 closes the flow question rather than opening it.
Either outcome is reportable; the point of writing it down first is that only one
of them can be claimed afterwards.

Correctness check built in
--------------------------
``log_pz + log_det_jac`` must reproduce the checkpoint's own ``log_likelihood``
to floating point. The script asserts it and refuses to write results otherwise —
a decomposition that does not sum back is not a decomposition.

Usage (EC2)
-----------
    python scripts/score_flow_terms.py \\
        --flow-path data/processed/fmcraw_combflow/flow_combprint.pt \\
        --window-flow-csv data/processed/fmcraw_combflow/window_flow_eval.csv \\
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \\
        --embedding combprint --max-duration 9 \\
        --out-dir reports/diagnostics/flow_terms_fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
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
logger = logging.getLogger("flow_terms")

SUM_TOLERANCE = 1e-4


def decompose(flow, x: np.ndarray) -> dict[str, np.ndarray]:
    """Return ``log_pz``, ``log_det_jac`` and their sum for each row of ``x``.

    ``log p_Z`` is evaluated against the flow's own prior mean, so the two terms
    add back to the checkpoint's ``log_likelihood`` exactly.
    """
    z, logdet = flow.latents(x)
    # The prior is N(prior_mean, I) with a single scalar shift on every latent
    # dimension (RealNVPConfig.prior_mean, mirroring MusicDET's mu_real). Reading
    # it from the config rather than assuming 0 is what makes the sum check below
    # pass for prior-shifted checkpoints.
    prior_mean = float(getattr(flow.config, "prior_mean", 0.0))
    d = z.shape[1]
    log_pz = -0.5 * d * np.log(2.0 * np.pi) - 0.5 * ((z - prior_mean) ** 2).sum(axis=1)
    return {"log_pz": log_pz, "log_det_jac": logdet, "log_prob": log_pz + logdet}


def _verify(flow, x: np.ndarray) -> None:
    terms = decompose(flow, x)
    reference = np.asarray(flow.log_likelihood(x), dtype=np.float64)
    err = float(np.nanmax(np.abs(terms["log_prob"] - reference)))
    if not np.isfinite(err) or err > SUM_TOLERANCE:
        raise SystemExit(
            f"DECOMPOSITION DOES NOT SUM BACK: max |log_pz + log_det_jac - log_likelihood| "
            f"= {err:.3e} > {SUM_TOLERANCE}. Refusing to write results — a decomposition "
            "that does not reconstruct the original score is not measuring the original "
            "score's parts."
        )
    logger.info("decomposition verified: max reconstruction error %.3e", err)


def _refit_and_rescore(windows_by_track, loaded_flow, args) -> list[dict]:
    """Fit a fresh flow on HALF the reals; score the other half plus every fake.

    The saved checkpoints were fitted on every real window in the corpus, so
    scoring those same reals is train-on-test — their likelihood is inflated and
    the resulting AUC measures memorisation as much as detection. Splitting by a
    hash of ``track_id`` keeps the split deterministic and, crucially, at TRACK
    level: windows from one track must never straddle the split, or the leak
    comes back through near-duplicate neighbouring windows.
    """
    import hashlib

    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    def in_fit_half(track_id: str) -> bool:
        return int(hashlib.md5(track_id.encode()).hexdigest(), 16) % 2 == 0

    fit_windows = [pw for tid, label, _alg, pw in windows_by_track if label == "real" and in_fit_half(tid)]
    if not fit_windows:
        raise SystemExit("no real tracks landed in the fit half — cannot refit")
    x_fit = np.concatenate(fit_windows, axis=0)
    logger.info(
        "refitting on %d windows from %d held-in real tracks (the other reals are scored)",
        len(x_fit),
        len(fit_windows),
    )

    cfg = RealNVPConfig(
        n_coupling_layers=getattr(loaded_flow.config, "n_coupling_layers", 2),
        hidden_dim=getattr(loaded_flow.config, "hidden_dim", 128),
        n_epochs=getattr(loaded_flow.config, "n_epochs", 200),
        device=args.device,
        prior_mean=getattr(loaded_flow.config, "prior_mean", 0.0),
    )
    fresh = RealNVPOneClass(cfg)
    fresh.fit(x_fit)

    out: list[dict] = []
    n_skipped = 0
    for tid, label, alg, pw in windows_by_track:
        if label == "real" and in_fit_half(tid):
            n_skipped += 1
            continue  # in the fit half: scoring it would be the leak
        terms = decompose(fresh, pw)
        row = {"track_id": tid, "label": label, "algorithm": alg, "n_windows": int(len(pw))}
        for k, v in terms.items():
            fin = v[np.isfinite(v)]
            row[f"neg_{k}"] = float(-fin.mean()) if len(fin) else float("nan")
        out.append(row)
    logger.info(
        "scored %d tracks (%d fit-half reals excluded — this is the fix for the " "train-on-test leak)",
        len(out),
        n_skipped,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embedding", default=None, help="Must match the checkpoint.")
    parser.add_argument("--audio-column", default="canonical_path")
    parser.add_argument("--per-stratum", type=int, default=600)
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument("--window-duration", type=float, default=4.0)
    parser.add_argument("--hop-duration", type=float, default=2.0)
    parser.add_argument("--level-match", action="store_true", default=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--refit-holdout",
        dest="refit_holdout",
        action="store_true",
        default=True,
        help=(
            "Refit the flow on HALF the reals and score the other half plus all fakes "
            "(default). The saved checkpoints were fitted on every real window, so scoring "
            "them is train-on-test: reals get an inflated likelihood and the AUC is not a "
            "detector number."
        ),
    )
    parser.add_argument("--no-refit-holdout", dest="refit_holdout", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import librosa

    from intrinsic_ai_music_detection.data.audio_preprocessing import level_match
    from intrinsic_ai_music_detection.features.frontend import EmptyFrontEndOutput, resolve_frontend, windows_or_raise
    from intrinsic_ai_music_detection.models.flow import RealNVPOneClass

    flow = RealNVPOneClass.load(args.flow_path, device=args.device)
    front_end = resolve_frontend(embedding=args.embedding, flow=flow, device=args.device)
    logger.info("front end (from checkpoint): %s", front_end.name)

    if not args.refit_holdout:
        logger.warning(
            "TRAIN-ON-TEST LEAK: --no-refit-holdout scores reals the checkpoint was fitted "
            "on, which inflates their likelihood and therefore the AUC. The first run of "
            "this script reported FMC neg_log_prob 0.7415 this way, against 0.450 from the "
            "symmetric-fold protocol — the whole gap is the leak. Do not report these "
            "numbers as detector performance; they are only valid for comparing the two "
            "TERMS against each other, where the leak applies equally to both."
        )

    df = pd.read_csv(args.manifest, low_memory=False)
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")
    rng = np.random.RandomState(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], dropna=False):
        n = min(len(grp), args.per_stratum)
        parts.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
    df = pd.concat(parts, ignore_index=True)
    logger.info("sampled %d tracks", len(df))

    verified = False
    rows: list[dict] = []
    windows_by_track: list[tuple[str, str, str, np.ndarray]] = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        path = str(getattr(r, args.audio_column))
        if not Path(path).exists():
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
                warnings.filterwarnings("ignore", message="PySoundFile failed")
                audio, sr = librosa.load(path, sr=None, mono=True, duration=args.max_duration, res_type="soxr_hq")
            audio = np.asarray(audio, dtype=np.float32)
            if args.level_match:
                audio = level_match(audio, sr)
            emb = front_end.extract(audio, sr)
            pw = windows_or_raise(
                emb,
                max(len(audio) / sr, 1e-6),
                args.window_duration,
                args.hop_duration,
                context=f"flow terms {getattr(r, 'track_id', '')}",
            )
            if not verified:
                _verify(flow, pw)
                verified = True
            terms = decompose(flow, pw)
        except EmptyFrontEndOutput as exc:
            logger.debug("skip %s: %s", path, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("skip %s: %s", path, exc)
            continue

        row = {
            "track_id": getattr(r, "track_id", ""),
            "label": r.label,
            "algorithm": getattr(r, "algorithm", ""),
            "n_windows": int(len(pw)),
        }
        if args.refit_holdout:
            windows_by_track.append(
                (str(getattr(r, "track_id", "")), str(r.label), str(getattr(r, "algorithm", "")), pw)
            )
        for k, v in terms.items():
            fin = v[np.isfinite(v)]
            # Anomaly convention: HIGHER = more AI-like, so each term is negated,
            # exactly as score_samples negates the full log-likelihood.
            row[f"neg_{k}"] = float(-fin.mean()) if len(fin) else float("nan")
        rows.append(row)
        if i % 200 == 0:
            logger.info("  %d/%d", i, len(df))

    if not rows:
        raise SystemExit("no tracks scored — refusing to write an empty result")

    if args.refit_holdout:
        rows = _refit_and_rescore(windows_by_track, flow, args)

    per_track = pd.DataFrame(rows)
    ctrl = "_lvl" if args.level_match else ""
    per_track.to_csv(out_dir / f"flow_terms_per_track{ctrl}.csv", index=False)

    is_real = (per_track["label"] == "real").to_numpy()
    report: list[dict] = []
    for col in ("neg_log_pz", "neg_log_det_jac", "neg_log_prob"):
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
                    "term": col,
                    "algorithm": alg,
                    "auc_signed": round(auc, 4),
                    "eer_pct": round(eer * 100, 2),
                }
            )

    if not report:
        raise SystemExit("no comparable real/fake pairs — check the manifest")
    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / f"flow_terms_auc{ctrl}.csv", index=False)

    pivot = rep.pivot_table(index="term", columns="algorithm", values="auc_signed")
    pivot["MACRO"] = pivot.mean(axis=1)
    logger.info("\nFLOW LIKELIHOOD DECOMPOSITION (SIGNED AUC; <0.5 = inverted):\n%s", pivot.to_string())

    macro = pivot["MACRO"].to_dict()
    pz, jac, tot = macro.get("neg_log_pz"), macro.get("neg_log_det_jac"), macro.get("neg_log_prob")
    logger.info(
        "\nPREDICTION recorded in advance: log_pz inverts and carries the inversion; "
        "log_det_jac sits at or above 0.5.\n"
        "  base density  neg_log_pz      macro %.4f\n"
        "  volume term   neg_log_det_jac macro %.4f\n"
        "  their sum     neg_log_prob    macro %.4f",
        pz or float("nan"),
        jac or float("nan"),
        tot or float("nan"),
    )
    if jac is not None and pz is not None:
        if jac > 0.5 >= pz:
            logger.info(
                "PREDICTION HELD. The Jacobian term is correctly oriented while the base "
                "density is not — there is a reals-only flow detector here, and it is a "
                "local-volume statistic, which is the same quantity eval_phd_detector.py "
                "measures topologically. Gate it before reporting.",
            )
        else:
            logger.info(
                "PREDICTION DID NOT HOLD. Record it as such, with both numbers: the "
                "log-det is not acting as a volume/ID proxy on this data, and F2 closes "
                "the flow question rather than opening it.",
            )
    logger.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
