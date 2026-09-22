"""DECISIVE CONFOUND CONTROL for the spectral-artifact detector.

Why this script exists
----------------------
``diagnose_descriptors.py --spectral`` shows the spectral (deconvolution-comb)
features separating real/fake at ~0.96-0.97 *cross-generator*, including
udio-120s where every embedding-geometry axis sat at ~0.40. That is either:

  (A) ARCHITECTURAL  -- the comb is the neural vocoder's strided-transposed-conv
      fakeprint (Afchar ISMIR 2025). It lives in the generated waveform itself,
      so it transfers across Suno/Udio and would generalise to new generators.
      LEGITIMATE.

  (B) CONFOUND       -- all SONICS fakes are ~8 kHz band-limited and upsampled
      from a low native rate; reals are full-band. The "comb" could merely be a
      band-limit / resampling-imaging signature of low-rate sources, not a
      generation artifact. Would NOT generalise to a full-band generator.
      A sophisticated bandwidth cheat.

Both hypotheses produce identical numbers on this corpus. This control breaks
the tie by giving the REAL canonical audio the fakes' *delivery history*
(band-limit to ~8 kHz + a 24k->16k->24k resampling round-trip, optional low-rate
MP3) and recomputing the spectral features.

Verdict
-------
  * If degraded-reals ACQUIRE the comb (comb_peak rises toward the fake mean and
    fake-vs-degraded-real AUC collapses toward 0.5) -> hypothesis (B), CONFOUND.
  * If degraded-reals STAY comb-free (comb_peak ~ real mean, fake-vs-degraded AUC
    stays high) -> hypothesis (A), ARCHITECTURAL. The detector is legitimate.

Run on EC2 (canonical WAVs live there):
    python scripts/control_recanonicalize_spectral.py \
        --manifest data/processed/canonical_sampler/canonical_manifest.csv \
        --spectral data/processed/multiembed_descriptors/spectral_features.csv \
        --out reports/diagnostics/control_recanonicalize.csv \
        --workers 4 --sample-reals 800
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.spectral_artifacts import (  # noqa: E402
    SpectralArtifactConfig,
    _resample,
    spectral_artifact_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("control_recanonicalize")

# features that carry the comb signal (presence-type; the robust discriminators)
_COMB_FEATURES = [
    "spec_comb_peak",
    "spec_residual_kurtosis",
    "spec_comb_peak_count",
    "spec_comb_ac_decay",
    "spec_residual_std",
]


def _fakeize(
    audio: np.ndarray,
    sr: int,
    lowpass_hz: float,
    intermediate_sr: int,
) -> np.ndarray:
    """Give a (full-band) real signal the fakes' delivery history.

    1. Brick-wall-ish low-pass to the fakes' band edge (~8 kHz).
    2. Downsample sr->intermediate_sr (anti-alias) then upsample back to sr.
       This reproduces *our* canonical 16k->24k upsampling that the band-limited
       fakes underwent -- using the same good resampler. If THAT step created the
       comb, the degraded reals will show it; if the comb predates our pipeline
       (i.e. the vocoder made it), they will not.
    """
    nyq = sr / 2.0
    wc = min(lowpass_hz, 0.99 * nyq)
    sos = sp_signal.butter(8, wc / nyq, btype="low", output="sos")
    y = sp_signal.sosfiltfilt(sos, audio).astype(np.float32)
    y = _resample(y, sr, intermediate_sr)
    y = _resample(y, intermediate_sr, sr)
    return y.astype(np.float32)


def _process_one(job: dict) -> dict:
    """Worker: recompute spec_* on the degraded version of one real track."""
    import librosa

    cfg: SpectralArtifactConfig = job["cfg"]
    base = {"track_id": job["track_id"]}
    try:
        audio, sr = librosa.load(
            job["path"], sr=cfg.analysis_sr, mono=True, duration=cfg.max_duration if cfg.max_duration else None
        )
        degraded = _fakeize(
            audio.astype(np.float32), int(sr), lowpass_hz=job["lowpass_hz"], intermediate_sr=job["intermediate_sr"]
        )
        feats = spectral_artifact_features(degraded, cfg.analysis_sr, cfg)
        if not feats:
            return {**base, "status": "error: empty"}
        return {**base, **{f"deg_{k}": v for k, v in feats.items()}, "status": "ok"}
    except Exception as exc:  # pragma: no cover
        return {**base, "status": f"error: {exc}"}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC of (pos > neg) for one feature (no sklearn dependency)."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/canonical_sampler/canonical_manifest.csv")
    ap.add_argument("--spectral", default="data/processed/multiembed_descriptors/spectral_features.csv")
    ap.add_argument("--out", default="reports/diagnostics/control_recanonicalize.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample-reals", type=int, default=800, help="0 = all reals.")
    ap.add_argument("--lowpass-hz", type=float, default=8000.0)
    ap.add_argument("--intermediate-sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = SpectralArtifactConfig()
    spec = pd.read_csv(args.spectral)
    spec["track_id"] = spec["track_id"].astype(str)
    if "y" not in spec.columns and "label" in spec.columns:
        spec["y"] = (spec["label"].astype(str) == "fake").astype(int)

    man = pd.read_csv(args.manifest, low_memory=False)
    man["track_id"] = man["track_id"].astype(str)
    label_col = "label" if "label" in man.columns else "fake_label"
    reals = man[man[label_col].astype(str).str.lower() != "fake"]
    path_col = "canonical_path"
    reals = reals[reals[path_col].astype(str).map(lambda p: bool(p) and Path(p).exists())]
    if args.sample_reals and len(reals) > args.sample_reals:
        reals = reals.sample(args.sample_reals, random_state=args.seed)
    logger.info(
        "Degrading %d real tracks (lowpass=%.0f Hz, via %d Hz)", len(reals), args.lowpass_hz, args.intermediate_sr
    )

    jobs = [
        {
            "track_id": r["track_id"],
            "path": str(r[path_col]),
            "cfg": cfg,
            "lowpass_hz": args.lowpass_hz,
            "intermediate_sr": args.intermediate_sr,
        }
        for _, r in reals.iterrows()
    ]
    rows: list[dict] = []
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, j): j["track_id"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            if not str(row.get("status", "")).startswith("ok"):
                n_err += 1
            if i % 200 == 0 or i == len(jobs):
                logger.info("  %d/%d (%d errors)", i, len(jobs), n_err)

    deg = pd.DataFrame(rows)
    deg = deg[deg["status"].astype(str) == "ok"].copy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deg.to_csv(out, index=False)

    # ---- comparison: real_orig vs real_degraded vs fake -------------------
    real_orig = spec[spec["y"] == 0]
    fake = spec[spec["y"] == 1]
    logger.info("\n=== RE-CANONICALIZATION CONTROL VERDICT ===")
    logger.info("  n: real_orig=%d  real_degraded=%d  fake=%d", len(real_orig), len(deg), len(fake))
    logger.info("  %-26s %10s %10s %10s | %8s %8s", "feature", "real_orig", "real_deg", "fake", "AUC_f|r", "AUC_f|deg")
    summary = []
    for f in _COMB_FEATURES:
        if f not in spec.columns or f"deg_{f}" not in deg.columns:
            continue
        ro = real_orig[f].to_numpy(dtype=float)
        fk = fake[f].to_numpy(dtype=float)
        dg = deg[f"deg_{f}"].to_numpy(dtype=float)
        auc_f_r = _auc(fk, ro)  # fake vs real_orig (baseline separation)
        auc_f_deg = _auc(fk, dg)  # fake vs degraded_real (the control)
        logger.info(
            "  %-26s %10.4f %10.4f %10.4f | %8.3f %8.3f",
            f,
            np.nanmean(ro),
            np.nanmean(dg),
            np.nanmean(fk),
            auc_f_r,
            auc_f_deg,
        )
        summary.append(
            {
                "feature": f,
                "mean_real_orig": float(np.nanmean(ro)),
                "mean_real_degraded": float(np.nanmean(dg)),
                "mean_fake": float(np.nanmean(fk)),
                "auc_fake_vs_realorig": auc_f_r,
                "auc_fake_vs_degraded": auc_f_deg,
            }
        )
    sm = pd.DataFrame(summary)
    sm.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)

    # ---- automated verdict on the primary comb feature --------------------
    if not sm.empty and "spec_comb_peak" in sm["feature"].values:
        r = sm[sm["feature"] == "spec_comb_peak"].iloc[0]
        gap_orig = r["mean_fake"] - r["mean_real_orig"]
        gap_after = r["mean_fake"] - r["mean_real_degraded"]
        retained = gap_after / gap_orig if abs(gap_orig) > 1e-9 else float("nan")
        logger.info(
            "\n  spec_comb_peak fake-vs-real gap: orig=%.4f  after-degrade=%.4f  (%.0f%% retained)",
            gap_orig,
            gap_after,
            100 * retained,
        )
        if retained >= 0.6:
            logger.info(
                "  VERDICT (comb features only): degrading reals did NOT make them acquire the "
                "comb (gap retained). So the comb-*presence* features are NOT a band-limit "
                "artifact. NOTE: this does NOT clear the spectral detector overall -- the strong "
                "cross-generator AUC came from band-edge SHAPE/ROLLOFF features (rolloff/hf_lf/"
                "slope), which this control does not test. Use the band-narrowing control "
                "(--band-hi-hz 6000) to expose those: if discrimination collapses inside the "
                "shared band, the win was a bandwidth confound."
            )
        elif retained <= 0.3:
            logger.info(
                "  VERDICT: CONFOUND -- degraded reals acquired the comb. The signal is a "
                "band-limit/resampling artifact, not a generation fakeprint."
            )
        else:
            logger.info(
                "  VERDICT: MIXED -- partial confound on comb features. Cross-check with the "
                "band-narrowing control and a full-band generator / raw-audio probe."
            )
    logger.info("\n  wrote %s and %s", out, out.with_name(out.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
