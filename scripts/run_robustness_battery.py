"""Audio-manipulation robustness battery — directly extends MusicDET Table 6.

MusicDET (arXiv 2605.18072) Table 6 reports EER of a FIXED, already-trained
zero-shot detector under: pitch shifting, time stretching, equalization,
reverberation, white noise, and MP3/AAC/Opus @ 64kbps. We already cover MP3 at
32/64/128 kbps (run_bitrate_sweep.sh). This script covers the rest: pitch
shift, time stretch, EQ, reverb, white noise (scripts/apply_audio_manipulations.py)
plus AAC/Opus @ 64kbps (ffmpeg), against the SAME frozen, already-trained
SONICS flow used for the headline result — matching MusicDET's own protocol of
testing a fixed detector's degradation, not retraining per manipulation.

Uses the same 5000-track balanced subset as run_bitrate_sweep.sh for direct
comparability with that result and to keep runtime bounded.

Protocol notes (state BOTH in the paper):
  - Randomized parameters: pitch shift draws n_steps ~ U(-2,+2) semitones and
    time stretch draws rate ~ U(0.8,1.2) PER TRACK (seeded, reproducible),
    matching MusicDET's stated protocol. (--fixed-params restores the legacy
    single-point +2 / 1.1x behaviour; legacy rows are NOT comparable to
    MusicDET's Table 6.)
  - Cascade caveat: manipulations are applied to already-canonicalized
    (MP3-64k round-tripped) audio, so aac_64k/opus_64k rows measure an
    MP3->AAC / MP3->Opus CASCADE, not a single clean transcode — unlike
    MusicDET, whose model is clean-trained. Report accordingly.

Usage:
    python scripts/run_robustness_battery.py \\
        --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\
        --output-dir data/processed/robustness_battery \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_audio_manipulations import MANIPULATIONS, randomized_manipulation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

ENCODEC_SR = 24_000
# ENCODEC_FPS was removed deliberately. The frame rate is DERIVED from the
# extracted array by features/frontend.py; a module-level 75 is exactly the
# constant that produced this script's 0-row run over 32,960 tracks.

# MusicDET's own published Table 6 EER (%) for direct side-by-side reference.
# (Zero-shot MusicDET column; baseline/untransformed EER was 4.51%.)
MUSICDET_TABLE6_EER = {
    "pitch_shift": 44.73,
    "time_stretch": 2.44,
    "equalization": 6.04,
    "reverberation": 4.04,
    "white_noise": 44.11,
    "mp3_64k": 41.75,
    "aac_64k": 35.85,
    "opus_64k": 22.15,
}


def _codec_roundtrip(audio: np.ndarray, sr: int, codec: str, bitrate_kbps: int = 64) -> np.ndarray:
    """Round-trip audio through the given ffmpeg codec at bitrate_kbps."""
    import soundfile as sf

    codec_map = {"aac": "aac", "opus": "libopus"}
    ext_map = {"aac": "m4a", "opus": "opus"}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        mid = Path(td) / f"mid.{ext_map[codec]}"
        dst = Path(td) / "out.wav"
        sf.write(str(src), audio, sr)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:a",
                codec_map[codec],
                "-b:a",
                f"{bitrate_kbps}k",
                str(mid),
            ],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mid), "-ar", str(sr), "-ac", "1", str(dst)],
            check=True,
        )
        out, _ = sf.read(str(dst))
    return out.astype(np.float32)


from intrinsic_ai_music_detection.features.pooling import pool_windows as _pool_windows  # noqa: E402


def score_audio_training_free(audio: np.ndarray, sr: int, detector: str, operator: dict | None = None) -> dict | None:
    """Score one buffer with the training-free detector — no flow, no checkpoint.

    This is the arm the paper actually reports, so the robustness battery has to
    be able to run without a ``--flow-path``. The flow arms remain available as
    the ablation via :func:`score_audio`.
    """
    from intrinsic_ai_music_detection.features.comb_artifacts import comb_features

    try:
        # The battery MUST use the operator the paper reports. Calling
        # comb_features(audio, sr) with library defaults measured `hull` + 5 dB
        # clip — the WORST variant in the grid — so the first battery run scored
        # a detector we do not report (baseline AUC 0.8110 against the reported
        # 0.9225).
        feats = comb_features(audio, sr, **(operator or {}))
    except Exception as exc:
        logger.debug("comb feature extraction failed: %s", exc)
        return None
    if not feats:
        return None
    # Any calibrated score is formed from the SAME feature dict by the library's
    # single definition, so the battery can measure the detector the paper proposes
    # rather than only the raw comb. Before this, --training-free accepted two raw
    # columns and nothing else, which is why the paper's robustness row is the comb
    # arm's (ledger R27.44).
    from intrinsic_ai_music_detection.features.comb_artifacts import comb_calibrated_score

    try:
        score = comb_calibrated_score(feats, detector)
    except (KeyError, ValueError) as exc:
        logger.debug("cannot form %s: %s", detector, exc)
        return None
    if not np.isfinite(score):
        return None
    # anomaly_score is the column the rest of the battery aggregates on. For the
    # comb, a LARGER value means "more decoder-like", so it is already oriented
    # the way the flow's anomaly score is and needs no sign flip.
    return {"comb_score": float(score), "anomaly_score": float(score), "wf_n_windows": 1}


def score_audio(
    audio: np.ndarray, sr: int, extractor, flow, window_duration: float, hop_duration: float
) -> dict | None:
    from intrinsic_ai_music_detection.features.frontend import EmptyFrontEndOutput, windows_or_raise

    try:
        emb = extractor.extract(audio, sr)
        if emb is None or len(emb) == 0:
            return None
        emb = np.asarray(emb, dtype=np.float32)
    except Exception as exc:
        logger.debug("extraction failed: %s", exc)
        return None

    # Window and hop are derived from the array's own measured rate by
    # features/frontend.py — this function takes DURATIONS, never frame counts,
    # so there is no frame rate for a caller to get wrong. This script previously
    # assumed EnCodec's 75 fps, so a combprint front end (~1 profile per second)
    # was asked for 300 rows from a 7-row matrix, every track returned None, and
    # the run wrote "robustness_scores.csv (0 rows)" after 32,960 tracks before
    # dying on KeyError: 'manipulation'.
    seconds = max(len(audio) / max(sr, 1), 1e-6)
    try:
        pw = windows_or_raise(emb, seconds, window_duration, hop_duration, context="robustness battery")
    except EmptyFrontEndOutput as exc:
        logger.debug("%s", exc)
        return None
    lls = -flow.score_samples(pw)
    finite = lls[np.isfinite(lls)]
    if len(finite) < 1:
        return None
    return {"wf_mean": float(finite.mean()), "wf_n_windows": int(len(finite)), "anomaly_score": float(-finite.mean())}


def _resolve_extractor(flow, requested: str, device: str):
    """Deprecated shim — front-end resolution now lives in features/frontend.py.

    Kept only so any external caller keeps working; it delegates so the mismatch
    rule cannot diverge between this script, measure_efficiency.py and
    build_reconstruction_control.py, which is how three independent local fixes
    for the same bug happened.
    """
    from intrinsic_ai_music_detection.features.frontend import resolve_frontend

    return resolve_frontend(embedding=requested, flow=flow, device=device).extractor


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--flow-path",
        default=None,
        help=(
            "Frozen RealNVPOneClass checkpoint (the flow ABLATION arm). Required unless "
            "--training-free is given \u2014 the reported detector has no checkpoint."
        ),
    )
    parser.add_argument(
        "--training-free",
        default=None,
        choices=["comb_strength", "comb_stat_strength", "comb_priormax2_margin", "comb_priormax4_margin", "comb_priormax2_hmargin", "comb_priormax4_hmargin", "comb_priormax4_latmargin", "comb_priormax4_unionmargin",
        "comb_priormax4_lo20margin",
        "comb_priormax4_widemargin",
        "comb_priormax4_lo20widemargin"],
        help=(
            "Run the battery on the TRAINING-FREE detector instead of a flow. This is the "
            "arm the paper reports; the flow arms are the ablation. Mutually exclusive "
            "with --flow-path."
        ),
    )
    parser.add_argument("--subset-manifest", default="data/processed/bitrate_sweep_subset.csv")
    parser.add_argument("--output-dir", default="data/processed/robustness_battery")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wf-window-duration", type=float, default=4.0)
    parser.add_argument("--wf-hop-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument(
        "--manipulations",
        nargs="*",
        default=["pitch_shift", "time_stretch", "equalization", "reverberation", "white_noise", "aac_64k", "opus_64k"],
        help="Subset to re-run only specific rows (e.g. --manipulations pitch_shift time_stretch).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for per-track parameter draws.")
    # Operator selection, so the battery scores the SAME detector the paper reports.
    # Defaults are the measured-best variant (`median_db`), not the library defaults.
    parser.add_argument("--residual", dest="residual_op", default="median", choices=["hull", "median"])
    parser.add_argument("--average", default="db", choices=["db", "power"])
    parser.add_argument("--f-min", type=float, default=1000.0)
    parser.add_argument("--hull-clip-db", type=float, default=0.0)
    parser.add_argument("--smooth-bins", type=int, default=5)
    parser.add_argument("--span-duration", type=float, default=3.0)
    parser.add_argument(
        "--null-priors",
        type=int,
        default=0,
        help=(
            "Decoy prior sets, needed to build the proposed comb_priormax<M>_margin "
            "score from this battery's output (scripts/derive_prior_margin.py). "
            "Default 0 reproduces the comb-arm rows the paper already reports; pass "
            "24 to measure the PROPOSED detector's degradation instead."
        ),
    )
    parser.add_argument(
        "--fixed-params",
        action="store_true",
        default=False,
        help=(
            "LEGACY: use the old fixed manipulation parameters (+2 semitones, 1.1x stretch, "
            "global noise seed) instead of MusicDET's randomized U(-2,+2)/U(0.8,1.2) protocol. "
            "Only for reproducing pre-correction numbers — not comparable to their Table 6."
        ),
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help="Front end for scoring (must match the checkpoint; auto-detected from it if omitted).",
    )
    args = parser.parse_args()

    if bool(args.flow_path) == bool(args.training_free):
        raise SystemExit(
            "Pass exactly one of --flow-path or --training-free. The reported detector is "
            "training-free (no checkpoint exists for it); the flow arms are the ablation."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = Path(args.subset_manifest)
    if not subset_path.exists():
        logger.error(
            "%s not found. Run `bash scripts/run_bitrate_sweep.sh` first (it builds this subset), "
            "or pass --subset-manifest pointing at any canonical_manifest.csv-shaped CSV.",
            subset_path,
        )
        sys.exit(1)
    subset = pd.read_csv(subset_path, low_memory=False)
    logger.info("Loaded subset: %d tracks", len(subset))

    from intrinsic_ai_music_detection.data.audio_utils import load_audio
    from intrinsic_ai_music_detection.features.embeddings import get_extractor
    from intrinsic_ai_music_detection.models.evaluate import bootstrap_auc, bootstrap_eer
    from intrinsic_ai_music_detection.models.flow import RealNVPOneClass

    operator = {
        "residual_op": args.residual_op,
        "average": args.average,
        "f_min": args.f_min,
        "hull_clip_db": args.hull_clip_db,
        "smooth_bins": args.smooth_bins,
        "span_duration": args.span_duration,
        "null_priors": args.null_priors,
    }
    if args.training_free:
        # Emit exactly the harmonic order the requested score needs, and skip the
        # decoys when it does not use them -- comb_priormax*_hmargin is deterministic,
        # so null_priors=0 is both correct and much faster.
        _order = re.search(r"comb_priormax(\d+)_", args.training_free)
        if _order:
            operator["n_harm_grid"] = (int(_order.group(1)),)
            operator["surrogates"] = 0
            if args.training_free.endswith(("_hmargin", "_latmargin", "lo20margin", "widemargin", "lo20widemargin")):
                operator["null_priors"] = 0
            elif operator["null_priors"] <= 0:
                operator["null_priors"] = 24
        flow = None
        extractor = None
        load_sr = 16_000
        logger.info("training-free operator: %s", operator)
        logger.info("TRAINING-FREE mode: scoring with %r — no flow, no checkpoint.", args.training_free)
    else:
        logger.info("Loading frozen flow from %s ...", args.flow_path)
        flow = RealNVPOneClass.load(args.flow_path, device=args.device)
        extractor = _resolve_extractor(flow, getattr(args, "embedding", None), args.device)
        load_sr = getattr(extractor, "sample_rate", ENCODEC_SR)

    def _score(audio: np.ndarray, sr: int) -> dict | None:
        if args.training_free:
            return score_audio_training_free(audio, sr, args.training_free, operator)
        return score_audio(audio, sr, extractor, flow, args.wf_window_duration, args.wf_hop_duration)

    checkpoint_path = output_dir / "robustness_checkpoint.jsonl"
    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rows.append(row)
                done.add((row["track_id"], row["manipulation"]))
        logger.info("Resuming: %d (track, manipulation) pairs already checkpointed", len(done))
    ckpt_fh = open(checkpoint_path, "a")

    def _emit(row: dict) -> None:
        rows.append(row)
        ckpt_fh.write(json.dumps(row) + "\n")
        ckpt_fh.flush()

    n = len(subset)
    for i, row in enumerate(subset.itertuples(), start=1):
        track_id = str(row.track_id)
        path = getattr(row, "canonical_path", None)
        if not path or not Path(path).exists():
            continue
        try:
            audio, sr = load_audio(
                str(path),
                target_sr=load_sr,
                max_duration=args.max_duration,
            )
        except Exception as exc:
            logger.debug("[%d] load failed for %s: %s", i, track_id, exc)
            continue

        # baseline (untransformed) — needed once per track for the AUC comparison
        if (track_id, "baseline") not in done:
            s = _score(audio, sr)
            if s is not None:
                _emit(
                    {
                        "track_id": track_id,
                        "label": row.label,
                        "algorithm": getattr(row, "algorithm", ""),
                        "manipulation": "baseline",
                        **s,
                    }
                )

        for manip in args.manipulations:
            if (track_id, manip) in done:
                continue
            try:
                manip_params: dict = {}
                if manip in MANIPULATIONS:
                    if args.fixed_params:
                        manip_audio = MANIPULATIONS[manip](audio, sr)
                    else:
                        # Deterministic per-(track, manipulation) draw — reruns
                        # and resumes see the same parameters for the same track.
                        track_rng = np.random.default_rng(
                            (zlib.crc32(f"{track_id}:{manip}".encode()) + args.seed) % (2**32)
                        )
                        manip_audio, manip_params = randomized_manipulation(manip, audio, sr, track_rng)
                elif manip == "aac_64k":
                    manip_audio = _codec_roundtrip(audio, sr, "aac", 64)
                elif manip == "opus_64k":
                    manip_audio = _codec_roundtrip(audio, sr, "opus", 64)
                else:
                    logger.warning("Unknown manipulation: %s", manip)
                    continue
                s = _score(manip_audio, sr)
                if s is not None:
                    _emit(
                        {
                            "track_id": track_id,
                            "label": row.label,
                            "algorithm": getattr(row, "algorithm", ""),
                            "manipulation": manip,
                            "manip_params": manip_params or None,
                            **s,
                        }
                    )
            except Exception as exc:
                logger.warning("[%d] manipulation %s failed for %s: %s", i, manip, track_id, exc)

        if i % 100 == 0 or i == n:
            logger.info("[%d/%d] processed", i, n)

    ckpt_fh.close()

    df = pd.DataFrame(rows).drop_duplicates(subset=["track_id", "manipulation"], keep="last")
    if df.empty:
        raise SystemExit(
            "robustness battery produced ZERO rows — every track failed to score. "
            "Refusing to write an empty CSV (it previously did, then died 5 lines later "
            "with KeyError: 'manipulation', hiding the real cause). Most likely the front "
            "end's frame rate does not match the window: check --embedding against the "
            "checkpoint and --wf-window-duration against the clip length."
        )
    df.to_csv(output_dir / "robustness_scores.csv", index=False)
    logger.info("Saved robustness_scores.csv (%d rows)", len(df))

    # --- AUC/EER per manipulation (real vs all-fake, pooled) ---
    result_rows = []
    for manip in ["baseline"] + list(args.manipulations):
        sub = df[df["manipulation"] == manip]
        real = sub[sub["label"] == "real"]["anomaly_score"].dropna().to_numpy()
        fake = sub[sub["label"] == "fake"]["anomaly_score"].dropna().to_numpy()
        if len(real) < 5 or len(fake) < 5:
            continue
        y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
        s = np.r_[real, fake]
        auc_res = bootstrap_auc(y, s)
        eer_res = bootstrap_eer(y, s)
        mdet_eer = MUSICDET_TABLE6_EER.get(manip, float("nan"))
        result_rows.append(
            {
                "manipulation": manip,
                "n_real": len(real),
                "n_fake": len(fake),
                "our_auc": round(auc_res["auc"], 4),
                "our_eer_pct": round(eer_res["eer"] * 100, 2),
                "musicdet_table6_eer_pct": mdet_eer,
            }
        )
        logger.info(
            "  %-15s  our AUC=%.4f  our EER=%.2f%%   MusicDET Table6 EER=%.2f%%",
            manip,
            auc_res["auc"],
            eer_res["eer"] * 100,
            mdet_eer,
        )

    out_df = pd.DataFrame(result_rows)
    out_df.to_csv(output_dir / "robustness_auc_eer.csv", index=False)
    logger.info("Saved robustness_auc_eer.csv")
    logger.info("\n%s", out_df.to_string(index=False))


if __name__ == "__main__":
    main()
