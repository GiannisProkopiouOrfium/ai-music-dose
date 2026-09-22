"""Reconstruction-based confound control (Afchar/MusicDET-appendix protocol).

This is the most decisive confound test in the paper: take the *same* real
audio track, reconstruct it through a neural codec (EnCodec 3/6/24 kbps) or a
Griffin–Lim mel-spectrogram inverter (GrifMel), and score both the original and
the reconstruction with the frozen SONICS-trained RealNVP flow.

Because original and reconstruction share identical musical content, genre, key,
tempo, and LUFS, any AUC difference cannot be attributed to genre or covariate
shift — it must reflect the **codec/vocoder decoder artifacts** that the flow
has learned to identify.

Expected outcome:
  - AUC(original vs EnCodec-reconstruction) > 0.70 for all bitrates =>
    flow detects codec artifacts independent of content.
  - AUC should *increase* for lower bitrates (more aggressive quantization
    => stronger artifacts).
  - GrifMel reconstructions (purely spectral, no neural codec) should show
    lower AUC, confirming the artifact is codec-quantization specific.

This directly parallels MusicDET Table A.2 (arXiv 2605.18072 Appendix A.2)
but on EnCodec latent space instead of STFT features.

Outputs
-------
  reconstruction_scores.csv    per-(track, reconstruction-type) anomaly scores
  reconstruction_auc.csv       AUC/EER per reconstruction type
  reconstruction_auc.png       ROC curves

Usage (EC2)
-----------
# Full run:
poetry run python scripts/build_reconstruction_control.py \\
    --real-dir data/processed/canonical_sonics_full/real \\
    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\
    --output-dir data/processed/reconstruction_control \\
    --n-tracks 500 \\
    --device cuda

# Quick pilot:
poetry run python scripts/build_reconstruction_control.py \\
    --real-dir data/processed/canonical_sonics_full/real \\
    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\
    --output-dir data/processed/reconstruction_control_pilot \\
    --n-tracks 50 \\
    --device cuda \\
    --recon-types encodec_3kbps encodec_6kbps
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.complexity import flac_bits_per_sec_array  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}
ENCODEC_FPS = 75
ENCODEC_SR = 24_000


# ---------------------------------------------------------------------------
# Reconstruction functions
# ---------------------------------------------------------------------------

# Module-level cache: loading EncodecModel from pretrained weights is expensive
# (~1-2s + disk/network I/O each time). Without caching, a run of 500 tracks x
# 3 bitrates would instantiate the model 1500 times — the dominant cost of the
# whole script. Cache one model instance per (bitrate, device) pair instead.
_ENCODEC_MODEL_CACHE: dict[tuple[float, str], object] = {}


def _get_encodec_model(bandwidth: float, device: str):
    """Return a cached EncodecModel configured for `bandwidth`, loading once."""
    key = (bandwidth, device)
    if key not in _ENCODEC_MODEL_CACHE:
        from encodec import EncodecModel

        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)
        model.eval()
        model.to(device)
        _ENCODEC_MODEL_CACHE[key] = model
        logger.info("Loaded EnCodec model for bandwidth=%.1f kbps on %s (cached for reuse)", bandwidth, device)
    return _ENCODEC_MODEL_CACHE[key]


def reconstruct_encodec(audio: np.ndarray, sr: int, bitrate_kbps: float, device: str) -> np.ndarray:
    """Round-trip audio through EnCodec at the specified bitrate.

    EnCodec quantizes the latent codes according to the target bitrate;
    the reconstructed waveform carries the quantization artifacts of that
    operating point without any change in musical content.

    Uses a module-level model cache (`_get_encodec_model`) so the pretrained
    weights are loaded once per bitrate, not once per (track, bitrate) call.
    """
    import torch
    from encodec.utils import convert_audio

    bandwidth_map = {1.5: 1.5, 3.0: 3.0, 6.0: 6.0, 12.0: 12.0, 24.0: 24.0}
    bw = bandwidth_map.get(bitrate_kbps, 6.0)
    model = _get_encodec_model(bw, device)

    wav_t = torch.from_numpy(audio).float()
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif wav_t.ndim == 2:
        wav_t = wav_t.unsqueeze(0)  # [1, C, T]

    # Resample if needed
    if sr != ENCODEC_SR:
        wav_t = convert_audio(wav_t, sr, ENCODEC_SR, 1)

    with torch.no_grad():
        encoded = model.encode(wav_t.to(device))
        decoded = model.decode(encoded)

    recon = decoded.squeeze().cpu().numpy()
    return recon.astype(np.float32)


def reconstruct_griffinlim(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 32,
) -> np.ndarray:
    """Round-trip through mel-spectrogram inversion (GrifMel).

    No neural codec is involved — this tests whether *spectral quantization*
    alone (mel-binning of frequencies) produces detectable artifacts.
    If the flow does NOT flag GrifMel reconstructions, it confirms the
    signal is specifically from neural codec quantization, not mere spectral
    imprecision.
    """
    import librosa

    # Forward: waveform -> mel -> magnitude
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    # Convert to amplitude (undo power)
    S_amp = np.sqrt(np.maximum(S, 1e-10))
    # Inverse: Griffin-Lim
    recon = librosa.griffinlim(
        librosa.feature.inverse.mel_to_stft(S_amp, sr=sr, n_fft=n_fft),
        n_iter=n_iter,
        hop_length=hop_length,
    )
    return recon.astype(np.float32)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _pool_windows(emb: np.ndarray, window_frames: int, hop_frames: int) -> np.ndarray:
    pooled = []
    n = len(emb)
    start = 0
    while start + window_frames <= n:
        sub = emb[start : start + window_frames]
        finite = sub[np.isfinite(sub).all(axis=1)]
        if len(finite) >= 1:
            pooled.append(finite.mean(axis=0))
        start += hop_frames
    return np.array(pooled, dtype=np.float64) if pooled else np.empty((0, emb.shape[1]))


def score_audio(
    audio: np.ndarray,
    sr: int,
    encodec_extractor,
    flow,
    window_frames: int,
    hop_frames: int,
) -> dict | None:
    """Extract EnCodec latents and score with the RealNVP flow."""
    try:
        emb = encodec_extractor.extract(audio, sr)
        if emb is None or len(emb) == 0:
            return None
        emb = np.asarray(emb, dtype=np.float32)
    except Exception as exc:
        logger.debug("EnCodec extraction failed: %s", exc)
        return None

    # Window and hop come from features/frontend.py, which derives the frame rate
    # from the array. This script hard-coded EnCodec's 75 fps, so combprint's
    # ~1 profile/second gave zero windows and every track returned None
    # ("No tracks scored successfully"). That derivation now exists once.
    from intrinsic_ai_music_detection.features.frontend import EmptyFrontEndOutput, windows_or_raise

    seconds = max(len(audio) / max(sr, 1), 1e-6)
    fps = max(len(emb) / seconds, 1e-9)
    try:
        pw = windows_or_raise(emb, seconds, window_frames / fps, hop_frames / fps, context="reconstruction control")
    except EmptyFrontEndOutput as exc:
        logger.debug("%s", exc)
        return None
    if len(pw) < 2:
        return None

    lls = -flow.score_samples(pw)  # score_samples = NLL; negate -> loglik
    finite = lls[np.isfinite(lls)]
    if len(finite) < 2:
        return None

    return {
        "wf_mean": float(finite.mean()),
        "wf_std": float(finite.std()),
        "wf_n_windows": int(len(finite)),
        "anomaly_score": float(-finite.mean()),  # anomaly = -mean_loglik
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_extractor(flow, requested: str, device: str):
    """Pick the front end for scoring and REFUSE a mismatch with the checkpoint.

    EnCodec latents and 128-bin log-mel are both 128-d, so a wrong front end
    passes every shape check and silently yields meaningless scores. The flow
    records the embedding it was trained on (RealNVPOneClass.embedding_name);
    honour it, and fail loudly when the request disagrees.
    """
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    trained_on = getattr(flow, "embedding_name", None)
    if trained_on and requested and trained_on != requested:
        raise SystemExit(
            f"EMBEDDING MISMATCH: checkpoint was trained on {trained_on!r} but --embedding is "
            f"{requested!r}. Scoring {trained_on!r}-trained density with {requested!r} features "
            "would produce meaningless numbers (several front ends share 128 dims). "
            f"Re-run with --embedding {trained_on}."
        )
    name = requested or trained_on or "encodec"
    if not trained_on:
        logger.warning(
            "Checkpoint has no embedding tag (saved before tagging existed) — assuming %r. "
            "Verify this matches how the flow was trained.",
            name,
        )
    logger.info("scoring front end: %s", name)
    return get_extractor(name, device=device)


def _save_variant(args, manifest_rows, track_id, variant, audio, sr) -> None:
    """Write one variant's audio and record it, if --save-audio-dir was given.

    ``pair_id`` is the track: gate C10 compares variants WITHIN a pair, because
    the whole point is that the content is identical and only the reconstruction
    differs.
    """
    if not args.save_audio_dir:
        return
    import soundfile as sf

    out = Path(args.save_audio_dir) / f"{track_id}__{variant}.wav"
    try:
        sf.write(str(out), np.asarray(audio, dtype=np.float32), sr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write %s: %s", out, exc)
        return
    manifest_rows.append(
        {
            "track_id": f"{track_id}__{variant}",
            "pair_id": track_id,
            "variant": variant,
            "label": "real",  # every variant derives from REAL audio
            "algorithm": variant,
            "path": str(out),
            "canonical_path": str(out),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruction-based confound control: real vs codec-reconstructed audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-dir", required=True, help="Directory of real canonical WAVs")
    parser.add_argument(
        "--flow-path",
        required=True,
        help="Path to saved RealNVP flow (.pt) trained on SONICS real windows",
    )
    parser.add_argument("--output-dir", default="data/processed/reconstruction_control")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-tracks", type=int, default=500, help="Number of real tracks to sample")
    parser.add_argument(
        "--recon-types",
        nargs="*",
        default=["encodec_3kbps", "encodec_6kbps", "encodec_24kbps", "griffinlim_128mel", "griffinlim_256mel"],
        help="Reconstruction types to test",
    )
    parser.add_argument("--wf-window-duration", type=float, default=4.0)
    parser.add_argument("--wf-hop-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-complexity",
        action="store_true",
        default=False,
        help=(
            "Also measure FLAC bits/sec of the original and each reconstruction (Serrà et al., "
            "ICLR 2020 complexity estimate). DECISIVE CONFOUND TEST: content is byte-identical "
            "across an original/reconstruction pair, so if the FLAC statistic separates them it "
            "is tracking reconstruction/codec artifacts (a legitimate detection signal); if it "
            "does NOT, then any real-vs-fake complexity gap on a corpus is production/mastering "
            "difference, i.e. a corpus confound. Adds ~1-2 s per track (ffmpeg)."
        ),
    )
    parser.add_argument(
        "--save-audio-dir",
        default=None,
        help=(
            "Also WRITE the reconstructions (and the originals) as WAVs here, plus a "
            "manifest with track_id/variant/pair_id/path. Without this the script only "
            "scores in memory with the FLOW, which is why gate C10 could never be run on "
            "the training-free comb or the NMF arm — the content-identical pairs existed "
            "only transiently. This closes the last SKIP."
        ),
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help="Front end for scoring (must match the checkpoint; auto-detected from it if omitted).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load flow ---
    from intrinsic_ai_music_detection.models.flow import RealNVPOneClass

    logger.info("Loading flow from %s ...", args.flow_path)
    flow = RealNVPOneClass.load(args.flow_path, device=args.device)
    logger.info("Flow loaded.")

    # --- Load EnCodec extractor ---
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    logger.info("Loading EnCodec extractor ...")
    extractor = _resolve_extractor(flow, getattr(args, "embedding", None), args.device)
    logger.info("EnCodec ready.")

    window_frames = max(int(args.wf_window_duration * ENCODEC_FPS), 5)
    hop_frames = max(int(args.wf_hop_duration * ENCODEC_FPS), 1)

    # --- Collect real tracks ---
    real_dir = Path(args.real_dir)
    audio_files = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(audio_files))[: args.n_tracks]
    audio_files = [audio_files[i] for i in sorted(idx)]
    logger.info("Selected %d real tracks for reconstruction control", len(audio_files))

    # --- Resume support -----------------------------------------------------
    # Originally this script only wrote output at the very end (single
    # `df.to_csv` after the loop), so a single interruption on a multi-hour
    # run (500 tracks x up to 5 reconstruction types, several of them
    # GPU-heavy EnCodec round-trips or CPU-heavy 32-iteration Griffin-Lim)
    # would silently lose ALL progress. Now every completed track is appended
    # to a JSONL checkpoint immediately, and already-complete tracks (original
    # + every requested recon_type present) are skipped on restart.
    import json

    checkpoint_path = output_dir / "reconstruction_checkpoint.jsonl"
    required_types = {"original", *args.recon_types}
    done_by_track: dict[str, set[str]] = {}
    checkpoint_rows: list[dict] = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                checkpoint_rows.append(row)
                done_by_track.setdefault(row["track_id"], set()).add(row["recon_type"])
        n_complete = sum(1 for tid, types in done_by_track.items() if required_types.issubset(types))
        logger.info(
            "Resuming from checkpoint: %d rows, %d/%d tracks already fully complete",
            len(checkpoint_rows),
            n_complete,
            len(audio_files),
        )

    # Rows for the exported-audio manifest, so the content-identical pairs can be
    # re-scored later by ANY detector rather than only by the flow loaded here.
    audio_manifest: list[dict] = []
    if args.save_audio_dir:
        Path(args.save_audio_dir).mkdir(parents=True, exist_ok=True)
        logger.info("exporting reconstruction audio to %s", args.save_audio_dir)

    checkpoint_fh = open(checkpoint_path, "a")

    def _emit(row: dict) -> None:
        checkpoint_rows.append(row)
        checkpoint_fh.write(json.dumps(row) + "\n")
        checkpoint_fh.flush()

    # --- Score each track: original + all reconstructions ---
    from intrinsic_ai_music_detection.data.audio_utils import load_audio

    n_skipped_complete = 0
    for i, path in enumerate(audio_files, start=1):
        track_id = path.stem
        have = done_by_track.get(track_id, set())
        if required_types.issubset(have):
            n_skipped_complete += 1
            continue

        try:
            audio, sr = load_audio(str(path), target_sr=ENCODEC_SR, max_duration=args.max_duration)
        except Exception as exc:
            logger.debug("[%d] load failed %s: %s", i, path.name, exc)
            continue

        # Score the ORIGINAL (skip if already checkpointed)
        if "original" not in have:
            orig_scores = score_audio(audio, sr, extractor, flow, window_frames, hop_frames)
            if orig_scores is None:
                continue
            if args.with_complexity:
                orig_scores["flac_bits_per_sec"] = flac_bits_per_sec_array(audio, sr)
            _emit({"track_id": track_id, "recon_type": "original", **orig_scores})
            _save_variant(args, audio_manifest, track_id, "source", audio, sr)

        # Score each reconstruction (skip ones already checkpointed)
        for recon_type in args.recon_types:
            if recon_type in have:
                continue
            try:
                if recon_type.startswith("encodec_"):
                    bw_str = recon_type.replace("encodec_", "").replace("kbps", "")
                    bw = float(bw_str)
                    recon_audio = reconstruct_encodec(audio, sr, bw, args.device)
                elif recon_type.startswith("griffinlim_"):
                    n_mels = int(recon_type.split("_")[1].replace("mel", ""))
                    recon_audio = reconstruct_griffinlim(audio, sr, n_mels=n_mels)
                else:
                    logger.warning("Unknown recon_type: %s", recon_type)
                    continue

                _save_variant(args, audio_manifest, track_id, recon_type, recon_audio, sr)
                recon_scores = score_audio(recon_audio, sr, extractor, flow, window_frames, hop_frames)
                if recon_scores is not None:
                    if args.with_complexity:
                        recon_scores["flac_bits_per_sec"] = flac_bits_per_sec_array(recon_audio, sr)
                    _emit({"track_id": track_id, "recon_type": recon_type, **recon_scores})

            except Exception as exc:
                logger.warning("[%d] recon %s failed for %s: %s", i, recon_type, path.name, exc)

        if i % 10 == 0 or i == len(audio_files):
            n_orig_done = sum(1 for r in checkpoint_rows if r["recon_type"] == "original")
            logger.info(
                "[%d/%d] ok=%d  (skipped %d already-complete)", i, len(audio_files), n_orig_done, n_skipped_complete
            )

    checkpoint_fh.close()

    if not checkpoint_rows:
        logger.error("No tracks scored successfully.")
        sys.exit(1)

    df = pd.DataFrame(checkpoint_rows).drop_duplicates(subset=["track_id", "recon_type"], keep="last")
    df.to_csv(output_dir / "reconstruction_scores.csv", index=False)
    logger.info("Saved reconstruction_scores.csv (%d rows)", len(df))

    # --- AUC: original vs each reconstruction ---
    from intrinsic_ai_music_detection.models.evaluate import auc_and_eer

    orig_scores = df.loc[df["recon_type"] == "original", ["track_id", "anomaly_score"]].set_index("track_id")

    auc_rows = []
    for rtype in args.recon_types:
        recon_df = df.loc[df["recon_type"] == rtype, ["track_id", "anomaly_score"]].set_index("track_id")
        common = orig_scores.index.intersection(recon_df.index)
        if len(common) < 5:
            continue

        # Pair (original=0, reconstruction=1)
        real_s = orig_scores.loc[common, "anomaly_score"].values
        recon_s = recon_df.loc[common, "anomaly_score"].values
        y = np.r_[np.zeros(len(real_s)), np.ones(len(recon_s))]
        sc = np.r_[real_s, recon_s]
        auc, eer = auc_and_eer(y, sc)

        delta_mean = float(np.mean(recon_s) - np.mean(real_s))
        logger.info(
            "  %-25s  n=%3d  AUC=%.4f  EER=%.4f  Δmean_anomaly=+%.3f",
            rtype,
            len(common),
            auc,
            eer,
            delta_mean,
        )
        auc_rows.append(
            {
                "recon_type": rtype,
                "n": len(common),
                "auc": auc,
                "eer": eer,
                "orig_anomaly_mean": float(np.mean(real_s)),
                "recon_anomaly_mean": float(np.mean(recon_s)),
                "delta_anomaly_mean": delta_mean,
            }
        )

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(output_dir / "reconstruction_auc.csv", index=False)
    logger.info("\nSaved reconstruction_auc.csv:\n%s", auc_df.to_string(index=False))

    # --- Complexity control: does the FLAC statistic separate CONTENT-IDENTICAL pairs? ---
    if args.with_complexity and "flac_bits_per_sec" in df.columns:
        orig_cx = df.loc[df["recon_type"] == "original", ["track_id", "flac_bits_per_sec"]].set_index("track_id")
        cx_rows = []
        for rtype in args.recon_types:
            rec_cx = df.loc[df["recon_type"] == rtype, ["track_id", "flac_bits_per_sec"]].set_index("track_id")
            common = orig_cx.index.intersection(rec_cx.index)
            o = orig_cx.loc[common, "flac_bits_per_sec"].to_numpy(float)
            r = rec_cx.loc[common, "flac_bits_per_sec"].to_numpy(float)
            ok = np.isfinite(o) & np.isfinite(r)
            o, r = o[ok], r[ok]
            if len(o) < 5:
                continue
            # Anomaly convention: LOWER complexity = more "AI-like" -> score = -bits/s
            y = np.r_[np.zeros(len(o)), np.ones(len(r))]
            auc, eer = auc_and_eer(y, np.r_[-o, -r])
            cx_rows.append(
                {
                    "recon_type": rtype,
                    "n": len(o),
                    "complexity_auc": round(auc, 4),
                    "complexity_eer": round(eer, 4),
                    "orig_flac_bps_mean": round(float(o.mean()), 1),
                    "recon_flac_bps_mean": round(float(r.mean()), 1),
                    "delta_flac_bps": round(float(r.mean() - o.mean()), 1),
                }
            )
        if cx_rows:
            cx_df = pd.DataFrame(cx_rows)
            cx_df.to_csv(output_dir / "reconstruction_complexity_auc.csv", index=False)
            logger.info("\nCOMPLEXITY CONTROL (content-identical pairs):\n%s", cx_df.to_string(index=False))
            logger.info(
                "INTERPRETATION: content is byte-identical within each pair, so complexity_auc "
                "well above 0.5 means the FLAC statistic tracks RECONSTRUCTION/CODEC ARTIFACTS "
                "(a legitimate, content-independent detection signal, and its real-vs-fake "
                "performance is then not merely corpus mastering). complexity_auc ~ 0.5 means the "
                "statistic cannot see codec artifacts at all, so any real-vs-fake complexity gap "
                "must be attributed to production/mastering differences between the corpora — "
                "report it as a CONFOUND and do not present it as a detector."
            )

    # --- ROC curves plot ---
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(args.recon_types)))
        for rtype, color in zip(args.recon_types, colors):
            recon_df = df.loc[df["recon_type"] == rtype, ["track_id", "anomaly_score"]].set_index("track_id")
            common = orig_scores.index.intersection(recon_df.index)
            if len(common) < 5:
                continue
            real_s = orig_scores.loc[common, "anomaly_score"].values
            recon_s = recon_df.loc[common, "anomaly_score"].values
            y = np.r_[np.zeros(len(real_s)), np.ones(len(recon_s))]
            sc = np.r_[real_s, recon_s]
            fpr, tpr, _ = roc_curve(y, sc)
            auc = float(np.trapz(tpr, fpr))
            ax.plot(fpr, tpr, color=color, label=f"{rtype} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate (original flagged as reconstruction)")
        ax.set_ylabel("True Positive Rate (reconstruction detected)")
        ax.set_title(
            "Reconstruction Control: Original vs Reconstructed Audio\n"
            "(same content, different codec — confound-proofs genre/content signal)"
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(output_dir / "reconstruction_auc.png", dpi=150)
        plt.close(fig)
        logger.info("Saved reconstruction_auc.png")

    except Exception as exc:
        logger.warning("Could not save ROC plot: %s", exc)

    if args.save_audio_dir and audio_manifest:
        man = pd.DataFrame(audio_manifest).drop_duplicates(subset=["track_id"])
        man_path = Path(args.save_audio_dir) / "recon_audio_manifest.csv"
        man.to_csv(man_path, index=False)
        logger.info(
            "Exported %d variant WAVs across %d content-identical pairs -> %s",
            len(man),
            man["pair_id"].nunique(),
            man_path,
        )
        logger.info(
            "Gate C10 for the REPORTED detectors (the flow is not the detector):\n"
            "  python scripts/eval_comb_detector.py \\\n"
            "    --manifest %s \\\n"
            "    --audio-column canonical_path --per-stratum 0 --workers 6 \\\n"
            "    --max-duration 9 --resample-hz 0 --level-match --smooth-bins 5 \\\n"
            "    --residual median --average db --f-min 1000 --span-duration 3 \\\n"
            "    --carry-columns variant pair_id \\\n"
            "    --out-dir reports/diagnostics/comb_recon_control",
            man_path,
        )

    # Interpretation
    min_auc = auc_df["auc"].min() if not auc_df.empty else float("nan")
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)
    if min_auc > 0.65:
        logger.info(
            "✓  Minimum reconstruction AUC=%.4f > 0.65 across all types. "
            "The flow detects codec/vocoder reconstruction artifacts even when "
            "musical content, genre, and acoustic covariates are held constant. "
            "This is strong evidence the signal is NOT a genre/content confound.",
            min_auc,
        )
    else:
        logger.warning(
            "⚠  Some reconstruction AUC values are low (min=%.4f). "
            "The detector may not reliably separate codec artifacts from content. "
            "Check which reconstruction types produce low AUC.",
            min_auc,
        )
    logger.info(
        "Note: EnCodec reconstructions at lower bitrates (3/6 kbps) should score "
        "HIGHER anomaly than originals, mirroring how AI generators produce "
        "periodic quantization patterns in the latent trajectory."
    )


if __name__ == "__main__":
    main()
