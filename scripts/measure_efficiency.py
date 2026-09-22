"""Measure inference efficiency of our EnCodec + RealNVP window-flow pipeline,
in the same units as MusicDET's Table 3 ("Comparison of Efficiency"):

    Speed (M/S)   — inference throughput, full tracks/sec ("music/s")
    FLOPs (G)     — floating-point operations for one forward pass
    Memory (GB)   — peak GPU memory during inference
    Param. (M)    — trainable parameter count

MusicDET's protocol (arXiv 2605.18072, Sec 4.2 "Analysis of Efficiency"):
loads pre-processed files, forward-pass-only, batch size 12, single GPU —
i.e. disk I/O and preprocessing are OUTSIDE their timed region; only the
batched forward pass over already-loaded tensors is timed. We mirror that
as closely as our pipeline allows: batch size 12 *tracks*, forward pass
only (no backward), same GPU, "Param" = trainable parameters only (their
MERT-AASIST-frozen row reports 0.45M despite a much larger frozen SSL
backbone underneath — i.e. frozen components don't count toward Param,
but DO count toward FLOPs/Memory/Speed since they're on the critical
inference path). Our EnCodec encoder is entirely frozen (never trained), so:
    Param  = RealNVP flow parameters only
    FLOPs / Memory / Speed = full pipeline (EnCodec encode + windowing + flow)

Measurement-hygiene protocol (repeated-trial version):
  - WARM-UP batches run and are DISCARDED before every timed region (cuDNN
    autotune / lazy CUDA init would otherwise land inside the first trial —
    the earlier single-shot version of this script swung 508→313 tracks/s
    between identical runs for exactly this reason).
  - Each timed region is repeated --n-trials times back-to-back; the JSON
    reports per-trial values plus mean±std. Cite mean±std, never one trial.
  - Throughput is ALSO reported as audio-seconds processed per second
    (speed_audio_s_per_sec_*): "tracks/s" depends on clip length, so the
    audio-seconds rate is the honest cross-paper unit. The default
    --e2e-clip-duration is 55 s to match the headline analysis duration
    (a 10 s-clip "tracks/s" number would look ~5.5x faster than the same
    pipeline on real 55 s inputs).
  - --amp runs EnCodec's frozen encoder under torch.autocast fp16 (I7
    mitigation); the flow stays fp32. Report fp32 and AMP as separate rows.

Two "end-to-end" (EnCodec + flow) numbers are reported:
  - speed_tracks_per_sec_end_to_end_unbatched_legacy: audio load + EnCodec
    encode call ONE TRACK AT A TIME inside the timed loop. NOT comparable to
    MusicDET's number — kept only so older reports remain reproducible.
  - speed_tracks_per_sec_true_batched_e2e: audio pre-loaded and fixed-length
    cropped/padded OUTSIDE the timed region, then EnCodec's encoder AND the
    flow both run as genuine batched forward passes INSIDE it. This is the
    like-for-like number to cite against MusicDET's 516 M/S.

Usage
-----
python scripts/measure_efficiency.py \
    --flow-path data/processed/full_sonics_all_k12/sonics_real_flow_k12_encodec.pt \
    --audio-dir data/processed/canonical_sonics_full/real \
    --n-tracks 60 --batch-size 12 --n-trials 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

# The project package isn't pip-installed in the venv (matches every other
# script here, e.g. run_robustness_battery.py) — import via src/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

N_WARMUP_BATCHES = 2


def _count_flow_params(flow) -> int:
    return sum(p.numel() for p in flow._model.parameters() if p.requires_grad)


def _to_flow_space(flow, pw: np.ndarray) -> np.ndarray:
    """Raw pooled windows → the flow's input space (PCA if attached, then standardise)."""
    if flow._pca_components is not None and pw.shape[1] != flow._dim and pw.shape[1] == flow._pca_components.shape[1]:
        pw = (pw - flow._pca_mean) @ flow._pca_components.T
    if pw.shape[1] != flow._mu.shape[-1]:
        raise SystemExit(
            f"FRONT-END MISMATCH: the checkpoint expects {flow._mu.shape[-1]}-d input but the "
            f"extractor produced {pw.shape[1]}-d. This script loaded EnCodec regardless of what "
            f"the flow was trained on — pass --embedding to match "
            f"{getattr(flow, 'embedding_name', None)!r}."
        )
    return (pw - flow._mu) / flow._sd


def _batched_encode(extractor, audios: list, device: str, amp: bool = False) -> list:
    """Run EnCodec's encoder on a zero-padded BATCH of equal-length raw audio
    arrays in a single forward pass, then trim each track's output back to its
    true (unpadded) frame count. With ``amp=True`` the (frozen, inference-only)
    encoder runs under torch.autocast fp16 — outputs are cast back to fp32.

    Front ends that are not a torch module with a ``.model.encoder`` — notably
    ``combprint``, which is a pure CPU signal-processing transform — cannot be
    batched this way. They fall back to per-track extraction, which is the
    honest thing to time for them: there is no batched forward pass to measure
    because there is no neural forward pass at all. That fallback is reported in
    the output JSON as ``batched_encode: false`` so the number is not silently
    compared against a genuinely batched one.
    """
    import torch

    if not hasattr(extractor, "model") or not hasattr(getattr(extractor, "model", None), "encoder"):
        return [np.asarray(extractor.extract(a, extractor.sample_rate), dtype=np.float32) for a in audios]

    max_len = max(len(a) for a in audios)
    batch = np.zeros((len(audios), 1, max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        batch[i, 0, : len(a)] = a
    wav = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        if amp and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = extractor.model.encoder(wav)  # [B, 128, T_padded]
            emb = emb.float()
        else:
            emb = extractor.model.encoder(wav)
    emb_np = emb.permute(0, 2, 1).cpu().numpy()  # [B, T_padded, 128]
    total_t = emb_np.shape[1]
    out = []
    for i, a in enumerate(audios):
        true_t = max(1, round(total_t * len(a) / max_len))
        out.append(emb_np[i, :true_t, :])
    return out


def _score_ok(feats: dict, name: str | None) -> bool:
    """Confirm the timed pass actually produced the requested score.

    Without this the loop counts a track as "scored" whenever comb_features returns
    anything at all, so a detector whose columns were never emitted would still report
    a throughput -- timing a computation nobody asked for.
    """
    from intrinsic_ai_music_detection.features.comb_artifacts import comb_calibrated_score

    if not name or name == "nmf_peak":
        return True
    try:
        comb_calibrated_score(feats, name)
    except (KeyError, ValueError):
        return False
    return True


def _measure_training_free(args) -> None:
    """Time the training-free detector: one STFT per track, no model.

    This is a genuinely different measurement from the flow path and is reported
    as such. There is no checkpoint, so Param = 0; there is no neural forward
    pass, so the "batched forward" that MusicDET's 516 M/S measures has no
    counterpart here and we time the WHOLE detector end to end instead — which
    is the conservative direction (it includes audio decode and the STFT).

    Cite the audio-seconds/s row. Our clips are not MusicDET's 4.04 s segments,
    so tracks/s is not comparable across papers; audio-seconds/s is.
    """
    from intrinsic_ai_music_detection.data.audio_utils import load_audio
    from intrinsic_ai_music_detection.features.comb_artifacts import comb_features

    audio_dir = Path(args.audio_dir)
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)[: args.n_tracks]
    if not files:
        raise SystemExit(f"No audio files found under {audio_dir}")
    logger.info("Timing training-free detector %r on %d tracks", args.training_free, len(files))

    clip_s = args.e2e_clip_duration
    audios = []
    for f in files:
        audio, sr = load_audio(f, target_sr=16_000)
        audio = audio.astype(np.float32)
        n = int(clip_s * sr)
        if len(audio) >= n:
            start = (len(audio) - n) // 2
            audio = audio[start : start + n]
        else:
            padded = np.zeros(n, dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded
        audios.append((audio, sr))

    # Time ONLY what the reported detector computes. The library defaults add a
    # log-frequency variant and per-span stationarity residuals — diagnostics, not
    # the detector — and default to the `hull` envelope, whose O(n) Python loop
    # dominates. Timing those made the first measurement 3x pessimistic
    # (7.3 tracks/s) for a detector that is one STFT and one autocorrelation.
    detector_kw = {
        "residual_op": "median",
        "average": "db",
        "f_min": 1000.0,
        "smooth_bins": 5,
        "span_duration": 0.0,
        "log_axis": False,
        # The PROPOSED detector needs only its own harmonic order and its decoys:
        # comb_priormax2_margin is one prior evaluation plus 24 decoy evaluations,
        # ~25 autocorrelation lookups. Timing the full emitted grid (three harmonic
        # orders x two prior forms x 24 decoys) would overstate a deployment cost
        # nobody would pay, so the timing uses the deployed configuration.
        "n_harm_grid": (2,),
        "surrogates": 0,
        "null_priors": args.null_priors,
    }
    # Time exactly what the requested detector computes. comb_priormax*_hmargin needs
    # NO decoys -- its null is deterministic -- so it is timed at null_priors=0, which
    # is the configuration a deployer runs and is markedly cheaper than the 24-decoy
    # margin the paper currently proposes.
    _order = re.search(r"comb_priormax(\d+)_", args.training_free or "")
    if _order:
        detector_kw["n_harm_grid"] = (int(_order.group(1)),)
        if args.training_free.endswith(("_hmargin", "_latmargin", "lo20margin", "widemargin", "lo20widemargin")):
            detector_kw["null_priors"] = 0
        elif detector_kw["null_priors"] <= 0:
            detector_kw["null_priors"] = 24
    logger.info("timing the detector only: %s", detector_kw)

    def _pass() -> int:
        scored = 0
        for audio, sr in audios:
            feats = comb_features(audio, sr, **detector_kw)
            if feats and _score_ok(feats, args.training_free):
                scored += 1
        return scored

    _pass()  # warm-up (discarded): FFT plan caching must not land in trial 1

    trial_times: list[float] = []
    n_scored = 0
    for _trial in range(max(1, args.n_trials)):
        t0 = time.time()
        n_scored = _pass()
        trial_times.append(time.time() - t0)

    if n_scored < len(audios):
        raise SystemExit(
            f"Only {n_scored}/{len(audios)} tracks produced comb features. An empty result is a "
            "hard error here — see the standing rule in features/frontend.py."
        )

    speeds = [len(audios) / t for t in trial_times]
    total_audio_s = len(audios) * clip_s
    result = {
        "method": f"{args.training_free} (training-free, ours)",
        "n_tracks_measured": len(audios),
        "n_trials": args.n_trials,
        "device": "cpu (no neural forward pass exists)",
        "param_M": 0.0,
        "trainable_parameters": 0,
        "batched_encode": False,
        "e2e_clip_duration_s": clip_s,
        "speed_tracks_per_sec": round(float(np.mean(speeds)), 2),
        "speed_tracks_per_sec_std": round(float(np.std(speeds)), 2),
        "speed_tracks_per_sec_trials": [round(s, 2) for s in speeds],
        "speed_audio_s_per_sec": round(total_audio_s / float(np.mean(trial_times)), 1),
        "note": (
            "Whole-detector timing including audio decode and STFT — not a forward-pass-only "
            "number. Compare on audio-seconds/s, not tracks/s: MusicDET's 516 M/S is over "
            "4.04 s segments and ours over "
            f"{clip_s:.0f} s clips."
        ),
        "musicdet_table3_reference": _MUSICDET_TABLE3,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "TRAINING-FREE %s: Param=0  Speed=%.1f±%.1f tracks/s (%.0f audio-s/s over %.0fs clips)",
        args.training_free,
        float(np.mean(speeds)),
        float(np.std(speeds)),
        total_audio_s / float(np.mean(trial_times)),
        clip_s,
    )
    logger.info("Saved → %s", out_path)


_MUSICDET_TABLE3 = {
    "MusicDET": {"speed_M_S": 516, "flops_G": 4.09, "memory_GB": 0.11, "param_M": 8.13, "eer_pct": 4.51},
    "AASIST": {"speed_M_S": 271, "flops_G": 9.62, "memory_GB": 0.20, "param_M": 0.30, "eer_pct": 32.73},
    "MERT-AASIST_frozen": {"speed_M_S": 175, "flops_G": 73.20, "memory_GB": 1.33, "param_M": 0.45, "eer_pct": 23.27},
    "SpecTTTra-alpha": {"speed_M_S": 810, "flops_G": 2.85, "memory_GB": 0.33, "param_M": 16.83, "eer_pct": 17.63},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--flow-path",
        default=None,
        help=(
            "Saved RealNVPOneClass checkpoint (for Param count). Required unless "
            "--training-free is given: the corrected detector has no flow at all."
        ),
    )
    parser.add_argument(
        "--training-free",
        default=None,
        choices=["comb_strength", "comb_stat_strength", "comb_priormax2_margin", "comb_priormax4_margin", "comb_priormax2_hmargin", "comb_priormax4_hmargin", "comb_priormax4_latmargin", "comb_priormax4_unionmargin",
        "comb_priormax4_lo20margin",
        "comb_priormax4_widemargin",
        "comb_priormax4_lo20widemargin"] + ["nmf_peak"],
        help=(
            "Time the TRAINING-FREE detector instead of a flow. There is no checkpoint and no "
            "neural forward pass, so Param=0 and FLOPs are the STFT's — which is the point of "
            "the table. Mutually exclusive with --flow-path."
        ),
    )
    parser.add_argument("--audio-dir", required=True, help="Directory of real canonical audio for timing.")
    parser.add_argument("--n-tracks", type=int, default=60)
    parser.add_argument(
        "--null-priors",
        type=int,
        default=0,
        help="0 times the comb arm the paper reports; 24 times the proposed "
        "prior-restricted, decoy-calibrated score of Sec. 7.",
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Matches MusicDET's Table 3 protocol.")
    parser.add_argument("--window-duration", type=float, default=4.0)
    parser.add_argument("--hop-duration", type=float, default=2.0)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help=(
            "Repeat each timed region this many times (after discarded warm-up batches) and "
            "report mean±std. 1 = legacy single-shot behaviour (do not cite single-shot numbers)."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help=(
            "Run EnCodec's frozen encoder under torch.autocast fp16 (CUDA only). The flow "
            "stays fp32. Report as a separate 'AMP' row next to the fp32 row, not instead of it."
        ),
    )
    parser.add_argument(
        "--e2e-clip-duration",
        type=float,
        default=55.0,
        help=(
            "Fixed clip length (s) for the true-batched end-to-end timing: each track is "
            "center-cropped/zero-padded to this length OUTSIDE the timed region. Default 55 "
            "matches the headline --analysis-duration so 'tracks/s' means the same thing as "
            "elsewhere in the paper; the audio-seconds/s output is clip-length-independent."
        ),
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help=(
            "Front end to time. MUST match the checkpoint — this script previously loaded "
            "EnCodec unconditionally and died with a broadcast error on any other flow. "
            "Auto-detected from the checkpoint's embedding_name when omitted."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default="reports/efficiency_comparison.json")
    args = parser.parse_args()

    if bool(args.flow_path) == bool(args.training_free):
        raise SystemExit(
            "Pass exactly one of --flow-path or --training-free. The corrected detector is "
            "training-free (no checkpoint exists for it); the flow arms are the ablation."
        )

    import torch

    from intrinsic_ai_music_detection.data.audio_utils import load_audio
    from intrinsic_ai_music_detection.features.frontend import resolve_frontend, windows_or_raise
    from intrinsic_ai_music_detection.models.flow import RealNVPOneClass

    if args.training_free:
        _measure_training_free(args)
        return

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        logger.warning("CUDA not available, falling back to CPU (numbers will not be comparable to Table 3).")
    is_cuda = str(device).startswith("cuda")

    logger.info("Loading flow checkpoint: %s", args.flow_path)
    flow = RealNVPOneClass.load(args.flow_path, device=device)
    n_params = _count_flow_params(flow)
    logger.info("Flow trainable parameters: %d (%.3fM)", n_params, n_params / 1e6)

    # The front end must follow the CHECKPOINT. This was hard-coded to EnCodec,
    # so timing any other flow died with a broadcast error deep in _to_flow_space
    # instead of saying which front end it wanted. Resolution now lives in
    # features/frontend.py so this script cannot disagree with the other two that
    # had the same bug.
    front_end = resolve_frontend(embedding=args.embedding, flow=flow, device=device)
    _name = front_end.name
    extractor = front_end.extractor
    logger.info("Front end (from checkpoint): %s", _name)
    _ = getattr(extractor, "model", None)  # force lazy load now, outside the timed region

    audio_dir = Path(args.audio_dir)
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)[: args.n_tracks]
    if len(files) < args.n_tracks:
        logger.warning("Only found %d audio files (< requested %d) in %s", len(files), args.n_tracks, audio_dir)
    if not files:
        raise SystemExit(f"No audio files found under {audio_dir}")

    def _sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    # --- pre-load + pre-encode all tracks with EnCodec (outside timed region) ---
    # so the flow-only "Speed" measurement below isolates the SAME thing MusicDET
    # measures: forward-pass compute given pre-processed input, batch size 12.
    # This loop also produces the LEGACY unbatched end-to-end timing further down
    # (kept for backward compatibility only).
    pooled_windows: list[np.ndarray] = []
    track_durations: list[float] = []
    t0_e2e = time.time()
    for f in files:
        audio, sr = load_audio(f, target_sr=extractor.sample_rate)
        emb = front_end.extract(audio.astype(np.float32), sr)
        dur = max(len(audio) / sr, 1.0)
        track_durations.append(dur)
        # Window derived from the array's own measured rate, not from EnCodec's 75 fps.
        pooled_windows.append(
            windows_or_raise(emb, dur, args.window_duration, args.hop_duration, context=f"{f.name} pre-encode")
        )
    e2e_encode_time = time.time() - t0_e2e
    mean_track_dur = float(np.mean(track_durations))
    total_audio_s = float(np.sum(track_durations))

    xz_list = [_to_flow_space(flow, pw) for pw in pooled_windows]
    mean_windows_per_track = float(np.mean([len(x) for x in xz_list]))

    # --- flow-only forward-pass timing (batched, warm-up + repeated trials) ---
    flow._model.eval()

    def _flow_only_pass() -> None:
        with torch.no_grad():
            for i in range(0, len(xz_list), args.batch_size):
                batch_tracks = xz_list[i : i + args.batch_size]
                batch_x = np.concatenate(batch_tracks, axis=0).astype(np.float32)
                xb = torch.from_numpy(batch_x).to(device)
                _ = flow._model.log_prob(xb)

    # warm-up (discarded): cuDNN autotune / lazy init must not land in trial 1
    with torch.no_grad():
        for i in range(0, min(N_WARMUP_BATCHES * args.batch_size, len(xz_list)), args.batch_size):
            xb = torch.from_numpy(np.concatenate(xz_list[i : i + args.batch_size], axis=0).astype(np.float32)).to(
                device
            )
            _ = flow._model.log_prob(xb)
    _sync()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    flow_trial_times: list[float] = []
    for _trial in range(max(1, args.n_trials)):
        _sync()
        t0 = time.time()
        _flow_only_pass()
        _sync()
        flow_trial_times.append(time.time() - t0)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else float("nan")

    flow_speeds = [len(files) / t for t in flow_trial_times]
    tracks_per_sec_flow_only = float(np.mean(flow_speeds))
    tracks_per_sec_end_to_end_unbatched_legacy = len(files) / (e2e_encode_time + float(np.mean(flow_trial_times)))

    # --- true batched end-to-end timing (EnCodec encode + flow, both batched) ---
    clip_samples = int(args.e2e_clip_duration * extractor.sample_rate)
    fixed_audios = []
    for f in files:
        audio, sr = load_audio(f, target_sr=extractor.sample_rate)
        audio = audio.astype(np.float32)
        if len(audio) >= clip_samples:
            start = (len(audio) - clip_samples) // 2
            fixed_audios.append(audio[start : start + clip_samples])
        else:
            padded = np.zeros(clip_samples, dtype=np.float32)
            padded[: len(audio)] = audio
            fixed_audios.append(padded)

    def _e2e_pass() -> None:
        with torch.no_grad():
            for i in range(0, len(fixed_audios), args.batch_size):
                batch_audios = fixed_audios[i : i + args.batch_size]
                batch_embs = _batched_encode(extractor, batch_audios, device, amp=args.amp)
                pooled = [
                    windows_or_raise(
                        emb,
                        args.e2e_clip_duration,
                        args.window_duration,
                        args.hop_duration,
                        context="batched e2e",
                    )
                    for emb in batch_embs
                ]
                xz_batch = [_to_flow_space(flow, pw) for pw in pooled]
                batch_x = np.concatenate(xz_batch, axis=0).astype(np.float32)
                xb = torch.from_numpy(batch_x).to(device)
                _ = flow._model.log_prob(xb)

    # warm-up (discarded) — first batch only, enough to trigger autotune/AMP init
    with torch.no_grad():
        _batched_encode(extractor, fixed_audios[: args.batch_size], device, amp=args.amp)
    _sync()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    e2e_trial_times: list[float] = []
    for _trial in range(max(1, args.n_trials)):
        _sync()
        t0 = time.time()
        _e2e_pass()
        _sync()
        e2e_trial_times.append(time.time() - t0)
    peak_mem_gb_e2e = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else float("nan")

    e2e_speeds = [len(fixed_audios) / t for t in e2e_trial_times]
    tracks_per_sec_true_batched_e2e = float(np.mean(e2e_speeds))

    # --- FLOPs (best-effort; requires torch>=2.1's FlopCounterMode) ---
    flops_g = None
    flops_g_per_track = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        n_sample = min(args.batch_size, len(xz_list[0]))
        sample_x = torch.from_numpy(xz_list[0][:n_sample].astype(np.float32)).to(device)
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                _ = flow._model.log_prob(sample_x)
        flops_g = fc.get_total_flops() / 1e9
        # Per-track estimate: FLOPs scale linearly in window count for the flow.
        flops_g_per_track = flops_g / n_sample * mean_windows_per_track
        logger.info(
            "Flow-only FLOPs: %.4f G for %d WINDOWS (≈%.4f G per track at %.1f windows/track). "
            "MusicDET's 4.09 G is per input sample — compare against the PER-TRACK figure, "
            "and note ours excludes EnCodec's encoder FLOPs (frozen but on the critical path).",
            flops_g,
            n_sample,
            flops_g_per_track,
            mean_windows_per_track,
        )
    except Exception as exc:
        logger.warning("FlopCounterMode unavailable (%s) — FLOPs not measured.", exc)

    result = {
        "method": "encodec_realnvp_window_flow (ours)",
        "n_tracks_measured": len(files),
        "batch_size": args.batch_size,
        "n_trials": args.n_trials,
        "n_warmup_batches": N_WARMUP_BATCHES,
        "amp": bool(args.amp),
        "device": device,
        "param_M_flow_only": round(n_params / 1e6, 4),
        "flops_G_flow_forward_per_window_batch": round(flops_g, 4) if flops_g is not None else None,
        "flops_G_flow_forward_per_track_est": (round(flops_g_per_track, 4) if flops_g_per_track is not None else None),
        "mean_windows_per_track": round(mean_windows_per_track, 2),
        "mean_track_duration_s": round(mean_track_dur, 2),
        "memory_GB_peak_flow_forward": round(peak_mem_gb, 4) if not np.isnan(peak_mem_gb) else None,
        "memory_GB_peak_true_batched_e2e": (round(peak_mem_gb_e2e, 4) if not np.isnan(peak_mem_gb_e2e) else None),
        "speed_tracks_per_sec_flow_forward_only": round(tracks_per_sec_flow_only, 2),
        "speed_tracks_per_sec_flow_forward_only_std": round(float(np.std(flow_speeds)), 2),
        "speed_tracks_per_sec_flow_forward_only_trials": [round(s, 2) for s in flow_speeds],
        "speed_audio_s_per_sec_flow_forward_only": round(total_audio_s / float(np.mean(flow_trial_times)), 1),
        "speed_tracks_per_sec_true_batched_e2e": round(tracks_per_sec_true_batched_e2e, 2),
        "speed_tracks_per_sec_true_batched_e2e_std": round(float(np.std(e2e_speeds)), 2),
        "speed_tracks_per_sec_true_batched_e2e_trials": [round(s, 2) for s in e2e_speeds],
        "speed_audio_s_per_sec_true_batched_e2e": round(
            len(fixed_audios) * args.e2e_clip_duration / float(np.mean(e2e_trial_times)), 1
        ),
        "e2e_clip_duration_s": args.e2e_clip_duration,
        "speed_tracks_per_sec_end_to_end_unbatched_legacy": round(tracks_per_sec_end_to_end_unbatched_legacy, 2),
        "musicdet_table3_reference": _MUSICDET_TABLE3,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    logger.info("=" * 70)
    logger.info("EFFICIENCY COMPARISON (vs. MusicDET Table 3)%s", "  [AMP fp16 encoder]" if args.amp else "")
    logger.info("=" * 70)
    logger.info(
        "Ours   : Param=%.3fM  Speed(flow-only)=%.1f±%.1f tracks/s (%.0f audio-s/s)  "
        "Speed(TRUE BATCHED e2e, %.0fs clips)=%.1f±%.1f tracks/s (%.0f audio-s/s)  "
        "Memory(flow)=%.3fGB  Memory(e2e)=%.3fGB  FLOPs(flow/track)=%s G   [%d trials, %d warm-up batches]",
        n_params / 1e6,
        tracks_per_sec_flow_only,
        float(np.std(flow_speeds)),
        total_audio_s / float(np.mean(flow_trial_times)),
        args.e2e_clip_duration,
        tracks_per_sec_true_batched_e2e,
        float(np.std(e2e_speeds)),
        len(fixed_audios) * args.e2e_clip_duration / float(np.mean(e2e_trial_times)),
        peak_mem_gb,
        peak_mem_gb_e2e,
        f"{flops_g_per_track:.3f}" if flops_g_per_track is not None else "N/A",
        args.n_trials,
        N_WARMUP_BATCHES,
    )
    logger.info("MusicDET: Param=8.13M  Speed=516 M/S  Memory=0.11GB  FLOPs=4.09G  EER=4.51%%")
    logger.warning(
        "The 'speed_tracks_per_sec_end_to_end_unbatched_legacy' figure (%.1f) is kept only for "
        "backward-compatible reproducibility and must NOT be cited — it is dominated by "
        "one-track-at-a-time Python/disk overhead. Cite the true-batched e2e mean±std, and "
        "note the clip duration (tracks/s is clip-length-dependent; audio-s/s is not).",
        tracks_per_sec_end_to_end_unbatched_legacy,
    )
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
