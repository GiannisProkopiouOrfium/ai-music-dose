"""Train and evaluate the convolutional-coupling flow on time-frequency windows.

Why a separate script
---------------------
`run_balanced_ablation.py` pools every window to a 1-D vector before the flow
sees it. This arm's entire point is to NOT do that, so it reads the same cached
frame matrices and builds `[N, 1, F, T]` windows instead. Same corpus, same
manifest, same symmetric fold protocol — only the model's view of a window
changes, which is what makes the comparison an ablation rather than a new system.

What it is testing
------------------
Whether MusicDET's advantage is the convolutional coupling. On FakeMusicCaps at
16 kHz our pooled arms give macro 0.589 (plain) and 0.621 (+ banding + global
prior), and widening the pooled vector to 2048 dims *inverts* to 0.413 because an
MLP coupling overfits 16k windows. A conv coupling models ~8x more values for the
same parameter count, so if the bottleneck is capacity-per-input-value this is
where it shows up.

Usage (EC2)
-----------
    python scripts/run_conv_flow.py \
        --manifest data/processed/canonical_fmc_raw16/canonical_manifest.csv \
        --embedding spec-musicdet \
        --embedding-cache-dir data/emb_cache_specmd_fmcraw \
        --per-stratum 4000 --max-duration 9 --device cuda \
        --output-dir data/processed/fmcraw_convflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.cache_keys import (  # noqa: E402
    band_match_tag,
    embedding_cache_key,
    level_match_tag,
)
from intrinsic_ai_music_detection.models.conv_flow import ConvFlowConfig, ConvRealNVPOneClass  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_conv_flow")

EMBEDDING_SR = {"encodec": 24_000, "spec": 24_000, "logmel": 24_000, "spec-musicdet": 16_000}


def make_windows(frames: np.ndarray, window_frames: int, hop_frames: int, freq_crop: int) -> np.ndarray:
    """Slice a ``[T_total, F]`` frame matrix into ``[n, 1, F, T]`` windows.

    Both axes are cropped, never padded, to a multiple of the squeeze factor.
    Padding would insert a constant region that the flow can key on, and clip
    length correlates with class on both corpora.
    """
    if frames.ndim != 2 or len(frames) < window_frames:
        return np.empty((0, 1, freq_crop, window_frames), dtype=np.float32)
    out: list[np.ndarray] = []
    start = 0
    while start + window_frames <= len(frames):
        block = frames[start : start + window_frames, :freq_crop]
        if np.isfinite(block).all():
            out.append(block.T[None, :, :])  # -> [1, F, T]
        start += hop_frames
    if not out:
        return np.empty((0, 1, freq_crop, window_frames), dtype=np.float32)
    return np.stack(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convolutional-coupling one-class flow over time-frequency windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding", default="spec-musicdet", choices=sorted(EMBEDDING_SR))
    parser.add_argument("--per-stratum", type=int, default=4000)
    parser.add_argument("--max-duration", type=float, default=9.0)
    parser.add_argument(
        "--fixed-duration",
        type=float,
        default=None,
        help=(
            "MUST match the --fixed-duration the cache was built with: it is part of the cache "
            "key (_fd<N>). Omitting it is why the first run resolved zero entries."
        ),
    )
    parser.add_argument("--window-duration", type=float, default=4.0)
    parser.add_argument("--hop-duration", type=float, default=2.0)
    parser.add_argument("--freq-crop", type=int, default=256, help="STFT bins kept (multiple of 2**n_scales)")
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--n-scales", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augment-p", type=float, default=0.0, help="SpecAugment probability (MusicDET uses 0.5)")
    parser.add_argument(
        "--window-normalize",
        action="store_true",
        default=False,
        help=(
            "Subtract each window's own mean before the flow — the log-domain equivalent of "
            "MusicDET's per-segment RMS normalisation (dataset.py normalises every 4.04 s "
            "segment; we normalise loudness once per track). Removes absolute level from the "
            "model's view entirely, which is worth testing because peak_dbfs remains a "
            "0.64-0.69 channel descriptor after every other control."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=20000,
        help=(
            "Cap on training windows per fold. At 256x400 a window is 0.4 MB, so 20k is ~8 GB — "
            "the first attempt indexed 144k windows (~58 GB) and was OOM-killed with no "
            "traceback. Windows are read from a memmap on demand, so only this budget and one "
            "batch are ever resident."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--level-match",
        action="store_true",
        default=False,
        help="Must match how the cache was built, or the key will miss.",
    )
    parser.add_argument(
        "--resample-hz", type=float, default=None, help="Must match how the cache was built, or the key will miss."
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"]
    df["track_id"] = df["track_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["algorithm"] = df.get("algorithm", pd.Series([""] * len(df))).fillna("").astype(str)

    rng = np.random.default_rng(args.seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], sort=True):
        n = min(args.per_stratum, len(grp))
        parts.append(grp.iloc[np.sort(rng.choice(len(grp), size=n, replace=False))])
    df = pd.concat(parts, ignore_index=True)
    logger.info("sampled %d tracks", len(df))

    target_sr = EMBEDDING_SR[args.embedding]
    cache_dir = Path(args.embedding_cache_dir) / args.embedding
    if not cache_dir.exists():
        raise SystemExit(f"embedding cache not found: {cache_dir}")
    tags = tuple(t for t in (band_match_tag(args.resample_hz), level_match_tag(args.level_match)) if t)

    div = 2**args.n_scales
    if args.freq_crop % div:
        raise SystemExit(f"--freq-crop {args.freq_crop} must be divisible by 2**n_scales={div}")

    def load(track_id: str) -> np.ndarray | None:
        key = embedding_cache_key(
            track_id,
            target_sr,
            args.max_duration,
            "preprocessed",
            fixed_duration=args.fixed_duration,
            extra_tags=tags,
        )
        path = cache_dir / f"{key}.npy"
        if not path.exists():
            return None
        try:
            return np.load(str(path)).astype(np.float32)
        except Exception:  # noqa: BLE001
            return None

    # Frame rate is derived from the data, never assumed: spec-musicdet is
    # 100 fps and encodec 75, and a hard-coded rate silently windows the wrong
    # amount of audio.
    probe = next((load(t) for t in df["track_id"]), None)
    if probe is None:
        raise SystemExit(
            f"no cache entries resolved in {cache_dir}.\n"
            f"The key is md5 of: <track_id>_{target_sr}_{args.max_duration}_preprocessed"
            f"{'_fd%g' % args.fixed_duration if args.fixed_duration else ''}{''.join(tags)}\n"
            f"Every one of those must match how the cache was built — --max-duration, "
            f"--fixed-duration, --level-match and --resample-hz all change the key."
        )
    fps = len(probe) / max(args.max_duration, 1e-6)
    wf = int(args.window_duration * fps)
    wf -= wf % div  # crop, never pad
    hf = max(int(args.hop_duration * fps), 1)
    logger.info("fps=%.1f -> window %d frames, hop %d, freq crop %d", fps, wf, hf, args.freq_crop)

    reals = df[df["label"] == "real"]
    fakes = df[df["label"] != "real"]

    # MEMORY. A [1, 256, 400] float32 window is 0.4 MB. Materialising every
    # window — 24k real plus ~120k fake at hop 1 s — is ~58 GB and silently OOM
    # -kills the process (the first attempt stopped right after the "real:
    # N tracks" line with no traceback). So windows are indexed, not stored:
    # each entry is (path, start_frame) and the array is read from a memmap on
    # demand. Peak memory is then one batch.
    def index_track(track_id: str) -> tuple[Path, int] | None:
        key = embedding_cache_key(
            track_id,
            target_sr,
            args.max_duration,
            "preprocessed",
            fixed_duration=args.fixed_duration,
            extra_tags=tags,
        )
        path = cache_dir / f"{key}.npy"
        if not path.exists():
            return None
        try:
            n_frames = np.load(str(path), mmap_mode="r").shape[0]
        except Exception:  # noqa: BLE001
            return None
        return (path, n_frames) if n_frames >= wf else None

    def window_starts(n_frames: int) -> list[int]:
        return list(range(0, n_frames - wf + 1, hf))

    def read_windows(path: Path, starts: list[int]) -> np.ndarray:
        """Read the listed windows from one cached matrix via memmap."""
        mat = np.load(str(path), mmap_mode="r")
        out = []
        for st in starts:
            block = np.asarray(mat[st : st + wf, : args.freq_crop], dtype=np.float32)
            if np.isfinite(block).all():
                out.append(block.T[None, :, :])
        if not out:
            return np.empty((0, 1, args.freq_crop, wf), dtype=np.float32)
        return np.stack(out)

    real_index: list[tuple[str, Path, list[int]]] = []
    for tid in reals["track_id"]:
        entry = index_track(tid)
        if entry is None:
            continue
        path, n_frames = entry
        real_index.append((tid, path, window_starts(n_frames)))
    if not real_index:
        raise SystemExit("no real windows indexed — refusing to continue")
    n_real_windows = sum(len(s) for _, _, s in real_index)
    logger.info("real: %d tracks, %d windows indexed", len(real_index), n_real_windows)

    fake_index: list[tuple[str, str, Path, list[int]]] = []
    for row in fakes.itertuples(index=False):
        entry = index_track(str(row.track_id))
        if entry is None:
            continue
        path, n_frames = entry
        fake_index.append((str(row.track_id), str(row.algorithm), path, window_starts(n_frames)))
    if not fake_index:
        raise SystemExit("no fake windows indexed — refusing to continue")
    logger.info(
        "fake: %d tracks, %d windows indexed",
        len(fake_index),
        sum(len(s) for _, _, _, s in fake_index),
    )

    win_mb = args.freq_crop * wf * 4 / 1e6
    logger.info(
        "one window = %.2f MB; training set capped at %d windows (%.1f GB) — raise "
        "--max-train-windows only if the host has the RAM",
        win_mb,
        args.max_train_windows,
        args.max_train_windows * win_mb / 1e3,
    )

    cfg = ConvFlowConfig(
        n_blocks=args.n_blocks,
        n_scales=args.n_scales,
        hidden_channels=args.hidden_channels,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        verbose=True,
        augment_p=args.augment_p,
        window_normalize=args.window_normalize,
    )

    # Symmetric per-fold scoring: the SAME fold flow scores this fold's held-out
    # reals AND the fakes, so no fold-flow-vs-full-flow offset is folded into the
    # AUC. This is the protocol every other arm in the project uses.
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_rows: list[dict] = []
    track_scores: dict[str, float] = {}
    rng_train = np.random.default_rng(args.seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(real_index)))):
        # Subsample the training windows to a fixed budget BEFORE reading them,
        # so peak memory does not depend on corpus size.
        pool = [(i, st) for i in train_idx for st in real_index[i][2]]
        if len(pool) > args.max_train_windows:
            keep = rng_train.choice(len(pool), size=args.max_train_windows, replace=False)
            pool = [pool[k] for k in np.sort(keep)]
        by_track: dict[int, list[int]] = {}
        for i, st in pool:
            by_track.setdefault(i, []).append(st)
        chunks = [read_windows(real_index[i][1], starts) for i, starts in by_track.items()]
        chunks = [c for c in chunks if len(c)]
        if not chunks:
            logger.warning("fold %d: no training windows readable — skipping", fold)
            continue
        x_train = np.concatenate(chunks, axis=0)
        del chunks
        logger.info("fold %d: training on %d real windows", fold, len(x_train))

        det = ConvRealNVPOneClass(cfg).fit(x_train)
        det.embedding_name = args.embedding
        del x_train

        held_real = []
        for i in test_idx:
            tid, path, starts = real_index[i]
            w = read_windows(path, starts)
            if not len(w):
                continue
            s_ = det.score_samples(w)
            s_ = s_[np.isfinite(s_)]
            if len(s_):
                held_real.append(float(s_.mean()))
                track_scores[tid] = float(s_.mean())
        if not held_real:
            continue
        real_arr = np.asarray(held_real)

        by_alg: dict[str, list[float]] = {}
        for tid, alg, path, starts in fake_index:
            if not alg:
                continue
            w = read_windows(path, starts)
            if not len(w):
                continue
            s_ = det.score_samples(w)
            s_ = s_[np.isfinite(s_)]
            if len(s_):
                by_alg.setdefault(alg, []).append(float(s_.mean()))
                track_scores[tid] = float(s_.mean())

        for alg, scores in sorted(by_alg.items()):
            if len(scores) < 5:
                continue
            y = np.r_[np.zeros(len(real_arr)), np.ones(len(scores))]
            auc, eer = auc_and_eer(y, np.r_[real_arr, np.asarray(scores)])
            fold_rows.append(
                {
                    "fold": fold,
                    "algorithm": alg,
                    "auc": round(auc, 4),
                    "eer": round(eer, 4),
                    "n_real": len(real_arr),
                    "n_fake": len(scores),
                }
            )

        if fold == 0:
            det.save(out_dir / "conv_flow_fold0.pt")
        del det

    if not fold_rows:
        raise SystemExit("no fold produced a comparable pair — refusing to write an empty result")

    res = pd.DataFrame(fold_rows)
    res.to_csv(out_dir / f"conv_flow_fold_auc_{args.embedding}.csv", index=False)
    per_gen = res.groupby("algorithm")[["auc", "eer"]].mean()
    logger.info("\nSYMMETRIC per-fold AUC/EER:\n%s", per_gen.to_string())
    logger.info("MACRO AUC %.4f | MACRO EER %.2f%%", per_gen["auc"].mean(), per_gen["eer"].mean() * 100)

    pd.DataFrame([{"track_id": k, "conv_flow_anomaly": v} for k, v in track_scores.items()]).to_csv(
        out_dir / "conv_flow_scores.csv", index=False
    )
    logger.info(
        "per-track scores -> %s (fuse with --extra-scores or --comb-csv)",
        out_dir / "conv_flow_scores.csv",
    )


if __name__ == "__main__":
    main()
