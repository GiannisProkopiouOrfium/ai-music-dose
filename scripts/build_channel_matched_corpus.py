"""Give both classes the SAME delivery chain, instead of deleting the band they differ in.

Why this replaces `--resample-hz`
---------------------------------
The bandwidth control worked — it took SONICS' channel-alone AUC from 1.0000 to
0.5778 — but the reconstruction control then showed what it cost. With the
controlled flow, on content-identical original/reconstruction pairs:

| reconstruction | uncontrolled flow | controlled flow |
|---|---|---|
| EnCodec 3 kbps  | 0.9799 | **0.5340** |
| EnCodec 6 kbps  | 0.9127 | **0.4891** |
| EnCodec 24 kbps | 0.8310 | **0.4973** |
| Griffin-Lim 128 | 0.7627 | **0.9903** |
| Griffin-Lim 256 | 0.7267 | **0.9889** |

Codec-quantisation sensitivity was destroyed; phase-reconstruction sensitivity
*improved*. So the two artifact families live in different bands: RVQ
quantisation noise sits above ~7.5 kHz and decimation deletes it, while
phase/coherence damage is broadband and survives. Deleting the band is therefore
the wrong control — it removes real evidence along with the leak.

The right control is to make the delivery chains identical. SONICS fakes are
distributed as 16 kHz MP3; our reals are a 48 kHz YouTube FLAC pull. Sending the
reals through the fakes' chain equalises the channel *without discarding any
band*, so whatever HF evidence survives the fakes' own delivery stays available
to the detector and is legitimately comparable.

What it does
------------
For every track in the manifest, from its ORIGINAL source:

    decode -> mono -> resample to --chain-sr -> MP3 at --chain-kbps
           -> decode -> resample to --target-sr -> LUFS normalise

Both classes get the identical chain, so this is symmetric preprocessing, not a
label-dependent correction. The fakes acquire one extra codec generation (their
own ~37 kbps plus ours); that residual asymmetry is far smaller than the 48 kHz
vs 16 kHz gap it replaces, and it is stated rather than hidden.

Run the channel diagnostic on the output before trusting any detector number
from it.

Usage (EC2)
-----------
    python scripts/build_channel_matched_corpus.py \
        --manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
        --out-dir data/processed/canonical_sonics_chainmatched \
        --chain-sr 16000 --chain-kbps 64 --workers 6
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_channel_matched_corpus")


def _chain_one(job: dict) -> dict:
    """Decode -> chain sample rate -> MP3 -> decode -> target rate -> LUFS."""
    import librosa

    from intrinsic_ai_music_detection.data.audio_preprocessing import normalise_loudness

    src, dst = Path(job["src"]), Path(job["dst"])
    out = {"track_id": job["track_id"], "canonical_path": str(dst)}

    if dst.exists() and dst.stat().st_size > 1024:
        out["status"] = "ok"
        return out

    try:
        audio, _ = librosa.load(
            str(src),
            sr=job["chain_sr"],
            mono=True,
            duration=job["max_duration"],
            res_type="soxr_hq",
        )
        if audio.size < job["chain_sr"] // 2:
            out["status"] = "too_short"
            return out

        if job["chain_kbps"] <= 0:
            # NO CODEC. Used when both classes are already delivered identically
            # and a codec pass would only DESTROY evidence: FakeMusicCaps ships
            # every clip at 16 kHz mono, so there is no format confound to
            # remove, and one MP3-64k pass measurably erases the deconvolution
            # comb (three generators sharing an exact 299.8 Hz autocorrelation
            # lag on raw audio spread to 380-464 Hz after it, with real music
            # landing inside that range).
            #
            # The int16 round-trip below is still applied to BOTH classes: the
            # FakeMusicCaps reals ship as pcm_s16le and the fakes as pcm_f32le,
            # so 16-bit quantisation noise is otherwise present on one side only.
            if job["target_sr"] != job["chain_sr"]:
                x = librosa.resample(
                    audio,
                    orig_sr=job["chain_sr"],
                    target_sr=job["target_sr"],
                    res_type="soxr_vhq",
                )
            else:
                x = audio
            sr_out = job["target_sr"]
            x = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).astype(np.float32) / 32768.0
        else:
            with tempfile.TemporaryDirectory() as tmp:
                wav_in = Path(tmp) / "in.wav"
                mp3 = Path(tmp) / "mid.mp3"
                wav_out = Path(tmp) / "out.wav"

                scipy.io.wavfile.write(
                    str(wav_in),
                    job["chain_sr"],
                    (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16),
                )
                # Encode AT the chain sample rate, so the codec's own low-pass and
                # quantisation are the ones the fake class actually received. Doing
                # the resample after the codec would give a different noise floor.
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(wav_in),
                        "-b:a",
                        f"{job['chain_kbps']}k",
                        "-ar",
                        str(job["chain_sr"]),
                        "-ac",
                        "1",
                        str(mp3),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=180,
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3), "-ar", str(job["target_sr"]), "-ac", "1", str(wav_out)],
                    capture_output=True,
                    check=True,
                    timeout=180,
                )
                sr_out, data = scipy.io.wavfile.read(str(wav_out))

            x = data.astype(np.float32)
            x = x / 32768.0 if data.dtype == np.int16 else x
        x = normalise_loudness(x, sr_out, target_lufs=job["target_lufs"])

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_suffix(".wav.tmp")
        scipy.io.wavfile.write(str(tmp_dst), sr_out, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
        tmp_dst.replace(dst)

        out["status"] = "ok"
        out["actual_duration"] = round(len(x) / sr_out, 3)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = str(exc)[:200]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a corpus so both classes share one delivery chain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--src-column",
        default="src_path",
        help="Column holding the ORIGINAL source. Must NOT be canonical_path: chaining an "
        "already-canonicalised file adds a codec generation to one class only.",
    )
    parser.add_argument(
        "--chain-sr",
        type=int,
        default=16_000,
        help="Sample rate of the shared chain. Set it to the FAKE class's native rate "
        "(SONICS and FakeMusicCaps are both 16 kHz).",
    )
    parser.add_argument(
        "--chain-kbps",
        type=int,
        default=64,
        help=(
            "MP3 bitrate of the shared chain, encoded AT --chain-sr. Pass 0 for NO codec: use "
            "that when both classes already share a delivery format, because a codec pass then "
            "removes no confound and measurably destroys the artifact (the deconvolution comb "
            "does not survive one MP3-64k pass). Both classes still get an int16 round-trip, "
            "which equalises FakeMusicCaps' pcm_s16le reals against its pcm_f32le fakes."
        ),
    )
    parser.add_argument("--target-sr", type=int, default=24_000, help="Output rate (encoder native).")
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N (smoke test).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"]
    if args.src_column not in df.columns:
        raise SystemExit(f"--src-column {args.src_column!r} not in manifest (has {list(df.columns)})")
    df["track_id"] = df["track_id"].astype(str)
    df["label"] = df["label"].astype(str)
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        raise SystemExit("no rows to process")

    missing = int((~df[args.src_column].astype(str).map(lambda p: Path(p).exists())).sum())
    if missing:
        logger.warning(
            "%d/%d source files do not resolve. If that miss rate is class-skewed it silently "
            "rebalances the corpus — check before using the output.",
            missing,
            len(df),
        )

    jobs = []
    for row in df.itertuples(index=False):
        src = str(getattr(row, args.src_column))
        if not Path(src).exists():
            continue
        sub = "real" if row.label == "real" else "fake"
        jobs.append(
            {
                "src": src,
                "dst": str(out_dir / sub / f"{row.track_id}.wav"),
                "track_id": row.track_id,
                "chain_sr": args.chain_sr,
                "chain_kbps": args.chain_kbps,
                "target_sr": args.target_sr,
                "target_lufs": args.target_lufs,
                "max_duration": args.max_duration,
            }
        )
    chain_desc = (
        f"-> {args.chain_sr} Hz -> MP3 {args.chain_kbps} kbps -> {args.target_sr} Hz"
        if args.chain_kbps > 0
        else f"-> {args.chain_sr} Hz -> int16 (NO codec) -> {args.target_sr} Hz"
    )
    logger.info("chaining %d tracks: %s -> %.0f LUFS", len(jobs), chain_desc, args.target_lufs)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_chain_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0 or i == len(futures):
                n_ok = sum(r["status"] == "ok" for r in results)
                logger.info("  %d/%d (%d ok)", i, len(futures), n_ok)

    res = pd.DataFrame(results)
    # Drop every column the rebuild recomputes BEFORE joining. Suffixing them
    # instead collides as soon as the input manifest already carries a suffixed
    # name (chaining a chain-matched corpus raised MergeError on
    # 'actual_duration_old'), and a half-suffixed manifest is worse than a
    # crash because downstream code silently reads the stale column.
    recomputed = [c for c in res.columns if c != "track_id"]
    df = df.drop(columns=[c for c in recomputed if c in df.columns])
    merged = df.merge(res, on="track_id", how="inner")

    manifest_path = out_dir / "canonical_manifest.csv"
    merged.to_csv(manifest_path, index=False)

    n_ok = int((merged["status"] == "ok").sum())
    by_label = merged[merged["status"] == "ok"]["label"].value_counts()
    logger.info("manifest -> %s (%d ok of %d)\n%s", manifest_path, n_ok, len(merged), by_label.to_string())

    if n_ok == 0:
        raise SystemExit("nothing was written — refusing to leave an empty corpus behind")
    fail_by_label = merged[merged["status"] != "ok"]["label"].value_counts()
    if len(fail_by_label):
        logger.warning("failures by class (a skew here rebalances the corpus):\n%s", fail_by_label.to_string())

    logger.info(
        "\nNEXT: run the channel diagnostic on this corpus BEFORE any detector.\n"
        "  python scripts/diagnose_channel_confound.py --manifest %s \\\n"
        "    --which canonical --mode profile --per-stratum 400 --workers 6 \\\n"
        "    --level-match --max-duration 25 --fixed-duration 20 \\\n"
        "    --out-dir reports/diagnostics/channel_chainmatched\n"
        "Note there is NO --resample-hz: the point of this corpus is that the band no longer "
        "has to be discarded.",
        manifest_path,
    )


if __name__ == "__main__":
    main()
