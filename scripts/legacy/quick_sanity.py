"""Quick sanity check: EnCodec embeddings + ID estimation on sample data."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio
from intrinsic_ai_music_detection.features.embeddings import EnCodecExtractor
from intrinsic_ai_music_detection.features.id_estimators import estimate_all


def main():
    print("Loading EnCodec model...")
    extractor = EnCodecExtractor(device="cpu")
    print(f"EnCodec: {extractor.embedding_dim}d, {extractor.sample_rate}Hz, metric={extractor.distance_metric}")
    print()

    results = []
    dirs = [
        ("real", "data/raw/sanity_check/real"),
        ("ai", "data/raw/sanity_check/ai_musicgen"),
    ]

    for label, d in dirs:
        if not os.path.isdir(d):
            print(f"SKIP: {d} not found")
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(d, f)
            vid_id = f.replace(".wav", "")

            # Load & embed
            audio, sr = load_audio(path, target_sr=24000)
            audio = normalize_audio(audio)
            emb = extractor.extract(audio, sr)
            print(f"{label:4s} | {vid_id} | embeddings: {emb.shape}")

            # Compute ID (euclidean for EnCodec)
            ids = estimate_all(emb, metric="euclidean")
            phd_val = ids.get("phd", float("nan"))
            twonn_val = ids.get("twonn", float("nan"))
            mle_val = ids.get("mle", float("nan"))
            print(f"       PHD={phd_val:.3f}  TwoNN={twonn_val:.3f}  MLE={mle_val:.3f}")
            results.append({"label": label, "vid_id": vid_id, **ids})

    print()
    print("=" * 60)
    print("SUMMARY: Real vs AI Intrinsic Dimension")
    print("=" * 60)
    for method in ["phd", "twonn", "mle"]:
        real_vals = [r[method] for r in results if r["label"] == "real" and not np.isnan(r[method])]
        ai_vals = [r[method] for r in results if r["label"] == "ai" and not np.isnan(r[method])]
        if real_vals and ai_vals:
            rm = np.mean(real_vals)
            am = np.mean(ai_vals)
            diff = rm - am
            tag = "<-- EXPECTED" if diff > 0 else "<-- UNEXPECTED"
            print(
                f"{method.upper():>6s}: Real={rm:.3f} (n={len(real_vals)})  AI={am:.3f} (n={len(ai_vals)})  diff={diff:+.3f} {tag}"
            )
        else:
            print(f"{method.upper():>6s}: insufficient data (real={len(real_vals)}, ai={len(ai_vals)})")
    print("=" * 60)

    # Also test fakeprint baseline
    print()
    print("Testing fakeprint baseline...")
    try:
        from intrinsic_ai_music_detection.features.fakeprints import compute_fakeprint

        for label, d in dirs:
            if not os.path.isdir(d):
                continue
            f = sorted(os.listdir(d))[0]
            path = os.path.join(d, f)
            audio, sr = load_audio(path, target_sr=44100)
            fp = compute_fakeprint(audio, sr)
            print(f"  {label:4s} | {f} | fakeprint shape: {fp.shape}, mean={fp.mean():.4f}")
        print("  Fakeprint baseline: OK")
    except Exception as e:
        print(f"  Fakeprint baseline: FAILED - {e}")

    print()
    print("Sanity check complete!")


if __name__ == "__main__":
    main()
