"""Build a COMMON-BANDWIDTH canonical corpus (bandwidth-confound elimination).

Why
---
The spectral "udio wall break" (0.96 LOGO) was traced to a *bandwidth confound*:
SONICS fakes are ~8 kHz band-limited while reals are full-band, so band-edge
shape features (rolloff/slope/flatness near 6-8 kHz) separate the classes for a
trivial, non-architectural reason. The band-narrowing control (--band-hi-hz 6000)
already exposed this. This script makes the fix *explicit and global*: it
low-passes BOTH reals and fakes to a single shared ceiling so no downstream
feature -- spectral OR embedding -- can exploit bandwidth.

The embedding pipeline already low-passes to 8 kHz for both classes, so geometry
features were never bandwidth-confounded; this corpus simply makes the whole
pipeline uniform and unassailable for the final paper numbers.

What it does
------------
Reads the canonical manifest, low-passes each canonical WAV to ``--ceiling-hz``
(default 7000, safely below the fakes' ~8 kHz edge) with a steep zero-phase
Butterworth filter, writes 24 kHz int16 WAVs to ``--out-dir`` and a new manifest
with identical ``track_id``s and updated ``canonical_path``. Re-run the spectral
extractor and the diagnostic against the new manifest to get confound-free
numbers.

Run on EC2 (canonical WAVs live there):
    python scripts/build_common_bandwidth_canonical.py \
        --manifest data/processed/canonical_sampler/canonical_manifest.csv \
        --out-dir data/processed/canonical_commonband \
        --ceiling-hz 7000 --workers 6

    # then re-extract spectral on the bandwidth-matched corpus:
    python scripts/compute_spectral_features.py \
        --manifest data/processed/canonical_commonband/canonical_manifest.csv \
        --audio-source canonical \
        --output data/processed/multiembed_descriptors/spectral_features_cb.csv \
        --workers 6
    python scripts/diagnose_descriptors.py \
        --features data/processed/multiembed_descriptors/ablation_features.csv \
        --spectral data/processed/multiembed_descriptors/spectral_features_cb.csv \
        --out-dir reports/diagnostics/commonband
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile
from scipy import signal as sp_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_common_bandwidth")


def _lowpass_to_ceiling(audio: np.ndarray, sr: int, ceiling_hz: float, order: int = 10) -> np.ndarray:
    """Zero-phase Butterworth low-pass; both real and fake get the SAME ceiling."""
    nyq = sr / 2.0
    wc = min(ceiling_hz, 0.99 * nyq) / nyq
    sos = sp_signal.butter(order, wc, btype="low", output="sos")
    return sp_signal.sosfiltfilt(sos, audio).astype(np.float32)


def _process_one(job: dict) -> dict:
    src = Path(job["src"])
    dst = Path(job["dst"])
    if dst.exists():
        return {"track_id": job["track_id"], "canonical_path": str(dst), "status": "ok"}
    try:
        sr, data = scipy.io.wavfile.read(str(src))
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        else:
            x = data.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        y = _lowpass_to_ceiling(x, sr, job["ceiling_hz"])
        y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(dst), sr, y_int16)
        return {"track_id": job["track_id"], "canonical_path": str(dst), "status": "ok"}
    except Exception as exc:  # pragma: no cover
        return {"track_id": job["track_id"], "canonical_path": "", "status": f"error: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/canonical_sampler/canonical_manifest.csv")
    ap.add_argument("--out-dir", default="data/processed/canonical_commonband")
    ap.add_argument("--ceiling-hz", type=float, default=7000.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, low_memory=False)
    man["track_id"] = man["track_id"].astype(str)
    if "status" in man.columns:
        man = man[man["status"].astype(str) == "ok"]
    out_dir = Path(args.out_dir)

    jobs = []
    for _, r in man.iterrows():
        src = str(r.get("canonical_path", ""))
        if not src or not Path(src).exists():
            continue
        label = str(r.get("label", "")).lower()
        sub = "fake" if label == "fake" else "real"
        dst = out_dir / sub / f"{r['track_id']}.wav"
        jobs.append({"track_id": r["track_id"], "src": src, "dst": str(dst), "ceiling_hz": args.ceiling_hz})
    if args.limit:
        jobs = jobs[: args.limit]
    logger.info("Low-passing %d tracks to %.0f Hz -> %s", len(jobs), args.ceiling_hz, out_dir)
    if not jobs:
        logger.error("No tracks found (check --manifest canonical_path).")
        sys.exit(1)

    results: dict[str, dict] = {}
    t0 = time.time()
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, j): j["track_id"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            results[row["track_id"]] = row
            if not str(row["status"]).startswith("ok"):
                n_err += 1
            if i % 500 == 0 or i == len(jobs):
                logger.info("  %d/%d (%.1f tracks/s, %d errors)", i, len(jobs), i / max(time.time() - t0, 1e-6), n_err)

    # new manifest: copy source rows, overwrite canonical_path with the band-limited file
    new = man.copy()
    new = new[new["track_id"].isin(results.keys())].copy()
    new["canonical_path"] = new["track_id"].map(lambda t: results[t]["canonical_path"])
    new["common_bandwidth_hz"] = args.ceiling_hz
    out_manifest = out_dir / "canonical_manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    new.to_csv(out_manifest, index=False)
    logger.info(
        "Wrote %d band-limited tracks + manifest %s (%d errors, %.1fs)", len(new), out_manifest, n_err, time.time() - t0
    )


if __name__ == "__main__":
    main()
