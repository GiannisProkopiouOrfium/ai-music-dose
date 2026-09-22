"""Measure, on REAL cached embeddings, what float16 storage costs us.

`--emb-cache-dtype float16` halves cache size, which is the difference between
the 256-d @ 100 fps spec-musicdet SONICS cache fitting on disk or not. Before
relying on it, verify empirically rather than by appeal to "float16 has ~3
decimal digits": load existing float32 cache entries, round-trip them through
float16, and measure the effect where it actually matters — on the pooled
window features the flow consumes, and on the flow's own log-likelihood
RANKING, which is what every AUC/EER in the project is computed from.

Decision rule applied by the script:
  * pooled-feature deviation, in standardised units, must be << 1
  * Spearman rank correlation of per-window log-likelihoods must be ~1.0
  * the induced change in any score must be tiny relative to the real-vs-fake
    score gap we are trying to measure

Exit code is 1 if any check fails, so it can gate the decision.

Usage:
    python scripts/check_float16_cache_fidelity.py \\
        --cache-dir data/emb_cache_specmd_fmc_native/spec-musicdet --n-files 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n-files", type=int, default=300)
    ap.add_argument("--window-frames", type=int, default=400, help="4 s at 100 fps")
    ap.add_argument("--hop-frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, "src")
    from intrinsic_ai_music_detection.features.pooling import pool_windows
    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    files = sorted(Path(args.cache_dir).glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy in {args.cache_dir}")
    rng = np.random.default_rng(args.seed)
    pick = [files[i] for i in rng.choice(len(files), size=min(args.n_files, len(files)), replace=False)]
    logger.info("sampling %d of %d cache entries", len(pick), len(files))

    p32, p16, raw_dev, f16_seen = [], [], [], False
    for f in pick:
        x = np.load(f)
        if x.dtype == np.float16:
            f16_seen = True
        x32 = x.astype(np.float32)
        x16 = x32.astype(np.float16).astype(np.float32)
        raw_dev.append(float(np.abs(x32 - x16).max()))
        w32 = pool_windows(x32, args.window_frames, args.hop_frames)
        w16 = pool_windows(x16, args.window_frames, args.hop_frames)
        if len(w32) and len(w32) == len(w16):
            p32.append(w32)
            p16.append(w16)
    if not p32:
        raise SystemExit("no windows produced — check --window-frames against the cache frame rate")
    if f16_seen:
        logger.warning(
            "some entries are ALREADY float16; their round-trip is a no-op and the "
            "measured loss is optimistic. Point at a float32 cache for a clean test."
        )

    a32, a16 = np.vstack(p32), np.vstack(p16)
    mu, sd = a32.mean(axis=0), a32.std(axis=0) + 1e-9
    std_dev = np.abs((a32 - a16) / sd)
    logger.info("raw feature       : max |delta| = %.3e", max(raw_dev))
    logger.info("pooled, standardised: max %.3e | median %.3e", std_dev.max(), float(np.median(std_dev)))

    # the decisive test: does the flow RANK windows the same way?
    cfg = RealNVPConfig(n_coupling_layers=4, hidden_dim=64, n_epochs=25, batch_size=256, seed=0)
    n_fit = min(len(a32), 20000)
    flow = RealNVPOneClass(cfg).fit(a32[:n_fit])
    s32, s16 = flow.score_samples(a32), flow.score_samples(a16)

    from scipy.stats import spearmanr

    rho = float(spearmanr(s32, s16).statistic)
    spread = float(np.percentile(s32, 95) - np.percentile(s32, 5))
    err = float(np.abs(s32 - s16).max())
    logger.info("flow score        : spearman rho = %.8f", rho)
    logger.info(
        "flow score        : max |delta| = %.4e vs 5-95%% score spread %.4e (%.4f%%)",
        err,
        spread,
        100 * err / max(spread, 1e-12),
    )

    ok = (std_dev.max() < 0.05) and (rho > 0.9999) and (err < 0.01 * spread)
    if ok:
        logger.info("PASS — float16 caching is safe for this representation; use --emb-cache-dtype float16")
        return 0
    logger.error("FAIL — float16 measurably perturbs the scores; keep float32 and buy disk instead")
    return 1


if __name__ == "__main__":
    sys.exit(main())
