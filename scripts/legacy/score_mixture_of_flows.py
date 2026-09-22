"""I2: Mixture-of-flows universal detector — post-hoc, fully unsupervised.

Motivation (detailed report §9.3): training ONE flow on SONICS-real +
MusicCaps-real raised FakeMusicCaps AUC to 0.706 but dropped EVERY SONICS
generator by 0.12-0.26 AUC — broadening one density model redistributes
detection power ("no free lunch"). This script tests the alternative the
report itself proposes: KEEP per-corpus flows sharp and score each track by
the flow that explains it best:

    anomaly(x) = min_i  calibrate_i( NLL_i(x) )

where calibrate_i maps flow i's raw NLL to the empirical quantile of that
flow's own HELD-OUT REAL score distribution (label-free — no fakes touched).
A real track from any covered domain gets a low calibrated score from its own
domain's flow; fakes score high under every flow. Success criterion:
FakeMusicCaps AUC >= the combined-flow's 0.706 while SONICS per-generator AUCs
stay at single-flow levels.

Inputs
------
--flow NAME=CHECKPOINT           (repeat per flow; PCA-aware checkpoints work)
--calibration NAME=CSV           (repeat; per-flow held-out REAL scores:
                                  columns track_id + one of wf_anomaly_score /
                                  anomaly / wf_mean (wf_mean is +loglik and is
                                  auto-negated); e.g. the real rows of that
                                  flow's window_flow_eval.csv)
--eval NAME=MANIFEST:CACHE:MAXDUR (repeat per eval corpus; MANIFEST needs
                                  track_id,label,algorithm; CACHE is the
                                  embedding-cache root containing encodec/;
                                  MAXDUR is the cache-key max-duration)

Output: per-corpus, per-generator AUC/EER for each single flow (calibrated)
AND the mixture — one CSV + a printed comparison table.

Usage (EC2):
    python scripts/score_mixture_of_flows.py \\
        --flow sonics_k12=data/processed/full_sonics_all_k12/sonics_real_flow_k12_encodec.pt \\
        --flow musiccaps_k2=data/processed/fmc_native_flow_k2/fmc_flow_encodec.pt \\
        --calibration sonics_k12=data/processed/full_sonics_all_k12/window_flow_eval.csv \\
        --calibration musiccaps_k2=data/processed/fmc_native_flow_k2/window_flow_eval.csv \\
        --eval sonics=data/processed/canonical_sonics_full/canonical_manifest.csv:data/emb_cache_encodec_full:55.0 \\
        --eval fakemusiccaps=data/processed/fmc_canonical/canonical_manifest.csv:data/emb_cache_fmc:10.0 \\
        --output-csv data/processed/mixture_of_flows/mixture_eval.csv --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.features.pooling import pool_windows  # noqa: E402
from intrinsic_ai_music_detection.models.aggregate import QuantileCalibrator, mixture_min_score  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402
from intrinsic_ai_music_detection.models.flow import RealNVPOneClass  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

ENCODEC_SR = 24_000
WINDOW_DUR = 4.0
HOP_DUR = 2.0


def _parse_kv(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--{what} expects NAME=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _load_windows(track_id: str, cache_root: Path, max_duration: float) -> np.ndarray | None:
    key = hashlib.md5(f"{track_id}_{ENCODEC_SR}_{max_duration}_preprocessed".encode()).hexdigest()
    cp = cache_root / "encodec" / f"{key}.npy"
    if not cp.exists():
        return None
    try:
        mat = np.load(str(cp))
        wf = max(int(WINDOW_DUR * 75), 5)
        hf = max(int(HOP_DUR * 75), 1)
        pw = pool_windows(mat, wf, hf)
        return pw if len(pw) >= 2 else None
    except Exception:
        return None


def _calibration_scores(csv_path: str) -> np.ndarray:
    """Held-out REAL anomaly scores for one flow, from its eval CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    real = df[df["label"] == "real"] if "label" in df.columns else df
    for col, negate in (("wf_anomaly_score", False), ("anomaly", False)):
        if col in real.columns:
            s = real[col].dropna().to_numpy(float)
            return s
    # wf_mean columns store +loglik → anomaly = -wf_mean
    wf_cols = [c for c in real.columns if c.endswith("_wf_mean") or c == "wf_mean"]
    if wf_cols:
        s = -real[wf_cols[0]].dropna().to_numpy(float)
        return s
    raise SystemExit(f"No usable score column (wf_anomaly_score/anomaly/*_wf_mean) in {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--flow", action="append", required=True, help="NAME=checkpoint.pt (repeatable)")
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        help="NAME=csv with that flow's held-out REAL scores (repeatable)",
    )
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        dest="evals",
        help="NAME=manifest.csv:emb_cache_root:max_duration (repeatable)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-per-group", type=int, default=5)
    args = parser.parse_args()

    flow_paths = _parse_kv(args.flow, "flow")
    calib_paths = _parse_kv(args.calibration, "calibration")
    missing = set(flow_paths) - set(calib_paths)
    if missing:
        raise SystemExit(
            f"flows without calibration CSVs: {sorted(missing)} — calibration is required "
            "(raw NLLs from different flows share no scale)"
        )

    flows = {name: RealNVPOneClass.load(p, device=args.device) for name, p in flow_paths.items()}
    calibrators = {name: QuantileCalibrator().fit(_calibration_scores(calib_paths[name])) for name in flow_paths}
    for name in flows:
        logger.info(
            "flow %-16s dim=%s  pca=%s  calibration n=%d",
            name,
            flows[name]._dim,
            "yes" if flows[name]._pca_components is not None else "no",
            len(calibrators[name].to_array()),
        )

    rows = []
    for spec in args.evals:
        if "=" not in spec:
            raise SystemExit(f"--eval expects NAME=manifest:cache:maxdur, got {spec!r}")
        corpus, rest = spec.split("=", 1)
        try:
            manifest_path, cache_root, maxdur_s = rest.rsplit(":", 2)
        except ValueError as exc:
            raise SystemExit(f"--eval {spec!r}: expected manifest:cache:max_duration") from exc
        max_duration = float(maxdur_s)
        manifest = pd.read_csv(manifest_path, low_memory=False)
        if "status" in manifest.columns:
            manifest = manifest[manifest["status"] == "ok"]
        manifest["track_id"] = manifest["track_id"].astype(str)
        cache = Path(cache_root)
        logger.info(
            "[%s] %d tracks (manifest=%s, cache=%s, maxdur=%.1f)",
            corpus,
            len(manifest),
            manifest_path,
            cache_root,
            max_duration,
        )

        # Per-track per-flow anomaly (mean NLL over windows), then calibrated.
        per_flow_scores: dict[str, list[float]] = {n: [] for n in flows}
        labels: list[str] = []
        algos: list[str] = []
        n_missing = 0
        for r in manifest.itertuples(index=False):
            pw = _load_windows(str(r.track_id), cache, max_duration)
            if pw is None:
                n_missing += 1
                continue
            labels.append(str(r.label))
            algos.append(str(getattr(r, "algorithm", "")))
            for name, flow in flows.items():
                try:
                    anom = float(flow.score_samples(pw).mean())
                except Exception:
                    anom = float("nan")
                per_flow_scores[name].append(anom)
        if n_missing:
            logger.warning("[%s] %d/%d tracks missing from cache — skipped", corpus, n_missing, len(manifest))
        labels_a = np.array(labels)
        algos_a = np.array(algos)
        raw = {n: np.array(v) for n, v in per_flow_scores.items()}
        cal = {n: calibrators[n].transform(raw[n]) for n in flows}
        mix = mixture_min_score({n: raw[n] for n in flows}, calibrators=calibrators)

        scored = dict(cal)
        scored["MIXTURE_MIN"] = mix
        real_mask = labels_a == "real"
        for method, s in scored.items():
            real_s = s[real_mask]
            real_s = real_s[np.isfinite(real_s)]
            if len(real_s) < args.min_per_group:
                continue
            for alg in sorted(set(algos_a[~real_mask])):
                if not alg.strip():
                    continue
                fake_s = s[(~real_mask) & (algos_a == alg)]
                fake_s = fake_s[np.isfinite(fake_s)]
                if len(fake_s) < args.min_per_group:
                    continue
                y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
                sc = np.r_[real_s, fake_s]
                auc, eer = auc_and_eer(y, sc)
                rows.append(
                    {
                        "corpus": corpus,
                        "scorer": method,
                        "algorithm": alg,
                        "auc": round(auc, 4),
                        "eer_pct": round(eer * 100, 2),
                        "n_real": len(real_s),
                        "n_fake": len(fake_s),
                    }
                )

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.info("Saved %s", args.output_csv)
    for corpus in out["corpus"].unique():
        pivot = out[out["corpus"] == corpus].pivot_table(index="scorer", columns="algorithm", values="auc")
        logger.info("\n[%s] AUC by scorer (rows) x generator (cols):\n%s", corpus, pivot.to_string())
    logger.info(
        "Reference points: combined single-flow gave FakeMusicCaps AUC 0.706 while dropping every "
        "SONICS generator by 0.12-0.26 (report §9.3). MIXTURE_MIN should match/beat 0.706 on "
        "FakeMusicCaps while keeping SONICS at its single-flow levels — that is the success criterion."
    )


if __name__ == "__main__":
    main()
