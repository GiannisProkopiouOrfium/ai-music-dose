"""Per-dimension NLL attribution and temporal anomaly localization (Plan Phase D2).

Two complementary analyses:

D2a — Per-dimension NLL contribution:
  For each of the 128 EnCodec dimensions, compute the mean NLL contribution
  separately for real and fake windows.  The "discriminative dimensions" are
  those with the largest real-fake gap.

  Why this matters scientifically:
  - If top dimensions correlate with frequency bands -> bandwidth confound.
  - If they correlate with time-position within a frame -> periodic upsampling.
  - If they are spread evenly across the 128 dims -> truly distributed codec-
    level statistics, not reducible to a simple spectral feature.

D2b — Temporal anomaly localization:
  For a sample of tracks, plot the per-window anomaly score over time.
  Identify whether the anomaly clusters in:
    (a) random positions -> noisy/distributed signal
    (b) rhythmic positions -> periodic quantizer artifacts
    (c) intro/outro -> energy envelope differences

  Cross-correlate window anomaly scores with:
    - Window RMS energy (strong beat = high energy -> higher anomaly?)
    - Window spectral flatness (tonal vs noisy)
    - Window tempo estimate (does anomaly correlate with beat structure?)

D2c — Genre / covariate correlation of per-window anomaly:
  Do high-anomaly windows in REAL tracks correlate with unusual genres/
  production styles (supporting the manifold-coverage explanation of FPR)?

Outputs
-------
  per_dim_nll_gap.csv        real-vs-fake NLL gap per EnCodec dimension
  per_dim_nll_plot.png       bar chart of per-dim gaps
  temporal_examples/         per-track temporal trajectory plots
  temporal_stats.csv         aggregate temporal statistics (mean window, peak pos, etc.)

Usage (EC2)
-----------
poetry run python scripts/analyze_anomaly_attribution.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --flow-path data/processed/full_sonics_all/sonics_real_flow_encodec.pt \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --covariate-csv data/processed/covariate_profile_canonical/covariate_profiles.csv \\
    --output-dir reports/anomaly_attribution \\
    --device cuda \\
    --n-tracks-per-class 200
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ENCODEC_FPS = 75
WINDOW_DUR = 4.0
HOP_DUR = 2.0
ENCODEC_SR = 24_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_cache_health(
    track_ids: list[str],
    emb_cache: Path,
    max_duration: float,
    n_probe: int = 50,
    min_hit_rate: float = 0.5,
) -> None:
    """Fail fast if the embedding cache key doesn't match (see coverage_curve.py)."""
    probe_ids = track_ids[:n_probe]
    hits = 0
    for tid in probe_ids:
        key = hashlib.md5(f"{tid}_{ENCODEC_SR}_{max_duration}_preprocessed".encode()).hexdigest()
        if (emb_cache / "encodec" / f"{key}.npy").exists():
            hits += 1
    hit_rate = hits / max(len(probe_ids), 1)
    logger.info(
        "Cache health check: %d/%d probed tracks hit the cache (%.0f%%) at %s",
        hits,
        len(probe_ids),
        100 * hit_rate,
        emb_cache / "encodec",
    )
    if hit_rate < min_hit_rate:
        logger.error(
            "Cache hit rate %.0f%% is below the %.0f%% threshold. Check --max-duration "
            "(currently %.1f) and --emb-cache. Aborting before wasting compute.",
            100 * hit_rate,
            100 * min_hit_rate,
            max_duration,
        )
        sys.exit(1)


def _pool_windows_full(emb: np.ndarray, wf: int, hf: int) -> list[np.ndarray]:
    """Return list of per-window mean-pooled embeddings (one per window)."""
    windows = []
    n = len(emb)
    start = 0
    while start + wf <= n:
        sub = emb[start : start + wf]
        fin = sub[np.isfinite(sub).all(axis=1)]
        if len(fin) >= 1:
            windows.append(fin.mean(axis=0))
        start += hf
    return windows


def _load_track_all_windows(tid: str, emb_cache: Path, max_dur: float) -> np.ndarray | None:
    wf = max(int(WINDOW_DUR * ENCODEC_FPS), 5)
    hf = max(int(HOP_DUR * ENCODEC_FPS), 1)
    key = hashlib.md5(f"{tid}_{ENCODEC_SR}_{max_dur}_preprocessed".encode()).hexdigest()
    cp = emb_cache / "encodec" / f"{key}.npy"
    if not cp.exists():
        return None
    try:
        mat = np.load(str(cp))
        wins = _pool_windows_full(mat, wf, hf)
        return np.array(wins, dtype=np.float32) if len(wins) >= 2 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# D2a: Per-dimension NLL contribution
# ---------------------------------------------------------------------------


def compute_per_dim_nll(
    flow,
    real_track_ids: list[str],
    fake_track_ids: list[str],
    emb_cache: Path,
    max_dur: float,
    device: str,
    n_tracks_per_class: int = 200,
) -> pd.DataFrame:
    """Per-EnCodec-dim NLL gap between real and fake windows."""
    import torch

    rng = np.random.default_rng(42)
    real_sample = rng.choice(real_track_ids, size=min(n_tracks_per_class, len(real_track_ids)), replace=False)
    fake_sample = rng.choice(fake_track_ids, size=min(n_tracks_per_class, len(fake_track_ids)), replace=False)

    def _collect_windows(ids: list[str]) -> np.ndarray:
        all_w = []
        for tid in ids:
            w = _load_track_all_windows(tid, emb_cache, max_dur)
            if w is not None:
                all_w.append(w)
        return np.concatenate(all_w, axis=0) if all_w else np.empty((0, 128))

    logger.info("Collecting real windows for per-dim NLL ...")
    real_wins = _collect_windows(real_sample.tolist())
    logger.info("Collecting fake windows for per-dim NLL ...")
    fake_wins = _collect_windows(fake_sample.tolist())

    if len(real_wins) < 10 or len(fake_wins) < 10:
        logger.error("Not enough windows for per-dim NLL analysis.")
        return pd.DataFrame()

    # Standardize using flow's internal params
    mu = flow._mu.astype(np.float32)
    sd = flow._sd.astype(np.float32)

    def _zdim_nll(wins: np.ndarray) -> np.ndarray:
        """Per-dim mean NLL (Gaussian base distribution approximation)."""
        z = (wins - mu) / sd  # standardized
        # log prob per dim: -0.5*(z^2 + log(2π))
        return -0.5 * (z**2 + np.log(2 * np.pi)).mean(axis=0)  # [128]

    real_nll_per_dim = _zdim_nll(real_wins)
    fake_nll_per_dim = _zdim_nll(fake_wins)

    gap = fake_nll_per_dim - real_nll_per_dim  # negative = fakes are lower NLL than reals in this dim

    df = pd.DataFrame(
        {
            "dim": np.arange(len(real_nll_per_dim)),
            "real_nll_per_dim": real_nll_per_dim,
            "fake_nll_per_dim": fake_nll_per_dim,
            "gap_fake_minus_real": gap,  # negative = dim is MORE anomalous for fakes
            "abs_gap": np.abs(gap),
        }
    ).sort_values("abs_gap", ascending=False)

    logger.info("Top 10 most discriminative EnCodec dimensions:")
    logger.info("\n%s", df.head(10).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# D2b: Temporal anomaly localization
# ---------------------------------------------------------------------------


def temporal_anomaly_localization(
    flow,
    real_track_ids: list[str],
    fake_track_ids_by_gen: dict[str, list[str]],
    emb_cache: Path,
    max_dur: float,
    output_dir: Path,
    n_tracks_per_class: int = 50,
) -> pd.DataFrame:
    """Per-window anomaly score statistics: where in the track do anomalies peak?"""
    rng = np.random.default_rng(42)
    temporal_dir = output_dir / "temporal_examples"
    temporal_dir.mkdir(exist_ok=True)

    rows = []

    def _process_tracks(track_ids: list[str], label: str, alg: str) -> None:
        sample = rng.choice(track_ids, size=min(n_tracks_per_class, len(track_ids)), replace=False)
        for tid in sample:
            pw = _load_track_all_windows(tid, emb_cache, max_dur)
            if pw is None:
                continue
            scores = -flow.score_samples(pw)  # loglik per window (higher = more real)
            anomaly = -scores  # anomaly (higher = more fake-like)
            fin = anomaly[np.isfinite(anomaly)]
            if len(fin) < 2:
                continue

            n_wins = len(fin)
            t = np.arange(n_wins) * HOP_DUR  # time in seconds

            rows.append(
                {
                    "track_id": tid,
                    "label": label,
                    "algorithm": alg,
                    "n_windows": n_wins,
                    "anomaly_mean": float(fin.mean()),
                    "anomaly_std": float(fin.std()),
                    "anomaly_max": float(fin.max()),
                    "peak_window_idx": int(np.argmax(fin)),
                    "peak_time_s": float(t[np.argmax(fin)]),
                    "peak_frac": float(np.argmax(fin) / n_wins),  # 0=intro, 0.5=middle, 1=outro
                    "early_mean": float(fin[: max(1, n_wins // 4)].mean()),  # first quartile
                    "late_mean": float(fin[-max(1, n_wins // 4) :].mean()),  # last quartile
                    "early_late_diff": float(fin[-max(1, n_wins // 4) :].mean() - fin[: max(1, n_wins // 4)].mean()),
                }
            )

    _process_tracks(real_track_ids, "real", "real")
    for alg, ids in fake_track_ids_by_gen.items():
        _process_tracks(ids, "fake", alg)

    df = pd.DataFrame(rows)

    if not df.empty:
        logger.info("\nTemporal anomaly statistics (mean over tracks):")
        logger.info(
            "\n%s",
            df.groupby("algorithm")
            .agg(
                {
                    "anomaly_mean": "mean",
                    "anomaly_std": "mean",
                    "peak_frac": "mean",  # where in track does peak anomaly occur?
                    "early_late_diff": "mean",  # is anomaly increasing or decreasing?
                }
            )
            .round(4)
            .to_string(),
        )

        logger.info(
            "\nInterpretation:"
            "\n  peak_frac ≈ 0: anomaly peaks at track intro (energy/production differences)"
            "\n  peak_frac ≈ 0.5: anomaly in middle (rhythmic structure)"
            "\n  peak_frac ≈ 1: anomaly at outro"
            "\n  early_late_diff > 0: anomaly INCREASES over track (gradual artifact build-up)"
        )

    return df


# ---------------------------------------------------------------------------
# D2c: Correlate window anomaly with genre / covariates
# ---------------------------------------------------------------------------


def correlate_anomaly_with_genre(
    df_temporal: pd.DataFrame,
    genres_df: pd.DataFrame | None,
    covariates_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Correlate track-level anomaly scores with genre and acoustic covariates."""
    if df_temporal.empty:
        return pd.DataFrame()

    merged = df_temporal.copy()
    merged["track_id"] = merged["track_id"].astype(str)

    if genres_df is not None and not genres_df.empty:
        genres_df["track_id"] = genres_df["track_id"].astype(str)
        merged = merged.merge(genres_df[["track_id", "genre_top1", "confidence_top1"]], on="track_id", how="left")

    cov_cols = []
    if covariates_df is not None and not covariates_df.empty:
        covariates_df["track_id"] = covariates_df["track_id"].astype(str)
        avail = [
            c
            for c in ["tempo", "loudness_lufs", "spectral_centroid_mean", "effective_bandwidth_hz"]
            if c in covariates_df.columns
        ]
        merged = merged.merge(covariates_df[["track_id"] + avail], on="track_id", how="left")
        cov_cols = avail

    # Correlations between anomaly_mean and acoustic covariates
    from scipy import stats as sp_stats

    corr_rows = []
    for cov in cov_cols:
        if cov not in merged.columns:
            continue
        sub = merged[["anomaly_mean", cov]].dropna()
        if len(sub) < 20:
            continue
        r, p = sp_stats.pearsonr(sub["anomaly_mean"], sub[cov])
        corr_rows.append({"covariate": cov, "pearson_r": round(r, 4), "p_value": round(p, 6), "n": len(sub)})
        logger.info("  anomaly vs %-35s  r=%.4f  p=%.4f  n=%d", cov, r, p, len(sub))

    # Per-genre mean anomaly
    if "genre_top1" in merged.columns:
        genre_anomaly = (
            merged.groupby(["label", "genre_top1"])["anomaly_mean"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "anomaly_mean", "std": "anomaly_std", "count": "n"})
            .reset_index()
            .sort_values("anomaly_mean", ascending=False)
        )
        logger.info(
            "\nAnomaly by genre (top 10 highest anomaly genres):\n%s", genre_anomaly.head(10).to_string(index=False)
        )

    return pd.DataFrame(corr_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-dimension NLL attribution and temporal anomaly localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--window-flow-csv", required=True)
    parser.add_argument("--emb-cache", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--sonics-manifest", required=True)
    parser.add_argument("--genre-csv", default=None)
    parser.add_argument("--covariate-csv", default=None)
    parser.add_argument("--output-dir", default="reports/anomaly_attribution")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-tracks-per-class", type=int, default=200)
    parser.add_argument("--max-duration", type=float, default=55.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = Path(args.emb_cache)

    # Load flow
    from intrinsic_ai_music_detection.models.flow import RealNVPOneClass

    logger.info("Loading flow from %s ...", args.flow_path)
    flow = RealNVPOneClass.load(args.flow_path, device=args.device)

    # Load manifest
    manifest = pd.read_csv(args.sonics_manifest, low_memory=False)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"]
    manifest["track_id"] = manifest["track_id"].astype(str)

    real_ids = manifest.loc[manifest["label"] == "real", "track_id"].tolist()
    _check_cache_health(real_ids, emb_cache, args.max_duration)

    fake_by_gen: dict[str, list[str]] = {}
    for alg, grp in manifest[manifest["label"] == "fake"].groupby("algorithm"):
        fake_by_gen[str(alg)] = grp["track_id"].tolist()

    all_fake_ids = manifest.loc[manifest["label"] == "fake", "track_id"].tolist()

    # Load optional tables
    genres_df = (
        pd.read_csv(args.genre_csv, low_memory=False) if args.genre_csv and Path(args.genre_csv).exists() else None
    )
    covariate_df = (
        pd.read_csv(args.covariate_csv, low_memory=False)
        if args.covariate_csv and Path(args.covariate_csv).exists()
        else None
    )

    # ===========================================================
    # D2a: Per-dimension NLL
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("D2a: Per-dimension NLL attribution")
    logger.info("=" * 60)
    dim_df = compute_per_dim_nll(
        flow,
        real_ids,
        all_fake_ids,
        emb_cache,
        args.max_duration,
        args.device,
        n_tracks_per_class=args.n_tracks_per_class,
    )
    if not dim_df.empty:
        dim_df.to_csv(output_dir / "per_dim_nll_gap.csv", index=False)
        try:
            from intrinsic_ai_music_detection.visualization.visualize import plot_per_dimension_nll

            plot_per_dimension_nll(
                dim_df.sort_values("dim")["real_nll_per_dim"].values,
                title="Real-vs-fake NLL gap per EnCodec dimension (real)",
                save_path=output_dir / "per_dim_nll_real.png",
            )
            plot_per_dimension_nll(
                dim_df.sort_values("dim")["fake_nll_per_dim"].values,
                title="Per-EnCodec-dimension NLL (fake)",
                save_path=output_dir / "per_dim_nll_fake.png",
            )
            # Gap plot
            import matplotlib.pyplot as plt

            gap = dim_df.sort_values("dim")["gap_fake_minus_real"].values
            fig, ax = plt.subplots(figsize=(14, 4))
            colors = ["#F44336" if g < 0 else "#2196F3" for g in gap]
            ax.bar(range(len(gap)), gap, color=colors, alpha=0.8)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel("EnCodec dimension index")
            ax.set_ylabel("NLL gap (fake − real)\nnegative = fakes more anomalous in this dim")
            ax.set_title("Per-dimension discriminativity: negative (red) dims separate real from fake")
            plt.tight_layout()
            fig.savefig(output_dir / "per_dim_nll_gap.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Could not save per-dim plots: %s", exc)

    # ===========================================================
    # D2b: Temporal localization
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("D2b: Temporal anomaly localization")
    logger.info("=" * 60)
    temporal_df = temporal_anomaly_localization(
        flow,
        real_ids,
        fake_by_gen,
        emb_cache,
        args.max_duration,
        output_dir,
        n_tracks_per_class=min(args.n_tracks_per_class, 100),
    )
    if not temporal_df.empty:
        temporal_df.to_csv(output_dir / "temporal_stats.csv", index=False)
        # Temporal trajectory plot for a few example tracks
        try:
            import matplotlib.pyplot as plt

            from intrinsic_ai_music_detection.visualization.visualize import plot_temporal_anomaly_trajectory

            sample_tracks = temporal_df.groupby("algorithm").head(2)
            for _, row in sample_tracks.iterrows():
                tid = row["track_id"]
                pw = _load_track_all_windows(tid, emb_cache, args.max_duration)
                if pw is None:
                    continue
                scores = -flow.score_samples(pw)
                anomaly = -scores
                plot_temporal_anomaly_trajectory(
                    anomaly,
                    label=f"{row['algorithm']} ({row['label']})",
                    save_path=output_dir / "temporal_examples" / f"trajectory_{row['algorithm']}_{tid[:8]}.png",
                )
        except Exception as exc:
            logger.warning("Temporal trajectory plots failed: %s", exc)

    # ===========================================================
    # D2c: Genre / covariate correlation
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("D2c: Anomaly vs genre / covariate correlation")
    logger.info("=" * 60)
    if not temporal_df.empty:
        corr_df = correlate_anomaly_with_genre(temporal_df, genres_df, covariate_df)
        if not corr_df.empty:
            corr_df.to_csv(output_dir / "anomaly_covariate_correlation.csv", index=False)

    logger.info("\nAll outputs saved to: %s", output_dir)
    logger.info(
        "Key outputs:\n"
        "  per_dim_nll_gap.csv          – per-EnCodec-dim NLL gap\n"
        "  per_dim_nll_gap.png          – bar chart\n"
        "  temporal_stats.csv           – where in track does anomaly peak?\n"
        "  temporal_examples/           – per-track trajectory plots\n"
        "  anomaly_covariate_correlation.csv – Pearson r vs covariates"
    )


if __name__ == "__main__":
    main()
