"""Validate whether hf_energy_ratio=1.0 AUC is a genuine codec artifact
or a data collection artifact (real audio downloaded at higher quality).

Test 1 (LP degradation): degrade real tracks by lowpass at target_bw_hz to
simulate codec bandwidth limits, then recompute hf_energy_ratio.  If it drops
to ~0, the HF difference is a format confound.

Test 2 (canonical comparison): if --canonical-csv is supplied (acoustic
features from the canonical preprocessing run), compare hf_energy_ratio and
other bandwidth-sensitive features between raw and canonical runs to measure
how much each feature changes after the MP3 round-trip equalises bandwidth.

Usage:
  # Classic LP test:
  poetry run python scripts/validate_hf_ratio.py \
    --acoustic-csv data/processed/acoustic_analysis_raw/acoustic_features.csv \
    --n-samples 100 \
    --output-dir data/processed/hf_validation

  # With canonical comparison:
  poetry run python scripts/validate_hf_ratio.py \
    --acoustic-csv data/processed/acoustic_analysis_raw/acoustic_features.csv \
    --canonical-csv data/processed/acoustic_analysis_canonical/acoustic_features.csv \
    --n-samples 100 \
    --output-dir data/processed/hf_validation
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.signal

ANALYSIS_SR = 24_000
MAX_DURATION = 60.0
HF_THRESHOLD = 8000  # Hz — same as acoustic script


def hf_energy_ratio(audio: np.ndarray, sr: int) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        S = np.abs(librosa.stft(audio, hop_length=512))
    freq_res = sr / (2 * S.shape[0])
    hf_bin = int(HF_THRESHOLD / freq_res)
    hf_e = float(np.sum(S[hf_bin:] ** 2))
    tot_e = float(np.sum(S**2))
    return hf_e / tot_e if tot_e > 0 else np.nan


def degrade_to_bandwidth(audio: np.ndarray, sr: int, target_bw_hz: int) -> np.ndarray:
    """Simulate codec bandwidth limit by lowpass at target_bw_hz."""
    nyq = sr / 2
    cutoff = min(target_bw_hz / nyq, 0.999)
    b, a = scipy.signal.butter(8, cutoff, btype="low")
    return scipy.signal.filtfilt(b, a, audio).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-csv", required=True, help="acoustic_features.csv with path column")
    parser.add_argument(
        "--canonical-csv", default=None, help="acoustic_features.csv from canonical run (for comparison)"
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of real tracks to test")
    parser.add_argument("--output-dir", default="data/processed/hf_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.acoustic_csv, low_memory=False)
    real_df = df[df["label"] == "real"].sample(n=min(args.n_samples, len(df[df["label"] == "real"])), random_state=42)
    fake_df = df[df["label"] == "fake"].sample(n=min(args.n_samples, len(df[df["label"] == "fake"])), random_state=42)

    # Bandwidth limits to test
    bandwidths = {
        "original_24kHz": None,  # no degradation
        "lowpass_8kHz": 8000,
        "lowpass_6kHz": 6000,
        "lowpass_4kHz": 4000,
    }

    rows = []
    for bw_name, bw_hz in bandwidths.items():
        print(f"\n--- {bw_name} ---")
        for label, subset in [("real", real_df), ("fake", fake_df)]:
            ratios = []
            for _, row in subset.iterrows():
                path = row.get("path", "")
                if not path or not Path(path).exists():
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        audio, sr = librosa.load(path, sr=ANALYSIS_SR, mono=True, duration=MAX_DURATION)
                    audio = audio.astype(np.float32)
                    if bw_hz is not None:
                        audio = degrade_to_bandwidth(audio, sr, bw_hz)
                    ratio = hf_energy_ratio(audio, sr)
                    if np.isfinite(ratio):
                        ratios.append(ratio)
                except Exception as e:
                    print(f"  Skipped {path}: {e}")
                    continue

            if ratios:
                mean_r = np.mean(ratios)
                std_r = np.std(ratios)
                print(f"  {label:<6}  mean={mean_r:.6f}  std={std_r:.6f}  n={len(ratios)}")
                rows.append(
                    {
                        "bandwidth": bw_name,
                        "label": label,
                        "mean_hf_ratio": mean_r,
                        "std_hf_ratio": std_r,
                        "n": len(ratios),
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "hf_validation.csv", index=False)
    print(f"\nSaved hf_validation.csv")

    # Interpretation guide
    print("\n=== Interpretation (LP degradation test) ===")
    orig_real = out_df[(out_df["bandwidth"] == "original_24kHz") & (out_df["label"] == "real")]["mean_hf_ratio"].values
    orig_fake = out_df[(out_df["bandwidth"] == "original_24kHz") & (out_df["label"] == "fake")]["mean_hf_ratio"].values
    lp8_real = out_df[(out_df["bandwidth"] == "lowpass_8kHz") & (out_df["label"] == "real")]["mean_hf_ratio"].values
    if len(orig_real) and len(orig_fake) and len(lp8_real):
        print(f"  Real original:      {orig_real[0]:.6f}")
        print(f"  Fake original:      {orig_fake[0]:.6f}")
        print(f"  Real after LP@8kHz: {lp8_real[0]:.6f}")
        if lp8_real[0] < orig_fake[0] * 1.5:
            print("  → FORMAT CONFOUND: degraded real ≈ fake → HF ratio is encoding artifact")
        else:
            print("  → GENUINE SIGNAL: degraded real still >> fake → codec bandwidth limitation confirmed")

        drop_fraction = 1 - (lp8_real[0] / (orig_real[0] + 1e-9))
        print(f"  HF content removed by LP@8kHz: {drop_fraction*100:.1f}%")
        if drop_fraction > 0.95:
            print("  → CONFOUND LIKELY: LP@8kHz removes 95%+ of HF → signal is format-dependent")
        elif drop_fraction > 0.5:
            print("  → MIXED: LP@8kHz removes substantial HF → partially format-dependent")
        else:
            print("  → GENUINE SIGNAL: HF persists after bandwidth matching")

    # -----------------------------------------------------------------------
    # Canonical comparison: compare raw vs canonical feature distributions
    # -----------------------------------------------------------------------
    if args.canonical_csv:
        canon_path = Path(args.canonical_csv)
        if not canon_path.exists():
            print(f"\n--canonical-csv not found: {args.canonical_csv}")
        else:
            import numpy as np
            from sklearn.metrics import roc_auc_score

            df_raw_full = df
            df_can = pd.read_csv(canon_path, low_memory=False)

            bandwidth_sensitive = [
                "hf_energy_ratio",
                "spectral_flatness_mean",
                "spectral_centroid_mean",
                "spectral_rolloff_mean",
                "zcr_mean",
                "zcr_cv",
            ]

            print("\n=== Canonical comparison: raw vs canonical feature distributions ===")
            print(f"{'feature':<35}  {'auc_raw':>8}  {'auc_canon':>9}  {'drop':>6}  status")
            print("-" * 80)

            canon_rows = []
            for feat in bandwidth_sensitive:
                auc_raw = auc_can = np.nan

                if feat in df_raw_full.columns:
                    y_r = (df_raw_full["label"] == "fake").astype(int).to_numpy()
                    x_r = pd.to_numeric(df_raw_full[feat], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(x_r)
                    if mask.sum() >= 20 and len(np.unique(y_r[mask])) == 2:
                        raw_auc = roc_auc_score(y_r[mask], x_r[mask])
                        auc_raw = max(raw_auc, 1 - raw_auc)

                if feat in df_can.columns:
                    y_c = (df_can["label"] == "fake").astype(int).to_numpy()
                    x_c = pd.to_numeric(df_can[feat], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(x_c)
                    if mask.sum() >= 20 and len(np.unique(y_c[mask])) == 2:
                        can_auc = roc_auc_score(y_c[mask], x_c[mask])
                        auc_can = max(can_auc, 1 - can_auc)

                drop = (auc_raw - auc_can) if (np.isfinite(auc_raw) and np.isfinite(auc_can)) else np.nan
                status = "FORMAT_CONFOUND" if (np.isfinite(drop) and drop > 0.10) else "ok"
                print(f"{feat:<35}  {auc_raw:8.3f}  {auc_can:9.3f}  {drop:+6.3f}  {status}")
                canon_rows.append(
                    {"feature": feat, "auc_raw": auc_raw, "auc_canonical": auc_can, "auc_drop": drop, "status": status}
                )

            pd.DataFrame(canon_rows).to_csv(output_dir / "hf_canonical_comparison.csv", index=False)
            print(f"\nSaved hf_canonical_comparison.csv")


if __name__ == "__main__":
    main()
