"""Download a small SONICS sample for sanity check.

Downloads 5 fake songs from the smallest HF zip part,
and 5 real songs from S3, then runs the full pipeline:
- Multiple embedding models (EnCodec, CLAP, MERT)
- ID estimation (PHD, TwoNN, MLE)
- Fourier fakeprint baseline
- Classification (LogReg, SVM, threshold)
- Statistical evaluation (Mann-Whitney, Cohen's d, KS)
"""

import os
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="skdim")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SONICS_DIR = Path("data/raw/sonics")
FAKE_DIR = SONICS_DIR / "fake_songs"
REAL_DIR = SONICS_DIR / "real_songs"
META_DIR = SONICS_DIR / "metadata"

N_SAMPLES = 5


def download_fake_sample():
    """Download part_10.zip (smallest) and extract N_SAMPLES songs."""
    from huggingface_hub import hf_hub_download

    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we already have enough fakes
    existing = list(FAKE_DIR.glob("*.mp3"))
    if len(existing) >= N_SAMPLES:
        print(f"Already have {len(existing)} fake songs, skipping download")
        return

    print("Downloading part_10.zip from HuggingFace (~2.7GB)...")
    zip_path = hf_hub_download(
        repo_id="awsaf49/sonics",
        repo_type="dataset",
        filename="fake_songs/part_10.zip",
        local_dir=str(SONICS_DIR / "hf_download"),
    )
    print(f"Downloaded to: {zip_path}")

    # Extract just N_SAMPLES mp3 files
    print(f"Extracting {N_SAMPLES} songs...")
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        mp3_names = [n for n in zf.namelist() if n.endswith(".mp3")]
        print(f"  Zip contains {len(mp3_names)} mp3 files")
        for name in mp3_names[:N_SAMPLES]:
            zf.extract(name, str(FAKE_DIR / "_tmp"))
            src = FAKE_DIR / "_tmp" / name
            dst = FAKE_DIR / Path(name).name
            shutil.move(str(src), str(dst))
            extracted += 1
            print(f"  Extracted: {dst.name}")

    # Clean up temp dir
    tmp = FAKE_DIR / "_tmp"
    if tmp.exists():
        shutil.rmtree(str(tmp))

    print(f"Extracted {extracted} fake songs")


def download_real_sample():
    """Download real songs from S3 using youtube_ids from real_songs.csv."""
    import subprocess

    REAL_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(REAL_DIR.glob("*.*"))
    if len(existing) >= N_SAMPLES:
        print(f"Already have {len(existing)} real songs, skipping download")
        return

    real_csv = META_DIR / "real_songs.csv"
    if not real_csv.exists():
        print("ERROR: real_songs.csv not found")
        return

    df = pd.read_csv(real_csv)
    # Get test split songs for consistency
    test_songs = df[df["split"] == "test"] if "split" in df.columns else df
    youtube_ids = test_songs["youtube_id"].dropna().unique()[:20]  # try up to 20

    downloaded = 0
    s3_prefix = "s3://ai-plagiarism-data/ai-downloads"

    for yt_id in youtube_ids:
        if downloaded >= N_SAMPLES:
            break

        yt_id = str(yt_id).strip()
        if not yt_id:
            continue

        # Check if already exists
        if list(REAL_DIR.glob(f"{yt_id}.*")):
            downloaded += 1
            continue

        # Try m4a and mp3
        for ext in [".m4a", ".mp3", ".wav"]:
            s3_key = f"{s3_prefix}/{yt_id}{ext}"
            local_path = REAL_DIR / f"{yt_id}{ext}"
            try:
                result = subprocess.run(
                    ["aws", "s3", "cp", s3_key, str(local_path), "--region", "eu-west-1"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    print(f"  Downloaded real: {yt_id}{ext}")
                    downloaded += 1
                    break
            except subprocess.TimeoutExpired:
                continue

    print(f"Downloaded {downloaded} real songs from S3")


def _collect_audio_files():
    """Collect fake and real audio files."""
    fake_files = sorted(FAKE_DIR.glob("*.mp3"))[:N_SAMPLES]
    real_files = sorted(f for f in REAL_DIR.iterdir() if f.suffix in {".mp3", ".m4a", ".wav"})[:N_SAMPLES]

    if not fake_files or not real_files:
        print("ERROR: Not enough audio files for sanity check")
        print(f"  Fake files: {len(fake_files)}")
        print(f"  Real files: {len(real_files)}")
        return None, None
    return real_files, fake_files


def run_sanity_check():
    """Run the full pipeline sanity check on the SONICS sample."""
    from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio
    from intrinsic_ai_music_detection.features.embeddings import get_extractor
    from intrinsic_ai_music_detection.features.fakeprints import compute_fakeprint
    from intrinsic_ai_music_detection.features.id_estimators import estimate_all
    from intrinsic_ai_music_detection.models.evaluate import compare_distributions, compute_binary_metrics
    from intrinsic_ai_music_detection.models.train_model import train_and_evaluate, train_threshold_classifier

    real_files, fake_files = _collect_audio_files()
    if real_files is None:
        return

    all_files = [(f, "real") for f in real_files] + [(f, "fake") for f in fake_files]
    print(f"\nSanity check: {len(real_files)} real, {len(fake_files)} fake")

    # ── 1. Load audio once at native SR, resample on demand ────────────
    print("\n[1/5] Loading audio...")
    import librosa

    raw_audio = {}  # {filename: (audio_native, native_sr)}

    for f, label in all_files:
        key = f.name
        try:
            audio_native, sr_native = librosa.load(str(f), sr=None, mono=True, duration=120)
            raw_audio[key] = (audio_native.astype(np.float32), sr_native)
            print(f"  Loaded: {key} ({sr_native}Hz, {len(audio_native)/sr_native:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {key} — {e}")

    def get_audio(key, target_sr):
        """Resample cached audio to target sample rate."""
        audio, sr = raw_audio[key]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return normalize_audio(audio), target_sr

    # ── 2. Embedding extraction + ID estimation ─────────────────────────
    embedding_configs = [
        ("encodec", 24000, "euclidean"),
        ("mert", 24000, "cosine"),
        ("clap", 48000, "cosine"),
    ]

    # Try MuQ if available
    try:
        import muq  # noqa: F401

        embedding_configs.append(("muq", 24000, "cosine"))
    except ImportError:
        print("  [INFO] muq not installed, skipping MuQ embeddings")

    all_id_results = []

    for emb_name, sr, metric in embedding_configs:
        print(f"\n[2/5] Extracting {emb_name.upper()} embeddings...")
        t0 = time.time()
        extractor = get_extractor(emb_name, device="cpu")

        for f, label in all_files:
            key = f.name
            if key not in raw_audio:
                continue
            try:
                audio, loaded_sr = get_audio(key, sr)
                emb = extractor.extract(audio, loaded_sr)
                ids = estimate_all(emb, metric=metric)

                row = {
                    "file": key,
                    "label": label,
                    "embedding": emb_name,
                    "emb_shape": str(emb.shape),
                    **{f"{m}": v for m, v in ids.items()},
                }
                all_id_results.append(row)

                phd = ids.get("phd", float("nan"))
                twonn = ids.get("twonn", float("nan"))
                mle = ids.get("mle", float("nan"))
                print(f"  {label:4s} | {key[:30]:30s} | {emb_name:7s} | PHD={phd:.2f} TwoNN={twonn:.2f} MLE={mle:.2f}")
            except Exception as e:
                print(f"  {label:4s} | {key[:30]:30s} | {emb_name:7s} | ERROR: {e}")

        print(f"  {emb_name.upper()} done in {time.time() - t0:.1f}s")

    # ── 3. Fourier Fakeprint baseline ────────────────────────────────────
    print("\n[3/5] Computing Fourier fakeprints...")
    fakeprint_results = []
    for f, label in all_files:
        key = f.name
        if key not in raw_audio:
            continue
        try:
            audio, sr = get_audio(key, 44100)
            fp = compute_fakeprint(audio, sr)
            fp_energy = float(np.mean(fp))
            fp_max = float(np.max(fp))
            fp_nonzero = float(np.count_nonzero(fp > 0.1) / len(fp))
            fakeprint_results.append(
                {
                    "file": key,
                    "label": label,
                    "fp_mean": fp_energy,
                    "fp_max": fp_max,
                    "fp_active_ratio": fp_nonzero,
                    "fp_len": len(fp),
                }
            )
            print(f"  {label:4s} | {key[:30]:30s} | mean={fp_energy:.4f} max={fp_max:.4f} active={fp_nonzero:.2%}")
        except Exception as e:
            print(f"  {label:4s} | {key[:30]:30s} | FAKEPRINT ERROR: {e}")

    # ── 4. Statistical analysis ──────────────────────────────────────────
    print(f"\n{'='*75}")
    print("[4/5] STATISTICAL ANALYSIS")
    print(f"{'='*75}")

    id_df = pd.DataFrame(all_id_results)

    for emb_name in id_df["embedding"].unique():
        emb_df = id_df[id_df["embedding"] == emb_name]
        real_df = emb_df[emb_df["label"] == "real"]
        fake_df = emb_df[emb_df["label"] == "fake"]

        print(f"\n  --- {emb_name.upper()} ---")
        for method in ["phd", "twonn", "mle"]:
            if method not in emb_df.columns:
                continue
            real_vals = real_df[method].dropna().values.astype(np.float64)
            fake_vals = fake_df[method].dropna().values.astype(np.float64)
            if len(real_vals) < 2 or len(fake_vals) < 2:
                continue

            result = compare_distributions(real_vals, fake_vals)
            diff = result.human_mean - result.ai_mean
            tag = "real > fake" if diff > 0 else "fake > real"
            print(
                f"  {method.upper():>6s}: Real={result.human_mean:.3f}±{result.human_std:.3f}  "
                f"Fake={result.ai_mean:.3f}±{result.ai_std:.3f}  "
                f"d={result.cohens_d:+.3f}  MW-p={result.mann_whitney_p:.3e}  [{tag}]"
            )

    if fakeprint_results:
        fp_df = pd.DataFrame(fakeprint_results)
        real_fp = fp_df[fp_df["label"] == "real"]["fp_mean"].values
        fake_fp = fp_df[fp_df["label"] == "fake"]["fp_mean"].values
        if len(real_fp) >= 2 and len(fake_fp) >= 2:
            fp_cmp = compare_distributions(real_fp, fake_fp)
            print(f"\n  --- FAKEPRINT ---")
            print(
                f"  fp_mean: Real={fp_cmp.human_mean:.4f}±{fp_cmp.human_std:.4f}  "
                f"Fake={fp_cmp.ai_mean:.4f}±{fp_cmp.ai_std:.4f}  "
                f"d={fp_cmp.cohens_d:+.3f}  MW-p={fp_cmp.mann_whitney_p:.3e}"
            )

    # ── 5. Classification ────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("[5/5] CLASSIFICATION (tiny sample — illustrative only)")
    print(f"{'='*75}")

    for emb_name in id_df["embedding"].unique():
        emb_df = id_df[id_df["embedding"] == emb_name]
        y = (emb_df["label"] == "fake").astype(int).values

        methods = [m for m in ["phd", "twonn", "mle"] if m in emb_df.columns]
        if not methods:
            continue

        print(f"\n  --- {emb_name.upper()} ---")

        # Threshold classifiers per method
        for m in methods:
            vals = emb_df[m].dropna().values
            if len(vals) != len(y):
                continue
            thresh, acc = train_threshold_classifier(vals, y, direction="lower")
            print(f"    Threshold ({m.upper()}): acc={acc:.3f}  thresh={thresh:.2f}")

        # Multi-ID classifier (if enough samples for 2-fold)
        X = emb_df[methods].dropna().values
        if len(X) == len(y) and len(X) >= 4:
            for clf_type in ["logistic_regression", "svm"]:
                try:
                    result = train_and_evaluate(
                        X,
                        y,
                        classifier_type=clf_type,
                        feature_set_name=f"{emb_name}_all_ids",
                        n_folds=min(2, len(X) // 2),
                    )
                    print(
                        f"    {clf_type}: acc={result.cv_accuracy:.3f}  "
                        f"f1={result.cv_f1:.3f}  auc={result.cv_auc:.3f}"
                    )
                except Exception as e:
                    print(f"    {clf_type}: FAILED — {e}")

    # ── FST baseline check ───────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("FST BASELINE STATUS")
    print(f"{'='*75}")
    fst_repo = Path("vendors/fst")
    if fst_repo.exists():
        ckpt1 = fst_repo / "checkpoints" / "stage1_mert_audiocat.ckpt"
        ckpt2 = fst_repo / "checkpoints" / "stage2_fst.ckpt"
        print(f"  Repo: EXISTS at {fst_repo}")
        print(f"  Stage-1 ckpt: {'EXISTS' if ckpt1.exists() else 'MISSING'}")
        print(f"  Stage-2 ckpt: {'EXISTS' if ckpt2.exists() else 'MISSING'}")
        if ckpt1.exists() and ckpt2.exists():
            print("  [INFO] FST is ready — will run in full EC2 experiment")
        else:
            print("  [INFO] Download checkpoints from Google Drive before EC2 run")
    else:
        print(f"  Repo: NOT FOUND at {fst_repo}")
        print("  [INFO] Clone repo + download checkpoints for EC2:")
        print("    git clone https://github.com/Mippia/FST-AI-Music-Detection.git vendors/fst")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("SANITY CHECK COMPLETE")
    print(f"{'='*75}")
    embeddings_tested = list(id_df["embedding"].unique())
    print(f"  Embeddings tested: {embeddings_tested}")
    print(f"  ID methods: PHD, TwoNN, MLE")
    print(f"  Fakeprints: {'YES' if fakeprint_results else 'NO'}")
    print(f"  Classification: threshold + LogReg + SVM")
    print(f"  Total ID results: {len(all_id_results)}")
    print(f"  NOTE: N={N_SAMPLES} per class — results are illustrative, not statistical")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SONICS sanity check — full pipeline")
    parser.add_argument("--skip-download-fakes", action="store_true")
    parser.add_argument("--skip-download-reals", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    # Clear sys.argv so laion-clap doesn't hijack argparse on import
    sys.argv = sys.argv[:1]

    if not args.skip_download_fakes:
        download_fake_sample()

    if not args.skip_download_reals:
        download_real_sample()

    if not args.skip_check:
        run_sanity_check()
