"""Train a generator-agnostic detector on real music alone, using synthesised negatives.

The method, in one line
-----------------------
Fit one flow to real windows and a second flow to windows of *reconstructions of
those same reals*, and score by the log-ratio — Afchar, Meseguer-Brocal &
Hennequin's real-vs-reconstruction task (arXiv:2501.10111), realised inside our
existing ``DualFlowDetector`` so that no fake-music label is used anywhere.

Why this and not more one-class score correction
------------------------------------------------
Three label-free corrections from the OOD literature have already been tried and
all three failed (typicality, complexity compensation, background likelihood
ratio — §5 of the handover), and the diagnosis was that reweighting the same
likelihood cannot separate synthesis from corpus identity, because a real-only
model never sees which direction "synthetic" points in. Proxy negatives supply
that direction from real audio alone.

It stays zero-shot in the sense that matters
--------------------------------------------
No output of any AI music generator is used in training. The detector is trained
before the generators it must catch exist, which is the property that has to hold
when a new model ships next quarter. State this precisely in the paper: it is
**generator-zero-shot**, not unsupervised — the negatives are synthetic, and the
claim is not the same as the one-class flow's.

The confound cannot follow
--------------------------
A reconstruction shares its source's content, genre, tempo, loudness and
bandwidth exactly, so the learned boundary cannot be a corpus or channel
boundary. That matters here specifically: on canonical SONICS a two-line channel
statistic reaches AUC 1.0000, so any boundary learned between *different* tracks
is suspect, and this one is not learned between different tracks.

Protocol
--------
Real tracks are split disjointly: the TRAIN half supplies both the positives and
(via reconstruction) the negatives; the EVAL half is scored and never seen. Fakes
are only ever scored, never trained on. The bandwidth control is applied
identically to positives, negatives and every evaluated track.

Usage (EC2)
-----------
    python scripts/run_proxy_negative_detector.py \
        --train-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
        --eval-manifest  data/processed/canonical_sonics_full/canonical_manifest.csv \
        --n-train-real 1500 --per-stratum-eval 800 \
        --resample-hz 15000 --device cuda \
        --output-dir data/processed/proxy_neg_sonics

    # cross-corpus: train the negatives on SONICS reals, score FakeMusicCaps
    python scripts/run_proxy_negative_detector.py \
        --train-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \
        --eval-manifest  data/processed/canonical_fmc_native/combined_manifest.csv \
        --n-train-real 1500 --per-stratum-eval 800 \
        --resample-hz 15000 --device cuda \
        --output-dir data/processed/proxy_neg_sonics2fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.data.audio_preprocessing import bandwidth_match  # noqa: E402
from intrinsic_ai_music_detection.data.audio_utils import load_audio  # noqa: E402
from intrinsic_ai_music_detection.features.cache_keys import band_match_tag, embedding_cache_key  # noqa: E402
from intrinsic_ai_music_detection.features.embeddings import get_extractor  # noqa: E402
from intrinsic_ai_music_detection.features.pooling import pool_windows  # noqa: E402
from intrinsic_ai_music_detection.features.proxy_negatives import RECON_TYPES, synthesise_negative  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer  # noqa: E402
from intrinsic_ai_music_detection.models.flow import DualFlowDetector, RealNVPConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("proxy_negative_detector")

EMBEDDING_SR = {"encodec": 24_000, "spec": 24_000, "logmel": 24_000, "spec-musicdet": 16_000}


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"]
    for col in ("track_id", "label", "canonical_path"):
        if col not in df.columns:
            raise SystemExit(f"{path} lacks required column {col!r} (has {list(df.columns)})")
    df["track_id"] = df["track_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["algorithm"] = df.get("algorithm", pd.Series([""] * len(df))).fillna("").astype(str)
    if df.empty:
        raise SystemExit(f"no usable rows in {path}")
    return df


def stratified_sample(df: pd.DataFrame, per_stratum: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby(["label", "algorithm"], sort=True):
        n = min(per_stratum, len(grp))
        parts.append(grp.iloc[np.sort(rng.choice(len(grp), size=n, replace=False))])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Embedding with a control-aware cache
# ---------------------------------------------------------------------------


class WindowEmbedder:
    """Embed audio into pooled windows, caching per (track, variant, control).

    The cache key carries the bandwidth control and the reconstruction variant.
    Omitting either would let a controlled run silently read an uncontrolled
    run's vectors — the failure mode that has invalidated four sets of results in
    this project (§8.4).
    """

    def __init__(
        self,
        embedding: str,
        cache_dir: Path,
        device: str,
        max_duration: float,
        window_duration: float,
        hop_duration: float,
        resample_hz: float | None,
        lowpass_hz: float | None,
    ) -> None:
        self.embedding = embedding
        self.cache_dir = cache_dir / embedding
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.target_sr = EMBEDDING_SR.get(embedding, 24_000)
        self.max_duration = max_duration
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.resample_hz = resample_hz
        self.lowpass_hz = lowpass_hz
        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            logger.info("Loading %s extractor on %s", self.embedding, self.device)
            self._extractor = get_extractor(self.embedding, device=self.device)
        return self._extractor

    def _key(self, track_id: str, variant: str) -> str:
        tags = tuple(t for t in (band_match_tag(self.resample_hz),) if t)
        if self.lowpass_hz:
            tags = tags + (f"_lp{int(self.lowpass_hz)}",)
        if variant != "real":
            tags = tags + (f"_v{variant}",)
        return embedding_cache_key(track_id, self.target_sr, self.max_duration, "preprocessed", extra_tags=tags)

    def windows(self, track_id: str, path: str, variant: str = "real") -> np.ndarray | None:
        """Return ``[n_windows, d]`` for one track/variant, or ``None`` on failure."""
        cache_path = self.cache_dir / f"{self._key(track_id, variant)}.npy"
        if cache_path.exists():
            try:
                frames = np.load(str(cache_path))
                return self._pool(frames)
            except Exception as exc:  # noqa: BLE001
                logger.debug("cache read failed for %s/%s: %s", track_id, variant, exc)

        try:
            audio, sr = load_audio(path, target_sr=self.target_sr, max_duration=self.max_duration)
        except Exception as exc:  # noqa: BLE001
            logger.debug("load failed for %s: %s", path, exc)
            return None

        if variant == "real":
            if self.resample_hz or self.lowpass_hz:
                audio = bandwidth_match(audio, sr, resample_hz=self.resample_hz, lowpass_hz=self.lowpass_hz)
        else:
            try:
                audio = synthesise_negative(
                    audio,
                    sr,
                    variant,
                    device=self.device,
                    resample_hz=self.resample_hz,
                    lowpass_hz=self.lowpass_hz,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("negative synthesis failed for %s/%s: %s", track_id, variant, exc)
                return None

        try:
            frames = np.asarray(self.extractor.extract(audio, sr), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.debug("extract failed for %s/%s: %s", track_id, variant, exc)
            return None
        if frames.size == 0:
            return None

        # Write through an open handle, NOT np.save(str(path)): np.save appends
        # ".npy" to any filename that does not already end in it, so a temp file
        # named "<key>.npy.tmp" was silently written as "<key>.npy.tmp.npy" and
        # the rename then failed with FileNotFoundError.
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, frames)
        tmp.replace(cache_path)
        return self._pool(frames)

    def _pool(self, frames: np.ndarray) -> np.ndarray | None:
        if frames.ndim != 2 or len(frames) == 0:
            return None
        # Derive the frame rate from the array itself rather than assuming 75 Hz:
        # spec-musicdet runs at 100 fps and a hard-coded rate would silently
        # window 3 s of audio while claiming 4.
        fps = len(frames) / max(self.max_duration, 1e-6)
        wf = max(int(self.window_duration * fps), 5)
        hf = max(int(self.hop_duration * fps), 1)
        pooled = pool_windows(frames, wf, hf, mode="mean")
        return pooled if len(pooled) else None


def collect(
    embedder: WindowEmbedder, rows: pd.DataFrame, variant: str, desc: str
) -> tuple[list[str], list[np.ndarray]]:
    ids: list[str] = []
    wins: list[np.ndarray] = []
    n_fail = 0
    for i, row in enumerate(rows.itertuples(index=False), 1):
        pw = embedder.windows(str(row.track_id), str(row.canonical_path), variant=variant)
        if pw is None:
            n_fail += 1
            continue
        ids.append(str(row.track_id))
        wins.append(pw)
        if i % 200 == 0 or i == len(rows):
            logger.info("  %s [%s]: %d/%d (%d failed)", desc, variant, i, len(rows), n_fail)
    if not wins:
        raise SystemExit(
            f"{desc}/{variant}: ZERO tracks produced windows. Refusing to continue — an empty "
            f"stage that returns quietly is how four sets of results in this project were "
            f"invalidated (§8.4)."
        )
    if n_fail > 0.2 * len(rows):
        logger.warning(
            "%s/%s: %d/%d tracks failed (>20%%). A class-skewed failure rate silently "
            "rebalances the comparison — check before trusting the AUCs.",
            desc,
            variant,
            n_fail,
            len(rows),
        )
    return ids, wins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generator-agnostic detector trained on real music plus synthesised negatives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-manifest", required=True, help="supplies REAL training tracks")
    parser.add_argument("--eval-manifest", required=True, help="scored; may be a different corpus")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding", default="encodec", choices=sorted(EMBEDDING_SR))
    parser.add_argument("--cache-dir", default="data/emb_cache_proxyneg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-train-real", type=int, default=1500)
    parser.add_argument("--per-stratum-eval", type=int, default=800)
    parser.add_argument(
        "--recon-types",
        nargs="+",
        default=["encodec_3kbps", "encodec_6kbps", "griffinlim_128mel"],
        choices=list(RECON_TYPES),
        help=(
            "Decoder families used to synthesise negatives. Use SEVERAL: training on one teaches "
            "'this is EnCodec' rather than 'this is a neural decoder', and the whole point is "
            "generalisation across decoder families."
        ),
    )
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=15000.0,
        help=(
            "Shared bandwidth ceiling, applied to positives, negatives and every evaluated track. "
            "Default 15000 (Nyquist 7.5 kHz) is the rate measured on 2026-08-13 to bring SONICS' "
            "channel-alone AUC from 1.0000 to 0.5778. Pass 0 to disable (NOT recommended: the "
            "resulting numbers are partly a delivery-chain measurement)."
        ),
    )
    parser.add_argument("--lowpass-hz", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--window-duration", type=float, default=4.0)
    parser.add_argument("--hop-duration", type=float, default=2.0)
    parser.add_argument("--n-coupling-layers", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--flow-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-detector", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resample_hz = args.resample_hz if args.resample_hz and args.resample_hz > 0 else None
    if resample_hz is None:
        logger.warning(
            "NO bandwidth control. On canonical SONICS a single channel descriptor reaches "
            "AUC 1.0000 without one, so these numbers cannot be described as measuring synthesis."
        )

    train_df = load_manifest(Path(args.train_manifest))
    eval_df = load_manifest(Path(args.eval_manifest))

    rng = np.random.default_rng(args.seed)
    train_reals = train_df[train_df["label"] == "real"]
    if len(train_reals) < 50:
        raise SystemExit(f"only {len(train_reals)} real tracks in --train-manifest; need >= 50")

    # Disjoint split. When train and eval are the same corpus, the eval reals
    # must never have contributed a training window OR a negative — otherwise the
    # detector is scored on audio whose reconstruction it memorised.
    same_corpus = Path(args.train_manifest).resolve() == Path(args.eval_manifest).resolve()
    perm = rng.permutation(len(train_reals))
    n_train = min(args.n_train_real, len(train_reals) if not same_corpus else len(train_reals) // 2)
    train_ids = set(train_reals.iloc[np.sort(perm[:n_train])]["track_id"])
    train_rows = train_reals[train_reals["track_id"].isin(train_ids)]
    logger.info("train reals: %d of %d available (same_corpus=%s)", len(train_rows), len(train_reals), same_corpus)

    if same_corpus:
        eval_df = eval_df[~((eval_df["label"] == "real") & (eval_df["track_id"].isin(train_ids)))]
        logger.info("eval set after removing training reals: %d rows", len(eval_df))
    eval_rows = stratified_sample(eval_df, args.per_stratum_eval, args.seed)

    embedder = WindowEmbedder(
        embedding=args.embedding,
        cache_dir=Path(args.cache_dir),
        device=args.device,
        max_duration=args.max_duration,
        window_duration=args.window_duration,
        hop_duration=args.hop_duration,
        resample_hz=resample_hz,
        lowpass_hz=args.lowpass_hz,
    )

    # --- positives: real training windows ---
    _, pos_wins = collect(embedder, train_rows, "real", "train")
    x_pos = np.concatenate(pos_wins, axis=0)

    # --- negatives: reconstructions of the SAME tracks ---
    neg_parts: list[np.ndarray] = []
    for recon_type in args.recon_types:
        _, wins = collect(embedder, train_rows, recon_type, "train")
        neg_parts.append(np.concatenate(wins, axis=0))
        logger.info("negatives from %s: %d windows", recon_type, len(neg_parts[-1]))
    x_neg = np.concatenate(neg_parts, axis=0)

    logger.info(
        "training set: %d real windows vs %d synthesised-negative windows (%d decoder families)",
        len(x_pos),
        len(x_neg),
        len(args.recon_types),
    )

    # --- fit the dual flow ---
    real_cfg = RealNVPConfig(
        device=args.device,
        n_epochs=args.flow_epochs,
        patience=20,
        seed=args.seed,
        n_coupling_layers=args.n_coupling_layers,
        hidden_dim=args.hidden_dim,
        verbose=True,
    )
    fake_cfg = RealNVPConfig(
        device=args.device,
        n_epochs=args.flow_epochs,
        patience=20,
        seed=args.seed + 1,
        n_coupling_layers=args.n_coupling_layers,
        hidden_dim=args.hidden_dim,
        verbose=True,
    )
    detector = DualFlowDetector(real_cfg=real_cfg, fake_cfg=fake_cfg)
    detector.fit_real(x_pos)
    detector.fit_fake(x_neg)
    if args.save_detector:
        detector.save(args.save_detector)
        logger.info("detector saved -> %s", args.save_detector)

    # --- score the eval corpus ---
    rows: list[dict] = []
    for i, row in enumerate(eval_rows.itertuples(index=False), 1):
        variant = "real"
        pw = embedder.windows(str(row.track_id), str(row.canonical_path), variant=variant)
        if pw is None:
            continue
        ratio = detector.score_samples(pw)  # higher = more fake-like
        real_only = detector.real_flow.score_samples(pw)  # the one-class baseline, same flow
        finite_r = ratio[np.isfinite(ratio)]
        finite_o = real_only[np.isfinite(real_only)]
        if len(finite_r) < 2:
            continue
        rows.append(
            {
                "track_id": str(row.track_id),
                "label": str(row.label),
                "algorithm": str(row.algorithm),
                "proxy_ratio": float(finite_r.mean()),
                "one_class_nll": float(finite_o.mean()) if len(finite_o) else float("nan"),
                "n_windows": int(len(finite_r)),
            }
        )
        if i % 500 == 0 or i == len(eval_rows):
            logger.info("  scored %d/%d", i, len(eval_rows))

    if not rows:
        raise SystemExit("no evaluation tracks scored — refusing to write an empty result")

    scores = pd.DataFrame(rows)
    scores_path = out_dir / "proxy_negative_scores.csv"
    scores.to_csv(scores_path, index=False)
    logger.info("per-track scores -> %s (%d rows)", scores_path, len(scores))

    # --- report: proxy ratio vs the one-class baseline, per generator ---
    is_real = (scores["label"] == "real").to_numpy()
    report: list[dict] = []
    for scorer in ("proxy_ratio", "one_class_nll"):
        s = scores[scorer].to_numpy(float)
        real_s = s[is_real]
        real_s = real_s[np.isfinite(real_s)]
        aucs = []
        for alg in sorted(set(scores.loc[~is_real, "algorithm"]) - {""}):
            fake_s = s[(~is_real) & (scores["algorithm"] == alg).to_numpy()]
            fake_s = fake_s[np.isfinite(fake_s)]
            if len(fake_s) < 5 or len(real_s) < 5:
                continue
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
            report.append(
                {
                    "scorer": scorer,
                    "algorithm": alg,
                    "auc": round(auc, 4),
                    "eer_pct": round(eer * 100, 2),
                    "n_real": len(real_s),
                    "n_fake": len(fake_s),
                }
            )
            aucs.append(auc)
        if aucs:
            report.append(
                {
                    "scorer": scorer,
                    "algorithm": "MACRO",
                    "auc": round(float(np.mean(aucs)), 4),
                    "eer_pct": float("nan"),
                    "n_real": len(real_s),
                    "n_fake": -1,
                }
            )

    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / "proxy_negative_auc.csv", index=False)
    pivot = rep.pivot_table(index="scorer", columns="algorithm", values="auc")
    logger.info("\nAUC by scorer x generator:\n%s", pivot.to_string())
    logger.info(
        "\nREAD: 'proxy_ratio' used ZERO generator outputs in training — its negatives are "
        "reconstructions of the training reals. 'one_class_nll' is the same real flow scored "
        "alone, i.e. the existing headline method on identical audio, so the difference between "
        "the two rows is exactly what the synthesised negatives bought."
    )
    logger.info(
        "CAVEAT to ship with any number here: this is GENERATOR-zero-shot, not unsupervised. "
        "The negatives are synthetic but they are supervision, and the claim differs from the "
        "one-class flow's. Say which one each reported number belongs to."
    )


if __name__ == "__main__":
    main()
