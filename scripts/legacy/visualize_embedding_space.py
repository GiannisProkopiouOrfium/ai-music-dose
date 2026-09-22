"""Visualize the EnCodec embedding space with UMAP / t-SNE (Plan Phase D1).

Generates publication-quality scatter plots of the pooled 4s-window EnCodec
embeddings, colored by:
  1. Real vs fake (binary label)
  2. Generator (5 AI generators + real)
  3. CLAP genre (top-1 genre from tag_all_genres.py)
  4. Anomaly score (continuous heatmap)

These visualizations serve two purposes:
  (a) Scientific: show which factors separate real from fake in latent space.
  (b) Paper figure: a striking UMAP that illustrates the "real manifold" concept.

Usage (EC2)
-----------
# Full UMAP (subsample 10k points, ~20 min on CPU or 2 min on GPU):
poetry run python scripts/visualize_embedding_space.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --genre-csv data/processed/genre_tags_all/all_genres.csv \\
    --output-dir reports/figures/embedding_space \\
    --method umap \\
    --n-max 15000

# Quick t-SNE (fast, no umap-learn needed):
poetry run python scripts/visualize_embedding_space.py \\
    --window-flow-csv data/processed/full_sonics_all/window_flow_eval.csv \\
    --emb-cache data/emb_cache_encodec_full \\
    --sonics-manifest data/processed/canonical_sonics_full/canonical_manifest.csv \\
    --output-dir reports/figures/embedding_space \\
    --method tsne \\
    --n-max 5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _pool_windows(emb: np.ndarray, wf: int, hf: int) -> np.ndarray:
    pooled = []
    n = len(emb)
    start = 0
    while start + wf <= n:
        sub = emb[start : start + wf]
        fin = sub[np.isfinite(sub).all(axis=1)]
        if len(fin) >= 1:
            pooled.append(fin.mean(axis=0))
        start += hf
    return np.array(pooled, dtype=np.float32) if pooled else np.empty((0, emb.shape[1]), dtype=np.float32)


def _load_track_pooled(track_id: str, emb_cache: Path, max_dur: float) -> np.ndarray | None:
    """Load one pooled window vector per track (track-level mean of all windows)."""
    wf = max(int(WINDOW_DUR * ENCODEC_FPS), 5)
    hf = max(int(HOP_DUR * ENCODEC_FPS), 1)
    key = hashlib.md5(f"{track_id}_{ENCODEC_SR}_{max_dur}_preprocessed".encode()).hexdigest()
    cp = emb_cache / "encodec" / f"{key}.npy"
    if not cp.exists():
        return None
    try:
        mat = np.load(str(cp))
        pw = _pool_windows(mat, wf, hf)
        if len(pw) >= 2:
            return pw.mean(axis=0).astype(np.float32)  # track-level mean embedding
        return None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMAP/t-SNE visualization of EnCodec embedding space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--window-flow-csv", required=True)
    parser.add_argument("--emb-cache", required=True)
    parser.add_argument("--sonics-manifest", required=True)
    parser.add_argument("--genre-csv", default=None)
    parser.add_argument("--output-dir", default="reports/figures/embedding_space")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--n-max", type=int, default=12000, help="Max samples to embed (stratified subsample)")
    parser.add_argument("--max-duration", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = Path(args.emb_cache)

    # --- Load manifest + window-flow scores ---
    manifest = pd.read_csv(args.sonics_manifest, low_memory=False)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"]
    manifest["track_id"] = manifest["track_id"].astype(str)

    wf = pd.read_csv(args.window_flow_csv, low_memory=False)
    wf["track_id"] = wf["track_id"].astype(str)
    wf_col = next((c for c in wf.columns if "wf_mean" in c), None)

    # --- Load genre tags ---
    genres_df = None
    if args.genre_csv and Path(args.genre_csv).exists():
        genres_df = pd.read_csv(args.genre_csv, low_memory=False)
        genres_df["track_id"] = genres_df["track_id"].astype(str)
        logger.info("Loaded genre tags: %d rows", len(genres_df))

    # --- Collect track-level embeddings ---
    logger.info("Loading track-level pooled embeddings from cache (%d tracks) ...", len(manifest))
    embeddings, track_ids_loaded = [], []
    for _, row in manifest.iterrows():
        tid = str(row["track_id"])
        emb = _load_track_pooled(tid, emb_cache, args.max_duration)
        if emb is not None:
            embeddings.append(emb)
            track_ids_loaded.append(tid)

    if not embeddings:
        logger.error("No embeddings loaded. Check --emb-cache path and that tracks are cached.")
        sys.exit(1)

    X = np.vstack(embeddings)
    logger.info("Loaded %d track embeddings (dim=%d)", len(X), X.shape[1])

    # Build attribute arrays aligned with X
    track_id_to_row = manifest.set_index("track_id")
    wf_scores = wf.set_index("track_id")[wf_col] if wf_col else None

    labels, generators, genres, anomaly_scores = [], [], [], []
    for tid in track_ids_loaded:
        row = track_id_to_row.loc[tid] if tid in track_id_to_row.index else None
        label = str(row["label"]) if row is not None else "unknown"
        alg = str(row.get("algorithm", "real")) if row is not None else "real"
        generator = alg if label == "fake" else "real"
        labels.append(label)
        generators.append(generator)

        # Genre
        if genres_df is not None and tid in genres_df.set_index("track_id").index:
            genre = genres_df.set_index("track_id").loc[tid]["genre_top1"]
        else:
            genre = "unknown"
        genres.append(str(genre))

        # Anomaly score (from window_flow_eval)
        if wf_scores is not None and tid in wf_scores.index:
            anomaly_scores.append(float(-wf_scores.loc[tid]))  # anomaly = -wf_mean
        else:
            anomaly_scores.append(float("nan"))

    anomaly_arr = np.array(anomaly_scores)

    # --- Save coordinates (so UMAP doesn't need to be re-run for every color scheme) ---
    from intrinsic_ai_music_detection.visualization.visualize import _reduce_embeddings

    logger.info("Computing %s reduction for %d -> %d samples ...", args.method, len(X), min(len(X), args.n_max))
    coords = None
    coords_cache = output_dir / f"embedding_coords_{args.method}.npy"
    if coords_cache.exists():
        logger.info("Loading cached %s coordinates from %s", args.method, coords_cache)
        coords_data = np.load(str(coords_cache), allow_pickle=True).item()
        coords = coords_data["coords"]
        idx_used = coords_data["idx"]
        X_used = X[idx_used]
        labels_used = np.array(labels)[idx_used]
        gen_used = np.array(generators)[idx_used]
        genres_used = np.array(genres)[idx_used]
        anomaly_used = anomaly_arr[idx_used]
    else:
        # Stratified subsample
        rng = np.random.default_rng(args.seed)
        all_labels = np.array(generators)
        uniq_gens = np.unique(all_labels)
        per_gen = max(args.n_max // len(uniq_gens), 10)
        idx_list = []
        for g in uniq_gens:
            mask = all_labels == g
            n_g = mask.sum()
            n_sel = min(per_gen, n_g)
            idx_list.append(rng.choice(np.where(mask)[0], size=n_sel, replace=False))
        idx_used = np.concatenate(idx_list)
        rng.shuffle(idx_used)

        X_used = X[idx_used].astype(np.float64)
        labels_used = np.array(labels)[idx_used]
        gen_used = np.array(generators)[idx_used]
        genres_used = np.array(genres)[idx_used]
        anomaly_used = anomaly_arr[idx_used]

        coords = _reduce_embeddings(X_used, method=args.method, seed=args.seed)
        np.save(str(coords_cache), {"coords": coords, "idx": idx_used})
        logger.info("Saved %s coordinates to %s", args.method, coords_cache)

    # --- Save coordinate table ---
    from intrinsic_ai_music_detection.visualization.visualize import plot_embedding_space

    coord_df = pd.DataFrame(
        {
            f"{args.method}_1": coords[:, 0],
            f"{args.method}_2": coords[:, 1],
            "label": labels_used,
            "generator": gen_used,
            "genre": genres_used,
            "anomaly_score": anomaly_used,
        }
    )
    coord_df.to_csv(output_dir / f"embedding_coords_{args.method}.csv", index=False)

    # --- Figure 1: colored by generator ---
    logger.info("Plotting Fig 1: colored by generator ...")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 7))
    gens_uniq = sorted(coord_df["generator"].unique())
    palette = dict(zip(gens_uniq, sns.color_palette("tab10", n_colors=len(gens_uniq))))
    for gen in gens_uniq:
        sub = coord_df[coord_df["generator"] == gen]
        ax.scatter(
            sub[f"{args.method}_1"], sub[f"{args.method}_2"], c=[palette[gen]], s=5, alpha=0.5, label=gen, linewidths=0
        )
    ax.legend(markerscale=4, fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"EnCodec embedding space ({args.method.upper()}) — colored by generator\n" f"n={len(coord_df):,}")
    ax.set_xlabel(f"{args.method.upper()} dim 1")
    ax.set_ylabel(f"{args.method.upper()} dim 2")
    plt.tight_layout()
    fig.savefig(output_dir / f"umap_by_generator.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: colored by anomaly score ---
    logger.info("Plotting Fig 2: colored by anomaly score ...")
    fin_mask = np.isfinite(coord_df["anomaly_score"].values)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        coord_df.loc[fin_mask, f"{args.method}_1"],
        coord_df.loc[fin_mask, f"{args.method}_2"],
        c=coord_df.loc[fin_mask, "anomaly_score"].values,
        cmap="RdYlGn_r",
        s=5,
        alpha=0.5,
        vmin=np.percentile(coord_df.loc[fin_mask, "anomaly_score"], 2),
        vmax=np.percentile(coord_df.loc[fin_mask, "anomaly_score"], 98),
    )
    plt.colorbar(sc, ax=ax, label="Anomaly score (−mean loglik; higher = more fake)")
    ax.set_title(f"EnCodec embedding space ({args.method.upper()}) — anomaly score heatmap\n" f"n={fin_mask.sum():,}")
    ax.set_xlabel(f"{args.method.upper()} dim 1")
    ax.set_ylabel(f"{args.method.upper()} dim 2")
    plt.tight_layout()
    fig.savefig(output_dir / "umap_anomaly_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: colored by genre (if available) ---
    if genres_df is not None:
        logger.info("Plotting Fig 3: colored by genre (top 10 genres) ...")
        top_genres = coord_df["genre"].value_counts().head(10).index.tolist()
        coord_df["genre_plot"] = coord_df["genre"].apply(lambda g: g if g in top_genres else "other")
        fig, ax = plt.subplots(figsize=(10, 7))
        g_uniq = sorted(coord_df["genre_plot"].unique())
        palette_g = dict(zip(g_uniq, sns.color_palette("tab20", n_colors=len(g_uniq))))
        for g in g_uniq:
            sub = coord_df[coord_df["genre_plot"] == g]
            alpha = 0.3 if g == "other" else 0.7
            s = 3 if g == "other" else 8
            ax.scatter(
                sub[f"{args.method}_1"],
                sub[f"{args.method}_2"],
                c=[palette_g[g]],
                s=s,
                alpha=alpha,
                label=g,
                linewidths=0,
            )
        ax.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_title(
            f"EnCodec embedding space ({args.method.upper()}) — colored by CLAP genre\n"
            f"n={len(coord_df):,} (top 10 genres highlighted)"
        )
        ax.set_xlabel(f"{args.method.upper()} dim 1")
        ax.set_ylabel(f"{args.method.upper()} dim 2")
        plt.tight_layout()
        fig.savefig(output_dir / "umap_by_genre.png", dpi=150)
        plt.close(fig)

    logger.info("All embedding visualizations saved to: %s", output_dir)
    logger.info("Files:\n  umap_by_generator.png\n  umap_anomaly_heatmap.png\n  umap_by_genre.png")


if __name__ == "__main__":
    main()
