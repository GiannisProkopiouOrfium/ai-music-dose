#!/usr/bin/env bash
# =============================================================================
# run_genre_balanced_flow.sh
#
# Train the EnCodec RealNVP flow with inverse-genre-frequency WEIGHTED
# sampling on the SAME 12,722-track SONICS-real corpus used for the headline
# result — isolating BALANCING from B2's broadening (which added ~24,985
# external tracks). No windows are added or removed here; only which windows
# get sampled more/less often during training changes.
#
# Hypothesis: if the genre confound is mainly a representation-imbalance
# problem (some genres are 10-100x rarer in training), balanced sampling
# should close the weak-genre gap (classical, orchestral, baroque, gothic,
# fusion, crunk, country, smooth jazz, new age) with LESS data cost than B2,
# since it doesn't dilute genres that are already well-represented.
#
# Comparison points:
#   SONICS-only, uniform sampling (headline): data/processed/full_sonics_all/
#   SONICS-only, genre-balanced sampling (new): data/processed/genre_balanced_flow/
#   SONICS + external, uniform (B2, broadening): data/processed/combined_flow_eval/
#
# Prereqs: scripts/tag_all_genres.py must have been run (all_genres.csv).
#
# Usage:
#   bash scripts/run_genre_balanced_flow.sh
# =============================================================================
set -euo pipefail

REPO="$HOME/intrinsic-ai-music-detection"
cd "$REPO"

S3="s3://orfium-dev-research-projects/ai-music-intrinsic-dimension"
SONICS_MANIFEST="data/processed/canonical_sonics_full/canonical_manifest.csv"
GENRE_CSV="data/processed/genre_tags_all/all_genres.csv"
EMB_CACHE="data/emb_cache_encodec_full"   # same cache as headline / B2 — no re-extraction needed
OUT_DIR="data/processed/genre_balanced_flow"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/genre_balanced_flow_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo; echo "=== [$(date '+%F %T')] $* ==="; echo; }

if [[ ! -f "$GENRE_CSV" ]]; then
    echo "ERROR: Genre CSV not found: $GENRE_CSV"
    echo "Run scripts/tag_all_genres.py first."
    exit 1
fi
if [[ ! -f "$SONICS_MANIFEST" ]]; then
    echo "ERROR: SONICS canonical manifest not found: $SONICS_MANIFEST"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Train the genre-balanced flow on the SAME SONICS-only real corpus
# ---------------------------------------------------------------------------
log "Step 1 — Train genre-balanced flow (inverse-frequency weighted sampling)"

python scripts/run_balanced_ablation.py \
    --embeddings          encodec \
    --per-stratum         50000 \
    --analysis-duration   55 \
    --window-duration     4 \
    --hop-duration        2 \
    --estimators          twonn \
    --canonical-manifest  "$SONICS_MANIFEST" \
    --preprocess-mode     preprocessed \
    --embedding-cache-dir "$EMB_CACHE" \
    --device              cuda \
    --window-flow-eval \
    --wf-window-duration   4.0 \
    --wf-hop-duration      2.0 \
    --wf-pca-components    128 \
    --wf-flow-epochs       200 \
    --wf-save-flow-path    "$OUT_DIR/genre_balanced_flow.pt" \
    --genre-csv            "$GENRE_CSV" \
    --wf-genre-balanced-sampling \
    --genre-weight-cap      20.0 \
    --seed                42 \
    --output-dir          "$OUT_DIR" \
    2>&1

log "Genre-balanced flow training + eval complete"

# ---------------------------------------------------------------------------
# Step 2: Comparison report — headline (uniform) vs genre-balanced vs B2 (broadened)
# ---------------------------------------------------------------------------
log "Step 2 — Comparison: uniform vs genre-balanced vs broadened (B2)"

python3 - <<'PYEOF'
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def per_gen_auc(csv_path, score_substr="wf_mean"):
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path, low_memory=False)
    col = next((c for c in df.columns if score_substr in c), None)
    if col is None:
        return {}
    real_s = df[df["label"] == "real"][col].dropna().to_numpy()
    out = {}
    for alg in sorted(df[df["label"] == "fake"]["algorithm"].dropna().unique()):
        fake_s = df[(df["label"] == "fake") & (df["algorithm"] == alg)][col].dropna().to_numpy()
        if len(fake_s) < 5:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        s = np.r_[-real_s, -fake_s]   # wf_mean is log-lik (higher=real) -> negate for anomaly-AUC
        out[alg] = float(roc_auc_score(y, s))
    return out

def per_genre_auc(wf_csv, genre_csv, score_substr="wf_mean", min_n=15):
    if not Path(wf_csv).exists() or not Path(genre_csv).exists():
        return {}
    df = pd.read_csv(wf_csv, low_memory=False)
    col = next((c for c in df.columns if score_substr in c), None)
    if col is None:
        return {}
    g = pd.read_csv(genre_csv, low_memory=False)[["track_id", "genre_top1"]].drop_duplicates("track_id")
    g["track_id"] = g["track_id"].astype(str)
    df["track_id"] = df["track_id"].astype(str)
    merged = df.merge(g, on="track_id", how="left")
    out = {}
    for genre, sub in merged.dropna(subset=["genre_top1", col]).groupby("genre_top1"):
        real_s = sub[sub["label"] == "real"][col].to_numpy()
        fake_s = sub[sub["label"] == "fake"][col].to_numpy()
        if len(real_s) < min_n or len(fake_s) < min_n:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        s = np.r_[-real_s, -fake_s]
        try:
            out[genre] = (float(roc_auc_score(y, s)), len(real_s), len(fake_s))
        except Exception:
            pass
    return out

headline = per_gen_auc("data/processed/full_sonics_all/window_flow_eval.csv")
balanced = per_gen_auc("data/processed/genre_balanced_flow/window_flow_eval.csv")
broadened = per_gen_auc("data/processed/combined_flow_eval/window_flow_eval.csv")

print("\n=== Per-generator AUC: uniform (headline) vs genre-balanced vs broadened (B2) ===")
print(f"{'Generator':<22} {'Uniform':>9} {'Balanced':>9} {'Δ bal':>8}  {'Broadened':>10} {'Δ broad':>8}")
print("-" * 74)
for alg in sorted(set(list(headline) + list(balanced) + list(broadened))):
    h, b, d = headline.get(alg, float("nan")), balanced.get(alg, float("nan")), broadened.get(alg, float("nan"))
    print(f"  {alg:<20} {h:>9.4f} {b:>9.4f} {(b-h):>+8.4f}  {d:>10.4f} {(d-h):>+8.4f}")

WEAK_GENRES = ["classical", "orchestral", "baroque", "gothic", "fusion", "crunk", "country", "smooth jazz", "new age"]
h_g = per_genre_auc("data/processed/full_sonics_all/window_flow_eval.csv", "data/processed/genre_tags_all/all_genres.csv")
b_g = per_genre_auc("data/processed/genre_balanced_flow/window_flow_eval.csv", "data/processed/genre_tags_all/all_genres.csv")
d_g = per_genre_auc("data/processed/combined_flow_eval/window_flow_eval.csv", "data/processed/genre_tags_all/all_genres.csv")

print("\n=== Per-GENRE AUC (the 9 previously weak genres): uniform vs genre-balanced vs broadened ===")
print(f"{'Genre':<15} {'n_real':>7} {'Uniform':>9} {'Balanced':>9} {'Δ bal':>8}  {'Broadened':>10} {'Δ broad':>8}")
print("-" * 82)
for genre in WEAK_GENRES:
    h = h_g.get(genre)
    b = b_g.get(genre)
    d = d_g.get(genre)
    h_auc = h[0] if h else float("nan")
    n_real = h[1] if h else (b[1] if b else 0)
    b_auc = b[0] if b else float("nan")
    d_auc = d[0] if d else float("nan")
    print(f"  {genre:<13} {n_real:>7} {h_auc:>9.4f} {b_auc:>9.4f} {(b_auc-h_auc):>+8.4f}  {d_auc:>10.4f} {(d_auc-h_auc):>+8.4f}")

print("\nInterpretation:")
print("  Δ bal > 0 and larger than Δ broad  -> balancing beats broadening for that genre (cheaper fix).")
print("  Δ bal < 0 for strong genres         -> expected cost of balancing (less relative weight on them).")
print("  If overall pooled AUC drops a lot while weak-genre AUC rises a little, balancing over-corrected —")
print("  consider lowering --genre-weight-cap (default 20x) and re-running.")
PYEOF

aws s3 sync "$OUT_DIR/" "$S3/processed/genre_balanced_flow/" --no-progress || true

log "Done. Compare uniform vs genre-balanced vs broadened results above."
