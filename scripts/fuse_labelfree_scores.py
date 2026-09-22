"""Label-free score fusion: codec-latent flow density + universal-compressor complexity.

Motivation (from the 2026-08-07 results): on SONICS the flow NLL and a
parameter-free FLAC-complexity statistic are strongly COMPLEMENTARY — the flow
dominates on the chirp/Suno generators (AUC 0.996/0.971/0.936) while complexity
alone dominates on udio-30s (0.982 vs the flow's 0.784), our worst case. Neither
component uses a single fake label, so their fusion is still a fully
unsupervised detector.

Every component is mapped to the empirical quantile of its own REFERENCE-real
distribution before combining (`QuantileCalibrator`), so components on wildly
different scales (log-density vs bits/second) become comparable without any
labelled tuning. Reference reals are a split-half disjoint from the evaluated
reals.

Components (all anomaly-oriented, higher = more likely AI):
  flow_nll    : -wf_mean (the headline scorer)
  complexity  : -flac_bits_per_sec  (AI music compresses smaller = less complex)
  dispersion  : |wf_std - median(reference real wf_std)| (trajectory shape)
  top25       : mean of the 25% most-anomalous windows (needs --trajectories)

Fusions: the mean of the calibrated components (primary), plus max/min variants
and every pair, so the ablation is fully visible rather than cherry-picked.

Reports per-generator AUC/EER, MACRO average, and POOLED (all fakes vs all
reals) — pooled EER is the number directly comparable to MusicDET's 9.72% on
this corpus — plus a DeLong test of each fusion against the flow-only baseline.

Usage:
    python scripts/fuse_labelfree_scores.py \\
        --window-flow-csv data/processed/full_sonics_k12_symmetric/window_flow_eval.csv \\
        --complexity-csv data/processed/full_sonics_k12_symmetric/complexity_compensated_complexity.csv \\
        --output-csv data/processed/full_sonics_k12_symmetric/labelfree_fusion.csv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.aggregate import QuantileCalibrator, aggregate_trajectory  # noqa: E402
from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, delong_test  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _precision_metrics(real_s: np.ndarray, fake_s: np.ndarray, prevalence: float | None = None) -> dict:
    """Deployment-oriented operating points, not EER.

    EER weights false alarms and misses equally, which is the wrong trade-off
    for content moderation: flagging genuine music is far costlier than missing
    a fake. These report how much fake material is caught at a FIXED, low
    false-alarm rate, plus the precision there — the numbers an operator
    actually chooses a threshold on. Thresholds come from the REAL score
    distribution only, so they remain label-free at deployment time.

    ``tpr_at_fpr*`` is prevalence-free and directly comparable across corpora.

    ``precision_at_fpr*`` is NOT. Precision depends on how many reals there are
    per fake, and in this evaluation that ratio is an artifact of how the corpus
    was sampled — SONICS evaluates ~1.3k reals against ~47k fakes, which is not a
    rate any deployment sees. Computing it from the evaluation's own class counts
    therefore produces a number that changes when the sampler changes and means
    nothing operationally.

    So ``prevalence`` (the assumed fraction of AI-generated tracks in the stream)
    is stated explicitly and precision is computed at that rate:

        precision = TPR·p / (TPR·p + FPR·(1−p))

    Passing ``None`` reproduces the historical eval-set-ratio behaviour and adds
    ``precision_prevalence`` to the row so the assumption is always visible in
    the output rather than implied.
    """
    out: dict = {}
    n_real, n_fake = len(real_s), len(fake_s)
    eval_prevalence = n_fake / (n_real + n_fake) if (n_real + n_fake) else float("nan")
    p = eval_prevalence if prevalence is None else float(prevalence)
    out["precision_prevalence"] = round(p, 6)
    out["precision_prevalence_source"] = "eval_set_ratio" if prevalence is None else "declared"

    for fpr_target in (0.01, 0.001):
        thr = float(np.percentile(real_s, 100 * (1 - fpr_target)))
        tpr = float(np.mean(fake_s > thr))  # detection rate at that alarm rate
        # Use the NOMINAL fpr_target as the false-alarm rate: it is the operating
        # point being reported, and the empirical rate at a percentile threshold
        # equals it by construction up to ties.
        denom = tpr * p + fpr_target * (1.0 - p)
        prec = (tpr * p) / denom if denom > 0 else float("nan")
        tag = f"{fpr_target:g}".replace("0.", "")
        out[f"tpr_at_fpr{tag}"] = round(tpr, 4)
        out[f"precision_at_fpr{tag}"] = round(prec, 4)
    return out


def _evaluate(
    name: str, scores: np.ndarray, is_real: np.ndarray, algos: np.ndarray, min_n: int, prevalence: float | None = None
) -> list[dict]:
    rows = []
    real_s = scores[is_real]
    real_s = real_s[np.isfinite(real_s)]
    if len(real_s) < min_n:
        return rows
    aucs, all_fake = [], []
    for alg in sorted(set(algos[~is_real]) - {""}):
        fake_s = scores[(~is_real) & (algos == alg)]
        fake_s = fake_s[np.isfinite(fake_s)]
        if len(fake_s) < min_n:
            continue
        y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
        auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
        rows.append(
            {
                "scorer": name,
                "algorithm": alg,
                "auc": round(auc, 4),
                "eer_pct": round(eer * 100, 2),
                **_precision_metrics(real_s, fake_s, prevalence),
                "n_real": len(real_s),
                "n_fake": len(fake_s),
            }
        )
        aucs.append(auc)
        all_fake.append(fake_s)
    if aucs:
        rows.append(
            {
                "scorer": name,
                "algorithm": "MACRO_AVG",
                "auc": round(float(np.mean(aucs)), 4),
                "eer_pct": float("nan"),
                "n_real": len(real_s),
                "n_fake": sum(len(f) for f in all_fake),
            }
        )
        pooled_fake = np.concatenate(all_fake)
        y = np.r_[np.zeros(len(real_s)), np.ones(len(pooled_fake))]
        auc, eer = auc_and_eer(y, np.r_[real_s, pooled_fake])
        rows.append(
            {
                "scorer": name,
                "algorithm": "POOLED",
                "auc": round(auc, 4),
                "eer_pct": round(eer * 100, 2),
                "n_real": len(real_s),
                "n_fake": len(pooled_fake),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--window-flow-csv", required=True)
    parser.add_argument(
        "--complexity-csv",
        default=None,
        help="track_id,flac_bits_per_sec (the *_complexity.csv written by eval_complexity_compensated.py)",
    )
    parser.add_argument("--trajectories", default=None, help="wf_trajectories_<emb>.npz (adds the top25 component)")
    parser.add_argument("--meta", default=None, help="wf_trajectories_<emb>_meta.csv")
    parser.add_argument(
        "--extra-scores",
        action="append",
        default=None,
        metavar="NAME=CSV",
        help=(
            "Add another flow's per-track scores as a fusion component (repeatable), e.g. "
            "'spec=data/processed/sonics_spec_pilot/window_flow_eval.csv'. The CSV must have "
            "track_id plus a *_wf_mean column (+log-likelihood orientation); it is inner-joined "
            "on track_id, so the evaluation is restricted to tracks BOTH arms scored. This is "
            "how the EnCodec and spectrogram arms get fused — both are label-free, so the "
            "combination remains a zero-shot detector."
        ),
    )
    parser.add_argument(
        "--fpr-scores",
        action="append",
        default=None,
        metavar="COMPONENT=CSV",
        help=(
            "Per-component scores for an EXTERNAL REAL corpus (e.g. FMA), one per component, "
            "e.g. 'flow_nll=.../fma_fpr_symmetric/external_scores.csv' "
            "'spec=.../fma_fpr_spec/external_scores.csv'. Each is calibrated with that "
            "component's own SONICS-real calibrator, fused identically, and the FUSED "
            "false-positive rate is reported at the SONICS operating points. This answers the "
            "question raw AUC cannot: does fusion buy in-domain accuracy AND cross-corpus "
            "generalisation, or only the former?"
        ),
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comb-csv",
        default=None,
        help=(
            "Per-track deconvolution-comb features from eval_comb_detector.py "
            "(comb_per_track*.csv). Adds the ARTIFACT branch to the fusion. Unlike the density "
            "score, its polarity is fixed a priori by the mechanism, so it is the natural "
            "partner for a likelihood whose sign is not guaranteed."
        ),
    )
    parser.add_argument(
        "--comb-feature",
        nargs="+",
        default=["comb_strength"],
        help="Which comb columns to fuse (repeatable). comb_strength and comb_sharpness are "
        "complementary per generator; comb_log_* are their frequency-scaling-invariant twins.",
    )
    parser.add_argument("--min-per-group", type=int, default=5)
    parser.add_argument(
        "--prevalence",
        type=float,
        default=None,
        help=(
            "Assumed fraction of AI-generated tracks in the stream, used for precision@FPR. "
            "Precision is prevalence-dependent, and this evaluation's real:fake ratio is a "
            "sampling artifact (SONICS scores ~1.3k reals against ~47k fakes), so leaving this "
            "unset reports precision at that meaningless rate. State the rate you mean, e.g. "
            "0.01. TPR@FPR is prevalence-free either way. The rate used is written into every "
            "output row as 'precision_prevalence'."
        ),
    )
    parser.add_argument(
        "--logo-selection",
        action="store_true",
        default=False,
        help=(
            "UNBIASED selection protocol. Reporting the best of N fusion variants chosen on the "
            "same generators it is evaluated on is test-set selection. With this flag, for each "
            "held-out generator the best fusion is chosen using ONLY the other generators, then "
            "applied to the held-out one — a leave-one-generator-out nested selection matching "
            "the zero-shot framing. Report THESE numbers in the paper; the full table is the "
            "ablation."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=["auc", "eer"],
        default="eer",
        help="Criterion used to pick the fusion on the selection generators.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.window_flow_csv, low_memory=False)
    df["track_id"] = df["track_id"].astype(str)
    mean_col = next((c for c in df.columns if c.endswith("_wf_mean") or c == "wf_mean"), None)
    if mean_col is None:
        raise SystemExit(f"no *_wf_mean column in {args.window_flow_csv}")
    std_col = mean_col.replace("_wf_mean", "_wf_std")

    df["flow_nll"] = -df[mean_col]
    components = ["flow_nll"]

    if std_col in df.columns:
        df["_wf_std"] = df[std_col]
        components.append("dispersion")

    for spec in args.extra_scores or []:
        if "=" not in spec:
            raise SystemExit(f"--extra-scores expects NAME=CSV, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            raise SystemExit(f"--extra-scores {name}: file not found: {path}")
        other = pd.read_csv(path, low_memory=False)
        if "track_id" not in other.columns:
            raise SystemExit(
                f"--extra-scores {name}: {path} has no 'track_id' column (cols: "
                f"{list(other.columns)[:8]}). Summary/aggregate tables (e.g. eval_typicality.py's "
                f"--output-csv) cannot be fused; pass a PER-TRACK score file."
            )
        other["track_id"] = other["track_id"].astype(str)
        ocol = next((c for c in other.columns if c.endswith("_wf_mean") or c == "wf_mean"), None)
        if ocol is None:
            raise SystemExit(f"--extra-scores {name}: no *_wf_mean column in {path}")
        other = other[["track_id", ocol]].drop_duplicates("track_id").rename(columns={ocol: f"_extra_{name}"})
        before = len(df)
        df = df.merge(other, on="track_id", how="inner")
        df[name] = -df[f"_extra_{name}"]  # anomaly-oriented
        components.append(name)
        logger.info(
            "component %r joined from %s: %d -> %d tracks (inner join on the tracks both arms scored)",
            name,
            path,
            before,
            len(df),
        )

    if args.comb_csv:
        if not Path(args.comb_csv).exists():
            raise SystemExit(f"--comb-csv not found: {args.comb_csv}")
        comb = pd.read_csv(args.comb_csv, low_memory=False)
        if "track_id" not in comb.columns:
            raise SystemExit(f"--comb-csv {args.comb_csv} has no track_id column")
        wanted = list(args.comb_feature)
        missing = [c for c in wanted if c not in comb.columns]
        if missing:
            raise SystemExit(
                f"--comb-feature {missing} not in {args.comb_csv} "
                f"(available: {[c for c in comb.columns if c.startswith('comb_')]})"
            )
        comb["track_id"] = comb["track_id"].astype(str)
        comb = comb[["track_id", *wanted]].drop_duplicates("track_id")
        before_merge = len(df)
        df = df.merge(comb, on="track_id", how="left")
        if len(df) != before_merge:
            raise SystemExit(f"--comb-csv merge changed the row count {before_merge} -> {len(df)}")
        n_matched = int(df[wanted[0]].notna().sum())
        logger.info("comb joined: %d/%d tracks matched", n_matched, len(df))
        if n_matched == 0:
            raise SystemExit("--comb-csv matched ZERO tracks on track_id")
        # Anomaly orientation: a STRONGER / SHARPER comb means more likely
        # generated, so these are already anomaly-oriented and used as-is. They
        # are the only components whose sign is fixed a priori by the mechanism
        # (Afchar et al., arXiv:2506.19108) rather than fitted -- which matters
        # because every fitted one-class score we have measured INVERTS.
        #
        # comb_strength and comb_sharpness are complementary per generator on
        # FakeMusicCaps: strength wins MusicGen (0.952 vs 0.514), sharpness wins
        # mustango (0.968 vs 0.932) and audioldm2 (0.923 vs 0.912). Fusing them
        # is the cheapest available improvement and needs no labels.
        for feat in wanted:
            name = feat.replace("comb_", "comb-")
            df[name] = df[feat]
            components.append(name)

    if args.complexity_csv and Path(args.complexity_csv).exists():
        comp = pd.read_csv(args.complexity_csv)
        if "track_id" not in comp.columns or "flac_bits_per_sec" not in comp.columns:
            raise SystemExit(
                f"--complexity-csv {args.complexity_csv} lacks track_id/flac_bits_per_sec "
                f"(cols: {list(comp.columns)[:8]}). That is the SUMMARY table; pass the file "
                f"written by eval_complexity_compensated.py --per-track-csv instead."
            )
        comp["track_id"] = comp["track_id"].astype(str)
        # Take ONLY the two columns we need. The per-track complexity file also
        # carries 'label', 'algorithm' and 'nll'; merging those in makes pandas
        # suffix the collisions to label_x / label_y, so `df["label"]` ceases to
        # exist and the fusion dies with KeyError: 'label' several steps later,
        # nowhere near the cause. Same family as §8.4: a silent structural change
        # that surfaces as an unrelated failure.
        comp = comp[["track_id", "flac_bits_per_sec"]].drop_duplicates("track_id")
        before_merge = len(df)
        df = df.merge(comp, on="track_id", how="left")
        if len(df) != before_merge:
            raise SystemExit(
                f"--complexity-csv merge changed the row count {before_merge} -> {len(df)}; "
                "the complexity file has duplicate track_ids and the join fanned out."
            )
        n_matched = int(df["flac_bits_per_sec"].notna().sum())
        logger.info("complexity joined: %d/%d tracks matched on track_id", n_matched, len(df))
        if n_matched == 0:
            raise SystemExit(
                "--complexity-csv matched ZERO tracks on track_id. The two files describe "
                "different corpora, or the ids are formatted differently."
            )
        df["complexity"] = -df["flac_bits_per_sec"]
        components.append("complexity")
    else:
        logger.warning("no --complexity-csv: fusion will not include the complexity component")

    if args.trajectories and args.meta and Path(args.trajectories).exists():
        npz = np.load(args.trajectories)
        meta = pd.read_csv(args.meta, low_memory=False)
        meta["track_id"] = meta["track_id"].astype(str)
        if "key" not in meta.columns:
            meta["key"] = meta["track_id"]
        if meta["key"].duplicated().any():
            logger.error("legacy colliding trajectory keys — skipping the top25 component")
        else:
            top25 = {
                r.track_id: aggregate_trajectory(npz[r.key], "top_25pct") for r in meta.itertuples() if r.key in npz
            }
            df["top25"] = df["track_id"].map(top25)
            components.append("top25")

    is_real = (df["label"] == "real").to_numpy()
    algos = df.get("algorithm", pd.Series([""] * len(df))).fillna("").astype(str).to_numpy()

    # --- split-half reals: reference (calibration) vs evaluated ---
    rng = np.random.default_rng(args.seed)
    real_pos = np.where(is_real)[0]
    perm = rng.permutation(len(real_pos))
    ref_rows = real_pos[perm[: len(real_pos) // 2]]
    eval_real_rows = real_pos[perm[len(real_pos) // 2 :]]
    keep = np.zeros(len(df), dtype=bool)
    keep[eval_real_rows] = True
    keep[~is_real] = True  # all fakes are evaluated
    logger.info(
        "components: %s | reference reals %d, evaluated reals %d, fakes %d",
        components,
        len(ref_rows),
        len(eval_real_rows),
        int((~is_real).sum()),
    )

    # --- calibrate each component on reference reals ---
    calibrated: dict[str, np.ndarray] = {}
    calibrators_used: dict[str, QuantileCalibrator] = {}
    raw_component: dict[str, np.ndarray] = {}
    for c in components:
        if c == "dispersion":
            ref_med = float(np.nanmedian(df["_wf_std"].to_numpy(float)[ref_rows]))
            raw = np.abs(df["_wf_std"].to_numpy(float) - ref_med)
        else:
            raw = df[c].to_numpy(float)
        ref_vals = raw[ref_rows]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if len(ref_vals) < 10:
            logger.warning("component %s: too few finite reference values — dropped", c)
            continue
        raw_component[c] = raw
        _cal = QuantileCalibrator().fit(ref_vals)
        calibrators_used[c] = _cal
        calibrated[c] = _cal.transform(raw)
    components = list(calibrated)

    rows: list[dict] = []
    scores_for_delong: dict[str, np.ndarray] = {}

    def _run(name: str, s: np.ndarray) -> None:
        rows.extend(_evaluate(name, s[keep], is_real[keep], algos[keep], args.min_per_group, args.prevalence))
        scores_for_delong[name] = s

    for c in components:
        _run(c, calibrated[c])

    # --- fusions: every pair, plus all-components, mean and max ---
    for r in range(2, len(components) + 1):
        for combo in itertools.combinations(components, r):
            stack = np.stack([calibrated[c] for c in combo], axis=1)
            with np.errstate(invalid="ignore"):
                _run("fuse_mean(" + "+".join(combo) + ")", np.nanmean(stack, axis=1))
                # MIN = unanimity: a track is anomalous only if EVERY component
                # says so. Conservative by construction, so it is the operator
                # to reach for when cross-corpus false alarms are the binding
                # constraint (mean inherits the worst component's false-alarm
                # behaviour; max amplifies it).
                _run("fuse_min(" + "+".join(combo) + ")", np.nanmin(stack, axis=1))
                if r == len(components):
                    _run("fuse_max(" + "+".join(combo) + ")", np.nanmax(stack, axis=1))

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.info("Saved %s", args.output_csv)

    # --- unbiased leave-one-generator-out selection (see --logo-selection) ---
    if args.logo_selection:
        ev = keep
        real_ev = is_real[ev]
        algos_ev = algos[ev]
        gens = sorted(set(algos_ev[~real_ev]) - {""})

        def _score_on(name: str, gen_list: list[str]) -> float | None:
            s = scores_for_delong[name][ev]
            real_s = s[real_ev]
            real_s = real_s[np.isfinite(real_s)]
            vals = []
            for g in gen_list:
                fake_s = s[(~real_ev) & (algos_ev == g)]
                fake_s = fake_s[np.isfinite(fake_s)]
                if len(fake_s) < args.min_per_group or len(real_s) < args.min_per_group:
                    continue
                y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
                auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
                vals.append(eer if args.selection_metric == "eer" else -auc)
            return float(np.mean(vals)) if vals else None

        logo_rows = []
        for held in gens:
            sel = [g for g in gens if g != held]
            best_name, best_val = None, None
            for name in scores_for_delong:
                v = _score_on(name, sel)
                if v is None:
                    continue
                if best_val is None or v < best_val:
                    best_val, best_name = v, name
            if best_name is None:
                continue
            s = scores_for_delong[best_name][ev]
            real_s = s[real_ev]
            real_s = real_s[np.isfinite(real_s)]
            fake_s = s[(~real_ev) & (algos_ev == held)]
            fake_s = fake_s[np.isfinite(fake_s)]
            y = np.r_[np.zeros(len(real_s)), np.ones(len(fake_s))]
            auc, eer = auc_and_eer(y, np.r_[real_s, fake_s])
            logo_rows.append(
                {
                    "held_out_generator": held,
                    "selected_fusion": best_name,
                    "heldout_auc": round(auc, 4),
                    "heldout_eer_pct": round(eer * 100, 2),
                    "n_real": len(real_s),
                    "n_fake": len(fake_s),
                }
            )
        if logo_rows:
            logo = pd.DataFrame(logo_rows)
            logo.to_csv(Path(args.output_csv).with_name(Path(args.output_csv).stem + "_logo.csv"), index=False)
            logger.info(
                "\nUNBIASED leave-one-generator-out selection (fusion chosen WITHOUT the held-out "
                "generator; selection metric = %s):\n%s",
                args.selection_metric,
                logo.to_string(index=False),
            )
            logger.info(
                "LOGO macro AUC = %.4f | macro EER = %.2f%% — THESE are the citable numbers "
                "(the sorted table above is an ablation whose top row is test-set-selected).",
                logo["heldout_auc"].mean(),
                logo["heldout_eer_pct"].mean(),
            )
            consensus = logo["selected_fusion"].value_counts()
            logger.info("Fusion selected per fold:\n%s", consensus.to_string())
            top = consensus.index[0]
            pooled_row = out[(out["scorer"] == top) & (out["algorithm"] == "POOLED")]
            if len(pooled_row):
                logger.info(
                    "Consensus fusion %r (chosen in %d/%d folds) has POOLED AUC %.4f / EER %.2f%% "
                    "— compare against MusicDET's 9.72%% pooled EER on this corpus.",
                    top,
                    int(consensus.iloc[0]),
                    len(logo),
                    float(pooled_row["auc"].iloc[0]),
                    float(pooled_row["eer_pct"].iloc[0]),
                )

    pivot = out[~out["algorithm"].isin(["MACRO_AVG", "POOLED"])].pivot_table(
        index="scorer", columns="algorithm", values="auc"
    )
    pivot["MACRO"] = out[out["algorithm"] == "MACRO_AVG"].set_index("scorer")["auc"]
    pivot["POOLED_AUC"] = out[out["algorithm"] == "POOLED"].set_index("scorer")["auc"]
    pivot["POOLED_EER%"] = out[out["algorithm"] == "POOLED"].set_index("scorer")["eer_pct"]
    pivot = pivot.sort_values("MACRO", ascending=False)
    logger.info("\nAUC by scorer (rows) x generator (cols), sorted by MACRO:\n%s", pivot.to_string())

    # --- DeLong: each fusion vs the flow-only baseline, pooled ---
    base = scores_for_delong.get("flow_nll")
    if base is not None:
        ev = keep & ~np.isnan(base)
        y = (~is_real[ev]).astype(int)
        logger.info("\nDeLong vs flow_nll (pooled real-vs-all-fake, evaluated reals only):")
        sig_rows = []
        for name, s in scores_for_delong.items():
            if name == "flow_nll":
                continue
            res = delong_test(y, s[ev], base[ev])
            sig_rows.append(
                {
                    "scorer": name,
                    "auc": round(res["auc_a"], 4),
                    "auc_flow_only": round(res["auc_b"], 4),
                    "delta_auc": round(res["delta_auc"], 4),
                    "z": round(res["z"], 2) if np.isfinite(res["z"]) else None,
                    "p_value": res["p_value"],
                }
            )
        sig = pd.DataFrame(sig_rows).sort_values("delta_auc", ascending=False)
        logger.info("\n%s", sig.to_string(index=False))
        sig.to_csv(Path(args.output_csv).with_name(Path(args.output_csv).stem + "_delong.csv"), index=False)

    # --- fused cross-corpus false-positive rate -----------------------------
    if args.fpr_scores:
        ext: dict[str, np.ndarray] = {}
        for spec_arg in args.fpr_scores:
            if "=" not in spec_arg:
                raise SystemExit(f"--fpr-scores expects COMPONENT=CSV, got {spec_arg!r}")
            cname, cpath = spec_arg.split("=", 1)
            if cname not in calibrated:
                raise SystemExit(
                    f"--fpr-scores component {cname!r} is not one of the fused components " f"{list(calibrated)}"
                )
            ecsv = pd.read_csv(cpath, low_memory=False)
            col = next((c for c in ("wf_anomaly_score", "anomaly") if c in ecsv.columns), None)
            if col is None:
                wfm = [c for c in ecsv.columns if "wf_mean" in c]
                if not wfm:
                    raise SystemExit(f"no anomaly column found in {cpath}")
                ecsv[col:="_anom"] = -ecsv[wfm[0]]
            ext[cname] = ecsv[col].dropna().to_numpy(float)

        n_ext = {k: len(v) for k, v in ext.items()}
        if len(set(n_ext.values())) != 1:
            logger.warning(
                "external corpora have differing row counts %s — truncating to the shortest. "
                "Rows are assumed to be in the SAME track order across components; verify this.",
                n_ext,
            )
        n_keep = min(n_ext.values())

        # Z-score, NOT quantile, for the FPR analysis. The quantile transform is
        # bounded at 1.0, so once >=1% of in-domain reals exceed the reference
        # maximum the 99th-percentile threshold saturates at 1.0 and the FPR
        # collapses to 0 by construction. A z-score fit on the same reference
        # reals is monotone, unbounded, and comparable across components.
        z_in, z_ext = {}, {}
        for cname in ext:
            raw_in = raw_component[cname]
            ref_vals = raw_in[ref_rows]
            ref_vals = ref_vals[np.isfinite(ref_vals)]
            mu, sd = float(np.mean(ref_vals)), float(np.std(ref_vals) + 1e-12)
            z_in[cname] = (raw_in - mu) / sd
            z_ext[cname] = (ext[cname][:n_keep] - mu) / sd

        logger.info("\nFUSED CROSS-CORPUS FPR on %d external real tracks:", n_keep)
        fpr_rows = []
        tag = "+".join(sorted(z_in))
        scorers = list(z_in) + [f"FUSED_mean({tag})", f"FUSED_min({tag})"]
        for name in scorers:
            if name.startswith("FUSED_"):
                op = np.nanmin if name.startswith("FUSED_min") else np.nanmean
                in_dom = op(np.stack([z_in[k] for k in z_in], axis=1), axis=1)
                out_dom = op(np.stack([z_ext[k] for k in z_in], axis=1), axis=1)
            else:
                in_dom, out_dom = z_in[name], z_ext[name]
            ref = in_dom[keep & is_real]
            ref = ref[np.isfinite(ref)]
            row = {"scorer": name, "n_external": int(np.isfinite(out_dom).sum())}
            for pct in (99, 95):
                t = float(np.percentile(ref, pct))
                row[f"fpr_at_{100 - pct}pct_operating_point"] = round(100 * float(np.nanmean(out_dom > t)), 2)
            fpr_rows.append(row)
        fpr_df = pd.DataFrame(fpr_rows)
        fpr_df.to_csv(Path(args.output_csv).with_name(Path(args.output_csv).stem + "_fpr.csv"), index=False)
        logger.info("\n%s", fpr_df.to_string(index=False))
        logger.info(
            "READ WITH THE AUC TABLE: an arm can win in-domain AUC and still be the worse "
            "detector if its cross-corpus FPR is far higher — genuine music from an unseen "
            "catalogue being flagged as AI is the failure that matters in deployment."
        )

    logger.info(
        "\nHONESTY NOTE: 'complexity' is a parameter-free FLAC-bitrate statistic with NO learned "
        "component. If it is competitive on its own, report it as a baseline AND a confound "
        "(is the corpus separable by production quality alone?), not only as a fusion component. "
        "The reconstruction-control complexity check (build_reconstruction_control.py "
        "--with-complexity) decides whether it reflects codec artifacts or corpus mastering."
    )


if __name__ == "__main__":
    main()
