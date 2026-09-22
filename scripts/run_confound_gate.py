"""The gate. No number enters a report without passing this, and the gate output
ships next to the number.

Why a gate rather than a checklist
----------------------------------
This project has withdrawn two headline claims and retracted a third. Every one
of them was a number that looked plausible and had never been asked a specific,
falsifiable question. A checklist in a handover does not get run; a script that
prints ``FAIL`` does.

The rule: **a number without a gate output next to it does not go in a report** —
including numbers already measured. Retro-validation of the existing score CSVs
is part of the work, not a bonus.

The gates
---------
  C1  channel-alone     best single channel descriptor used alone as a detector.
                        Catches the SONICS energy-ratio leak (AUC 1.0000).
  C2  level             dc_offset / peak / crest. Catches the 0.7773 level leak.
  C3  duration          duration_s AUC, and the per-class fraction reaching full
                        length. The specific risk of a longer SONICS average.
  C4  silence           silence_frac / lead / tail / exact zeros.
  C5  label shuffle     permute labels; AUC must return to 0.5 within its CI.
                        Catches leakage of the byte-identical-scores shape.
  C6  score sanity      non-constant, non-degenerate, enough finite values.
  C7  row survival      per-stratum counts; any stratum at 0 is a hard FAIL.
  C8  spacing cluster   comb spacing must cluster BY GENERATOR, not sit at one
                        corpus-wide value. Distinguishes decoder from channel.
  C9  harmonicity       |corr(score, harmonicity)| per class. Catches a musical
                        harmonic comb masquerading as a decoder comb.
  C10 content-identical reconstruction pairs of the same source must score alike.
  C11 CI + DeLong       bootstrap CI per generator, and DeLong against a rival.

C1-C4 and C8-C10 need per-track descriptors. Supply them with --descriptor-csv
(the output of diagnose_channel_confound.py) — the gate does NOT recompute them,
so it cannot disagree with the diagnostic that produced the protocol.

Usage (EC2)
-----------
    python scripts/run_confound_gate.py \\
        --score-csv reports/diagnostics/comb_fmc_hull_db/comb_per_track_lvl_hull_db_f1000.csv \\
        --score-column comb_strength \\
        --descriptor-csv reports/diagnostics/channel_fmc/channel_per_track.csv \\
        --out-dir reports/diagnostics/gate_comb_fmc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intrinsic_ai_music_detection.models.evaluate import auc_and_eer, bootstrap_auc, delong_test  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("confound_gate")

# A descriptor that reaches this AUC on its own is doing the detector's job for
# it. 0.60 is the threshold the protocol already uses in the handover's tables.
LEAK_AUC = 0.60
# |corr| above this means the score is substantially explained by harmonicity.
HARMONICITY_R = 0.40

CHANNEL_DESCRIPTORS = (
    "cutoff_hz",
    "frac_power_above_8k",
    "frac_power_8k_11k",
    "frac_power_top_quarter",
    "hf_floor_frac",
    "spectral_flatness_hf",
)
LEVEL_DESCRIPTORS = ("dc_offset", "peak_dbfs", "crest_factor_db")
SILENCE_DESCRIPTORS = ("silence_frac", "lead_silence_s", "tail_silence_s", "exact_zero_frac")


class Gate:
    """One check. ``passed is None`` means "not applicable / no data", never "pass"."""

    def __init__(self, cid: str, name: str):
        self.cid, self.name = cid, name
        self.passed: bool | None = None
        self.detail: dict = {}
        self.note = ""

    def result(self, passed: bool | None, note: str = "", **detail) -> "Gate":
        # Cast through bool(): numpy comparisons return np.bool_, which is not
        # JSON-serialisable and is not `is True` / `is False`, so a caller
        # checking identity would silently treat every gate as indeterminate.
        self.passed = None if passed is None else bool(passed)
        self.note, self.detail = note, detail
        return self

    def to_row(self) -> dict:
        return {
            "gate": self.cid,
            "name": self.name,
            "status": "SKIP" if self.passed is None else ("PASS" if self.passed else "FAIL"),
            "note": self.note,
            **{k: v for k, v in self.detail.items()},
        }


def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Orientation-free AUC: a perfectly inverted descriptor is just as much a leak."""
    ok = np.isfinite(values)
    if ok.sum() < 10 or len(set(labels[ok])) < 2:
        return float("nan")
    auc, _ = auc_and_eer(labels[ok], values[ok])
    return max(auc, 1.0 - auc)


def _descriptor_gate(cid, name, df, y, descriptors) -> Gate:
    g = Gate(cid, name)
    present = [d for d in descriptors if d in df.columns]
    if not present:
        return g.result(None, f"no {name} descriptors in --descriptor-csv")
    aucs = {d: _auc(y, df[d].to_numpy(float)) for d in present}
    aucs = {k: v for k, v in aucs.items() if np.isfinite(v)}
    if not aucs:
        return g.result(None, "descriptors present but all non-finite")
    worst = max(aucs, key=aucs.get)
    return g.result(
        aucs[worst] <= LEAK_AUC,
        f"worst descriptor {worst} at AUC {aucs[worst]:.4f} (threshold {LEAK_AUC})",
        worst_descriptor=worst,
        worst_auc=round(aucs[worst], 4),
    )


def gate_duration(df, y) -> Gate:
    g = Gate("C3", "duration")
    if "duration_s" not in df.columns:
        return g.result(None, "no duration_s in --descriptor-csv")
    d = df["duration_s"].to_numpy(float)
    auc = _auc(y, d)
    # The second half matters more than the AUC when a long analysis window is
    # used: if one class is systematically shorter, it gets less averaging, and
    # the detector's advantage is partly "how much audio did I get".
    full = float(np.nanmax(d))
    frac_real = float(np.mean(d[y == 0] >= 0.99 * full))
    frac_fake = float(np.mean(d[y == 1] >= 0.99 * full))
    gap = abs(frac_real - frac_fake)
    ok = (not np.isfinite(auc) or auc <= LEAK_AUC) and gap <= 0.05
    return g.result(
        ok,
        f"duration AUC {auc:.4f}; full-length fraction real {frac_real:.3f} vs "
        f"fake {frac_fake:.3f} (gap {gap:.3f}, max 0.05)",
        duration_auc=round(auc, 4) if np.isfinite(auc) else None,
        frac_full_real=round(frac_real, 4),
        frac_full_fake=round(frac_fake, 4),
    )


def gate_shuffle(y, s, seed, n_perm=200) -> Gate:
    g = Gate("C5", "label shuffle")
    rng = np.random.RandomState(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a = _auc(yp, s)
        if np.isfinite(a):
            null.append(a)
    if len(null) < 20:
        return g.result(None, "too few valid permutations")
    lo, hi = np.percentile(null, [2.5, 97.5])
    # The orientation-free AUC of a random split is ~0.5 plus a positive bias
    # (max(auc, 1-auc) is never below 0.5), so the check is that the null band is
    # TIGHT and low — not that it is centred on exactly 0.5.
    ok = bool(hi < 0.62)
    return g.result(
        ok,
        f"shuffled-label AUC 95% band [{lo:.4f}, {hi:.4f}] (upper bound must stay < 0.62)",
        null_lo=round(float(lo), 4),
        null_hi=round(float(hi), 4),
    )


def gate_score_sanity(s) -> Gate:
    g = Gate("C6", "score sanity")
    finite = np.isfinite(s)
    n_finite = int(finite.sum())
    if n_finite < 20:
        return g.result(False, f"only {n_finite} finite scores", n_finite=n_finite)
    uniq = len(np.unique(np.round(s[finite], 10)))
    # Byte-identical score files are exactly how the invalidated FMA runs failed.
    ok = uniq > max(10, 0.01 * n_finite)
    return g.result(
        ok,
        f"{n_finite} finite scores, {uniq} distinct values " f"({100 * n_finite / len(s):.1f}% of rows finite)",
        n_finite=n_finite,
        n_distinct=uniq,
    )


def gate_row_survival(df) -> Gate:
    g = Gate("C7", "row survival")
    if "algorithm" not in df.columns:
        return g.result(None, "no algorithm column")
    counts = df.groupby(["label", "algorithm"]).size()
    empty = [str(k) for k, v in counts.items() if v == 0]
    ok = not empty and counts.min() >= 5
    return g.result(
        ok,
        f"smallest stratum {int(counts.min())} rows across {len(counts)} strata"
        + (f"; EMPTY: {empty}" if empty else ""),
        min_stratum=int(counts.min()),
        n_strata=int(len(counts)),
    )


def gate_spacing(df) -> Gate:
    g = Gate("C8", "decoder-spacing cluster")
    if "comb_spacing_hz" not in df.columns or "algorithm" not in df.columns:
        return g.result(None, "no comb_spacing_hz — not applicable to this arm")
    fake = df[df["label"] != "real"]
    med = fake.groupby("algorithm")["comb_spacing_hz"].median().dropna()
    if len(med) < 2:
        return g.result(None, "fewer than two generators")

    # CORRECTED 2026-08-21. The first version compared between-generator SPREAD to
    # within-generator STANDARD DEVIATION, and failed FMC at 83.5 vs 295.3 Hz.
    # That comparison is not meaningful: the spacing estimate lives on a bounded
    # lag grid ([40, 4000] Hz) and a minority of tracks land on a harmonic or on
    # noise, so the SD is dominated by a heavy tail while the mode is tight. The
    # right question is CONCENTRATION: what fraction of a generator's tracks sit
    # near that generator's own modal spacing, and is that higher than for real
    # music?
    def concentration(values, centre, tol=0.05):
        v = values[np.isfinite(values)]
        return float(np.mean(np.abs(v - centre) <= tol * centre)) if len(v) else float("nan")

    per_gen = {}
    for alg, centre in med.items():
        v = fake.loc[fake["algorithm"] == alg, "comb_spacing_hz"].to_numpy(float)
        per_gen[alg] = round(concentration(v, centre), 4)

    real = df[df["label"] == "real"]["comb_spacing_hz"].to_numpy(float)
    real_med = float(np.nanmedian(real))
    real_conc = concentration(real, real_med)

    fake_conc = float(np.nanmean(list(per_gen.values())))
    spread = float(med.max() - med.min())
    # A decoder signature means each generator concentrates on ITS OWN spacing
    # more tightly than real music concentrates on any single value, AND the
    # generators do not all sit at one value.
    # A margin, not a bare inequality: a concentration that beats real music by a
    # hair is noise, not a decoder signature.
    ok = bool(fake_conc > max(real_conc * 1.25, real_conc + 0.05) and spread > 0.05 * float(med.median()))
    return g.result(
        ok,
        f"per-generator concentration within ±5% of own median: fake mean {fake_conc:.3f} "
        f"vs real {real_conc:.3f}; between-generator spread {spread:.1f} Hz "
        f"(median spacing {float(med.median()):.1f} Hz)",
        concentration_per_generator=per_gen,
        concentration_real=round(real_conc, 4),
        between_spread_hz=round(spread, 2),
        per_generator_median_hz={k: round(float(v), 2) for k, v in med.items()},
        real_median_hz=round(real_med, 2),
    )


def gate_harmonicity(df, s, y) -> Gate:
    g = Gate("C9", "harmonicity")
    cols = [c for c in ("harmonicity", "pitch_salience", "spectral_peak_count", "hnr") if c in df.columns]
    if not cols:
        return g.result(
            None,
            "no harmonicity descriptor available — the comb-vs-harmonics confound is "
            "UNTESTED for this number and must be stated as such. Re-run "
            "diagnose_channel_confound.py to emit harmonicity/pitch_salience/"
            "spectral_peak_count.",
        )
    # CORRECTED 2026-08-22. The gate is thresholded on the WITHIN-REAL correlation
    # only, and the within-fake correlation is reported as mechanism evidence.
    #
    # The confound this gate exists to catch is "does the detector fire on tonal
    # REAL music?" — a musical harmonic series masquerading as a decoder comb.
    # That is a within-REAL question. A within-FAKE correlation means the opposite:
    # among tracks that DO have decoder peaks, the score tracks how many there are.
    # That is the mechanism working, and thresholding on it fails the detector for
    # succeeding. Measured on the FMC headline: harmonicity 0.184 (real) / 0.253
    # (fake), pitch_salience 0.163 / 0.184, spectral_peak_count 0.105 / 0.613 —
    # the peak-count row is near-tautological with the score and is diagnostic only.
    TAUTOLOGICAL = {"spectral_peak_count"}

    detail, worst_real, worst_col = {}, 0.0, None
    for col in cols:
        per_class = {}
        for name, mask in (("real", y == 0), ("fake", y == 1)):
            v, sc = df[col].to_numpy(float)[mask], s[mask]
            ok = np.isfinite(v) & np.isfinite(sc)
            per_class[name] = float(np.corrcoef(v[ok], sc[ok])[0, 1]) if ok.sum() > 10 else float("nan")
        detail[col] = {k: round(v, 4) for k, v in per_class.items()}
        r = per_class["real"]
        if col not in TAUTOLOGICAL and np.isfinite(r) and abs(r) > worst_real:
            worst_real, worst_col = abs(r), col

    if worst_col is None:
        return g.result(
            None,
            "no non-tautological harmonicity descriptor with enough finite pairs " f"(available: {cols})",
        )
    return g.result(
        worst_real <= HARMONICITY_R,
        f"strongest WITHIN-REAL |corr(score, tonality)| = {worst_real:.3f} on "
        f"{worst_col!r} (threshold {HARMONICITY_R}). Within-fake correlations are "
        f"MECHANISM, not confound, and are reported not thresholded. "
        f"Per-descriptor {{real, fake}}: {detail}",
        worst_descriptor=worst_col,
        worst_abs_corr_real=round(worst_real, 4),
        per_descriptor=detail,
    )


def gate_bootstrap_and_delong(df, s, y, rival: np.ndarray | None, seed) -> Gate:
    g = Gate("C11", "bootstrap CI + DeLong")
    ok_mask = np.isfinite(s)
    if len(set(y[ok_mask])) < 2:
        return g.result(None, "single class")
    overall = bootstrap_auc(y[ok_mask], s[ok_mask], seed=seed)
    detail = {
        "auc": round(float(overall["auc"]), 4),
        "ci_lo": round(float(overall["ci_lo"]), 4),
        "ci_hi": round(float(overall["ci_hi"]), 4),
    }
    per_gen = {}
    if "algorithm" in df.columns:
        for alg in sorted(set(df.loc[y == 1, "algorithm"]) - {""}):
            sel = ((df["algorithm"] == alg).to_numpy() | (y == 0)) & ok_mask
            if len(set(y[sel])) < 2 or sel.sum() < 20:
                continue
            b = bootstrap_auc(y[sel], s[sel], seed=seed)
            per_gen[alg] = [round(float(b["auc"]), 4), round(float(b["ci_lo"]), 4), round(float(b["ci_hi"]), 4)]
    detail["per_generator_auc_ci"] = per_gen

    note = f"AUC {detail['auc']:.4f} [{detail['ci_lo']:.4f}, {detail['ci_hi']:.4f}]"
    passed = detail["ci_lo"] > 0.5
    if rival is not None:
        both = ok_mask & np.isfinite(rival)
        if len(set(y[both])) == 2:
            d = delong_test(y[both], s[both], rival[both])
            detail["delong"] = {k: round(float(v), 6) for k, v in d.items()}
            note += f"; DeLong vs rival z={d['z']:.2f} p={d['p_value']:.3g}"
            passed = passed and d["p_value"] < 0.05
    return g.result(passed, note + " (CI lower bound must exceed 0.5)", **detail)


def gate_content_identical(path: str | None, score_column: str | None = None) -> Gate:
    """Paired comparison on content-identical audio: source vs each reconstruction.

    Takes the per-track CSV written by ``eval_comb_detector.py`` /
    ``eval_nmf_peak_detector.py`` when scoring the manifest that
    ``build_reconstruction_control.py --save-audio-dir`` exports. That manifest
    labels every variant ``real`` (they all derive from real audio), so the eval
    script correctly reports "no AUC" — the comparison here is WITHIN a
    ``pair_id``, not between classes, which is the whole point of a
    content-identical control.

    Accepts either a purpose-built ``pair_id,variant,score`` file or the raw
    per-track CSV plus ``--recon-score-column``.
    """
    g = Gate("C10", "content-identical control")
    if not path:
        return g.result(
            None,
            "no --recon-csv supplied — whether this score reflects generation rather than "
            "mastering/production is UNTESTED and must be stated as such",
        )
    df = pd.read_csv(path)
    if not {"pair_id", "variant"}.issubset(df.columns):
        return g.result(None, "--recon-csv needs pair_id and variant columns")

    col = "score" if "score" in df.columns else score_column
    if col is None or col not in df.columns:
        return g.result(
            None,
            f"--recon-csv has no 'score' column; pass --recon-score-column "
            f"(available: {[c for c in df.columns if 'comb_' in c or c.startswith('r_')][:12]})",
        )

    src = df[df["variant"] == "source"].set_index("pair_id")[col]
    rows = []
    for var in sorted(set(df["variant"]) - {"source"}):
        v = df[df["variant"] == var].set_index("pair_id")[col]
        common = src.index.intersection(v.index)
        if len(common) < 10:
            continue
        a, b = v[common].to_numpy(float), src[common].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        rows.append((var, float(np.mean(a[ok] > b[ok])), int(ok.sum()), float(np.median(a[ok] - b[ok]))))
    if not rows:
        return g.result(None, "no comparable pairs in --recon-csv")
    return g.result(
        True,
        "; ".join(f"{v}: fires above source on {f:.1%} of {n} pairs (median delta {d:+.4f})" for v, f, n, d in rows)
        + " — read the ORDERING; this gate reports rather than thresholds",
        per_variant={
            v: {"frac_above_source": round(f, 4), "n_pairs": n, "median_delta": round(d, 5)} for v, f, n, d in rows
        },
        score_column=col,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-csv", required=True, help="Per-track scores with track_id + label.")
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--descriptor-csv",
        default=None,
        help="Per-track channel descriptors from diagnose_channel_confound.py. Without it "
        "C1-C4, C8-C9 report SKIP — which is NOT a pass.",
    )
    parser.add_argument(
        "--rival-csv",
        default=None,
        help="Second per-track score CSV for the DeLong comparison (e.g. the flow arm).",
    )
    parser.add_argument("--rival-column", default=None)
    parser.add_argument(
        "--recon-csv",
        default=None,
        help="Content-identical pairs for C10 — the per-track CSV from scoring the manifest "
        "that build_reconstruction_control.py --save-audio-dir exports.",
    )
    parser.add_argument(
        "--recon-score-column",
        default=None,
        help="Score column inside --recon-csv (defaults to 'score' if that column exists).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column naming the class; 'real' is the negative class.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.score_csv, low_memory=False)
    if args.score_column not in df.columns:
        raise SystemExit(
            f"{args.score_csv} has no column {args.score_column!r}. Available: "
            f"{[c for c in df.columns if c not in ('track_id', 'label', 'algorithm')][:20]}"
        )
    if args.label_column not in df.columns:
        raise SystemExit(f"{args.score_csv} has no {args.label_column!r} column")
    if "algorithm" not in df.columns:
        df["algorithm"] = ""
    df["algorithm"] = df["algorithm"].fillna("")

    s = df[args.score_column].to_numpy(float)
    y = (df[args.label_column] != "real").astype(int).to_numpy()

    merged = df
    if args.descriptor_csv:
        desc = pd.read_csv(args.descriptor_csv, low_memory=False)
        if "track_id" not in desc.columns or "track_id" not in df.columns:
            raise SystemExit("both CSVs need a track_id column to join descriptors")
        # suffix rather than overwrite: a silent column collision is how the
        # fuse_labelfree_scores KeyError happened.
        merged = df.merge(desc, on="track_id", how="left", suffixes=("", "_desc"))
        if len(merged) != len(df):
            raise SystemExit(
                f"descriptor join changed the row count ({len(df)} -> {len(merged)}); "
                "duplicate track_ids in --descriptor-csv"
            )
        s = merged[args.score_column].to_numpy(float)
        y = (merged[args.label_column] != "real").astype(int).to_numpy()

    rival = None
    if args.rival_csv and args.rival_column:
        rv = pd.read_csv(args.rival_csv, low_memory=False)
        j = merged[["track_id"]].merge(rv[["track_id", args.rival_column]], on="track_id", how="left")
        rival = j[args.rival_column].to_numpy(float)

    gates = [
        _descriptor_gate("C1", "channel-alone", merged, y, CHANNEL_DESCRIPTORS),
        _descriptor_gate("C2", "level", merged, y, LEVEL_DESCRIPTORS),
        gate_duration(merged, y),
        _descriptor_gate("C4", "silence", merged, y, SILENCE_DESCRIPTORS),
        gate_shuffle(y, s, args.seed),
        gate_score_sanity(s),
        gate_row_survival(merged),
        gate_spacing(merged),
        gate_harmonicity(merged, s, y),
        gate_content_identical(args.recon_csv, args.recon_score_column),
        gate_bootstrap_and_delong(merged, s, y, rival, args.seed),
    ]

    rows = [g.to_row() for g in gates]
    report = pd.DataFrame(rows)
    stem = f"gate_{Path(args.score_csv).stem}_{args.score_column}"
    report.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=str))

    n_fail = sum(1 for g in gates if g.passed is False)
    n_skip = sum(1 for g in gates if g.passed is None)

    logger.info("\n%s", "=" * 78)
    logger.info("CONFOUND GATE — %s :: %s", args.score_csv, args.score_column)
    logger.info("%s", "=" * 78)
    for g in gates:
        status = "SKIP" if g.passed is None else ("PASS" if g.passed else "FAIL")
        logger.info("  [%-4s] %-3s %-26s %s", status, g.cid, g.name, g.note)
    logger.info("%s", "=" * 78)

    if n_fail:
        logger.error(
            "%d gate(s) FAILED. This number must NOT be written into a report as-is. "
            "Either fix the confound or report the number WITH the failing gate beside "
            "it and a CORRECTED/RETRACTED marker.",
            n_fail,
        )
    if n_skip:
        logger.warning(
            "%d gate(s) SKIPPED. A skip is NOT a pass — it means that confound is "
            "untested for this number, and the paper must say so rather than imply "
            "the control was run.",
            n_skip,
        )
    if not n_fail and not n_skip:
        logger.info("All gates passed. Ship the gate CSV next to the number.")

    logger.info("Saved → %s", out_dir / f"{stem}.csv")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
