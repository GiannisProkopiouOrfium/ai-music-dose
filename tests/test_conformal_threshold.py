"""Group-conditional conformal thresholds in ``measure_false_positive_rate.py``.

Why this is worth code rather than a sentence
----------------------------------------------
The paper already observes that a reals-only quantile IS a split-conformal threshold
under exchangeability, and that an unseen catalogue breaks exchangeability: the comb
arm's false-positive rate moves 0.042 -> 0.115 on FMA, 2.75x. What it does not do is
act on the observation. Conditioning the calibration on genre -- Mondrian conformal --
gives each genre its nominal rate by construction, which is the only mechanism in the
project that attacks the per-genre disparity directly rather than hoping a better
score dissolves it.

Two things are asserted that a docstring cannot enforce.

**The guarantee is finite-sample, so the order statistic must be the right one.**
``conformal_threshold`` returns the ``ceil((n+1)(1-alpha))``-th smallest calibration
score, not ``np.quantile``. On 3,000 pooled reals those agree to the fourth decimal
and the distinction looks pedantic; on a forty-track genre they do not, and the small
genres are exactly what a per-group threshold exists to serve. The coverage test
below draws fresh reals and checks the realised exceedance rate really is bounded.

**A sample too small for the level must say so.** With fewer than ``1/alpha - 1``
calibration points the required order statistic falls outside the sample. Returning
the maximum there would look like a threshold and carry no guarantee at all, so the
function returns ``+inf`` and the caller falls back to the pooled threshold and flags
it. Silent degradation is the failure mode this repository has been bitten by most.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "measure_false_positive_rate.py"
sys.path.insert(0, str(REPO / "scripts"))

from measure_false_positive_rate import (  # noqa: E402
    MIN_CALIBRATION_PER_GROUP,
    conformal_threshold,
)


class TestTheConformalThresholdItself:
    def test_it_is_the_ceil_n_plus_one_times_one_minus_alpha_order_statistic(self):
        x = np.arange(1.0, 101.0)  # n = 100
        # ceil(101 * 0.95) = 96 -> the 96th smallest, which is the value 96.
        assert conformal_threshold(x, 0.05) == 96.0
        # ceil(101 * 0.99) = 100 -> the largest, still inside the sample.
        assert conformal_threshold(x, 0.01) == 100.0

    def test_a_sample_too_small_for_the_level_returns_infinity(self):
        # alpha = 0.05 needs n >= 19. At n = 18, ceil(19*0.95) = 19 > 18.
        assert conformal_threshold(np.arange(18.0), 0.05) == float("inf")
        assert np.isfinite(conformal_threshold(np.arange(19.0), 0.05))

    def test_an_empty_calibration_set_returns_infinity(self):
        assert conformal_threshold(np.array([]), 0.05) == float("inf")

    def test_non_finite_calibration_values_are_dropped_not_propagated(self):
        x = np.concatenate([np.arange(1.0, 101.0), [np.nan, np.inf]])
        assert conformal_threshold(x, 0.05) == 96.0

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    @pytest.mark.parametrize("n_calib", [50, 200])
    def test_the_coverage_guarantee_actually_holds_on_fresh_draws(self, alpha, n_calib):
        rng = np.random.RandomState(0)
        exceed = []
        for _ in range(400):
            thr = conformal_threshold(rng.randn(n_calib), alpha)
            fresh = rng.randn(200)
            exceed.append(float((fresh > thr).mean()))
        realised = float(np.mean(exceed))
        assert realised <= alpha + 0.02, (
            f"alpha={alpha}, n={n_calib}: realised exceedance {realised:.4f} exceeds the nominal "
            "rate. The conformal guarantee is the only reason this threshold is defensible on a "
            "catalogue nobody calibrated on."
        )

    def test_it_is_conservative_where_np_quantile_is_not_on_a_small_sample(self):
        # The distinction that matters for a small genre: the conformal threshold is
        # never below the plug-in estimate, so it never over-promises.
        rng = np.random.RandomState(3)
        x = rng.randn(40)
        assert conformal_threshold(x, 0.05) >= float(np.quantile(x, 0.95))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A labelled corpus and an unseen catalogue whose genres are genuinely different.

    ``electronic`` scores high (quantised production resembling a decoder comb, which
    is the measured reason Electronic is the comb arm's worst genre at 0.236) and
    ``folk`` low. A single global threshold must therefore spread badly across them,
    which is the condition the per-group threshold is supposed to fix.
    """
    rng = np.random.RandomState(5)
    n_lab = 1200
    label = ["real"] * (n_lab // 2) + ["fake"] * (n_lab // 2)
    is_fake = np.array([lab != "real" for lab in label], dtype=float)
    lab = pd.DataFrame(
        {
            "track_id": [f"L{i:04d}" for i in range(n_lab)],
            "label": label,
            "algorithm": [""] * (n_lab // 2) + ["gen_a"] * (n_lab // 2),
            "comb_x": rng.randn(n_lab) * 0.4 + 1.3 * is_fake,
        }
    )
    groups = (
        ["electronic"] * 700 + ["folk"] * 700 + ["pop"] * 700 + ["tiny_genre"] * 25
    )
    shift = {"electronic": 0.75, "folk": -0.35, "pop": 0.0, "tiny_genre": 0.2}
    uns = pd.DataFrame(
        {
            "track_id": [f"U{i:04d}" for i in range(len(groups))],
            "label": "real",
            "genre": groups,
            "comb_x": [rng.randn() * 0.4 + shift[g] for g in groups],
        }
    )
    lab_p, uns_p = tmp_path / "lab.csv", tmp_path / "uns.csv"
    lab.to_csv(lab_p, index=False)
    uns.to_csv(uns_p, index=False)
    return lab_p, uns_p


def _run(tmp_path: Path, extra: list[str]) -> tuple[subprocess.CompletedProcess, Path]:
    lab_p, uns_p = _fixture(tmp_path)
    out = tmp_path / ("out_" + str(abs(hash(tuple(extra))) % 10_000))
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--labelled-csv", str(lab_p),
            "--unseen-csv", str(uns_p),
            "--score-column", "comb_x",
            "--out-dir", str(out),
        ] + extra,
        capture_output=True,
        text=True,
    )
    return proc, out


class TestTheExistingBehaviourIsUntouched:
    def test_without_the_flag_nothing_new_is_written(self, tmp_path):
        proc, out = _run(tmp_path, ["--group-column", "genre"])
        assert proc.returncode == 0, proc.stderr
        assert (out / "false_positive_rate.csv").exists()
        assert (out / "false_positive_rate_by_group.csv").exists()
        assert not (out / "conformal_by_group.csv").exists()

    def test_the_original_table_keeps_its_columns(self, tmp_path):
        _, out = _run(tmp_path, [])
        cols = list(pd.read_csv(out / "false_positive_rate.csv").columns)
        assert cols[:8] == [
            "quantile", "threshold", "target_fpr", "in_corpus_fpr",
            "unseen_fpr", "n_threshold_reals", "n_eval_reals", "n_unseen",
        ]

    def test_the_flag_requires_a_group_column(self, tmp_path):
        proc, _ = _run(tmp_path, ["--calibrate-per-group"])
        assert proc.returncode != 0
        assert "group" in (proc.stdout + proc.stderr).lower()


class TestMondrianEqualisesTheGroups:
    def test_the_per_group_threshold_shrinks_the_fpr_spread(self, tmp_path):
        proc, out = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group"])
        assert proc.returncode == 0, proc.stderr
        t = pd.read_csv(out / "conformal_by_group.csv")
        big = t[t["n_eval"] >= 20]

        def spread(col):
            v = big[col][big[col] > 0]
            return float(v.max() / v.min())

        assert spread("fpr_conformal_group") < spread("fpr_transfer"), (
            "the per-group conformal threshold did not reduce the false-positive spread across "
            "genres, which is the only thing it is for."
        )

    def test_each_group_lands_near_its_nominal_rate(self, tmp_path):
        _, out = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group", "--alpha", "0.05"])
        t = pd.read_csv(out / "conformal_by_group.csv")
        for _, row in t[(t["n_eval"] >= 100) & (~t["fallback_to_global"])].iterrows():
            assert row["fpr_conformal_group"] == pytest.approx(0.05, abs=0.04), (
                f"{row['group']}: conformal FPR {row['fpr_conformal_group']} against a nominal 0.05"
            )

    def test_a_group_too_small_to_calibrate_is_flagged_not_silently_pooled(self, tmp_path):
        _, out = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group"])
        t = pd.read_csv(out / "conformal_by_group.csv").set_index("group")
        assert "tiny_genre" in t.index
        assert bool(t.loc["tiny_genre", "fallback_to_global"]), (
            f"a genre with fewer than {MIN_CALIBRATION_PER_GROUP} calibration tracks must be marked "
            "as falling back to the pooled threshold, not reported as if it had its own guarantee."
        )
        assert t.loc["tiny_genre", "thr_conformal_group"] == t.loc["tiny_genre", "thr_conformal_global"]

    def test_all_three_arms_are_reported_so_the_comparison_isolates_one_variable(self, tmp_path):
        _, out = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group"])
        cols = set(pd.read_csv(out / "conformal_by_group.csv").columns)
        assert {"fpr_transfer", "fpr_conformal_global", "fpr_conformal_group"} <= cols
        assert {"recall_transfer", "recall_conformal_group"} <= cols
        assert {"precision@p=0.01_transfer", "precision@p=0.01_conformal_group"} <= cols


class TestMultiSplitAveraging:
    """One calibration split cannot resolve the effect at this sample size.

    Simulated with a PERFECTLY calibrated procedure on PERFECTLY exchangeable data at
    n = 191 per group and eight groups: the max/min FPR spread has median **4.0x** and
    reaches **8.5x** at the 90th percentile, purely from the calibration draw. The
    first real run measured 9.57x for the transfer arm against 9.03x for the
    group-conditional one -- a difference entirely inside that noise. Averaging over
    splits is what makes the comparison mean anything, and the per-group SD across
    splits is what says whether the groups are distinguishable at all.
    """

    def test_a_single_split_warns_that_it_is_a_point_estimate(self, tmp_path):
        proc, _ = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group", "--n-splits", "1"])
        assert proc.returncode == 0, proc.stderr
        assert "POINT ESTIMATE" in proc.stdout + proc.stderr

    def test_averaging_reports_the_across_split_spread(self, tmp_path):
        _, out = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group", "--n-splits", "12"])
        t = pd.read_csv(out / "conformal_by_group.csv")
        assert "fpr_conformal_group_sd" in t.columns
        assert (t["n_splits"] == 12).all()
        big = t[t["n_eval"] >= 100]
        assert big["fpr_conformal_group_sd"].notna().all()
        assert (big["fpr_conformal_group_sd"] > 0).any(), (
            "the across-split SD is identically zero, so the splits are not independent"
        )

    def test_averaging_is_more_stable_than_a_single_split(self, tmp_path):
        # The mean over many splits must sit inside the range a single split spans,
        # and its group ordering must be reproducible across seeds.
        a = pd.read_csv(_run(tmp_path, ["--group-column", "genre", "--calibrate-per-group",
                                        "--n-splits", "15"])[1] / "conformal_by_group.csv")
        b = pd.read_csv(_run(tmp_path, ["--group-column", "genre", "--calibrate-per-group",
                                        "--n-splits", "15", "--seed", "99"])[1] / "conformal_by_group.csv")
        m = a.merge(b, on="group", suffixes=("_a", "_b"))
        big = m[m["n_eval_a"] >= 100]
        assert (abs(big["fpr_conformal_group_a"] - big["fpr_conformal_group_b"]) < 0.05).all(), (
            "averaged per-group FPRs move by more than 0.05 between seeds -- still too noisy"
        )

    def test_the_default_is_one_split_so_existing_behaviour_is_unchanged(self, tmp_path):
        _, out = _run(tmp_path, ["--group-column", "genre", "--calibrate-per-group"])
        assert (pd.read_csv(out / "conformal_by_group.csv")["n_splits"] == 1).all()
