"""``scripts/analyze_detector.py``: the left tail, and that the old outputs did not move.

Two jobs.

**Guard the contract.** This script's four original outputs carry published numbers
(ledger R25 reads per-family recall out of ``errors_by_generator.csv``, and the
deployment precision of 0.1265 out of ``operating_points.csv``). The extension that
adds a left-tail table must not perturb a single column of them, and "I did not mean
to change it" is not a control. The tests below pin the exact column list of
``operating_points.csv`` and require the new per-quantile table to agree with the old
one at q=0.95 by construction.

**Pin the scale of the new number.** ``partial_auc`` returns the MEAN TPR over an FPR
range, not an AUC: its chance level is ``max_fpr / 2``, which is 0.005 at the default
0.01, not 0.5. That is an easy number to misread as "barely above chance" when it is
in fact a hundredfold above it, so it is asserted rather than left to the docstring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "analyze_detector.py"
sys.path.insert(0, str(REPO / "scripts"))

from analyze_detector import (  # noqa: E402
    PAUC_MAX_FPR,
    partial_auc,
    roc_curve,
    threshold_grid,
)

# The columns operating_points.csv has always had. R25 and the paper read from these.
HISTORICAL_OPERATING_POINT_COLUMNS = [
    "real_quantile",
    "threshold",
    "fpr_point",
    "fpr_upper95",
    "n_false_pos",
    "n_eval_reals",
    "tpr_recall",
    "precision@p=0.1",
    "precision@p=0.05",
    "precision@p=0.01",
    "precision@p=0.001",
]


def _fixture_csv(path: Path, n: int = 1200, separation: float = 1.2) -> Path:
    rng = np.random.RandomState(7)
    label = ["real"] * (n // 2) + ["fake"] * (n // 2)
    algorithm = [""] * (n // 2) + ["gen_a"] * (n // 4) + ["gen_b"] * (n - n // 2 - n // 4)
    is_fake = np.array([lab != "real" for lab in label], dtype=float)
    pd.DataFrame(
        {
            "track_id": [f"t{i:04d}" for i in range(n)],
            "label": label,
            "algorithm": algorithm,
            "comb_x": rng.randn(n) * 0.4 + separation * is_fake,
        }
    ).to_csv(path, index=False)
    return path


def _run(tmp_path: Path, extra: list[str] | None = None) -> Path:
    src = _fixture_csv(tmp_path / "fx.csv")
    out = tmp_path / ("out" + ("_x" if extra else ""))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--score-csv", str(src), "--score-column", "comb_x", "--out-dir", str(out)]
        + (extra or []),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return out


class TestTheOriginalOutputsDidNotMove:
    def test_operating_points_keeps_exactly_its_historical_columns(self, tmp_path):
        op = pd.read_csv(_run(tmp_path) / "operating_points.csv")
        assert list(op.columns) == HISTORICAL_OPERATING_POINT_COLUMNS, (
            "operating_points.csv's column list changed. Ledger R25 and the paper's deployment "
            "numbers are read from this file by column name."
        )

    def test_the_default_quantile_grid_is_the_historical_inline_one(self, tmp_path):
        op = pd.read_csv(_run(tmp_path) / "operating_points.csv")
        assert op["real_quantile"].tolist() == [0.90, 0.95, 0.99, 0.999]

    def test_all_four_original_files_are_still_written(self, tmp_path):
        out = _run(tmp_path)
        for name in (
            "operating_points.csv",
            "precision_targets.csv",
            "errors_by_generator.csv",
            "per_track_with_errors.csv",
        ):
            assert (out / name).exists()

    def test_errors_by_generator_is_unaffected_by_the_quantiles_flag(self, tmp_path):
        # The q=0.95 error breakdown is set by an explicit 0.95, not by the grid, so a
        # caller sweeping quantiles cannot silently move a published number.
        a = (_run(tmp_path) / "errors_by_generator.csv").read_text()
        b = (_run(tmp_path, ["--quantiles", "0.5", "0.8"]) / "errors_by_generator.csv").read_text()
        assert a == b


class TestThePerQuantileBreakdown:
    def test_it_agrees_with_the_original_table_at_q95(self, tmp_path):
        out = _run(tmp_path)
        old = pd.read_csv(out / "errors_by_generator.csv").set_index("algorithm")["recall_at_q95"]
        new = pd.read_csv(out / "errors_by_generator_by_q.csv")
        new = new[new["real_quantile"] == 0.95].set_index("algorithm")["recall"]
        for gen in old.index:
            assert new[gen] == pytest.approx(old[gen], abs=1e-9), (
                f"{gen}: the per-quantile table disagrees with errors_by_generator.csv at q=0.95. "
                "They are the same quantity and must be computed the same way."
            )

    def test_recall_falls_as_the_threshold_tightens(self, tmp_path):
        by_q = pd.read_csv(_run(tmp_path) / "errors_by_generator_by_q.csv")
        for _, sub in by_q.groupby("algorithm"):
            sub = sub.sort_values("real_quantile")
            assert sub["recall"].is_monotonic_decreasing


class TestTheLeftTail:
    def test_the_table_exists_and_is_keyed_by_target_fpr(self, tmp_path):
        tail = pd.read_csv(_run(tmp_path) / "left_tail.csv")
        assert tail["target_fpr"].tolist() == [0.001, 0.01, 0.05]
        assert f"pauc_fpr{PAUC_MAX_FPR:g}_meantpr" in tail.columns

    def test_the_achieved_fpr_tracks_the_target(self, tmp_path):
        tail = pd.read_csv(_run(tmp_path) / "left_tail.csv")
        for _, row in tail.iterrows():
            assert row["achieved_fpr_point"] <= row["target_fpr"] + 0.02

    def test_recall_rises_as_the_allowed_fpr_rises(self, tmp_path):
        tail = pd.read_csv(_run(tmp_path) / "left_tail.csv").sort_values("target_fpr")
        assert tail["tpr_at_fpr"].is_monotonic_increasing

    def test_the_pauc_is_one_value_not_four(self, tmp_path):
        # It does not depend on the real-quantile grid, which is why it is here and
        # not repeated down operating_points.csv.
        tail = pd.read_csv(_run(tmp_path) / "left_tail.csv")
        assert tail[f"pauc_fpr{PAUC_MAX_FPR:g}_meantpr"].nunique() == 1


class TestPartialAucScale:
    """It is a MEAN TPR, so chance is ``max_fpr / 2`` -- not 0.5."""

    def test_perfect_separation_gives_one(self):
        real = np.linspace(0.0, 1.0, 500)
        fake = np.linspace(2.0, 3.0, 500)
        assert partial_auc(real, fake, 0.01) == pytest.approx(1.0, abs=1e-6)

    def test_chance_sits_at_half_the_max_fpr_not_at_half(self):
        rng = np.random.RandomState(0)
        real, fake = rng.randn(20_000), rng.randn(20_000)
        got = partial_auc(real, fake, 0.01)
        assert got == pytest.approx(0.005, abs=0.004), (
            f"chance gave {got:.4f}. This column is the mean TPR over an FPR range, so its chance "
            "level is max_fpr/2 = 0.005. Reading it as an AUC would make a strong detector look "
            "like a failure and a chance one look catastrophic."
        )

    def test_it_is_bounded_below_by_the_tpr_at_the_range_edge(self):
        rng = np.random.RandomState(1)
        real, fake = rng.randn(4_000), rng.randn(4_000) + 1.5
        thr = np.quantile(real, 0.99)
        tpr_at_edge = float((fake > thr).mean())
        assert partial_auc(real, fake, 0.01) <= tpr_at_edge + 1e-9

    def test_the_roc_starts_at_the_origin(self):
        fpr, tpr = roc_curve(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)

    def test_an_empty_class_is_nan_rather_than_a_crash(self):
        assert np.isnan(partial_auc(np.array([]), np.array([1.0]), 0.01))
        assert np.isnan(partial_auc(np.array([1.0]), np.array([]), 0.01))


class TestThePrecisionTargetGridResolvesTheTail:
    """The two tables must not contradict each other. On 2026-09-17 they did.

    ``analysis_pm4_sonics`` reported precision 0.9092 at 1% prevalence in
    ``operating_points.csv`` and, from the same run, called precision 0.90 at 1%
    unreachable in ``precision_targets.csv`` with a best of 0.8830. The cause was a
    threshold grid uniform in the quantile: its last step spanned 0.99874 to 0.99999,
    eight tracks wide on 6,361 reals, straight over the operating point the first
    table was standing on.

    The invariant below is the one that was violated, and it is checkable without
    knowing the right answer: whatever the best reachable precision is, it cannot be
    lower than one the script itself already achieved.
    """

    def test_best_reachable_is_never_below_an_operating_point_actually_reached(self, tmp_path):
        out = _run(tmp_path)
        op = pd.read_csv(out / "operating_points.csv")
        pt = pd.read_csv(out / "precision_targets.csv")
        for prevalence in (0.1, 0.05, 0.01, 0.001):
            reached = op[f"precision@p={prevalence:g}"].max()
            claimed = pt.loc[pt["prevalence"] == prevalence, "best_precision_reachable"].max()
            assert claimed >= reached - 1e-6, (
                f"p={prevalence}: precision_targets.csv claims the best reachable precision is "
                f"{claimed:.4f}, but operating_points.csv from the same run already reached "
                f"{reached:.4f}. The threshold grid cannot see an operating point the script "
                "itself reports."
            )

    def test_the_tail_grid_puts_most_of_its_points_where_precision_is_decided(self):
        rng = np.random.RandomState(0)
        reals = rng.randn(6_361)
        counts = {}
        for mode in ("linear", "tail"):
            g = threshold_grid(reals, mode=mode)
            counts[mode] = int(sum((reals > t).mean() <= 0.01 for t in g))
        assert counts["tail"] > 10 * counts["linear"], (
            f"linear grid {counts['linear']} points below FPR 0.01, tail grid {counts['tail']}. "
            "Precision 0.9 at 1% prevalence needs FPR near 0.00076, so a grid that spends its "
            "budget above FPR 0.01 cannot find it."
        )

    def test_the_tail_grid_is_a_strict_superset_so_it_cannot_regress(self):
        # Adding candidate thresholds can only raise the best precision the search
        # finds. Making 'tail' a superset of 'linear' therefore guarantees the fix is
        # an improvement everywhere rather than a different trade-off.
        rng = np.random.RandomState(2)
        reals = rng.randn(3_000)
        lin = set(np.round(threshold_grid(reals, mode="linear"), 12))
        tail = set(np.round(threshold_grid(reals, mode="tail"), 12))
        assert lin <= tail, f"{len(lin - tail)} historical grid points are missing from the tail grid"
        assert len(tail) > len(lin)

    def test_best_reachable_never_falls_when_switching_to_the_tail_grid(self, tmp_path):
        lin = pd.read_csv(_run(tmp_path, ["--precision-grid", "linear"]) / "precision_targets.csv")
        tail = pd.read_csv(_run(tmp_path) / "precision_targets.csv")
        m = lin.merge(tail, on=["prevalence", "target_precision"], suffixes=("_lin", "_tail"))
        worse = m[m["best_precision_reachable_tail"] < m["best_precision_reachable_lin"] - 1e-9]
        assert worse.empty, worse.to_string(index=False)

    def test_the_linear_mode_still_reproduces_the_historical_grid_exactly(self):
        rng = np.random.RandomState(1)
        reals = rng.randn(500)
        assert np.allclose(
            threshold_grid(reals, mode="linear"),
            np.quantile(reals, np.linspace(0.5, 0.99999, 400)),
        ), "the superseded grid must stay reproducible, per the CORRECTED-marker discipline"

    def test_an_empty_input_is_an_empty_grid_not_a_crash(self):
        assert len(threshold_grid(np.array([]), mode="tail")) == 0
