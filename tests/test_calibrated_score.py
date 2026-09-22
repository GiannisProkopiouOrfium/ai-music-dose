"""``comb_calibrated_score``: one definition of the margin, two call paths.

Why this file exists
--------------------
The margin ``real - max(null)`` was only ever formed at CSV level, by
``scripts/derive_analytic_null.py`` and ``scripts/derive_prior_margin.py``. The two
deployment scripts -- ``run_robustness_battery.py`` and ``measure_efficiency.py`` --
score one audio buffer at a time and had no way to reach it: their ``--training-free``
flag accepted ``comb_strength`` and ``comb_stat_strength`` and nothing else. **So
neither could measure the detector the paper proposes**, and the robustness and cost
figures in the paper are the raw comb arm's (ledger R27.44).

``comb_calibrated_score`` gives the per-buffer path the same arithmetic. Two code paths
computing "the margin" is exactly how a published number drifts from the thing it is
named after, so the tests below assert they agree rather than trusting that they do.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import comb_calibrated_score


def _feats(order: int = 4, n_decoys: int = 24, seed: int = 0) -> dict[str, float]:
    """A feature dict shaped exactly like one row of ``comb_features()`` output."""
    rng = np.random.RandomState(seed)
    f = {
        "comb_strength": 0.44,
        "comb_stat_strength": 0.31,
        f"comb_priormax{order}_strength": 0.50,
        f"comb_priormax{order}_hmax_strength": 0.12,
        f"comb_priormax{order}_latmax_strength": 0.20,
        f"comb_priormax{order}_unionmax_strength": 0.22,
        f"comb_harm{order}_strength": 0.33,
        f"comb_harm{order}_sharpness": 6.6,
    }
    for j in range(n_decoys):
        f[f"comb_priormax{order}_decoy{j:02d}_strength"] = float(rng.uniform(0.05, 0.31))
    return f


# --- the CSV-level definitions, transcribed from the scripts as oracles ------
def _csv_analytic_margin(row: dict, order: int, null_key: str) -> float:
    """``derive_analytic_null.py``: ``num(real_col) - num(null_col)``."""
    return row[f"comb_priormax{order}_strength"] - row[f"comb_priormax{order}_{null_key}"]


def _csv_decoy_margin(row: dict, order: int) -> float:
    """``derive_prior_margin.py:94``: ``real - np.nanmax(decoys, axis=1)``."""
    decoys = [v for k, v in row.items() if re.fullmatch(rf"comb_priormax{order}_decoy\d+_strength", k)]
    return row[f"comb_priormax{order}_strength"] - float(np.nanmax(decoys))


class TestTheTwoPathsAgree:
    """The per-buffer score must equal the per-track CSV column, exactly."""

    @pytest.mark.parametrize(
        "suffix,null_key",
        [("hmargin", "hmax_strength"), ("latmargin", "latmax_strength"), ("unionmargin", "unionmax_strength")],
    )
    @pytest.mark.parametrize("order", [2, 4])
    def test_analytic_margins_match_the_csv_definition(self, suffix, null_key, order):
        f = _feats(order=order)
        got = comb_calibrated_score(f, f"comb_priormax{order}_{suffix}")
        want = _csv_analytic_margin(f, order, null_key)
        assert got == pytest.approx(want, abs=0), (
            "the per-buffer margin and the per-track CSV margin disagree. Two definitions of "
            "the same published number is how R27.44's gap appeared in the first place."
        )

    @pytest.mark.parametrize("order", [2, 4])
    def test_the_decoy_margin_matches_derive_prior_margin(self, order):
        f = _feats(order=order)
        got = comb_calibrated_score(f, f"comb_priormax{order}_margin")
        assert got == pytest.approx(_csv_decoy_margin(f, order), abs=0)

    def test_a_larger_null_gives_a_smaller_margin(self):
        f = _feats()
        h = comb_calibrated_score(f, "comb_priormax4_hmargin")
        u = comb_calibrated_score(f, "comb_priormax4_unionmargin")
        assert h > u, "hmax 0.12 < unionmax 0.22, so the hmargin must be the larger"


class TestBackwardsCompatibility:
    """The two names the scripts accepted before must behave exactly as before."""

    @pytest.mark.parametrize("name,value", [("comb_strength", 0.44), ("comb_stat_strength", 0.31)])
    def test_raw_columns_pass_through_untouched(self, name, value):
        assert comb_calibrated_score(_feats(), name) == value


class TestItFailsLoudlyRatherThanSilently:
    """A missing column must raise, never return 0.0 or a raw score by accident."""

    def test_a_margin_whose_null_column_is_absent_raises(self):
        f = _feats(order=4)
        del f["comb_priormax4_hmax_strength"]
        with pytest.raises(KeyError, match="n_harm_grid"):
            comb_calibrated_score(f, "comb_priormax4_hmargin")

    def test_a_harmonic_order_that_was_not_emitted_raises(self):
        with pytest.raises(KeyError):
            comb_calibrated_score(_feats(order=4), "comb_priormax8_hmargin")

    def test_a_decoy_margin_with_no_decoys_raises_and_names_the_flag(self):
        f = {k: v for k, v in _feats().items() if "decoy" not in k}
        with pytest.raises(KeyError, match="null_priors"):
            comb_calibrated_score(f, "comb_priormax4_margin")

    def test_a_non_finite_input_raises_rather_than_propagating_nan(self):
        f = _feats()
        f["comb_priormax4_hmax_strength"] = float("nan")
        with pytest.raises(ValueError):
            comb_calibrated_score(f, "comb_priormax4_hmargin")
        f2 = _feats()
        f2["comb_strength"] = float("nan")
        with pytest.raises(ValueError):
            comb_calibrated_score(f2, "comb_strength")

    def test_an_unknown_name_raises(self):
        with pytest.raises(KeyError, match="unknown calibrated score"):
            comb_calibrated_score(_feats(), "comb_not_a_score")


class TestTheHarmonicMarginNeedsNoDecoys:
    """Its null is deterministic, so it can be computed -- and TIMED -- at null_priors=0.

    This is why the efficiency script sets ``null_priors = 0`` for ``*_hmargin``: the
    24 decoy evaluations per harmonic order are the bulk of the proposed detector's
    cost, and the harmonic-protected null does not need them.
    """

    def test_it_works_with_no_decoy_columns_at_all(self):
        f = {k: v for k, v in _feats().items() if "decoy" not in k}
        assert comb_calibrated_score(f, "comb_priormax4_hmargin") == pytest.approx(0.38)
        assert comb_calibrated_score(f, "comb_priormax4_latmargin") == pytest.approx(0.30)

    def test_but_the_decoy_margin_does_not(self):
        f = {k: v for k, v in _feats().items() if "decoy" not in k}
        with pytest.raises(KeyError):
            comb_calibrated_score(f, "comb_priormax4_margin")
