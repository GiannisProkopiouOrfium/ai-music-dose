"""``scripts/derive_analytic_null.py``: the floor is recoverable, and exactly.

The whole script rests on one identity -- that ``comb_harmonic``'s sharpness is the
peak divided by the floor, so the floor comes back as ``strength / sharpness`` from a
CSV that was written months ago. If that identity ever stops holding (someone changes
the guard constant, or divides by something else), every number derived here silently
becomes a different quantity with the same name. The first test checks the inversion
against the real ``comb_harmonic`` rather than against a restatement of it.

The remaining tests cover the degenerate inputs this repository has actually been
bitten by: an ALL-REAL corpus (the FMA transfer analysis and the C10 reconstruction
control are both all-real, and an unconditional sort on a missing MACRO column is
ledger R17), a column that is entirely NaN, and a CSV missing the paired column.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import (
    _harmonic_curve,
    _normalised_acf,
    comb_harmonic,
)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "derive_analytic_null.py"

BIN_HZ = 1.0
N_BINS = 4000


def _comb(spacing_bins: int, tooth: float, noise: float = 0.30, seed: int = 0) -> np.ndarray:
    """A residual with teeth every ``spacing_bins``, on a noise floor."""
    rng = np.random.RandomState(seed)
    r = rng.rand(N_BINS) * noise
    r[::spacing_bins] += tooth
    return r


def _floor_directly(residual: np.ndarray, n_harm: int) -> float:
    """``median(|H|)`` computed the way ``comb_harmonic`` computes it internally.

    Mirrors ``comb_artifacts.py:487-493`` -- deliberately recomputed from the shared
    primitives rather than copied as a constant, so a change to the band or the
    harmonic curve breaks this test instead of passing silently.
    """
    acf = _normalised_acf(residual)
    lo = max(int(round(40.0 / BIN_HZ)), 1)
    hi = min(int(round(1_500.0 / BIN_HZ)), (len(acf) - 1) // max(n_harm, 1))
    _, curve = _harmonic_curve(acf, lo, hi, n_harm)
    return float(np.median(np.abs(curve)))


class TestTheFloorInversionIsExact:
    """``strength / sharpness`` must return ``comb_harmonic``'s own floor."""

    @pytest.mark.parametrize("n_harm", [2, 4, 8])
    @pytest.mark.parametrize("tooth", [0.05, 0.30, 1.00])
    def test_round_trip_against_comb_harmonic(self, n_harm, tooth):
        sys.path.insert(0, str(REPO / "scripts"))
        from derive_analytic_null import recover_floor

        residual = _comb(84, tooth, seed=3)
        h = comb_harmonic(residual, bin_hz=BIN_HZ, n_harm=n_harm)
        recovered = float(
            recover_floor(
                np.array([h["comb_harm_strength"]]),
                np.array([h["comb_harm_sharpness"]]),
            )[0]
        )
        assert recovered == pytest.approx(_floor_directly(residual, n_harm), rel=1e-9), (
            "strength / sharpness no longer returns comb_harmonic's floor. Every column "
            "derive_analytic_null.py writes is then a different quantity under the same name."
        )

    def test_a_degenerate_sharpness_gives_nan_not_infinity(self):
        sys.path.insert(0, str(REPO / "scripts"))
        from derive_analytic_null import recover_floor

        out = recover_floor(np.array([1.0, 1.0, np.nan]), np.array([0.0, np.nan, 1.0]))
        assert np.isnan(out).all(), "a track with no usable null must be NaN, never inf"


def _frame(n: int = 240, all_real: bool = False, nan_sharpness: bool = False) -> pd.DataFrame:
    """A per-track CSV shaped like ``comb_per_track*.csv``, with a real separation.

    Fakes get a stronger harmonic peak than reals, so ``_nmargin`` has a signal to
    find and ``n_inverted`` is meaningful rather than an artefact of symmetric noise.
    """
    rng = np.random.RandomState(11)
    label = ["real"] * n if all_real else ["real"] * (n // 2) + ["fake"] * (n // 2)
    algorithm = [""] * n if all_real else [""] * (n // 2) + ["gen_a"] * (n // 4) + ["gen_b"] * (n - n // 2 - n // 4)
    is_fake = np.array([lab != "real" for lab in label], dtype=float)
    strength = 0.10 + 0.25 * is_fake + rng.rand(n) * 0.04
    floor = 0.05 + rng.rand(n) * 0.01
    df = pd.DataFrame(
        {
            "track_id": [f"t{i:04d}" for i in range(n)],
            "label": label,
            "algorithm": algorithm,
            "comb_harm4_strength": strength,
            "comb_harm4_sharpness": np.nan if nan_sharpness else strength / floor,
            "comb_priormax4_strength": strength * 0.9,
        }
    )
    return df


def _run(tmp_path: Path, df: pd.DataFrame):
    src = tmp_path / "per_track.csv"
    out = tmp_path / "analytic_null.csv"
    df.to_csv(src, index=False)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--score-csv", str(src), "--out-csv", str(out)],
        capture_output=True,
        text=True,
    )
    return proc, out, out.with_name(out.stem + "_auc.csv")


class TestEndToEnd:
    def test_it_writes_the_columns_and_the_auc_table(self, tmp_path):
        proc, out, auc = _run(tmp_path, _frame())
        assert proc.returncode == 0, proc.stderr
        got = pd.read_csv(out)
        for col in ("comb_harm4_floor", "comb_harm4_nmargin", "comb_priormax4_xfloor"):
            assert col in got.columns
        # The recovered floor must match the one the fixture was built from.
        expected = got["comb_harm4_strength"] / got["comb_harm4_sharpness"]
        assert np.allclose(got["comb_harm4_floor"], expected)
        assert auc.exists()
        table = pd.read_csv(auc)
        assert {"score", "MACRO", "n_inverted"} <= set(table.columns)
        assert set(table["score"]) == {"comb_harm4_floor", "comb_harm4_nmargin", "comb_priormax4_xfloor"}

    def test_the_difference_form_is_correctly_oriented_on_a_separable_fixture(self, tmp_path):
        _, out, auc = _run(tmp_path, _frame())
        table = pd.read_csv(auc).set_index("score")
        assert table.loc["comb_harm4_nmargin", "MACRO"] > 0.9
        assert table.loc["comb_harm4_nmargin", "n_inverted"] == 0

    def test_an_all_real_corpus_writes_no_auc_table_and_does_not_crash(self, tmp_path):
        # Ledger R17: derive_prior_margin.py raised KeyError('MACRO') on exactly this
        # input. An all-real corpus is legitimate -- it is what C10 and the FMA
        # false-positive analysis both use.
        proc, out, auc = _run(tmp_path, _frame(all_real=True))
        assert proc.returncode == 0, proc.stderr
        assert out.exists() and not auc.exists()
        assert "ALL-REAL CORPUS" in proc.stdout

    def test_an_all_nan_sharpness_column_yields_nan_not_a_crash(self, tmp_path):
        proc, out, _ = _run(tmp_path, _frame(nan_sharpness=True))
        assert proc.returncode == 0, proc.stderr
        assert pd.read_csv(out)["comb_harm4_floor"].isna().all()

    def test_a_missing_sharpness_column_is_skipped_loudly(self, tmp_path):
        df = _frame().drop(columns=["comb_harm4_sharpness"])
        proc, _, _ = _run(tmp_path, df)
        assert proc.returncode != 0
        assert "--n-harm" in proc.stdout + proc.stderr, "the failure must name the flag that causes it"

    def test_a_csv_with_no_harmonic_columns_exits_with_an_explanation(self, tmp_path):
        df = _frame()[["track_id", "label", "algorithm"]]
        proc, _, _ = _run(tmp_path, df)
        assert proc.returncode != 0
        assert "nothing to derive" in proc.stdout + proc.stderr


def _detnull_frame(n: int = 240, overlap: bool = False) -> pd.DataFrame:
    """A per-track CSV shaped like a `_DETNULL` run: lattice null columns present.

    ``overlap=True`` simulates a null that contains the prior's own winning lag, which
    forces the margin to exactly zero -- the pathology the zero-atom diagnostic exists
    to surface.
    """
    rng = np.random.RandomState(23)
    label = ["real"] * (n // 2) + ["fake"] * (n // 2)
    is_fake = np.array([lab != "real" for lab in label], dtype=float)
    real = 0.10 + 0.25 * is_fake + rng.rand(n) * 0.04
    null = real.copy() if overlap else 0.06 + rng.rand(n) * 0.02
    return pd.DataFrame(
        {
            "track_id": [f"t{i:04d}" for i in range(n)],
            "label": label,
            "algorithm": [""] * (n // 2) + ["gen_a"] * (n - n // 2),
            "comb_priormax4_strength": real,
            "comb_priormax4_latmax_strength": null,
            "comb_priormax4_latpval": rng.rand(n) * (1.0 - 0.9 * is_fake),
            "comb_acf_floor": 0.03 + rng.rand(n) * 0.005,
        }
    )


class TestTheLatticeFamily:
    def test_it_derives_a_margin_a_confidence_and_an_analytic_margin(self, tmp_path):
        proc, out, auc = _run(tmp_path, _detnull_frame())
        assert proc.returncode == 0, proc.stderr
        got = pd.read_csv(out)
        for col in ("comb_priormax4_latmargin", "comb_priormax4_latconf", "comb_priormax4_afmargin"):
            assert col in got.columns, col
        assert np.allclose(
            got["comb_priormax4_latmargin"],
            got["comb_priormax4_strength"] - got["comb_priormax4_latmax_strength"],
        )
        # The p-value is flipped so higher means more generated, like every other score.
        assert np.allclose(got["comb_priormax4_latconf"], 1.0 - got["comb_priormax4_latpval"])

    def test_the_exact_acf_floor_is_preferred_over_the_approximate_cross_form(self, tmp_path):
        _, out, _ = _run(tmp_path, _detnull_frame())
        cols = pd.read_csv(out, nrows=1).columns
        assert "comb_priormax4_afmargin" in cols
        assert "comb_priormax4_xfloor" not in cols, (
            "comb_acf_floor was present, so the dimensionally exact margin should have been used "
            "and the approximate cross-form suppressed"
        )

    def test_the_zero_atom_diagnostic_catches_a_null_that_contains_the_prior(self, tmp_path):
        _, out, _ = _run(tmp_path, _detnull_frame(overlap=True))
        diag = pd.read_csv(out.with_name(out.stem + "_zeroatom.csv")).set_index("score")
        assert diag.loc["comb_priormax4_latmargin", "frac_exactly_zero"] == 1.0, (
            "a null equal to the prior's own score must show up as a margin pinned at zero -- that "
            "is the mechanism behind comb_priormax8_margin's collapse"
        )

    def test_a_clean_null_shows_no_zero_atom(self, tmp_path):
        _, out, _ = _run(tmp_path, _detnull_frame(overlap=False))
        diag = pd.read_csv(out.with_name(out.stem + "_zeroatom.csv")).set_index("score")
        assert diag.loc["comb_priormax4_latmargin", "frac_exactly_zero"] == 0.0
