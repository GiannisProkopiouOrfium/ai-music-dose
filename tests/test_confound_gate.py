"""Gate the gate.

A control that cannot fail is not a control — that is the lesson of the
Butterworth low-pass, which "passed" as a bandwidth control while relocating the
leak from an energy ratio (1.0000) to spectral flatness (0.9287).

So every gate here is exercised twice: once on data with a DELIBERATELY PLANTED
confound, where it must FAIL, and once on clean data, where it must PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_confound_gate import (  # noqa: E402
    CHANNEL_DESCRIPTORS,
    LEVEL_DESCRIPTORS,
    _descriptor_gate,
    gate_content_identical,
    gate_duration,
    gate_harmonicity,
    gate_row_survival,
    gate_score_sanity,
    gate_shuffle,
    gate_spacing,
)

N = 400


def _frame(leak: str | None = None, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A balanced two-class frame; ``leak`` plants one specific confound.

    A FRESH seeded RNG per call. A module-level RNG makes the fixture depend on
    how many tests ran before it, so the same assertion passes alone and fails in
    the suite — which is exactly what happened here.
    """
    RNG = np.random.RandomState(seed)
    y = np.r_[np.zeros(N // 2, int), np.ones(N // 2, int)]
    alg = np.where(y == 1, RNG.choice(["gen_a", "gen_b"], N), "")
    # Each generator has ITS OWN spacing, tightly clustered — the decoder
    # signature the gate is supposed to accept.
    # Generators sit tightly on their own stride-determined spacing; REAL music's
    # "spacing" is whatever the melody's f0 happens to be, so it is broad. That
    # contrast — concentration, not spread — is what C8 tests.
    per_gen = np.where(alg == "gen_a", 200.0, np.where(alg == "gen_b", 400.0, 300.0))
    jitter = np.where(alg == "", RNG.randn(N) * 90.0, RNG.randn(N) * 4.0)
    df = pd.DataFrame(
        {
            "track_id": [f"t{i}" for i in range(N)],
            "label": np.where(y == 1, "fake", "real"),
            "algorithm": alg,
            # clean defaults: identically distributed in both classes
            "frac_power_8k_11k": RNG.randn(N),
            "cutoff_hz": RNG.randn(N),
            "dc_offset": RNG.randn(N),
            "peak_dbfs": RNG.randn(N),
            "crest_factor_db": RNG.randn(N),
            "duration_s": np.full(N, 9.0),
            "harmonicity": RNG.randn(N),
            "comb_spacing_hz": per_gen + jitter,
        }
    )
    score = RNG.randn(N) + y * 1.2  # a genuinely informative detector

    if leak == "channel":
        df["frac_power_8k_11k"] = y * 6.0 + RNG.randn(N) * 0.1
    elif leak == "level":
        df["dc_offset"] = y * 6.0 + RNG.randn(N) * 0.1
    elif leak == "duration":
        df["duration_s"] = np.where(y == 1, 4.0, 9.0)
    elif leak == "harmonicity":
        df["harmonicity"] = score * 3.0 + RNG.randn(N) * 0.05
    elif leak == "spacing":
        # A channel artifact pushes BOTH classes to one value with the same
        # scatter: generators no longer concentrate more than real music does.
        df["comb_spacing_hz"] = 300.0 + RNG.randn(N) * 90.0
    elif leak == "constant_scores":
        score = np.full(N, 0.5)
    elif leak == "leaky_score":
        score = y * 1000.0 + RNG.randn(N) * 1e-6  # near-perfect, for the shuffle gate
    return df, y, score


class TestChannelGate:
    def test_fails_on_a_planted_channel_leak(self):
        df, y, _ = _frame("channel")
        assert _descriptor_gate("C1", "channel-alone", df, y, CHANNEL_DESCRIPTORS).passed is False

    def test_passes_on_clean_data(self):
        df, y, _ = _frame()
        assert _descriptor_gate("C1", "channel-alone", df, y, CHANNEL_DESCRIPTORS).passed is True

    def test_skips_rather_than_passes_when_descriptors_are_absent(self):
        """A missing control is UNTESTED, never PASS. This distinction is the
        whole reason `passed` is tri-state."""
        df, y, _ = _frame()
        bare = df[["track_id", "label", "algorithm"]]
        assert _descriptor_gate("C1", "channel-alone", bare, y, CHANNEL_DESCRIPTORS).passed is None


class TestLevelGate:
    def test_fails_on_a_planted_level_leak(self):
        df, y, _ = _frame("level")
        assert _descriptor_gate("C2", "level", df, y, LEVEL_DESCRIPTORS).passed is False

    def test_passes_on_clean_data(self):
        df, y, _ = _frame()
        assert _descriptor_gate("C2", "level", df, y, LEVEL_DESCRIPTORS).passed is True


class TestDurationGate:
    def test_fails_when_one_class_is_systematically_shorter(self):
        df, y, _ = _frame("duration")
        assert gate_duration(df, y).passed is False

    def test_passes_when_durations_match(self):
        df, y, _ = _frame()
        assert gate_duration(df, y).passed is True


class TestHarmonicityGate:
    def test_fails_when_the_score_tracks_harmonicity(self):
        df, y, s = _frame("harmonicity")
        assert gate_harmonicity(df, s, y).passed is False

    def test_passes_when_it_does_not(self):
        df, y, s = _frame()
        assert gate_harmonicity(df, s, y).passed is True

    def test_skips_when_no_harmonicity_descriptor_exists(self):
        df, y, s = _frame()
        assert gate_harmonicity(df.drop(columns=["harmonicity"]), s, y).passed is None


class TestSpacingGate:
    def test_fails_when_every_generator_sits_at_one_value(self):
        """A channel artifact pushes all generators to a single spacing."""
        df, y, _ = _frame("spacing")
        assert gate_spacing(df).passed is False

    def test_passes_when_spacing_clusters_by_generator(self):
        df, y, _ = _frame()
        assert gate_spacing(df).passed is True


class TestScoreSanityGate:
    def test_fails_on_constant_scores(self):
        """The shape of the invalidated FMA runs: byte-identical score files."""
        _, _, s = _frame("constant_scores")
        assert gate_score_sanity(s).passed is False

    def test_passes_on_varied_scores(self):
        _, _, s = _frame()
        assert gate_score_sanity(s).passed is True


class TestShuffleGate:
    def test_passes_for_a_normal_detector(self):
        _, y, s = _frame()
        assert gate_shuffle(y, s, seed=0).passed is True

    def test_shuffling_destroys_even_a_near_perfect_score(self):
        """Sanity on the gate itself: permuting labels must collapse ANY score."""
        _, y, s = _frame("leaky_score")
        assert gate_shuffle(y, s, seed=0).passed is True


class TestRowSurvivalGate:
    def test_fails_when_a_stratum_is_nearly_empty(self):
        df, _, _ = _frame()
        thin = pd.concat([df[df["label"] == "real"], df[df["algorithm"] == "gen_a"].head(2)])
        assert gate_row_survival(thin).passed is False

    def test_passes_on_a_balanced_frame(self):
        df, _, _ = _frame()
        assert gate_row_survival(df).passed is True


@pytest.mark.parametrize(
    "leak,gate_fn",
    [
        ("channel", lambda d, y, s: _descriptor_gate("C1", "c", d, y, CHANNEL_DESCRIPTORS)),
        ("level", lambda d, y, s: _descriptor_gate("C2", "l", d, y, LEVEL_DESCRIPTORS)),
        ("duration", lambda d, y, s: gate_duration(d, y)),
        ("harmonicity", lambda d, y, s: gate_harmonicity(d, s, y)),
        ("spacing", lambda d, y, s: gate_spacing(d)),
    ],
)
def test_every_planted_leak_is_caught_by_exactly_its_own_gate(leak, gate_fn):
    df, y, s = _frame(leak)
    assert gate_fn(df, y, s).passed is False, f"gate did not catch planted {leak!r} leak"


class TestC9ThresholdsWithinRealOnly:
    """CORRECTED 2026-08-22: C9 must threshold the WITHIN-REAL correlation.

    The confound is "does the detector fire on tonal REAL music?". A within-FAKE
    correlation means the score tracks peak count among tracks that HAVE decoder
    peaks — the mechanism working. Thresholding on it failed the FMC headline for
    succeeding (spectral_peak_count: real 0.105, fake 0.613).
    """

    @staticmethod
    def _frame_with(real_corr: float, fake_corr: float, n=400, seed=0):
        rng = np.random.RandomState(seed)
        y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
        score = rng.randn(n) + y * 1.5
        harm = np.empty(n)
        for cls, c in ((0, real_corr), (1, fake_corr)):
            m = y == cls
            z = rng.randn(m.sum())
            harm[m] = c * score[m] + np.sqrt(max(1 - c**2, 0.0)) * z
        df = pd.DataFrame(
            {
                "track_id": [f"t{i}" for i in range(n)],
                "label": np.where(y == 1, "fake", "real"),
                "algorithm": np.where(y == 1, "gen_a", ""),
                "harmonicity": harm,
            }
        )
        return df, y, score

    def test_high_within_real_correlation_FAILS(self):
        df, y, s = self._frame_with(real_corr=0.85, fake_corr=0.1)
        assert gate_harmonicity(df, s, y).passed is False

    def test_high_within_FAKE_correlation_alone_PASSES(self):
        """This is the mechanism, and it must not fail the gate."""
        df, y, s = self._frame_with(real_corr=0.05, fake_corr=0.85)
        assert gate_harmonicity(df, s, y).passed is True

    def test_spectral_peak_count_alone_is_not_thresholded(self):
        """It is near-tautological with a peak-energy score — diagnostic only."""
        df, y, s = self._frame_with(real_corr=0.9, fake_corr=0.9)
        df = df.rename(columns={"harmonicity": "spectral_peak_count"})
        assert gate_harmonicity(df, s, y).passed is None


class TestC10ReadsThePerTrackCsv:
    """C10 must work on the raw per-track CSV, not only a purpose-built file.

    `build_reconstruction_control.py --save-audio-dir` labels every variant
    `real` (they all derive from real audio), so the eval scripts correctly
    report "no AUC" — the comparison is WITHIN a pair_id. If C10 cannot read
    their output directly it stays SKIP forever, which is what happened.
    """

    @staticmethod
    def _pairs(tmp_path, delta, score_col="comb_strength"):
        rng = np.random.RandomState(0)
        n = 60
        base = rng.rand(n)
        rows = []
        for i in range(n):
            rows.append({"pair_id": f"p{i}", "variant": "source", score_col: base[i]})
            rows.append(
                {"pair_id": f"p{i}", "variant": "encodec_24kbps", score_col: base[i] + delta + rng.randn() * 0.01}
            )
        path = tmp_path / "recon.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_reads_per_track_csv_with_explicit_column(self, tmp_path):
        g = gate_content_identical(self._pairs(tmp_path, delta=0.2), "comb_strength")
        assert g.passed is True
        assert g.detail["per_variant"]["encodec_24kbps"]["frac_above_source"] > 0.9

    def test_reports_the_ordering_when_reconstruction_scores_lower(self, tmp_path):
        g = gate_content_identical(self._pairs(tmp_path, delta=-0.2), "comb_strength")
        assert g.detail["per_variant"]["encodec_24kbps"]["frac_above_source"] < 0.1

    def test_missing_score_column_is_SKIP_not_a_crash(self, tmp_path):
        g = gate_content_identical(self._pairs(tmp_path, delta=0.2), None)
        assert g.passed is None and "no 'score' column" in g.note

    def test_prefers_an_explicit_score_column_named_score(self, tmp_path):
        g = gate_content_identical(self._pairs(tmp_path, delta=0.2, score_col="score"))
        assert g.passed is True
