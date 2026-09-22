"""The 2x2 floor x ceiling probe: ledger R27.62 (the 40 Hz floor) and R27.64 (the null ceiling).

Both factors ride in one extraction, so both are pinned here before any number from
either is quoted. The measured values in the docstrings come from a local run at the
FakeMusicCaps geometry (``bin_hz = 16000/16384 = 0.9765625``); the asserts are on
*orderings and identities*, not on those figures, so a change in numpy's FFT cannot
turn a real regression into a green suite.

Written from two directions on purpose. Several tests check that the probe REPRODUCES
the published behaviour when its knobs are at their published values -- that is the
property that would actually break silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.comb_artifacts import (
    CALIBRATED_NULL_KEY,
    CALIBRATED_REAL_KEY,
    DECODER_FUNDAMENTALS_HZ,
    LO20_HZ,
    WIDE_HI_HZ,
    comb_calibrated_score,
    comb_features,
    comb_prior_harmonic_null,
    comb_prior_max,
)

BIN_HZ = 16000 / 16384
SR = 16000


def _residual(delta_hz: float | None = None, *, loud_k: int = 1, n: int = 8000, seed: int = 0) -> np.ndarray:
    """A peak residual with an optional comb, at the real bin resolution."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    if delta_hz:
        step = delta_hz / BIN_HZ
        k = 1
        while True:
            b = int(round(k * step))
            if b >= n - 1:
                break
            x[b] += 6.0 * np.exp(-abs(k - loud_k) / 3.0)
            k += 1
    return x


def _tone_audio(freqs_hz, seconds: float = 9.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    a = rng.normal(0.0, 0.05, t.size)
    for f in freqs_hz:
        a += 0.004 * np.sin(2 * np.pi * f * t)
    return a


# --------------------------------------------------------------------------- constants


def test_probe_constants_are_the_values_the_ledger_quotes():
    assert LO20_HZ == 20.0, "R27.62 measured the lowered floor at 20 Hz"
    assert WIDE_HI_HZ == 1000.0, "R27.64 widened the ceiling to n_harm * 1000 Hz"


def test_every_probe_arm_has_both_a_real_and_a_null_column():
    for kind in ("lo20margin", "widemargin", "lo20widemargin"):
        assert kind in CALIBRATED_NULL_KEY, f"{kind} has no null column"
        assert kind in CALIBRATED_REAL_KEY, f"{kind} has no real column"


def test_the_lowered_floor_reaches_the_prior_as_well_as_the_null():
    """The floor must move on BOTH sides or the comparison is rigged.

    A lowered floor that only widened the null would make the null strictly larger
    and the margin strictly smaller -- a guaranteed loss dressed up as a test.
    """
    assert CALIBRATED_REAL_KEY["lo20margin"] == "lo20_strength"
    assert CALIBRATED_REAL_KEY["lo20widemargin"] == "lo20_strength"
    # `wide` moves only the ceiling, which lives in the null alone.
    assert CALIBRATED_REAL_KEY["widemargin"] == "strength"


# --------------------------------------------------------------------------- the floor


def test_the_published_floor_rejects_stable_audios_fundamental_at_every_m():
    """21.53 Hz -> lag 22, below the 40 Hz floor (lag 41). This is R27.62's premise."""
    lag = int(round(21.53 / BIN_HZ))
    assert lag == 22
    assert lag < int(round(40.0 / BIN_HZ)) == 41
    for m in (1, 2, 4, 8):
        hz = comb_prior_max(_residual(21.53, seed=1), bin_hz=BIN_HZ, n_harm=m)["hz"]
        assert hz > 40.0, f"M={m}: the 40 Hz floor should exclude the 21.53 Hz fundamental"


def test_lowering_the_floor_admits_exactly_one_more_prior_lag():
    r = _residual(seed=2)
    for m in (2, 4):
        n40 = len({k for f in DECODER_FUNDAMENTALS_HZ for k in [int(round(mm * f / BIN_HZ)) for mm in range(1, m + 1)] if k >= 41})
        n20 = len({k for f in DECODER_FUNDAMENTALS_HZ for k in [int(round(mm * f / BIN_HZ)) for mm in range(1, m + 1)] if k >= int(round(LO20_HZ / BIN_HZ))})
        assert n20 == n40 + 1, f"M={m}: only lag 22 should be admitted, got {n20 - n40} extra"
    assert np.isfinite(comb_prior_max(r, bin_hz=BIN_HZ, n_harm=4, min_spacing_hz=LO20_HZ)["strength"])


def test_lowering_the_floor_also_enlarges_the_null():
    """If it did not, the floor would be reaching the prior only -- the rigged case."""
    r = _residual(seed=3)
    narrow = comb_prior_harmonic_null(r, bin_hz=BIN_HZ, n_harm=4)["n_lags"]
    lowered = comb_prior_harmonic_null(r, bin_hz=BIN_HZ, n_harm=4, min_spacing_hz=LO20_HZ)["n_lags"]
    assert lowered > narrow, f"null did not grow: {narrow} -> {lowered}"


# ------------------------------------------------------------------------- the ceiling


def test_widening_the_ceiling_enlarges_the_null_by_roughly_the_range_ratio():
    """441 -> 3130 lags at M=4 on this geometry (R27.61's measured table)."""
    r = _residual(seed=4)
    narrow = comb_prior_harmonic_null(r, bin_hz=BIN_HZ, n_harm=4)["n_lags"]
    wide = comb_prior_harmonic_null(r, bin_hz=BIN_HZ, n_harm=4, hi_hz=WIDE_HI_HZ)["n_lags"]
    assert narrow == 441, f"the published null is 441 lags at M=4, got {narrow}"
    assert wide > 6 * narrow, f"a 1000/150 ceiling should be several times larger, got {wide}"


def test_the_wide_null_still_protects_every_harmonic():
    """Widening must not readmit a tooth. If it did, it would be the lag-axis complement
    all over again -- the failure that scores AUC 0.0000 (handover section 2)."""
    r = _residual(seed=5)
    n = len(r)
    lo, hi = 41, min(n - 1, int(round(4 * WIDE_HI_HZ / BIN_HZ)))
    keep = np.ones(hi - lo + 1, dtype=bool)
    for f in DECODER_FUNDAMENTALS_HZ:
        k = 1
        while True:
            lag = int(round(k * f / BIN_HZ))
            if lag - 1 > hi:
                break
            a, b = max(lag - 1, lo), min(lag + 1, hi)
            if b >= a:
                keep[a - lo : b - lo + 1] = False
            k += 1
    survivors = np.flatnonzero(keep) + lo
    for f in DECODER_FUNDAMENTALS_HZ:
        for k in range(1, int(hi * BIN_HZ / f) + 1):
            lag = int(round(k * f / BIN_HZ))
            if lo <= lag <= hi:
                assert lag not in survivors, f"{k}*{f} Hz = lag {lag} leaked into the wide null"


# ----------------------------------------------------------- the published path is intact


def test_the_probe_does_not_move_the_published_columns():
    """Every pre-existing key must keep the value it had. This is the one that matters."""
    audio = _tone_audio([1000 + k * 100 for k in range(1, 60)], seed=6)
    kw = dict(
        residual_op="median", average="db", f_min=1000.0, smooth_bins=5,
        span_duration=0.0, n_harm_grid=(4,), surrogates=0, null_priors=0,
    )
    feats = comb_features(audio, SR, **kw)
    # The published harmonic margin must be formed from the published columns only.
    assert feats["comb_priormax4_hnlags"] == 441
    direct = feats["comb_priormax4_strength"] - feats["comb_priormax4_hmax_strength"]
    assert comb_calibrated_score(feats, "comb_priormax4_hmargin") == pytest.approx(direct, abs=0.0)


def test_each_probe_arm_resolves_to_its_own_two_columns():
    audio = _tone_audio([1000 + k * 100 for k in range(1, 60)], seed=7)
    feats = comb_features(
        audio, SR, residual_op="median", average="db", f_min=1000.0, smooth_bins=5,
        span_duration=0.0, n_harm_grid=(4,), surrogates=0, null_priors=0,
    )
    for kind in ("hmargin", "lo20margin", "widemargin", "lo20widemargin"):
        real = feats[f"comb_priormax4_{CALIBRATED_REAL_KEY[kind]}"]
        null = feats[f"comb_priormax4_{CALIBRATED_NULL_KEY[kind]}"]
        got = comb_calibrated_score(feats, f"comb_priormax4_{kind}")
        assert got == pytest.approx(real - null, abs=0.0), f"{kind} is not real - null"


def test_null_sizes_are_ordered_narrow_lo20_wide_lo20wide():
    audio = _tone_audio([1000 + k * 100 for k in range(1, 60)], seed=8)
    feats = comb_features(
        audio, SR, residual_op="median", average="db", f_min=1000.0, smooth_bins=5,
        span_duration=0.0, n_harm_grid=(4,), surrogates=0, null_priors=0,
    )
    n = {t: feats[f"comb_priormax4_{t}hnlags"] for t in ("", "lo20_", "wide_", "lo20wide_")}
    assert n[""] < n["lo20_"] < n["wide_"] < n["lo20wide_"], n


def test_a_missing_probe_column_raises_rather_than_scoring_zero():
    """A CSV extracted before the probe existed must fail loudly, not silently
    degrade the margin into the uncalibrated score."""
    feats = {"comb_priormax4_strength": 0.5}
    for kind in ("lo20margin", "widemargin", "lo20widemargin"):
        with pytest.raises(KeyError):
            comb_calibrated_score(feats, f"comb_priormax4_{kind}")


def test_unknown_probe_like_name_still_raises():
    with pytest.raises(KeyError):
        comb_calibrated_score({"comb_priormax4_strength": 0.5}, "comb_priormax4_lo30margin")


# ------------------------------------------------- the decoupled arms (ledger R27.68)


def _probe_frame():
    """Two tracks through the real feature path, as a one-row-per-track frame."""
    import pandas as pd

    rows = []
    for i in range(4):
        audio = _tone_audio([1000 + k * 100 for k in range(1, 60)] if i % 2 == 0 else [1234.0], seed=40 + i)
        feats = comb_features(
            audio, SR, residual_op="median", average="db", f_min=1000.0, smooth_bins=5,
            span_duration=0.0, n_harm_grid=(2, 4), surrogates=0, null_priors=0,
        )
        # derive_analytic_null.py refuses a frame with no label column, by design.
        feats.update(track_id=f"t{i}", label="fake" if i % 2 == 0 else "real",
                     algorithm="synth" if i % 2 == 0 else "")
        rows.append(feats)
    return pd.DataFrame(rows)


def _derive(df, tmp_path):
    import subprocess, sys, pandas as pd

    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    df.to_csv(src, index=False)
    r = subprocess.run(
        [sys.executable, "scripts/derive_analytic_null.py", "--score-csv", str(src), "--out-csv", str(dst)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return pd.read_csv(dst)


def test_decoupled_arm_equals_the_coupled_one_at_the_widest_m(tmp_path):
    """At the M that owns the shared null, decoupling must be a no-op.

    If this ever fails, the cross is reading the wrong null column and every
    decoupled number is against a different baseline than it claims.
    """
    out = _derive(_probe_frame(), tmp_path)
    for a, b in (("xwidemargin", "widemargin"), ("xlo20widemargin", "lo20widemargin")):
        np.testing.assert_array_equal(
            out[f"comb_priormax4_{a}"].to_numpy(),
            out[f"comb_priormax4_{b}"].to_numpy(),
            err_msg=f"comb_priormax4_{a} should be identical to comb_priormax4_{b}",
        )


def test_decoupled_arm_differs_at_the_shallower_m(tmp_path):
    """At M=2 the coupled null stops at 2*1000 Hz and the shared one at 4*1000.

    A zero difference here would mean the decoupling silently did nothing.
    """
    out = _derive(_probe_frame(), tmp_path)
    d = (out["comb_priormax2_xwidemargin"] - out["comb_priormax2_widemargin"]).abs().max()
    assert d > 0.0, "the m=2 cross is not using a wider null than the coupled arm"


def test_decoupled_arm_subtracts_a_null_shared_across_prior_depths(tmp_path):
    """The whole point: the same null for every prior depth, so depth reads alone."""
    out = _derive(_probe_frame(), tmp_path)
    shared = out["comb_priormax4_strength"] - out["comb_priormax4_xwidemargin"]
    for m in (2, 4):
        implied = out[f"comb_priormax{m}_strength"] - out[f"comb_priormax{m}_xwidemargin"]
        np.testing.assert_allclose(
            implied.to_numpy(), shared.to_numpy(), rtol=0, atol=0,
            err_msg=f"m={m} is not subtracting the same null as m=4",
        )


def test_lowered_floor_cross_uses_the_lowered_prior(tmp_path):
    """`xlo20wide` must read `_lo20_strength`, not `_strength` -- otherwise the floor
    reaches the null only, which enlarges it and shrinks every margin."""
    out = _derive(_probe_frame(), tmp_path)
    implied_null = out["comb_priormax2_lo20_strength"] - out["comb_priormax2_xlo20widemargin"]
    direct_null = out["comb_priormax4_lo20wide_hmax_strength"]
    np.testing.assert_allclose(implied_null.to_numpy(), direct_null.to_numpy(), rtol=0, atol=1e-12)


def test_a_csv_without_the_probe_columns_still_derives(tmp_path):
    """Pass-through for the older extractions. Family G must simply not fire."""
    base = _probe_frame()
    df = base.drop(columns=[c for c in base.columns if "wide" in c or "lo20" in c])
    out = _derive(df, tmp_path)
    assert not [c for c in out.columns if "xwide" in c or "xlo20" in c]
    assert "comb_priormax4_hmargin" in out.columns, "the published margin must still be derived"
