"""Arm F2's correctness condition: the decomposition must sum back.

    log p(x) = log p_Z(f(x)) + log|det J_f(x)|

If the two terms do not reconstruct the checkpoint's own log-likelihood to
floating point, then whatever they are, they are not the parts of the score the
project has been reporting — and any conclusion drawn from scoring them
separately would be about a different quantity.

The prior-shifted case is tested explicitly because `prior_mean` is a scalar
offset applied to every latent dimension (mirroring MusicDET's mu_real), and
assuming it is zero was the first thing this script got wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score_flow_terms import SUM_TOLERANCE, decompose  # noqa: E402

from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass  # noqa: E402


def _fit(dim=8, n=400, prior_mean=0.0, seed=0, epochs=25):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, dim) @ np.diag(np.linspace(0.5, 2.0, dim))
    flow = RealNVPOneClass(
        RealNVPConfig(
            n_coupling_layers=2,
            hidden_dim=16,
            n_epochs=epochs,
            device="cpu",
            prior_mean=prior_mean,
        )
    )
    flow.fit(x)
    return flow, rng.randn(64, dim)


class TestDecompositionSumsBack:
    def test_zero_prior_mean(self):
        flow, x = _fit()
        terms = decompose(flow, x)
        ref = np.asarray(flow.log_likelihood(x), dtype=np.float64)
        assert np.max(np.abs(terms["log_prob"] - ref)) < SUM_TOLERANCE

    @pytest.mark.parametrize("prior_mean", [0.5, -1.0, 2.0])
    def test_shifted_prior_mean(self, prior_mean):
        """Reading prior_mean from the config rather than assuming 0."""
        flow, x = _fit(prior_mean=prior_mean)
        terms = decompose(flow, x)
        ref = np.asarray(flow.log_likelihood(x), dtype=np.float64)
        err = float(np.max(np.abs(terms["log_prob"] - ref)))
        assert err < SUM_TOLERANCE, f"prior_mean={prior_mean} gave reconstruction error {err:.3e}"

    def test_terms_are_not_degenerate(self):
        """A decomposition where one term is constant carries no information."""
        flow, x = _fit()
        terms = decompose(flow, x)
        assert terms["log_pz"].std() > 1e-6
        assert terms["log_det_jac"].std() > 1e-9

    def test_returns_one_value_per_row(self):
        flow, x = _fit()
        for v in decompose(flow, x).values():
            assert v.shape == (len(x),)


class TestOneDimensionalFallback:
    """The degenerate path (<2-d) uses a diagonal affine map instead of a flow.

    It is still an exact bijection, so the same identity must hold — and this is
    the path a badly-configured run silently falls into.
    """

    def test_sums_back(self):
        rng = np.random.RandomState(0)
        x = rng.randn(300, 1) * 2.0 + 1.0
        flow = RealNVPOneClass(RealNVPConfig(n_coupling_layers=2, hidden_dim=8, n_epochs=5, device="cpu"))
        flow.fit(x)
        xs = rng.randn(32, 1)
        terms = decompose(flow, xs)
        ref = np.asarray(flow.log_likelihood(xs), dtype=np.float64)
        assert np.max(np.abs(terms["log_prob"] - ref)) < SUM_TOLERANCE
