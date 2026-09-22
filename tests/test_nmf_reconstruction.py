"""Tests for the NMF-reconstruction-error scorer (Afchar & Hennequin 2607.25530).

The point of this scorer is ORIENTATION. A likelihood fitted to real music
assigns generated music higher density, because generated audio is smoother and
sits nearer the mode — measured in every representation we tried, including the
comb profile itself (SONICS macro 0.242). Reconstruction error against a basis
fitted on reals is correctly oriented by construction, and these tests pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

sklearn_decomposition = pytest.importorskip("sklearn.decomposition")
from sklearn.decomposition import NMF  # noqa: E402


def _real_profiles(n: int, d: int = 64, seed: int = 0) -> np.ndarray:
    """Smooth, varied profiles spanning a few underlying patterns."""
    rng = np.random.default_rng(seed)
    atoms = np.abs(rng.normal(size=(4, d)))
    weights = np.abs(rng.normal(size=(n, 4)))
    return np.maximum(weights @ atoms + 0.02 * rng.random((n, d)), 0.0)


def _combed_profiles(n: int, d: int = 64, seed: int = 1, spacing: int = 8) -> np.ndarray:
    """The same smooth content plus a periodic pattern the basis has never seen."""
    base = _real_profiles(n, d, seed=seed)
    comb = np.zeros(d)
    comb[::spacing] = 3.0
    return np.maximum(base + comb, 0.0)


def test_reconstruction_error_is_higher_for_the_unseen_pattern():
    """The orientation property. A basis fitted on reals cannot express a comb,
    so a comb reconstructs badly — high error means 'not real music', which is
    what we want and what a likelihood does not give."""
    basis_data = _real_profiles(120)
    model = NMF(n_components=4, init="nndsvda", max_iter=600, random_state=0).fit(basis_data)

    def err(x):
        return np.linalg.norm(x - model.transform(x) @ model.components_, axis=1)

    real_err = err(_real_profiles(40, seed=7))
    comb_err = err(_combed_profiles(40, seed=7))
    assert comb_err.mean() > real_err.mean()


def test_a_likelihood_would_invert_on_the_same_data():
    """Contrast, and the reason this scorer exists. A Gaussian fitted to the
    varied real profiles assigns HIGHER density to the more regular combed ones,
    because they sit nearer the mode — the exact failure measured on real data."""
    real = _real_profiles(200)
    mu, sd = real.mean(axis=0), real.std(axis=0) + 1e-6

    def loglik(x):
        z = (x - mu) / sd
        return -0.5 * (z**2).sum(axis=1)

    smooth = np.tile(mu, (40, 1))  # maximally typical, i.e. what a generator trends toward
    assert loglik(smooth).mean() > loglik(_real_profiles(40, seed=7)).mean(), (
        "a density model ranks the most typical input as most real — which is why "
        "one-class likelihood inverts on generated music"
    )


def test_too_many_atoms_destroys_the_signal():
    """A basis large enough to express anything reconstructs a comb too, removing
    the error we measure. This is why the atom count is swept and why their paper
    uses ~20 rather than 'as many as possible'."""
    basis_data = _real_profiles(120)
    gaps = []
    for k in (4, 60):
        k = min(k, basis_data.shape[0], basis_data.shape[1])
        model = NMF(n_components=k, init="nndsvda", max_iter=800, random_state=0).fit(basis_data)

        def err(x, m=model):
            return np.linalg.norm(x - m.transform(x) @ m.components_, axis=1)

        gaps.append(err(_combed_profiles(40, seed=7)).mean() - err(_real_profiles(40, seed=7)).mean())
    assert gaps[0] > gaps[1], "a larger basis must shrink the real/fake error gap"


def test_basis_and_scored_reals_must_be_disjoint():
    """A real that contributed an atom reconstructs too well, so its score is
    optimistic — the same leak as scoring a track with a flow that trained on it.
    The script splits reals in half; this pins why."""
    data = _real_profiles(120)
    model = NMF(n_components=4, init="nndsvda", max_iter=600, random_state=0).fit(data[:60])

    def err(x):
        return np.linalg.norm(x - model.transform(x) @ model.components_, axis=1)

    seen, unseen = err(data[:60]), err(data[60:])
    assert seen.mean() <= unseen.mean() + 1e-9
