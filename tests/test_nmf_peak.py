"""Orientation of the NMF peak-energy score, pinned.

This is the test whose absence let a misimplementation be written up as the
finding "NMF reconstruction error inverts". The claim it pins is the one the
paper's formula makes: r_i = ‖H_iW − H_i(W∗G)‖₂ measures how much PEAK STRUCTURE
sample i's reconstruction carries, so a comb-like profile must score HIGHER than
a smooth one.

If `test_comb_profiles_score_higher_than_smooth_ones` fails, no number produced
by scripts/eval_nmf_peak_detector.py may be reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_nmf_peak_detector import peak_energy_scores  # noqa: E402

D = 445  # their published fakeprint dimension
N_PER_CLASS = 60


def _smooth_profiles(n=N_PER_CLASS, seed=0):
    """Real-music-like: irregular micro-texture, no periodic teeth."""
    rng = np.random.RandomState(seed)
    from scipy.ndimage import gaussian_filter1d

    x = rng.rand(n, D)
    return np.clip(gaussian_filter1d(x, sigma=6.0, axis=1), 0.0, None)


def _comb_profiles(n=N_PER_CLASS, spacing=17, seed=1):
    """Decoder-like: a clean periodic comb on a faint floor."""
    rng = np.random.RandomState(seed)
    x = rng.rand(n, D) * 0.05
    x[:, ::spacing] += 1.0
    return x


class TestOrientation:
    def test_comb_profiles_score_higher_than_smooth_ones(self):
        X = np.vstack([_smooth_profiles(), _comb_profiles()])
        r = peak_energy_scores(X, n_atoms=20, blur_sigma=3.0)
        smooth, comb = r[:N_PER_CLASS], r[N_PER_CLASS:]
        assert comb.mean() > smooth.mean(), (
            f"comb {comb.mean():.4f} must exceed smooth {smooth.mean():.4f} — "
            "the score measures peak energy, not novelty"
        )

    def test_separation_is_decisive_not_marginal(self):
        X = np.vstack([_smooth_profiles(), _comb_profiles()])
        r = peak_energy_scores(X, n_atoms=20, blur_sigma=3.0)
        y = np.r_[np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)]
        from sklearn.metrics import roc_auc_score

        assert roc_auc_score(y, r) > 0.9

    @pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0, 5.0, 8.0])
    def test_orientation_holds_across_the_swept_blur_widths(self, sigma):
        X = np.vstack([_smooth_profiles(), _comb_profiles()])
        r = peak_energy_scores(X, n_atoms=20, blur_sigma=sigma)
        assert r[N_PER_CLASS:].mean() > r[:N_PER_CLASS].mean()


def _legacy_recon_error(reals: np.ndarray, scored: np.ndarray, n_atoms: int = 20) -> np.ndarray:
    """The RETRACTED quantity: ‖x − H W‖ with W fitted on reals only."""
    import warnings

    from sklearn.decomposition import NMF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmf = NMF(n_components=n_atoms, init="nndsvda", random_state=0, max_iter=600).fit(reals)
    return np.linalg.norm(scored - nmf.transform(scored) @ nmf.components_, axis=1)


def _symmetric_profiles(n, regular, seed):
    """Max-normalised residual profiles, as the real pipeline produces them.

    ``regular=True`` gives a decoder-like comb (evenly spaced teeth);
    ``regular=False`` gives real-music-like irregular micro-texture.

    The fixture is deliberately **symmetric**: both classes get the same number
    of peaks drawn from the same amplitude distribution and are max-normalised to
    a unit peak, so the ONLY difference is whether the peak positions are
    periodic. An earlier version of this fixture gave the two classes different
    peak amplitudes; an L2 score then separated them on magnitude and produced
    the opposite conclusion about the retracted quantity. That near-miss is the
    reason this docstring exists.
    """
    rng = np.random.RandomState(seed)
    x = rng.rand(n, D) ** 3 * 0.3
    for i in range(n):
        if regular:
            spacing = 17
            idx = np.arange(rng.randint(0, spacing), D, spacing)
        else:
            idx = rng.choice(D, size=rng.randint(20, 40), replace=False)
        x[i, idx] += 0.8 + 0.2 * rng.rand(len(idx))
    x = np.clip(x, 0.0, None)
    return x / (1e-6 + x.max(axis=1, keepdims=True))


class TestTheRetractedQuantityInvertsByConstruction:
    """§0.1's first defect, measured rather than argued.

    A k-atom basis reconstructs a clean periodic comb WELL and irregular
    micro-texture BADLY, so ‖x − HW‖ gives the comb the LOWER score. Verified
    across 8 seeds and a capacity grid; the inversion is not a capacity artifact.
    """

    @pytest.mark.parametrize("seed", range(4))
    def test_inverts_across_seeds(self, seed):
        from sklearn.metrics import roc_auc_score

        R = _symmetric_profiles(120, regular=False, seed=seed)
        F = _symmetric_profiles(60, regular=True, seed=seed + 100)
        fit, held = R[:60], R[60:]
        scores = np.r_[_legacy_recon_error(fit, held), _legacy_recon_error(fit, F)]
        y = np.r_[np.zeros(len(held)), np.ones(len(F))]
        auc = roc_auc_score(y, scores)
        assert auc < 0.5, f"retracted quantity should invert; got AUC {auc:.4f}"

    @pytest.mark.parametrize("n_atoms", [5, 10, 20, 40])
    def test_inversion_is_not_a_capacity_artifact(self, n_atoms):
        from sklearn.metrics import roc_auc_score

        R = _symmetric_profiles(120, regular=False, seed=0)
        F = _symmetric_profiles(60, regular=True, seed=100)
        fit, held = R[:60], R[60:]
        scores = np.r_[_legacy_recon_error(fit, held, n_atoms), _legacy_recon_error(fit, F, n_atoms)]
        y = np.r_[np.zeros(len(held)), np.ones(len(F))]
        assert roc_auc_score(y, scores) < 0.5

    @pytest.mark.parametrize("seed", range(4))
    def test_the_papers_quantity_is_correctly_oriented_on_the_same_fixture(self, seed):
        """The contrast that makes the retraction meaningful: same data, right answer."""
        from sklearn.metrics import roc_auc_score

        R = _symmetric_profiles(120, regular=False, seed=seed)[60:]
        F = _symmetric_profiles(60, regular=True, seed=seed + 100)
        r = peak_energy_scores(np.vstack([R, F]), n_atoms=20, blur_sigma=3.0)
        y = np.r_[np.zeros(len(R)), np.ones(len(F))]
        assert roc_auc_score(y, r) > 0.9


class TestResamplingTo445DestroysTheSignal:
    """The second, measured defect: the old arm's 445-bin descriptor.

    Real teeth are one to three bins wide at ~1 Hz resolution. Resampling ~7,000
    bins to 445 averages each output bin over ~16 input bins. This is what makes
    the old NMF row uninterpretable independently of the scoring rule.
    """

    @staticmethod
    def _hi_res(regular, n=40, n_bins=7168, tooth=205, seed=0):
        from intrinsic_ai_music_detection.features.comb_artifacts import afchar_hull_curve

        rng = np.random.RandomState(seed)
        freqs = np.linspace(1000.0, 8000.0, n_bins)
        out = []
        for _ in range(n):
            spec = -45 + 8 * np.exp(-np.arange(n_bins) / 2500) + rng.randn(n_bins) * 0.8
            if regular:
                for k in range(rng.randint(0, tooth), n_bins, tooth):
                    spec[k : k + 2] += 4.0  # narrow decoder teeth
            else:
                for p in rng.choice(n_bins, size=30, replace=False):
                    spec[max(0, p - 6) : p + 6] += 3.0  # broad musical partials
            r = np.clip(np.clip(spec - afchar_hull_curve(freqs, spec, area=10), 0, None), 0, 5)
            out.append(r / (1e-6 + r.max()))
        return np.array(out)

    def test_full_resolution_beats_445_bins(self):
        from sklearn.metrics import roc_auc_score

        R = self._hi_res(regular=False, seed=1)
        F = self._hi_res(regular=True, seed=2)
        X_full = np.vstack([R, F])
        y = np.r_[np.zeros(len(R)), np.ones(len(F))]

        grid = np.linspace(0, X_full.shape[1] - 1, 445)
        X_445 = np.stack([np.interp(grid, np.arange(X_full.shape[1]), row) for row in X_full], axis=0)

        auc_full = roc_auc_score(y, peak_energy_scores(X_full, n_atoms=20, blur_sigma=3.0))
        auc_445 = roc_auc_score(y, peak_energy_scores(X_445, n_atoms=20, blur_sigma=3.0))
        assert auc_full > auc_445 + 0.1, (
            f"full resolution {auc_full:.4f} must clearly beat 445 bins {auc_445:.4f}; "
            "this is the measured basis for the second half of the §0.1 retraction"
        )


class TestGuards:
    def test_too_few_samples_is_a_hard_error(self):
        with pytest.raises(ValueError, match="at least 20 pooled samples"):
            peak_energy_scores(np.random.rand(5, D), n_atoms=20)

    def test_returns_one_score_per_row(self):
        X = np.vstack([_smooth_profiles(), _comb_profiles()])
        assert peak_energy_scores(X, n_atoms=20, blur_sigma=3.0).shape == (len(X),)


class TestDictionarySourceDeterminesOrientation:
    """The zero-shot claim's exact boundary, measured.

    `r_i = ||H_iW - H_i(W*G)||` reads out how much of a sample's reconstruction
    is peak structure. That only works if the DICTIONARY contains comb-like atoms
    — and a dictionary learned from real music alone does not. Then H projects a
    comb onto smooth atoms, the reconstruction has little peak structure, and the
    score INVERTS.

    So the method is **label-free but not reals-only**: no fake label is ever
    used, but unlabelled audio containing some synthetic content must be present
    when the dictionary is learned. The threshold is reals-only. Saying "trained
    on reals only" would be false, and this test exists so nobody says it.
    """

    @staticmethod
    def _corpus(seed):
        R = _symmetric_profiles(120, regular=False, seed=seed)
        F = _symmetric_profiles(120, regular=True, seed=seed + 50)
        return R, F, np.vstack([R, F]), np.r_[np.zeros(len(R)), np.ones(len(F))]

    def test_pooled_dictionary_is_correctly_oriented(self):
        from sklearn.metrics import roc_auc_score

        _, _, X, y = self._corpus(0)
        assert roc_auc_score(y, peak_energy_scores(X, n_atoms=20, blur_sigma=3.0)) > 0.9

    def test_reals_only_dictionary_INVERTS(self):
        """The boundary. If this ever stops inverting, the claim can be strengthened —
        but it must be re-measured on real corpora before anyone says 'reals only'."""
        from sklearn.metrics import roc_auc_score

        R, _, X, y = self._corpus(0)
        auc = roc_auc_score(y, peak_energy_scores(X, n_atoms=20, blur_sigma=3.0, fit_matrix=R))
        assert auc < 0.5, (
            f"a reals-only dictionary should invert (got {auc:.4f}); if it no longer does, "
            "re-check the mechanism before widening the claim"
        )

    def test_external_pooled_dictionary_transfers(self):
        """The claim we CAN make: the dictionary need not come from the corpus being
        scored, so the target generator is never required."""
        from sklearn.metrics import roc_auc_score

        _, _, X, y = self._corpus(0)
        _, _, X_ext, _ = self._corpus(7)
        auc = roc_auc_score(y, peak_energy_scores(X, n_atoms=20, blur_sigma=3.0, fit_matrix=X_ext))
        assert auc > 0.9, f"external dictionary must transfer; got {auc:.4f}"

    def test_dimension_mismatch_is_a_hard_error(self):
        _, _, X, _ = self._corpus(0)
        bad = np.random.RandomState(0).rand(60, D // 2)
        with pytest.raises(ValueError, match="dimensions"):
            peak_energy_scores(X, n_atoms=20, fit_matrix=bad)

    @pytest.mark.parametrize("learner", ["nmf", "sparse", "kmeans"])
    def test_the_boundary_is_a_property_of_DICTIONARY_LEARNING_not_of_NMF(self, learner):
        """The paper's central claim, generalised past one implementation.

        A reviewer is entitled to ask whether the reals-only inversion is a quirk
        of ``sklearn.NMF``'s Frobenius objective. It is not: L1-penalised
        non-negative sparse coding and plain centroid atoms (K-SVD's clustering
        step) both show the same pooled-vs-reals-only reversal. Only the way W and
        H are obtained changes; the statistic is identical in all three.

        If a learner ever stops inverting here, the claim narrows to the learners
        that do — it does not silently widen.
        """
        from sklearn.metrics import roc_auc_score

        R, _, X, y = self._corpus(0)
        pooled = roc_auc_score(y, peak_energy_scores(X, n_atoms=20, blur_sigma=3.0, dictionary=learner))
        reals_only = roc_auc_score(
            y, peak_energy_scores(X, n_atoms=20, blur_sigma=3.0, dictionary=learner, fit_matrix=R)
        )
        assert pooled > 0.9, f"{learner}: pooled dictionary should work, got {pooled:.4f}"
        assert reals_only < 0.5, f"{learner}: reals-only dictionary should invert, got {reals_only:.4f}"

    def test_default_learner_is_nmf_so_published_numbers_are_unchanged(self):
        """The default path must stay bit-identical — published columns depend on it."""
        _, _, X, _ = self._corpus(0)
        explicit = peak_energy_scores(X, n_atoms=20, blur_sigma=3.0, dictionary="nmf")
        implicit = peak_energy_scores(X, n_atoms=20, blur_sigma=3.0)
        np.testing.assert_array_equal(explicit, implicit)

    def test_unknown_learner_is_a_hard_error(self):
        _, _, X, _ = self._corpus(0)
        with pytest.raises(ValueError, match="unknown dictionary learner"):
            peak_energy_scores(X, n_atoms=20, dictionary="pca")


class TestPeakinessIsNotEvidenceOfACombButPeriodicityIs:
    """The Figure-2 statistic, and why it had to change.

    `visualize_nmf_space.py` ranked atoms by `peakiness = ||W - W*G|| / ||W||`,
    which is the SCORE'S OWN SENSITIVITY — how much blurring destroys the atom.
    The 2026-08-21 figure notes then read that number as evidence for "the
    dictionary learns decoder combs". It is not: a forest of RANDOM spikes is
    just as peaky as a comb with the same number of spikes.

    `comb_strength` — the autocorrelation statistic the detector already applies
    to a track's residual — separates them by an order of magnitude and recovers
    the true spacing. That is the mechanism test, and it is what the paper quotes.
    """

    D = 4779
    BIN_HZ = (8000.0 - 1000.0) / (D - 1)
    SPACING_BINS = 57

    @classmethod
    def _pair(cls, seed=0):
        rng = np.random.RandomState(seed)
        n_spikes = cls.D // cls.SPACING_BINS
        comb = np.full(cls.D, 0.05)
        comb[np.arange(20, cls.D, cls.SPACING_BINS)] += 1.0
        forest = np.full(cls.D, 0.05)
        forest[rng.choice(cls.D, size=n_spikes, replace=False)] += 1.0
        return comb, forest

    @staticmethod
    def _peakiness(w, sigma=5.0):
        from scipy.ndimage import gaussian_filter1d

        return float(np.linalg.norm(w - gaussian_filter1d(w, sigma=sigma)) / (np.linalg.norm(w) + 1e-9))

    def test_peakiness_cannot_tell_a_comb_from_a_random_spike_forest(self):
        comb, forest = self._pair()
        assert abs(self._peakiness(comb) - self._peakiness(forest)) < 0.05, (
            "if peakiness ever separates these two, this test's premise is wrong — "
            "but then the figure argument would need re-deriving, not assuming"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_periodicity_does_and_recovers_the_spacing(self, seed):
        from intrinsic_ai_music_detection.features.comb_artifacts import comb_strength

        comb, forest = self._pair(seed)
        p_comb, sp_comb, _ = comb_strength(comb, bin_hz=self.BIN_HZ)
        p_forest, _, _ = comb_strength(forest, bin_hz=self.BIN_HZ)
        assert p_comb > 0.8, f"a clean comb should be highly periodic, got {p_comb:.3f}"
        assert p_forest < 0.3, f"random spikes should not be, got {p_forest:.3f}"
        expected = self.SPACING_BINS * self.BIN_HZ
        assert (
            abs(sp_comb - expected) < 0.05 * expected
        ), f"spacing should be recovered: got {sp_comb:.1f} Hz, expected {expected:.1f} Hz"


class TestDictionaryLevelTrainOnTestLeak:
    """The fifth retraction, and the flag that closes it.

    ``r_i`` is read off a reconstruction through learned atoms. A track that was IN
    the fit set is reconstructed through atoms it helped create, so its ``r_i`` is
    inflated relative to a track that was not. When the fit reals are also the
    scored reals, real music scores systematically higher and the AUC **inverts**
    — for reasons that have nothing to do with AI music.

    Measured on FakeMusicCaps with a 600-real dictionary, everything else identical:
    the macro AUC reads **0.1679** when those 600 are 100% of the scored reals and
    **0.5690** when they are 11% of them. The published "reals-only inverts to
    0.167" was that leak.

    Same defect class as the ``score_flow_terms.py`` train-on-test found in R2.3,
    one level down — which is why it gets a test rather than a note.
    """

    D = 400
    SPACING = 13

    @classmethod
    def _profiles(cls, n, regular, rng):
        X = []
        for _ in range(n):
            r = rng.rand(cls.D) * 0.2
            idx = (
                np.arange(rng.randint(4, 9), cls.D, cls.SPACING)
                if regular
                else rng.choice(cls.D, size=len(np.arange(6, cls.D, cls.SPACING)), replace=False)
            )
            r[idx] += 1.0
            X.append(r / r.max())
        return np.array(X)

    def test_auc_degrades_monotonically_with_fit_set_overlap(self):
        """The leak is a dose-response in overlap, which is why subsampling the
        SCORED set changed a headline while nothing about the method changed."""
        from sklearn.metrics import roc_auc_score

        aucs = []
        for n_scored_real in (60, 200, 600, 2000):
            rng = np.random.RandomState(0)
            reals = self._profiles(n_scored_real, False, rng)
            fakes = self._profiles(600, True, rng)
            X = np.vstack([reals, fakes])
            y = np.r_[np.zeros(len(reals)), np.ones(len(fakes))]
            scores = peak_energy_scores(X, n_atoms=20, blur_sigma=2.0, fit_matrix=reals[:60])
            aucs.append(roc_auc_score(y, scores))

        assert aucs[0] < 0.1, f"100% overlap should invert hard, got {aucs[0]:.4f}"
        assert all(
            a < b for a, b in zip(aucs, aucs[1:])
        ), f"AUC must rise monotonically as overlap falls; got {[round(a, 4) for a in aucs]}"
        assert aucs[-1] - aucs[0] > 0.25, "the leak must account for a large share of the apparent inversion"

    def test_holding_the_fit_reals_out_removes_the_artefact(self):
        """The same dictionary, scored on reals it never saw. Whatever inversion
        survives here is the real effect; everything above it was the leak."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(0)
        fit_reals = self._profiles(60, False, rng)
        fakes = self._profiles(600, True, rng)

        leaked = roc_auc_score(
            np.r_[np.zeros(60), np.ones(600)],
            peak_energy_scores(np.vstack([fit_reals, fakes]), n_atoms=20, blur_sigma=2.0, fit_matrix=fit_reals),
        )
        held_out_reals = self._profiles(60, False, np.random.RandomState(1))
        held = roc_auc_score(
            np.r_[np.zeros(60), np.ones(600)],
            peak_energy_scores(np.vstack([held_out_reals, fakes]), n_atoms=20, blur_sigma=2.0, fit_matrix=fit_reals),
        )
        assert held - leaked > 0.25, (
            f"holding the fit reals out must move the number materially " f"(leaked {leaked:.4f}, held-out {held:.4f})"
        )
