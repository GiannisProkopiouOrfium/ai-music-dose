"""Regression tests for the 2026-08 methodology-audit fixes.

Covers, with small synthetic CPU-only data:
  - RealNVPOneClass save/load round-trip, incl. PCA serialization and the
    load(device=...) override (the old load() kept the pickled training device
    and crashed CPU-only consumers of cuda-trained checkpoints).
  - The sign convention: score_samples == -log_likelihood (anomaly-oriented).
  - Global-RNG isolation of fit() (fit used to reseed torch/numpy globally).
  - Canonical window pooling semantics (drop-trailing-partial, non-finite
    filtering, empty-on-short) + meanstd/frame modes.
  - Trajectory aggregators, quantile calibration, and mixture-min scoring.
  - _EarlyStopper multi-module restore (conditional flow + projector).
"""

from __future__ import annotations

import numpy as np
import pytest

from intrinsic_ai_music_detection.features.pooling import frame_features, pool_windows
from intrinsic_ai_music_detection.models.aggregate import (
    AGGREGATOR_NAMES,
    QuantileCalibrator,
    aggregate_all,
    aggregate_trajectory,
    mixture_min_score,
)
from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass


@pytest.fixture(scope="module")
def fitted_flow_with_pca():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    x_raw = rng.normal(size=(500, 12))
    pca = PCA(n_components=6, random_state=0).fit(x_raw)
    x_red = pca.transform(x_raw)
    cfg = RealNVPConfig(n_epochs=25, n_coupling_layers=4, device="cpu", seed=1)
    flow = RealNVPOneClass(cfg).fit(x_red).attach_pca(pca)
    return flow, pca, x_raw, x_red


def test_save_load_roundtrip_with_pca(tmp_path, fitted_flow_with_pca):
    flow, _pca, x_raw, x_red = fitted_flow_with_pca
    p = tmp_path / "flow.pt"
    flow.save(p)
    loaded = RealNVPOneClass.load(p, device="cpu")

    # device override must win over the pickled training device
    assert loaded.config.device == "cpu"

    s_ref = flow.score_samples(x_red[:40])
    # raw-dim input goes through the serialized PCA automatically
    np.testing.assert_allclose(loaded.score_samples(x_raw[:40]), s_ref, atol=1e-6)
    # already-reduced input passes through unchanged
    np.testing.assert_allclose(loaded.score_samples(x_red[:40]), s_ref, atol=1e-6)


def test_dim_mismatch_hard_fails(fitted_flow_with_pca):
    flow, _pca, _x_raw, _x_red = fitted_flow_with_pca
    with pytest.raises(ValueError, match="dim"):
        flow.score_samples(np.zeros((3, 5)))


def test_sign_convention_anomaly_oriented(fitted_flow_with_pca):
    flow, _pca, _x_raw, x_red = fitted_flow_with_pca
    ll = flow.log_likelihood(x_red[:10])
    np.testing.assert_allclose(flow.score_samples(x_red[:10]), -ll)


def test_fit_does_not_mutate_global_rng():
    import torch

    rng_state_np = np.random.get_state()[1][:10].copy()
    rng_state_torch = torch.random.get_rng_state().clone()
    x = np.random.default_rng(3).normal(size=(200, 6))
    RealNVPOneClass(RealNVPConfig(n_epochs=5, n_coupling_layers=2, device="cpu")).fit(x)
    assert (np.random.get_state()[1][:10] == rng_state_np).all()
    assert torch.equal(torch.random.get_rng_state(), rng_state_torch)


def test_pool_windows_canonical_semantics():
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(25, 4))
    mat[7] = np.nan  # a poisoned frame is dropped inside its windows
    out = pool_windows(mat, window_frames=10, hop_frames=5)
    # starts at 0, 5, 10, 15 (trailing partial dropped)
    assert out.shape == (4, 4)
    w0 = mat[0:10]
    w0 = w0[np.isfinite(w0).all(axis=1)]
    np.testing.assert_allclose(out[0], w0.mean(axis=0))
    # shorter than one window -> EMPTY, never a whole-track fallback
    assert pool_windows(mat[:5], 10, 5).shape == (0, 4)


def test_pool_windows_meanstd_and_frame_modes():
    rng = np.random.default_rng(2)
    mat = rng.normal(size=(30, 4))
    ms = pool_windows(mat, 10, 5, mode="meanstd")
    assert ms.shape[1] == 8
    np.testing.assert_allclose(ms[0][:4], mat[0:10].mean(axis=0))
    np.testing.assert_allclose(ms[0][4:], mat[0:10].std(axis=0))
    ff = frame_features(np.vstack([mat, np.full((3, 4), np.inf)]), stride=2)
    assert ff.shape == (15, 4)
    with pytest.raises(ValueError):
        pool_windows(mat, 10, 5, mode="bogus")


def test_aggregators():
    traj = np.array([1.0, 2.0, 3.0, 100.0, np.nan])
    r = aggregate_all(traj)
    assert set(r) == set(AGGREGATOR_NAMES)
    assert r["mean"] == pytest.approx(26.5)
    assert r["median"] == pytest.approx(2.5)
    assert r["max"] == 100.0
    assert r["top_10pct"] == 100.0
    assert r["trimmed_mean_10"] == pytest.approx(2.5)
    assert np.isnan(aggregate_trajectory([np.nan], "mean"))
    with pytest.raises(ValueError):
        aggregate_trajectory(traj, "bogus")


def test_quantile_calibrator_and_mixture():
    cal = QuantileCalibrator().fit(np.arange(100.0))
    assert cal.transform([-5])[0] == 0.0
    assert cal.transform([500])[0] == 1.0
    assert cal.transform([49.5])[0] == pytest.approx(0.5, abs=0.01)
    cal2 = QuantileCalibrator.from_array(cal.to_array())
    np.testing.assert_allclose(cal2.transform([10.0, 60.0]), cal.transform([10.0, 60.0]))

    mix = mixture_min_score({"a": [0.9, 0.2, np.nan], "b": [0.1, 0.8, 0.4]})
    np.testing.assert_allclose(mix, [0.1, 0.2, 0.4])
    with pytest.raises(ValueError):
        mixture_min_score({"a": [0.1, 0.2], "b": [0.3]})


def test_early_stopper_multi_module_restore():
    import torch

    from intrinsic_ai_music_detection.models.flow import _EarlyStopper

    m1, m2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    stopper = _EarlyStopper(patience=2)
    stopper.update(1.0, m1, m2)  # best epoch snapshot of BOTH modules
    best_w1 = m1.weight.detach().clone()
    best_w2 = m2.weight.detach().clone()
    with torch.no_grad():
        m1.weight += 1.0
        m2.weight += 1.0
    stopper.update(2.0, m1, m2)  # worse epoch — snapshot must not move
    stopper.restore(m1, m2)
    assert torch.equal(m1.weight, best_w1)
    assert torch.equal(m2.weight, best_w2)
    with pytest.raises(ValueError):
        stopper.restore(m1)  # module-count mismatch must fail loudly


def test_conditional_flow_trains_and_scores_finite():
    from intrinsic_ai_music_detection.models.flow import ConditionalRealNVPConfig, ConditionalRealNVPOneClass

    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 6))
    ctx = rng.normal(size=(300, 3))
    cfg = ConditionalRealNVPConfig(n_epochs=15, n_coupling_layers=2, device="cpu", patience=5)
    det = ConditionalRealNVPOneClass(cfg).fit(x, ctx)
    s = det.score_samples(x[:20], ctx[:20])
    assert np.isfinite(s).all()


# --- Spectrogram front end (Lever 3: representation-controlled comparison) -----
# Skips locally when librosa is absent; runs on EC2 / CI where it is installed.


def test_spectrogram_extractor_matches_encodec_rate_and_dim():
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import SpectrogramExtractor

    ex = SpectrogramExtractor()
    sr = 24_000
    for dur in (4.0, 10.0, 55.0):
        audio = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(sr * dur)) / sr)).astype(np.float32)
        feats = ex.extract(audio, sr)
        # 128-d to match EnCodec's latent width, ~75 fps to match its frame rate
        assert feats.shape[1] == 128
        assert abs(feats.shape[0] / dur - 75.0) < 1.0, (dur, feats.shape)
        assert np.isfinite(feats).all()
        assert feats.dtype == np.float32


def test_spectrogram_extractor_resamples_and_discriminates():
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import SpectrogramExtractor

    ex = SpectrogramExtractor()
    sr = 24_000
    t = np.arange(sr * 4) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = (tone + 0.2 * np.random.default_rng(0).normal(size=len(t))).astype(np.float32)
    # a noise floor must move the log-mel representation substantially
    assert abs(ex.extract(noisy, sr).mean() - ex.extract(tone, sr).mean()) > 0.5
    # non-native sample rates are resampled to the configured 24 kHz
    audio48 = (0.3 * np.sin(2 * np.pi * 440 * np.arange(48_000 * 2) / 48_000)).astype(np.float32)
    feats = ex.extract(audio48, 48_000)
    assert feats.shape[1] == 128 and abs(feats.shape[0] / 2.0 - 75.0) < 1.5


def test_spec_registered_in_extractor_factory():
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    for name in ("spec", "logmel"):
        assert get_extractor(name, device="cpu").embedding_dim == 128


# ---------------------------------------------------------------------------
# Frequency-banded flow + global prior (MusicDET's architecture axis)
# ---------------------------------------------------------------------------


def _banded_cfg():
    from intrinsic_ai_music_detection.models.flow import RealNVPConfig

    return RealNVPConfig(n_coupling_layers=2, hidden_dim=8, n_epochs=3, batch_size=32, seed=0)


def test_latents_reconstruct_log_likelihood_exactly():
    """log_likelihood(x) must equal log N(z; prior_mean, I) + logdet.

    This identity is what licenses swapping the per-band base density for a
    joint global prior while keeping the Jacobian term.
    """
    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    for mu in (0.0, 1.5):
        cfg = RealNVPConfig(n_coupling_layers=2, hidden_dim=8, n_epochs=3, batch_size=32, seed=0, prior_mean=mu)
        f = RealNVPOneClass(cfg).fit(x)
        z, logdet = f.latents(x)
        base = -0.5 * ((z - mu) ** 2 + np.log(2 * np.pi)).sum(axis=1)
        assert np.allclose(base + logdet, f.log_likelihood(x), atol=1e-4)


def test_banded_flow_sums_bands_and_is_anomaly_oriented():
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass, RealNVPOneClass

    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 8))
    b = BandedRealNVPOneClass(_banded_cfg(), n_bands=2).fit(x)
    assert b._slices == [(0, 4), (4, 8)]

    # summation identity against the two independently-refit band flows
    parts = np.zeros(len(x))
    for f, (lo, hi) in zip(b._flows, b._slices):
        parts += f.log_likelihood(x[:, lo:hi])
    assert np.allclose(parts, b.log_likelihood(x), atol=1e-8)
    # higher score == more anomalous
    assert np.allclose(b.score_samples(x), -b.log_likelihood(x))
    assert b.score_samples(rng.normal(loc=8.0, size=(50, 8))).mean() > b.score_samples(x).mean()
    assert isinstance(b._flows[0], RealNVPOneClass) and b._global is None


def test_global_prior_is_a_proper_density_and_differs_from_summation():
    """The global stage must change the score (it models cross-band dependence)."""
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass

    rng = np.random.default_rng(2)
    # strongly correlated bands: summation cannot see the coupling, the global can
    a = rng.normal(size=(400, 4))
    x = np.hstack([a, a * 0.9 + 0.1 * rng.normal(size=(400, 4))])

    summed = BandedRealNVPOneClass(_banded_cfg(), n_bands=2).fit(x)
    joint = BandedRealNVPOneClass(_banded_cfg(), n_bands=2, global_prior=True).fit(x)
    assert joint._global is not None

    ll_s, ll_j = summed.log_likelihood(x), joint.log_likelihood(x)
    assert np.isfinite(ll_j).all() and ll_j.shape == (len(x),)
    assert not np.allclose(ll_s, ll_j)
    # the global prior must not silently invert the anomaly orientation
    far = rng.normal(loc=10.0, size=(50, 8))
    assert joint.score_samples(far).mean() > joint.score_samples(x).mean()


def test_banded_roundtrip_preserves_scores_with_and_without_global_prior(tmp_path):
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass

    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 8))
    for gp in (False, True):
        b = BandedRealNVPOneClass(_banded_cfg(), n_bands=2, global_prior=gp).fit(x)
        b.embedding_name = "spec-musicdet"
        p = tmp_path / f"banded_gp{int(gp)}.pt"
        b.save(p)
        r = BandedRealNVPOneClass.load(p, device="cpu")
        assert r.global_prior is gp and (r._global is not None) is gp
        assert r.embedding_name == "spec-musicdet" and r.n_bands == 2
        assert np.allclose(b.log_likelihood(x), r.log_likelihood(x), atol=1e-6)


def test_banded_rejects_too_many_bands_and_wrong_dim():
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass

    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="use at most"):
        BandedRealNVPOneClass(_banded_cfg(), n_bands=8).fit(rng.normal(size=(50, 8)))
    b = BandedRealNVPOneClass(_banded_cfg(), n_bands=2).fit(rng.normal(size=(200, 8)))
    with pytest.raises(ValueError, match="!= fitted dim"):
        b.log_likelihood(rng.normal(size=(10, 6)))


def test_musicdet_spec_config_is_band_divisible():
    """257 one-sided bins is PRIME; MusicDET's own assert F % n_bands == 0 only
    holds because the (0, 8 kHz) crop lands on bin 256."""
    from intrinsic_ai_music_detection.config import MusicDETSpecConfig

    c = MusicDETSpecConfig()
    k_hi = int(round(c.fmax * c.n_fft / c.sample_rate))
    k_lo = int(round(c.fmin * c.n_fft / c.sample_rate))
    assert k_hi - k_lo == c.embedding_dim == 256
    assert c.sample_rate / c.hop_length == 100.0 and c.spec_type == "linear"
    assert all(c.embedding_dim % nb == 0 for nb in (2, 4, 8))


def test_musicdet_spec_extractor_is_linear_and_cropped():
    pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.features.embeddings import get_extractor

    ex = get_extractor("spec-musicdet", device="cpu")
    assert ex.embedding_dim == 256 and ex.sample_rate == 16_000
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16_000 * 4) / 16_000)).astype(np.float32)
    feats = ex.extract(audio, 16_000)
    assert feats.shape[1] == 256 and abs(feats.shape[0] / 4.0 - 100.0) < 1.5
    assert np.isfinite(feats).all()
    # the default 'spec' arm must be untouched by the linear-STFT addition
    assert get_extractor("spec", device="cpu").embedding_dim == 128


# ---------------------------------------------------------------------------
# Feature-masking augmentation (ported from MusicDET's SpecAugmentFT)
# ---------------------------------------------------------------------------


def test_augment_is_off_by_default_and_never_touches_scoring():
    """Default must be a no-op: every result before 2026-08-12 was unaugmented."""
    import torch

    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    f = RealNVPOneClass(RealNVPConfig())
    assert f.config.augment_p == 0.0
    b = torch.randn(8, 32)
    assert torch.equal(f._augment(b), b)

    # with augmentation ON, scoring is still deterministic and unmasked
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8))
    cfg = RealNVPConfig(
        n_coupling_layers=2,
        hidden_dim=8,
        n_epochs=3,
        batch_size=32,
        seed=0,
        augment_p=1.0,
        augment_width=(2, 3),
    )
    g = RealNVPOneClass(cfg).fit(x)
    assert np.allclose(g.log_likelihood(x), g.log_likelihood(x))


def test_augment_masks_contiguous_block_to_zero():
    import torch

    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    torch.manual_seed(0)
    f = RealNVPOneClass(RealNVPConfig(augment_p=1.0, augment_n_masks=1, augment_width=(4, 4)))
    b = torch.ones(16, 32)
    out = f._augment(b)
    zeroed = (out[0] == 0).nonzero().flatten().tolist()
    assert len(zeroed) == 4, zeroed
    assert zeroed == list(range(zeroed[0], zeroed[0] + 4)), "mask must be CONTIGUOUS"
    # same columns masked for every row in the batch (a batch-level mask)
    assert torch.equal((out == 0).all(dim=0), (out[0] == 0))
    assert not torch.equal(out, b) and torch.equal(b, torch.ones(16, 32)), "input mutated"


def test_augment_degenerate_widths_are_safe():
    import torch

    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    b = torch.ones(4, 8)
    # width >= dim -> no-op rather than an error or a fully-zeroed batch
    wide = RealNVPOneClass(RealNVPConfig(augment_p=1.0, augment_width=(8, 12)))
    assert torch.equal(wide._augment(b), b)
    # a mask can never consume every dimension
    torch.manual_seed(1)
    narrow = RealNVPOneClass(RealNVPConfig(augment_p=1.0, augment_n_masks=3, augment_width=(1, 7)))
    assert (narrow._augment(b) != 0).any()


# ---------------------------------------------------------------------------
# Background-model likelihood ratio (Ren et al. 1906.02845, adapted)
# ---------------------------------------------------------------------------


def test_shuffle_corruption_preserves_marginals_and_kills_dependence():
    """The defining property of the default background: same marginals, no structure."""
    from intrinsic_ai_music_detection.models.background import corrupt

    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 1))
    x = np.hstack([a, a * 0.95 + 0.05 * rng.normal(size=(2000, 1)), rng.normal(size=(2000, 2))])
    xb = corrupt(x, "shuffle", np.random.default_rng(1))

    # every column is a permutation of the original column -> marginals identical
    for j in range(x.shape[1]):
        assert np.allclose(np.sort(x[:, j]), np.sort(xb[:, j]))
    # correlation between the two dependent columns is destroyed
    assert abs(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) > 0.9
    assert abs(np.corrcoef(xb[:, 0], xb[:, 1])[0, 1]) < 0.1


def test_noise_and_mask_corruptions_are_sane():
    from intrinsic_ai_music_detection.models.background import corrupt

    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 32))
    n = corrupt(x, "noise", np.random.default_rng(1), noise_sigma=0.5)
    assert n.shape == x.shape and not np.allclose(n, x) and np.isfinite(n).all()
    m = corrupt(x, "mask", np.random.default_rng(1), mask_width=(4, 8))
    assert m.shape == x.shape and not np.allclose(m, x)
    # masking must not blank an entire row
    assert (m != np.broadcast_to(x.mean(axis=0), x.shape)).any(axis=1).all()
    with pytest.raises(ValueError, match="unknown corruption"):
        corrupt(x, "nope", rng)


def test_likelihood_ratio_cancels_a_shared_marginal_shift():
    """The point of the method: a corpus-identity-like marginal offset must
    move the raw likelihood far more than it moves the ratio."""
    from intrinsic_ai_music_detection.models.background import LikelihoodRatioOneClass
    from intrinsic_ai_music_detection.models.flow import RealNVPConfig, RealNVPOneClass

    rng = np.random.default_rng(0)
    cfg = RealNVPConfig(n_coupling_layers=4, hidden_dim=32, n_epochs=40, batch_size=128, seed=0)

    def make(n, shift):  # correlated features + a per-corpus marginal offset
        a = rng.normal(size=(n, 4))
        return np.hstack([a, a * 0.9 + 0.1 * rng.normal(size=(n, 4))]) + shift

    train = make(3000, 0.0)
    same_corpus = make(600, 0.0)
    other_corpus = make(600, 1.2)  # SAME structure, different marginals

    plain = RealNVPOneClass(cfg).fit(train)
    lr = LikelihoodRatioOneClass(cfg, corruption="shuffle").fit(train)

    def gap(model):  # how much the corpus shift alone moves the anomaly score
        s0, s1 = model.score_samples(same_corpus), model.score_samples(other_corpus)
        return abs(s1.mean() - s0.mean()) / (s0.std() + 1e-9)

    assert gap(lr) < gap(plain), (gap(lr), gap(plain))


def test_likelihood_ratio_api_matches_flow_and_roundtrips(tmp_path):
    from intrinsic_ai_music_detection.models.background import LikelihoodRatioOneClass
    from intrinsic_ai_music_detection.models.flow import RealNVPConfig

    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 8))
    cfg = RealNVPConfig(n_coupling_layers=2, hidden_dim=8, n_epochs=3, batch_size=64, seed=0)
    d = LikelihoodRatioOneClass(cfg, corruption="shuffle").fit(x)
    d.embedding_name = "spec-musicdet"

    assert np.allclose(d.score_samples(x), -d.log_likelihood(x))
    comp = d.component_scores(x)
    assert np.allclose(comp["ratio"], comp["foreground"] - comp["background"])
    assert np.allclose(comp["ratio"], d.log_likelihood(x))

    p = tmp_path / "lr.pt"
    d.save(p)
    assert (tmp_path / "lr_fg.pt").exists() and (tmp_path / "lr_bg.pt").exists()
    r = LikelihoodRatioOneClass.load(p, device="cpu")
    assert r.corruption == "shuffle" and r.embedding_name == "spec-musicdet"
    assert np.allclose(d.log_likelihood(x), r.log_likelihood(x), atol=1e-6)


def test_likelihood_ratio_composes_with_banding():
    from intrinsic_ai_music_detection.models.background import LikelihoodRatioOneClass
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass, RealNVPConfig

    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 8))
    cfg = RealNVPConfig(n_coupling_layers=2, hidden_dim=8, n_epochs=3, batch_size=64, seed=0)
    d = LikelihoodRatioOneClass(cfg, n_bands=2, global_prior=True, global_n_coupling_layers=1).fit(x)
    assert isinstance(d._fg, BandedRealNVPOneClass) and isinstance(d._bg, BandedRealNVPOneClass)
    assert d._fg._global is not None and d._fg._global.config.n_coupling_layers == 1
    assert np.isfinite(d.score_samples(x)).all()


def test_load_detector_dispatches_on_checkpoint_type(tmp_path):
    """Every consumer of --flow-path must get back the model that was saved."""
    from intrinsic_ai_music_detection.models.background import LikelihoodRatioOneClass
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass, RealNVPConfig, RealNVPOneClass
    from intrinsic_ai_music_detection.models.loading import load_detector

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 8))
    cfg = RealNVPConfig(n_coupling_layers=2, hidden_dim=8, n_epochs=3, batch_size=64, seed=0)

    cases = [
        ("plain.pt", RealNVPOneClass(cfg).fit(x), RealNVPOneClass),
        ("banded.pt", BandedRealNVPOneClass(cfg, n_bands=2).fit(x), BandedRealNVPOneClass),
        ("ratio.pt", LikelihoodRatioOneClass(cfg).fit(x), LikelihoodRatioOneClass),
    ]
    for name, model, cls in cases:
        p = tmp_path / name
        model.save(p)
        back = load_detector(p, device="cpu")
        assert isinstance(back, cls), (name, type(back))
        assert np.allclose(model.score_samples(x), back.score_samples(x), atol=1e-6), name
