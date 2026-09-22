"""Tests for the convolutional-coupling flow.

This model exists to close the one architectural gap against MusicDET: they
model a 32x64x100 map with convolutional couplings, we mean-pool it to a vector.
The tests below pin the properties that make that possible — invertibility
bookkeeping, weight sharing, and sensitivity to time structure that mean pooling
provably cannot have.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from intrinsic_ai_music_detection.models.conv_flow import (  # noqa: E402
    ActNorm2d,
    ConvCoupling,
    ConvFlowConfig,
    ConvRealNVP,
    ConvRealNVPOneClass,
    Squeeze,
    _spec_augment,
)

SHAPE = (1, 16, 16)  # C, F, T — divisible by 2**n_scales


def _windows(n: int, seed: int = 0, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1.0, size=(n, *SHAPE)).astype(np.float32)
    # a little frequency structure, so the flow has something to model
    x += np.linspace(0, 1, SHAPE[1], dtype=np.float32)[None, None, :, None]
    return x


def test_squeeze_preserves_volume():
    x = torch.randn(2, 3, 8, 8)
    y = Squeeze()(x)
    assert y.shape == (2, 12, 4, 4)
    assert y.numel() == x.numel()


def test_actnorm_normalises_on_first_forward():
    layer = ActNorm2d(4).train()
    x = torch.randn(64, 4, 8, 8) * 7.0 + 3.0
    y, logdet = layer(x)
    assert y.mean().abs() < 0.2, "actnorm must centre its first batch"
    assert abs(float(y.std()) - 1.0) < 0.2, "actnorm must scale its first batch"
    assert logdet.shape == (64,)


def test_coupling_starts_as_the_identity():
    """The last conv is zero-initialised, so training starts well-conditioned."""
    layer = ConvCoupling(4, hidden=8)
    x = torch.randn(2, 4, 8, 8)
    y, logdet = layer(x)
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.allclose(logdet, torch.zeros(2), atol=1e-6)


def test_coupling_leaves_its_conditioning_half_untouched():
    """Half the channels must pass through unchanged or the map is not a coupling
    and the log-determinant is wrong."""
    layer = ConvCoupling(4, hidden=8)
    with torch.no_grad():
        layer.net[-1].weight.normal_(0, 0.1)
        layer.net[-1].bias.normal_(0, 0.1)
    x = torch.randn(2, 4, 8, 8)
    y, _ = layer(x)
    assert torch.allclose(y[:, :2], x[:, :2])
    assert not torch.allclose(y[:, 2:], x[:, 2:])


def test_parameter_count_is_independent_of_input_size():
    """The entire argument for convolutions. An MLP coupling on 2048 pooled dims
    overfits 16k windows and inverts (measured: macro 0.413 vs 0.621 at 256
    dims); a conv coupling modelling 8x more values costs the same parameters.
    """
    cfg = ConvFlowConfig(n_blocks=2, n_scales=1, hidden_channels=16)
    small = sum(p.numel() for p in ConvRealNVP(1, cfg).parameters())
    # 4x the values per window, same architecture
    large = sum(p.numel() for p in ConvRealNVP(1, cfg).parameters())
    assert small == large


def test_log_prob_is_finite_and_per_sample():
    cfg = ConvFlowConfig(n_blocks=2, n_scales=2, hidden_channels=8)
    model = ConvRealNVP(1, cfg)
    x = torch.randn(5, 1, 16, 16)
    lp = model.log_prob(x)
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()


def test_fit_then_score_gives_higher_anomaly_to_shifted_data():
    """The one-class property: windows unlike the training distribution must
    score higher."""
    cfg = ConvFlowConfig(
        n_blocks=2, n_scales=2, hidden_channels=16, n_epochs=12, batch_size=32, device="cpu", patience=0
    )
    det = ConvRealNVPOneClass(cfg).fit(_windows(160, seed=1))
    in_dist = det.score_samples(_windows(40, seed=2))
    out_dist = det.score_samples(_windows(40, seed=3, shift=4.0))
    assert np.isfinite(in_dist).all() and np.isfinite(out_dist).all()
    assert out_dist.mean() > in_dist.mean()


def test_score_distinguishes_time_reversal():
    """Mean pooling is provably invariant to time reversal, so it cannot encode
    a temporal artifact signature. A conv flow over [C,F,T] can — this is the
    capability the whole model exists to add."""
    cfg = ConvFlowConfig(
        n_blocks=2, n_scales=2, hidden_channels=16, n_epochs=12, batch_size=32, device="cpu", patience=0
    )
    train = _windows(160, seed=4)
    # give the training set a consistent temporal ramp
    train = train + np.linspace(0, 2, SHAPE[2], dtype=np.float32)[None, None, None, :]
    det = ConvRealNVPOneClass(cfg).fit(train)

    probe = _windows(40, seed=5) + np.linspace(0, 2, SHAPE[2], dtype=np.float32)[None, None, None, :]
    forward = det.score_samples(probe)
    reversed_ = det.score_samples(np.ascontiguousarray(probe[:, :, :, ::-1]))
    assert not np.allclose(forward, reversed_), "a conv flow must not be invariant to time reversal"
    # mean pooling, for contrast, is exactly invariant
    assert np.allclose(probe.mean(axis=3), probe[:, :, :, ::-1].mean(axis=3))


def test_non_divisible_shape_is_rejected_with_a_reason():
    cfg = ConvFlowConfig(n_blocks=1, n_scales=2, device="cpu", n_epochs=1)
    with pytest.raises(ValueError, match="divisible"):
        ConvRealNVPOneClass(cfg).fit(np.zeros((4, 1, 15, 16), dtype=np.float32))


def test_wrong_rank_input_is_rejected():
    cfg = ConvFlowConfig(n_blocks=1, n_scales=1, device="cpu", n_epochs=1)
    with pytest.raises(ValueError, match=r"\[N, C, F, T\]"):
        ConvRealNVPOneClass(cfg).fit(np.zeros((4, 256), dtype=np.float32))


def test_spec_augment_masks_and_does_not_mutate_the_input():
    x = torch.ones(2, 1, 32, 32)
    out = _spec_augment(x, p=1.0, max_freq=8, max_time=8)
    assert (out == 0).any(), "with p=1 something must be masked"
    assert (x == 1).all(), "the caller's tensor must not be modified in place"
    assert torch.equal(_spec_augment(x, p=0.0, max_freq=8, max_time=8), x)


def test_roundtrip_save_load_preserves_scores(tmp_path):
    cfg = ConvFlowConfig(n_blocks=2, n_scales=1, hidden_channels=8, n_epochs=5, batch_size=32, device="cpu", patience=0)
    det = ConvRealNVPOneClass(cfg).fit(_windows(96, seed=6))
    det.embedding_name = "spec-musicdet"
    probe = _windows(8, seed=7)
    before = det.score_samples(probe)

    path = tmp_path / "conv.pt"
    det.save(path)
    restored = ConvRealNVPOneClass.load(path, device="cpu")
    assert restored.embedding_name == "spec-musicdet"
    assert np.allclose(before, restored.score_samples(probe), atol=1e-5)


def test_window_normalize_removes_absolute_level():
    """MusicDET RMS-normalises every 4.04 s segment; we normalise per track. On a
    log spectrogram a waveform gain is an additive constant, so the equivalent is
    subtracting each window's mean — and the resulting score must then be
    invariant to that gain. peak_dbfs is still a 0.64-0.69 channel descriptor
    after every other control, so this is worth being exact about.
    """
    cfg = ConvFlowConfig(
        n_blocks=2,
        n_scales=1,
        hidden_channels=8,
        n_epochs=6,
        batch_size=32,
        device="cpu",
        patience=0,
        window_normalize=True,
    )
    det = ConvRealNVPOneClass(cfg).fit(_windows(96, seed=8))
    probe = _windows(16, seed=9)
    base = det.score_samples(probe)
    gained = det.score_samples(probe + 3.0)  # a constant log-domain gain
    assert np.allclose(base, gained, atol=1e-4)


def test_without_window_normalize_level_still_matters():
    """The contrast: the default keeps absolute level, so a gain changes the
    score. Both behaviours are legitimate; the point is that the flag does
    something and which one is in force is recorded."""
    cfg = ConvFlowConfig(
        n_blocks=2,
        n_scales=1,
        hidden_channels=8,
        n_epochs=6,
        batch_size=32,
        device="cpu",
        patience=0,
        window_normalize=False,
    )
    det = ConvRealNVPOneClass(cfg).fit(_windows(96, seed=8))
    probe = _windows(16, seed=9)
    assert not np.allclose(det.score_samples(probe), det.score_samples(probe + 3.0), atol=1e-4)
