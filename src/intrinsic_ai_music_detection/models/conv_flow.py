"""Convolutional-coupling normalizing flow over time-frequency windows.

The gap this closes
-------------------
MusicDET (arXiv:2605.18072) models a 32x64x100 feature map — 204,800 values —
with **convolutional** coupling networks, in 8.13M parameters. We mean-pool a
4 s window's ~400 STFT frames into a single 256-d vector and run MLP couplings on
that. §9.7.19 called this "the honest remaining gap"; every result since has
narrowed the field down to it.

Measured evidence that the bottleneck is the binding constraint, not a detail:

* Mean pooling is **invariant to time reversal**, so it cannot represent a
  temporal artifact signature at all.
* Widening the pooled vector does not help — `--wf-pooling blocks 8` on
  spec-musicdet gives 8x256 = 2048 dims and *inverts* (macro 0.413 against
  0.621 for the 256-d version), because an MLP coupling on 2048 dims fitted to
  16,065 windows simply overfits.

That second point is the whole argument for convolutions. A conv coupling
**shares weights across time and frequency**, so modelling 200k values costs no
more parameters than modelling 2k. Capacity stops scaling with input size, which
is exactly why MusicDET can model the full map and our MLP cannot.

What this is, precisely
-----------------------
A RealNVP over ``[B, C, F, T]`` tensors:

* ``Squeeze`` — invertible space-to-depth, trading resolution for channels so
  couplings see a wider receptive field cheaply (Glow's trick; logdet 0).
* ``ActNorm2d`` — per-channel affine, data-dependent init.
* ``ChannelPermute`` — fixed permutation. MusicDET's spec-nf sets
  ``flow_permutation="permute"``, i.e. it does **not** use the invertible 1x1
  convolution, so we match that rather than the Glow default.
* ``ConvCoupling`` — affine coupling whose scale/shift come from a small CNN.

Still one-class and label-free: trained by maximum likelihood on real windows
only, scored by negative log-likelihood. The representation stays frozen — the
convolutions are inside the *flow*, not a learned front end. That distinction is
the paper's contrast with MusicDET and it is preserved here deliberately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ConvFlowConfig:
    """Hyper-parameters for the convolutional one-class flow."""

    n_blocks: int = 8  # coupling layers per scale
    n_scales: int = 2  # squeeze operations
    hidden_channels: int = 64  # width of the coupling CNNs
    n_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 50.0
    device: str = "cpu"
    seed: int = 42
    patience: int = 15
    verbose: bool = False
    # MusicDET trains with SpecAugment at p=0.5 (model.py). Masking forces the
    # density to be supported by evidence spread across the map rather than by
    # one narrow band, which is the frequency-domain analogue of dropout.
    augment_p: float = 0.0
    augment_max_freq: int = 20
    augment_max_time: int = 20
    # MusicDET RMS-normalises EVERY 4.04 s segment (dataset.py: audio / sqrt(mean(x^2))),
    # whereas we normalise loudness once per track. Per-window normalisation removes
    # all absolute level from the model's view, which matters here because peak_dbfs
    # is still a 0.64-0.69 channel-alone descriptor on both corpora after every other
    # control. On a LOG spectrogram a waveform gain is an additive constant, so the
    # equivalent operation is subtracting each window's own mean.
    window_normalize: bool = False


class Squeeze(nn.Module):
    """Space-to-depth: ``[B,C,H,W] -> [B,4C,H/2,W/2]``. Volume-preserving."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * 4, h // 2, w // 2)


class ActNorm2d(nn.Module):
    """Per-channel affine with data-dependent initialisation.

    Without it the first coupling sees wildly-scaled inputs and training
    diverges; this is the same role it plays in Glow.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("initialised", torch.tensor(0, dtype=torch.uint8))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.initialised.item() == 0 and self.training:
            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                std = x.std(dim=(0, 2, 3), keepdim=True) + 1e-6
                self.bias.data = -mean / std
                self.log_scale.data = -torch.log(std)
                self.initialised.fill_(1)
        y = x * torch.exp(self.log_scale) + self.bias
        # One log-scale per channel applies at every spatial position.
        logdet = self.log_scale.sum() * x.shape[2] * x.shape[3]
        return y, logdet.expand(x.shape[0])


class ChannelPermute(nn.Module):
    """Fixed channel permutation so successive couplings see different splits."""

    def __init__(self, num_channels: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("perm", torch.randperm(num_channels, generator=g))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]


class ConvCoupling(nn.Module):
    """Affine coupling whose scale and shift are produced by a small CNN.

    The final convolution is zero-initialised, so the layer starts as the
    identity and the flow begins training from a well-conditioned point.
    """

    def __init__(self, num_channels: int, hidden: int) -> None:
        super().__init__()
        self.split = num_channels // 2
        rest = num_channels - self.split
        self.net = nn.Sequential(
            nn.Conv2d(self.split, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rest * 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x[:, : self.split], x[:, self.split :]
        h = self.net(x_a)
        shift, scale = h.chunk(2, dim=1)
        # tanh-bounded log-scale: an unbounded scale makes the likelihood
        # explode on out-of-distribution inputs, which is precisely the regime a
        # one-class detector spends its life in.
        log_s = torch.tanh(scale) * 2.0
        y_b = x_b * torch.exp(log_s) + shift
        return torch.cat([x_a, y_b], dim=1), log_s.flatten(1).sum(dim=1)


class ConvRealNVP(nn.Module):
    """Multi-scale convolutional RealNVP producing a Gaussian latent."""

    def __init__(self, in_channels: int, cfg: ConvFlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.squeeze = Squeeze()
        self.scales = nn.ModuleList()
        channels = in_channels
        for scale in range(cfg.n_scales):
            channels *= 4  # each squeeze quadruples the channel count
            layers = nn.ModuleList()
            for block in range(cfg.n_blocks):
                layers.append(ActNorm2d(channels))
                layers.append(ChannelPermute(channels, seed=cfg.seed + 100 * scale + block))
                layers.append(ConvCoupling(channels, cfg.hidden_channels))
            self.scales.append(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layers in self.scales:
            x = self.squeeze(x)
            for layer in layers:
                if isinstance(layer, ChannelPermute):
                    x = layer(x)
                else:
                    x, ld = layer(x)
                    logdet = logdet + ld
        return x, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x)
        prior = -0.5 * (z.pow(2) + np.log(2.0 * np.pi))
        return prior.flatten(1).sum(dim=1) + logdet


def _spec_augment(
    x: torch.Tensor, p: float, max_freq: int, max_time: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Zero a random frequency band and time span, per MusicDET's SpecAugment.

    Applied only during training, and to a copy — masking the stored tensor would
    corrupt the cache for every later epoch.
    """
    if p <= 0.0 or torch.rand(1, generator=generator).item() > p:
        return x
    x = x.clone()
    _, _, f, t = x.shape
    if max_freq > 0 and f > 1:
        width = int(torch.randint(1, min(max_freq, f) + 1, (1,), generator=generator).item())
        start = int(torch.randint(0, f - width + 1, (1,), generator=generator).item())
        x[:, :, start : start + width, :] = 0.0
    if max_time > 0 and t > 1:
        width = int(torch.randint(1, min(max_time, t) + 1, (1,), generator=generator).item())
        start = int(torch.randint(0, t - width + 1, (1,), generator=generator).item())
        x[:, :, :, start : start + width] = 0.0
    return x


class ConvRealNVPOneClass:
    """One-class detector API matching ``RealNVPOneClass``, over 4-D windows.

    ``fit`` takes ``[N, C, F, T]`` real windows; ``score_samples`` returns a
    per-window anomaly score (negative log-likelihood, higher = more anomalous),
    so every downstream aggregator and calibrator works unchanged.
    """

    def __init__(self, cfg: ConvFlowConfig | None = None) -> None:
        self.cfg = cfg or ConvFlowConfig()
        self.model: ConvRealNVP | None = None
        self.embedding_name: str | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.window_normalize:
            # Per-window, before the global per-bin statistics: this is the
            # analogue of MusicDET's per-segment RMS normalisation.
            x = x - x.mean(axis=(1, 2, 3), keepdims=True)
        return (x - self._mean) / self._std

    def fit(self, x: npt.NDArray[np.float64]) -> "ConvRealNVPOneClass":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"expected [N, C, F, T], got shape {x.shape}")
        n, c, f, t = x.shape
        div = 2**self.cfg.n_scales
        if f % div or t % div:
            raise ValueError(
                f"F and T must be divisible by 2**n_scales={div}; got F={f}, T={t}. "
                "Crop the window rather than pad — padding injects a constant region "
                "the flow can key on."
            )
        self._shape = (c, f, t)

        # Per-frequency-bin standardisation: a spectrogram's dynamic range varies
        # by orders of magnitude across frequency, and an un-normalised flow
        # spends its capacity on that instead of on structure.
        centred = x - x.mean(axis=(1, 2, 3), keepdims=True) if self.cfg.window_normalize else x
        self._mean = centred.mean(axis=(0, 3), keepdims=True)
        self._std = centred.std(axis=(0, 3), keepdims=True) + 1e-6

        torch.manual_seed(self.cfg.seed)
        device = torch.device(self.cfg.device)
        self.model = ConvRealNVP(c, self.cfg).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "ConvRealNVP: %d params modelling %d values per window (%dx%dx%d); "
            "weight sharing is why these two numbers can differ by orders of magnitude",
            n_params,
            c * f * t,
            c,
            f,
            t,
        )

        data = torch.from_numpy(self._standardise(x))
        gen = torch.Generator().manual_seed(self.cfg.seed)
        best, bad = float("inf"), 0
        best_state = None

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen)
            total, seen = 0.0, 0
            for i in range(0, n, self.cfg.batch_size):
                batch = data[perm[i : i + self.cfg.batch_size]].to(device)
                batch = _spec_augment(
                    batch,
                    self.cfg.augment_p,
                    self.cfg.augment_max_freq,
                    self.cfg.augment_max_time,
                    gen,
                )
                loss = -self.model.log_prob(batch).mean() / (c * f * t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
                total += float(loss) * len(batch)
                seen += len(batch)

            epoch_loss = total / max(seen, 1)
            if self.cfg.verbose and (epoch % 10 == 0 or epoch == self.cfg.n_epochs - 1):
                logger.info("  conv-flow epoch %3d  NLL/dim=%.5f", epoch, epoch_loss)

            if epoch_loss < best - 1e-5:
                best, bad = epoch_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if self.cfg.patience and bad >= self.cfg.patience:
                    logger.info("  early stop at epoch %d (best NLL/dim=%.5f)", epoch, best)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def log_likelihood(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.model is None:
            raise RuntimeError("fit() must be called before scoring")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != self._shape:
            raise ValueError(f"expected [N, {self._shape}], got {x.shape}")
        device = torch.device(self.cfg.device)
        self.model.eval()
        out: list[np.ndarray] = []
        data = torch.from_numpy(self._standardise(x))
        with torch.no_grad():
            for i in range(0, len(data), self.cfg.batch_size):
                batch = data[i : i + self.cfg.batch_size].to(device)
                out.append(self.model.log_prob(batch).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def score_samples(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Anomaly score: higher = more anomalous. Normalised per value so the
        scale is comparable across window shapes."""
        c, f, t = self._shape
        return -self.log_likelihood(x) / (c * f * t)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save: fit() was never called")
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "conv_realnvp",
                "cfg": self.cfg.__dict__,
                "state_dict": self.model.state_dict(),
                "mean": self._mean,
                "std": self._std,
                "shape": self._shape,
                "embedding_name": self.embedding_name,
            },
            dst,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ConvRealNVPOneClass":
        payload = torch.load(str(path), map_location=device, weights_only=False)
        if payload.get("kind") != "conv_realnvp":
            raise ValueError(f"{path} is not a conv_realnvp checkpoint (kind={payload.get('kind')})")
        cfg = ConvFlowConfig(**payload["cfg"])
        cfg.device = device
        obj = cls(cfg)
        obj._shape = tuple(payload["shape"])
        obj._mean, obj._std = payload["mean"], payload["std"]
        obj.embedding_name = payload.get("embedding_name")
        obj.model = ConvRealNVP(obj._shape[0], cfg).to(device)
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        return obj
