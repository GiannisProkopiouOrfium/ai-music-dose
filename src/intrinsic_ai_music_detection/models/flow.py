"""Lightweight normalizing-flow one-class detector for AI-music detection.

This is the density-estimation upgrade to the LedoitWolf-Mahalanobis one-class
detector used in the embedding-geometry pipeline. A RealNVP flow models the
distribution of *real* music in a frozen embedding (or descriptor) space and
scores any sample by its log-likelihood under that learned distribution.

Motivation
----------
- Mahalanobis assumes a single Gaussian real manifold. Real music is multimodal
  (genre / instrumentation / production), so a Gaussian ball is a crude fit in a
  high-dimensional space.
- MusicDET (ICML 2026) independently shows a real-only normalizing flow is a
  strong, generator-agnostic detector — but it operates on a *raw STFT energy
  spectrogram* (which their own robustness study shows collapses under MP3 /
  band-limiting). Here we instead place the flow on *frozen self-supervised
  embeddings* (extracted from band-limited / canonicalised audio), so the
  density model is orthogonal to — and not attackable by — the bandwidth
  confound that the spectral axis suffers from.

The detector exposes a scikit-learn-like API mirroring ``_real_manifold_scores``
so it is a drop-in alternative:

    det = RealNVPOneClass().fit(X_real)
    anomaly = det.score_samples(X_query)   # higher = more anomalous = more "AI"

``score_samples`` returns the *negative* log-likelihood, so that — exactly like
Mahalanobis distance — a larger value means a more anomalous (more AI-like)
sample. ``log_likelihood`` returns the raw log-density if the sign matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class RealNVPConfig:
    """Hyper-parameters for the RealNVP one-class detector."""

    n_coupling_layers: int = 8
    hidden_dim: int = 128
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    seed: int = 42
    standardize: bool = True
    # Early-stopping patience on the (real) training NLL; 0 disables.
    patience: int = 20
    verbose: bool = False
    # Feature-masking augmentation during training, ported from MusicDET's
    # SpecAugmentFT (model.py) -- they train their flow with p=0.5 masking while
    # we train on clean features. With probability ``augment_p``, each minibatch
    # has ``augment_n_masks`` contiguous blocks of feature dimensions replaced by
    # the training mean (which is 0 in standardised space, so masking is exactly
    # "no information from these dims"). Block width is drawn uniformly from
    # ``augment_width``.
    #
    # Rationale for a ONE-CLASS density: the flow is free to reach high
    # likelihood by keying on narrow, corpus-specific structure (the (B)
    # production-identity confound) rather than on broad properties of real
    # music. Masking removes any single block's ability to carry the whole
    # score, so the density must be supported by evidence spread across the
    # feature axis. For frequency-ordered features this is literally MusicDET's
    # frequency masking; for EnCodec latents it is structured dropout.
    # 0.0 disables it, which is the default -- every result before 2026-08-12
    # was produced without augmentation.
    augment_p: float = 0.0
    augment_n_masks: int = 1
    augment_width: tuple[int, int] = (6, 20)
    # Base-distribution mean for the real-only prior N(prior_mean, I) in latent
    # space (all dims shifted by the same scalar). Default 0.0 = standard
    # normal, matching the original RealNVP formulation. MusicDET (arXiv
    # 2605.18072, Sec 4.3 "Effect of the Prior Mean mu") reports EER
    # monotonically decreasing as mu_real is shifted away from 0, because it
    # increases latent-space margin between the real mode and everything
    # else. Exposed here so we can reproduce that ablation on our own flow.
    prior_mean: float = 0.0


class RealNVPOneClass:
    """Real-only RealNVP density model used as a label-free anomaly detector.

    The flow is trained by maximum likelihood on real-music feature vectors only.
    At inference time, samples that fall in low-density regions of the learned
    real distribution receive high anomaly scores.
    """

    def __init__(self, config: RealNVPConfig | None = None) -> None:
        self.config = config or RealNVPConfig()
        self._model = None  # lazily built torch module
        self._mu: npt.NDArray[np.float64] | None = None
        self._sd: npt.NDArray[np.float64] | None = None
        self._dim: int | None = None
        self._gaussian_fallback: tuple[np.ndarray, np.ndarray] | None = None
        # Optional linear projection (e.g. a fitted sklearn PCA) applied to raw
        # inputs BEFORE standardisation/scoring. Stored so a saved checkpoint is
        # self-contained: consumers can feed raw pooled windows and get correct
        # likelihoods without separately reconstructing the reduction.
        self._pca_components: npt.NDArray[np.float64] | None = None  # [k, d_raw]
        self._pca_mean: npt.NDArray[np.float64] | None = None  # [d_raw]
        # Which embedding this flow was trained on. Persisted so consumers can
        # refuse to score features from a DIFFERENT front end — several of our
        # representations share 128 dimensions (EnCodec latents and 128-bin
        # log-mel), so a mismatch would otherwise pass every shape check and
        # silently produce meaningless scores.
        self.embedding_name: str | None = None

    def attach_pca(self, pca) -> "RealNVPOneClass":
        """Attach a fitted sklearn ``PCA`` (or anything with ``components_`` /
        ``mean_``) whose transform was applied to the training data. It is then
        serialized with the checkpoint and auto-applied to raw-dim inputs."""
        self._pca_components = np.asarray(pca.components_, dtype=np.float64)
        self._pca_mean = np.asarray(pca.mean_, dtype=np.float64)
        return self

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        x_real: npt.NDArray[np.float64],
        sample_weights: npt.NDArray[np.float64] | None = None,
    ) -> "RealNVPOneClass":
        """Fit the flow on real-only feature vectors of shape ``(n, d)``.

        Parameters
        ----------
        x_real : [n, d] real-music feature vectors (e.g. EnCodec windows).
        sample_weights : optional [n] non-negative weights. When provided,
            minibatches are drawn by weighted sampling *with replacement*
            (like ``torch.utils.data.WeightedRandomSampler``) instead of a
            uniform permutation, so the *effective* training distribution
            is reweighted without duplicating any data in memory. This is
            the mechanism behind genre-balanced (inverse-frequency) resampling:
            pass ``weight[i] = 1 / genre_count[track_of(window_i)]`` to make
            every genre contribute roughly equally per epoch, as distinct
            from simply adding more real data (broadening).
        """
        x = np.asarray(x_real, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x_real must be 2-D (n, d); got shape {x.shape}")
        finite_mask = np.isfinite(x).all(axis=1)
        w = None
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64)
            if len(w) != len(x):
                raise ValueError(f"sample_weights length {len(w)} != x_real length {len(x)}")
            finite_mask &= np.isfinite(w) & (w > 0)
            w = w[finite_mask]
        x = x[finite_mask]
        if len(x) < 10:
            raise ValueError(f"need >=10 finite real samples to fit; got {len(x)}")

        self._dim = x.shape[1]
        if self.config.standardize:
            self._mu = x.mean(axis=0)
            self._sd = x.std(axis=0) + 1e-9
        else:
            self._mu = np.zeros(self._dim)
            self._sd = np.ones(self._dim)
        xz = (x - self._mu) / self._sd

        # A RealNVP needs >=2 dims to split into two coupling halves. For d==1
        # fall back to a 1-D Gaussian density (still a valid likelihood model).
        if self._dim < 2:
            self._gaussian_fallback = (xz.mean(axis=0), xz.std(axis=0) + 1e-9)
            return self

        self._fit_flow(xz, sample_weights=w)
        return self

    def log_likelihood(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return log p(x) under the learned real-music distribution, shape ``(n,)``."""
        if self._mu is None or self._sd is None or self._dim is None:
            raise RuntimeError("call fit() before scoring")
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        # Auto-apply the attached PCA when inputs arrive in the raw (pre-PCA)
        # dimensionality. Inputs already in flow space (d == self._dim) pass
        # through untouched, so callers that transform externally still work.
        if self._pca_components is not None and x.shape[1] != self._dim and x.shape[1] == self._pca_components.shape[1]:
            x = (x - self._pca_mean) @ self._pca_components.T
        if x.shape[1] != self._dim:
            raise ValueError(
                f"input dim {x.shape[1]} does not match flow dim {self._dim}"
                + (
                    f" (nor the attached PCA input dim {self._pca_components.shape[1]})"
                    if self._pca_components is not None
                    else " (no PCA attached to this checkpoint)"
                )
            )
        xz = (x - self._mu) / self._sd
        # log-density correction for the standardisation (constant per-sample, so
        # it does not affect ranking / AUC / EER, but keep it for correctness).
        std_logdet = -np.log(self._sd).sum()

        if self._gaussian_fallback is not None:
            mu1, sd1 = self._gaussian_fallback
            ll = -0.5 * np.log(2 * np.pi) - np.log(sd1) - 0.5 * ((xz - mu1) / sd1) ** 2
            ll = ll.reshape(len(xz), -1).sum(axis=1) + std_logdet
            return ll[0] if single else ll

        ll = self._flow_log_prob(xz) + std_logdet
        return ll[0] if single else ll

    def latents(self, x: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return ``(z, logdet)``: the flow latent and total log|det J| per sample.

        ``log_likelihood(x) == log N(z; prior_mean, I) + logdet`` exactly, so a
        caller can substitute a different (e.g. joint, cross-band) density over
        ``z`` and keep the Jacobian term. Used by BandedRealNVPOneClass's
        global-prior mode.
        """
        if self._mu is None or self._sd is None or self._dim is None:
            raise RuntimeError("call fit() before scoring")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if self._pca_components is not None and x.shape[1] != self._dim and x.shape[1] == self._pca_components.shape[1]:
            x = (x - self._pca_mean) @ self._pca_components.T
        if x.shape[1] != self._dim:
            raise ValueError(f"input dim {x.shape[1]} does not match flow dim {self._dim}")
        xz = (x - self._mu) / self._sd
        std_logdet = -np.log(self._sd).sum()

        if self._gaussian_fallback is not None:
            # Degenerate (<2-d) case: the "flow" is a diagonal affine map, which
            # is still an exact bijection to a standard normal.
            mu1, sd1 = self._gaussian_fallback
            z = (xz - mu1) / sd1
            logdet = np.full(len(xz), std_logdet - np.log(sd1).sum(), dtype=np.float64)
            return z.reshape(len(xz), -1), logdet

        import torch

        with torch.no_grad():
            t = torch.as_tensor(xz, dtype=torch.float32, device=self.config.device)
            z_t, ld_t = self._model.transform(t)
        z = z_t.cpu().numpy().astype(np.float64)
        logdet = ld_t.cpu().numpy().astype(np.float64) + std_logdet
        return z, logdet

    def score_samples(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Anomaly score = negative log-likelihood (higher = more anomalous/AI-like)."""
        return -self.log_likelihood(x)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted flow to disk for later scoring without retraining.

        Saves model weights, standardisation parameters, and config.
        Restore with ``RealNVPOneClass.load(path)``.
        """
        import torch

        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "mu": self._mu,
            "sd": self._sd,
            "dim": self._dim,
            "gaussian_fallback": self._gaussian_fallback,
            "state_dict": self._model.state_dict() if self._model is not None else None,
            # Optional input reduction \u2014 see attach_pca(). Older checkpoints
            # lack these keys; load() treats them as absent.
            "pca_components": self._pca_components,
            "pca_mean": self._pca_mean,
            "embedding_name": self.embedding_name,
        }
        torch.save(payload, dst)
        logger.info("Flow saved \u2192 %s", dst)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RealNVPOneClass":
        """Restore a previously saved flow from disk."""
        import torch

        # weights_only=False is required because the payload contains a dataclass
        # (RealNVPConfig) written by this codebase; only load from trusted paths.
        payload = torch.load(Path(path), map_location=device, weights_only=False)  # noqa: S614
        obj = cls(config=payload["config"])
        # The pickled config carries the TRAINING device (e.g. "cuda"); scoring
        # placement must follow the device requested here, otherwise loading a
        # cuda-trained checkpoint with device="cpu" crashes in _flow_log_prob.
        obj.config.device = device
        obj._mu = payload["mu"]
        obj._sd = payload["sd"]
        obj._dim = payload["dim"]
        obj._gaussian_fallback = payload["gaussian_fallback"]
        obj._pca_components = payload.get("pca_components")
        obj._pca_mean = payload.get("pca_mean")
        obj.embedding_name = payload.get("embedding_name")
        if payload["state_dict"] is not None and obj._dim is not None and obj._dim >= 2:
            obj._model = _build_realnvp(
                dim=obj._dim,
                n_layers=obj.config.n_coupling_layers,
                hidden=obj.config.hidden_dim,
                prior_mean=getattr(obj.config, "prior_mean", 0.0),
            ).to(device)
            obj._model.load_state_dict(payload["state_dict"])
            obj._model.eval()
        logger.info("Flow loaded from %s (dim=%d)", path, obj._dim or -1)
        return obj

    # -- internals ----------------------------------------------------------

    def _fit_flow(
        self,
        xz: npt.NDArray[np.float64],
        sample_weights: npt.NDArray[np.float64] | None = None,
    ) -> None:
        import torch

        device = torch.device(self.config.device)

        # Seed inside a forked-RNG scope: training is deterministic for a given
        # config.seed, but the process-global torch/numpy RNG state is restored
        # afterwards — repeated fits in one process no longer silently reseed
        # every downstream consumer (bootstraps, shuffles created after a fit).
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.config.seed)

            self._model = _build_realnvp(
                dim=self._dim,
                n_layers=self.config.n_coupling_layers,
                hidden=self.config.hidden_dim,
                prior_mean=self.config.prior_mean,
            ).to(device)

            data = torch.from_numpy(xz.astype(np.float32)).to(device)
            weights_t = None
            if sample_weights is not None:
                weights_t = torch.from_numpy(sample_weights.astype(np.float64)).to(device)
                weights_t = weights_t / weights_t.sum()
            opt = torch.optim.Adam(self._model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

            n = len(data)
            bs = min(self.config.batch_size, n)
            stopper = _EarlyStopper(self.config.patience)
            for epoch in range(self.config.n_epochs):
                epoch_nll = self._train_one_epoch(data, opt, n, bs, device, weights=weights_t)
                if self.config.verbose and (epoch % 25 == 0 or epoch == self.config.n_epochs - 1):
                    logger.info("  flow epoch %3d  NLL=%.4f", epoch, epoch_nll)
                if stopper.update(epoch_nll, self._model):
                    break
            stopper.restore(self._model)
        self._model.eval()

    def _augment(self, batch):
        """Mask contiguous feature blocks (MusicDET's SpecAugment, ported).

        Applied to TRAINING batches only; scoring never sees a masked input.
        Values are in standardised space, so writing 0 substitutes the training
        mean for the masked dimensions rather than injecting an outlier.
        """
        import torch

        p = float(getattr(self.config, "augment_p", 0.0))
        if p <= 0.0:
            return batch
        d = batch.shape[1]
        lo, hi = getattr(self.config, "augment_width", (6, 20))
        lo, hi = max(1, int(lo)), max(1, int(hi))
        if lo >= d:
            return batch
        out = None
        for _ in range(int(getattr(self.config, "augment_n_masks", 1))):
            if torch.rand(1, device=batch.device).item() > p:
                continue
            w = int(torch.randint(lo, min(hi, d - 1) + 1, (1,), device=batch.device).item())
            f0 = int(torch.randint(0, d - w + 1, (1,), device=batch.device).item())
            if out is None:
                out = batch.clone()
            out[:, f0 : f0 + w] = 0.0
        return batch if out is None else out

    def _train_one_epoch(self, data, opt, n: int, bs: int, device, weights=None) -> float:
        import torch

        # Weighted resampling (with replacement, like WeightedRandomSampler) so
        # rare-genre windows are seen roughly as often as common-genre ones per
        # epoch, without duplicating any data in memory. Uniform permutation
        # (without replacement) is used when no weights are given (headline path).
        if weights is not None:
            idx_all = torch.multinomial(weights, n, replacement=True)
        else:
            idx_all = torch.randperm(n, device=device)
        epoch_nll = 0.0
        self._model.train()
        for i in range(0, n, bs):
            idx = idx_all[i : i + bs]
            batch = self._augment(data[idx])
            loss = -self._model.log_prob(batch).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
            opt.step()
            epoch_nll += float(loss) * len(idx)
        return epoch_nll / n

    def _flow_log_prob(self, xz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        import torch

        device = torch.device(self.config.device)
        self._model.eval()
        out = np.empty(len(xz), dtype=np.float64)
        bs = 4096
        with torch.no_grad():
            for i in range(0, len(xz), bs):
                batch = torch.from_numpy(xz[i : i + bs].astype(np.float32)).to(device)
                out[i : i + bs] = self._model.log_prob(batch).cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# Torch RealNVP implementation (imported lazily inside the class above)
# ---------------------------------------------------------------------------


def _build_realnvp_modules():
    """Define the torch modules. Done inside a function so importing this file
    does not require torch unless the flow is actually trained."""
    import torch
    import torch.nn as nn

    class _AffineCoupling(nn.Module):
        """Affine coupling layer with a fixed binary mask (RealNVP)."""

        def __init__(self, dim: int, hidden: int, mask: "torch.Tensor") -> None:
            super().__init__()
            self.register_buffer("mask", mask)
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim * 2),
            )
            # Initialise last layer to zero -> starts as identity transform.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.scale_clamp = 5.0

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            x_masked = x * self.mask
            st = self.net(x_masked)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s) * self.scale_clamp
            s = s * (1 - self.mask)
            t = t * (1 - self.mask)
            z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=1)
            return z, log_det

    class _RealNVPModule(nn.Module):
        def __init__(self, dim: int, n_layers: int, hidden: int, prior_mean: float = 0.0) -> None:
            super().__init__()
            self.dim = dim
            masks = []
            base = torch.arange(dim) % 2
            for i in range(n_layers):
                masks.append(base if i % 2 == 0 else 1 - base)
            self.layers = nn.ModuleList([_AffineCoupling(dim, hidden, m.float()) for m in masks])
            # Base distribution N(prior_mean, I) — same scalar shift on every
            # latent dim (mirrors MusicDET's mu_real; see RealNVPConfig.prior_mean).
            # persistent=False: this is a hyperparameter fully determined by
            # config.prior_mean at construction time (not a learned/stateful
            # value), so it must NOT be saved into / loaded from state_dict —
            # otherwise every checkpoint saved before this field existed fails
            # to load with "Missing key(s): prior_mean" under strict loading.
            self.register_buffer("prior_mean", torch.full((dim,), float(prior_mean)), persistent=False)

        def transform(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """Return the latent ``z`` and the accumulated log|det J|.

            Split out from ``log_prob`` so a downstream stage can model the
            joint density of several flows' latents instead of assuming each
            is independently N(prior_mean, I) -- see BandedRealNVPOneClass's
            global-prior mode.
            """
            z = x
            total_log_det = torch.zeros(len(x), device=x.device)
            for layer in self.layers:
                z, log_det = layer(z)
                total_log_det = total_log_det + log_det
            return z, total_log_det

        def log_prob(self, x: "torch.Tensor") -> "torch.Tensor":
            z, total_log_det = self.transform(x)
            log_base = -0.5 * ((z - self.prior_mean) ** 2 + np.log(2 * np.pi)).sum(dim=1)
            return log_base + total_log_det

    return _RealNVPModule


def _build_realnvp(dim: int, n_layers: int, hidden: int, prior_mean: float = 0.0):
    module_cls = _build_realnvp_modules()
    return module_cls(dim=dim, n_layers=n_layers, hidden=hidden, prior_mean=prior_mean)


# ---------------------------------------------------------------------------
# ConditionalRealNVP — genre-conditioned one-class density model (Plan C1)
# ---------------------------------------------------------------------------


@dataclass
class ConditionalRealNVPConfig:
    """Hyper-parameters for the conditional RealNVP (genre-conditioned)."""

    n_coupling_layers: int = 8
    hidden_dim: int = 128
    context_dim: int = 64  # CLAP genre embedding dimension (after projection)
    context_hidden: int = 64  # MLP hidden size for context projection
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    seed: int = 42
    standardize: bool = True
    patience: int = 20
    verbose: bool = False


class ConditionalRealNVPOneClass:
    """RealNVP conditioned on a context vector (e.g. CLAP genre embedding).

    Motivation (Plan Phase C1):
    - Conditioning on genre serves a dual scientific purpose:
      (a) If genre-conditioned AUC matches unconditional AUC, the signal is
          NOT genre — it's genuine codec artifact.
      (b) A genre-aware manifold may improve within-genre discrimination and
          reduce FMA FPR for out-of-distribution genres.

    - Context vector: the CLAP genre posterior (n_genres-dim similarity vector)
      from tag_all_genres.py, projected to ``context_dim`` by a small MLP.
    - At inference, the context is the CLAP genre posterior of the track being
      scored.  For tracks without genre info, falls back to all-zeros context
      (equivalent to unconditional behavior on a zero-conditioned flow).

    Usage
    -----
        cfg = ConditionalRealNVPConfig(context_dim=64)
        det = ConditionalRealNVPOneClass(cfg)
        det.fit(x_real, context_real)        # context: [n_windows, n_genres]
        det.score_samples(x_query, context)  # context can be per-track broadcast
    """

    def __init__(self, config: ConditionalRealNVPConfig | None = None) -> None:
        self.config = config or ConditionalRealNVPConfig()
        self._model = None
        self._context_proj = None  # projects raw context to config.context_dim
        self._mu: npt.NDArray | None = None
        self._sd: npt.NDArray | None = None
        self._ctx_mu: npt.NDArray | None = None
        self._ctx_sd: npt.NDArray | None = None
        self._dim: int | None = None
        self._raw_ctx_dim: int | None = None

    def fit(
        self,
        x_real: npt.NDArray[np.float64],
        context_real: npt.NDArray[np.float64],
    ) -> "ConditionalRealNVPOneClass":
        """Fit the conditional flow.

        Parameters
        ----------
        x_real        : [n_windows, feature_dim] — EnCodec window embeddings.
        context_real  : [n_windows, ctx_raw_dim] — genre posterior per window.
                        If [n_tracks, ctx_raw_dim], the caller must broadcast
                        track context to all windows of that track.
        """
        x = np.asarray(x_real, dtype=np.float64)
        ctx = np.asarray(context_real, dtype=np.float64)
        if ctx.ndim == 1:
            ctx = ctx[None, :].repeat(len(x), axis=0)
        # Align lengths BEFORE masking (ctx may have been passed per-track / pre-aligned).
        # Filtering x alone (as before) and then truncating ctx from index 0 misaligns
        # x[i] with ctx[i] whenever any row of x is dropped — silently pairing windows
        # with the wrong context. Also, an unfiltered NaN/Inf anywhere in ctx poisons
        # ctx.mean()/ctx.std() for that whole column, which then makes ctx_z (and thus
        # every downstream log_prob) NaN for ALL rows, not just the bad one. Both bugs
        # together are why conditional-flow training silently produced NaN for every
        # single held-out generator. Fix: align first, then drop rows where EITHER
        # x or ctx has a non-finite value, keeping the pairing intact.
        min_len = min(len(x), len(ctx))
        x, ctx = x[:min_len], ctx[:min_len]
        finite_mask = np.isfinite(x).all(axis=1) & np.isfinite(ctx).all(axis=1)
        n_dropped = len(finite_mask) - int(finite_mask.sum())
        if n_dropped:
            logger.warning(
                "ConditionalRealNVPOneClass.fit: dropping %d/%d rows with non-finite x or context",
                n_dropped,
                len(finite_mask),
            )
        x, ctx = x[finite_mask], ctx[finite_mask]
        if len(x) < 10:
            raise ValueError(
                f"ConditionalRealNVPOneClass.fit: only {len(x)} finite (x, context) rows "
                "remain after filtering — cannot fit."
            )

        self._dim = x.shape[1]
        self._raw_ctx_dim = ctx.shape[1]

        # Standardize x
        if self.config.standardize:
            self._mu = x.mean(axis=0)
            self._sd = x.std(axis=0) + 1e-9
        else:
            self._mu = np.zeros(self._dim)
            self._sd = np.ones(self._dim)
        xz = (x - self._mu) / self._sd

        # Standardize context
        self._ctx_mu = ctx.mean(axis=0)
        self._ctx_sd = ctx.std(axis=0) + 1e-9
        ctx_z = (ctx - self._ctx_mu) / self._ctx_sd

        self._fit_flow(xz, ctx_z)
        return self

    def log_likelihood(
        self,
        x: npt.NDArray[np.float64],
        context: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return log p(x | context) under the conditional real-music distribution."""
        if self._mu is None:
            raise RuntimeError("Call fit() before scoring.")
        x = np.asarray(x, dtype=np.float64)
        ctx = np.asarray(context, dtype=np.float64)
        if ctx.ndim == 1:
            ctx = np.tile(ctx, (len(x), 1))
        xz = (x - self._mu) / self._sd
        ctx_z = (ctx - self._ctx_mu) / self._ctx_sd
        std_logdet = -np.log(self._sd).sum()
        ll = self._flow_log_prob(xz, ctx_z) + std_logdet
        return ll

    def score_samples(
        self,
        x: npt.NDArray[np.float64],
        context: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Anomaly score = negative log-likelihood (higher = more anomalous)."""
        return -self.log_likelihood(x, context)

    def save(self, path: str | Path) -> None:
        import torch

        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "mu": self._mu,
            "sd": self._sd,
            "ctx_mu": self._ctx_mu,
            "ctx_sd": self._ctx_sd,
            "dim": self._dim,
            "raw_ctx_dim": self._raw_ctx_dim,
            "state_dict": self._model.state_dict() if self._model else None,
            "ctx_proj_state_dict": self._context_proj.state_dict() if self._context_proj else None,
        }
        torch.save(payload, dst)
        logger.info("Conditional flow saved → %s", dst)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ConditionalRealNVPOneClass":
        import torch

        payload = torch.load(Path(path), map_location=device, weights_only=False)  # noqa: S614
        obj = cls(config=payload["config"])
        obj.config.device = device  # scoring follows the requested device, not the training one
        obj._mu, obj._sd = payload["mu"], payload["sd"]
        obj._ctx_mu, obj._ctx_sd = payload["ctx_mu"], payload["ctx_sd"]
        obj._dim, obj._raw_ctx_dim = payload["dim"], payload["raw_ctx_dim"]
        if payload.get("state_dict") and obj._dim:
            obj._model, obj._context_proj = _build_conditional_realnvp(
                dim=obj._dim,
                raw_ctx_dim=obj._raw_ctx_dim,
                ctx_dim=obj.config.context_dim,
                ctx_hidden=obj.config.context_hidden,
                n_layers=obj.config.n_coupling_layers,
                hidden=obj.config.hidden_dim,
            )
            obj._model.to(device).load_state_dict(payload["state_dict"])
            obj._context_proj.to(device).load_state_dict(payload["ctx_proj_state_dict"])
            obj._model.eval()
            obj._context_proj.eval()
        return obj

    def _fit_flow(self, xz: npt.NDArray, ctx_z: npt.NDArray) -> None:
        import torch

        device = torch.device(self.config.device)
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.config.seed)
            self._fit_flow_seeded(xz, ctx_z, device)

    def _fit_flow_seeded(self, xz: npt.NDArray, ctx_z: npt.NDArray, device) -> None:
        import torch

        self._model, self._context_proj = _build_conditional_realnvp(
            dim=self._dim,
            raw_ctx_dim=self._raw_ctx_dim,
            ctx_dim=self.config.context_dim,
            ctx_hidden=self.config.context_hidden,
            n_layers=self.config.n_coupling_layers,
            hidden=self.config.hidden_dim,
        )
        self._model.to(device)
        self._context_proj.to(device)

        params = list(self._model.parameters()) + list(self._context_proj.parameters())
        opt = torch.optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)

        x_tensor = torch.from_numpy(xz.astype(np.float32)).to(device)
        ctx_tensor = torch.from_numpy(ctx_z.astype(np.float32)).to(device)
        n = len(x_tensor)
        bs = min(self.config.batch_size, n)
        stopper = _EarlyStopper(self.config.patience)

        for epoch in range(self.config.n_epochs):
            self._model.train()
            self._context_proj.train()
            perm = torch.randperm(n, device=device)
            epoch_nll = 0.0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                xb, cb = x_tensor[idx], ctx_tensor[idx]
                ctx_emb = self._context_proj(cb)  # [B, ctx_dim]
                loss = -self._model.log_prob(xb, ctx_emb).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                epoch_nll += float(loss) * len(idx)
            epoch_nll /= n
            if self.config.verbose and (epoch % 25 == 0 or epoch == self.config.n_epochs - 1):
                logger.info("  cond_flow epoch %3d  NLL=%.4f", epoch, epoch_nll)
            # Track flow AND projector together — they are trained jointly, so
            # restoring only the flow would pair best-epoch flow weights with
            # final-epoch projector weights (an inconsistent model).
            if stopper.update(epoch_nll, self._model, self._context_proj):
                break
        stopper.restore(self._model, self._context_proj)
        self._model.eval()
        self._context_proj.eval()

    def _flow_log_prob(self, xz: npt.NDArray, ctx_z: npt.NDArray) -> npt.NDArray:
        import torch

        device = torch.device(self.config.device)
        self._model.eval()
        self._context_proj.eval()
        out = np.empty(len(xz), dtype=np.float64)
        bs = 4096
        with torch.no_grad():
            for i in range(0, len(xz), bs):
                xb = torch.from_numpy(xz[i : i + bs].astype(np.float32)).to(device)
                cb = torch.from_numpy(ctx_z[i : i + bs].astype(np.float32)).to(device)
                ctx_emb = self._context_proj(cb)
                out[i : i + bs] = self._model.log_prob(xb, ctx_emb).cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# DualFlowDetector — likelihood-ratio detector (Plan C2)
# ---------------------------------------------------------------------------


class DualFlowDetector:
    """Likelihood-ratio detector: logp_real(x) - logp_fake(x).

    Trains two RealNVP flows: one on real windows, one on fake windows.
    Detection score = log p_real(x) - log p_fake(x).

    This is the supervised upper-bound ablation (Plan Phase C2).  The
    HEADLINE detector is still the real-only flow; the dual flow is reported
    separately as a ceiling / ablation to show how much labeled fake data helps.

    Protocol (LOGO for fairness):
    - For each LOGO fold: train real-flow on all-folds-except-held-out real,
      train fake-flow on all-folds-except-held-out FAKE of 4 generators
      (held-out generator is NEVER seen by either flow).
    - Score held-out real + held-out fake generator -> AUC/EER.
    """

    def __init__(
        self,
        real_cfg: RealNVPConfig | None = None,
        fake_cfg: RealNVPConfig | None = None,
    ) -> None:
        self._real_flow = RealNVPOneClass(real_cfg or RealNVPConfig())
        self._fake_flow: RealNVPOneClass | None = None
        self._fake_cfg = fake_cfg or RealNVPConfig()

    @property
    def real_flow(self) -> "RealNVPOneClass":
        """The real-only flow, exposed so callers can report the one-class
        baseline on the SAME audio without re-fitting or reaching into privates.

        Needed by ``run_proxy_negative_detector.py``: the whole point of that
        experiment is the delta between the ratio and the one-class score, and
        the comparison is only meaningful if both come from the same flow.
        """
        return self._real_flow

    def fit_real(self, x_real: npt.NDArray[np.float64]) -> "DualFlowDetector":
        """Fit the real-music flow."""
        self._real_flow.fit(x_real)
        return self

    def fit_fake(self, x_fake: npt.NDArray[np.float64]) -> "DualFlowDetector":
        """Fit the fake-music flow (requires labeled fake windows)."""
        self._fake_flow = RealNVPOneClass(self._fake_cfg)
        self._fake_flow.fit(x_fake)
        return self

    def score_samples(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Anomaly score = -(logp_real - logp_fake) = logp_fake - logp_real.

        Higher score = more likely fake = more anomalous.
        If fake flow is not fitted, falls back to the real-only flow score.
        """
        if self._fake_flow is None:
            logger.warning("DualFlowDetector.fit_fake() not called; falling back to real-only flow.")
            return self._real_flow.score_samples(x)
        ll_real = self._real_flow.log_likelihood(x)
        ll_fake = self._fake_flow.log_likelihood(x)
        return ll_fake - ll_real  # higher = more fake-like

    @staticmethod
    def _component_path(path: str | Path, suffix: str) -> Path:
        # Suffix only the filename stem — str.replace(".pt", …) would corrupt
        # any *directory* name containing ".pt" anywhere in the path.
        p = Path(path)
        return p.with_name(f"{p.stem}_{suffix}{p.suffix or '.pt'}")

    def save(self, path: str | Path) -> None:
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._real_flow.save(self._component_path(dst, "real"))
        if self._fake_flow is not None:
            self._fake_flow.save(self._component_path(dst, "fake"))
        logger.info("DualFlowDetector saved → %s", dst)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DualFlowDetector":
        real_path = cls._component_path(path, "real")
        fake_path = cls._component_path(path, "fake")
        obj = cls()
        obj._real_flow = RealNVPOneClass.load(real_path, device=device)
        if Path(fake_path).exists():
            obj._fake_flow = RealNVPOneClass.load(fake_path, device=device)
        return obj


class _EarlyStopper:
    """Tracks the best (lowest) training NLL and restores the best weights.

    Accepts one or more modules trained jointly (e.g. a conditional flow plus
    its context projector): the best state of EVERY module is snapshotted at the
    same epoch, so a restore never mixes weights from different epochs.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_nll = np.inf
        self.best_states: list[dict] | None = None
        self.bad = 0

    def update(self, nll: float, *models) -> bool:
        """Record ``nll``; return True if training should stop early."""
        if self.patience <= 0:
            return False
        if nll < self.best_nll - 1e-4:
            self.best_nll = nll
            self.best_states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore(self, *models) -> None:
        if self.best_states is None:
            return
        if len(models) != len(self.best_states):
            raise ValueError(f"restore() got {len(models)} modules but {len(self.best_states)} were tracked")
        for m, state in zip(models, self.best_states):
            m.load_state_dict(state)


# ---------------------------------------------------------------------------
# Conditional RealNVP torch modules (lazy import, like the base modules)
# ---------------------------------------------------------------------------


def _build_conditional_realnvp_modules():
    """Define context-conditioned coupling layers.

    Each affine coupling layer receives both the data half and the projected
    context embedding, concatenating them before the coupling MLP.
    This is the standard FiLM-free conditioning approach for normalizing flows.
    """
    import torch
    import torch.nn as nn

    class _ContextProjector(nn.Module):
        """Projects raw context (e.g. CLAP genre posterior) to ctx_dim."""

        def __init__(self, raw_dim: int, hidden: int, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(raw_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
                nn.Tanh(),
            )

        def forward(self, ctx: "torch.Tensor") -> "torch.Tensor":
            return self.net(ctx)

    class _ConditionalAffineCoupling(nn.Module):
        """Affine coupling layer conditioned on a context vector."""

        def __init__(self, dim: int, hidden: int, ctx_dim: int, mask: "torch.Tensor") -> None:
            super().__init__()
            self.register_buffer("mask", mask)
            self.net = nn.Sequential(
                nn.Linear(dim + ctx_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim * 2),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.scale_clamp = 5.0

        def forward(self, x: "torch.Tensor", ctx: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
            x_masked = x * self.mask
            # Concatenate context to masked input before MLP
            inp = torch.cat([x_masked, ctx], dim=1)
            st = self.net(inp)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s) * self.scale_clamp
            s = s * (1 - self.mask)
            t = t * (1 - self.mask)
            z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=1)
            return z, log_det

    class _ConditionalRealNVPModule(nn.Module):
        def __init__(self, dim: int, ctx_dim: int, n_layers: int, hidden: int) -> None:
            super().__init__()
            self.dim = dim
            base = torch.arange(dim) % 2
            masks = [base if i % 2 == 0 else 1 - base for i in range(n_layers)]
            self.layers = nn.ModuleList([_ConditionalAffineCoupling(dim, hidden, ctx_dim, m.float()) for m in masks])

        def log_prob(self, x: "torch.Tensor", ctx: "torch.Tensor") -> "torch.Tensor":
            z = x
            total_log_det = torch.zeros(len(x), device=x.device)
            for layer in self.layers:
                z, log_det = layer(z, ctx)
                total_log_det = total_log_det + log_det
            import math

            log_base = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            return log_base + total_log_det

    return _ConditionalRealNVPModule, _ContextProjector


def _build_conditional_realnvp(
    dim: int,
    raw_ctx_dim: int,
    ctx_dim: int,
    ctx_hidden: int,
    n_layers: int,
    hidden: int,
):
    """Build a (ConditionalRealNVP, ContextProjector) pair."""
    module_cls, proj_cls = _build_conditional_realnvp_modules()
    model = module_cls(dim=dim, ctx_dim=ctx_dim, n_layers=n_layers, hidden=hidden)
    projector = proj_cls(raw_dim=raw_ctx_dim, hidden=ctx_hidden, out_dim=ctx_dim)
    return model, projector


# ---------------------------------------------------------------------------
# BandedRealNVP — frequency-guided one-class flow (MusicDET's core architecture)
# ---------------------------------------------------------------------------


def _restore_flow_from_payload(p: dict, device: str) -> "RealNVPOneClass":
    """Rebuild a RealNVPOneClass from the dict written by BandedRealNVPOneClass.save."""
    f = RealNVPOneClass(config=p["config"])
    f.config.device = device
    f._mu, f._sd, f._dim = p["mu"], p["sd"], p["dim"]
    f._gaussian_fallback = p["gaussian_fallback"]
    if p["state_dict"] is not None and f._dim and f._dim >= 2:
        f._model = _build_realnvp(
            dim=f._dim,
            n_layers=f.config.n_coupling_layers,
            hidden=f.config.hidden_dim,
            prior_mean=getattr(f.config, "prior_mean", 0.0),
        ).to(device)
        f._model.load_state_dict(p["state_dict"])
        f._model.eval()
    return f


class BandedRealNVPOneClass:
    """Independent RealNVP flows over contiguous FREQUENCY BANDS, summed.

    MusicDET (arXiv:2605.18072) describes itself as a *frequency-guided*
    normalizing flow: instead of one joint density over the whole spectral
    vector, it splits the spectrogram into bands and fits a shallow flow (K=2)
    per band. We had no analogue — our single 128-d vector has no band axis —
    and that is the one architectural axis of theirs we never tested.

    Why it should matter, in our own numbers: on FakeMusicCaps the flow sees
    only ~21k training windows for a 128-d joint density, and fails (macro AUC
    0.585-0.621). Splitting into B bands replaces one D-dimensional density
    estimate with B independent D/B-dimensional ones, each far better determined
    by the same data — the standard curse-of-dimensionality factorisation. On
    SONICS (330k windows) the joint estimate is already well-determined, which
    predicts banding helps little there and a lot on FakeMusicCaps. That
    asymmetry is the falsifiable prediction.

    Independence across bands is an approximation (it drops cross-band
    correlations), which is precisely the bias/variance trade being made.

    ``log p(x) = sum_b log p_b(x_b)`` — valid as a density over the product
    space, so scores stay comparable in the usual anomaly-oriented way.

    NOTE: bands are contiguous column blocks, so this is only *frequency*
    banding when the features are frequency-ordered (log-mel). Do NOT combine
    with PCA, which mixes frequencies and destroys the band semantics.
    """

    def __init__(
        self,
        config: RealNVPConfig | None = None,
        n_bands: int = 2,
        global_prior: bool = False,
        global_n_coupling_layers: int | None = None,
    ) -> None:
        if n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {n_bands}")
        self.config = config or RealNVPConfig()
        self.n_bands = n_bands
        self.global_prior = bool(global_prior)
        # MusicDET builds the global stage at global_K=1 -- a single coupling
        # layer -- while each band flow gets K. None reuses the band depth
        # (more capacity than theirs); set 1 to match them exactly.
        self.global_n_coupling_layers = global_n_coupling_layers
        self._flows: list[RealNVPOneClass] = []
        self._slices: list[tuple[int, int]] = []
        self._global: RealNVPOneClass | None = None
        self._dim: int | None = None
        self.embedding_name: str | None = None

    def _make_slices(self, dim: int) -> list[tuple[int, int]]:
        if self.n_bands * 2 > dim:
            raise ValueError(
                f"n_bands={self.n_bands} needs >=2 dims per band but input has {dim}; " f"use at most {dim // 2} bands"
            )
        edges = np.linspace(0, dim, self.n_bands + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(self.n_bands)]

    def fit(self, x_real: npt.NDArray[np.float64], sample_weights=None) -> "BandedRealNVPOneClass":
        x = np.asarray(x_real, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x_real must be 2-D (n, d); got {x.shape}")
        self._dim = x.shape[1]
        self._slices = self._make_slices(self._dim)
        self._flows = []
        for bi, (lo, hi) in enumerate(self._slices):
            cfg = RealNVPConfig(**{**self.config.__dict__, "seed": self.config.seed + bi})
            f = RealNVPOneClass(cfg).fit(x[:, lo:hi], sample_weights=sample_weights)
            self._flows.append(f)
            logger.info("  band %d/%d: dims [%d:%d] fitted", bi + 1, self.n_bands, lo, hi)

        if self.global_prior and self.n_bands > 1:
            # Second stage: model the JOINT density of the concatenated band
            # latents instead of assuming each is independently N(mu, I).
            # This restores the cross-band dependence that plain summation
            # drops, and mirrors MusicDET's own design (glow_model.py:669-794,
            # "band-wise flows followed by a global prior").
            z_train = np.concatenate(
                [f.latents(x[:, lo:hi])[0] for f, (lo, hi) in zip(self._flows, self._slices)], axis=1
            )
            gk = self.global_n_coupling_layers or self.config.n_coupling_layers
            gcfg = RealNVPConfig(**{**self.config.__dict__, "seed": self.config.seed + 1000, "n_coupling_layers": gk})
            self._global = RealNVPOneClass(gcfg).fit(z_train, sample_weights=sample_weights)
            logger.info("  global prior over %d concatenated latent dims fitted", z_train.shape[1])
        return self

    def log_likelihood(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not self._flows:
            raise RuntimeError("call fit() before scoring")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self._dim:
            raise ValueError(f"input dim {x.shape[1]} != fitted dim {self._dim}")
        if self._global is not None:
            # log p(x) = log p_global(z) + sum_b logdet_b   -- an exact density,
            # since the per-band maps are bijections and the global stage
            # supplies the base density over their stacked latents.
            zs, logdet = [], np.zeros(len(x), dtype=np.float64)
            for f, (lo, hi) in zip(self._flows, self._slices):
                z_b, ld_b = f.latents(x[:, lo:hi])
                zs.append(z_b)
                logdet += ld_b
            return self._global.log_likelihood(np.concatenate(zs, axis=1)) + logdet

        total = np.zeros(len(x), dtype=np.float64)
        for f, (lo, hi) in zip(self._flows, self._slices):
            total += f.log_likelihood(x[:, lo:hi])
        return total

    def score_samples(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Anomaly score = negative summed log-likelihood (higher = more AI-like)."""
        return -self.log_likelihood(x)

    def save(self, path: str | Path) -> None:
        import torch

        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "banded": True,
            "n_bands": self.n_bands,
            "global_prior": self.global_prior,
            "global_n_coupling_layers": self.global_n_coupling_layers,
            "global": (
                {
                    "config": self._global.config,
                    "mu": self._global._mu,
                    "sd": self._global._sd,
                    "dim": self._global._dim,
                    "gaussian_fallback": self._global._gaussian_fallback,
                    "state_dict": (self._global._model.state_dict() if self._global._model is not None else None),
                }
                if self._global is not None
                else None
            ),
            "slices": self._slices,
            "dim": self._dim,
            "config": self.config,
            "embedding_name": self.embedding_name,
            "bands": [
                {
                    "config": f.config,
                    "mu": f._mu,
                    "sd": f._sd,
                    "dim": f._dim,
                    "gaussian_fallback": f._gaussian_fallback,
                    "state_dict": f._model.state_dict() if f._model is not None else None,
                }
                for f in self._flows
            ],
        }
        torch.save(payload, dst)
        logger.info("Banded flow (%d bands) saved → %s", self.n_bands, dst)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BandedRealNVPOneClass":
        import torch

        payload = torch.load(Path(path), map_location=device, weights_only=False)  # noqa: S614
        if not payload.get("banded"):
            raise ValueError(f"{path} is not a banded flow checkpoint")
        obj = cls(
            config=payload["config"],
            n_bands=payload["n_bands"],
            global_prior=payload.get("global_prior", False),
            global_n_coupling_layers=payload.get("global_n_coupling_layers"),
        )
        obj._slices = [tuple(s) for s in payload["slices"]]
        obj._dim = payload["dim"]
        obj.embedding_name = payload.get("embedding_name")
        for band in payload["bands"]:
            obj._flows.append(_restore_flow_from_payload(band, device))

        gp = payload.get("global")
        if gp is not None:
            obj._global = _restore_flow_from_payload(gp, device)
        return obj
