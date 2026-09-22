"""Background-model likelihood ratio — a label-free correction for corpus identity.

Motivation
----------
Our raw one-class score ``-log p_real(x)`` is confounded. It decomposes into

    (A) genuine synthesis artifacts        <- what we want to detect
    (B) corpus / production identity       <- loudness, spectral tilt, bandwidth,
                                              mastering, recording chain

(B) is why the same flow reaches ~0.99 AUC on SONICS, only ~0.62 on
FakeMusicCaps, and produces a 21% false-positive rate on FMA: it has learned
"sounds like *this corpus's* real music" rather than "is not synthesised". It is
also why likelihoods *invert* (AUC < 0.5) on some corpora — the classic
likelihood-OOD pathology of Nalisnick et al. (ICLR 2019, arXiv:1810.09136), which
we observe both in our XLS-R arm and in MusicDET's own released model.

Ren et al. (NeurIPS 2019, arXiv:1906.02845) give the label-free fix: fit a second
"background" density on a corrupted version of the same training data, and score
with the **likelihood ratio**

    LLR(x) = log p_foreground(x) - log p_background(x)

The background model absorbs whatever survives the corruption — the generic,
population-level statistics — so the ratio keeps only the structure the
foreground model captures *beyond* them.

Choice of corruption
--------------------
The corruption defines what counts as "background", so it is the design decision
that matters:

``shuffle`` (default)
    Permute every feature dimension independently across samples. This exactly
    preserves each dimension's marginal distribution and destroys all
    cross-dimensional dependence, so the background model is the product of
    marginals and ``LLR`` estimates the **total correlation** of the features.
    Rationale specific to this problem: production identity is largely a
    *marginal* property (a track's overall spectral envelope, level, bandwidth),
    whereas vocoder and codec artifacts are *relational* — phase-incoherent
    partials, band-edge correlations, unnatural inter-band structure. Ratioing
    against the marginals is therefore a principled way to subtract (B) while
    keeping (A).

``noise``
    Add per-dimension Gaussian noise (Ren's input-perturbation, adapted to a
    continuous space). Blurs fine structure while keeping coarse structure.

``mask``
    Randomly zero contiguous feature blocks (MusicDET's SpecAugment as a
    corruption). Keeps everything except locally-contiguous detail.

The wrapper exposes the same ``fit`` / ``score_samples`` / ``log_likelihood``
API as ``RealNVPOneClass``, so it is a drop-in inside the existing fold and
symmetric-scoring machinery.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass, RealNVPConfig, RealNVPOneClass

logger = logging.getLogger(__name__)

CORRUPTIONS = ("shuffle", "noise", "mask")


def corrupt(
    x: npt.NDArray[np.float64],
    mode: str,
    rng: np.random.Generator,
    noise_sigma: float = 0.5,
    mask_width: tuple[int, int] = (6, 20),
) -> npt.NDArray[np.float64]:
    """Return a corrupted copy of ``x`` (n, d). See module docstring for modes."""
    x = np.asarray(x, dtype=np.float64)
    if mode == "shuffle":
        # independent permutation per column -> marginals preserved exactly,
        # every cross-dimensional dependence destroyed
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            out[:, j] = x[rng.permutation(len(x)), j]
        return out
    if mode == "noise":
        sd = x.std(axis=0, keepdims=True) + 1e-9
        return x + noise_sigma * sd * rng.standard_normal(x.shape)
    if mode == "mask":
        out = x.copy()
        col_mean = x.mean(axis=0)
        d = x.shape[1]
        lo, hi = max(1, int(mask_width[0])), max(1, int(mask_width[1]))
        if lo >= d:
            return out
        hi = min(hi, d - 1)
        widths = rng.integers(lo, hi + 1, size=len(x))
        starts = (rng.random(len(x)) * (d - widths + 1)).astype(int)
        for i, (s, w) in enumerate(zip(starts, widths)):
            out[i, s : s + w] = col_mean[s : s + w]
        return out
    raise ValueError(f"unknown corruption {mode!r}; choose from {CORRUPTIONS}")


class LikelihoodRatioOneClass:
    """Foreground/background flow pair scored by their log-likelihood ratio.

    ``log_likelihood`` returns ``log p_fg(x) - log p_bg(x)`` and ``score_samples``
    returns its negation, so the anomaly orientation (higher = more AI-like) is
    unchanged and every downstream consumer keeps working.
    """

    def __init__(
        self,
        config: RealNVPConfig | None = None,
        corruption: str = "shuffle",
        noise_sigma: float = 0.5,
        mask_width: tuple[int, int] = (6, 20),
        n_bands: int = 1,
        global_prior: bool = False,
        global_n_coupling_layers: int | None = None,
    ) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {corruption!r}")
        self.config = config or RealNVPConfig()
        self.corruption = corruption
        self.noise_sigma = float(noise_sigma)
        self.mask_width = mask_width
        self.n_bands = n_bands
        self.global_prior = global_prior
        self.global_n_coupling_layers = global_n_coupling_layers
        self._fg: RealNVPOneClass | BandedRealNVPOneClass | None = None
        self._bg: RealNVPOneClass | BandedRealNVPOneClass | None = None
        self.embedding_name: str | None = None

    # -- construction -------------------------------------------------------

    def _inner(self, seed_offset: int):
        cfg = RealNVPConfig(**{**self.config.__dict__, "seed": self.config.seed + seed_offset})
        if self.n_bands and self.n_bands > 1:
            return BandedRealNVPOneClass(
                cfg,
                n_bands=self.n_bands,
                global_prior=self.global_prior,
                global_n_coupling_layers=self.global_n_coupling_layers,
            )
        return RealNVPOneClass(cfg)

    # -- API mirroring RealNVPOneClass --------------------------------------

    def fit(self, x_real, sample_weights=None) -> "LikelihoodRatioOneClass":
        x = np.asarray(x_real, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x_real must be 2-D (n, d); got {x.shape}")
        self._fg = self._inner(0).fit(x, sample_weights=sample_weights)
        rng = np.random.default_rng(self.config.seed + 9973)
        x_bg = corrupt(x, self.corruption, rng, self.noise_sigma, self.mask_width)
        # NOTE the background flow is trained on the SAME real tracks, only
        # corrupted -- no fakes and no labels enter anywhere.
        self._bg = self._inner(5000).fit(x_bg, sample_weights=sample_weights)
        logger.info("likelihood-ratio detector fitted (corruption=%s)", self.corruption)
        return self

    def log_likelihood(self, x) -> npt.NDArray[np.float64]:
        if self._fg is None or self._bg is None:
            raise RuntimeError("call fit() before scoring")
        return self._fg.log_likelihood(x) - self._bg.log_likelihood(x)

    def score_samples(self, x) -> npt.NDArray[np.float64]:
        """Anomaly score = -(log p_fg - log p_bg); higher = more AI-like."""
        return -self.log_likelihood(x)

    def component_scores(self, x) -> dict[str, npt.NDArray[np.float64]]:
        """Foreground / background / ratio log-likelihoods, for diagnostics.

        Useful for showing *how much* of the raw score was corpus identity: if
        foreground and background AUCs are similar and the ratio's is much
        higher, the (B) component was dominating.
        """
        if self._fg is None or self._bg is None:
            raise RuntimeError("call fit() before scoring")
        fg = self._fg.log_likelihood(x)
        bg = self._bg.log_likelihood(x)
        return {"foreground": fg, "background": bg, "ratio": fg - bg}

    def attach_pca(self, pca) -> "LikelihoodRatioOneClass":
        for m in (self._fg, self._bg):
            if m is not None and hasattr(m, "attach_pca"):
                m.attach_pca(pca)
        return self

    # -- persistence --------------------------------------------------------

    def _paths(self, path: str | Path) -> tuple[Path, Path]:
        p = Path(path)
        return p.with_name(p.stem + "_fg" + p.suffix), p.with_name(p.stem + "_bg" + p.suffix)

    def save(self, path: str | Path) -> None:
        import torch

        if self._fg is None or self._bg is None:
            raise RuntimeError("call fit() before save()")
        fg_p, bg_p = self._paths(path)
        self._fg.save(fg_p)
        self._bg.save(bg_p)
        meta = {
            "likelihood_ratio": True,
            "corruption": self.corruption,
            "noise_sigma": self.noise_sigma,
            "mask_width": tuple(self.mask_width),
            "n_bands": self.n_bands,
            "global_prior": self.global_prior,
            "global_n_coupling_layers": self.global_n_coupling_layers,
            "config": self.config,
            "embedding_name": self.embedding_name,
            "fg_path": fg_p.name,
            "bg_path": bg_p.name,
        }
        torch.save(meta, Path(path))
        logger.info("likelihood-ratio detector saved → %s (+ _fg/_bg)", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "LikelihoodRatioOneClass":
        import torch

        meta = torch.load(Path(path), map_location=device, weights_only=False)  # noqa: S614
        if not meta.get("likelihood_ratio"):
            raise ValueError(f"{path} is not a likelihood-ratio checkpoint")
        obj = cls(
            config=meta["config"],
            corruption=meta["corruption"],
            noise_sigma=meta["noise_sigma"],
            mask_width=tuple(meta["mask_width"]),
            n_bands=meta["n_bands"],
            global_prior=meta["global_prior"],
            global_n_coupling_layers=meta.get("global_n_coupling_layers"),
        )
        obj.embedding_name = meta.get("embedding_name")
        base = Path(path)
        loader = BandedRealNVPOneClass if meta["n_bands"] > 1 else RealNVPOneClass
        obj._fg = loader.load(base.with_name(meta["fg_path"]), device=device)
        obj._bg = loader.load(base.with_name(meta["bg_path"]), device=device)
        return obj
