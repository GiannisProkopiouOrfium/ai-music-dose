"""Checkpoint-type dispatch for the one-class detectors.

We now serialize three shapes of detector — plain ``RealNVPOneClass``, the
frequency-banded ``BandedRealNVPOneClass``, and the foreground/background
``LikelihoodRatioOneClass``. They share the ``score_samples`` /
``log_likelihood`` API, so every consumer (external-corpus FPR, robustness
battery, reconstruction control, fusion) can be agnostic about which one it is
— but only if loading is agnostic too.

Loading the wrong class is the failure mode this prevents: ``RealNVPOneClass.load``
on a banded or ratio checkpoint either raises deep inside a state-dict load or,
worse, silently produces a detector that scores with only part of the model.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_detector(path: str | Path, device: str = "cpu"):
    """Load whichever detector was saved at *path*, dispatching on the payload.

    Returns an object exposing ``log_likelihood(x)`` and ``score_samples(x)``
    with the usual anomaly orientation (higher = more AI-like).
    """
    import torch

    from intrinsic_ai_music_detection.models.background import LikelihoodRatioOneClass
    from intrinsic_ai_music_detection.models.flow import BandedRealNVPOneClass, RealNVPOneClass

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    payload = torch.load(p, map_location="cpu", weights_only=False)  # noqa: S614

    if isinstance(payload, dict) and payload.get("likelihood_ratio"):
        logger.info("loading likelihood-ratio detector (corruption=%s) from %s", payload.get("corruption"), p)
        return LikelihoodRatioOneClass.load(p, device=device)
    if isinstance(payload, dict) and payload.get("banded"):
        logger.info("loading banded flow (%d bands) from %s", payload.get("n_bands", "?"), p)
        return BandedRealNVPOneClass.load(p, device=device)
    logger.info("loading single flow from %s", p)
    return RealNVPOneClass.load(p, device=device)
