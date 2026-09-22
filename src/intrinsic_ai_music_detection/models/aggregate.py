"""Trajectory -> track-score aggregation, and multi-flow (mixture) scoring.

Sign convention (repo-wide going forward): every function here operates on
**anomaly scores** (negative log-likelihood, higher = more anomalous = more
AI-like), i.e. the output of ``RealNVPOneClass.score_samples``. Never pass raw
log-likelihoods — the historical dual convention (``wf_mean`` = +LL in CSVs vs
``score_samples`` = -LL) has already caused one silent AUC inversion.

Aggregators (I1 ablation)
-------------------------
The headline track score has always been the arithmetic mean over per-window
scores. Patch/segment-based anomaly-detection literature consistently finds
that when the anomalous evidence is *localized* (a few windows carry the
artifact), max / top-k% aggregation outperforms the mean, while the mean wins
when evidence is diffuse. These aggregators turn saved per-window trajectories
into an ablation table with zero extra GPU time.

Mixture-of-flows (I2)
---------------------
Per-corpus real flows are kept separate and a track is scored by the flow that
explains it BEST: ``anomaly(x) = min_i cal_i(anomaly_i(x))``. Because absolute
NLLs from independently trained flows are not on a shared scale, each flow's
scores are first mapped to the empirical quantile of that flow's own held-out
REAL score distribution (label-free calibration). A real track from any covered
domain then gets a low calibrated score from its own domain's flow; fakes score
high under every flow.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Ordered so ablation tables print consistently.
AGGREGATOR_NAMES = (
    "mean",
    "median",
    "trimmed_mean_10",
    "top_10pct",
    "top_25pct",
    "max",
    "logsumexp",
)


def aggregate_trajectory(anomaly_traj: npt.ArrayLike, method: str = "mean") -> float:
    """Aggregate a per-window anomaly-score trajectory into one track score.

    Non-finite entries are dropped first; returns ``nan`` if nothing remains.
    ``top_Xpct`` = mean of the ceil(X%) MOST anomalous windows (localized
    evidence); ``trimmed_mean_10`` drops the top and bottom 10% of windows
    before averaging (robust diffuse evidence); ``logsumexp`` is a smooth max.
    """
    traj = np.asarray(anomaly_traj, dtype=np.float64).ravel()
    traj = traj[np.isfinite(traj)]
    if len(traj) == 0:
        return float("nan")
    if method == "mean":
        return float(traj.mean())
    if method == "median":
        return float(np.median(traj))
    if method == "trimmed_mean_10":
        if len(traj) < 3:
            return float(traj.mean())
        lo, hi = np.percentile(traj, [10, 90])
        core = traj[(traj >= lo) & (traj <= hi)]
        return float(core.mean()) if len(core) else float(traj.mean())
    if method in ("top_10pct", "top_25pct"):
        frac = 0.10 if method == "top_10pct" else 0.25
        k = max(1, int(np.ceil(frac * len(traj))))
        return float(np.sort(traj)[-k:].mean())
    if method == "max":
        return float(traj.max())
    if method == "logsumexp":
        m = traj.max()
        return float(m + np.log(np.exp(traj - m).sum()) - np.log(len(traj)))
    raise ValueError(f"unknown aggregation method: {method!r}")


def aggregate_all(anomaly_traj: npt.ArrayLike) -> dict[str, float]:
    """All aggregators for one trajectory — one row of the I1 ablation table."""
    return {m: aggregate_trajectory(anomaly_traj, m) for m in AGGREGATOR_NAMES}


class QuantileCalibrator:
    """Maps anomaly scores to empirical quantiles of a reference (real) sample.

    Fully label-free: the reference is a flow's own held-out REAL score
    distribution. After calibration, scores from different flows live on a
    common [0, 1] scale and can be compared / combined (mixture scoring).
    Scores above every reference value map to 1.0, below every value to 0.0;
    interpolation is linear between reference order statistics.
    """

    def __init__(self) -> None:
        self._sorted_ref: np.ndarray | None = None

    def fit(self, reference_scores: npt.ArrayLike) -> "QuantileCalibrator":
        ref = np.asarray(reference_scores, dtype=np.float64).ravel()
        ref = ref[np.isfinite(ref)]
        if len(ref) < 10:
            raise ValueError(f"need >=10 finite reference scores; got {len(ref)}")
        self._sorted_ref = np.sort(ref)
        return self

    def transform(self, scores: npt.ArrayLike) -> np.ndarray:
        if self._sorted_ref is None:
            raise RuntimeError("call fit() before transform()")
        s = np.asarray(scores, dtype=np.float64)
        ref = self._sorted_ref
        # Empirical CDF with linear interpolation between order statistics.
        q = np.interp(s, ref, np.linspace(0.0, 1.0, len(ref)), left=0.0, right=1.0)
        out = np.where(np.isfinite(s), q, np.nan)
        return out

    # -- persistence (plain npz-friendly) -----------------------------------
    def to_array(self) -> np.ndarray:
        if self._sorted_ref is None:
            raise RuntimeError("not fitted")
        return self._sorted_ref

    @classmethod
    def from_array(cls, sorted_ref: npt.ArrayLike) -> "QuantileCalibrator":
        obj = cls()
        obj._sorted_ref = np.asarray(sorted_ref, dtype=np.float64)
        return obj


def mixture_min_score(
    scores_by_flow: dict[str, npt.ArrayLike],
    calibrators: dict[str, QuantileCalibrator] | None = None,
) -> np.ndarray:
    """Mixture-of-flows anomaly score: minimum over per-flow (calibrated) scores.

    Parameters
    ----------
    scores_by_flow : flow name -> per-track anomaly scores (aligned arrays).
    calibrators : flow name -> fitted QuantileCalibrator on that flow's own
        held-out-real scores. STRONGLY recommended — raw NLLs from different
        flows are not on a shared scale; omit only when the caller has already
        calibrated. Missing keys fall back to raw scores with no warning
        suppression (a warning is logged).

    A track is real-like if ANY flow explains it (low score under that flow),
    so the min is taken; nan entries are ignored per-track (all-nan -> nan).
    """
    import logging

    logger = logging.getLogger(__name__)
    names = list(scores_by_flow)
    if not names:
        raise ValueError("scores_by_flow is empty")
    cols = []
    for name in names:
        s = np.asarray(scores_by_flow[name], dtype=np.float64)
        if calibrators is not None and name in calibrators:
            s = calibrators[name].transform(s)
        elif calibrators is not None:
            logger.warning("mixture_min_score: no calibrator for flow %r — using raw scores", name)
        cols.append(s)
    mat = np.stack(cols, axis=1)  # [n_tracks, n_flows]
    lengths = {len(c) for c in cols}
    if len(lengths) != 1:
        raise ValueError(f"per-flow score arrays have differing lengths: {lengths}")
    with np.errstate(invalid="ignore"):
        out = np.nanmin(mat, axis=1)
    return out
