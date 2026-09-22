"""Model-agnostic geometric descriptors of sliding-window embedding clouds.

This module answers the question "what else, besides intrinsic dimension, can a
sliding window over an embedding trajectory reveal?".  Every descriptor is
computed *directly from the cached embedding matrix* (frames x dim) — no extra
model forward pass — so new descriptors iterate in CPU-seconds via
``run_balanced_ablation.py --recompute-features``.

Three complementary, cheap descriptor families:

  1. Local geometry of each window's point cloud (linear-algebraic; O(d^2) or
     O(n^2) on a capped point set):
       - effective_rank   : exp(spectral entropy of singular values)
       - participation_ratio : (sum λ)^2 / sum λ^2 of covariance eigenvalues
       - anisotropy       : fraction of variance on the top principal component

  2. Local density (kNN on the capped window cloud):
       - mean_nn_distance : average distance to k nearest neighbours

  3. Trajectory dynamics over the *ordered* frame sequence (whole track):
       - velocity / acceleration : norms of 1st / 2nd frame differences
       - curvature        : turning angle between consecutive step vectors
       - novelty          : off-diagonal self-similarity contrast (how often the
                            trajectory revisits earlier states)
       - structure        : long-range temporal coherence + segment structure
                            read from the frame self-similarity matrix (SSM).
                            Motivated by SpecTTTra / SONICS: generative models
                            produce good local texture but weak long-range
                            structure (drift, missing verse/chorus recurrence,
                            smeared segment boundaries). All measures are
                            scale-relative so they are model- and
                            sample-rate-agnostic.

Per-window descriptors are summarised across windows with the same statistics
used for temporal ID (mean / std / range / slope / first_last) so they slot
directly into the ablation feature table and classifier bundles.  The intuition:
AI generators with finite context windows are expected to produce trajectories
that are smoother, lower-rank and more self-similar (more revisiting) than human
recordings.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---- defaults ----
KNN_K: int = 5  # neighbours for local density
MAX_WINDOW_POINTS: int = 400  # cap per-window cloud before O(n^2) work
EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Local geometry of a single window cloud
# ---------------------------------------------------------------------------


def _centered_singular_values(window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Singular values of the mean-centred window (frames x dim)."""
    centered = window - window.mean(axis=0, keepdims=True)
    # economy SVD; values are non-negative, descending
    try:
        sv = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.array([], dtype=float)
    return sv[sv > EPS]


def effective_rank(window: npt.NDArray[np.float64]) -> float:
    """exp(Shannon entropy of the normalised singular-value spectrum).

    A continuous, scale-invariant measure of how many directions the cloud
    meaningfully occupies (1 == perfectly 1-D, higher == more isotropic).
    """
    sv = _centered_singular_values(window)
    if sv.size < 2:
        return float(sv.size)
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p + EPS))
    return float(np.exp(entropy))


def participation_ratio(window: npt.NDArray[np.float64]) -> float:
    """(sum λ)^2 / sum λ^2 over covariance eigenvalues (λ = singular_value^2).

    Equals the number of equally-weighted dimensions that would yield the same
    spread; lower => variance concentrated in few directions.
    """
    sv = _centered_singular_values(window)
    if sv.size == 0:
        return 0.0
    lam = sv**2
    denom = float(np.sum(lam**2))
    if denom <= EPS:
        return 0.0
    return float((np.sum(lam) ** 2) / denom)


def anisotropy(window: npt.NDArray[np.float64]) -> float:
    """Fraction of total variance captured by the top principal component."""
    sv = _centered_singular_values(window)
    if sv.size == 0:
        return 0.0
    lam = sv**2
    total = float(np.sum(lam))
    if total <= EPS:
        return 0.0
    return float(lam[0] / total)


def mean_nn_distance(window: npt.NDArray[np.float64], k: int = KNN_K) -> float:
    """Average distance to the ``k`` nearest neighbours within the window.

    A local-density proxy: lower => points are packed more tightly (the cloud
    is locally denser / more repetitive).
    """
    n = len(window)
    if n < k + 1:
        return float("nan")
    # pairwise squared Euclidean distances
    sq = np.sum(window**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (window @ window.T)
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 0.0)
    # k smallest per row
    part = np.partition(d2, k, axis=1)[:, :k]
    return float(np.mean(np.sqrt(part)))


def window_geometry(window: npt.NDArray[np.float64], k: int = KNN_K) -> dict[str, float]:
    """All per-window scalar descriptors for one (frames x dim) cloud."""
    if len(window) > MAX_WINDOW_POINTS:
        idx = np.linspace(0, len(window) - 1, MAX_WINDOW_POINTS).astype(int)
        window = window[idx]
    return {
        "effrank": effective_rank(window),
        "partratio": participation_ratio(window),
        "anisotropy": anisotropy(window),
        "nndist": mean_nn_distance(window, k=k),
    }


# ---------------------------------------------------------------------------
# Trajectory dynamics over the ordered frame sequence
# ---------------------------------------------------------------------------


def _structural_descriptors(sim: npt.NDArray[np.float64], kernel: int = 8) -> dict[str, float]:
    """Long-range coherence + segment-structure descriptors from an SSM.

    ``sim`` is the cosine self-similarity matrix of the (subsampled) frame
    trajectory. These features target the axis where generative models are
    hypothesised to fail relative to human recordings: macro-structure. They
    are SpecTTTra/SONICS-inspired but computed *without any trained classifier*
    (purely from embedding geometry), so they remain model-agnostic. Every
    quantity is a ratio, normalised slope or fraction, hence robust to frame
    subsampling and embedding scale.
    """
    n = sim.shape[0]
    out: dict[str, float] = {}
    if n < 2 * kernel + 1:
        return out

    # --- time-lag similarity profile: mean similarity at each temporal lag ---
    lags = np.arange(1, n)
    lag_profile = np.array([float(np.mean(np.diagonal(sim, offset=int(lag)))) for lag in lags])
    if lag_profile.size < 4 or not np.isfinite(lag_profile).all():
        return out

    norm_lag = lags / lags[-1]  # normalised lag in (0, 1]

    # decay: how fast similarity falls with lag. Strong negative slope = the
    # trajectory drifts away from earlier states (no return); ~0 = recurrent.
    coeffs = np.polyfit(norm_lag, lag_profile, 1)
    out["struct_decay_slope"] = float(coeffs[0])

    # long-range coherence: mean similarity at lags > 50% of the track. High =
    # the trajectory revisits earlier states (verse/chorus-like recurrence).
    long_mask = norm_lag > 0.5
    if long_mask.any():
        out["struct_longrange"] = float(np.mean(lag_profile[long_mask]))

    # recurrence strength: tallest repetition peak after removing the decay
    # trend, normalised by the profile's own variability.
    resid = lag_profile - np.polyval(coeffs, norm_lag)
    rstd = float(np.std(resid))
    if rstd > EPS:
        out["struct_recurrence"] = float(np.max(resid) / rstd)

    # --- checkerboard-kernel novelty: segment-boundary strength ---
    kern = np.ones((2 * kernel, 2 * kernel))
    kern[:kernel, kernel:] = -1.0
    kern[kernel:, :kernel] = -1.0
    nov = np.empty(n - 2 * kernel)
    for j, c in enumerate(range(kernel, n - kernel)):
        nov[j] = float(np.sum(sim[c - kernel : c + kernel, c - kernel : c + kernel] * kern))
    nov /= (2 * kernel) ** 2  # scale-normalise by kernel area
    if nov.size >= 2:
        out["struct_novelty_std"] = float(np.std(nov))
        thr = float(np.mean(nov) + np.std(nov))
        out["struct_boundary_rate"] = float(np.mean(nov > thr))

    return out


def trajectory_dynamics(emb_matrix: npt.NDArray[np.float64], max_points: int = 1500) -> dict[str, float]:
    """Velocity / acceleration / curvature / novelty of the frame trajectory.

    Operates on the time-ordered sequence of frame embeddings (subsampled to
    ``max_points`` for the O(n^2) novelty term).
    """
    X = emb_matrix[np.isfinite(emb_matrix).all(axis=1)]
    if len(X) < 4:
        return {}
    if len(X) > max_points:
        idx = np.linspace(0, len(X) - 1, max_points).astype(int)
        X = X[idx]

    out: dict[str, float] = {}

    # velocity: norm of consecutive frame differences
    vel = np.linalg.norm(np.diff(X, axis=0), axis=1)
    if vel.size:
        out["vel_mean"] = float(np.mean(vel))
        out["vel_std"] = float(np.std(vel))

    # acceleration: norm of second differences
    acc = np.linalg.norm(np.diff(X, n=2, axis=0), axis=1)
    if acc.size:
        out["acc_mean"] = float(np.mean(acc))
        out["acc_std"] = float(np.std(acc))

    # curvature: turning angle between consecutive step vectors
    steps = np.diff(X, axis=0)
    sn = np.linalg.norm(steps, axis=1)
    valid = sn > EPS
    if valid.sum() >= 2:
        unit = steps[valid] / sn[valid][:, None]
        cos = np.sum(unit[:-1] * unit[1:], axis=1)
        angles = np.arccos(np.clip(cos, -1.0, 1.0))
        out["curv_mean"] = float(np.mean(angles))
        out["curv_std"] = float(np.std(angles))

    # novelty: off-diagonal self-similarity contrast (revisiting behaviour)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    x_unit = X / np.maximum(norms, EPS)
    sim = x_unit @ x_unit.T
    n = len(X)
    off = sim[~np.eye(n, dtype=bool)]
    out["selfsim_mean"] = float(np.mean(off))
    out["selfsim_max"] = float(np.max(off))

    # long-range coherence + segment structure (SpecTTTra/SONICS-inspired,
    # reuses the SSM already computed above — no extra O(n^2) pass)
    out.update(_structural_descriptors(sim))

    return out


# ---------------------------------------------------------------------------
# Public entry point: windowed descriptors + summary statistics
# ---------------------------------------------------------------------------


def _summarise(values: list[float], times: npt.NDArray[np.float64], prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if finite.size < 2:
        return {}
    t = times[finite_mask]
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_range": float(np.max(finite) - np.min(finite)),
        f"{prefix}_slope": float(np.polyfit(t, finite, 1)[0]),
        f"{prefix}_first_last": float(finite[-1] - finite[0]),
    }


def window_descriptor_features(
    emb_matrix: npt.NDArray[np.float64],
    audio_duration: float,
    window_duration: float,
    hop_duration: float,
    k: int = KNN_K,
) -> dict[str, float]:
    """Sliding-window geometric descriptors + whole-track trajectory dynamics.

    Mirrors the windowing in ``run_balanced_ablation._temporal_id_from_matrix``
    (same window/hop frame logic, including the short-track fallback) so the
    descriptors align window-for-window with the temporal ID features.

    Returns a flat dict of ``wd_*`` scalar features ready to merge into the
    ablation feature table.
    """
    n_frames = len(emb_matrix)
    if n_frames < 2 or audio_duration <= 0:
        return {}

    fps = n_frames / audio_duration
    window_frames = max(int(window_duration * fps), 10)
    hop_frames = max(int(hop_duration * fps), 1)

    if window_frames >= n_frames:
        window_frames = max(n_frames // 3, 10)
        hop_frames = max(window_frames // 2, 1)
        if window_frames < 10:
            return {}

    per_window: dict[str, list[float]] = {}
    times: list[float] = []
    start = 0
    while start + window_frames <= n_frames:
        sub = emb_matrix[start : start + window_frames]
        sub = sub[np.isfinite(sub).all(axis=1)]
        times.append((start + window_frames / 2) / fps)
        geom = window_geometry(sub, k=k) if len(sub) >= 5 else {}
        for name in ("effrank", "partratio", "anisotropy", "nndist"):
            per_window.setdefault(name, []).append(geom.get(name, np.nan))
        start += hop_frames

    if not times:
        return {}

    times_arr = np.asarray(times)
    result: dict[str, float] = {}
    for name, values in per_window.items():
        result.update(_summarise(values, times_arr, prefix=f"wd_{name}"))

    # Window count is a DURATION PROXY, not a geometric property. Emitted once as
    # a diagnostic ONLY (named wd_n_windows). Classifier bundles must exclude any
    # *_n_windows column so the detector cannot shortcut on track length — see
    # run_balanced_ablation.run_classification_and_eval / _run_generator_shift_eval.
    result["wd_n_windows"] = int(len(times))

    # whole-track trajectory dynamics (single set of scalars)
    for name, val in trajectory_dynamics(emb_matrix).items():
        result[f"wd_traj_{name}"] = val

    return result
