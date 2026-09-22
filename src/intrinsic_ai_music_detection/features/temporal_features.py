"""Temporal ID time-series analysis: transition entropy, HMM, change-point detection.

Four analysis paths operating on a 1-D sequence of per-window ID estimates:

  1. Summary statistics (already in run_balanced_ablation.py)
       slope, std, range, first_last

  2. Transition-matrix entropy  (this module)
       k-means cluster window IDs into K ``ID regimes`` → build K×K transition
       matrix → compute Shannon entropy.  AI generators with finite context
       windows are expected to revisit fewer regimes (lower entropy) and to
       transition more predictably.

  3. State-space / HMM (this module)
       Fit a 1-D Gaussian HMM to the ID sequence; features:
         - log-likelihood per window (higher = more predictable given learned
           Markov structure)
         - number of distinct states (estimated via AIC/BIC sweep)
         - dominant state dwell-time statistics
         - Viterbi-decoded state sequence entropy

  4. Change-point detection (this module)
       ``ruptures`` Pelt or BinSeg algorithm on the ID sequence; features:
         - n_changepoints: total detected change points
         - mean_segment_length: average run length between changes
         - first_change_relative: position of first change (normalised 0–1)
         - changepoint_density: change points per window

All functions accept a 1-D float array of window IDs and return a dict of
scalar features that can be concatenated into the ablation feature table.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Suppress hmmlearn's logging-level warnings (it uses logging.warning, not
# warnings.warn, so warnings.catch_warnings() has no effect on them).
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# ---- defaults ----
N_REGIME_CLUSTERS: int = 4  # number of ID regimes for transition entropy
HMM_MAX_STATES: int = 6  # max states in BIC sweep for Gaussian HMM
HMM_MIN_STATES: int = 2
CHANGEPOINT_MIN_SIZE: int = 2  # minimum segment size for ruptures


# ---------------------------------------------------------------------------
# Helper: clean array
# ---------------------------------------------------------------------------


def _clean(seq: list | npt.NDArray) -> npt.NDArray[np.float64]:
    arr = np.asarray(seq, dtype=float)
    return arr[np.isfinite(arr)]


# ---------------------------------------------------------------------------
# 1. Transition-matrix entropy
# ---------------------------------------------------------------------------


def transition_entropy(
    window_ids: list | npt.NDArray,
    n_clusters: int = N_REGIME_CLUSTERS,
) -> dict[str, float]:
    """Compute transition-matrix entropy of the ID sequence.

    Steps
    -----
    1. Quantise the continuous ID values into ``n_clusters`` discrete regimes
       via 1-D k-means (or quantiles for robustness).
    2. Build a normalised K×K transition matrix T where T[i,j] is the empirical
       probability of transitioning from regime i to regime j.
    3. Compute the Shannon entropy of T (flattened) = - ∑ T log T.
    4. Also compute the per-row conditional entropy (stationary-dist-weighted mean).

    A deterministic/repetitive sequence produces low entropy; a diverse,
    non-repeating one produces high entropy.  AI generators with finite context
    are expected to produce lower entropy (more predictable regime cycling).
    """
    arr = _clean(window_ids)
    if len(arr) < max(n_clusters + 1, 4):
        return {}

    # Quantise into regimes using quantile-based bins (robust to outliers)
    edges = np.quantile(arr, np.linspace(0, 1, n_clusters + 1))
    edges[0] -= 1e-6  # ensure first bin captures minimum
    edges[-1] += 1e-6
    labels = np.digitize(arr, edges[:-1]) - 1
    labels = np.clip(labels, 0, n_clusters - 1)

    # Build transition matrix
    T = np.zeros((n_clusters, n_clusters), dtype=float)
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1

    # Normalise rows
    row_sums = T.sum(axis=1, keepdims=True) + 1e-12
    T_norm = T / row_sums

    # Stationary distribution (from empirical state visits)
    state_counts = np.bincount(labels, minlength=n_clusters).astype(float)
    state_dist = state_counts / (state_counts.sum() + 1e-12)

    # Shannon entropy of transition matrix (flat)
    flat = T_norm.ravel()
    flat = flat[flat > 1e-12]
    entropy_flat = float(-np.sum(flat * np.log2(flat)))

    # Conditional entropy H(X_{t+1} | X_t)
    cond_entropy = 0.0
    for i in range(n_clusters):
        row = T_norm[i]
        nonzero = row[row > 1e-12]
        h_i = float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) else 0.0
        cond_entropy += state_dist[i] * h_i

    # Self-transition probability (how often does the state stay the same)
    self_trans = float(np.trace(T_norm) / n_clusters)

    # Number of distinct transitions observed
    n_distinct_transitions = int((T > 0).sum())

    return {
        "trans_entropy_flat": entropy_flat,
        "trans_entropy_conditional": cond_entropy,
        "trans_self_prob": self_trans,
        "trans_n_distinct": n_distinct_transitions,
        "trans_n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# 2. HMM / state-space model
# ---------------------------------------------------------------------------


def hmm_features(
    window_ids: list | npt.NDArray,
    min_states: int = HMM_MIN_STATES,
    max_states: int = HMM_MAX_STATES,
) -> dict[str, float]:
    """Fit a Gaussian HMM to the ID sequence and extract model features.

    Requires ``hmmlearn``.

    Features extracted
    ------------------
    hmm_n_states_bic      Optimal number of Gaussian HMM states (BIC criterion)
    hmm_log_likelihood    Log-likelihood of best model per window
    hmm_bic               BIC of best model (lower = simpler / more predictable)
    hmm_viterbi_entropy   Entropy of the Viterbi-decoded state sequence
    hmm_dwell_mean        Mean dwell time in each decoded state (windows)
    hmm_dwell_cv          Coefficient of variation of dwell times (regularity)
    """
    arr = _clean(window_ids)
    if len(arr) < max(min_states * 2 + 2, 8):
        return {}

    try:
        from hmmlearn import hmm  # noqa: PLC0415
    except ImportError:
        logger.debug("hmmlearn not installed — HMM features unavailable. pip install hmmlearn")
        return {}

    X = arr.reshape(-1, 1)
    best_bic = np.inf
    best_n = min_states
    best_model = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in range(min_states, min(max_states + 1, len(arr) // 2)):
            try:
                model = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                    verbose=False,
                )
                # Skip models with more free params than data points.
                # Rule: k_params = n^2 (transition) + 2n (means+vars for 1-D diag).
                # Fitting k >= T is numerically degenerate and biases BIC selection.
                k_params = n * n + 2 * n
                if k_params >= len(arr):
                    continue
                model.fit(X)
                log_lik = model.score(X)
                bic = -2 * log_lik + k_params * np.log(len(arr))
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        return {}

    try:
        log_lik = best_model.score(X)
        _, viterbi_states = best_model.decode(X, algorithm="viterbi")

        # Viterbi sequence entropy
        state_counts = np.bincount(viterbi_states, minlength=best_n)
        state_probs = state_counts / (state_counts.sum() + 1e-12)
        viterbi_entropy = float(-np.sum(state_probs[state_probs > 1e-12] * np.log2(state_probs[state_probs > 1e-12])))

        # Dwell time statistics
        dwell_times = []
        if len(viterbi_states) > 0:
            current_state = viterbi_states[0]
            count = 1
            for s in viterbi_states[1:]:
                if s == current_state:
                    count += 1
                else:
                    dwell_times.append(count)
                    current_state = s
                    count = 1
            dwell_times.append(count)

        dwell_arr = np.array(dwell_times, dtype=float)
        dwell_mean = float(np.mean(dwell_arr)) if len(dwell_arr) else float("nan")
        dwell_cv = float(np.std(dwell_arr) / (np.mean(dwell_arr) + 1e-9)) if len(dwell_arr) > 1 else float("nan")

        return {
            "hmm_n_states_bic": best_n,
            "hmm_log_likelihood": float(log_lik / len(arr)),
            "hmm_bic": float(best_bic),
            "hmm_viterbi_entropy": viterbi_entropy,
            "hmm_dwell_mean": dwell_mean,
            "hmm_dwell_cv": dwell_cv,
        }
    except Exception as exc:
        logger.debug("HMM feature extraction failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# 3. Change-point detection
# ---------------------------------------------------------------------------


def changepoint_features(
    window_ids: list | npt.NDArray,
    min_size: int = CHANGEPOINT_MIN_SIZE,
    penalty: float | None = None,
) -> dict[str, float]:
    """Detect structural change points in the ID trajectory via ``ruptures``.

    Uses the Pelt algorithm (optimal, linear-complexity) with an RBF cost
    function.  The penalty is chosen as ``penalty = log(n)`` (BIC-like) if
    not specified.

    Features
    --------
    cp_n_changepoints       Number of detected change points
    cp_mean_segment_len     Mean length of segments between change points (windows)
    cp_first_change_rel     Relative position of first change point (0 = start, 1 = end)
    cp_last_change_rel      Relative position of last change point
    cp_density              Change points per window
    """
    arr = _clean(window_ids)
    if len(arr) < max(2 * min_size + 1, 6):
        return {}

    try:
        import ruptures as rpt  # noqa: PLC0415
    except ImportError:
        logger.debug("ruptures not installed — changepoint features unavailable. pip install ruptures")
        return {}

    try:
        signal = arr.reshape(-1, 1)
        n = len(arr)
        pen = penalty if penalty is not None else np.log(n) * np.var(arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
            algo.fit(signal)
            breakpoints = algo.predict(pen=pen)

        # ruptures returns breakpoints as indices including the final index (= n)
        if n in breakpoints:
            breakpoints = breakpoints[:-1]

        n_cp = len(breakpoints)
        if n_cp == 0:
            return {
                "cp_n_changepoints": 0,
                "cp_mean_segment_len": float(n),
                "cp_first_change_rel": float("nan"),
                "cp_last_change_rel": float("nan"),
                "cp_density": 0.0,
            }

        # Segment lengths
        edges = [0] + list(breakpoints) + [n]
        seg_lengths = np.diff(edges).astype(float)

        return {
            "cp_n_changepoints": n_cp,
            "cp_mean_segment_len": float(np.mean(seg_lengths)),
            "cp_first_change_rel": float(breakpoints[0] / n),
            "cp_last_change_rel": float(breakpoints[-1] / n),
            "cp_density": float(n_cp / n),
        }
    except Exception as exc:
        logger.debug("Change-point detection failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Combined: all temporal features from a window ID sequence
# ---------------------------------------------------------------------------


def all_temporal_features(
    window_ids: list | npt.NDArray,
    n_clusters: int = N_REGIME_CLUSTERS,
    run_hmm: bool = True,
    run_changepoint: bool = True,
) -> dict[str, float]:
    """Compute all temporal analysis features from a window ID sequence.

    Parameters
    ----------
    window_ids
        1-D sequence of per-window ID estimates (NaN values are dropped).
    n_clusters
        Number of ID regimes for transition entropy.
    run_hmm
        Whether to run HMM fitting (slower; requires hmmlearn).
    run_changepoint
        Whether to run change-point detection (requires ruptures).

    Returns
    -------
    dict mapping feature name → scalar value
    """
    result: dict[str, float] = {}
    result.update(transition_entropy(window_ids, n_clusters=n_clusters))
    if run_hmm:
        result.update(hmm_features(window_ids))
    if run_changepoint:
        result.update(changepoint_features(window_ids))
    return result


# ---------------------------------------------------------------------------
# Feature name registry (for downstream column selection)
# ---------------------------------------------------------------------------

TRANSITION_FEATURE_NAMES = [
    "trans_entropy_flat",
    "trans_entropy_conditional",
    "trans_self_prob",
    "trans_n_distinct",
]

HMM_FEATURE_NAMES = [
    "hmm_n_states_bic",
    "hmm_log_likelihood",
    "hmm_bic",
    "hmm_viterbi_entropy",
    "hmm_dwell_mean",
    "hmm_dwell_cv",
]

CHANGEPOINT_FEATURE_NAMES = [
    "cp_n_changepoints",
    "cp_mean_segment_len",
    "cp_first_change_rel",
    "cp_last_change_rel",
    "cp_density",
]

ALL_TEMPORAL_EXTENSION_NAMES = TRANSITION_FEATURE_NAMES + HMM_FEATURE_NAMES + CHANGEPOINT_FEATURE_NAMES
