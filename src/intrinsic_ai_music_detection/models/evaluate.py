"""Evaluation utilities for AI music detection experiments.

Provides statistical tests, effect sizes, and visualisation-ready
result tables for comparing intrinsic dimension distributions between
human-made and AI-generated music.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class DistributionComparison:
    """Statistical comparison of ID distributions between two groups."""

    human_mean: float
    human_std: float
    ai_mean: float
    ai_std: float
    mann_whitney_u: float
    mann_whitney_p: float
    cohens_d: float
    ks_statistic: float
    ks_p: float


def cohens_d(group1: npt.NDArray[np.float64], group2: npt.NDArray[np.float64]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compare_distributions(
    human_ids: npt.NDArray[np.float64],
    ai_ids: npt.NDArray[np.float64],
) -> DistributionComparison:
    """Run statistical tests comparing human vs AI intrinsic dimensions.

    Tests performed:
    - Mann-Whitney U: non-parametric rank test
    - Cohen's d: standardised effect size
    - Kolmogorov-Smirnov: distribution difference
    """
    u_stat, u_p = stats.mannwhitneyu(human_ids, ai_ids, alternative="two-sided")
    ks_stat, ks_p = stats.ks_2samp(human_ids, ai_ids)
    d = cohens_d(human_ids, ai_ids)

    result = DistributionComparison(
        human_mean=float(np.mean(human_ids)),
        human_std=float(np.std(human_ids, ddof=1)),
        ai_mean=float(np.mean(ai_ids)),
        ai_std=float(np.std(ai_ids, ddof=1)),
        mann_whitney_u=float(u_stat),
        mann_whitney_p=float(u_p),
        cohens_d=d,
        ks_statistic=float(ks_stat),
        ks_p=float(ks_p),
    )

    logger.info(
        "Human ID: %.2f ± %.2f | AI ID: %.2f ± %.2f | d=%.3f | U-test p=%.2e",
        result.human_mean,
        result.human_std,
        result.ai_mean,
        result.ai_std,
        result.cohens_d,
        result.mann_whitney_p,
    )

    return result


def compute_binary_metrics(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
    y_prob: npt.NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : ground truth labels (0 = human, 1 = AI)
    y_pred : predicted labels
    y_prob : predicted probabilities for class 1 (for AUC-ROC)
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["eer"] = equal_error_rate(y_true, y_prob)

    return metrics


def compute_roc_curve(
    y_true: npt.NDArray[np.int64],
    y_prob: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute ROC curve arrays for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def equal_error_rate(
    y_true: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
) -> float:
    """Equal Error Rate (EER) — the threshold-independent operating point where the
    false-acceptance rate equals the false-rejection rate.

    EER is the primary metric used by MusicDET (ICML 2026) and the audio
    anti-spoofing literature, reported alongside ROC-AUC so results are
    comparable to that body of work.

    Parameters
    ----------
    y_true : ground-truth labels (1 = AI/positive, 0 = real/negative).
    scores : detection scores where *higher* means *more likely AI*. (For a
        one-class anomaly score such as Mahalanobis distance or a flow's negative
        log-likelihood, higher = more anomalous = more likely AI, which matches.)

    Returns
    -------
    EER in [0, 1]; ``nan`` if only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true, scores = y_true[finite], scores[finite]
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    # Find the crossover point between FPR and FNR.
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def auc_and_eer(
    y_true: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Convenience wrapper returning ``(roc_auc, eer)`` for a score vector where
    higher means more likely AI. Returns ``(nan, nan)`` if a single class."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true, scores = y_true[finite], scores[finite]
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, scores)), equal_error_rate(y_true, scores)


def bootstrap_auc(
    y_true: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute AUC with percentile bootstrap confidence interval.

    Parameters
    ----------
    y_true  : ground-truth labels (1 = AI/fake, 0 = real).
    scores  : detection scores (higher = more likely AI).
    n_boot  : number of bootstrap resamples.
    seed    : random seed for reproducibility.
    alpha   : significance level; returns (1-alpha) CI. Default 0.05 -> 95% CI.

    Returns
    -------
    dict with keys ``auc``, ``ci_lo``, ``ci_hi``, ``se``, ``n_boot_valid``.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true, scores = y_true[finite], scores[finite]
    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        return {"auc": nan, "ci_lo": nan, "ci_hi": nan, "se": nan, "n_boot_valid": 0}

    point_auc = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_true[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boot_aucs.append(float(roc_auc_score(yb, sb)))
        except Exception:
            continue

    boot_arr = np.array(boot_aucs)
    return {
        "auc": point_auc,
        "ci_lo": float(np.percentile(boot_arr, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot_arr, 100 * (1 - alpha / 2))),
        "se": float(boot_arr.std()),
        "n_boot_valid": len(boot_aucs),
    }


def bootstrap_eer(
    y_true: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute EER with percentile bootstrap confidence interval."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true, scores = y_true[finite], scores[finite]
    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        return {"eer": nan, "ci_lo": nan, "ci_hi": nan, "se": nan, "n_boot_valid": 0}

    point_eer = equal_error_rate(y_true, scores)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_eers: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_true[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boot_eers.append(equal_error_rate(yb, sb))
        except Exception:
            continue

    boot_arr = np.array(boot_eers)
    return {
        "eer": point_eer,
        "ci_lo": float(np.percentile(boot_arr, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot_arr, 100 * (1 - alpha / 2))),
        "se": float(boot_arr.std()),
        "n_boot_valid": len(boot_eers),
    }


def delong_test(
    y_true: npt.NDArray[np.int64],
    scores_a: npt.NDArray[np.float64],
    scores_b: npt.NDArray[np.float64],
) -> dict[str, float]:
    """DeLong et al. (1988) paired AUC comparison test.

    Tests H0: AUC(scores_a) == AUC(scores_b) on the SAME set of samples.

    Returns
    -------
    dict with ``auc_a``, ``auc_b``, ``delta_auc``, ``z``, ``p_value``.
    ``p_value`` is two-sided.

    Reference: DeLong, DeLong, Clarke-Pearson (1988), Biometrics 44:837-845.
    Implementation follows the structural component decomposition.
    """
    y_true = np.asarray(y_true).astype(int)
    s_a = np.asarray(scores_a, dtype=float)
    s_b = np.asarray(scores_b, dtype=float)

    finite = np.isfinite(s_a) & np.isfinite(s_b)
    y_true, s_a, s_b = y_true[finite], s_a[finite], s_b[finite]

    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        return {"auc_a": nan, "auc_b": nan, "delta_auc": nan, "z": nan, "p_value": nan}

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    def _midrank(x: np.ndarray) -> np.ndarray:
        """Tie-averaged ranks (1-based), O(n log n) — Sun & Xu (2014) helper."""
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        n = len(x)
        ranks_sorted = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j < n - 1 and xs[j + 1] == xs[i]:
                j += 1
            ranks_sorted[i : j + 1] = 0.5 * (i + j) + 1.0  # average rank over the tie block
            i = j + 1
        ranks = np.empty(n, dtype=float)
        ranks[order] = ranks_sorted
        return ranks

    def _structural_components(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample structural components (V10 and V01).

        Computed via the fast midrank identity (Sun & Xu, IEEE SPL 2014) instead
        of the O(n_pos*n_neg) double loop the original implementation used —
        the naive form takes hours at SONICS scale (~6k x 47k) and looked like a
        hang. Results are identical up to floating point, ties included.
        """
        sp = scores[pos_idx]
        sn = scores[neg_idx]
        r_all = _midrank(np.concatenate([sp, sn]))
        r_pos_within = _midrank(sp)
        r_neg_within = _midrank(sn)
        # V10[i] = (rank of sp[i] among all − rank among positives) / n_neg
        v10 = (r_all[:n_pos] - r_pos_within) / n_neg
        # V01[j] = 1 − (rank of sn[j] among all − rank among negatives) / n_pos
        v01 = 1.0 - (r_all[n_pos:] - r_neg_within) / n_pos
        return v10, v01

    v10_a, v01_a = _structural_components(s_a)
    v10_b, v01_b = _structural_components(s_b)

    auc_a = float(v10_a.mean())
    auc_b = float(v10_b.mean())

    # Variance-covariance matrix of (AUC_a, AUC_b)
    s10 = np.cov(np.vstack([v10_a, v10_b])) / n_pos  # 2x2
    s01 = np.cov(np.vstack([v01_a, v01_b])) / n_neg  # 2x2

    var_diff = s10[0, 0] + s10[1, 1] - 2 * s10[0, 1] + s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]
    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "delta_auc": auc_a - auc_b, "z": float("nan"), "p_value": float("nan")}

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = float(2 * stats.norm.sf(abs(z)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": auc_a - auc_b,
        "z": float(z),
        "p_value": p_value,
    }


def per_fold_variance(
    fold_aucs: list[float],
    fold_eers: list[float],
) -> dict[str, float]:
    """Compute mean, std, and 95% CI from per-fold AUC/EER values.

    Input: list of per-fold AUC values (one per LOGO fold).
    """
    a = np.asarray([x for x in fold_aucs if np.isfinite(x)])
    e = np.asarray([x for x in fold_eers if np.isfinite(x)])
    n = len(a)
    ci_half = 1.96 * a.std() / np.sqrt(max(n, 1))
    return {
        "n_folds": n,
        "auc_mean": float(a.mean()) if n > 0 else float("nan"),
        "auc_std": float(a.std()) if n > 0 else float("nan"),
        "auc_ci_lo": float(a.mean() - ci_half) if n > 0 else float("nan"),
        "auc_ci_hi": float(a.mean() + ci_half) if n > 0 else float("nan"),
        "eer_mean": float(e.mean()) if len(e) > 0 else float("nan"),
        "eer_std": float(e.std()) if len(e) > 0 else float("nan"),
    }


def format_results_table(
    results: list[dict[str, float | str]],
    caption: str = "Results",
) -> str:
    """Format a list of result dicts as a markdown table.

    Each dict should contain 'method', 'accuracy', 'f1', 'auc_roc', etc.
    """
    if not results:
        return ""

    keys = list(results[0].keys())
    header = "| " + " | ".join(keys) + " |"
    separator = "| " + " | ".join("---" for _ in keys) + " |"
    rows = []
    for r in results:
        row = "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in (r[k] for k in keys)) + " |"
        rows.append(row)

    table = f"**{caption}**\n\n{header}\n{separator}\n" + "\n".join(rows)
    return table


def generate_report(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> str:
    """Generate a sklearn classification report string."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["Human", "AI-Generated"],
        digits=3,
    )
