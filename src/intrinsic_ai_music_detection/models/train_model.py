"""Classification models for AI-generated music detection.

Trains and evaluates binary classifiers (human vs AI) using intrinsic
dimension features, fakeprint features, or combinations thereof.

Classifier options
------------------
- Logistic Regression (default): interpretable, fast, good baseline
- SVM (RBF kernel): captures non-linear decision boundaries
- Threshold-based: simple ID threshold, no training needed

Feature configurations
----------------------
- Single-ID: one estimator × one embedding model → 1 feature
- Multi-ID: all estimators × all embeddings → N features
- Fakeprint: spectral residual vector → M features
- Hybrid: concatenation of ID features + fakeprint features
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


ClassifierType = Literal["logistic_regression", "svm"]


@dataclass
class ClassificationResult:
    """Result container for a single experiment run."""

    classifier: str
    feature_set: str
    cv_accuracy: float
    cv_precision: float
    cv_recall: float
    cv_f1: float
    cv_auc: float
    cv_scores: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    model: Pipeline | None = None


def build_pipeline(
    classifier_type: ClassifierType = "logistic_regression",
    random_state: int = 42,
) -> Pipeline:
    """Build a sklearn Pipeline with standardisation + classifier.

    Parameters
    ----------
    classifier_type : ``'logistic_regression'`` or ``'svm'``
    random_state : random seed for reproducibility
    """
    if classifier_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif classifier_type == "svm":
        clf = SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type!r}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_and_evaluate(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    classifier_type: ClassifierType = "logistic_regression",
    feature_set_name: str = "unknown",
    n_folds: int = 5,
    random_state: int = 42,
) -> ClassificationResult:
    """Run stratified k-fold cross-validation and return metrics.

    Parameters
    ----------
    X : feature matrix of shape ``[n_samples, n_features]``
    y : binary labels (0 = human, 1 = AI)
    classifier_type : which classifier to use
    feature_set_name : descriptive label for logging
    n_folds : number of CV folds
    random_state : random seed
    """
    pipeline = build_pipeline(classifier_type, random_state)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_estimator=False,
        n_jobs=-1,
    )

    result = ClassificationResult(
        classifier=classifier_type,
        feature_set=feature_set_name,
        cv_accuracy=float(np.mean(cv_results["test_accuracy"])),
        cv_precision=float(np.mean(cv_results["test_precision"])),
        cv_recall=float(np.mean(cv_results["test_recall"])),
        cv_f1=float(np.mean(cv_results["test_f1"])),
        cv_auc=float(np.mean(cv_results["test_roc_auc"])),
        cv_scores={k: v for k, v in cv_results.items() if k.startswith("test_")},
    )

    logger.info(
        "%-30s | %s | Acc=%.3f  F1=%.3f  AUC=%.3f",
        feature_set_name,
        classifier_type,
        result.cv_accuracy,
        result.cv_f1,
        result.cv_auc,
    )

    # Refit on full data for the final model
    pipeline.fit(X, y)
    result.model = pipeline

    return result


def train_threshold_classifier(
    id_values: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    direction: Literal["lower", "higher"] = "lower",
) -> tuple[float, float]:
    """Find the optimal threshold for a single-ID feature.

    Searches for the threshold that maximises accuracy, given that
    AI-generated music typically has *lower* intrinsic dimension.

    Parameters
    ----------
    id_values : 1-D array of intrinsic dimension values
    y : binary labels (0 = human, 1 = AI)
    direction : ``'lower'`` if AI tracks have lower ID; ``'higher'`` otherwise

    Returns
    -------
    (threshold, accuracy) — the best threshold and its accuracy
    """
    sorted_vals = np.sort(np.unique(id_values))
    best_thresh = float("nan")
    best_acc = 0.0

    for t in sorted_vals:
        if direction == "lower":
            preds = (id_values < t).astype(int)
        else:
            preds = (id_values > t).astype(int)

        acc = float(np.mean(preds == y))
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(t)

    return best_thresh, best_acc


def save_model(model: Pipeline, path: str | Path) -> None:
    """Serialise a trained pipeline to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> Pipeline:
    """Deserialise a trained pipeline from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted model files only
