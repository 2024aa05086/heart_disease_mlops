"""Model training utilities.

This module defines functions to build and train classification models
for the heart disease dataset. It supports logistic regression and
random forest classifiers and provides functionality for hyperparameter
tuning and cross‑validation. Metrics such as accuracy, precision,
recall, and ROC‑AUC are computed for evaluation.

Models are trained on the preprocessed feature matrix produced by
data.preprocess_features().
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score


def build_models() -> Dict[str, Any]:
    """Define the candidate models with reasonable default hyperparameters.

    Returns:
        A dictionary mapping model names to estimator instances.
    """
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    return models


def evaluate_models(models: Dict[str, Any], X: Any, y: pd.Series, cv_splits: int = 5) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple models using cross‑validation.

    Args:
        models: A dictionary of model name to estimator.
        X: Preprocessed feature matrix (numpy array or sparse matrix).
        y: Target vector.
        cv_splits: Number of cross‑validation folds.

    Returns:
        A dictionary mapping model names to metric dicts.
    """
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "roc_auc": make_scorer(roc_auc_score),
    }
    results: Dict[str, Dict[str, float]] = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
        # Compute mean of each metric across folds
        metrics_mean = {metric: np.mean(scores) for metric, scores in cv_scores.items() if metric.startswith("test_")}
        # Remove the 'test_' prefix
        metrics_mean = {metric.replace("test_", ""): value for metric, value in metrics_mean.items()}
        results[name] = metrics_mean
    return results


def train_final_model(model: Any, X: Any, y: pd.Series) -> Any:
    """Train a model on the full dataset.

    Args:
        model: The estimator to train.
        X: Preprocessed feature matrix.
        y: Target vector.

    Returns:
        The fitted model.
    """
    model.fit(X, y)
    return model