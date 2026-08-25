"""Probability-first binary classification metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def safe_probabilities(probabilities: Iterable[float], epsilon: float = 1e-6) -> np.ndarray:
    if np.isscalar(probabilities):
        values = np.asarray(probabilities, dtype=np.float64)
    elif isinstance(probabilities, np.ndarray):
        values = np.asarray(probabilities, dtype=np.float64)
    else:
        values = np.asarray(list(probabilities), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain non-finite values")
    return np.clip(values, epsilon, 1.0 - epsilon)


def expected_calibration_error(targets: Iterable[float], probabilities: Iterable[float], bins: int = 10) -> float:
    y = np.asarray(list(targets), dtype=float)
    p = safe_probabilities(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1])
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def binary_metrics(targets: Iterable[float], probabilities: Iterable[float]) -> Dict[str, float]:
    y = np.asarray(list(targets), dtype=float)
    p = safe_probabilities(probabilities)
    result = {
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0.0, 1.0])),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "sensitivity": float(((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "specificity": float(((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "ece": expected_calibration_error(y, p),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result
