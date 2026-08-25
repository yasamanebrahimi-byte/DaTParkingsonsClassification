"""OOF ensemble selection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..utils.metrics import safe_probabilities


def optimize_weights(targets: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=float)
    matrix = np.asarray(prediction_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(targets):
        raise ValueError("prediction_matrix must be [n_samples, n_models]")

    def objective(weights: np.ndarray) -> float:
        normalized = weights / max(weights.sum(), 1e-12)
        probabilities = safe_probabilities(matrix @ normalized)
        return float(-np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)))

    result = minimize(objective, np.ones(matrix.shape[1]), bounds=[(0.0, 1.0)] * matrix.shape[1], constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0})
    if not result.success:
        return np.ones(matrix.shape[1]) / matrix.shape[1]
    return result.x / result.x.sum()

