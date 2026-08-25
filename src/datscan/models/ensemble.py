"""OOF-validated probability and logit ensemble helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..utils.metrics import safe_probabilities


def average_probabilities(predictions: Sequence[Iterable[float]], weights: Sequence[float] | None = None) -> np.ndarray:
    values = np.asarray([safe_probabilities(p) for p in predictions], dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected one prediction vector per ensemble member")
    if weights is None:
        weights = np.ones(values.shape[0], dtype=float) / values.shape[0]
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (values.shape[0],) or weights_array.sum() <= 0:
        raise ValueError("Ensemble weights must match members and have positive sum")
    weights_array = weights_array / weights_array.sum()
    return safe_probabilities((values * weights_array[:, None]).sum(axis=0))


def aggregate_member_logits(
    logits: Sequence[Iterable[float]] | np.ndarray,
    method: str = "probability_mean",
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Aggregate a [members, samples] logit matrix for training/inference code."""

    matrix = np.asarray(logits, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("logits must be a finite non-empty [members, samples] matrix")
    normalized = str(method).lower()
    probabilities = 1.0 / (1.0 + np.exp(-matrix))
    if normalized in {"logit_mean", "mean_logits"}:
        return safe_probabilities(1.0 / (1.0 + np.exp(-matrix.mean(axis=0))))
    if normalized in {"probability_mean", "mean_probability", "mean_probabilities"}:
        return average_probabilities(probabilities, weights=weights)
    if normalized in {"median_probability", "median_probabilities"}:
        if weights is not None:
            raise ValueError("median probability aggregation does not accept weights")
        return safe_probabilities(np.median(probabilities, axis=0))
    if normalized in {"weighted_probability_mean", "weighted_probabilities"}:
        return average_probabilities(probabilities, weights=weights)
    raise ValueError(f"Unknown ensemble aggregation method: {method}")
