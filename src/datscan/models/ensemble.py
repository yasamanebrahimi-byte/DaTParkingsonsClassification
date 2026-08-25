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

