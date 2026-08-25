"""Small dependency-light ensemble/calibration math used by submission inference.

This module intentionally mirrors ``datscan.training.calibrate`` and
``datscan.training.ensemble`` without importing the training package, because
the submission ZIP contains only ``submission/``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


DEFAULT_EPSILON = 1e-6


def clip_probabilities(values, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 0.5 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and in (0, 0.5)")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    return np.clip(values, epsilon, 1.0 - epsilon)


def combine_logits(logits, method: str = "logit_mean", weights=None) -> tuple[np.ndarray, str]:
    """Return the ensemble input and whether it is a logit or probability."""

    matrix = np.asarray(logits, dtype=float)
    if matrix.ndim != 1 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("logits must be a finite non-empty one-dimensional array")
    normalized = str(method).lower()
    if normalized in {"logit_mean", "mean_logits"}:
        return np.asarray(matrix.mean(), dtype=float), "logit"
    probabilities = expit(matrix)
    if normalized in {"probability_mean", "mean_probabilities"}:
        return np.asarray(clip_probabilities(probabilities.mean()), dtype=float), "probability"
    if normalized in {"median_probability", "median_probabilities"}:
        if weights is not None:
            raise ValueError("median probability aggregation does not accept weights")
        return np.asarray(clip_probabilities(np.median(probabilities)), dtype=float), "probability"
    if normalized in {"weighted_probability_mean", "weighted_probabilities"}:
        if weights is None:
            raise ValueError("weighted probability mean requires weights")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != matrix.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("fold weights must be finite, non-negative, and match the number of models")
        weights = weights / weights.sum()
        return np.asarray(clip_probabilities(np.sum(weights * probabilities)), dtype=float), "probability"
    raise ValueError(f"Unknown ensemble method: {method}")


def apply_calibration(values, artifact: dict) -> np.ndarray:
    """Apply legacy temperature, Platt, or no calibration to one or many values."""

    payload = dict(artifact or {})
    input_type = str(payload.get("input_type", "logit")).lower()
    input_type = {"mean_logit": "logit", "mean_logits": "logit", "mean_probability": "probability", "mean_probabilities": "probability"}.get(input_type, input_type)
    epsilon = float(payload.get("final_clip", payload.get("epsilon", DEFAULT_EPSILON)))
    if input_type == "probability":
        probabilities = clip_probabilities(values, epsilon)
        logits = logit(probabilities)
    elif input_type == "logit":
        logits = np.asarray(values, dtype=float)
        if not np.isfinite(logits).all():
            raise ValueError("logits contain non-finite values")
        probabilities = clip_probabilities(expit(logits), epsilon)
    else:
        raise ValueError("Calibration input_type must be logit or probability")
    method = str(payload.get("calibration_method", payload.get("method", "none"))).lower()
    enabled = bool(payload.get("enabled", True))
    if not enabled or method == "none":
        return probabilities
    if method in {"temperature", "temperature_scaling", "legacy_temperature"}:
        temperature = float(payload.get("temperature", 1.0))
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("Calibration temperature must be finite and positive")
        return clip_probabilities(expit(logits / temperature), epsilon)
    if method in {"platt", "platt_scaling"}:
        slope = float(payload["slope"])
        intercept = float(payload["intercept"])
        if not 0.0 < slope <= 20.0 or not np.isfinite(slope):
            raise ValueError("Platt slope must be finite and in (0, 20]")
        if not -20.0 <= intercept <= 20.0 or not np.isfinite(intercept):
            raise ValueError("Platt intercept must be finite and in [-20, 20]")
        return clip_probabilities(expit(slope * logits + intercept), epsilon)
    raise ValueError(f"Unknown calibration method: {method}")
