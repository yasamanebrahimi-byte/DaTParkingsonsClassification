"""OOF-only probability calibration utilities.

The original project exposed temperature scaling only.  The legacy functions
remain intact, while the versioned artifact helpers below support the
ensemble-like calibration experiments without changing historical packages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

from ..utils.metrics import binary_metrics, safe_probabilities


DEFAULT_EPSILON = 1e-6
CALIBRATION_VERSION = 2
_METHOD_ALIASES = {
    "legacy_temperature": "temperature",
    "temperature_scaling": "temperature",
    "temperature": "temperature",
    "platt_scaling": "platt",
    "platt": "platt",
    "none": "none",
}


def _arrays(values: Iterable[float], targets: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.shape != y.shape:
        raise ValueError("calibration values and targets must have identical shapes")
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("calibration values must be a non-empty one-dimensional array")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration values and targets must be finite")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("calibration targets must be binary 0/1 values")
    return x, y


def _probabilities_from_logits(logits: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    return safe_probabilities(expit(np.asarray(logits, dtype=float)), epsilon=epsilon)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit the historical positive temperature on logits using OOF rows only."""

    logits, targets = _arrays(logits, targets)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scores = logits / temperature
        # logaddexp is stable even for very large neural logits.
        return float(np.mean(np.logaddexp(0.0, scores) - targets * scores))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    return safe_probabilities(expit(np.asarray(logits, dtype=float) / float(temperature)), epsilon=epsilon)


def fit_platt(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Fit bounded Platt parameters ``sigmoid(slope * logit + intercept)``.

    The non-negative slope keeps the calibrator monotonic, while finite bounds
    prevent tiny calibration datasets from producing numerically unstable
    probabilities.
    """

    logits, targets = _arrays(logits, targets)

    def objective(parameters: np.ndarray) -> float:
        slope, intercept = parameters
        scores = float(slope) * logits + float(intercept)
        return float(np.mean(np.logaddexp(0.0, scores) - targets * scores))

    result = minimize(
        objective,
        np.asarray([1.0, 0.0], dtype=float),
        method="L-BFGS-B",
        bounds=((1e-3, 20.0), (-20.0, 20.0)),
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Platt optimization failed: {result.message}")
    slope, intercept = (float(value) for value in result.x)
    return slope, intercept


def apply_platt(
    logits: np.ndarray,
    slope: float,
    intercept: float,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    slope = float(slope)
    intercept = float(intercept)
    if not np.isfinite(slope) or not 0.0 < slope <= 20.0:
        raise ValueError("Platt slope must be finite and in (0, 20]")
    if not np.isfinite(intercept) or not -20.0 <= intercept <= 20.0:
        raise ValueError("Platt intercept must be finite and in [-20, 20]")
    return safe_probabilities(expit(slope * np.asarray(logits, dtype=float) + intercept), epsilon=epsilon)


def probabilities_to_logits(probabilities: Iterable[float], epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Convert probabilities to finite logits using the documented epsilon."""

    return logit(safe_probabilities(probabilities, epsilon=epsilon))


def fit_platt_from_probabilities(
    probabilities: Iterable[float], targets: Iterable[float], epsilon: float = DEFAULT_EPSILON
) -> tuple[float, float]:
    return fit_platt(probabilities_to_logits(probabilities, epsilon), np.asarray(list(targets), dtype=float))


def _canonical_method(method: str | None) -> str:
    normalized = "none" if method is None else str(method).strip().lower()
    if normalized not in _METHOD_ALIASES:
        raise ValueError(f"Unknown calibration method: {method}")
    return _METHOD_ALIASES[normalized]


def apply_calibration(values: Iterable[float], artifact: Mapping, epsilon: float | None = None) -> np.ndarray:
    """Apply a versioned artifact to logits or probabilities.

    ``input_type`` is part of the artifact so the training and submission
    paths cannot silently disagree about whether a probability-to-logit
    conversion is required.
    """

    payload = dict(artifact)
    input_type = str(payload.get("input_type", "logit")).lower()
    input_type = {"mean_logit": "logit", "mean_logits": "logit", "mean_probability": "probability", "mean_probabilities": "probability"}.get(input_type, input_type)
    if input_type not in {"logit", "probability"}:
        raise ValueError("Calibration input_type must be 'logit' or 'probability'")
    clip = float(epsilon if epsilon is not None else payload.get("final_clip", payload.get("epsilon", DEFAULT_EPSILON)))
    if not 0.0 < clip < 0.5 or not np.isfinite(clip):
        raise ValueError("Calibration clipping epsilon must be finite and in (0, 0.5)")
    values_array = np.asarray(values, dtype=float)
    method = _canonical_method(payload.get("calibration_method", payload.get("method", "none")))
    enabled = bool(payload.get("enabled", True))
    if input_type == "probability":
        raw_probabilities = safe_probabilities(values_array, epsilon=clip)
        logits = probabilities_to_logits(raw_probabilities, epsilon=clip)
    else:
        logits = values_array
        raw_probabilities = _probabilities_from_logits(logits, epsilon=clip)
    if not enabled or method == "none":
        return raw_probabilities
    if method == "temperature":
        result = apply_temperature(logits, float(payload.get("temperature", 1.0)), epsilon=clip)
    elif method == "platt":
        result = apply_platt(logits, float(payload["slope"]), float(payload["intercept"]), epsilon=clip)
    else:  # pragma: no cover - _canonical_method makes this unreachable
        raise ValueError(f"Unsupported calibration method: {method}")
    return safe_probabilities(result, epsilon=clip)


def fit_calibration_artifact(
    values: Iterable[float],
    targets: Iterable[float],
    *,
    method: str = "temperature",
    input_type: str = "logit",
    ensemble_method: str = "logit_mean",
    stage: str = "after_ensemble",
    epsilon: float = DEFAULT_EPSILON,
    enabled: bool = True,
) -> dict:
    """Fit and return a self-describing production calibration artifact."""

    method = _canonical_method(method)
    input_type = {"mean_logit": "logit", "mean_logits": "logit", "mean_probability": "probability", "mean_probabilities": "probability"}.get(str(input_type).lower(), str(input_type).lower())
    if input_type not in {"logit", "probability"}:
        raise ValueError("input_type must be 'logit' or 'probability'")
    if stage not in {"before_ensemble", "after_ensemble"}:
        raise ValueError("stage must be before_ensemble or after_ensemble")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    values_array = np.asarray(values, dtype=float)
    targets_array = np.asarray(targets, dtype=float)
    if method == "none":
        _arrays(values_array, targets_array)
        parameters = {}
    else:
        calibration_logits = (
            probabilities_to_logits(values_array, epsilon=float(epsilon)) if input_type == "probability" else values_array
        )
        if method == "temperature":
            parameters = {"temperature": fit_temperature(calibration_logits, targets_array)}
        else:
            slope, intercept = fit_platt(calibration_logits, targets_array)
            parameters = {"slope": slope, "intercept": intercept}
    artifact = {
        "version": CALIBRATION_VERSION,
        "input_type": input_type,
        "ensemble_method": str(ensemble_method),
        "calibration_method": method,
        "stage": stage,
        "epsilon": float(epsilon),
        "final_clip": float(epsilon),
        "enabled": bool(enabled),
        **parameters,
    }
    return artifact


def cross_fitted_calibration(
    values: Iterable[float],
    targets: Iterable[float],
    calibration_folds: Iterable[int],
    *,
    method: str = "temperature",
    input_type: str = "logit",
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """Fit a fresh calibrator outside each validation fold and score its rows.

    This deliberately returns fold-specific parameters.  Averaging them is
    not a valid production calibrator; production parameters must be refit on
    all eligible OOF ensemble rows after a method is selected.
    """

    values_array = np.asarray(values, dtype=float)
    targets_array = np.asarray(targets, dtype=float)
    folds = np.asarray(calibration_folds)
    if values_array.shape != targets_array.shape or values_array.shape != folds.shape:
        raise ValueError("values, targets, and calibration_folds must have identical shapes")
    if values_array.ndim != 1 or len(values_array) == 0:
        raise ValueError("cross-fitted calibration requires non-empty one-dimensional inputs")
    if np.isnan(folds.astype(float)).any():
        raise ValueError("calibration_folds cannot contain missing values")
    unique_folds = pd_unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("cross-fitted calibration requires at least two folds")
    calibrated = np.empty(len(values_array), dtype=float)
    fold_parameters = []
    for fold in unique_folds:
        validation_mask = folds == fold
        training_mask = ~validation_mask
        if not training_mask.any() or not validation_mask.any():
            raise ValueError("Each calibration fold must have both training and validation rows")
        artifact = fit_calibration_artifact(
            values_array[training_mask],
            targets_array[training_mask],
            method=method,
            input_type=input_type,
            ensemble_method="cross_fitted",
            stage="after_ensemble",
            epsilon=epsilon,
        )
        validation_probabilities = apply_calibration(values_array[validation_mask], artifact, epsilon=epsilon)
        calibrated[validation_mask] = validation_probabilities
        fold_parameters.append(
            {
                "fold": fold.item() if hasattr(fold, "item") else fold,
                **artifact,
                "training_count": int(training_mask.sum()),
                "validation_count": int(validation_mask.sum()),
                "validation_metrics": binary_metrics(targets_array[validation_mask], validation_probabilities),
            }
        )
    parameter_names = ("temperature",) if _canonical_method(method) == "temperature" else ("slope", "intercept")
    stability = {
        f"{name}_std": float(np.std([float(row[name]) for row in fold_parameters]))
        for name in parameter_names
    }
    return {
        "probabilities": safe_probabilities(calibrated, epsilon=epsilon),
        "fold_parameters": fold_parameters,
        "metrics": binary_metrics(targets_array, calibrated),
        "stability": stability,
    }


def pd_unique(values: np.ndarray) -> list:
    """Small dependency-free equivalent of pandas.unique for calibration folds."""

    result = []
    for value in values.tolist():
        if value not in result:
            result.append(value)
    return result


def save_calibration(
    temperature: float,
    path: str | Path,
    stage: str = "before_ensemble",
    enabled: bool = True,
    raw_log_loss: float | None = None,
    calibrated_log_loss: float | None = None,
) -> None:
    """Write the original temperature JSON shape for legacy compatibility."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "temperature_scaling",
        "temperature": float(temperature),
        "stage": stage,
        "enabled": bool(enabled),
    }
    if raw_log_loss is not None:
        payload["raw_log_loss"] = float(raw_log_loss)
    if calibrated_log_loss is not None:
        payload["calibrated_log_loss"] = float(calibrated_log_loss)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_calibration_artifact(artifact: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    payload.setdefault("version", CALIBRATION_VERSION)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_calibration_artifact(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration artifact must contain a JSON object")
    # A missing method is the historical temperature format.
    if "calibration_method" not in payload and "method" not in payload:
        payload["method"] = "temperature_scaling"
    return payload


def load_calibration(path: str | Path) -> float:
    """Historical loader: return only the legacy temperature value."""

    payload = load_calibration_artifact(path)
    return float(payload.get("temperature", 1.0))


# Friendly aliases used by scripts and downstream experiments.
fit_platt_scaling = fit_platt
apply_platt_scaling = apply_platt
cross_fit_calibration = cross_fitted_calibration
