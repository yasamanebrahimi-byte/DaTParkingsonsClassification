"""OOF ensemble selection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr

from ..utils.metrics import binary_metrics, safe_probabilities


ENSEMBLE_METHODS = {
    "logit_mean",
    "mean_logits",
    "probability_mean",
    "mean_probabilities",
    "weighted_probability_mean",
    "weighted_probabilities",
}


def _member_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [n_members, n_samples] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def normalize_weights(weights: np.ndarray | list[float], n_members: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (n_members,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Ensemble weights must be finite, non-negative, and match the number of members")
    return values / values.sum()


def mean_logits(logits: np.ndarray) -> np.ndarray:
    """Mean raw model logits, then leave sigmoid application to the caller."""

    return _member_matrix(logits, "logits").mean(axis=0)


def mean_probabilities(logits: np.ndarray, weights: np.ndarray | list[float] | None = None) -> np.ndarray:
    """Mean individual sigmoid probabilities, never sigmoid(mean logits)."""

    matrix = _member_matrix(logits, "logits")
    probabilities = expit(matrix)
    normalized = np.ones(matrix.shape[0], dtype=float) / matrix.shape[0] if weights is None else normalize_weights(weights, matrix.shape[0])
    return safe_probabilities((probabilities * normalized[:, None]).sum(axis=0))


def weighted_probability_mean(probabilities: np.ndarray, weights: np.ndarray | list[float]) -> np.ndarray:
    """Weighted probability mean using only weights learned from aligned OOF rows."""

    matrix = _member_matrix(probabilities, "probabilities")
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Probability ensemble members must be in [0, 1]")
    normalized = normalize_weights(weights, matrix.shape[0])
    return safe_probabilities((matrix * normalized[:, None]).sum(axis=0))


def combine_logits(
    logits: np.ndarray,
    method: str = "logit_mean",
    weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Return the pre-calibration ensemble input in the requested strategy."""

    normalized_method = str(method).lower()
    if normalized_method in {"logit_mean", "mean_logits"}:
        return mean_logits(logits)
    if normalized_method in {"probability_mean", "mean_probabilities"}:
        return mean_probabilities(logits, weights=weights)
    if normalized_method in {"weighted_probability_mean", "weighted_probabilities"}:
        if weights is None:
            raise ValueError("weighted probability mean requires weights")
        return weighted_probability_mean(expit(_member_matrix(logits, "logits")), weights)
    raise ValueError(f"Unknown ensemble method: {method}")


def ensemble_probabilities(
    logits: np.ndarray,
    method: str = "logit_mean",
    weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Combine model logits exactly as competition inference should."""

    normalized_method = str(method).lower()
    if normalized_method in {"logit_mean", "mean_logits"}:
        return safe_probabilities(expit(mean_logits(logits)))
    return combine_logits(logits, normalized_method, weights=weights)


def validate_member_oof_matrix(
    targets: np.ndarray,
    prediction_matrix: np.ndarray,
    *,
    member_training_masks: np.ndarray | None = None,
) -> None:
    """Guard against learning fold weights from invalid ordinary OOF data.

    A standard K-fold OOF matrix has one prediction per row, not one prediction
    from every fold model.  Callers must supply a genuinely out-of-sample
    training mask for every member before using ``optimize_weights`` on a
    matrix of model predictions.
    """

    matrix = _member_matrix(prediction_matrix, "prediction_matrix")
    if matrix.shape[1] != len(np.asarray(targets)):
        raise ValueError("prediction_matrix must have one column per target row")
    if member_training_masks is None:
        raise ValueError(
            "Cannot learn per-fold weights from ordinary one-prediction-per-UID OOF data; "
            "provide repeated or otherwise genuinely out-of-sample member predictions"
        )
    masks = np.asarray(member_training_masks, dtype=bool)
    if masks.shape != matrix.shape:
        raise ValueError("member_training_masks must match prediction_matrix shape")
    if masks.any():
        raise ValueError("member_training_masks marks a training-leaked prediction")


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


def blend_probabilities(global_probability: np.ndarray, roi_probability: np.ndarray, global_weight: float) -> np.ndarray:
    """Blend two already-calculated probabilities with a normalized weight."""
    if not 0.0 <= float(global_weight) <= 1.0:
        raise ValueError("global_weight must be between 0 and 1")
    global_probability = safe_probabilities(global_probability)
    roi_probability = safe_probabilities(roi_probability)
    if global_probability.shape != roi_probability.shape:
        raise ValueError("Global and ROI probabilities must have the same shape")
    return safe_probabilities(float(global_weight) * global_probability + (1.0 - float(global_weight)) * roi_probability)


def blend_logits(first_probability: np.ndarray, second_probability: np.ndarray, first_weight: float) -> np.ndarray:
    """Blend clipped logits and return a bounded probability vector."""
    if not 0.0 <= float(first_weight) <= 1.0:
        raise ValueError("first_weight must be between 0 and 1")
    first = logit(safe_probabilities(first_probability))
    second = logit(safe_probabilities(second_probability))
    if first.shape != second.shape:
        raise ValueError("Logit ensemble members must have the same shape")
    return safe_probabilities(expit(float(first_weight) * first + (1.0 - float(first_weight)) * second))


def grid_search_two_model_logit_weights(
    targets: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    step: float = 0.05,
) -> tuple[float, list[dict]]:
    """Grid-search a two-member logit blend using only aligned OOF rows."""
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    weights = np.unique(np.round(np.arange(0.0, 1.0 + step * 0.5, step), 10))
    rows = []
    for weight in weights:
        probability = blend_logits(first_probability, second_probability, float(weight))
        rows.append({"first_weight": float(weight), "second_weight": float(1.0 - weight), **binary_metrics(targets, probability)})
    best = min(rows, key=lambda row: (row["log_loss"], abs(row["first_weight"] - 0.5)))
    return float(best["first_weight"]), rows


def grid_search_two_model_weights(
    targets: np.ndarray,
    global_probability: np.ndarray,
    roi_probability: np.ndarray,
    step: float = 0.05,
) -> tuple[float, np.ndarray, list[dict]]:
    """Select the global probability weight using concatenated OOF rows only."""
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    weights = np.unique(np.round(np.arange(0.0, 1.0 + step * 0.5, step), 10))
    rows = []
    for weight in weights:
        probability = blend_probabilities(global_probability, roi_probability, float(weight))
        rows.append({"global_weight": float(weight), "roi_weight": float(1.0 - weight), **binary_metrics(targets, probability)})
    best = min(rows, key=lambda row: (row["log_loss"], abs(row["global_weight"] - 0.5)))
    return float(best["global_weight"]), np.asarray([best["global_weight"], 1.0 - best["global_weight"]], dtype=float), rows


def prediction_diversity(targets: np.ndarray, global_probability: np.ndarray, roi_probability: np.ndarray) -> dict:
    """Describe complementary errors on the same OOF rows."""
    global_probability = safe_probabilities(global_probability)
    roi_probability = safe_probabilities(roi_probability)
    if global_probability.shape != roi_probability.shape:
        raise ValueError("Global and ROI probabilities must have the same shape")
    disagreement = (global_probability >= 0.5) != (roi_probability >= 0.5)
    global_loss = -(targets * np.log(global_probability) + (1.0 - targets) * np.log(1.0 - global_probability))
    roi_loss = -(targets * np.log(roi_probability) + (1.0 - targets) * np.log(1.0 - roi_probability))
    pearson = float(np.corrcoef(global_probability, roi_probability)[0, 1]) if len(global_probability) > 1 else float("nan")
    spearman = spearmanr(global_probability, roi_probability).statistic if len(global_probability) > 1 else float("nan")
    return {
        "pearson_probability_correlation": pearson,
        "spearman_probability_correlation": float(spearman),
        "classification_disagreement_count": int(disagreement.sum()),
        "classification_disagreement_fraction": float(disagreement.mean()) if len(disagreement) else 0.0,
        "disagreement_log_loss_global": float(binary_metrics(targets[disagreement], global_probability[disagreement])["log_loss"]) if disagreement.any() else float("nan"),
        "disagreement_log_loss_roi": float(binary_metrics(targets[disagreement], roi_probability[disagreement])["log_loss"]) if disagreement.any() else float("nan"),
        "global_confident_wrong_feature_correct_count": int(((global_loss > 0.693147) & (roi_loss < global_loss)).sum()),
        "feature_confident_wrong_global_correct_count": int(((roi_loss > 0.693147) & (global_loss < roi_loss)).sum()),
    }
