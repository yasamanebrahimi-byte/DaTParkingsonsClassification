"""OOF-only temperature scaling for probability calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import log_loss


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = expit(logits / temperature)
        return float(log_loss(targets, probabilities, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float, epsilon: float = 1e-6) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    return np.clip(expit(np.asarray(logits, dtype=float) / temperature), epsilon, 1.0 - epsilon)


def save_calibration(temperature: float, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": "temperature_scaling", "temperature": temperature}, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(payload["temperature"])

