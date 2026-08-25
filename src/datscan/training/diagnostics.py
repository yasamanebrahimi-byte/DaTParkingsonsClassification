"""Calibration plots and confidence-extreme diagnostics from OOF rows only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from ..utils.metrics import binary_metrics, safe_probabilities


CONFIDENCE_BINS = [(-np.inf, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.75), (0.75, 0.90), (0.90, 0.95), (0.95, np.inf)]


def confidence_extremes(targets, probabilities) -> pd.DataFrame:
    y = np.asarray(targets, dtype=float)
    p = safe_probabilities(probabilities)
    rows = []
    for lower, upper in CONFIDENCE_BINS:
        mask = (p < upper) & (p >= lower)
        if not mask.any():
            rows.append({"bin": f"{lower:g}–{upper:g}", "count": 0, "mean_probability": np.nan, "actual_positive_fraction": np.nan, "mean_log_loss": np.nan})
            continue
        probability = p[mask]
        target = y[mask]
        loss = -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
        rows.append(
            {
                "bin": f"{lower:g}–{upper:g}",
                "count": int(mask.sum()),
                "mean_probability": float(probability.mean()),
                "actual_positive_fraction": float(target.mean()),
                "mean_log_loss": float(loss.mean()),
            }
        )
    return pd.DataFrame(rows)


def clipping_results(targets, probabilities, epsilons=(1e-6, 0.001, 0.005, 0.01, 0.02)) -> pd.DataFrame:
    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    rows = []
    for epsilon in epsilons:
        clipped = np.clip(p, float(epsilon), 1.0 - float(epsilon))
        rows.append({"epsilon": float(epsilon), **binary_metrics(y, clipped)})
    return pd.DataFrame(rows)


def write_calibration_diagnostics(frame: pd.DataFrame, output_dir: str | Path, probability_column: str = "probability") -> dict:
    if not {"target", probability_column}.issubset(frame.columns):
        raise ValueError(f"Diagnostics require target and {probability_column} columns")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y = frame["target"].to_numpy(dtype=float)
    p = safe_probabilities(frame[probability_column].to_numpy(dtype=float))
    fraction, mean_predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(mean_predicted, fraction, "o-", label="OOF")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axis.set(xlabel="Mean predicted probability", ylabel="Fraction positive", title="Reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reliability_diagram.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(p, bins=20, range=(0, 1), edgecolor="black")
    axis.set(xlabel="Predicted probability", ylabel="Count", title="OOF probability histogram")
    figure.tight_layout()
    figure.savefig(output / "probability_histogram.png", dpi=160)
    plt.close(figure)

    extremes = confidence_extremes(y, p)
    extremes.to_csv(output / "confidence_extremes.csv", index=False)
    clipping = clipping_results(y, p)
    clipping.to_csv(output / "clipping_results.csv", index=False)
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    summary = {
        "metrics": binary_metrics(y, p),
        "mean_predicted_probability": float(p.mean()),
        "true_positive_prevalence": float(y.mean()),
        "confidence_extremes": extremes.to_dict(orient="records"),
        "clipping": clipping.to_dict(orient="records"),
    }
    import json

    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    return summary
