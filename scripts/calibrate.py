"""Fit legacy or versioned OOF calibration artifacts.

The legacy invocation remains supported:
``--oof file.csv --output temperature.json``.
Use ``--method platt`` or ``--cross-fit`` for the Priority 6 experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.calibrate import (  # noqa: E402
    apply_calibration,
    cross_fitted_calibration,
    fit_calibration_artifact,
    save_calibration,
    save_calibration_artifact,
)
from datscan.utils.metrics import binary_metrics  # noqa: E402


def _values(frame: pd.DataFrame, probability_column: str | None, input_type: str | None) -> tuple[np.ndarray, str]:
    if probability_column:
        if probability_column not in frame.columns:
            raise ValueError(f"Missing probability column: {probability_column}")
        return frame[probability_column].to_numpy(dtype=float), "probability"
    if input_type == "probability":
        column = "probability"
        if column not in frame.columns:
            raise ValueError("OOF file requires probability or --probability-column when --input-type probability is used")
        return frame[column].to_numpy(dtype=float), "probability"
    if "logit" not in frame.columns:
        raise ValueError("OOF file requires logit unless --probability-column is supplied")
    return frame["logit"].to_numpy(dtype=float), "logit"


def _folds(frame: pd.DataFrame, column: str | None, targets: np.ndarray, n_splits: int) -> np.ndarray:
    if column and column not in frame.columns:
        raise ValueError(f"Missing calibration fold column: {column}")
    if column:
        return frame[column].to_numpy()
    if "fold" in frame.columns:
        return frame["fold"].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260824)
    result = np.empty(len(frame), dtype=int)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(frame)), targets.astype(int))):
        result[validation] = fold
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probability-column", help="Use this probability column for probability-space calibration")
    parser.add_argument("--method", choices=["none", "temperature", "legacy_temperature", "platt"], default=None)
    parser.add_argument("--input-type", choices=["logit", "probability"], default=None)
    parser.add_argument("--ensemble-method", default="logit_mean")
    parser.add_argument("--stage", default=None, choices=["before_ensemble", "after_ensemble"])
    parser.add_argument("--cross-fit", action="store_true", help="Also perform cross-fitted OOF evaluation")
    parser.add_argument("--calibration-fold-column", default=None)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--metrics-output")
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.oof)
    if "target" not in frame.columns:
        raise ValueError("OOF file requires target")
    targets = frame["target"].to_numpy(dtype=float)
    values, inferred_type = _values(frame, args.probability_column, args.input_type)
    input_type = args.input_type or inferred_type
    method = args.method or "temperature"
    stage = args.stage or ("after_ensemble" if input_type == "probability" else "before_ensemble")

    raw_probability = expit(values) if input_type == "logit" else np.clip(values, args.epsilon, 1.0 - args.epsilon)
    raw_metrics = binary_metrics(targets, raw_probability)
    artifact = fit_calibration_artifact(
        values,
        targets,
        method=method,
        input_type=input_type,
        ensemble_method=args.ensemble_method,
        stage=stage,
        epsilon=args.epsilon,
    )
    calibrated_probability = apply_calibration(values, artifact)
    in_sample = binary_metrics(targets, calibrated_probability)
    artifact["raw_metrics"] = raw_metrics
    artifact["in_sample_metrics"] = in_sample

    report = {
        "artifact": artifact,
        "raw_metrics": raw_metrics,
        "in_sample_metrics": in_sample,
    }
    if args.cross_fit:
        folds = _folds(frame, args.calibration_fold_column, targets, args.calibration_folds)
        cross_fitted = cross_fitted_calibration(
            values,
            targets,
            folds,
            method=method,
            input_type=input_type,
            epsilon=args.epsilon,
        )
        report["cross_fitted_metrics"] = cross_fitted["metrics"]
        report["cross_fitted_fold_parameters"] = cross_fitted["fold_parameters"]
        report["cross_fitted_stability"] = cross_fitted["stability"]
        artifact["cross_fitted_metrics"] = cross_fitted["metrics"]
        artifact["cross_fitted_fold_parameters"] = cross_fitted["fold_parameters"]
        artifact["cross_fitted_stability"] = cross_fitted["stability"]

    # Preserve the exact old JSON contract for the original CLI, including
    # its historical before-ensemble stage and enabled-if-better behavior.
    if args.method is None and args.input_type is None and not args.cross_fit:
        enabled = in_sample["log_loss"] < raw_metrics["log_loss"]
        save_calibration(
            artifact.get("temperature", 1.0) if enabled else 1.0,
            args.output,
            stage=stage,
            enabled=enabled,
            raw_log_loss=raw_metrics["log_loss"],
            calibrated_log_loss=in_sample["log_loss"],
        )
    else:
        save_calibration_artifact(artifact, args.output)
    if args.metrics_output:
        path = Path(args.metrics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Raw: {raw_metrics}")
    print(f"In-sample calibrated: {in_sample}")
    if "cross_fitted_metrics" in report:
        print(f"Cross-fitted calibrated: {report['cross_fitted_metrics']}")
    print(f"Saved calibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
