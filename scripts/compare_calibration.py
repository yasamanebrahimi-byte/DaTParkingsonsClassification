"""Compare calibration methods on single or repeated OOF representations."""

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

from datscan.training.calibrate import apply_calibration, cross_fitted_calibration, fit_calibration_artifact  # noqa: E402
from datscan.training.repeated import aggregate_repeated_oof  # noqa: E402
from datscan.utils.metrics import binary_metrics  # noqa: E402


def _calibration_folds(frame: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    if "fold" in frame.columns:
        return frame["fold"].to_numpy()
    result = np.empty(len(targets), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260824)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(targets)), targets.astype(int))):
        result[validation] = fold
    return result


def _row(representation: str, ensemble: str, calibration: str, targets: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = binary_metrics(targets, probabilities)
    return {
        "oof_representation": representation,
        "ensemble_method": ensemble,
        "calibration": calibration,
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier_score"],
        "auroc": metrics["auroc"],
    }


def _compare_one(
    representation: str,
    ensemble: str,
    values: np.ndarray,
    input_type: str,
    targets: np.ndarray,
    folds: np.ndarray,
) -> list[dict]:
    raw = expit(values) if input_type == "logit" else np.clip(values, 1e-6, 1.0 - 1e-6)
    rows = [_row(representation, ensemble, "None", targets, raw)]
    legacy = fit_calibration_artifact(values, targets, method="temperature", input_type=input_type, ensemble_method=ensemble, stage="after_ensemble")
    rows.append(_row(representation, ensemble, "Legacy Temperature", targets, apply_calibration(values, legacy)))
    for method, label in (("temperature", "Cross-Fitted Temperature"), ("platt", "Cross-Fitted Platt")):
        result = cross_fitted_calibration(values, targets, folds, method=method, input_type=input_type)
        rows.append(_row(representation, ensemble, label, targets, result["probabilities"]))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="Canonical OOF CSV or repeated long-format OOF CSV")
    parser.add_argument("--repeated-summary", help="Optional summary emitted by aggregate_repeated_oof.py")
    parser.add_argument("--output", required=True, help="CSV comparison table")
    parser.add_argument("--json-output")
    parser.add_argument("--plots-dir")
    args = parser.parse_args(argv)
    source = pd.read_csv(args.oof, dtype={"uid": str})
    if "repeat" in source.columns:
        summary = aggregate_repeated_oof(source)
    elif {"mean_logit", "mean_probability"}.issubset(source.columns):
        summary = source
    else:
        summary = None
    rows = []
    if summary is None:
        if not {"target", "logit"}.issubset(source.columns):
            raise ValueError("Single OOF input requires target and logit columns")
        targets = source["target"].to_numpy(dtype=float)
        folds = _calibration_folds(source, targets)
        rows.extend(_compare_one("Single-model OOF", "single_model", source["logit"].to_numpy(dtype=float), "logit", targets, folds))
    else:
        if args.repeated_summary:
            summary = pd.read_csv(args.repeated_summary, dtype={"uid": str})
        targets = summary["target"].to_numpy(dtype=float)
        folds = _calibration_folds(summary, targets)
        rows.extend(_compare_one("Repeated OOF", "mean_logits", summary["mean_logit"].to_numpy(dtype=float), "logit", targets, folds))
        rows.extend(_compare_one("Repeated OOF", "mean_probabilities", summary["mean_probability"].to_numpy(dtype=float), "probability", targets, folds))
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(orient="records"), indent=2, allow_nan=False), encoding="utf-8")
    if args.plots_dir:
        from datscan.training.diagnostics import write_calibration_diagnostics

        if summary is None:
            diagnostic_frame = source[["target", "probability"]].copy()
        else:
            diagnostic_frame = summary[["target", "mean_probability"]].rename(columns={"mean_probability": "probability"})
        write_calibration_diagnostics(diagnostic_frame, args.plots_dir)
    print(result.to_string(index=False))
    print(f"Wrote comparison table to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

