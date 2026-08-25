import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from datscan.training.calibrate import (
    apply_calibration,
    apply_platt,
    apply_temperature,
    cross_fitted_calibration,
    fit_calibration_artifact,
    fit_platt,
    load_calibration_artifact,
    save_calibration_artifact,
)
from datscan.training.ensemble import (
    combine_logits,
    ensemble_probabilities,
    mean_logits,
    mean_probabilities,
    validate_member_oof_matrix,
)
from datscan.training.repeated import aggregate_repeated_oof, validate_repeated_oof_assignments


def _repeated_frame() -> pd.DataFrame:
    logits = np.asarray([[-2.0, 0.0, 1.0, 2.0], [-1.0, 0.5, 1.5, 2.5]])
    targets = [0.0, 0.0, 1.0, 1.0]
    rows = []
    for repeat in range(logits.shape[0]):
        for uid, target, value in zip(["a", "b", "c", "d"], targets, logits[repeat]):
            rows.append({"uid": uid, "target": target, "repeat": repeat, "fold": int(uid in {"c", "d"}), "logit": value, "probability": float(expit(value))})
    return pd.DataFrame(rows)


def test_platt_scaling_is_bounded_and_identity_is_sigmoid():
    logits = np.asarray([-3.0, -0.5, 0.0, 1.5, 3.0])
    targets = np.asarray([0, 0, 1, 1, 1], dtype=float)
    slope, intercept = fit_platt(logits, targets)
    probabilities = apply_platt(logits, slope, intercept)
    assert 0.0 <= probabilities.min() <= probabilities.max() <= 1.0
    assert np.allclose(apply_platt(logits, 1.0, 0.0), expit(logits))


def test_explicit_ensemble_math_is_not_interchangeable():
    logits = np.asarray([[-2.0, 0.5], [2.0, 1.5]])
    assert np.allclose(mean_logits(logits), logits.mean(axis=0))
    assert np.allclose(mean_probabilities(logits), expit(logits).mean(axis=0))
    assert np.allclose(ensemble_probabilities(logits, "logit_mean"), expit(logits.mean(axis=0)))
    assert np.allclose(ensemble_probabilities(logits, "probability_mean"), expit(logits).mean(axis=0))
    assert np.allclose(combine_logits(logits, "weighted_probability_mean", [0.25, 0.75]), np.asarray([0.75 * expit(2.0) + 0.25 * expit(-2.0), 0.75 * expit(1.5) + 0.25 * expit(0.5)]), atol=1e-6)


def test_repeated_oof_aggregation_and_missing_repeat_guard():
    source = _repeated_frame()
    summary = aggregate_repeated_oof(source, n_repeats=2)
    assert len(summary) == 4
    assert (summary["n_predictions"] == 2).all()
    assert np.allclose(summary["mean_probability"], source.groupby("uid")["probability"].mean().to_numpy())
    with pytest.raises(ValueError, match="exactly 2"):
        aggregate_repeated_oof(source.drop(source.index[-1]), n_repeats=2)


def test_cross_fitted_calibration_does_not_use_validation_targets():
    values = np.asarray([-2.0, -0.5, 0.5, 2.0, -1.5, 1.5])
    targets = np.asarray([0, 0, 1, 1, 0, 1], dtype=float)
    folds = np.asarray([0, 0, 0, 1, 1, 1])
    first = cross_fitted_calibration(values, targets, folds, method="temperature")
    changed = targets.copy()
    changed[:3] = 1.0 - changed[:3]
    second = cross_fitted_calibration(values, changed, folds, method="temperature")
    assert np.allclose(first["probabilities"][:3], second["probabilities"][:3])
    assert all(row["training_count"] == 3 and row["validation_count"] == 3 for row in first["fold_parameters"])


def test_calibration_artifact_round_trip_and_legacy_equivalence(tmp_path):
    logits = np.asarray([-2.0, -0.5, 0.5, 2.0])
    targets = np.asarray([0, 0, 1, 1], dtype=float)
    artifact = fit_calibration_artifact(logits, targets, method="platt", input_type="logit", ensemble_method="logit_mean")
    path = tmp_path / "calibration.json"
    save_calibration_artifact(artifact, path)
    loaded = load_calibration_artifact(path)
    assert loaded["version"] == 2
    assert np.allclose(apply_calibration(logits, loaded), apply_platt(logits, loaded["slope"], loaded["intercept"]))

    legacy = {"method": "temperature_scaling", "temperature": 1.7, "enabled": True}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
    from datscan_inference.calibration import apply_calibration as submission_apply

    assert np.allclose(
        apply_calibration(expit(logits), {**legacy, "input_type": "probability"}),
        submission_apply(expit(logits), {**legacy, "input_type": "probability"}),
    )


def test_ordinary_single_prediction_oof_cannot_learn_fold_weights():
    with pytest.raises(ValueError, match="ordinary one-prediction-per-UID"):
        validate_member_oof_matrix(np.asarray([0, 1]), np.asarray([[0.2, 0.8], [0.3, 0.7]]))


def test_repeated_oof_assignment_check_keeps_held_out_rows_out_of_training():
    source = _repeated_frame()
    metadata = pd.DataFrame({"uid": ["a", "b", "c", "d"], "label": [0.0, 0.0, 1.0, 1.0]})
    folds = {repeat: group[["uid", "fold"]].copy() for repeat, group in source.groupby("repeat")}
    validate_repeated_oof_assignments(metadata, folds, source)
    bad = source.copy()
    bad.loc[bad.index[0], "fold"] = 9
    with pytest.raises(ValueError, match="fold assignments"):
        validate_repeated_oof_assignments(metadata, folds, bad)
