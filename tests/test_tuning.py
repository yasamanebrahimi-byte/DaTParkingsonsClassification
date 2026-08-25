import json

import numpy as np
import pandas as pd
import yaml

from datscan.training.optimization import gradient_accumulation_optimizer_steps
from datscan.training.tuning import (
    apply_trial_parameters,
    physical_fov,
    promote_trials,
    run_screening_study,
    sample_trial_configs,
    score_oof_predictions,
)


def test_sampling_is_deterministic_and_inside_space():
    space = {
        "learning_rate": {"type": "categorical", "values": [1e-4, 2e-4]},
        "base_channels": {"type": "int", "low": 16, "high": 32},
    }
    first = sample_trial_configs(space, 8, 123)
    second = sample_trial_configs(space, 8, 123)
    assert first == second
    assert all(row["learning_rate"] in {1e-4, 2e-4} for row in first)
    assert all(16 <= row["base_channels"] <= 32 for row in first)


def test_paired_resolution_override_and_physical_fov():
    base = {"preprocessing": {"target_spacing_mm": 3.0, "output_shape": [96, 96, 96]}}
    space = {
        "resolution_pair": {
            "type": "categorical",
            "overrides": {"spacing": "preprocessing.target_spacing_mm", "shape": "preprocessing.output_shape"},
            "values": [{"spacing": 2.5, "shape": [112, 112, 112]}],
        }
    }
    result = apply_trial_parameters(base, space, {"resolution_pair": {"spacing": 2.5, "shape": [112, 112, 112]}})
    assert result["preprocessing"]["output_shape"] == [112, 112, 112]
    assert physical_fov(result) == [280.0, 280.0, 280.0]


def test_grouped_augmentation_severity_uses_preset_instead_of_stale_base_sections():
    base = {"augmentation": {"name": "scanner_robust", "severity": "moderate", "gaussian_blur": {"probability": 0.5}}}
    space = {"augmentation_severity": {"path": "augmentation.severity", "grouped_preset": True, "type": "categorical", "values": ["mild", "strong"]}}
    result = apply_trial_parameters(base, space, {"augmentation_severity": "strong"})
    assert result["augmentation"]["severity"] == "strong"
    assert "gaussian_blur" not in result["augmentation"]


def test_gradient_accumulation_step_frequency():
    assert gradient_accumulation_optimizer_steps(7, 1) == 7
    assert gradient_accumulation_optimizer_steps(7, 2) == 4
    assert gradient_accumulation_optimizer_steps(7, 4) == 2


def test_oof_objective_uses_concatenated_rows_not_unweighted_fold_mean():
    oof = pd.DataFrame({
        "fold": [0, 0, 0, 1],
        "target": [0.0, 0.0, 1.0, 1.0],
        "probability": [0.1, 0.1, 0.9, 0.6],
    })
    scored = score_oof_predictions(oof)
    fold_mean = np.mean([row["log_loss"] for row in scored["folds"]])
    assert not np.isclose(scored["overall"]["log_loss"], fold_mean)


def test_study_failure_isolated_resume_and_trial_configs_persisted(tmp_path, monkeypatch):
    search = tmp_path / "search.yaml"
    base = tmp_path / "base.yaml"
    metadata = tmp_path / "metadata.csv"
    folds = tmp_path / "folds.csv"
    base.write_text(yaml.safe_dump({"seed": 7, "model": {"name": "resnet3d", "base_channels": 2}, "preprocessing": {"output_shape": [8, 8, 8], "target_spacing_mm": 1.0}, "training": {"epochs": 1, "patience": 1}}), encoding="utf-8")
    search.write_text(yaml.safe_dump({"base_config": str(base), "seed": 99, "n_trials": 3, "screening": {"folds": 2, "epochs": 1, "patience": 1}, "search_space": {"base_channels": {"type": "categorical", "values": [2, 4]}}}), encoding="utf-8")
    rows = pd.DataFrame({"uid": [f"u{i}" for i in range(8)], "label": [i % 2 for i in range(8)], "filepath": [f"f{i}" for i in range(8)]})
    rows.to_csv(metadata, index=False)
    pd.DataFrame({"uid": rows.uid, "fold": [i % 2 for i in range(8)], "label": rows.label}).to_csv(folds, index=False)

    calls = []

    def fake_candidate(metadata_frame, config, output_dir, *, stage, training_seed, trial_id):
        calls.append(trial_id)
        if trial_id == "trial_001":
            raise RuntimeError("synthetic failure")
        predictions = pd.DataFrame({"uid": metadata_frame.uid, "target": metadata_frame.label.astype(float), "probability": np.where(metadata_frame.label == 1, 0.8, 0.2), "fold": metadata_frame.fold})
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_dir / "oof.csv", index=False)
        return {"trial_id": trial_id, "stage": stage, "status": "completed", "mean_log_loss": 0.2 if trial_id == "trial_000" else 0.3, "std_log_loss": 0.01, "worst_fold_log_loss": 0.21, "auroc": 0.9, "brier": 0.1, "runtime_seconds": 0.01}, predictions

    monkeypatch.setattr("datscan.training.tuning._candidate_result", fake_candidate)
    study = tmp_path / "study"
    run_screening_study(search, metadata, folds, study, resume=True)
    assert calls == ["trial_000", "trial_001", "trial_002"]
    assert (study / "trials/trial_000/config.yaml").exists()
    failed = json.loads((study / "trials/trial_001/screening.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert set(pd.read_csv(study / "results.csv")["trial_id"]) == {"trial_000", "trial_001", "trial_002"}
    calls.clear()
    run_screening_study(search, metadata, folds, study, resume=True)
    assert calls == []
    promote_trials(study, folds, promote_top=1)
    assert calls == ["trial_000"]
    assert (study / "trials/trial_000/full_result.json").exists()
