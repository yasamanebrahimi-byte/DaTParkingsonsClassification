import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from datscan.training.domains import assign_domain_groups, domain_summary
from datscan.training.evaluate import evaluate_oof_by_domain
from datscan.training.folds import (
    create_domain_folds,
    create_folds,
    fold_quality,
    leave_one_domain_out_splits,
    validate_domain_folds,
)


def _metadata(n_groups=5, samples_per_group=4):
    rows = []
    for group in range(n_groups):
        for offset in range(samples_per_group):
            rows.append(
                {
                    "uid": f"scan_{group}_{offset}",
                    "label": float(offset % 2),
                    "shape_x": 32 + group * 8,
                    "shape_y": 32 + group * 8,
                    "shape_z": 16 + group * 4,
                    "spacing_x": 2.0 + group * 0.2,
                    "spacing_y": 2.0 + group * 0.2,
                    "spacing_z": 2.0 + group * 0.2,
                    "physical_extent_x": (32 + group * 8) * (2.0 + group * 0.2),
                    "physical_extent_y": (32 + group * 8) * (2.0 + group * 0.2),
                    "physical_extent_z": (16 + group * 4) * (2.0 + group * 0.2),
                    "nonzero_fraction": 0.1 + group * 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_domain_assignment_is_deterministic_and_does_not_use_labels():
    metadata = _metadata(n_groups=4, samples_per_group=3)
    first = assign_domain_groups(metadata, n_domains=4, seed=7)
    changed_labels = metadata.copy()
    changed_labels["label"] = 1.0 - changed_labels["label"]
    second = assign_domain_groups(changed_labels, n_domains=4, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["uid"].nunique() == len(metadata)
    assert first["domain_group"].notna().all()


def test_domain_summary_has_expected_label_counts():
    metadata = _metadata(n_groups=3, samples_per_group=4)
    assignments = pd.DataFrame(
        {
            "uid": metadata["uid"],
            "domain_group": [f"domain_{index // 4}" for index in range(len(metadata))],
        }
    )
    summary = domain_summary(metadata, assignments)
    assert summary["sample_count"].sum() == len(metadata)
    assert summary["normal_count"].sum() == 6
    assert summary["pathologic_count"].sum() == 6


def test_domain_folds_have_no_domain_leakage_and_preserve_labels():
    metadata = _metadata()
    assignments = pd.DataFrame(
        {"uid": metadata["uid"], "domain_group": metadata["uid"].str.split("_").str[1].map(lambda x: f"domain_{x}")}
    )
    folds = create_domain_folds(metadata, assignments, n_splits=5, seed=11)
    validate_domain_folds(folds)
    assert set(folds["uid"]) == set(metadata["uid"])
    merged = metadata[["uid", "label"]].merge(folds[["uid", "label"]], on="uid", suffixes=("_source", "_fold"))
    assert np.array_equal(merged["label_source"], merged["label_fold"])
    for fold in folds["fold"].unique():
        train_domains = set(folds.loc[folds["fold"] != fold, "domain_group"])
        valid_domains = set(folds.loc[folds["fold"] == fold, "domain_group"])
        assert train_domains.isdisjoint(valid_domains)
    assert folds.attrs["n_splits"] == 5


def test_domain_fold_generation_reproducible_and_reduces_too_many_requested_folds():
    metadata = _metadata(n_groups=3, samples_per_group=4)
    assignments = pd.DataFrame(
        {"uid": metadata["uid"], "domain_group": metadata["uid"].str.split("_").str[1].map(lambda x: f"domain_{x}")}
    )
    with pytest.warns(UserWarning):
        first = create_domain_folds(metadata, assignments, n_splits=5, seed=11)
    with pytest.warns(UserWarning):
        second = create_domain_folds(metadata, assignments, n_splits=5, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert first["fold"].nunique() == 3
    with pytest.raises(ValueError, match="at least 2"):
        create_domain_folds(metadata, assignments, n_splits=1)


def test_standard_stratified_folds_remain_stratified_kfold_behavior():
    metadata = _metadata(n_groups=4, samples_per_group=3)
    actual = create_folds(metadata, n_splits=3, seed=13)
    expected = pd.DataFrame({"uid": metadata["uid"], "label": metadata["label"], "fold": -1})
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
    for fold, (_, validation_indices) in enumerate(splitter.split(expected, expected["label"].astype(int))):
        expected.iloc[validation_indices, expected.columns.get_loc("fold")] = fold
    expected = expected.sort_values("uid").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_domain_evaluation_handles_one_class_domain():
    oof = pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d"],
            "target": [0, 0, 1, 1],
            "probability": [0.1, 0.3, 0.7, 0.8],
        }
    )
    assignments = pd.DataFrame(
        {"uid": ["a", "b", "c", "d"], "domain_group": ["normal_only", "normal_only", "mixed", "mixed"]}
    )
    results = evaluate_oof_by_domain(oof, assignments)
    normal_only = results.loc[results["domain_group"] == "normal_only"].iloc[0]
    assert np.isfinite(normal_only["log_loss"])
    assert np.isfinite(normal_only["brier"])
    assert np.isnan(normal_only["auroc"])


def test_domain_fold_validation_rejects_duplicate_uids():
    frame = pd.DataFrame(
        {
            "uid": ["a", "a"],
            "label": [0, 0],
            "fold": [0, 1],
            "domain_group": ["d0", "d1"],
        }
    )
    with pytest.raises(ValueError, match="duplicate UIDs"):
        validate_domain_folds(frame)


def test_leave_one_domain_out_skips_tiny_domains():
    metadata = _metadata(n_groups=3, samples_per_group=4)
    assignments = pd.DataFrame(
        {"uid": metadata["uid"], "domain_group": metadata["uid"].str.split("_").str[1].map(lambda x: f"domain_{x}")}
    )
    splits = leave_one_domain_out_splits(metadata, assignments, min_validation_samples=4)
    assert [domain for domain, _, _ in splits] == ["domain_0", "domain_1", "domain_2"]
    assert all(len(validation) == 4 and len(training) == 8 for _, training, validation in splits)
    with pytest.raises(ValueError, match="No domain"):
        leave_one_domain_out_splits(metadata, assignments, min_validation_samples=5)
