import numpy as np
import pandas as pd

from datscan.features.striatal_features import extract_striatal_features_from_roi, feature_family
from datscan.training.feature_cv import make_feature_estimator, train_feature_cv


def _bilateral_roi(background=0.1, uptake=1.0):
    roi = np.full((20, 20, 12), background, dtype=np.float32)
    roi[3:7, 7:13, 3:7] = uptake
    roi[13:17, 7:13, 3:7] = uptake
    return roi


def test_feature_extraction_is_deterministic_finite_and_named():
    roi = _bilateral_roi()
    first = extract_striatal_features_from_roi(roi, (2.0, 3.0, 4.0))
    second = extract_striatal_features_from_roi(roi.copy(), (2.0, 3.0, 4.0))
    assert first == second
    assert len(first) >= 60
    assert {"roi_mean", "left_mean", "right_mean", "asymmetry_mean_abs", "left_striatum_to_background", "left_posterior_to_anterior", "number_of_connected_components", "left_bounding_box_extent_x_mm"}.issubset(first)
    assert np.isfinite(np.asarray(list(first.values()), dtype=float)).all()


def test_canonical_axis_split_assigns_left_and_right_without_merging():
    roi = np.full((20, 20, 12), 0.1, dtype=np.float32)
    roi[:10, 7:13, 3:7] = 0.4
    roi[10:, 7:13, 3:7] = 0.9
    features = extract_striatal_features_from_roi(roi, 1.0)
    assert features["left_mean"] < features["right_mean"]
    assert features["asymmetry_mean_abs"] > 0
    assert feature_family("left_bounding_box_extent_x_mm") == "shape"


def test_symmetric_scan_has_near_zero_asymmetry():
    features = extract_striatal_features_from_roi(_bilateral_roi(), 1.0)
    assert features["asymmetry_mean_abs"] < 1e-7
    assert features["asymmetry_mean_normalized"] < 1e-6


def test_posterior_loss_reduces_posterior_anterior_ratio():
    normal = np.full((20, 20, 12), 0.1, dtype=np.float32)
    abnormal = normal.copy()
    for array in (normal, abnormal):
        array[:10, 2:6, 3:7] = 0.9
        array[:10, 14:18, 3:7] = 0.9
        array[10:, 2:6, 3:7] = 0.9
        array[10:, 14:18, 3:7] = 0.9
    abnormal[:, 2:6, 3:7] = 0.25
    normal_features = extract_striatal_features_from_roi(normal, 1.0)
    abnormal_features = extract_striatal_features_from_roi(abnormal, 1.0)
    assert abnormal_features["minimum_posterior_to_anterior"] < normal_features["minimum_posterior_to_anterior"]


def test_background_increase_reduces_ratio():
    low_background = extract_striatal_features_from_roi(_bilateral_roi(background=0.1), 1.0)
    high_background = extract_striatal_features_from_roi(_bilateral_roi(background=0.2), 1.0)
    assert high_background["background_uptake"] > low_background["background_uptake"]
    assert high_background["bilateral_striatum_to_background"] < low_background["bilateral_striatum_to_background"]


def test_morphology_uses_physical_voxel_volume():
    roi = np.zeros((12, 12, 8), dtype=np.float32)
    roi[2:4, 2:5, 2:6] = 1.0
    features = extract_striatal_features_from_roi(roi, (2.0, 3.0, 4.0))
    assert features["high_uptake_voxel_count"] == 24
    assert features["high_uptake_physical_volume_mm3"] == 24 * 2 * 3 * 4
    assert features["number_of_connected_components"] == 1


def test_scaler_is_fit_on_outer_training_partition_only():
    estimator = make_feature_estimator("logistic")
    train = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    target = np.asarray([0, 0, 1, 1])
    estimator.fit(train, target)
    scaler = estimator.named_steps["scaler"]
    assert np.allclose(scaler.mean_, train.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.vstack([train, [[100.0, 100.0]]]).mean(axis=0))


def test_feature_oof_has_every_uid_once_and_probabilities_bounded(tmp_path):
    rows = []
    for index in range(20):
        rows.append({"uid": f"scan_{index:02d}", "label": float(index % 2), "roi_mean": float(index), "asymmetry_mean_abs": float(index % 3)})
    features = pd.DataFrame(rows)
    folds = pd.DataFrame({"uid": features["uid"], "fold": [index % 5 for index in range(len(features))]})
    oof, metrics = train_feature_cv(features, folds, "logistic", tmp_path / "oof.csv", tmp_path / "models")
    assert len(oof) == len(features)
    assert oof["uid"].is_unique
    assert set(oof["uid"]) == set(features["uid"])
    assert ((oof["probability"] >= 0) & (oof["probability"] <= 1)).all()
    assert metrics["feature_count"] == 2
