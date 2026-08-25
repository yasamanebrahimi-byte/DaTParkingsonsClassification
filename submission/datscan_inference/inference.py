"""Offline competition inference for global-only or global + ROI packages."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit, logit

from .feature_support import PreprocessConfig as FeaturePreprocessConfig, ROIConfig as FeatureROIConfig
from .calibration import apply_calibration, combine_logits
from .models import load_model_details
from .preprocessing import Config, ROIConfig, preprocess_views
try:
    from .striatal_features import StriatalFeatureConfig, extract_striatal_features_from_roi
except ImportError:  # Feature files are optional in legacy CNN-only packages.
    StriatalFeatureConfig = None
    extract_striatal_features_from_roi = None
from .utils import validate_submission


def _load_family(paths, device):
    models = []
    preprocess_config = None
    roi_config = None
    for path in paths:
        model, raw_preprocess, payload = load_model_details(path, device)
        models.append(model)
        current_preprocess = Config.from_mapping(raw_preprocess)
        current_roi = ROIConfig.from_mapping(payload.get("roi")) if payload.get("roi") else None
        if preprocess_config is None:
            preprocess_config = current_preprocess
            roi_config = current_roi
        elif current_preprocess != preprocess_config or current_roi != roi_config:
            raise ValueError("Packaged checkpoints use different preprocessing/ROI configurations")
    return models, preprocess_config, roi_config


def _temperature_probability(probability: float, temperature: float) -> float:
    """Legacy public helper retained for old package behavior and tests."""

    return float(
        apply_calibration(
            probability,
            {"method": "temperature_scaling", "temperature": temperature, "input_type": "probability"},
        )
    )


def _calibrate_ensemble_input(logit_value: float, probability_value: float, calibration: dict) -> float:
    """Use the artifact's declared input type, defaulting legacy JSON to p."""

    payload = dict(calibration or {})
    input_type = str(payload.get("input_type", "probability")).lower()
    input_type = {"mean_logit": "logit", "mean_logits": "logit", "mean_probability": "probability", "mean_probabilities": "probability"}.get(input_type, input_type)
    value = logit_value if input_type == "logit" else probability_value
    return float(np.asarray(apply_calibration(value, payload)).reshape(-1)[0])


def _load_feature_family(assets: Path):
    paths = sorted(assets.glob("feature_model_fold*.pkl"))
    if not paths:
        return [], None, None, None, None
    if StriatalFeatureConfig is None or extract_striatal_features_from_roi is None:
        raise RuntimeError("Feature model assets are present but the packaged feature extractor is missing")
    columns_payload = json.loads((assets / "feature_columns.json").read_text(encoding="utf-8"))
    columns = columns_payload.get("feature_columns")
    if not columns:
        raise ValueError("feature_columns.json does not contain feature_columns")
    raw_config = yaml.safe_load((assets / "striatal_features.yaml").read_text(encoding="utf-8")) or {}
    feature_config = StriatalFeatureConfig.from_mapping(raw_config.get("features", raw_config))
    preprocess_config = FeaturePreprocessConfig.from_mapping(raw_config.get("preprocessing"))
    roi_config = FeatureROIConfig.from_mapping(dict(raw_config.get("roi") or {}, enabled=True))
    models = []
    for path in paths:
        with path.open("rb") as handle:
            models.append(pickle.load(handle))
    return models, columns, feature_config, preprocess_config, roi_config


def _feature_probability(models, columns, feature_config, roi, spacing_mm: float) -> float:
    values = extract_striatal_features_from_roi(roi, spacing_mm, feature_config)
    missing = [column for column in columns if column not in values]
    if missing:
        raise ValueError(f"Feature extraction is missing packaged columns: {missing}")
    matrix = np.asarray([[values[column] for column in columns]], dtype=float)
    probabilities = [float(model.predict_proba(matrix)[:, 1][0]) for model in models]
    return float(np.mean(np.clip(probabilities, 1e-6, 1.0 - 1e-6)))


def run_inference(root: Path) -> None:
    data_dir = root / "data"
    assets = root / "assets"
    template_path = data_dir / "submission_format.csv"
    output_path = root / "submission.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing submission template: {template_path}")
    global_paths = sorted(assets.glob("global_model_*.pt"))
    roi_paths = sorted(assets.glob("roi_model_*.pt"))
    legacy_paths = sorted(assets.glob("model_*.pt"))
    if not global_paths and not legacy_paths:
        raise FileNotFoundError("No packaged global model checkpoints found")
    if global_paths and legacy_paths:
        raise ValueError("Package contains both legacy and new global checkpoint names")
    if not global_paths:
        global_paths = legacy_paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_models, preprocess_config, _ = _load_family(global_paths, device)
    roi_models = []
    roi_config = None
    if roi_paths:
        roi_models, roi_preprocess_config, roi_config = _load_family(roi_paths, device)
        if roi_preprocess_config != preprocess_config:
            raise ValueError("Global and ROI checkpoints must share the same base preprocessing configuration")
        if roi_config is None or not roi_config.enabled:
            raise ValueError("ROI checkpoint metadata must include roi.enabled=true")

    manifest_path = assets / "ensemble.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    fold_ensemble_method = str((manifest or {}).get("ensemble_method", "logit_mean"))
    global_fold_weights = (manifest or {}).get("global_fold_weights", (manifest or {}).get("fold_weights"))
    roi_fold_weights = (manifest or {}).get("roi_fold_weights", (manifest or {}).get("fold_weights"))
    if roi_models:
        if manifest is None:
            raise FileNotFoundError("Global + ROI package is missing assets/ensemble.json")
        weights = manifest.get("weights", [0.5, 0.5])
        if len(weights) != 2 or not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("Invalid global/ROI ensemble weights")
        global_weight = float(weights[0]) / float(sum(weights))
        roi_weight = 1.0 - global_weight
    else:
        global_weight = 1.0
        roi_weight = 0.0

    feature_models, feature_columns, feature_config, feature_preprocess_config, feature_roi_config = _load_feature_family(assets)
    if feature_models:
        if feature_preprocess_config.__dict__ != preprocess_config.__dict__:
            raise ValueError("Packaged CNN and quantitative feature models use different preprocessing configurations")
        if roi_models and feature_roi_config.__dict__ != roi_config.__dict__:
            raise ValueError("Packaged ROI and quantitative feature models use different ROI configurations")

    calibration_path = assets / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {"temperature": 1.0, "enabled": False}
    # New calibration artifacts carry the fold-combination contract.  This is
    # the source of truth for global-only packages without an ensemble.json;
    # old packages retain the historical logit-mean default.
    if manifest is None or "ensemble_method" not in manifest:
        fold_ensemble_method = str(calibration.get("ensemble_method", fold_ensemble_method))
        if global_fold_weights is None:
            global_fold_weights = calibration.get("global_fold_weights", calibration.get("fold_weights"))
        if roi_fold_weights is None:
            roi_fold_weights = calibration.get("roi_fold_weights", calibration.get("fold_weights"))
    temperature = float(calibration.get("temperature", 1.0))
    calibration_enabled = bool(calibration.get("enabled", True))
    after_ensemble = bool(manifest and manifest.get("calibration_stage") == "after_ensemble") or calibration.get("stage") == "after_ensemble"
    feature_weight = None
    cnn_weight = None
    feature_method = "probability"
    if feature_models:
        if manifest is None:
            raise FileNotFoundError("Feature model assets require an OOF-derived assets/ensemble.json")
        member_names = [str(value).lower() for value in manifest.get("member_names", [])]
        weights = np.asarray(manifest.get("weights", []), dtype=float)
        if len(weights) != 2 or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Feature ensemble manifest must contain two non-negative finite weights")
        if member_names and len(member_names) == 2:
            feature_index = next((index for index, name in enumerate(member_names) if "feature" in name or "striat" in name), 1)
            cnn_index = 1 - feature_index
        else:
            cnn_index, feature_index = 0, 1
        normalized_weights = weights / weights.sum()
        cnn_weight = float(normalized_weights[cnn_index])
        feature_weight = float(normalized_weights[feature_index])
        feature_method = "logit" if "logit" in str(manifest.get("method", "")) else "probability"

    template = pd.read_csv(template_path, dtype={"uid": str})
    probabilities = []
    n = len(template)
    print("Loading models")
    with torch.inference_mode():
        for index, uid in enumerate(template["uid"].astype(str)):
            path = data_dir / "niftis" / f"{uid}.nii.gz"
            if not path.exists():
                matches = list((data_dir / "niftis").rglob(f"{uid}.nii.gz"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected exactly one NIfTI for UID {uid}")
                path = matches[0]
            view_roi_config = feature_roi_config if feature_models else (roi_config if roi_models else None)
            views = preprocess_views(path, preprocess_config, view_roi_config)
            global_tensor = torch.from_numpy(views["global"]).unsqueeze(0).to(device)
            global_logits = torch.stack([model(global_tensor).float().squeeze(0).cpu() for model in global_models]).numpy()
            global_ensemble_value, global_ensemble_type = combine_logits(global_logits, fold_ensemble_method, global_fold_weights)
            global_probability = float(expit(global_ensemble_value)) if global_ensemble_type == "logit" else float(global_ensemble_value)
            global_logit = float(global_ensemble_value) if global_ensemble_type == "logit" else float(logit(np.clip(global_probability, 1e-6, 1.0 - 1e-6)))
            if roi_models:
                roi_tensor = torch.from_numpy(views["roi"]).unsqueeze(0).to(device)
                roi_logits = torch.stack([model(roi_tensor).float().squeeze(0).cpu() for model in roi_models]).numpy()
                roi_ensemble_value, roi_ensemble_type = combine_logits(roi_logits, fold_ensemble_method, roi_fold_weights)
                roi_probability = float(expit(roi_ensemble_value)) if roi_ensemble_type == "logit" else float(roi_ensemble_value)
                roi_logit = float(roi_ensemble_value) if roi_ensemble_type == "logit" else float(logit(np.clip(roi_probability, 1e-6, 1.0 - 1e-6)))
                cnn_probability = global_weight * global_probability + roi_weight * roi_probability
                cnn_logit = float(logit(np.clip(cnn_probability, 1e-6, 1.0 - 1e-6)))
            else:
                cnn_probability = global_probability
                cnn_logit = global_logit
            if feature_models:
                feature_probability = _feature_probability(feature_models, feature_columns, feature_config, views["roi"], feature_preprocess_config.target_spacing_mm)
                if calibration_enabled and not after_ensemble:
                    cnn_probability = _calibrate_ensemble_input(cnn_logit, cnn_probability, calibration)
                    cnn_logit = float(logit(np.clip(cnn_probability, 1e-6, 1.0 - 1e-6)))
                if feature_method == "logit":
                    probability = float(expit(cnn_weight * logit(np.clip(cnn_probability, 1e-6, 1.0 - 1e-6)) + feature_weight * logit(np.clip(feature_probability, 1e-6, 1.0 - 1e-6))))
                else:
                    probability = cnn_weight * cnn_probability + feature_weight * feature_probability
                if calibration_enabled and after_ensemble:
                    probability = _calibrate_ensemble_input(float(logit(np.clip(probability, 1e-6, 1.0 - 1e-6))), probability, calibration)
            elif roi_models:
                probability = cnn_probability
                if calibration_enabled and after_ensemble:
                    probability = _calibrate_ensemble_input(cnn_logit, probability, calibration)
            else:
                probability = _calibrate_ensemble_input(cnn_logit, cnn_probability, calibration) if calibration_enabled else cnn_probability
            probabilities.append(float(np.clip(probability, 1e-6, 1.0 - 1e-6)))
            if n and (index + 1) in {max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4)}:
                print(f"{round((index + 1) * 100 / n)}% complete")
    output = pd.DataFrame({"uid": template["uid"].astype(str), "is_pathologic": probabilities})
    validate_submission(output, template)
    output.to_csv(output_path, index=False)
    print("Writing submission")
    print("Submission validation passed")
