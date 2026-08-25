"""Offline competition inference for global-only or global + ROI packages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

from .models import load_model_details
from .preprocessing import Config, ROIConfig, preprocess_views
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
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Calibration temperature must be finite and positive")
    return float(expit(logit(np.clip(probability, 1e-6, 1.0 - 1e-6)) / temperature))


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

    calibration_path = assets / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {"temperature": 1.0, "enabled": False}
    temperature = float(calibration.get("temperature", 1.0))
    calibration_enabled = bool(calibration.get("enabled", True))
    after_ensemble = bool(manifest and manifest.get("calibration_stage") == "after_ensemble") or calibration.get("stage") == "after_ensemble"

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
            views = preprocess_views(path, preprocess_config, roi_config if roi_models else None)
            global_tensor = torch.from_numpy(views["global"]).unsqueeze(0).to(device)
            global_logits = torch.stack([model(global_tensor).float().squeeze(0).cpu() for model in global_models])
            if roi_models:
                roi_tensor = torch.from_numpy(views["roi"]).unsqueeze(0).to(device)
                roi_logits = torch.stack([model(roi_tensor).float().squeeze(0).cpu() for model in roi_models])
                global_probability = float(expit(global_logits.mean().item()))
                roi_probability = float(expit(roi_logits.mean().item()))
                probability = global_weight * global_probability + roi_weight * roi_probability
                if calibration_enabled and after_ensemble:
                    probability = _temperature_probability(probability, temperature)
            else:
                probability = float(expit(global_logits.mean().item() / temperature)) if calibration_enabled else float(expit(global_logits.mean().item()))
            probabilities.append(float(np.clip(probability, 1e-6, 1.0 - 1e-6)))
            if n and (index + 1) in {max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4)}:
                print(f"{round((index + 1) * 100 / n)}% complete")
    output = pd.DataFrame({"uid": template["uid"].astype(str), "is_pathologic": probabilities})
    validate_submission(output, template)
    output.to_csv(output_path, index=False)
    print("Writing submission")
    print("Submission validation passed")
