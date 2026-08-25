"""Concise competition inference orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from .models import load_model
from .preprocessing import Config, preprocess
from .utils import validate_submission


def run_inference(root: Path) -> None:
    data_dir = root / "data"
    assets = root / "assets"
    template_path = data_dir / "submission_format.csv"
    output_path = root / "submission.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing submission template: {template_path}")
    checkpoint_paths = sorted(assets.glob("model_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError("No packaged model_*.pt assets found")
    print("Loading models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    preprocess_config = None
    for path in checkpoint_paths:
        model, raw_config = load_model(path, device)
        models.append(model)
        current_config = Config.from_mapping(raw_config)
        if preprocess_config is None:
            preprocess_config = current_config
        elif current_config != preprocess_config:
            raise ValueError("Packaged checkpoints use different preprocessing configurations")
    calibration_path = assets / "calibration.json"
    temperature = 1.0
    if calibration_path.exists():
        temperature = float(json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"])
    template = pd.read_csv(template_path, dtype={"uid": str})
    probabilities = []
    n = len(template)
    print("Models loaded")
    with torch.inference_mode():
        for index, uid in enumerate(template["uid"].astype(str)):
            path = data_dir / "niftis" / f"{uid}.nii.gz"
            if not path.exists():
                matches = list((data_dir / "niftis").rglob(f"{uid}.nii.gz"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected exactly one NIfTI for UID {uid}")
                path = matches[0]
            tensor = torch.from_numpy(preprocess(path, preprocess_config)).unsqueeze(0).to(device)
            logits = torch.stack([model(tensor).float().squeeze(0).cpu() for model in models])
            probabilities.append(float(expit(logits.mean().item() / temperature)))
            if n and (index + 1) in {max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4)}:
                print(f"{round((index + 1) * 100 / n)}% complete")
    output = pd.DataFrame({"uid": template["uid"].astype(str), "is_pathologic": np.clip(probabilities, 1e-6, 1 - 1e-6)})
    validate_submission(output, template)
    output.to_csv(output_path, index=False)
    print("Writing submission")
    print("Submission validation passed")
