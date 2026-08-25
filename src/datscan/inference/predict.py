"""Batch-independent, deterministic prediction functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from ..data.preprocessing import preprocess_nifti
from ..utils.metrics import safe_probabilities
from ..models.resnet3d import build_model


def predict_paths(model: torch.nn.Module, paths: Sequence[str | Path], preprocess_config, device: torch.device) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for path in paths:
            tensor = torch.from_numpy(preprocess_nifti(str(path), preprocess_config)).unsqueeze(0).to(device)
            values.append(float(model(tensor).detach().cpu().item()))
    return expit(np.asarray(values, dtype=float))


def validate_submission(submission: pd.DataFrame, template: pd.DataFrame) -> None:
    if list(submission.columns) != ["uid", "is_pathologic"]:
        raise ValueError("Submission columns must be exactly ['uid', 'is_pathologic']")
    if len(submission) != len(template) or submission["uid"].tolist() != template["uid"].astype(str).tolist():
        raise ValueError("Submission UIDs or ordering do not match submission_format.csv")
    if submission["uid"].duplicated().any():
        raise ValueError("Submission UIDs are not unique")
    probabilities = submission["is_pathologic"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise ValueError("Submission probabilities must be finite and in [0, 1]")

