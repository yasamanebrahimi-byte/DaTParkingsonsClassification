"""Checkpoint loading with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from ..models.resnet3d import build_model
from ..utils.config import ModelConfig, PreprocessConfig


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, PreprocessConfig, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    model_config = ModelConfig.from_mapping(payload.get("model"))
    preprocess_config = PreprocessConfig.from_mapping(payload.get("preprocess"))
    model = build_model(model_config.name, model_config.base_channels, model_config.groups, model_config.layers)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, preprocess_config, payload
