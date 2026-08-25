"""Configuration loading with small, dependency-light YAML support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def load_config(path: str | Path, base_path: str | Path | None = None) -> Dict[str, Any]:
    config = load_yaml(path)
    if base_path is not None:
        config = _deep_merge(load_yaml(base_path), config)
    return config


def get_path(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


@dataclass(frozen=True)
class PreprocessConfig:
    version: str = "1"
    target_spacing_mm: float = 3.0
    output_shape: Sequence[int] = (96, 96, 96)
    foreground_quantile: float = 0.01
    foreground_threshold_fraction: float = 0.05
    intensity_percentile: float = 99.5
    clip_max: float = 2.0
    eps: float = 1e-6
    pad_value: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "PreprocessConfig":
        values = dict(mapping or {})
        if "output_shape" in values:
            values["output_shape"] = tuple(int(x) for x in values["output_shape"])
        return cls(**{key: values[key] for key in values if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet3d"
    base_channels: int = 16
    groups: int = 8
    layers: Sequence[int] = (2, 2, 2, 2)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ModelConfig":
        values = dict(mapping or {})
        if "layers" in values:
            values["layers"] = tuple(int(x) for x in values["layers"])
        return cls(**{key: values[key] for key in values if key in cls.__dataclass_fields__})
