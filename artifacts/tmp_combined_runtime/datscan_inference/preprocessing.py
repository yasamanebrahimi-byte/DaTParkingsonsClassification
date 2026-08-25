"""Dependency-light copy of canonical global and bilateral ROI preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class Config:
    target_spacing_mm: float = 3.0
    output_shape: Tuple[int, int, int] = (96, 96, 96)
    foreground_quantile: float = 0.01
    foreground_threshold_fraction: float = 0.05
    intensity_percentile: float = 99.5
    clip_max: float = 2.0
    eps: float = 1e-6
    pad_value: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "Config":
        values = dict(mapping or {})
        if "output_shape" in values:
            values["output_shape"] = tuple(int(size) for size in values["output_shape"])
        return cls(**{key: values[key] for key in values if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ROIConfig:
    enabled: bool = False
    roi_shape: Tuple[int, int, int] = (64, 64, 48)
    center_max_shift_fraction: float = 0.25

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "ROIConfig":
        values = dict(mapping or {})
        if "roi_shape" in values:
            values["roi_shape"] = tuple(int(size) for size in values["roi_shape"])
        return cls(**{key: values[key] for key in values if key in cls.__dataclass_fields__})


def _crop_or_pad(volume: np.ndarray, output_shape: Tuple[int, int, int], center: Tuple[float, float, float], pad_value: float) -> np.ndarray:
    output = np.full(output_shape, pad_value, dtype=np.float32)
    source_center = np.rint(np.asarray(center)).astype(int)
    source_start = source_center - np.asarray(output_shape) // 2
    source_end = source_start + np.asarray(output_shape)
    source_slices, target_slices = [], []
    for start, end, size in zip(source_start, source_end, volume.shape):
        src_start, src_end = max(int(start), 0), min(int(end), int(size))
        dst_start = max(-int(start), 0)
        source_slices.append(slice(src_start, src_end))
        target_slices.append(slice(dst_start, dst_start + max(src_end - src_start, 0)))
    if all(s.start < s.stop for s in source_slices):
        output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def _normalize(volume: np.ndarray, config: Config) -> np.ndarray:
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    scale = max(float(np.percentile(positive, config.intensity_percentile)), config.eps)
    return np.asarray(np.clip(volume / scale, 0.0, config.clip_max), dtype=np.float32)


def _foreground_center(volume: np.ndarray, config: Config) -> Tuple[float, float, float]:
    positive = volume[volume > 0]
    if positive.size == 0:
        return tuple((np.asarray(volume.shape, dtype=float) - 1.0) / 2.0)
    threshold = max(float(np.quantile(positive, config.foreground_quantile)), float(positive.max()) * config.foreground_threshold_fraction)
    coordinates = np.argwhere(volume >= threshold)
    return tuple(float(x) for x in coordinates.mean(axis=0)) if coordinates.size else tuple((np.asarray(volume.shape) - 1) / 2)


def _roi_center(volume: np.ndarray, config: Config, roi_config: ROIConfig) -> Tuple[float, float, float]:
    positive = volume[np.isfinite(volume) & (volume > config.eps)]
    volume_center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    if positive.size == 0:
        return tuple(float(x) for x in volume_center)
    threshold = max(float(np.quantile(positive, min(config.foreground_quantile, 0.01))), config.eps)
    coordinates = np.argwhere(volume >= threshold)
    center = coordinates.mean(axis=0) if coordinates.size else volume_center
    roi_shape = np.asarray(roi_config.roi_shape, dtype=float)
    max_shift = np.maximum(roi_shape * float(roi_config.center_max_shift_fraction), 0.0)
    bounded = np.minimum(np.maximum(center, volume_center - max_shift), volume_center + max_shift)
    return tuple(float(x) for x in bounded)


def _load_normalized(path: str | Path, config: Config) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")
    from nibabel.processing import resample_to_output

    resampled = resample_to_output(image, voxel_sizes=(config.target_spacing_mm,) * 3, order=1)
    volume = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"Non-finite NIfTI values: {path}")
    return _normalize(volume, config)


def preprocess_views(path: str | Path, config: Config, roi_config: ROIConfig | None = None) -> Dict[str, np.ndarray]:
    volume = _load_normalized(path, config)
    global_crop = _crop_or_pad(volume, config.output_shape, _foreground_center(volume, config), config.pad_value)
    views = {"global": global_crop[None, ...].astype(np.float32, copy=False)}
    if roi_config is not None and roi_config.enabled:
        roi_crop = _crop_or_pad(volume, roi_config.roi_shape, _roi_center(volume, config, roi_config), config.pad_value)
        views["roi"] = roi_crop[None, ...].astype(np.float32, copy=False)
    return views


def preprocess(path: str | Path, config: Config, data_view: str = "global", roi_config: ROIConfig | None = None) -> np.ndarray:
    views = preprocess_views(path, config, roi_config)
    if data_view not in views:
        raise ValueError(f"Requested unavailable data view: {data_view}")
    return views[data_view]
