"""Inference copy of canonical preprocessing; kept dependency-light and deterministic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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


def preprocess(path: str | Path, config: Config) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")
    from nibabel.processing import resample_to_output

    resampled = resample_to_output(image, voxel_sizes=(config.target_spacing_mm,) * 3, order=1)
    volume = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"Non-finite NIfTI values: {path}")
    positive = volume[volume > 0]
    if positive.size:
        scale = max(float(np.percentile(positive, config.intensity_percentile)), config.eps)
        volume = np.clip(volume / scale, 0.0, config.clip_max)
        threshold = max(float(np.quantile(volume[volume > 0], config.foreground_quantile)), float(volume.max()) * config.foreground_threshold_fraction)
        coordinates = np.argwhere(volume >= threshold)
        center = tuple(float(x) for x in coordinates.mean(axis=0)) if coordinates.size else tuple((np.asarray(volume.shape) - 1) / 2)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
        center = tuple((np.asarray(volume.shape) - 1) / 2)
    return _crop_or_pad(volume, config.output_shape, center, config.pad_value)[None, ...]
