"""Small preprocessing shim used when the feature extractor is packaged."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import nibabel as nib
import numpy as np


@dataclass
class LoadedNifti:
    data: np.ndarray
    affine: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class PreprocessConfig:
    target_spacing_mm: float = 3.0
    output_shape: Tuple[int, int, int] = (96, 96, 96)
    foreground_quantile: float = 0.01
    foreground_threshold_fraction: float = 0.05
    intensity_percentile: float = 99.5
    clip_max: float = 2.0
    eps: float = 1.0e-6
    pad_value: float = 0.0


@dataclass(frozen=True)
class ROIConfig:
    enabled: bool = True
    roi_shape: Tuple[int, int, int] = (64, 64, 48)
    center_max_shift_fraction: float = 0.25


def _mapping_values(mapping: dict[str, Any] | None, cls):
    values = dict(mapping or {})
    if "output_shape" in values:
        values["output_shape"] = tuple(int(x) for x in values["output_shape"])
    if "roi_shape" in values:
        values["roi_shape"] = tuple(int(x) for x in values["roi_shape"])
    return cls(**{key: values[key] for key in values if key in cls.__dataclass_fields__})


PreprocessConfig.from_mapping = classmethod(lambda cls, mapping=None: _mapping_values(mapping, cls))
ROIConfig.from_mapping = classmethod(lambda cls, mapping=None: _mapping_values(mapping, cls))


def voxel_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
    values = np.sqrt((np.asarray(affine, dtype=float)[:3, :3] ** 2).sum(axis=0))
    return tuple(float(x) for x in values)


def load_nifti(path: str | Path, canonical: bool = True) -> LoadedNifti:
    image = nib.load(str(path))
    if canonical:
        image = nib.as_closest_canonical(image)
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"Expected a finite 3-D NIfTI: {path}")
    return LoadedNifti(data, np.asarray(image.affine, dtype=float), Path(path))


def resample_to_spacing(loaded: LoadedNifti, spacing_mm: float) -> LoadedNifti:
    from nibabel.processing import resample_to_output

    image = nib.Nifti1Image(loaded.data, loaded.affine)
    result = resample_to_output(image, voxel_sizes=(float(spacing_mm),) * 3, order=1)
    return LoadedNifti(np.asarray(result.get_fdata(dtype=np.float32), dtype=np.float32), np.asarray(result.affine, dtype=float), loaded.source_path)


def normalize_intensity(volume: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    scale = max(float(np.percentile(positive, config.intensity_percentile)), config.eps)
    return np.asarray(np.clip(volume / scale, 0.0, config.clip_max), dtype=np.float32)


def crop_or_pad_center(volume: np.ndarray, output_shape: Tuple[int, int, int], center, pad_value: float = 0.0) -> np.ndarray:
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


def roi_foreground_center(volume: np.ndarray, config: PreprocessConfig, roi_config: ROIConfig):
    positive = volume[np.isfinite(volume) & (volume > config.eps)]
    center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    if positive.size == 0:
        return tuple(float(x) for x in center)
    threshold = max(float(np.quantile(positive, min(config.foreground_quantile, 0.01))), config.eps)
    coordinates = np.argwhere(volume >= threshold)
    candidate = coordinates.mean(axis=0) if coordinates.size else center
    roi_shape = np.asarray(roi_config.roi_shape, dtype=float)
    max_shift = roi_shape * float(roi_config.center_max_shift_fraction)
    bounded = np.minimum(np.maximum(candidate, center - max_shift), center + max_shift)
    return tuple(float(x) for x in bounded)
