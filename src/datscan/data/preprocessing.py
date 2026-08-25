"""Deterministic physical-space preprocessing for training and inference."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from ..utils.config import PreprocessConfig
from .nifti import LoadedNifti, load_nifti, resample_to_spacing


def normalize_intensity(volume: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    scale = float(np.percentile(positive, config.intensity_percentile))
    scale = max(scale, config.eps)
    normalized = np.clip(volume / scale, 0.0, config.clip_max)
    return np.asarray(normalized, dtype=np.float32)


def foreground_center(volume: np.ndarray, config: PreprocessConfig) -> Tuple[float, float, float]:
    positive = volume[volume > 0]
    if positive.size == 0:
        return tuple((np.asarray(volume.shape, dtype=float) - 1.0) / 2.0)
    threshold = max(float(np.quantile(positive, config.foreground_quantile)), float(positive.max()) * config.foreground_threshold_fraction)
    coordinates = np.argwhere(volume >= threshold)
    if coordinates.size == 0:
        return tuple((np.asarray(volume.shape, dtype=float) - 1.0) / 2.0)
    return tuple(float(x) for x in coordinates.mean(axis=0))


def crop_or_pad_center(volume: np.ndarray, output_shape: Tuple[int, int, int], center: Tuple[float, float, float], pad_value: float = 0.0) -> np.ndarray:
    """Extract a centered fixed-size tensor and pad out-of-bounds regions."""
    output = np.full(output_shape, pad_value, dtype=np.float32)
    source_center = np.rint(np.asarray(center)).astype(int)
    source_start = source_center - np.asarray(output_shape) // 2
    source_end = source_start + np.asarray(output_shape)
    source_slices = []
    target_slices = []
    for axis, (start, end, size) in enumerate(zip(source_start, source_end, volume.shape)):
        src_start = max(int(start), 0)
        src_end = min(int(end), int(size))
        dst_start = max(-int(start), 0)
        dst_end = dst_start + max(src_end - src_start, 0)
        source_slices.append(slice(src_start, src_end))
        target_slices.append(slice(dst_start, dst_end))
    if all(s.start < s.stop for s in source_slices):
        output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def preprocess_loaded(loaded: LoadedNifti, config: PreprocessConfig) -> np.ndarray:
    resampled = resample_to_spacing(loaded, config.target_spacing_mm)
    normalized = normalize_intensity(resampled.data, config)
    center = foreground_center(normalized, config)
    cropped = crop_or_pad_center(normalized, tuple(int(x) for x in config.output_shape), center, config.pad_value)
    if not np.isfinite(cropped).all():
        raise ValueError(f"Preprocessing produced non-finite values: {loaded.source_path}")
    return cropped[None, ...].astype(np.float32, copy=False)


def preprocess_nifti(path: str, config: PreprocessConfig) -> np.ndarray:
    return preprocess_loaded(load_nifti(path, canonical=True), config)


def preprocessing_fingerprint(config: PreprocessConfig) -> Dict[str, object]:
    return asdict(config)

