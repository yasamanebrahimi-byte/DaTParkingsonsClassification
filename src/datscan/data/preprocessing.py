"""Deterministic physical-space preprocessing for training and inference."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from ..utils.config import PreprocessConfig, ROIConfig
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


def roi_foreground_center(volume: np.ndarray, config: PreprocessConfig, roi_config: ROIConfig) -> Tuple[float, float, float]:
    """Find a stable crop center from foreground geometry, not peak uptake.

    All sufficiently positive support contributes to the centroid.  The
    resulting center is bounded around the physical-volume center by a
    fraction of the requested crop size, so a unilateral abnormal scan cannot
    shift the crop far enough to lose the weaker hemisphere.
    """
    positive = volume[np.isfinite(volume) & (volume > config.eps)]
    volume_center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    if positive.size == 0:
        return tuple(float(x) for x in volume_center)

    # A low support threshold deliberately includes both a strong and a weak
    # striatum.  It is based on geometry rather than the single brightest
    # voxel, while the normal foreground crop remains unchanged.
    threshold = max(float(np.quantile(positive, min(config.foreground_quantile, 0.01))), config.eps)
    coordinates = np.argwhere(volume >= threshold)
    center = coordinates.mean(axis=0) if coordinates.size else volume_center
    roi_shape = np.asarray(tuple(int(x) for x in roi_config.roi_shape), dtype=float)
    max_shift = np.maximum(roi_shape * float(roi_config.center_max_shift_fraction), 0.0)
    bounded = np.minimum(np.maximum(center, volume_center - max_shift), volume_center + max_shift)
    return tuple(float(x) for x in bounded)


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


def _resample_and_normalize(loaded: LoadedNifti, config: PreprocessConfig) -> np.ndarray:
    resampled = resample_to_spacing(loaded, config.target_spacing_mm)
    return normalize_intensity(resampled.data, config)


def preprocess_loaded_roi(loaded: LoadedNifti, config: PreprocessConfig, roi_config: ROIConfig) -> np.ndarray:
    """Preprocess one canonical scan into the fixed bilateral ROI tensor."""
    if not roi_config.enabled:
        raise ValueError("ROI preprocessing requested but roi.enabled is false")
    normalized = _resample_and_normalize(loaded, config)
    center = roi_foreground_center(normalized, config, roi_config)
    cropped = crop_or_pad_center(normalized, tuple(int(x) for x in roi_config.roi_shape), center, config.pad_value)
    if not np.isfinite(cropped).all():
        raise ValueError(f"ROI preprocessing produced non-finite values: {loaded.source_path}")
    return cropped[None, ...].astype(np.float32, copy=False)


def preprocess_loaded_views(
    loaded: LoadedNifti,
    config: PreprocessConfig,
    roi_config: ROIConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Return global and, when configured, ROI tensors from one base volume."""
    normalized = _resample_and_normalize(loaded, config)
    global_center = foreground_center(normalized, config)
    global_crop = crop_or_pad_center(normalized, tuple(int(x) for x in config.output_shape), global_center, config.pad_value)
    views = {"global": global_crop[None, ...].astype(np.float32, copy=False)}
    if roi_config is not None and roi_config.enabled:
        center = roi_foreground_center(normalized, config, roi_config)
        roi_crop = crop_or_pad_center(normalized, tuple(int(x) for x in roi_config.roi_shape), center, config.pad_value)
        views["roi"] = roi_crop[None, ...].astype(np.float32, copy=False)
    if not all(np.isfinite(value).all() for value in views.values()):
        raise ValueError(f"Preprocessing produced non-finite values: {loaded.source_path}")
    return views


def preprocess_nifti(
    path: str,
    config: PreprocessConfig,
    data_view: str = "global",
    roi_config: ROIConfig | None = None,
) -> np.ndarray:
    """Preprocess a scan for the requested ``global`` or ``roi`` view."""
    loaded = load_nifti(path, canonical=True)
    if data_view == "global":
        return preprocess_loaded(loaded, config)
    if data_view == "roi":
        if roi_config is None:
            raise ValueError("roi_config is required for the ROI data view")
        return preprocess_loaded_roi(loaded, config, roi_config)
    raise ValueError(f"Unknown data_view: {data_view}")


def preprocess_nifti_views(path: str, config: PreprocessConfig, roi_config: ROIConfig | None = None) -> Dict[str, np.ndarray]:
    return preprocess_loaded_views(load_nifti(path, canonical=True), config, roi_config)


def preprocessing_fingerprint(
    config: PreprocessConfig,
    data_view: str = "global",
    roi_config: ROIConfig | None = None,
) -> Dict[str, object]:
    fingerprint: Dict[str, object] = {"data_view": data_view, "preprocessing": asdict(config)}
    if roi_config is not None:
        fingerprint["roi"] = asdict(roi_config)
    return fingerprint
