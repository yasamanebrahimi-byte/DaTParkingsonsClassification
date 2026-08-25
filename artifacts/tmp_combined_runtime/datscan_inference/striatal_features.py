"""Deterministic, image-derived quantitative DaT-SPECT features.

The extractor intentionally shares the image-model preprocessing foundation:
NIfTI data are loaded in canonical RAS orientation, resampled to isotropic
physical spacing, normalized per examination, and cropped with the existing
bilateral ROI center.  It does not use labels, dataset-wide statistics, or
acquisition metadata.

Axis convention
---------------
After :func:`nibabel.as_closest_canonical`, the array is RAS+.  Axis 0 is the
world X axis; its low-index half is the left side and its high-index half is
the right side because positive X points toward the right.  Axis 1 is the
world Y axis; its low-index half is posterior and its high-index half is
anterior.  These are geometric approximations, not anatomical segmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

try:  # The fallback keeps the feature extractor self-contained in submissions.
    from ..data.nifti import load_nifti, resample_to_spacing, voxel_spacing
    from ..data.preprocessing import crop_or_pad_center, normalize_intensity, roi_foreground_center
    from ..utils.config import PreprocessConfig, ROIConfig
except ImportError:  # pragma: no cover - exercised by the packaged inference runtime
    from .feature_support import (  # type: ignore
        PreprocessConfig,
        ROIConfig,
        crop_or_pad_center,
        load_nifti,
        normalize_intensity,
        resample_to_spacing,
        roi_foreground_center,
        voxel_spacing,
    )


LEFT_RIGHT_AXIS = 0
POSTERIOR_ANTERIOR_AXIS = 1


@dataclass(frozen=True)
class StriatalFeatureConfig:
    """Configuration for the compact clinical/image-derived feature set."""

    source: str = "roi"
    tissue_min_intensity: float = 1.0e-6
    epsilon: float = 1.0e-6
    background_method: str = "surrounding_tissue"
    background_border_fraction: float = 0.15
    background_exclude_relative: float = 0.40
    high_uptake_relative_thresholds: Sequence[float] = (0.40, 0.50, 0.60)
    primary_threshold: float = 0.50
    include_uptake: bool = True
    include_asymmetry: bool = True
    include_background_ratios: bool = True
    include_anterior_posterior: bool = True
    include_morphology: bool = True
    include_shape: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "StriatalFeatureConfig":
        values = dict(mapping or {})
        background = dict(values.get("background") or {})
        high_uptake = dict(values.get("high_uptake") or {})
        include = dict(values.get("include") or {})
        aliases = {
            "background_method": background.get("method", values.get("background_method", cls.background_method)),
            "background_border_fraction": background.get(
                "border_fraction", values.get("background_border_fraction", cls.background_border_fraction)
            ),
            "background_exclude_relative": background.get(
                "exclude_relative", values.get("background_exclude_relative", cls.background_exclude_relative)
            ),
            "epsilon": background.get("epsilon", values.get("epsilon", cls.epsilon)),
            "high_uptake_relative_thresholds": high_uptake.get(
                "relative_thresholds",
                values.get("high_uptake_relative_thresholds", cls.high_uptake_relative_thresholds),
            ),
            "primary_threshold": high_uptake.get(
                "primary_threshold", values.get("primary_threshold", cls.primary_threshold)
            ),
            "include_uptake": include.get("uptake", values.get("include_uptake", cls.include_uptake)),
            "include_asymmetry": include.get("asymmetry", values.get("include_asymmetry", cls.include_asymmetry)),
            "include_background_ratios": include.get(
                "background_ratios", values.get("include_background_ratios", cls.include_background_ratios)
            ),
            "include_anterior_posterior": include.get(
                "anterior_posterior", values.get("include_anterior_posterior", cls.include_anterior_posterior)
            ),
            "include_morphology": include.get("morphology", values.get("include_morphology", cls.include_morphology)),
            "include_shape": include.get("shape", values.get("include_shape", cls.include_shape)),
        }
        direct = {key: values[key] for key in ("source", "tissue_min_intensity") if key in values}
        direct.update(aliases)
        direct["high_uptake_relative_thresholds"] = tuple(float(x) for x in direct["high_uptake_relative_thresholds"])
        direct["source"] = str(direct.get("source", cls.source))
        config = cls(**direct)
        config.validate()
        return config

    def validate(self) -> None:
        if self.source != "roi":
            raise ValueError("Only the canonical bilateral ROI feature source is supported")
        if self.epsilon <= 0 or self.tissue_min_intensity < 0:
            raise ValueError("epsilon must be positive and tissue_min_intensity must be non-negative")
        if not 0 < self.background_border_fraction < 0.5:
            raise ValueError("background border_fraction must be in (0, 0.5)")
        if not 0 < self.background_exclude_relative < 1:
            raise ValueError("background exclude_relative must be in (0, 1)")
        thresholds = tuple(float(x) for x in self.high_uptake_relative_thresholds)
        if not thresholds or any(not 0 < value < 1 for value in thresholds):
            raise ValueError("high_uptake relative thresholds must be non-empty and in (0, 1)")
        if not 0 < self.primary_threshold < 1:
            raise ValueError("primary_threshold must be in (0, 1)")


def _threshold_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "_")


def _finite_positive(values: np.ndarray, minimum: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values) & (values > float(minimum))]


def _stats(values: np.ndarray, prefix: str, output: dict[str, float], minimum: float) -> None:
    values = _finite_positive(values, minimum)
    names = ("mean", "std", "max", "p50", "p75", "p90", "p95", "p99", "p99_5")
    if values.size == 0:
        for name in names:
            output[f"{prefix}_{name}"] = 0.0
        return
    output[f"{prefix}_mean"] = float(values.mean())
    output[f"{prefix}_std"] = float(values.std(ddof=0))
    output[f"{prefix}_max"] = float(values.max())
    percentiles = np.percentile(values, [50, 75, 90, 95, 99, 99.5])
    for name, value in zip(("p50", "p75", "p90", "p95", "p99", "p99_5"), percentiles):
        output[f"{prefix}_{name}"] = float(value)


def _side_slices(shape: Sequence[int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    midpoint = int(shape[LEFT_RIGHT_AXIS] // 2)
    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[LEFT_RIGHT_AXIS] = slice(0, midpoint)
    right[LEFT_RIGHT_AXIS] = slice(midpoint, int(shape[LEFT_RIGHT_AXIS]))
    return tuple(left), tuple(right)


def _asymmetry(left: float, right: float, epsilon: float) -> tuple[float, float]:
    absolute = abs(float(left) - float(right))
    normalized = absolute / (0.5 * (float(left) + float(right)) + epsilon)
    return float(absolute), float(normalized)


def _background_estimate(roi: np.ndarray, tissue_mask: np.ndarray, reference_max: float, config: StriatalFeatureConfig) -> float:
    if config.background_method != "surrounding_tissue":
        raise ValueError(f"Unknown background method: {config.background_method}")
    shape = np.asarray(roi.shape, dtype=int)
    border = np.zeros(shape, dtype=bool)
    widths = np.maximum(np.ceil(shape * config.background_border_fraction).astype(int), 1)
    for axis, width in enumerate(widths):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(0, int(width))
        high[axis] = slice(int(shape[axis] - width), int(shape[axis]))
        border[tuple(low)] = True
        border[tuple(high)] = True
    high_mask = roi >= reference_max * config.background_exclude_relative if reference_max > config.epsilon else np.zeros_like(roi, dtype=bool)
    candidates = roi[tissue_mask & border & ~high_mask]
    if candidates.size < 8:
        candidates = roi[tissue_mask & ~high_mask]
    if candidates.size == 0:
        candidates = roi[tissue_mask]
    if candidates.size == 0:
        return 0.0
    return float(np.median(candidates))


def _connected_component_volumes(mask: np.ndarray, voxel_volume: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:].astype(np.float64)
    order = np.argsort(-counts, kind="stable")
    return counts[order] * voxel_volume, labels


def _shape_features(mask: np.ndarray, spacing: tuple[float, float, float], prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    if not np.any(mask):
        for axis in "xyz":
            output[f"{prefix}_bounding_box_extent_{axis}_mm"] = 0.0
        for index in (1, 2, 3):
            output[f"{prefix}_principal_axis_length_{index}_mm"] = 0.0
        output[f"{prefix}_lambda2_over_lambda1"] = 0.0
        output[f"{prefix}_lambda3_over_lambda1"] = 0.0
        output[f"{prefix}_elongation_ratio"] = 0.0
        output[f"{prefix}_compactness_like"] = 0.0
        return output
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count < 1:
        return _shape_features(np.zeros_like(mask), spacing, prefix)
    counts = np.bincount(labels.ravel())[1:]
    component = int(np.argmax(counts)) + 1
    coordinates = np.argwhere(labels == component)
    extent_voxels = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    extent_mm = extent_voxels.astype(float) * np.asarray(spacing, dtype=float)
    for axis, value in zip("xyz", extent_mm):
        output[f"{prefix}_bounding_box_extent_{axis}_mm"] = float(value)

    physical_coordinates = coordinates.astype(float) * np.asarray(spacing, dtype=float)
    if len(physical_coordinates) > 1:
        covariance = np.cov(physical_coordinates, rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
        eigenvalues = np.sort(np.maximum(np.asarray(eigenvalues, dtype=float), 0.0))[::-1]
        lengths = 4.0 * np.sqrt(eigenvalues)
    else:
        eigenvalues = np.zeros(3, dtype=float)
        lengths = np.zeros(3, dtype=float)
    for index, value in enumerate(lengths, start=1):
        output[f"{prefix}_principal_axis_length_{index}_mm"] = float(value)
    denominator = float(eigenvalues[0] + 1.0e-12)
    output[f"{prefix}_lambda2_over_lambda1"] = float(eigenvalues[1] / denominator)
    output[f"{prefix}_lambda3_over_lambda1"] = float(eigenvalues[2] / denominator)
    output[f"{prefix}_elongation_ratio"] = float(lengths[0] / (lengths[1] + 1.0e-12)) if lengths[0] > 0 else 0.0
    bbox_volume = float(np.prod(extent_mm))
    output[f"{prefix}_compactness_like"] = float((len(coordinates) * np.prod(spacing)) / (bbox_volume + 1.0e-12))
    return output


def _side_component(mask: np.ndarray, side: str) -> np.ndarray:
    side_slices = _side_slices(mask.shape)[0 if side == "left" else 1]
    side_mask = np.zeros_like(mask, dtype=bool)
    side_mask[side_slices] = mask[side_slices]
    return side_mask


def extract_striatal_features_from_roi(
    roi: np.ndarray,
    spacing_mm: Sequence[float] | float,
    config: StriatalFeatureConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract an ordered feature dictionary from a normalized bilateral ROI.

    ``roi`` is expected to be a 3-D normalized crop.  The function is public
    so tests and inference can reuse exactly the same measurements without
    loading a NIfTI twice.
    """
    feature_config = config if isinstance(config, StriatalFeatureConfig) else StriatalFeatureConfig.from_mapping(config)
    feature_config.validate()
    array = np.asarray(roi, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D bilateral ROI, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("ROI contains non-finite values")
    if np.isscalar(spacing_mm):
        spacing = (float(spacing_mm),) * 3
    else:
        spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError("spacing_mm must contain three positive finite values")
    voxel_volume = float(np.prod(spacing))

    tissue_mask = np.isfinite(array) & (array > feature_config.tissue_min_intensity)
    tissue_values = _finite_positive(array, feature_config.tissue_min_intensity)
    reference_max = float(tissue_values.max()) if tissue_values.size else 0.0
    output: dict[str, float] = {}

    left_slice, right_slice = _side_slices(array.shape)
    left = array[left_slice]
    right = array[right_slice]
    left_mask = tissue_mask[left_slice]
    right_mask = tissue_mask[right_slice]
    left_values = left[left_mask]
    right_values = right[right_mask]

    thresholds = tuple(float(value) for value in feature_config.high_uptake_relative_thresholds)
    primary = float(feature_config.primary_threshold)
    primary_level = reference_max * primary if reference_max > feature_config.epsilon else np.inf
    primary_mask = tissue_mask & (array >= primary_level)

    if feature_config.include_uptake:
        _stats(array[tissue_mask], "roi", output, feature_config.tissue_min_intensity)
        _stats(left_values, "left", output, feature_config.tissue_min_intensity)
        _stats(right_values, "right", output, feature_config.tissue_min_intensity)
        side_means = [float(left_values.mean()) if left_values.size else 0.0, float(right_values.mean()) if right_values.size else 0.0]
        output["mean_bilateral_uptake"] = float(np.mean(side_means))
        output["minimum_side_uptake"] = float(min(side_means))
        output["maximum_side_uptake"] = float(max(side_means))
        output["roi_positive_fraction"] = float(tissue_mask.mean())
        output["roi_high_uptake_fraction"] = float(primary_mask.sum() / max(int(tissue_mask.sum()), 1))
        for threshold in thresholds:
            level = reference_max * threshold if reference_max > feature_config.epsilon else np.inf
            mask = tissue_mask & (array >= level)
            token = _threshold_token(threshold)
            output[f"high_uptake_fraction_rel_{token}"] = float(mask.sum() / max(int(tissue_mask.sum()), 1))

    if feature_config.include_asymmetry:
        metrics = {
            "mean": (float(left_values.mean()) if left_values.size else 0.0, float(right_values.mean()) if right_values.size else 0.0),
            "p95": (float(np.percentile(left_values, 95)) if left_values.size else 0.0, float(np.percentile(right_values, 95)) if right_values.size else 0.0),
            "high_uptake_volume_mm3": (float((primary_mask[left_slice]).sum() * voxel_volume), float((primary_mask[right_slice]).sum() * voxel_volume)),
        }
        for name, (left_value, right_value) in metrics.items():
            absolute, normalized = _asymmetry(left_value, right_value, feature_config.epsilon)
            output[f"asymmetry_{name}_abs"] = absolute
            output[f"asymmetry_{name}_normalized"] = normalized

    if feature_config.include_background_ratios:
        background = _background_estimate(array, tissue_mask, reference_max, feature_config)
        output["background_uptake"] = background
        left_mean = float(left_values.mean()) if left_values.size else 0.0
        right_mean = float(right_values.mean()) if right_values.size else 0.0
        bilateral_mean = float(tissue_values.mean()) if tissue_values.size else 0.0
        denominator = background + feature_config.epsilon
        output["left_striatum_to_background"] = float(left_mean / denominator)
        output["right_striatum_to_background"] = float(right_mean / denominator)
        output["bilateral_striatum_to_background"] = float(bilateral_mean / denominator)
        output["minimum_side_to_background"] = float(min(left_mean, right_mean) / denominator)
        output["left_sbr_like"] = float((left_mean - background) / denominator)
        output["right_sbr_like"] = float((right_mean - background) / denominator)
        output["bilateral_sbr_like"] = float((bilateral_mean - background) / denominator)

    if feature_config.include_anterior_posterior:
        ap_midpoint = int(array.shape[POSTERIOR_ANTERIOR_AXIS] // 2)
        side_ap: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, side_array, side_mask in (("left", left, left_mask), ("right", right, right_mask)):
            posterior = [slice(None)] * 3
            anterior = [slice(None)] * 3
            posterior[POSTERIOR_ANTERIOR_AXIS] = slice(0, ap_midpoint)
            anterior[POSTERIOR_ANTERIOR_AXIS] = slice(ap_midpoint, array.shape[POSTERIOR_ANTERIOR_AXIS])
            posterior_values = side_array[tuple(posterior)][side_mask[tuple(posterior)]]
            anterior_values = side_array[tuple(anterior)][side_mask[tuple(anterior)]]
            side_ap[name] = (posterior_values, anterior_values)
            posterior_mean = float(posterior_values.mean()) if posterior_values.size else 0.0
            anterior_mean = float(anterior_values.mean()) if anterior_values.size else 0.0
            output[f"{name}_posterior_uptake"] = posterior_mean
            output[f"{name}_anterior_uptake"] = anterior_mean
            output[f"{name}_posterior_to_anterior"] = float(posterior_mean / (anterior_mean + feature_config.epsilon))
        ratios = [output["left_posterior_to_anterior"], output["right_posterior_to_anterior"]]
        output["minimum_posterior_to_anterior"] = float(min(ratios))
        output["mean_posterior_to_anterior"] = float(np.mean(ratios))

    if feature_config.include_morphology or feature_config.include_shape:
        component_volumes, _ = _connected_component_volumes(primary_mask, voxel_volume)
        if feature_config.include_morphology:
            output["high_uptake_voxel_count"] = float(primary_mask.sum())
            output["high_uptake_physical_volume_mm3"] = float(primary_mask.sum() * voxel_volume)
            output["left_high_uptake_volume_mm3"] = float(primary_mask[left_slice].sum() * voxel_volume)
            output["right_high_uptake_volume_mm3"] = float(primary_mask[right_slice].sum() * voxel_volume)
            output["number_of_connected_components"] = float(len(component_volumes))
            output["largest_component_volume_mm3"] = float(component_volumes[0]) if len(component_volumes) else 0.0
            output["second_largest_component_volume_mm3"] = float(component_volumes[1]) if len(component_volumes) > 1 else 0.0
            output["component_volume_ratio"] = float(component_volumes[0] / (component_volumes[1] + feature_config.epsilon)) if len(component_volumes) > 1 else (1.0 if len(component_volumes) else 0.0)
            for threshold in thresholds:
                level = reference_max * threshold if reference_max > feature_config.epsilon else np.inf
                mask = tissue_mask & (array >= level)
                token = _threshold_token(threshold)
                volumes, _ = _connected_component_volumes(mask, voxel_volume)
                output[f"high_uptake_volume_mm3_rel_{token}"] = float(mask.sum() * voxel_volume)
                output[f"number_of_connected_components_rel_{token}"] = float(len(volumes))
        if feature_config.include_shape:
            output.update(_shape_features(_side_component(primary_mask, "left"), spacing, "left"))
            output.update(_shape_features(_side_component(primary_mask, "right"), spacing, "right"))

    # The insertion order is part of the reproducibility contract.  Sorting
    # also makes CSV output independent of implementation dictionary details.
    result = {name: float(value) for name, value in sorted(output.items())}
    invalid = [name for name, value in result.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"Feature extraction produced non-finite values: {invalid}")
    return result


def extract_striatal_features(
    path: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    roi_config: ROIConfig | None = None,
    feature_config: StriatalFeatureConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Load one scan and extract its deterministic quantitative features."""
    preprocess = preprocess_config or PreprocessConfig()
    roi = roi_config or ROIConfig(enabled=True)
    if not roi.enabled:
        raise ValueError("Quantitative striatal features require an enabled bilateral ROI")
    loaded = load_nifti(path, canonical=True)
    resampled = resample_to_spacing(loaded, preprocess.target_spacing_mm)
    normalized = normalize_intensity(resampled.data, preprocess)
    center = roi_foreground_center(normalized, preprocess, roi)
    cropped = crop_or_pad_center(normalized, tuple(int(x) for x in roi.roi_shape), center, preprocess.pad_value)
    if not np.isfinite(cropped).all():
        raise ValueError(f"Feature preprocessing produced non-finite values: {path}")
    return extract_striatal_features_from_roi(cropped, voxel_spacing(resampled.affine), feature_config)


def feature_family(name: str) -> str:
    """Map a feature name to an interpretable ablation family."""
    lower = str(name).lower()
    if any(token in lower for token in ("bounding_box", "principal_axis", "lambda", "elongation", "compactness")):
        return "shape"
    if any(token in lower for token in ("posterior", "anterior", "posterior_to_anterior")):
        return "anterior_posterior"
    if any(token in lower for token in ("background", "sbr_like")):
        return "background_ratio"
    if "asymmetry" in lower:
        return "asymmetry"
    if any(token in lower for token in ("component", "high_uptake_volume", "voxel_count", "physical_volume")):
        return "morphology"
    if lower.startswith(("left_", "right_", "minimum_side", "maximum_side", "mean_bilateral")):
        return "left_right"
    return "uptake"


def select_feature_columns(columns: Iterable[str], families: Sequence[str] | None = None) -> list[str]:
    selected = [str(column) for column in columns]
    if families:
        wanted = {str(family).strip() for family in families if str(family).strip()}
        selected = [column for column in selected if feature_family(column) in wanted]
    if not selected:
        raise ValueError("No quantitative feature columns remain after family selection")
    return selected


def validate_feature_frame(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return per-feature validation statistics and human-readable issues."""
    excluded = {"uid", "label", "target", "fold"}
    columns = list(feature_columns or [column for column in frame.columns if column not in excluded])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame is missing columns: {missing}")
    issues: list[str] = []
    rows = []
    duplicate_names: dict[tuple[float, ...], str] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        unique = np.unique(values[finite])
        standard_deviation = float(np.std(values[finite], ddof=0)) if finite.any() else float("nan")
        rows.append(
            {
                "feature": column,
                "min": float(np.min(values[finite])) if finite.any() else float("nan"),
                "max": float(np.max(values[finite])) if finite.any() else float("nan"),
                "mean": float(np.mean(values[finite])) if finite.any() else float("nan"),
                "std": standard_deviation,
                "missing_count": int((~finite).sum()),
                "unique_count": int(len(unique)),
                "family": feature_family(column),
            }
        )
        if not finite.all():
            issues.append(f"{column}: contains {int((~finite).sum())} NaN or infinite values")
        if len(unique) <= 1:
            issues.append(f"{column}: constant or empty")
        elif standard_deviation <= 1.0e-12:
            issues.append(f"{column}: near-zero variance")
        key = tuple(np.round(values, 12)) if finite.all() else None
        if key is not None and key in duplicate_names:
            issues.append(f"{column}: duplicate values with {duplicate_names[key]}")
        elif key is not None:
            duplicate_names[key] = column
    return pd.DataFrame(rows), issues


def write_feature_validation_report(stats: pd.DataFrame, issues: Sequence[str], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Striatal feature validation",
        "",
        "This report describes deterministic image-derived features. Acquisition metadata are not included in the primary feature model.",
        "",
        "## Issues",
        "",
    ]
    lines.extend([f"- {issue}" for issue in issues] or ["- None detected."])
    lines.extend(["", "## Feature statistics", ""])
    columns = list(stats.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in stats.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.8g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
