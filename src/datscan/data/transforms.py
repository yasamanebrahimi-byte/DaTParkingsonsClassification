"""Torch-only training augmentations applied after deterministic preprocessing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from torch.nn import functional as F


class MildVolumeAugmentation:
    """Conservative intensity and geometric augmentations for SPECT volumes.

    This is the historical augmentation used by the baseline and existing
    experiments. Keep it separate from the scanner-robust experiment so
    comparisons remain controlled.
    """

    def __init__(self, flip_probability: float = 0.5, noise_std: float = 0.015) -> None:
        self.flip_probability = flip_probability
        self.noise_std = noise_std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor
        if torch.rand(()) < self.flip_probability:
            # Canonical RAS convention places the left/right axis first in the
            # volume tensor [channel, x, y, z].
            result = torch.flip(result, dims=(-3,))
        scale = 1.0 + (torch.rand((), device=result.device) - 0.5) * 0.10
        result = result * scale
        if torch.rand((), device=result.device) < 0.35:
            gamma = 0.95 + torch.rand((), device=result.device) * 0.10
            result = torch.clamp(result, min=0).pow(gamma)
        if torch.rand((), device=result.device) < 0.25:
            result = result + torch.randn_like(result) * self.noise_std
        return torch.clamp(result, min=0.0)


def _check_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _check_range(low: float, high: float, name: str, *, nonnegative: bool = False) -> tuple[float, float]:
    low, high = float(low), float(high)
    if not (torch.isfinite(torch.tensor(low)) and torch.isfinite(torch.tensor(high))):
        raise ValueError(f"{name} must be finite")
    if low > high:
        raise ValueError(f"{name} minimum must not exceed maximum")
    if nonnegative and low < 0:
        raise ValueError(f"{name} must be nonnegative")
    return low, high


def _uniform(low: float, high: float, device: torch.device) -> float:
    if low == high:
        return low
    return float((low + (high - low) * torch.rand((), device=device)).item())


def _should_apply(probability: float, device: torch.device) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(torch.rand((), device=device).item() < probability)


def _as_batched_volume(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 4:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 5:
        return tensor, False
    raise ValueError(f"Expected [C, X, Y, Z] or [B, C, X, Y, Z], got {tuple(tensor.shape)}")


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(int(torch.ceil(torch.tensor(3.0 * sigma)).item()), 1)
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable, grouped 3D Gaussian smoothing to a 5D tensor."""
    channels = int(volume.shape[1])
    kernel = _gaussian_kernel(max(float(sigma), 1e-4), volume.device, volume.dtype)
    result = volume

    for axis in (0, 1, 2):
        radius = (kernel.numel() - 1) // 2
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        weight = kernel.reshape(shape).expand(channels, 1, *shape[2:])
        padding = [0, 0, 0]
        padding[axis] = radius
        result = F.conv3d(result, weight, padding=tuple(padding), groups=channels)
    return result


def _affine_3d(volume: torch.Tensor, rotation_degrees: torch.Tensor, translation_voxels: torch.Tensor) -> torch.Tensor:
    """Apply one centered 3D affine perturbation with zero padding."""
    radians = rotation_degrees * (torch.pi / 180.0)
    rx, ry, rz = radians
    zeros = torch.zeros((), device=volume.device, dtype=volume.dtype)
    ones = torch.ones((), device=volume.device, dtype=volume.dtype)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    rotation_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros]),
            torch.stack([zeros, cx, -sx]),
            torch.stack([zeros, sx, cx]),
        ]
    )
    rotation_y = torch.stack(
        [
            torch.stack([cy, zeros, sy]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([-sy, zeros, cy]),
        ]
    )
    rotation_z = torch.stack(
        [
            torch.stack([cz, -sz, zeros]),
            torch.stack([sz, cz, zeros]),
            torch.stack([zeros, zeros, ones]),
        ]
    )
    rotation = rotation_x @ rotation_y @ rotation_z
    sizes = torch.tensor(volume.shape[-3:], device=volume.device, dtype=volume.dtype)
    normalized_translation = 2.0 * translation_voxels / sizes
    theta = torch.cat([rotation, normalized_translation.reshape(3, 1)], dim=1).unsqueeze(0)
    theta = theta.expand(volume.shape[0], -1, -1)
    grid = F.affine_grid(theta, volume.shape, align_corners=False)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class ScannerRobustVolumeAugmentation:
    """Realistic scanner/reconstruction perturbations for preprocessed volumes.

    The transform accepts the dataset's ``[C, X, Y, Z]`` tensors and also
    supports batched ``[B, C, X, Y, Z]`` tensors for convenient diagnostics.
    Every call samples its own parameters. ``last_trace`` is populated with
    human-readable diagnostics for local augmentation inspection only.
    """

    _PRESETS: dict[str, dict[str, dict[str, float]]] = {
        "mild": {
            "intensity_scale": {"probability": 0.6, "min": 0.90, "max": 1.10},
            "gamma": {"probability": 0.4, "min": 0.90, "max": 1.10},
            "gaussian_noise": {"probability": 0.35, "std_min": 0.005, "std_max": 0.020},
            "gaussian_blur": {"probability": 0.25, "sigma_min": 0.25, "sigma_max": 0.75},
            "resolution_degradation": {"probability": 0.20, "scale_min": 0.80, "scale_max": 1.0},
            "poisson_noise": {"probability": 0.0, "count_scale_min": 100.0, "count_scale_max": 500.0},
            "affine": {"probability": 0.25, "max_rotation_degrees": 3.0, "max_translation_voxels": 2.0},
        },
        "moderate": {
            "intensity_scale": {"probability": 0.7, "min": 0.85, "max": 1.15},
            "gamma": {"probability": 0.5, "min": 0.85, "max": 1.15},
            "gaussian_noise": {"probability": 0.5, "std_min": 0.005, "std_max": 0.040},
            "gaussian_blur": {"probability": 0.5, "sigma_min": 0.25, "sigma_max": 1.25},
            "resolution_degradation": {"probability": 0.4, "scale_min": 0.65, "scale_max": 1.0},
            "poisson_noise": {"probability": 0.3, "count_scale_min": 50.0, "count_scale_max": 500.0},
            "affine": {"probability": 0.4, "max_rotation_degrees": 5.0, "max_translation_voxels": 4.0},
        },
        "strong": {
            "intensity_scale": {"probability": 0.8, "min": 0.80, "max": 1.20},
            "gamma": {"probability": 0.65, "min": 0.80, "max": 1.20},
            "gaussian_noise": {"probability": 0.65, "std_min": 0.005, "std_max": 0.060},
            "gaussian_blur": {"probability": 0.65, "sigma_min": 0.30, "sigma_max": 1.50},
            "resolution_degradation": {"probability": 0.55, "scale_min": 0.55, "scale_max": 1.0},
            "poisson_noise": {"probability": 0.4, "count_scale_min": 30.0, "count_scale_max": 300.0},
            "affine": {"probability": 0.5, "max_rotation_degrees": 6.0, "max_translation_voxels": 5.0},
        },
    }

    def __init__(
        self,
        *,
        flip_probability: float = 0.5,
        intensity_scale_probability: float = 0.7,
        intensity_scale_min: float = 0.85,
        intensity_scale_max: float = 1.15,
        gamma_probability: float = 0.5,
        gamma_min: float = 0.85,
        gamma_max: float = 1.15,
        gaussian_noise_probability: float = 0.5,
        noise_std_min: float = 0.005,
        noise_std_max: float = 0.04,
        gaussian_blur_probability: float = 0.5,
        sigma_min: float = 0.25,
        sigma_max: float = 1.25,
        resolution_degradation_probability: float = 0.4,
        scale_min: float = 0.65,
        scale_max: float = 1.0,
        poisson_probability: float = 0.3,
        count_scale_min: float = 50.0,
        count_scale_max: float = 500.0,
        additive_offset_probability: float = 0.0,
        offset_min: float = -0.002,
        offset_max: float = 0.002,
        affine_probability: float = 0.4,
        max_rotation_degrees: float = 5.0,
        max_translation_voxels: float = 4.0,
        severity: str = "moderate",
    ) -> None:
        self.flip_probability = _check_probability(flip_probability, "flip_probability")
        self.intensity_scale_probability = _check_probability(intensity_scale_probability, "intensity_scale_probability")
        self.intensity_scale_min, self.intensity_scale_max = _check_range(intensity_scale_min, intensity_scale_max, "intensity_scale")
        self.gamma_probability = _check_probability(gamma_probability, "gamma_probability")
        self.gamma_min, self.gamma_max = _check_range(gamma_min, gamma_max, "gamma")
        self.gaussian_noise_probability = _check_probability(gaussian_noise_probability, "gaussian_noise_probability")
        self.noise_std_min, self.noise_std_max = _check_range(noise_std_min, noise_std_max, "noise_std", nonnegative=True)
        self.gaussian_blur_probability = _check_probability(gaussian_blur_probability, "gaussian_blur_probability")
        self.sigma_min, self.sigma_max = _check_range(sigma_min, sigma_max, "sigma", nonnegative=True)
        self.resolution_degradation_probability = _check_probability(resolution_degradation_probability, "resolution_degradation_probability")
        self.scale_min, self.scale_max = _check_range(scale_min, scale_max, "resolution scale", nonnegative=True)
        if self.scale_max > 1.0:
            raise ValueError("resolution scale_max must be <= 1")
        self.poisson_probability = _check_probability(poisson_probability, "poisson_probability")
        self.count_scale_min, self.count_scale_max = _check_range(count_scale_min, count_scale_max, "count_scale", nonnegative=True)
        self.additive_offset_probability = _check_probability(additive_offset_probability, "additive_offset_probability")
        self.offset_min, self.offset_max = _check_range(offset_min, offset_max, "offset")
        self.affine_probability = _check_probability(affine_probability, "affine_probability")
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.max_translation_voxels = float(max_translation_voxels)
        if self.max_rotation_degrees < 0 or self.max_translation_voxels < 0:
            raise ValueError("affine bounds must be nonnegative")
        self.severity = str(severity)
        self.last_trace: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "ScannerRobustVolumeAugmentation":
        values = dict(config or {})
        severity = str(values.get("severity", "moderate")).lower()
        if severity not in cls._PRESETS:
            raise ValueError(f"Unknown scanner augmentation severity: {severity}")
        preset = deepcopy(cls._PRESETS[severity])
        for section in preset:
            override = values.get(section)
            if isinstance(override, Mapping):
                preset[section].update(override)
        offset = values.get("additive_offset")
        offset = offset if isinstance(offset, Mapping) else {}

        return cls(
            severity=severity,
            flip_probability=values.get("flip_probability", 0.5),
            intensity_scale_probability=preset["intensity_scale"]["probability"],
            intensity_scale_min=preset["intensity_scale"]["min"],
            intensity_scale_max=preset["intensity_scale"]["max"],
            gamma_probability=preset["gamma"]["probability"],
            gamma_min=preset["gamma"]["min"],
            gamma_max=preset["gamma"]["max"],
            gaussian_noise_probability=preset["gaussian_noise"]["probability"],
            noise_std_min=preset["gaussian_noise"]["std_min"],
            noise_std_max=preset["gaussian_noise"]["std_max"],
            gaussian_blur_probability=preset["gaussian_blur"]["probability"],
            sigma_min=preset["gaussian_blur"]["sigma_min"],
            sigma_max=preset["gaussian_blur"]["sigma_max"],
            resolution_degradation_probability=preset["resolution_degradation"]["probability"],
            scale_min=preset["resolution_degradation"]["scale_min"],
            scale_max=preset["resolution_degradation"]["scale_max"],
            poisson_probability=preset["poisson_noise"]["probability"],
            count_scale_min=preset["poisson_noise"]["count_scale_min"],
            count_scale_max=preset["poisson_noise"]["count_scale_max"],
            additive_offset_probability=offset.get("probability", 0.0),
            offset_min=offset.get("min", -0.002),
            offset_max=offset.get("max", 0.002),
            affine_probability=preset["affine"]["probability"],
            max_rotation_degrees=preset["affine"]["max_rotation_degrees"],
            max_translation_voxels=preset["affine"]["max_translation_voxels"],
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(tensor.shape)
        result = tensor.clone()
        if not torch.is_floating_point(result):
            result = result.float()
        volume, squeezed = _as_batched_volume(result)
        device = volume.device
        self.last_trace = []

        if _should_apply(self.flip_probability, device):
            volume = torch.flip(volume, dims=(-3,))
            self.last_trace.append({"name": "left_right_flip"})

        if _should_apply(self.affine_probability, device):
            rotation = (torch.rand(3, device=device, dtype=volume.dtype) * 2.0 - 1.0) * self.max_rotation_degrees
            translation = (torch.rand(3, device=device, dtype=volume.dtype) * 2.0 - 1.0) * self.max_translation_voxels
            volume = _affine_3d(volume, rotation, translation)
            self.last_trace.append(
                {
                    "name": "affine",
                    "rotation_degrees": [round(float(x), 4) for x in rotation.detach().cpu()],
                    "translation_voxels": [round(float(x), 4) for x in translation.detach().cpu()],
                }
            )

        if _should_apply(self.resolution_degradation_probability, device):
            scale = _uniform(self.scale_min, self.scale_max, device)
            original_size = tuple(int(size) for size in volume.shape[-3:])
            degraded_size = tuple(max(1, int(round(size * scale))) for size in original_size)
            if degraded_size != original_size:
                volume = F.interpolate(volume, size=degraded_size, mode="trilinear", align_corners=False)
                volume = F.interpolate(volume, size=original_size, mode="trilinear", align_corners=False)
            self.last_trace.append({"name": "resolution_degradation", "scale": round(scale, 4)})

        if _should_apply(self.gaussian_blur_probability, device):
            sigma = _uniform(self.sigma_min, self.sigma_max, device)
            if sigma > 0:
                volume = _blur_3d(volume, sigma)
            self.last_trace.append({"name": "gaussian_blur", "sigma": round(sigma, 4)})

        if _should_apply(self.intensity_scale_probability, device):
            scale = _uniform(self.intensity_scale_min, self.intensity_scale_max, device)
            volume = volume * scale
            self.last_trace.append({"name": "intensity_scale", "value": round(scale, 4)})

        if _should_apply(self.gamma_probability, device):
            gamma = _uniform(self.gamma_min, self.gamma_max, device)
            volume = volume.clamp_min(0.0).pow(gamma)
            self.last_trace.append({"name": "gamma", "value": round(gamma, 4)})

        if _should_apply(self.additive_offset_probability, device):
            offset = _uniform(self.offset_min, self.offset_max, device)
            volume = volume + offset
            self.last_trace.append({"name": "additive_offset", "value": round(offset, 5)})

        if _should_apply(self.gaussian_noise_probability, device):
            std = _uniform(self.noise_std_min, self.noise_std_max, device)
            volume = volume + torch.randn_like(volume) * std
            self.last_trace.append({"name": "gaussian_noise", "std": round(std, 5)})

        if _should_apply(self.poisson_probability, device):
            count_scale = _uniform(self.count_scale_min, self.count_scale_max, device)
            # Intensities are nonnegative normalized values. Mapping each
            # normalized unit to synthetic counts gives signal-dependent noise
            # without assuming scanner-specific count units.
            volume = torch.poisson(volume.clamp_min(0.0) * count_scale) / count_scale
            self.last_trace.append({"name": "poisson_noise", "count_scale": round(count_scale, 4)})

        volume = volume.clamp_min(0.0)
        if not torch.isfinite(volume).all():
            raise ValueError("Scanner augmentation produced non-finite values")
        if squeezed:
            volume = volume.squeeze(0)
        if tuple(volume.shape) != original_shape:
            raise RuntimeError(f"Scanner augmentation changed shape from {original_shape} to {tuple(volume.shape)}")
        return volume


def resolve_augmentation_config(config: Mapping[str, Any] | str | None, legacy_augment: bool = True) -> dict[str, Any]:
    """Normalize explicit or legacy augmentation settings for metadata."""
    if config is None:
        return {"name": "mild" if legacy_augment else "none"}
    if isinstance(config, str):
        return {"name": config}
    return deepcopy(dict(config))


def build_augmentation(
    config: Mapping[str, Any] | str | None = None,
    *,
    legacy_augment: bool = True,
) -> MildVolumeAugmentation | ScannerRobustVolumeAugmentation | None:
    """Build an augmentation from YAML, with ``training.augment`` fallback."""
    resolved = resolve_augmentation_config(config, legacy_augment=legacy_augment)
    name = str(resolved.get("name", "mild")).lower().replace("-", "_")
    if name in {"none", "off", "disabled"}:
        return None
    if name in {"mild", "baseline"}:
        return MildVolumeAugmentation(
            flip_probability=float(resolved.get("flip_probability", 0.5)),
            noise_std=float(resolved.get("noise_std", 0.015)),
        )
    if name in {"scanner_robust", "scanner"}:
        return ScannerRobustVolumeAugmentation.from_config(resolved)
    raise ValueError(f"Unknown augmentation name: {resolved.get('name')!r}")
