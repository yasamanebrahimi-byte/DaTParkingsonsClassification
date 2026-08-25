"""Torch-only training augmentations applied after deterministic preprocessing."""

from __future__ import annotations

import torch


class MildVolumeAugmentation:
    """Conservative intensity and geometric augmentations for SPECT volumes."""

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
        if torch.rand(()) < 0.35:
            gamma = 0.95 + torch.rand((), device=result.device) * 0.10
            result = torch.clamp(result, min=0).pow(gamma)
        if torch.rand(()) < 0.25:
            result = result + torch.randn_like(result) * self.noise_std
        return torch.clamp(result, min=0.0)
