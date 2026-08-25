"""Compact 3D ResNet variants with GroupNorm for small medical-imaging batches."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _group_count(channels: int, requested: int) -> int:
    for count in range(min(requested, channels), 0, -1):
        if channels % count == 0:
            return count
    return 1


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + identity)


class ResNet3D(nn.Module):
    """Baseline ResNet with the original approximately 32x spatial reduction."""

    spatial_strides = (2, 2, 1, 2, 2, 2)

    def __init__(self, layers: Sequence[int] = (2, 2, 2, 2), base_channels: int = 16, groups: int = 8) -> None:
        super().__init__()
        layers = tuple(int(count) for count in layers)
        if len(layers) != 4 or any(count < 1 for count in layers):
            raise ValueError("layers must contain four positive block counts")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv3d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(channels[0], groups), channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        in_channels = channels[0]
        for stage, (out_channels, block_count) in enumerate(zip(channels, layers)):
            stride = 1 if stage == 0 else 2
            stage_blocks = [BasicBlock3D(in_channels, out_channels, stride=stride, groups=groups)]
            stage_blocks.extend(BasicBlock3D(out_channels, out_channels, groups=groups) for _ in range(block_count - 1))
            blocks.append(nn.Sequential(*stage_blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(channels[-1], 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.stages(x)

    def feature_map_shape(self, input_shape: Sequence[int] = (1, 1, 96, 96, 96)) -> tuple[int, int, int]:
        """Return the analytical spatial shape immediately before global pooling."""
        if len(input_shape) != 5:
            raise ValueError("input_shape must be [batch, channels, depth, height, width]")
        shape = tuple(int(size) for size in input_shape[-3:])
        for stride in self.spatial_strides:
            shape = tuple((size + stride - 1) // stride for size in shape)
        return shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.forward_features(x)).flatten(1)
        return self.classifier(x).squeeze(1)


class HighResolutionResNet3D(ResNet3D):
    """ResNet variant that keeps an approximately 14^3 map for a 112^3 input."""

    spatial_strides = (1, 1, 1, 2, 2, 2)

    def __init__(self, layers: Sequence[int] = (2, 2, 2, 2), base_channels: int = 16, groups: int = 8) -> None:
        nn.Module.__init__(self)
        layers = tuple(int(count) for count in layers)
        if len(layers) != 4 or any(count < 1 for count in layers):
            raise ValueError("layers must contain four positive block counts")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv3d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0], groups), channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_channels = channels[0]
        for stage, (out_channels, block_count) in enumerate(zip(channels, layers)):
            stride = 1 if stage == 0 else 2
            stage_blocks = [BasicBlock3D(in_channels, out_channels, stride=stride, groups=groups)]
            stage_blocks.extend(BasicBlock3D(out_channels, out_channels, groups=groups) for _ in range(block_count - 1))
            blocks.append(nn.Sequential(*stage_blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(channels[-1], 1)
        self._initialize()


# Readable alias for callers that prefer the shorter class name.
ResNet3DHighRes = HighResolutionResNet3D


def build_model(
    name: str = "resnet3d",
    base_channels: int = 16,
    groups: int = 8,
    layers: Sequence[int] = (2, 2, 2, 2),
) -> nn.Module:
    if name.lower() in {"resnet3d", "resnet18", "resnet18_3d"}:
        return ResNet3D(layers=layers, base_channels=base_channels, groups=groups)
    if name.lower() in {"resnet3d_highres", "resnet3d-highres", "highres_resnet3d"}:
        return HighResolutionResNet3D(layers=layers, base_channels=base_channels, groups=groups)
    raise ValueError(f"Unknown model: {name}")
