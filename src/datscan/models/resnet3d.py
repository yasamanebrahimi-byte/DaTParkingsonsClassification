"""Compact 3D ResNet with GroupNorm for small medical-imaging batches."""

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
    def __init__(self, layers: Sequence[int] = (2, 2, 2, 2), base_channels: int = 16, groups: int = 8) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)


def build_model(name: str = "resnet3d", base_channels: int = 16, groups: int = 8) -> nn.Module:
    if name.lower() in {"resnet3d", "resnet18", "resnet18_3d"}:
        return ResNet3D(base_channels=base_channels, groups=groups)
    raise ValueError(f"Unknown model: {name}")

