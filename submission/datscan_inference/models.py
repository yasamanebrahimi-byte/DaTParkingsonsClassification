"""Self-contained 3D ResNet variants for competition inference."""

from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int, requested: int) -> int:
    for value in range(min(channels, requested), 0, -1):
        if channels % value == 0:
            return value
    return 1


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = None if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride, bias=False), nn.GroupNorm(_groups(out_channels, groups), out_channels))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)


class ResNet3D(nn.Module):
    def __init__(self, base_channels: int = 16, groups: int = 8, layers=(2, 2, 2, 2)) -> None:
        super().__init__()
        layers = tuple(int(count) for count in layers)
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(nn.Conv3d(1, channels[0], 7, 2, 3, bias=False), nn.GroupNorm(_groups(channels[0], groups), channels[0]), nn.ReLU(inplace=True), nn.MaxPool3d(3, 2, 1))
        stages = []
        incoming = channels[0]
        for stage, outgoing in enumerate(channels):
            blocks = [Block(incoming, outgoing, 1 if stage == 0 else 2, groups)]
            blocks.extend(Block(outgoing, outgoing, 1, groups) for _ in range(layers[stage] - 1))
            stages.append(nn.Sequential(*blocks))
            incoming = outgoing
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x):
        return self.classifier(self.pool(self.stages(self.stem(x))).flatten(1)).squeeze(1)


class HighResolutionResNet3D(ResNet3D):
    def __init__(self, base_channels: int = 16, groups: int = 8, layers=(2, 2, 2, 2)) -> None:
        nn.Module.__init__(self)
        layers = tuple(int(count) for count in layers)
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(nn.Conv3d(1, channels[0], 3, 1, 1, bias=False), nn.GroupNorm(_groups(channels[0], groups), channels[0]), nn.ReLU(inplace=True))
        stages = []
        incoming = channels[0]
        for stage, outgoing in enumerate(channels):
            blocks = [Block(incoming, outgoing, 1 if stage == 0 else 2, groups)]
            blocks.extend(Block(outgoing, outgoing, 1, groups) for _ in range(layers[stage] - 1))
            stages.append(nn.Sequential(*blocks))
            incoming = outgoing
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(channels[-1], 1)


def build_model(name="resnet3d", base_channels=16, groups=8, layers=(2, 2, 2, 2)):
    normalized = str(name).lower()
    if normalized in {"resnet3d", "resnet18", "resnet18_3d"}:
        return ResNet3D(base_channels, groups, layers)
    if normalized in {"resnet3d_highres", "resnet3d-highres", "highres_resnet3d"}:
        return HighResolutionResNet3D(base_channels, groups, layers)
    raise ValueError(f"Unknown model: {name}")


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = payload.get("model", {}) or {}
    model = build_model(
        model_config.get("name", "resnet3d"),
        int(model_config.get("base_channels", 16)),
        int(model_config.get("groups", 8)),
        model_config.get("layers", (2, 2, 2, 2)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload.get("preprocess", {})
