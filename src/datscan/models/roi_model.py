"""Compact high-resolution classifier for the bilateral striatal crop."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from .resnet3d import BasicBlock3D, _group_count


class ROIResNet3D(nn.Module):
    """A true ROI-only ResNet with no early pooling.

    The canonical RAS tensor uses axis 0 (the first spatial axis) for
    left/right.  The model receives both hemispheres in the same tensor and
    preserves that axis until the progressively downsampled feature maps.
    For a 64 x 64 x 48 input, the final map is 8 x 8 x 6 (8x downsampling).
    """

    spatial_strides = (1, 1, 1, 2, 2, 2)

    def __init__(self, layers: Sequence[int] = (2, 2, 2, 2), base_channels: int = 16, groups: int = 8) -> None:
        super().__init__()
        layers = tuple(int(count) for count in layers)
        if len(layers) != 4 or any(count < 1 for count in layers):
            raise ValueError("layers must contain four positive block counts")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv3d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0], groups), channels[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        incoming = channels[0]
        for stage, (outgoing, block_count) in enumerate(zip(channels, layers)):
            stride = 1 if stage == 0 else 2
            blocks = [BasicBlock3D(incoming, outgoing, stride=stride, groups=groups)]
            blocks.extend(BasicBlock3D(outgoing, outgoing, groups=groups) for _ in range(block_count - 1))
            stages.append(nn.Sequential(*blocks))
            incoming = outgoing
        self.stages = nn.Sequential(*stages)
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

    def forward_features(self, x):
        return self.stages(self.stem(x))

    def feature_map_shape(self, input_shape=(1, 1, 64, 64, 48)) -> tuple[int, int, int]:
        if len(input_shape) != 5:
            raise ValueError("input_shape must be [batch, channels, depth, height, width]")
        shape = tuple(int(size) for size in input_shape[-3:])
        for stride in self.spatial_strides:
            shape = tuple((size + stride - 1) // stride for size in shape)
        return shape

    def forward(self, x):
        return self.classifier(self.pool(self.forward_features(x)).flatten(1)).squeeze(1)
