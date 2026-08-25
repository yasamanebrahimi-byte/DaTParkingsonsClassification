import torch

from datscan.models.resnet3d import ResNet3D


def test_resnet_returns_raw_logit():
    model = ResNet3D(base_channels=4, groups=2)
    output = model(torch.randn(2, 1, 32, 32, 32))
    assert output.shape == (2,)
    assert output.dtype == torch.float32

