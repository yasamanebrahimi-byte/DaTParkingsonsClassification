"""ROI model hook; it intentionally shares the stable global architecture."""

from .resnet3d import ResNet3D


class ROIResNet3D(ResNet3D):
    """A separate model type for future ROI experiments and checkpoint clarity."""

