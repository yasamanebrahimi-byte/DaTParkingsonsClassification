import torch

from datscan.models.resnet3d import HighResolutionResNet3D, ResNet3D, build_model
from datscan.utils.config import ModelConfig, PreprocessConfig
from datscan.inference.loader import load_checkpoint


def test_resnet_returns_raw_logit():
    model = ResNet3D(base_channels=4, groups=2)
    output = model(torch.randn(2, 1, 32, 32, 32))
    assert output.shape == (2,)
    assert output.dtype == torch.float32


def test_baseline_model_is_selected_by_name():
    model = build_model("resnet3d", base_channels=2, groups=1)
    assert isinstance(model, ResNet3D)
    assert not isinstance(model, HighResolutionResNet3D)
    assert model.feature_map_shape((1, 1, 96, 96, 96)) == (3, 3, 3)


def test_highres_model_preserves_spatial_resolution_and_returns_batch_logits():
    model = build_model("resnet3d_highres", base_channels=2, groups=1)
    features = model.forward_features(torch.randn(1, 1, 112, 112, 112))
    output = model(torch.randn(1, 1, 112, 112, 112))
    assert features.shape[-3:] == (14, 14, 14)
    assert model.feature_map_shape((1, 1, 112, 112, 112)) == (14, 14, 14)
    assert output.shape == (1,)
    assert model.feature_map_shape((1, 1, 112, 112, 112))[0] > model.feature_map_shape((1, 1, 96, 96, 96))[0]


def test_checkpoint_metadata_reconstructs_highres_model(tmp_path):
    model_config = ModelConfig(name="resnet3d_highres", base_channels=2, groups=1)
    preprocess_config = PreprocessConfig(output_shape=(112, 112, 112), target_spacing_mm=2.5)
    checkpoint = tmp_path / "highres.pt"
    torch.save(
        {
            "checkpoint_version": 2,
            "state_dict": build_model(model_config.name, model_config.base_channels, model_config.groups, model_config.layers).state_dict(),
            "model": {"name": model_config.name, "base_channels": model_config.base_channels, "groups": model_config.groups, "layers": list(model_config.layers)},
            "preprocess": {"target_spacing_mm": 2.5, "output_shape": [112, 112, 112]},
        },
        checkpoint,
    )
    loaded, loaded_preprocess, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert isinstance(loaded, HighResolutionResNet3D)
    assert tuple(loaded_preprocess.output_shape) == tuple(preprocess_config.output_shape)
    assert payload["model"]["name"] == "resnet3d_highres"
