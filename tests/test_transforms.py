import numpy as np
import nibabel as nib
import pandas as pd
import torch

from datscan.data.dataset import DaTSPECTDataset
from datscan.data.transforms import (
    MildVolumeAugmentation,
    ScannerRobustVolumeAugmentation,
    build_augmentation,
)
from datscan.utils.config import PreprocessConfig


def _transform(**overrides):
    values = {
        "flip_probability": 0.0,
        "intensity_scale_probability": 0.0,
        "gamma_probability": 0.0,
        "gaussian_noise_probability": 0.0,
        "gaussian_blur_probability": 0.0,
        "resolution_degradation_probability": 0.0,
        "poisson_probability": 0.0,
        "additive_offset_probability": 0.0,
        "affine_probability": 0.0,
    }
    values.update(overrides)
    return ScannerRobustVolumeAugmentation(**values)


def test_augmentation_factory_preserves_none_and_mild_options():
    assert build_augmentation({"name": "none"}) is None
    assert isinstance(build_augmentation({"name": "mild"}), MildVolumeAugmentation)
    assert isinstance(build_augmentation({"name": "scanner_robust", "severity": "moderate"}), ScannerRobustVolumeAugmentation)


def test_gaussian_blur_smooths_impulse_and_preserves_shape():
    value = torch.zeros(1, 17, 17, 17)
    value[:, 8, 8, 8] = 1.0
    transform = _transform(gaussian_blur_probability=1.0, sigma_min=1.0, sigma_max=1.0)
    output = transform(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert 0.0 < output[:, 8, 8, 8].item() < 1.0
    assert output[:, 8, 8, 7].item() > 0.0


def test_resolution_degradation_reduces_high_frequency_structure():
    coordinates = torch.arange(16).reshape(1, 1, 1, 16)
    value = ((coordinates % 2) * torch.ones(1, 16, 16, 1)).float()
    value = value.expand(1, 16, 16, 16).clone()
    transform = _transform(resolution_degradation_probability=1.0, scale_min=0.5, scale_max=0.5)
    output = transform(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert output.std() < value.std()


def test_gaussian_noise_is_stochastic_but_seed_reproducible():
    value = torch.ones(1, 12, 12, 12)
    transform = _transform(gaussian_noise_probability=1.0, noise_std_min=0.02, noise_std_max=0.02)
    torch.manual_seed(17)
    first = transform(value)
    torch.manual_seed(17)
    second = transform(value)
    assert torch.allclose(first, second)
    assert not torch.allclose(first, value)


def test_poisson_noise_is_finite_and_nonnegative():
    value = torch.full((1, 12, 12, 12), 0.8)
    transform = _transform(poisson_probability=1.0, count_scale_min=100.0, count_scale_max=100.0)
    output = transform(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert output.min() >= 0


def test_affine_perturbation_keeps_center_object_in_volume():
    value = torch.zeros(1, 24, 24, 24)
    value[:, 9:15, 9:15, 9:15] = 1.0
    transform = _transform(affine_probability=1.0, max_rotation_degrees=5.0, max_translation_voxels=2.0)
    output = transform(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert output.sum() > 10.0
    assert output.max() > 0.1


def test_combined_scanner_augmentation_is_functional_and_nonnegative():
    value = torch.rand(1, 20, 20, 20)
    original = value.clone()
    transform = ScannerRobustVolumeAugmentation.from_config({"severity": "moderate"})
    output = transform(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert output.min() >= 0
    assert torch.equal(value, original)


def test_batched_scanner_augmentation_preserves_batch_shape():
    value = torch.rand(2, 1, 12, 12, 12)
    transform = _transform(affine_probability=1.0, max_rotation_degrees=2.0, max_translation_voxels=1.0)
    output = transform(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()


def test_training_dataset_is_stochastic_but_validation_is_deterministic(tmp_path):
    path = tmp_path / "scan.nii.gz"
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[5:11, 5:11, 5:11] = 1.0
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    frame = pd.DataFrame([{"uid": "scan", "filepath": str(path), "label": 1.0}])
    config = PreprocessConfig(output_shape=(16, 16, 16), target_spacing_mm=1.0)
    transform = _transform(gaussian_noise_probability=1.0, noise_std_min=0.03, noise_std_max=0.03)
    training = DaTSPECTDataset(frame, config, augment=transform, cache_dir=tmp_path / "cache")
    validation = DaTSPECTDataset(frame, config, cache_dir=tmp_path / "cache")
    train_first = training[0]["image"]
    train_second = training[0]["image"]
    valid_first = validation[0]["image"]
    valid_second = validation[0]["image"]
    assert not torch.equal(train_first, train_second)
    assert torch.equal(valid_first, valid_second)
    assert np.allclose(valid_first.numpy(), np.load(next((tmp_path / "cache").glob("*.npy"))))
