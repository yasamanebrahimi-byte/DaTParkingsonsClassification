import numpy as np
import nibabel as nib

from datscan.data.preprocessing import crop_or_pad_center, normalize_intensity, preprocess_nifti
from datscan.utils.config import PreprocessConfig


def test_normalization_is_per_scan_and_finite():
    config = PreprocessConfig()
    result = normalize_intensity(np.array([0, 1, 2, 100], dtype=np.float32), config)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.max() <= config.clip_max


def test_crop_or_pad_keeps_center():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[2, 2, 2] = 1
    result = crop_or_pad_center(volume, (7, 7, 7), (2, 2, 2))
    assert result.shape == (7, 7, 7)
    assert result[3, 3, 3] == 1


def test_full_preprocessing_has_fixed_shape(tmp_path):
    data = np.zeros((16, 14, 12), dtype=np.uint16)
    data[5:10, 4:9, 3:8] = 100
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0])), path)
    config = PreprocessConfig(output_shape=(16, 16, 16), target_spacing_mm=2.0)
    result = preprocess_nifti(path, config)
    assert result.shape == (1, 16, 16, 16)
    assert result.dtype == np.float32

