import numpy as np
import nibabel as nib
import pandas as pd
import torch

from datscan.data.dataset import DaTSPECTDataset
from datscan.data.preprocessing import preprocess_nifti, preprocess_nifti_views
from datscan.models.resnet3d import build_model
from datscan.utils.config import PreprocessConfig, ROIConfig


def _bilateral_scan(tmp_path, spacing=1.0, weak=False):
    volume = np.zeros((120, 120, 80), dtype=np.float32)
    volume[43:49, 56:64, 34:42] = 1.0
    volume[71:77, 56:64, 34:42] = 0.25 if weak else 1.0
    path = tmp_path / f"bilateral_{spacing}.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.diag([spacing, spacing, spacing, 1.0])), path)
    return path


def test_roi_shape_and_determinism(tmp_path):
    path = _bilateral_scan(tmp_path)
    preprocess = PreprocessConfig(target_spacing_mm=1.0, output_shape=(112, 112, 80))
    roi = ROIConfig(enabled=True, roi_shape=(64, 64, 48))
    first = preprocess_nifti(path, preprocess, data_view="roi", roi_config=roi)
    second = preprocess_nifti(path, preprocess, data_view="roi", roi_config=roi)
    assert first.shape == (1, 64, 64, 48)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


def test_bilateral_and_asymmetric_uptake_remain_in_roi(tmp_path):
    preprocess = PreprocessConfig(target_spacing_mm=1.0, output_shape=(112, 112, 80))
    roi = ROIConfig(enabled=True, roi_shape=(64, 64, 48))
    for weak in (False, True):
        result = preprocess_nifti(_bilateral_scan(tmp_path, weak=weak), preprocess, "roi", roi)[0]
        left_signal = result[10:28].max()
        right_signal = result[36:54].max()
        assert left_signal > 0
        assert right_signal > 0


def test_roi_respects_physical_resampling_and_shared_views(tmp_path):
    volume = np.zeros((12, 12, 12), dtype=np.float32)
    volume[4:8, 4:8, 4:8] = 5.0
    path = tmp_path / "physical.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.diag([2.0, 2.0, 2.0, 1.0])), path)
    preprocess = PreprocessConfig(target_spacing_mm=1.0, output_shape=(32, 32, 32))
    roi = ROIConfig(enabled=True, roi_shape=(16, 16, 16))
    views = preprocess_nifti_views(path, preprocess, roi)
    assert views["global"].shape == (1, 32, 32, 32)
    assert views["roi"].shape == (1, 16, 16, 16)
    assert views["global"].sum() >= views["roi"].sum() > 0
    assert np.count_nonzero(views["global"]) > np.count_nonzero(volume)


def test_roi_dataset_cache_is_isolated_from_global(tmp_path):
    path = _bilateral_scan(tmp_path)
    frame = pd.DataFrame([{"uid": "scan", "filepath": str(path), "label": 1.0}])
    preprocess = PreprocessConfig(target_spacing_mm=1.0, output_shape=(64, 64, 48))
    roi = ROIConfig(enabled=True, roi_shape=(32, 32, 24))
    cache = tmp_path / "cache"
    global_item = DaTSPECTDataset(frame, preprocess, cache_dir=cache, data_view="global")[0]
    roi_item = DaTSPECTDataset(frame, preprocess, cache_dir=cache, data_view="roi", roi_config=roi)[0]
    assert tuple(global_item["image"].shape) == (1, 64, 64, 48)
    assert tuple(roi_item["image"].shape) == (1, 32, 32, 24)
    assert len(list(cache.glob("*.npy"))) == 2


def test_roi_model_accepts_configured_dimensions_and_returns_one_logit():
    model = build_model("roi_resnet3d", base_channels=2, groups=1)
    output = model(torch.randn(2, 1, 32, 32, 24))
    assert output.shape == (2,)
    assert model.feature_map_shape((2, 1, 64, 64, 48)) == (8, 8, 6)
