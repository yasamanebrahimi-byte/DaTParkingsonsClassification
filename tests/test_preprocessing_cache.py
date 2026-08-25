import numpy as np
import nibabel as nib
import pandas as pd

import datscan.data.dataset as dataset_module
from datscan.data.dataset import DaTSPECTDataset
from datscan.data.preprocessing import preprocess_nifti
from datscan.utils.config import PreprocessConfig


def _dataset_frame(path):
    return pd.DataFrame([{"uid": "scan-1", "filepath": str(path), "label": 1.0}])


def _make_scan(tmp_path):
    path = tmp_path / "scan.nii.gz"
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[3:7, 3:7, 3:7] = 4.0
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    return path


def test_cache_populates_and_reuses_uncached_result(tmp_path, monkeypatch):
    path = _make_scan(tmp_path)
    config = PreprocessConfig(output_shape=(8, 8, 8), target_spacing_mm=1.0)
    uncached = preprocess_nifti(path, config)
    calls = 0
    original = dataset_module.preprocess_nifti

    def counted_preprocess(source, preprocess_config):
        nonlocal calls
        calls += 1
        return original(source, preprocess_config)

    monkeypatch.setattr(dataset_module, "preprocess_nifti", counted_preprocess)
    dataset = DaTSPECTDataset(_dataset_frame(path), config, cache_dir=tmp_path / "cache")
    first = dataset[0]["image"].numpy()
    second = dataset[0]["image"].numpy()

    cache_files = list((tmp_path / "cache").glob("*.npy"))
    assert calls == 1
    assert len(cache_files) == 1
    assert np.allclose(first, uncached)
    assert np.allclose(second, uncached)
    assert np.allclose(np.load(cache_files[0]), uncached)


def test_config_change_uses_a_different_cache_entry(tmp_path, monkeypatch):
    path = _make_scan(tmp_path)
    frame = _dataset_frame(path)
    calls = 0
    original = dataset_module.preprocess_nifti

    def counted_preprocess(source, preprocess_config):
        nonlocal calls
        calls += 1
        return original(source, preprocess_config)

    monkeypatch.setattr(dataset_module, "preprocess_nifti", counted_preprocess)
    first_config = PreprocessConfig(output_shape=(8, 8, 8), target_spacing_mm=1.0)
    second_config = PreprocessConfig(output_shape=(8, 8, 8), target_spacing_mm=2.0)
    DaTSPECTDataset(frame, first_config, cache_dir=tmp_path / "cache")[0]
    DaTSPECTDataset(frame, second_config, cache_dir=tmp_path / "cache")[0]

    assert calls == 2
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2


def test_corrupted_cache_entry_is_regenerated(tmp_path, monkeypatch):
    path = _make_scan(tmp_path)
    config = PreprocessConfig(output_shape=(8, 8, 8), target_spacing_mm=1.0)
    calls = 0
    original = dataset_module.preprocess_nifti

    def counted_preprocess(source, preprocess_config):
        nonlocal calls
        calls += 1
        return original(source, preprocess_config)

    monkeypatch.setattr(dataset_module, "preprocess_nifti", counted_preprocess)
    dataset = DaTSPECTDataset(_dataset_frame(path), config, cache_dir=tmp_path / "cache")
    expected = dataset[0]["image"].numpy()
    cache_file = next((tmp_path / "cache").glob("*.npy"))
    cache_file.write_bytes(b"not a numpy file")

    regenerated = dataset[0]["image"].numpy()
    assert calls == 2
    assert np.allclose(regenerated, expected)
    assert np.allclose(np.load(cache_file), expected)


def test_augmentation_runs_after_cached_volume_is_loaded(tmp_path, monkeypatch):
    path = _make_scan(tmp_path)
    config = PreprocessConfig(output_shape=(8, 8, 8), target_spacing_mm=1.0)
    original = dataset_module.preprocess_nifti
    base = original(path, config)
    dataset = DaTSPECTDataset(_dataset_frame(path), config, cache_dir=tmp_path / "cache")
    dataset[0]

    def fail_if_preprocessed_again(source, preprocess_config):
        raise AssertionError("valid cached volume should be used")

    monkeypatch.setattr(dataset_module, "preprocess_nifti", fail_if_preprocessed_again)
    augmented_dataset = DaTSPECTDataset(
        _dataset_frame(path),
        config,
        augment=lambda tensor: tensor + 1.0,
        cache_dir=tmp_path / "cache",
    )
    augmented = augmented_dataset[0]["image"].numpy()
    assert np.allclose(augmented, base + 1.0)
    assert np.allclose(np.load(next((tmp_path / "cache").glob("*.npy"))), base)
