from pathlib import Path

import numpy as np
import nibabel as nib
import sys


def test_submission_source_layout():
    root = Path(__file__).resolve().parents[1]
    assert (root / "submission" / "main.py").exists()
    assert (root / "submission" / "datscan_inference" / "preprocessing.py").exists()


def test_submission_preprocessing_matches_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    from datscan.data.preprocessing import preprocess_nifti
    from datscan.utils.config import PreprocessConfig
    from datscan_inference.preprocessing import Config, preprocess

    path = tmp_path / "aligned.nii.gz"
    volume = np.zeros((20, 18, 16), dtype=np.float32)
    volume[6:14, 5:13, 4:12] = 3.0
    nib.save(nib.Nifti1Image(volume, np.diag([2.0, 2.0, 2.5, 1.0])), path)
    training = preprocess_nifti(path, PreprocessConfig(output_shape=(16, 16, 16), target_spacing_mm=2.5))
    packaged = preprocess(path, Config(output_shape=(16, 16, 16), target_spacing_mm=2.5))
    assert np.allclose(training, packaged, atol=1e-6)
