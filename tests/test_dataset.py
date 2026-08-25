import numpy as np
import nibabel as nib
import pandas as pd

from datscan.data.dataset import DaTSPECTDataset
from datscan.utils.config import PreprocessConfig


def test_dataset_returns_tensor_and_target(tmp_path):
    path = tmp_path / "abc.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), path)
    frame = pd.DataFrame([{"uid": "abc", "filepath": str(path), "label": 1.0}])
    item = DaTSPECTDataset(frame, PreprocessConfig(output_shape=(8, 8, 8), target_spacing_mm=1.0))[0]
    assert tuple(item["image"].shape) == (1, 8, 8, 8)
    assert float(item["target"]) == 1.0

