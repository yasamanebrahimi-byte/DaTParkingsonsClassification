import numpy as np
import nibabel as nib
import pandas as pd
import torch

from datscan.training.train import train_one_fold
from datscan.utils.config import ModelConfig, PreprocessConfig


def test_one_fold_training_smoke(tmp_path):
    rows = []
    for index in range(6):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[7:13, 7:13, 7:13] = 1.0 + index
        path = tmp_path / f"scan_{index}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        rows.append({"uid": f"scan_{index}", "filepath": str(path), "label": float(index % 2), "fold": 0 if index < 2 else 1})
    predictions, metrics = train_one_fold(
        pd.DataFrame(rows),
        fold=0,
        preprocess_config=PreprocessConfig(output_shape=(16, 16, 16), target_spacing_mm=1.0),
        model_config=ModelConfig(base_channels=2, groups=1),
        training_config={"epochs": 1, "batch_size": 2, "num_workers": 0, "device": "cpu", "amp": False, "augment": False, "patience": 1},
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert len(predictions) == 2
    assert np.isfinite(predictions["logit"]).all()
    assert "log_loss" in metrics
    payload = torch.load(tmp_path / "checkpoints" / "resnet3d_fold0.pt", map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == 2
    assert payload["model"]["name"] == "resnet3d"
    assert tuple(payload["preprocess"]["output_shape"]) == (16, 16, 16)
