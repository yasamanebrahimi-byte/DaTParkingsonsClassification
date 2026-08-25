from pathlib import Path

import numpy as np
import nibabel as nib
import sys
import torch


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


def test_submission_reconstructs_baseline_and_highres_checkpoints(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    from datscan.models.resnet3d import build_model as build_training_model
    from datscan_inference.models import load_model

    for name in ("resnet3d", "resnet3d_highres"):
        model = build_training_model(name, base_channels=2, groups=1)
        path = tmp_path / f"{name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model": {"name": name, "base_channels": 2, "groups": 1, "layers": [2, 2, 2, 2]},
                "preprocess": {"target_spacing_mm": 2.5 if "highres" in name else 3.0, "output_shape": [112, 112, 112] if "highres" in name else [96, 96, 96]},
            },
            path,
        )
        loaded, raw_preprocess = load_model(path, torch.device("cpu"))
        assert loaded(torch.zeros(1, 1, 112 if "highres" in name else 96, 112 if "highres" in name else 96, 112 if "highres" in name else 96)).shape == (1,)
        assert raw_preprocess["output_shape"] in ([112, 112, 112], [96, 96, 96])
