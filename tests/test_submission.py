from pathlib import Path
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
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


def test_submission_roi_preprocessing_matches_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    from datscan.data.preprocessing import preprocess_nifti
    from datscan.utils.config import PreprocessConfig, ROIConfig
    from datscan_inference.preprocessing import Config, ROIConfig as SubmissionROIConfig, preprocess

    path = tmp_path / "roi_aligned.nii.gz"
    volume = np.zeros((80, 80, 64), dtype=np.float32)
    volume[28:34, 35:42, 26:34] = 4.0
    volume[46:52, 35:42, 26:34] = 1.0
    nib.save(nib.Nifti1Image(volume, np.diag([1.5, 1.5, 1.5, 1.0])), path)
    preprocess_config = PreprocessConfig(output_shape=(64, 64, 64), target_spacing_mm=2.5)
    roi_config = ROIConfig(enabled=True, roi_shape=(32, 32, 24))
    training = preprocess_nifti(path, preprocess_config, data_view="roi", roi_config=roi_config)
    packaged = preprocess(path, Config(output_shape=(64, 64, 64), target_spacing_mm=2.5), data_view="roi", roi_config=SubmissionROIConfig(enabled=True, roi_shape=(32, 32, 24)))
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


def test_submission_reconstructs_roi_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    from datscan.models.resnet3d import build_model as build_training_model
    from datscan_inference.models import load_model

    model = build_training_model("roi_resnet3d", base_channels=2, groups=1)
    path = tmp_path / "roi.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": {"name": "roi_resnet3d", "base_channels": 2, "groups": 1, "layers": [2, 2, 2, 2]},
            "preprocess": {"target_spacing_mm": 2.5, "output_shape": [112, 112, 112]},
            "data_view": "roi",
            "roi": {"enabled": True, "roi_shape": [64, 64, 48], "center_max_shift_fraction": 0.25},
        },
        path,
    )
    loaded, raw_preprocess = load_model(path, torch.device("cpu"))
    assert loaded(torch.zeros(1, 1, 64, 64, 48)).shape == (1,)
    assert raw_preprocess["target_spacing_mm"] == 2.5


def test_submission_inference_runs_global_and_roi_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    from datscan.models.resnet3d import build_model
    from datscan_inference.inference import run_inference

    runtime = tmp_path / "runtime"
    shutil.copytree(root / "submission" / "datscan_inference", runtime / "datscan_inference")
    (runtime / "assets").mkdir(parents=True)
    (runtime / "data" / "niftis").mkdir(parents=True)
    (runtime / "data" / "submission_format.csv").write_text("uid,is_pathologic\ncase,0\n", encoding="utf-8")
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[7:13, 6:10, 8:12] = 3.0
    nib.save(nib.Nifti1Image(volume, np.eye(4)), runtime / "data" / "niftis" / "case.nii.gz")
    global_model = build_model("resnet3d", base_channels=2, groups=1)
    roi_model = build_model("roi_resnet3d", base_channels=2, groups=1)
    common_preprocess = {"target_spacing_mm": 1.0, "output_shape": [16, 16, 16]}
    torch.save({"state_dict": global_model.state_dict(), "model": {"name": "resnet3d", "base_channels": 2, "groups": 1, "layers": [2, 2, 2, 2]}, "preprocess": common_preprocess, "data_view": "global", "roi": None}, runtime / "assets" / "global_model_fold0.pt")
    torch.save({"state_dict": roi_model.state_dict(), "model": {"name": "roi_resnet3d", "base_channels": 2, "groups": 1, "layers": [2, 2, 2, 2]}, "preprocess": common_preprocess, "data_view": "roi", "roi": {"enabled": True, "roi_shape": [8, 8, 8], "center_max_shift_fraction": 0.25}}, runtime / "assets" / "roi_model_fold0.pt")
    (runtime / "assets" / "ensemble.json").write_text('{"version": 2, "weights": [0.5, 0.5], "calibration_stage": "after_ensemble"}', encoding="utf-8")
    (runtime / "assets" / "calibration.json").write_text('{"temperature": 1.0, "enabled": false}', encoding="utf-8")
    run_inference(runtime)
    output = pd.read_csv(runtime / "submission.csv")
    assert list(output.columns) == ["uid", "is_pathologic"]
    assert len(output) == 1 and 0.0 <= float(output.iloc[0]["is_pathologic"]) <= 1.0
