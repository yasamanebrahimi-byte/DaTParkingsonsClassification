import json
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import nn

from datscan.models.ensemble import aggregate_member_logits
from datscan.training.folds import create_repeated_folds, load_repeated_folds, save_repeated_folds
from datscan.training.repeated import aggregate_repeated_oof, validate_repeated_oof


def _oof():
    rows = []
    logits = {0: [-2.0, 2.0, 0.5, 1.0], 1: [-1.0, 1.0, -0.5, 1.5]}
    for repeat in range(2):
        for index, (uid, target) in enumerate(zip(["a", "b", "c", "d"], [0, 1, 0, 1])):
            value = logits[repeat][index]
            rows.append({
                "uid": uid,
                "target": target,
                "repeat": repeat,
                "fold": index % 2,
                "fold_seed": 100 + repeat,
                "training_seed": 200 + repeat,
                "experiment_name": "test",
                "logit": value,
                "probability": float(1.0 / (1.0 + np.exp(-value))),
            })
    return pd.DataFrame(rows)


def test_repeated_fold_seed_metadata_round_trip(tmp_path):
    metadata = pd.DataFrame({"uid": [f"u{i}" for i in range(12)], "label": [i % 2 for i in range(12)]})
    repeats = create_repeated_folds(metadata, n_splits=3, n_repeats=2, seed=123)
    assert repeats[0].attrs["seed"] == 123
    assert repeats[1].attrs["seed"] == 124
    save_repeated_folds(repeats, tmp_path / "folds", base_seed=123)
    loaded = load_repeated_folds(tmp_path / "folds")
    assert loaded[0].attrs["seed"] == 123
    assert loaded[1].attrs["seed"] == 124
    for frame in loaded.values():
        assert frame["uid"].nunique() == len(metadata)
        assert frame["fold"].value_counts().sum() == len(metadata)


def test_repeated_oof_aggregation_includes_all_raw_representations():
    source = _oof()
    validate_repeated_oof(source, n_repeats=2)
    summary = aggregate_repeated_oof(source, n_repeats=2)
    first = source[source["uid"] == "a"]
    assert np.isclose(summary.loc[summary["uid"] == "a", "mean_probability"].iloc[0], first["probability"].mean())
    assert np.isclose(summary.loc[summary["uid"] == "a", "median_probability"].iloc[0], first["probability"].median())
    assert (summary["n_predictions"] == 2).all()


def test_logit_probability_and_median_aggregation_are_distinct_and_bounded():
    logits = np.asarray([[-4.0, 0.0], [0.0, 4.0], [2.0, 2.0]])
    probability = aggregate_member_logits(logits, "probability_mean")
    logit = aggregate_member_logits(logits, "logit_mean")
    median = aggregate_member_logits(logits, "median_probability")
    assert not np.allclose(probability, logit)
    assert np.all((probability >= 0) & (probability <= 1))
    assert np.all((logit >= 0) & (logit <= 1))
    assert np.all((median >= 0) & (median <= 1))


def test_manifest_payload_has_reloadable_model_metadata(tmp_path):
    from scripts.build_model_manifest import build_manifest

    checkpoint_root = tmp_path / "checkpoints" / "repeat_0" / "seed_20"
    checkpoint_root.mkdir(parents=True)
    path = checkpoint_root / "resnet3d_fold0.pt"
    torch.save({
        "state_dict": {},
        "model": {"name": "resnet3d", "base_channels": 2, "groups": 1, "layers": [1, 1, 1, 1]},
        "preprocess": {"output_shape": [8, 8, 8]},
        "fold": 0,
        "repeat": 0,
        "fold_seed": 10,
        "training_seed": 20,
        "experiment_name": "test",
    }, path)
    output = tmp_path / "manifest.json"
    payload = build_manifest(checkpoint_root.parent.parent.parent, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert payload == loaded
    assert loaded["models"][0]["fold"] == 0
    assert loaded["models"][0]["training_seed"] == 20


def test_ensemble_inference_reuses_one_preprocess_per_scan(monkeypatch):
    import datscan.inference.predict as predict_module

    calls = []

    def fake_preprocess(path, config, data_view, roi_config):
        calls.append(path)
        return np.ones((1, 2, 2, 2), dtype=np.float32)

    class Constant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)

        def forward(self, tensor):
            return torch.full((tensor.shape[0],), self.value)

    monkeypatch.setattr(predict_module, "preprocess_nifti", fake_preprocess)
    result = predict_module.predict_ensemble_paths(
        [Constant(-1.0), Constant(1.0)],
        ["a.nii.gz", "b.nii.gz"],
        object(),
        torch.device("cpu"),
        aggregation="probability_mean",
    )
    assert calls == ["a.nii.gz", "b.nii.gz"]
    assert np.allclose(result, [0.5, 0.5])


def test_manifest_packaging_copies_only_selected_checkpoints(tmp_path):
    from datscan.models.resnet3d import build_model
    from scripts.build_model_manifest import build_manifest
    from scripts.package_submission import main as package_main

    checkpoint_root = tmp_path / "models"
    checkpoint_root.mkdir()
    model = build_model("resnet3d", base_channels=2, groups=1, layers=(1, 1, 1, 1))
    checkpoint = checkpoint_root / "resnet3d_fold0.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": {"name": "resnet3d", "base_channels": 2, "groups": 1, "layers": [1, 1, 1, 1]},
        "preprocess": {"target_spacing_mm": 1.0, "output_shape": [8, 8, 8]},
        "data_view": "global",
        "fold": 0,
        "repeat": 0,
        "training_seed": 20,
    }, checkpoint)
    manifest = tmp_path / "manifest.json"
    build_manifest(checkpoint_root, manifest)
    output = tmp_path / "submission.zip"
    package_main(["--manifest", str(manifest), "--output", str(output)])
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "assets/ensemble_manifest.json" in names
    assert "assets/global_model_member_000.pt" in names

    runtime = tmp_path / "runtime"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(runtime)
    (runtime / "data" / "niftis").mkdir(parents=True)
    (runtime / "data" / "submission_format.csv").write_text("uid,is_pathologic\ncase,0\n", encoding="utf-8")
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), runtime / "data" / "niftis" / "case.nii.gz")
    import sys
    sys.path.insert(0, str(runtime))
    from datscan_inference.inference import run_inference
    run_inference(runtime)
    prediction = pd.read_csv(runtime / "submission.csv")
    assert len(prediction) == 1
    assert 0.0 <= float(prediction.iloc[0]["is_pathologic"]) <= 1.0
