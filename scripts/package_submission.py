"""Build and verify a root-correct CNN or CNN + quantitative-feature ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def _absolute(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints", help="Legacy global-only checkpoint directory")
    parser.add_argument("--global-checkpoint-dir")
    parser.add_argument("--roi-checkpoint-dir")
    parser.add_argument("--ensemble", help="OOF-derived global/ROI ensemble JSON")
    parser.add_argument("--feature-model-dir", help="Directory containing model_fold*.pkl and feature_columns.json")
    parser.add_argument("--feature-config", help="Quantitative feature YAML configuration")
    parser.add_argument("--calibration", default="artifacts/calibration/temperature.json")
    parser.add_argument("--global-config")
    parser.add_argument("--roi-config")
    parser.add_argument("--output", default="submission.zip")
    parser.add_argument("--allow-empty", action="store_true", help="Build source-only package for scaffold testing")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output_path = _absolute(root, args.output)
    calibration_path = _absolute(root, args.calibration)
    global_dir = _absolute(root, args.global_checkpoint_dir or args.checkpoint_dir)
    roi_dir = _absolute(root, args.roi_checkpoint_dir) if args.roi_checkpoint_dir else None
    feature_dir = _absolute(root, args.feature_model_dir) if args.feature_model_dir else None
    global_checkpoints = sorted(global_dir.glob("resnet3d_fold*.pt"))
    if not global_checkpoints:
        global_checkpoints = sorted(global_dir.glob("global_model_fold*.pt"))
    roi_checkpoints = sorted(roi_dir.glob("roi_resnet3d_fold*.pt")) if roi_dir else []
    if roi_dir and not roi_checkpoints:
        roi_checkpoints = sorted(roi_dir.glob("roi_model_fold*.pt"))
    feature_models = sorted(feature_dir.glob("model_fold*.pkl")) if feature_dir else []
    new_package = bool(args.global_checkpoint_dir or args.roi_checkpoint_dir or args.ensemble or feature_dir)
    if roi_checkpoints and not args.ensemble:
        raise SystemExit("ROI checkpoints require --ensemble with OOF-derived weights")
    if not global_checkpoints and not args.allow_empty:
        raise SystemExit("No trained global checkpoints found; train the model before packaging")
    if roi_checkpoints and len(roi_checkpoints) != len(global_checkpoints):
        raise SystemExit("Global and ROI checkpoint counts do not match")
    if feature_dir and not feature_models:
        raise SystemExit("--feature-model-dir must contain model_fold*.pkl files")
    if feature_models and not args.feature_config:
        raise SystemExit("Quantitative feature models require --feature-config")
    if feature_models and not (feature_dir / "feature_columns.json").exists():
        raise SystemExit("Quantitative feature model directory is missing feature_columns.json")

    source = root / "submission"
    with tempfile.TemporaryDirectory(prefix="datscan_submission_") as temp:
        staging = Path(temp) / "package"
        shutil.copytree(source, staging)
        assets = staging / "assets"
        assets.mkdir(exist_ok=True)
        if feature_models:
            package_module = staging / "datscan_inference" / "striatal_features.py"
            package_module.write_bytes((root / "src" / "datscan" / "features" / "striatal_features.py").read_bytes())
            for model_path in feature_models:
                shutil.copy2(model_path, assets / f"feature_{model_path.name}")
            shutil.copy2(feature_dir / "feature_columns.json", assets / "feature_columns.json")
            shutil.copy2(_absolute(root, args.feature_config), assets / "striatal_features.yaml")
            shutil.copy2(root / "submission" / "datscan_inference" / "feature_support.py", staging / "datscan_inference" / "feature_support.py")
        if new_package:
            for checkpoint in global_checkpoints:
                shutil.copy2(checkpoint, assets / checkpoint.name.replace("resnet3d_", "global_model_"))
            for checkpoint in roi_checkpoints:
                shutil.copy2(checkpoint, assets / checkpoint.name.replace("roi_resnet3d_", "roi_model_"))
        else:
            for checkpoint in global_checkpoints:
                shutil.copy2(checkpoint, assets / checkpoint.name.replace("resnet3d_", "model_"))
        if args.ensemble:
            shutil.copy2(_absolute(root, args.ensemble), assets / "ensemble.json")
        if calibration_path.exists():
            shutil.copy2(calibration_path, assets / "calibration.json")
        else:
            (assets / "calibration.json").write_text(json.dumps({"temperature": 1.0, "enabled": False}, indent=2), encoding="utf-8")
        for source_config, asset_name in ((args.global_config, "global_config.yaml"), (args.roi_config, "roi_config.yaml")):
            if source_config:
                shutil.copy2(_absolute(root, source_config), assets / asset_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(staging.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(staging).as_posix())
        with zipfile.ZipFile(output_path) as archive:
            names = set(archive.namelist())
            if "main.py" not in names or not any(name.startswith("datscan_inference/") for name in names):
                raise RuntimeError("submission.zip does not have the required root layout")
            if roi_checkpoints and "ensemble.json" not in names:
                raise RuntimeError("ROI package is missing ensemble.json")
    print(f"Wrote {output_path} with {len(global_checkpoints)} global and {len(roi_checkpoints)} ROI checkpoint(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
