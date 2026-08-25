"""Build and verify a root-correct competition ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--calibration", default="artifacts/calibration/temperature.json")
    parser.add_argument("--output", default="submission.zip")
    parser.add_argument("--allow-empty", action="store_true", help="Build source-only package for scaffold testing")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = root / checkpoint_dir
    calibration_path = Path(args.calibration)
    if not calibration_path.is_absolute():
        calibration_path = root / calibration_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    checkpoints = sorted(checkpoint_dir.glob("resnet3d_fold*.pt"))
    if not checkpoints and not args.allow_empty:
        raise SystemExit("No trained checkpoints found; train the baseline before packaging")
    source = root / "submission"
    with tempfile.TemporaryDirectory(prefix="datscan_submission_") as temp:
        staging = Path(temp) / "package"
        shutil.copytree(source, staging)
        assets = staging / "assets"
        assets.mkdir(exist_ok=True)
        for checkpoint in checkpoints:
            shutil.copy2(checkpoint, assets / checkpoint.name.replace("resnet3d_", "model_"))
        if calibration_path.exists():
            shutil.copy2(calibration_path, assets / "calibration.json")
        else:
            (assets / "calibration.json").write_text(json.dumps({"temperature": 1.0}), encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(staging.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(staging).as_posix())
        with zipfile.ZipFile(output_path) as archive:
            names = set(archive.namelist())
            if "main.py" not in names or not any(name.startswith("datscan_inference/") for name in names):
                raise RuntimeError("submission.zip does not have the required root layout")
    print(f"Wrote {output_path} with {len(checkpoints)} model checkpoint(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
