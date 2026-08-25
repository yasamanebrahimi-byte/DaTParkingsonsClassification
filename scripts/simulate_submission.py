"""Run the actual packaged entry point in a competition-like local runtime."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.inference.predict import validate_submission


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="submission.zip")
    parser.add_argument("--niftis", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="datscan_mock_runtime_") as temp:
        runtime = Path(temp)
        data = runtime / "data"
        source = runtime / "src"
        (data / "niftis").mkdir(parents=True)
        shutil.copytree(args.niftis, data / "niftis", dirs_exist_ok=True)
        shutil.copy2(args.template, data / "submission_format.csv")
        with zipfile.ZipFile(args.package) as archive:
            archive.extractall(source)
        subprocess.run([sys.executable, str(source / "main.py")], cwd=runtime, check=True)
        output = runtime / "submission.csv"
        if not output.exists():
            raise RuntimeError("Packaged inference did not write submission.csv")
        validate_submission(pd.read_csv(output), pd.read_csv(data / "submission_format.csv"))
        print(f"Packaged simulation passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
