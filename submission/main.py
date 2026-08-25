"""Offline competition entry point.

The evaluator runs this file from ``/code_execution/src/main.py`` and expects
``/code_execution/submission.csv``.  Paths are derived from this file so the
current working directory is irrelevant.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datscan_inference.inference import run_inference


if __name__ == "__main__":
    run_inference(ROOT)

