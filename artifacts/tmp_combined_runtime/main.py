"""Offline competition entry point.

The submission is unpacked directly into ``/code_execution/`` and the evaluator
runs ``/code_execution/main.py``. It expects
``/code_execution/submission.csv``.  Paths are derived from this file so the
current working directory is irrelevant.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datscan_inference.inference import run_inference


if __name__ == "__main__":
    run_inference(ROOT)
