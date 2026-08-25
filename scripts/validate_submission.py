"""Validate an output against the competition template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.inference.predict import validate_submission


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args(argv)
    submission = pd.read_csv(args.submission)
    template = pd.read_csv(args.template)
    validate_submission(submission, template)
    print("Submission validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
