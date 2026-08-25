"""Write OOF calibration plots, extreme-confidence tables, and clipping results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.diagnostics import write_calibration_diagnostics  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probability-column", default="probability")
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.oof)
    summary = write_calibration_diagnostics(frame, args.output_dir, args.probability_column)
    print(summary["metrics"])
    print(f"Wrote calibration diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

