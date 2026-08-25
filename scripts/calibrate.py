"""Fit temperature scaling using OOF logits only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.calibrate import fit_temperature, save_calibration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.oof)
    temperature = fit_temperature(frame["logit"].to_numpy(), frame["target"].to_numpy())
    save_calibration(temperature, args.output)
    print(f"Saved temperature={temperature:.6f} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
