"""Fit temperature scaling using OOF logits only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.special import logit
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.calibrate import fit_temperature, save_calibration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probability-column", help="Use this probability column for after-ensemble calibration")
    parser.add_argument("--stage", default=None, choices=["before_ensemble", "after_ensemble"])
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.oof)
    if args.probability_column:
        if args.probability_column not in frame.columns:
            raise SystemExit(f"Missing probability column: {args.probability_column}")
        probability = np.clip(frame[args.probability_column].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        logits = logit(probability)
        stage = args.stage or "after_ensemble"
    else:
        if "logit" not in frame.columns:
            raise SystemExit("OOF file requires logit unless --probability-column is supplied")
        logits = frame["logit"].to_numpy(dtype=float)
        stage = args.stage or "before_ensemble"
    targets = frame["target"].to_numpy(dtype=float)
    temperature = fit_temperature(logits, targets)
    raw_probability = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1.0 - 1e-6)
    calibrated_probability = np.clip(1.0 / (1.0 + np.exp(-logits / temperature)), 1e-6, 1.0 - 1e-6)
    raw_loss = float(log_loss(targets, raw_probability, labels=[0, 1]))
    calibrated_loss = float(log_loss(targets, calibrated_probability, labels=[0, 1]))
    enabled = calibrated_loss < raw_loss
    save_calibration(temperature if enabled else 1.0, args.output, stage=stage, enabled=enabled, raw_log_loss=raw_loss, calibrated_log_loss=calibrated_loss)
    print(f"Raw log loss={raw_loss:.6f}; calibrated log loss={calibrated_loss:.6f}; applied={enabled}")
    print(f"Saved calibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
