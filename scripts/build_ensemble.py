"""Build a simple OOF-validated ensemble manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.training.ensemble import optimize_weights


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    frames = [pd.read_csv(path) for path in args.oof]
    base = frames[0][["uid", "target"]].copy()
    probabilities = []
    for index, frame in enumerate(frames):
        if not frame["uid"].equals(base["uid"]):
            frame = base[["uid"]].merge(frame, on="uid", how="left", validate="one_to_one")
        probabilities.append(frame["probability"].to_numpy(dtype=float))
    weights = optimize_weights(base["target"].to_numpy(dtype=float), np.column_stack(probabilities))
    payload = {"method": "weighted_probability_mean", "members": [str(path) for path in args.oof], "weights": weights.tolist()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

