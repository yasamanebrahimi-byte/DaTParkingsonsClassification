"""Create and save the canonical CV assignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.folds import create_folds, save_folds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    save_folds(create_folds(metadata, args.n_splits, args.seed), args.output)
    print(f"Wrote {args.n_splits}-fold assignment to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
