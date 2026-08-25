"""Create deterministic repeat-specific folds without overwriting canonical folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.folds import create_repeated_folds, fold_quality, save_repeated_folds  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    repeats = create_repeated_folds(metadata, args.n_splits, args.n_repeats, args.seed)
    save_repeated_folds(repeats, args.output_dir, base_seed=args.seed)
    for repeat, frame in repeats.items():
        print(f"repeat={repeat} seed={frame.attrs['seed']} folds={frame['fold'].nunique()} rows={len(frame)}")
        print(fold_quality(frame).to_string(index=False))
    print(f"Wrote {len(repeats)} repeated fold tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

