"""Train a fold-safe logistic, nonlinear, or optional FeatureMLP model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from datscan.training.feature_cv import train_feature_cv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--folds", default="artifacts/folds/folds.csv")
    parser.add_argument("--model", choices=["logistic", "histgb", "feature_mlp"], default="logistic")
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-families", help="Comma-separated ablation families, e.g. uptake,asymmetry,background_ratio")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--tune-C", action="store_true", help="Tune logistic C inside each outer training fold")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    features = pd.read_csv(args.features, dtype={"uid": str})
    folds = pd.read_csv(args.folds, dtype={"uid": str})
    families = [value.strip() for value in args.feature_families.split(",")] if args.feature_families else None
    oof, metrics = train_feature_cv(
        features,
        folds,
        args.model,
        args.oof,
        args.output_dir,
        feature_families=families,
        C=args.C,
        tune_c=args.tune_C,
        random_state=args.seed,
    )
    print("Overall OOF metrics:", metrics["overall"])
    print(f"Fold log-loss standard deviation: {metrics['fold_log_loss_std']:.6f}")
    print(f"Features used: {metrics['feature_count']}; removed: {len(metrics['removed_features'])}")
    print(f"Wrote {len(oof)} OOF predictions to {args.oof}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
