"""Evaluate OOF predictions by acquisition domain and write a report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.evaluate import evaluate_oof_by_domain, render_domain_validation_report
from datscan.training.folds import fold_quality, validate_domain_folds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", help="OOF CSV, normally emitted by train_cv.py")
    parser.add_argument("--domains", required=True, help="CSV with uid and domain_group")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--standard-oof",
        help="Optional standard-CV OOF CSV for a direct metric comparison",
    )
    parser.add_argument(
        "--domain-folds",
        help="Optional domain_folds.csv to include fold distributions and re-check leakage",
    )
    args = parser.parse_args(argv)

    domains = pd.read_csv(args.domains)
    oof = pd.read_csv(args.oof) if args.oof else None
    if oof is None and args.standard_oof:
        parser.error("--oof is required when --standard-oof is supplied")
    if oof is None:
        results = pd.DataFrame(
            columns=[
                "domain_group",
                "n",
                "normal",
                "pathologic",
                "log_loss",
                "auroc",
                "brier",
                "mean_predicted_probability",
                "true_pathologic_fraction",
                "calibration_error",
            ]
        )
    else:
        results = evaluate_oof_by_domain(oof, domains)
    quality = None
    if args.domain_folds:
        folds = pd.read_csv(args.domain_folds)
        validate_domain_folds(folds)
        quality = fold_quality(folds)
    standard = pd.read_csv(args.standard_oof) if args.standard_oof else None
    render_domain_validation_report(
        results,
        args.output,
        standard_oof=standard,
        domain_oof=oof,
        fold_quality=quality,
    )
    print(results.to_string(index=False))
    print(f"Wrote domain evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
