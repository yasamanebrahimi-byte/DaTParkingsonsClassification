"""Aggregate long repeated OOF predictions into ensemble-like per-UID rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.repeated import aggregate_repeated_oof, repeated_summary_metrics, save_repeated_summary, variance_loss_diagnostics  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-repeats", type=int, default=None)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.oof, dtype={"uid": str})
    summary = aggregate_repeated_oof(frame, n_repeats=args.n_repeats)
    save_repeated_summary(summary, args.output)
    uncertainty = variance_loss_diagnostics(summary)
    uncertainty_path = Path(args.output).with_name(f"{Path(args.output).stem}_uncertainty.csv")
    uncertainty.to_csv(uncertainty_path, index=False)
    association = float(uncertainty["prediction_std"].corr(uncertainty["per_sample_log_loss"])) if len(uncertainty) > 1 else float("nan")
    print(pd.DataFrame(repeated_summary_metrics(summary)).T[["log_loss", "brier_score", "auroc"]].to_string())
    print(f"Prediction-std/log-loss Pearson correlation: {association:.6f}")
    print(f"Wrote repeated OOF summary with {len(summary)} UIDs to {args.output}")
    print(f"Wrote uncertainty diagnostics to {uncertainty_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
