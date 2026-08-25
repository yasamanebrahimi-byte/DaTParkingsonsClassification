"""Compare mild and scanner-robust OOF probabilities by confidence error."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.utils.metrics import binary_metrics  # noqa: E402


def _per_sample_log_loss(target: pd.Series, probability: pd.Series) -> np.ndarray:
    y = target.to_numpy(dtype=float)
    p = np.clip(probability.to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mild-oof", required=True)
    parser.add_argument("--scanner-oof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    mild = pd.read_csv(args.mild_oof)[["uid", "target", "probability", "fold"]].rename(columns={"probability": "mild_probability", "fold": "mild_fold"})
    scanner = pd.read_csv(args.scanner_oof)[["uid", "target", "probability", "fold"]].rename(columns={"probability": "scanner_aug_probability", "fold": "scanner_fold", "target": "scanner_target"})
    result = mild.merge(scanner, on="uid", how="inner", validate="one_to_one")
    if len(result) != len(mild) or len(result) != len(scanner):
        raise ValueError("Mild and scanner OOF files must contain exactly the same UIDs")
    if not np.allclose(result["target"], result["scanner_target"]):
        raise ValueError("Mild and scanner OOF targets do not match")
    if not np.array_equal(result["mild_fold"], result["scanner_fold"]):
        raise ValueError("Mild and scanner OOF fold assignments do not match")

    result["mild_log_loss"] = _per_sample_log_loss(result["target"], result["mild_probability"])
    result["scanner_aug_log_loss"] = _per_sample_log_loss(result["target"], result["scanner_aug_probability"])
    result["difference"] = result["mild_log_loss"] - result["scanner_aug_log_loss"]
    result = result.rename(columns={"mild_fold": "fold"})
    columns = ["uid", "target", "mild_probability", "scanner_aug_probability", "mild_log_loss", "scanner_aug_log_loss", "difference", "fold"]
    result = result.sort_values("difference", ascending=False)[columns]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    mild_metrics = binary_metrics(result["target"], result["mild_probability"])
    scanner_metrics = binary_metrics(result["target"], result["scanner_aug_probability"])
    fold_summary = result.groupby("fold").agg(mild_log_loss=("mild_log_loss", "mean"), scanner_aug_log_loss=("scanner_aug_log_loss", "mean"))
    print(pd.DataFrame({"mild": mild_metrics, "scanner_robust": scanner_metrics})[["mild", "scanner_robust"]].loc[["log_loss", "auroc", "brier_score"]].to_string())
    print("Fold log loss:")
    print(fold_summary.to_string())
    print(f"Fold log-loss standard deviation: mild={fold_summary['mild_log_loss'].std(ddof=0):.6f}, scanner_robust={fold_summary['scanner_aug_log_loss'].std(ddof=0):.6f}")
    print(f"Wrote confidence-error comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
