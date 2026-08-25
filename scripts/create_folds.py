"""Create and save standard or acquisition-domain-aware CV assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.folds import (
    create_domain_folds,
    create_folds,
    fold_quality,
    save_folds,
    validate_domain_folds,
)
from datscan.utils.reporting import markdown_table


def _write_quality_report(frame: pd.DataFrame, metadata: pd.DataFrame, path: str) -> None:
    quality = fold_quality(frame)
    global_fraction = float(metadata["label"].mean())
    lines = [
        "# Fold quality report",
        "",
        f"- Strategy: {'stratified_group' if 'domain_group' in frame else 'stratified'}",
        f"- Requested/created folds: {frame['fold'].nunique()}",
        f"- Samples: {len(frame)}",
        f"- Global pathologic fraction: {global_fraction:.6f}",
        "",
        "## Validation-fold distribution",
        "",
        markdown_table(quality),
        "",
    ]
    if "domain_group" in frame:
        lines.extend(
            [
                "## Leakage check",
                "",
                "No domain appears in both training and validation within any fold.",
                "",
            ]
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strategy",
        choices=("stratified", "stratified_group"),
        default="stratified",
        help="Fold strategy; stratified is the unchanged canonical IID default",
    )
    parser.add_argument(
        "--groups",
        help="CSV with uid and domain_group columns for stratified_group",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--report", help="Optional Markdown fold-quality report")
    args = parser.parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    if args.strategy == "stratified_group":
        if not args.groups:
            parser.error("--groups is required with --strategy stratified_group")
        folds = create_domain_folds(metadata, pd.read_csv(args.groups), args.n_splits, args.seed)
        validate_domain_folds(folds)
    else:
        if args.groups:
            parser.error("--groups is only valid with --strategy stratified_group")
        folds = create_folds(metadata, args.n_splits, args.seed)
    save_folds(folds, args.output)
    if args.report:
        _write_quality_report(folds, metadata, args.report)
    print(
        f"Wrote {folds['fold'].nunique()}-fold {args.strategy} assignment to {args.output}"
    )
    print(fold_quality(folds).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
