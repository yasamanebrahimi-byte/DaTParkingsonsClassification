"""Persist leave-one-acquisition-domain-out split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.folds import leave_one_domain_out_splits
from datscan.utils.reporting import markdown_table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--min-samples", type=int, default=30)
    args = parser.parse_args(argv)

    metadata = pd.read_csv(args.metadata)
    groups = pd.read_csv(args.groups)
    splits = leave_one_domain_out_splits(metadata, groups, args.min_samples)
    rows = []
    summaries = []
    for validation_domain, training, validation in splits:
        for _, row in training.iterrows():
            rows.append(
                {
                    "validation_domain": validation_domain,
                    "uid": row["uid"],
                    "label": row["label"],
                    "split": "train",
                    "domain_group": row["domain_group"],
                }
            )
        for _, row in validation.iterrows():
            rows.append(
                {
                    "validation_domain": validation_domain,
                    "uid": row["uid"],
                    "label": row["label"],
                    "split": "validation",
                    "domain_group": row["domain_group"],
                }
            )
        summaries.append(
            {
                "validation_domain": validation_domain,
                "sample_count": len(validation),
                "normal_count": int((validation["label"] == 0).sum()),
                "pathologic_count": int((validation["label"] == 1).sum()),
                "pathologic_fraction": float(validation["label"].mean()),
            }
        )
    manifest = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    report = Path(args.report) if args.report else output.with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Leave-one-domain-out split manifest",
        "",
        f"- Minimum validation-domain size: {args.min_samples}",
        f"- Eligible validation domains: {len(summary)}",
        "- Each row belongs to one diagnostic run identified by `validation_domain`; this is not a standard single-fold assignment file.",
        "",
        markdown_table(summary),
        "",
        "For each run, train on rows with `split=train` and evaluate on rows with `split=validation`.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} leave-one-domain-out runs to {output}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

