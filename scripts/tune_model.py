"""Run or resume targeted random search and promote finalists to full CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.tuning import promote_trials, run_screening_study, write_tuning_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OOF-log-loss-first targeted hyperparameter tuning")
    parser.add_argument("--search-config", help="YAML search-space definition for a new/resumed screening study")
    parser.add_argument("--metadata", help="Metadata CSV; required for a new study")
    parser.add_argument("--folds", help="Fixed screening-fold CSV; created deterministically if absent")
    parser.add_argument("--output-dir", help="Study directory for a new/resumed screening study")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--study", help="Existing study directory for promotion/reporting")
    parser.add_argument("--promote-top", type=int, help="Run full CV for this many best screening trials")
    parser.add_argument("--full-folds", help="Canonical full-CV folds CSV")
    parser.add_argument("--full-epochs", type=int)
    parser.add_argument("--full-patience", type=int)
    parser.add_argument("--confirm-seed", type=int, help="Optional second training seed for the top promoted trials")
    parser.add_argument("--confirm-top", type=int, default=2)
    parser.add_argument("--domain-folds", help="Optional domain-aware folds CSV for the top promoted trial(s)")
    parser.add_argument("--domain-top", type=int, default=1)
    parser.add_argument("--report", nargs="?", const="", help="Write the final Markdown report, optionally to this path")
    parser.add_argument("--baseline-oof", help="Optional base-model OOF CSV for comparison")
    args = parser.parse_args(argv)

    study_value = args.study or args.output_dir
    if not study_value:
        parser.error("provide --output-dir for screening or --study for promotion/reporting")
    study_dir = Path(study_value)

    if args.search_config or args.output_dir:
        if not args.search_config or not args.metadata or not args.folds or not args.output_dir:
            parser.error("screening requires --search-config, --metadata, --folds, and --output-dir")
        run_screening_study(
            args.search_config,
            args.metadata,
            args.folds,
            args.output_dir,
            n_trials=args.n_trials,
            seed=args.seed,
            resume=args.resume,
        )
        study_dir = Path(args.output_dir)

    if args.promote_top is not None:
        if not args.full_folds:
            parser.error("promotion requires --full-folds")
        promote_trials(
            study_dir,
            args.full_folds,
            promote_top=args.promote_top,
            full_epochs=args.full_epochs,
            full_patience=args.full_patience,
            confirm_seed=args.confirm_seed,
            confirm_top=args.confirm_top,
            domain_folds_path=args.domain_folds,
            domain_top=args.domain_top,
        )

    if args.report is not None or args.promote_top is not None:
        report_path = None if args.report in {None, ""} else args.report
        written = write_tuning_report(study_dir, report_path, baseline_oof=args.baseline_oof)
        print(f"Wrote tuning report to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
