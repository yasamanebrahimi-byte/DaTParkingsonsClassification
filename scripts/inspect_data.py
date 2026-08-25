"""Extract an archive and generate the first dataset validation report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.data.discover import safe_extract_zip
from datscan.data.validation import validate_dataset, validation_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    extract_dir = Path(args.extract_dir)
    if not any(extract_dir.rglob("*.nii.gz")):
        safe_extract_zip(args.archive, extract_dir)
    result = validate_dataset(extract_dir, args.labels, max_workers=args.max_workers)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    result.metadata.to_csv(args.metadata, index=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(validation_report(result), encoding="utf-8")
    print(f"Validated {len(result.metadata)} scans; status={'PASS' if result.ok else 'ISSUES FOUND'}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
