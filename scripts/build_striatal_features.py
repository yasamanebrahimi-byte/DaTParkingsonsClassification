"""Build deterministic quantitative striatal features for each examination."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from datscan.features.striatal_features import (
    StriatalFeatureConfig,
    extract_striatal_features,
    validate_feature_frame,
    write_feature_validation_report,
)
from datscan.utils.config import PreprocessConfig, ROIConfig, load_config


def _resolve_scan_path(row: pd.Series, data_dir: Path) -> Path:
    candidates = []
    if "filepath" in row and pd.notna(row["filepath"]):
        candidates.append(Path(str(row["filepath"])))
    uid = str(row["uid"])
    candidates.append(data_dir / f"{uid}.nii.gz")
    candidates.extend(sorted(data_dir.rglob(f"{uid}.nii.gz"))) if data_dir.exists() else None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve NIfTI for UID {uid}; checked metadata filepath and {data_dir / (uid + '.nii.gz')}"
    )


def build_features(
    metadata: pd.DataFrame,
    config: dict,
    data_dir: str | Path,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = {"uid", "label"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    if metadata["uid"].duplicated().any():
        raise ValueError("Metadata contains duplicate UIDs")
    preprocess = PreprocessConfig.from_mapping(config.get("preprocessing"))
    roi_mapping = dict(config.get("roi") or {})
    roi_mapping.setdefault("enabled", True)
    roi = ROIConfig.from_mapping(roi_mapping)
    feature = StriatalFeatureConfig.from_mapping(config.get("features", config))
    source_root = Path(data_dir)
    rows = []
    selected = metadata.sort_values("uid").reset_index(drop=True)
    if limit is not None:
        selected = selected.iloc[: int(limit)]
    def extract_row(row: pd.Series) -> dict:
        path = _resolve_scan_path(row, source_root)
        values = extract_striatal_features(path, preprocess, roi, feature)
        return {"uid": str(row["uid"]), "label": float(row["label"]), **values}
    records = [row for _, row in selected.iterrows()]
    worker_count = max(1, int(workers))
    if worker_count == 1:
        for index, row in enumerate(records, start=1):
            rows.append(extract_row(row))
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"Extracted {index}/{len(records)} scans")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for index, result in enumerate(executor.map(extract_row, records), start=1):
                rows.append(result)
                if index == 1 or index % 100 == 0 or index == len(records):
                    print(f"Extracted {index}/{len(records)} scans")
    frame = pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)
    feature_columns = [column for column in frame.columns if column not in {"uid", "label"}]
    stats, issues = validate_feature_frame(frame, feature_columns)
    invalid_issues = [issue for issue in issues if "NaN" in issue or "infinite" in issue]
    if invalid_issues:
        raise ValueError("Invalid feature values detected: " + "; ".join(invalid_issues))
    return frame, stats, issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/striatal_features.yaml")
    parser.add_argument("--data-dir", default="data/extracted", help="Fallback directory for metadata paths from another machine")
    parser.add_argument("--report", default="artifacts/reports/striatal_feature_validation.md")
    parser.add_argument("--limit", type=int, help="Extract only the first sorted UIDs for a smoke test")
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    metadata = pd.read_csv(args.metadata, dtype={"uid": str})
    frame, stats, issues = build_features(metadata, config, args.data_dir, args.limit, args.workers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    write_feature_validation_report(stats, issues, args.report)
    print(f"Wrote {len(frame)} rows and {len(stats)} features to {output}")
    print(f"Validation report: {args.report}")
    if issues:
        print(f"Validation warnings: {len(issues)} (see report; non-finite values are fatal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
