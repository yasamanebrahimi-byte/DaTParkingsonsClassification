"""Dataset audit and human-readable validation report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .discover import discover_niftis
from .metadata import extract_metadata


@dataclass
class ValidationResult:
    metadata: pd.DataFrame
    missing_images: List[str]
    extra_images: List[str]
    duplicate_uids: Dict[str, List[str]]
    invalid_labels: List[str]
    unreadable_images: Dict[str, str]

    @property
    def ok(self) -> bool:
        return not (self.missing_images or self.extra_images or self.duplicate_uids or self.invalid_labels or self.unreadable_images)


def read_labels(labels_path: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    expected = {"uid", "is_pathologic"}
    if not expected.issubset(labels.columns):
        raise ValueError(f"Labels must include {sorted(expected)}; found {list(labels.columns)}")
    labels = labels[["uid", "is_pathologic"]].copy()
    labels["uid"] = labels["uid"].astype(str)
    if labels["uid"].duplicated().any():
        duplicates = labels.loc[labels["uid"].duplicated(keep=False), "uid"].tolist()
        raise ValueError(f"Duplicate label UIDs: {duplicates[:10]}")
    return labels


def validate_dataset(images_root: str | Path, labels_path: str | Path, max_workers: int = 4) -> ValidationResult:
    labels = read_labels(labels_path)
    discovered = discover_niftis(images_root)
    label_uids = set(labels["uid"])
    image_uids = set(discovered)
    missing = sorted(label_uids - image_uids)
    extra = sorted(image_uids - label_uids)
    duplicate_uids = {uid: [str(p) for p in paths] for uid, paths in discovered.items() if len(paths) != 1}
    invalid_labels = sorted(labels.loc[~labels["is_pathologic"].isin([0, 1, 0.0, 1.0]), "uid"].tolist())
    rows: List[dict] = []
    unreadable: Dict[str, str] = {}

    def inspect(label_row) -> tuple[str, dict | None, str | None]:
        uid = str(label_row["uid"])
        paths = discovered.get(uid, [])
        if len(paths) != 1:
            return uid, None, None
        try:
            return uid, extract_metadata(paths[0], float(label_row["is_pathologic"]), uid), None
        except Exception as exc:  # report all bad files in one audit
            return uid, None, f"{type(exc).__name__}: {exc}"

    rows_by_uid: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(inspect, row) for _, row in labels.iterrows()]
        for future in as_completed(futures):
            uid, row, error = future.result()
            if row is not None:
                rows_by_uid[uid] = row
            if error is not None:
                unreadable[uid] = error
    rows = [rows_by_uid[uid] for uid in labels["uid"] if uid in rows_by_uid]
    metadata = pd.DataFrame(rows)
    return ValidationResult(metadata, missing, extra, duplicate_uids, invalid_labels, unreadable)


def validation_report(result: ValidationResult) -> str:
    frame = result.metadata
    lines = ["# Dataset validation report", "", f"- Valid metadata rows: {len(frame)}", f"- Dataset status: {'PASS' if result.ok else 'ISSUES FOUND'}"]
    if len(frame):
        lines += ["", "## Label balance", "", frame["label"].value_counts(dropna=False).sort_index().to_string()]
        lines += ["", "## Numeric summary", "", frame.select_dtypes(include=[np.number]).describe().T.to_string()]
    issues = {
        "Missing images": result.missing_images,
        "Extra images": result.extra_images,
        "Duplicate UIDs": list(result.duplicate_uids),
        "Invalid labels": result.invalid_labels,
        "Unreadable images": result.unreadable_images,
    }
    lines += ["", "## Findings", ""]
    for title, values in issues.items():
        lines.append(f"### {title}: {len(values)}")
        if values:
            lines.append("; ".join(map(str, values[:20])))
        lines.append("")
    return "\n".join(lines)
