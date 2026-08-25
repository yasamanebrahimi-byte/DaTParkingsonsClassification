"""Report deterministic ROI coverage and optionally save local slice checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.data.nifti import load_nifti, resample_to_spacing  # noqa: E402
from datscan.data.preprocessing import (  # noqa: E402
    normalize_intensity,
    preprocess_loaded_views,
    roi_foreground_center,
)
from datscan.utils.config import PreprocessConfig, ROIConfig, load_config  # noqa: E402


def _orthogonal(volume: np.ndarray, path: Path, title: str) -> None:
    center = tuple(size // 2 for size in volume.shape)
    slices = [volume[center[0], :, :], volume[:, center[1], :], volume[:, :, center[2]]]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, label in zip(axes, slices, ("axis 0", "axis 1", "axis 2")):
        axis.imshow(np.rot90(image), cmap="inferno")
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", default="configs/roi_resnet.yaml")
    parser.add_argument("--uids", nargs="*", help="Optional explicit UID list for local visualization")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="artifacts/reports/roi_sanity.csv")
    parser.add_argument("--visualize-dir")
    args = parser.parse_args(argv)
    config = load_config(args.config, "configs/baseline.yaml" if Path(args.config).name != "baseline.yaml" else None)
    preprocess_config = PreprocessConfig.from_mapping(config.get("preprocessing"))
    roi_config = ROIConfig.from_mapping(config.get("roi"))
    if not roi_config.enabled:
        raise SystemExit("ROI inspection requires roi.enabled=true")
    metadata = pd.read_csv(args.metadata, dtype={"uid": str})
    selected = set(str(uid) for uid in args.uids) if args.uids else None
    if selected is not None:
        rows = metadata[metadata["uid"].astype(str).isin(selected)]
    else:
        rows = metadata.head(max(args.limit, 0))
    report = []
    visualization_dir = Path(args.visualize_dir) if args.visualize_dir else None
    if visualization_dir:
        visualization_dir.mkdir(parents=True, exist_ok=True)
    for _, row in rows.iterrows():
        loaded = load_nifti(row["filepath"], canonical=True)
        resampled = resample_to_spacing(loaded, preprocess_config.target_spacing_mm)
        normalized = normalize_intensity(resampled.data, preprocess_config)
        views = preprocess_loaded_views(loaded, preprocess_config, roi_config)
        positive_total = float(normalized[normalized > 0].sum())
        high_threshold = float(np.percentile(normalized[normalized > 0], 95)) if np.any(normalized > 0) else 0.0
        roi = views["roi"][0]
        roi_high = roi >= high_threshold if high_threshold > 0 else np.zeros_like(roi, dtype=bool)
        report.append({
            "uid": str(row["uid"]),
            "full_preprocessed_shape": "x".join(str(x) for x in views["global"].shape[1:]),
            "roi_shape": "x".join(str(x) for x in roi.shape),
            "roi_center_resampled_voxels": ",".join(f"{x:.2f}" for x in roi_foreground_center(normalized, preprocess_config, roi_config)),
            "positive_intensity_retained_pct": 100.0 * float(roi[roi > 0].sum()) / max(positive_total, 1e-12),
            "high_intensity_voxels_retained_pct": 100.0 * float(roi_high.sum()) / max(float((normalized >= high_threshold).sum()), 1.0),
        })
        if visualization_dir:
            _orthogonal(views["global"][0], visualization_dir / f"{row['uid']}_global.png", f"{row['uid']} global")
            _orthogonal(roi, visualization_dir / f"{row['uid']}_roi.png", f"{row['uid']} ROI")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).to_csv(output, index=False)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Wrote ROI sanity report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
