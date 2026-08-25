"""Save orthogonal slice montages and statistics for local training scans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.data.dataset import DaTSPECTDataset  # noqa: E402
from datscan.data.transforms import build_augmentation  # noqa: E402
from datscan.utils.config import PreprocessConfig, load_config  # noqa: E402


def _stats(uid: str, sample: str, original: np.ndarray, value: np.ndarray, threshold: float) -> dict[str, object]:
    original_high = original >= threshold
    return {
        "uid": uid,
        "sample": sample,
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "positive_fraction": float((value > 0).mean()),
        "high_intensity_fraction": float((value >= threshold).mean()),
        "high_intensity_retention": float((value[original_high] >= threshold).mean()) if original_high.any() else 0.0,
        "finite": bool(np.isfinite(value).all()),
    }


def _slices(volume: np.ndarray) -> list[np.ndarray]:
    center = tuple(size // 2 for size in volume.shape)
    return [volume[:, :, center[2]].T, volume[:, center[1], :].T, volume[center[0], :, :].T]


def _save_montage(path: Path, uid: str, original: np.ndarray, augmented: list[np.ndarray], traces: list[list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes = [original, *augmented]
    labels = ["original", *[f"augmented {index + 1}" for index in range(len(augmented))]]
    figure, axes = plt.subplots(len(volumes), 3, figsize=(10, max(3, 2.7 * len(volumes))), squeeze=False)
    for row, (label, volume) in enumerate(zip(labels, volumes)):
        slices = _slices(volume)
        vmax = max(float(volume.max()), 1e-6)
        for column, (axis, image, title) in enumerate(zip(axes[row], slices, ("axial", "coronal", "sagittal"))):
            axis.imshow(image, cmap="hot", vmin=0.0, vmax=vmax)
            axis.set_title(title if row == 0 else "")
            axis.axis("off")
        axes[row, 0].set_ylabel(label, rotation=90, va="center")
    figure.suptitle(f"UID {uid}")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Training metadata CSV; only explicitly selected UIDs are read")
    parser.add_argument("--uid", nargs="+", required=True)
    parser.add_argument("--config", default="configs/highres_scanner_aug.yaml")
    parser.add_argument("--output-dir", default="artifacts/plots/scanner_augmentation")
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args(argv)
    if args.samples < 1:
        raise SystemExit("--samples must be positive")

    config_path = Path(args.config)
    config = load_config(config_path, "configs/baseline.yaml" if config_path.name != "baseline.yaml" else None)
    metadata = pd.read_csv(args.metadata, dtype={"uid": str})
    requested = [str(uid) for uid in args.uid]
    selected = metadata[metadata["uid"].astype(str).isin(requested)].copy()
    if len(selected) != len(set(requested)):
        missing = sorted(set(requested) - set(selected["uid"].astype(str)))
        raise SystemExit(f"UIDs missing from the supplied training metadata: {missing}")
    selected["uid"] = selected["uid"].astype(str)
    preprocess = PreprocessConfig.from_mapping(config.get("preprocessing"))
    cache_dir = config.get("preprocessing", {}).get("cache_dir")
    base_dataset = DaTSPECTDataset(selected, preprocess, cache_dir=cache_dir)
    augmentation = build_augmentation(config.get("augmentation"), legacy_augment=bool(config.get("training", {}).get("augment", True)))
    if augmentation is None:
        raise SystemExit("The inspection config must select a stochastic augmentation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in range(len(base_dataset)):
        item = base_dataset[index]
        uid = str(item["uid"])
        original = item["image"].detach().cpu().numpy()[0]
        positive = original[original > 0]
        threshold = float(np.percentile(positive, 95)) if positive.size else 0.0
        rows.append(_stats(uid, "original", original, original, threshold))
        augmented_values = []
        traces = []
        for sample_index in range(args.samples):
            augmented = augmentation(item["image"])
            value = augmented.detach().cpu().numpy()[0]
            augmented_values.append(value)
            traces.append(list(getattr(augmentation, "last_trace", [])))
            rows.append(_stats(uid, f"augmented_{sample_index + 1}", original, value, threshold))
            print(f"{uid} sample {sample_index + 1}: {traces[-1]}")
        _save_montage(output_dir / f"{uid}_montage.png", uid, original, augmented_values, traces)

    stats_path = output_dir / "augmentation_stats.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Wrote augmentation montages to {output_dir}")
    print(f"Wrote quantitative sanity checks to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
