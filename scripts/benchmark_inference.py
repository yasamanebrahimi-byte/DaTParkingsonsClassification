"""Measure relative preprocessing, forward-pass, and checkpoint costs locally."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.data.preprocessing import preprocess_nifti  # noqa: E402
from datscan.models.resnet3d import build_model  # noqa: E402
from datscan.utils.config import PreprocessConfig, ROIConfig, load_config  # noqa: E402


def _load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = payload.get("model", {}) or {}
    model = build_model(model_config.get("name", "resnet3d"), int(model_config.get("base_channels", 16)), int(model_config.get("groups", 8)), model_config.get("layers", (2, 2, 2, 2)))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="CSV with filepath or path column")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-view", default=None, choices=["global", "roi"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else {}
    model_paths = [Path(value) for value in args.checkpoints]
    device = torch.device(args.device)
    models, payloads = zip(*[_load_checkpoint(path, device) for path in model_paths])
    preprocess_config = PreprocessConfig.from_mapping(config.get("preprocessing") or payloads[0].get("preprocess"))
    data_view = args.data_view or str(payloads[0].get("data_view", "global"))
    roi_config = ROIConfig.from_mapping(payloads[0].get("roi")) if data_view == "roi" else None
    frame = pd.read_csv(args.metadata)
    path_column = next((column for column in ("filepath", "path", "nifti_path") if column in frame.columns), None)
    if path_column is None:
        raise ValueError("Metadata must contain filepath, path, or nifti_path")
    paths = [Path(value) for value in frame[path_column].head(max(int(args.limit), 1))]
    preprocess_seconds = []
    forward_seconds = [[] for _ in models]
    for path in paths:
        start = time.perf_counter()
        tensor = torch.from_numpy(preprocess_nifti(str(path), preprocess_config, data_view, roi_config)).unsqueeze(0).to(device)
        preprocess_seconds.append(time.perf_counter() - start)
        with torch.inference_mode():
            for index, model in enumerate(models):
                start = time.perf_counter()
                model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_seconds[index].append(time.perf_counter() - start)
    preprocess_per_scan = float(np.mean(preprocess_seconds)) if preprocess_seconds else float("nan")
    forward_per_model = [float(np.mean(values)) if values else float("nan") for values in forward_seconds]
    rows = []
    for count in range(1, len(models) + 1):
        rows.append({
            "ensemble_size": count,
            "preprocessing_seconds_per_scan": preprocess_per_scan,
            "forward_seconds_per_scan": float(np.sum(forward_per_model[:count])),
            "total_seconds_per_scan": preprocess_per_scan + float(np.sum(forward_per_model[:count])),
            "relative_total_cost": (preprocess_per_scan + float(np.sum(forward_per_model[:count]))) / max(preprocess_per_scan + forward_per_model[0], 1e-12),
            "checkpoint_bytes": int(sum(path.stat().st_size for path in model_paths[:count])),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"n_scans": len(paths), "models": [str(path) for path in model_paths], "rows": rows}, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(rows, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
