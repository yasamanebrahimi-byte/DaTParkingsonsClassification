"""Build a reproducible checkpoint manifest for arbitrary same-family ensembles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


AGGREGATIONS = {"logit_mean", "probability_mean", "median_probability", "weighted_probability_mean"}


def _numbers(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part) for part in value.split(",") if part.strip()}


def _relative_checkpoint(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = payload.get("model", {}) or {}
    stem = path.stem
    fold_match = re.search(r"fold[_-]?(\d+)", stem)
    fold = payload.get("fold")
    if fold is None and fold_match:
        fold = int(fold_match.group(1))
    repeat = payload.get("repeat")
    if repeat is None:
        repeat_match = re.search(r"repeat[_-]?(\d+)", str(path))
        repeat = int(repeat_match.group(1)) if repeat_match else None
    return {
        "repeat": int(repeat) if repeat is not None else None,
        "fold": int(fold) if fold is not None else None,
        "fold_seed": int(payload["fold_seed"]) if payload.get("fold_seed") is not None else None,
        "training_seed": int(payload["training_seed"]) if payload.get("training_seed") is not None else None,
        "experiment_name": payload.get("experiment_name"),
        "data_view": payload.get("data_view", "roi" if str(model.get("name", "")).lower().startswith("roi") else "global"),
        "model_family": model.get("name", "unknown"),
        "model": model,
        "preprocess": payload.get("preprocess", {}),
        "roi": payload.get("roi"),
    }


def build_manifest(
    checkpoint_root: str | Path,
    output: str | Path,
    aggregation: str = "probability_mean",
    experiment: str | None = None,
    repeats: set[int] | None = None,
    folds: set[int] | None = None,
    training_seeds: set[int] | None = None,
    weights: list[float] | None = None,
) -> dict:
    aggregation = str(aggregation).lower()
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    root = Path(checkpoint_root)
    paths = sorted(root.rglob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt checkpoints found under {root}")
    entries = []
    for path in paths:
        metadata = _metadata(path)
        if repeats is not None and metadata["repeat"] not in repeats:
            continue
        if folds is not None and metadata["fold"] not in folds:
            continue
        if training_seeds is not None and metadata["training_seed"] not in training_seeds:
            continue
        entry = {"checkpoint": _relative_checkpoint(path, Path.cwd()), **metadata}
        if metadata["training_seed"] is not None:
            entry["seed"] = metadata["training_seed"]
        entries.append(entry)
    if not entries:
        raise ValueError("Checkpoint filters selected no checkpoints")
    if weights is not None and len(weights) != len(entries):
        raise ValueError("--weights must contain one value per selected checkpoint")
    if aggregation == "weighted_probability_mean" and weights is None:
        raise ValueError("weighted_probability_mean requires --weights")
    if weights is not None and (any(value < 0 for value in weights) or sum(weights) <= 0):
        raise ValueError("Manifest weights must be non-negative with positive sum")
    payload = {
        "version": 1,
        "experiment": experiment or next((entry["experiment_name"] for entry in entries if entry["experiment_name"]), root.name),
        "aggregation": aggregation,
        "weights": weights,
        "models": entries,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--aggregation", default="probability_mean", choices=sorted(AGGREGATIONS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--repeats", help="Comma-separated repeat IDs")
    parser.add_argument("--folds", help="Comma-separated fold IDs")
    parser.add_argument("--training-seeds", help="Comma-separated training seeds")
    parser.add_argument("--weights", type=float, nargs="+")
    args = parser.parse_args(argv)
    payload = build_manifest(
        args.checkpoint_root,
        args.output,
        args.aggregation,
        args.experiment,
        _numbers(args.repeats),
        _numbers(args.folds),
        _numbers(args.training_seeds),
        args.weights,
    )
    print(json.dumps({"experiment": payload["experiment"], "aggregation": payload["aggregation"], "models": len(payload["models"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
