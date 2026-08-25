"""Evaluate and save an OOF-calibrated global/ROI probability ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datscan.training.ensemble import (  # noqa: E402
    blend_probabilities,
    grid_search_two_model_weights,
    optimize_weights,
    prediction_diversity,
)
from datscan.utils.metrics import binary_metrics  # noqa: E402


def _read_aligned(global_path: str, roi_path: str) -> pd.DataFrame:
    global_frame = pd.read_csv(global_path)
    roi_frame = pd.read_csv(roi_path)
    required = {"uid", "target", "probability"}
    for path, frame in ((global_path, global_frame), (roi_path, roi_frame)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    left = global_frame.rename(columns={"target": "target_global", "probability": "global_probability"})
    right = roi_frame.rename(columns={"target": "target_roi", "probability": "roi_probability"})
    if ("fold" in global_frame) != ("fold" in roi_frame):
        raise ValueError("Global and ROI OOF files must both contain fold assignments")
    columns = ["uid", "global_probability", "target_global"]
    right_columns = ["uid", "roi_probability", "target_roi"]
    if "fold" in global_frame:
        columns.append("fold")
        right = right.rename(columns={"fold": "fold_roi"})
        right_columns.append("fold_roi")
    merged = left[columns].merge(right[right_columns], on="uid", how="inner", validate="one_to_one")
    if len(merged) != len(global_frame) or len(merged) != len(roi_frame):
        raise ValueError("Global and ROI OOF files must contain exactly the same UIDs")
    if not np.allclose(merged["target_global"], merged["target_roi"]):
        raise ValueError("Global and ROI OOF targets do not match")
    if "fold_roi" in merged and not np.array_equal(merged["fold"].to_numpy(), merged["fold_roi"].to_numpy()):
        raise ValueError("Global and ROI OOF fold assignments do not match")
    merged["target"] = merged["target_global"].astype(float)
    return merged.sort_values("uid").reset_index(drop=True)


def _write_two_model_ensemble(global_path: str, roi_path: str, output: str, step: float) -> dict:
    aligned = _read_aligned(global_path, roi_path)
    targets = aligned["target"].to_numpy(dtype=float)
    global_probability = aligned["global_probability"].to_numpy(dtype=float)
    roi_probability = aligned["roi_probability"].to_numpy(dtype=float)
    global_weight, weights, grid = grid_search_two_model_weights(targets, global_probability, roi_probability, step=step)
    fifty_fifty = blend_probabilities(global_probability, roi_probability, 0.5)
    optimized = blend_probabilities(global_probability, roi_probability, global_weight)
    report = {
        "Global": binary_metrics(targets, global_probability),
        "ROI": binary_metrics(targets, roi_probability),
        "50/50 Ensemble": binary_metrics(targets, fifty_fifty),
        "Optimized Ensemble": binary_metrics(targets, optimized),
    }
    diversity = prediction_diversity(targets, global_probability, roi_probability)
    aligned["ensemble_probability"] = optimized
    aligned["probability"] = optimized
    aligned["logit"] = logit(optimized)
    keep = ["uid"]
    if "fold" in aligned:
        keep.append("fold")
    keep.extend(["target", "global_probability", "roi_probability", "ensemble_probability", "probability", "logit"])
    output_path = Path(output)
    oof_path = output_path.with_name(f"{output_path.stem}_oof.csv")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    aligned[keep].to_csv(oof_path, index=False)
    payload = {
        "version": 2,
        "method": "grid_search_weighted_probability_mean",
        "members": [str(global_path), str(roi_path)],
        "member_names": ["global", "roi"],
        "weights": weights.tolist(),
        "global_weight": float(global_weight),
        "roi_weight": float(1.0 - global_weight),
        "grid_step": float(step),
        "grid": grid,
        "metrics": report,
        "diversity": diversity,
        "oof_path": str(oof_path),
        "calibration_stage": "after_ensemble",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))
    print("OOF comparison:")
    print(pd.DataFrame(report).T[["log_loss", "auroc", "brier_score"]].to_string())
    print(f"Wrote ensemble OOF predictions to {oof_path}")
    return payload


def _write_generic_ensemble(paths: list[str], output: str) -> None:
    frames = [pd.read_csv(path) for path in paths]
    base = frames[0][["uid", "target"]].copy()
    probabilities = []
    for frame in frames:
        aligned = base.merge(frame[["uid", "probability"]], on="uid", how="left", validate="one_to_one")
        if aligned["probability"].isna().any():
            raise ValueError("Every ensemble member must contain every base UID")
        probabilities.append(aligned["probability"].to_numpy(dtype=float))
    matrix = np.column_stack(probabilities)
    weights = optimize_weights(base["target"].to_numpy(dtype=float), matrix)
    payload = {"version": 1, "method": "weighted_probability_mean", "members": paths, "weights": weights.tolist()}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-oof")
    parser.add_argument("--roi-oof")
    parser.add_argument("--oof", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args(argv)
    if bool(args.global_oof) != bool(args.roi_oof):
        parser.error("--global-oof and --roi-oof must be supplied together")
    if args.global_oof and args.roi_oof:
        _write_two_model_ensemble(args.global_oof, args.roi_oof, args.output, args.grid_step)
    elif args.oof:
        _write_generic_ensemble(args.oof, args.output)
    else:
        parser.error("provide --global-oof/--roi-oof or --oof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
