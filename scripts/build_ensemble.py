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
    blend_logits,
    blend_probabilities,
    grid_search_two_model_weights,
    grid_search_two_model_logit_weights,
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


def _inferred_member_names(paths: list[str], supplied: list[str] | None) -> list[str]:
    if supplied:
        if len(supplied) != len(paths):
            raise ValueError("--member-names must contain one name per OOF file")
        return [str(name) for name in supplied]
    names = []
    for index, path in enumerate(paths):
        stem = Path(path).stem.lower()
        if "feature" in stem or "striatal" in stem:
            name = "feature"
        elif index == 0 or any(token in stem for token in ("cnn", "resnet", "roi", "highres")):
            name = "cnn"
        else:
            name = f"member_{index + 1}"
        if name in names:
            name = f"{name}_{index + 1}"
        names.append(name)
    return names


def _read_generic_aligned(paths: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if len(paths) < 2:
        raise ValueError("At least two OOF files are required for an ensemble")
    frames = [pd.read_csv(path, dtype={"uid": str}) for path in paths]
    required = {"uid", "target", "probability"}
    for path, frame in zip(paths, frames):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["uid"].duplicated().any():
            raise ValueError(f"{path} contains duplicate UIDs")
    if any(("fold" in frame) != ("fold" in frames[0]) for frame in frames):
        raise ValueError("All ensemble OOF files must either contain fold assignments or omit them")
    names = [f"probability_{index}" for index in range(len(paths))]
    base = frames[0][["uid", "target"]].copy()
    base["uid"] = base["uid"].astype(str)
    if "fold" in frames[0]:
        base["fold"] = frames[0]["fold"].to_numpy()
    for index, frame in enumerate(frames):
        current = frame[["uid", "target", "probability"] + (["fold"] if "fold" in frame else [])].copy()
        current["uid"] = current["uid"].astype(str)
        if set(current["uid"]) != set(base["uid"]):
            raise ValueError("All ensemble OOF files must contain exactly the same UIDs")
        current = current.rename(columns={"target": f"target_{index}", "probability": names[index], "fold": f"fold_{index}"})
        keep = ["uid", f"target_{index}", names[index]] + ([f"fold_{index}"] if f"fold_{index}" in current else [])
        base = base.merge(current[keep], on="uid", how="inner", validate="one_to_one")
        if not np.allclose(base["target"].to_numpy(dtype=float), base[f"target_{index}"].to_numpy(dtype=float)):
            raise ValueError("Ensemble OOF targets do not match")
        if "fold" in base and f"fold_{index}" in base and not np.array_equal(base["fold"], base[f"fold_{index}"]):
            raise ValueError("Ensemble OOF fold assignments do not match")
    return base.sort_values("uid").reset_index(drop=True), names


def _write_generic_ensemble(paths: list[str], output: str, member_names: list[str] | None, step: float) -> None:
    aligned, probability_columns = _read_generic_aligned(paths)
    targets = aligned["target"].to_numpy(dtype=float)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = _inferred_member_names(paths, member_names)
    if len(paths) == 2:
        first = aligned[probability_columns[0]].to_numpy(dtype=float)
        second = aligned[probability_columns[1]].to_numpy(dtype=float)
        probability_weight, probability_weights, probability_grid = grid_search_two_model_weights(targets, first, second, step=step)
        logit_weight, logit_grid = grid_search_two_model_logit_weights(targets, first, second, step=step)
        p50 = blend_probabilities(first, second, 0.5)
        p_probability = blend_probabilities(first, second, probability_weight)
        p_logit = blend_logits(first, second, logit_weight)
        probability_metrics = binary_metrics(targets, p_probability)
        logit_metrics = binary_metrics(targets, p_logit)
        use_logit = logit_metrics["log_loss"] < probability_metrics["log_loss"]
        selected = p_logit if use_logit else p_probability
        selected_method = "grid_search_weighted_logit_mean" if use_logit else "grid_search_weighted_probability_mean"
        report = {
            names[0]: binary_metrics(targets, first),
            names[1]: binary_metrics(targets, second),
            "50/50 blend": binary_metrics(targets, p50),
            "optimized probability blend": probability_metrics,
            "optimized logit blend": logit_metrics,
        }
        diversity = prediction_diversity(targets, first, second)
        selected_weight = logit_weight if use_logit else probability_weight
        weights = [selected_weight, 1.0 - selected_weight]
        grid = probability_grid if not use_logit else logit_grid
    else:
        matrix = np.column_stack([aligned[column].to_numpy(dtype=float) for column in probability_columns])
        weights = optimize_weights(targets, matrix).tolist()
        selected = np.clip(matrix @ np.asarray(weights), 1e-6, 1.0 - 1e-6)
        selected_method = "weighted_probability_mean"
        report = {name: binary_metrics(targets, matrix[:, index]) for index, name in enumerate(names)}
        report["optimized ensemble"] = binary_metrics(targets, selected)
        diversity = {}
        grid = []
    result = aligned[["uid"] + (["fold"] if "fold" in aligned else []) + ["target"]].copy()
    for index, name in enumerate(names):
        probability = aligned[probability_columns[index]].to_numpy(dtype=float)
        result[f"{name}_probability"] = probability
        result[f"{name}_log_loss"] = -(targets * np.log(np.clip(probability, 1e-6, 1 - 1e-6)) + (1 - targets) * np.log(1 - np.clip(probability, 1e-6, 1 - 1e-6)))
    result["ensemble_probability"] = selected
    result["probability"] = selected
    result["logit"] = logit(np.clip(selected, 1e-6, 1.0 - 1e-6))
    oof_path = output_path.with_name(f"{output_path.stem}_oof.csv")
    result.to_csv(oof_path, index=False)
    cases_path = output_path.with_name(f"{output_path.stem}_high_loss_cases.csv")
    result.sort_values(f"{names[0]}_log_loss", ascending=False).to_csv(cases_path, index=False)
    worse_path = output_path.with_name(f"{output_path.stem}_ensemble_worse_cases.csv")
    first_loss = result[f"{names[0]}_log_loss"]
    ensemble_loss = -(targets * np.log(np.clip(selected, 1e-6, 1 - 1e-6)) + (1 - targets) * np.log(1 - np.clip(selected, 1e-6, 1 - 1e-6)))
    worse = result.loc[ensemble_loss > first_loss].copy()
    worse["ensemble_log_loss"] = ensemble_loss[ensemble_loss > first_loss]
    worse.sort_values("ensemble_log_loss", ascending=False).to_csv(worse_path, index=False)
    payload = {
        "version": 3,
        "method": selected_method,
        "members": paths,
        "member_names": names,
        "weights": weights,
        "grid_step": float(step),
        "grid": grid,
        "metrics": report,
        "diversity": diversity,
        "oof_path": str(oof_path),
        "high_loss_cases_path": str(cases_path),
        "ensemble_worse_cases_path": str(worse_path),
        "calibration_stage": "after_ensemble",
    }
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))
    print("OOF comparison:")
    print(pd.DataFrame(report).T[["log_loss", "auroc", "brier_score"]].to_string())
    print(f"Wrote ensemble OOF predictions to {oof_path}")
    print(f"Wrote high-loss comparison to {cases_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-oof")
    parser.add_argument("--roi-oof")
    parser.add_argument("--oof", "--predictions", dest="oof", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--member-names", nargs="+", help="Names for generic ensemble members, e.g. cnn feature")
    args = parser.parse_args(argv)
    if bool(args.global_oof) != bool(args.roi_oof):
        parser.error("--global-oof and --roi-oof must be supplied together")
    if args.global_oof and args.roi_oof:
        _write_two_model_ensemble(args.global_oof, args.roi_oof, args.output, args.grid_step)
    elif args.oof:
        _write_generic_ensemble(args.oof, args.output, args.member_names, args.grid_step)
    else:
        parser.error("provide --global-oof/--roi-oof or --oof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
