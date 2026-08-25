"""Train repeated stratified CV and preserve every raw repeated OOF row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.folds import load_repeated_folds  # noqa: E402
from datscan.training.repeated import validate_repeated_oof  # noqa: E402
from datscan.training.train import train_one_fold  # noqa: E402
from datscan.utils.config import ModelConfig, PreprocessConfig, ROIConfig, load_config  # noqa: E402
from datscan.utils.logging import configure_logging  # noqa: E402
from datscan.utils.reproducibility import seed_everything  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof", help="Optional raw long-format OOF path; defaults inside output-dir")
    parser.add_argument("--seed", type=int, default=None, help="Base training seed; repeat index is added when --training-seeds is omitted")
    parser.add_argument("--training-seeds", type=int, nargs="+", help="One or more training seeds; one seed per repeat is the recommended design")
    parser.add_argument("--all-seeds-each-repeat", action="store_true", help="Train every supplied training seed on every repeat partition")
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    config = load_config(args.config, "configs/baseline.yaml" if Path(args.config).name != "baseline.yaml" else None)
    base_seed = int(args.seed if args.seed is not None else config.get("seed", 20260824))
    metadata_frame = pd.read_csv(args.metadata)
    if metadata_frame["uid"].duplicated().any():
        raise ValueError("Metadata contains duplicate UIDs")
    repeats = load_repeated_folds(args.fold_dir)
    preprocess = PreprocessConfig.from_mapping(config.get("preprocessing"))
    model = ModelConfig.from_mapping(config.get("model"))
    data_view = str(config.get("data_view", "roi" if model.name.lower().startswith("roi") else "global"))
    roi_config = ROIConfig.from_mapping(config.get("roi")) if data_view == "roi" else None
    cache_dir = config.get("preprocessing", {}).get("cache_dir")
    output_dir = Path(args.output_dir)
    experiment_name = str(args.experiment_name or config.get("experiment_name") or Path(args.config).stem)
    supplied_training_seeds = list(args.training_seeds or [])
    all_predictions = []
    fold_metrics = []
    for repeat, fold_table in sorted(repeats.items()):
        if "label" in fold_table.columns:
            labels = metadata_frame[["uid", "label"]].copy()
            checked = labels.merge(fold_table[["uid", "label"]], on="uid", how="inner", validate="one_to_one", suffixes=("_metadata", "_fold"))
            if len(checked) != len(labels) or not np.array_equal(checked["label_metadata"].astype(float), checked["label_fold"].astype(float)):
                raise ValueError(f"Repeat {repeat} fold labels do not match metadata")
        fold_table = fold_table[["uid", "fold"]].copy()
        fold_table["uid"] = fold_table["uid"].astype(str)
        current = metadata_frame.copy()
        current["uid"] = current["uid"].astype(str)
        metadata = current.merge(fold_table, on="uid", how="inner", validate="one_to_one")
        if len(metadata) != len(metadata_frame):
            raise ValueError(f"Repeat {repeat} fold file does not cover metadata exactly")
        fold_seed = int(fold_table.attrs.get("seed", base_seed + int(repeat)))
        if not supplied_training_seeds:
            training_seeds = [base_seed + int(repeat)]
        elif args.all_seeds_each_repeat or len(supplied_training_seeds) not in {1, len(repeats)}:
            training_seeds = supplied_training_seeds
        else:
            training_seeds = [supplied_training_seeds[0] if len(supplied_training_seeds) == 1 else supplied_training_seeds[int(repeat)]]
        for training_seed in training_seeds:
            training_seed = int(training_seed)
            repeat_checkpoint_dir = output_dir / f"repeat_{int(repeat)}" / f"seed_{training_seed}"
            seed_everything(training_seed)
            for fold in sorted(metadata["fold"].unique()):
                seed_everything(training_seed + int(fold))
                predictions, metrics = train_one_fold(
                    metadata,
                    int(fold),
                    preprocess,
                    model,
                    config.get("training", {}),
                    repeat_checkpoint_dir,
                    cache_dir=cache_dir,
                    data_view=data_view,
                    roi_config=roi_config,
                    augmentation_config=config.get("augmentation"),
                    seed=training_seed,
                    repeat=int(repeat),
                    fold_seed=fold_seed,
                    training_seed=training_seed,
                    experiment_name=experiment_name,
                )
                all_predictions.append(predictions)
                fold_metrics.append({"repeat": int(repeat), **metrics})
                print(fold_metrics[-1])
    oof = pd.concat(all_predictions, ignore_index=True).sort_values(["uid", "repeat"]).reset_index(drop=True)
    validate_repeated_oof(oof, n_repeats=len(repeats))
    oof_path = Path(args.oof) if args.oof else output_dir / "repeated_oof.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(oof_path, index=False)
    run_manifest = {
        "version": 1,
        "experiment": experiment_name,
        "folds_per_repeat": int(max(len(frame["fold"].unique()) for frame in repeats.values())),
        "repeats": int(len(repeats)),
        "training_seeds": sorted({int(value) for value in oof["training_seed"].dropna().unique()}) if "training_seed" in oof else [],
        "fold_seeds": {str(int(repeat)): int(frame.attrs.get("seed", base_seed + int(repeat))) for repeat, frame in repeats.items()},
        "models": [
            {
                "repeat": int(row["repeat"]),
                "fold": int(row["fold"]),
                "fold_seed": int(row["fold_seed"]),
                "training_seed": int(row["training_seed"]),
                "checkpoint": str((output_dir / f"repeat_{int(row['repeat'])}" / f"seed_{int(row['training_seed'])}" / f"{('roi_resnet3d' if data_view == 'roi' else 'resnet3d')}_fold{int(row['fold'])}.pt").as_posix()),
                "experiment_name": experiment_name,
            }
            for row in pd.DataFrame(fold_metrics).drop_duplicates(["repeat", "fold", "training_seed"]).to_dict("records")
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Trained {len(fold_metrics)} models across {len(repeats)} repeats")
    print(f"Wrote {len(oof)} raw repeated OOF rows to {oof_path}")
    print(f"Wrote run manifest to {output_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
