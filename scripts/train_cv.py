"""Train the baseline model across saved folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from datscan.training.train import train_one_fold
from datscan.utils.config import ModelConfig, PreprocessConfig, ROIConfig, load_config
from datscan.utils.logging import configure_logging
from datscan.utils.reproducibility import seed_everything


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--metadata", default="artifacts/metadata/train_metadata.csv")
    parser.add_argument("--folds", default="artifacts/folds/folds.csv")
    parser.add_argument("--oof", default="artifacts/metrics/resnet18_oof.csv")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    args = parser.parse_args(argv)
    configure_logging()
    config = load_config(args.config, "configs/baseline.yaml" if Path(args.config).name != "baseline.yaml" else None)
    seed_everything(int(config.get("seed", 20260824)))
    metadata = pd.read_csv(args.metadata).merge(pd.read_csv(args.folds)[["uid", "fold"]], on="uid", how="inner", validate="one_to_one")
    preprocess = PreprocessConfig.from_mapping(config.get("preprocessing"))
    model = ModelConfig.from_mapping(config.get("model"))
    data_view = str(config.get("data_view", "roi" if model.name.lower().startswith("roi") else "global"))
    roi_config = ROIConfig.from_mapping(config.get("roi")) if data_view == "roi" else None
    cache_dir = config.get("preprocessing", {}).get("cache_dir")
    all_predictions = []
    fold_metrics = []
    for fold in sorted(metadata["fold"].unique()):
        predictions, metrics = train_one_fold(
            metadata,
            int(fold),
            preprocess,
            model,
            config.get("training", {}),
            args.checkpoint_dir,
            cache_dir=cache_dir,
            data_view=data_view,
            roi_config=roi_config,
        )
        all_predictions.append(predictions)
        fold_metrics.append(metrics)
        print(metrics)
    oof = pd.concat(all_predictions, ignore_index=True).sort_values("uid")
    Path(args.oof).parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.oof, index=False)
    from datscan.training.evaluate import evaluate_oof

    summary = evaluate_oof(oof)
    print(f"Overall OOF metrics: {summary['overall']}")
    if summary["folds"]:
        fold_losses = [row["log_loss"] for row in summary["folds"]]
        print(f"Fold log-loss variance: {float(pd.Series(fold_losses).var(ddof=0)):.6f}")
    best = min(fold_metrics, key=lambda metrics: metrics["best_validation_log_loss"])
    training = config.get("training", {})
    print(
        "Best validation log loss: "
        f"{best['best_validation_log_loss']:.6f} "
        f"(fold={best['fold']}, epoch={best['best_epoch']})"
    )
    print(
        "Parameters: "
        f"model={model.name}, base_channels={model.base_channels}, groups={model.groups}, "
        f"learning_rate={training.get('learning_rate', 2e-4)}, "
        f"weight_decay={training.get('weight_decay', 1e-3)}, "
        f"batch_size={training.get('batch_size', 2)}, "
        f"patience={training.get('patience', 10)}"
    )
    print(f"Wrote OOF predictions to {args.oof}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
