"""Cross-validation training loop shared by all configured 3D models."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data.dataset import DaTSPECTDataset
from ..data.transforms import build_augmentation, resolve_augmentation_config
from ..models.resnet3d import build_model
from ..utils.config import ModelConfig, PreprocessConfig, ROIConfig
from ..utils.metrics import binary_metrics


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)
    return autocast(enabled=enabled, dtype=torch.bfloat16 if enabled else torch.float32)


def _seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the DataLoader worker's torch seed."""
    import random

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_fold(
    frame: pd.DataFrame,
    fold: int,
    preprocess_config: PreprocessConfig,
    model_config: ModelConfig,
    training_config: Dict,
    checkpoint_dir: str | Path,
    cache_dir: str | Path | None = None,
    data_view: str = "global",
    roi_config: ROIConfig | None = None,
    checkpoint_prefix: str | None = None,
    augmentation_config: dict | None = None,
    seed: int | None = None,
    repeat: int | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    device = _device(str(training_config.get("device", "auto")))
    train_frame = frame[frame["fold"] != fold].reset_index(drop=True)
    valid_frame = frame[frame["fold"] == fold].reset_index(drop=True)
    if augmentation_config is None and "augmentation" in training_config:
        augmentation_config = training_config.get("augmentation")
    legacy_augment = bool(training_config.get("augment", True))
    augmentation = build_augmentation(augmentation_config, legacy_augment=legacy_augment)
    resolved_augmentation = resolve_augmentation_config(augmentation_config, legacy_augment=legacy_augment)
    train_dataset = DaTSPECTDataset(
        train_frame,
        preprocess_config,
        augment=augmentation,
        cache_dir=cache_dir,
        data_view=data_view,
        roi_config=roi_config,
    )
    valid_dataset = DaTSPECTDataset(
        valid_frame,
        preprocess_config,
        cache_dir=cache_dir,
        data_view=data_view,
        roi_config=roi_config,
    )
    loader_args = {"batch_size": int(training_config.get("batch_size", 2)), "num_workers": int(training_config.get("num_workers", 0)), "pin_memory": device.type == "cuda"}
    train_generator = None
    worker_init_fn = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(int(seed) + int(fold) * 1009)
        worker_init_fn = _seed_worker
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        worker_init_fn=worker_init_fn,
        **loader_args,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        **loader_args,
    )
    model = build_model(
        model_config.name,
        model_config.base_channels,
        model_config.groups,
        model_config.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config.get("learning_rate", 2e-4)), weight_decay=float(training_config.get("weight_decay", 1e-3)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(training_config.get("epochs", 50)), 1))
    criterion = nn.BCEWithLogitsLoss()
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = _scaler(use_amp)
    best_loss = float("inf")
    best_state = None
    patience = int(training_config.get("patience", 10))
    stale = 0
    best_epoch = None
    for epoch in range(int(training_config.get("epochs", 50))):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if training_config.get("grad_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        validation = predict_loader(model, valid_loader, device, use_amp)
        validation_metrics = binary_metrics(validation["target"], validation["probability"])
        if validation_metrics["log_loss"] < best_loss:
            best_loss = validation_metrics["log_loss"]
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for fold {fold}")
    model.load_state_dict(best_state)
    predictions = predict_loader(model, valid_loader, device, use_amp)
    out = valid_frame[["uid", "fold", "label"]].copy().rename(columns={"label": "target"})
    out["logit"] = predictions["logit"]
    out["probability"] = predictions["probability"]
    if repeat is not None:
        out["repeat"] = int(repeat)
    if checkpoint_prefix is None:
        checkpoint_prefix = "roi_resnet3d" if data_view == "roi" else "resnet3d"
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_prefix}_fold{fold}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "state_dict": best_state,
            "fold": fold,
            "repeat": int(repeat) if repeat is not None else None,
            "preprocess": asdict(preprocess_config),
            "model": asdict(model_config),
            "data_view": data_view,
            "roi": asdict(roi_config) if roi_config is not None else None,
            "augmentation": resolved_augmentation,
        },
        checkpoint_path,
    )
    return out, {
        "fold": fold,
        **binary_metrics(out["target"], out["probability"]),
        "best_validation_log_loss": float(best_loss),
        "best_epoch": best_epoch,
    }


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    logits = []
    targets = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with _autocast(use_amp):
                output = model(images)
            logits.append(output.float().cpu().numpy())
            targets.append(batch["target"].numpy())
    logits_array = np.concatenate(logits) if logits else np.empty(0, dtype=float)
    return {"logit": logits_array, "probability": 1.0 / (1.0 + np.exp(-logits_array)), "target": np.concatenate(targets) if targets else np.empty(0)}
