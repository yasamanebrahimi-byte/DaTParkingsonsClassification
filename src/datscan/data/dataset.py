"""PyTorch dataset backed by a metadata table."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.config import PreprocessConfig, ROIConfig
from .preprocessing import preprocess_nifti
from .preprocessing_cache import PreprocessingCache


class DaTSPECTDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        config: PreprocessConfig,
        augment: Optional[Callable] = None,
        cache_dir: str | Path | None = None,
        data_view: str = "global",
        roi_config: ROIConfig | None = None,
    ) -> None:
        required = {"uid", "filepath", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.config = config
        self.augment = augment
        if data_view not in {"global", "roi"}:
            raise ValueError(f"Unknown data_view: {data_view}")
        if data_view == "roi" and (roi_config is None or not roi_config.enabled):
            raise ValueError("ROI datasets require roi.enabled=true")
        self.data_view = data_view
        self.roi_config = roi_config
        self.cache = PreprocessingCache(cache_dir, config, data_view, roi_config) if cache_dir is not None else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        source_path = Path(row["filepath"])
        def process():
            # Keep the historical two-argument call for the global path so
            # downstream users that wrap the baseline preprocessor continue
            # to work unchanged.
            if self.data_view == "global" and self.roi_config is None:
                return preprocess_nifti(str(source_path), self.config)
            return preprocess_nifti(str(source_path), self.config, self.data_view, self.roi_config)
        if self.cache is None:
            array = process()
        else:
            array = self.cache.get_or_create(
                source_path,
                str(row["uid"]),
                process,
            )
        tensor = torch.from_numpy(array)
        if self.augment is not None:
            tensor = self.augment(tensor)
        return {"image": tensor, "target": torch.tensor(float(row["label"]), dtype=torch.float32), "uid": str(row["uid"])}
