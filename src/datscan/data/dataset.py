"""PyTorch dataset backed by a metadata table."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.config import PreprocessConfig
from .preprocessing import preprocess_nifti
from .preprocessing_cache import PreprocessingCache


class DaTSPECTDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        config: PreprocessConfig,
        augment: Optional[Callable] = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        required = {"uid", "filepath", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.config = config
        self.augment = augment
        self.cache = PreprocessingCache(cache_dir, config) if cache_dir is not None else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        source_path = Path(row["filepath"])
        if self.cache is None:
            array = preprocess_nifti(str(source_path), self.config)
        else:
            array = self.cache.get_or_create(
                source_path,
                str(row["uid"]),
                lambda: preprocess_nifti(str(source_path), self.config),
            )
        tensor = torch.from_numpy(array)
        if self.augment is not None:
            tensor = self.augment(tensor)
        return {"image": tensor, "target": torch.tensor(float(row["label"]), dtype=torch.float32), "uid": str(row["uid"])}
