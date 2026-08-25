"""Reproducible stratified fold creation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(metadata: pd.DataFrame, n_splits: int = 5, seed: int = 20260824) -> pd.DataFrame:
    if metadata["label"].isna().any():
        raise ValueError("Cannot create folds with missing labels")
    result = metadata[["uid", "label"]].copy()
    result["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(splitter.split(result, result["label"].astype(int))):
        result.loc[validation_indices, "fold"] = fold
    return result.sort_values("uid").reset_index(drop=True)


def save_folds(folds: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)

