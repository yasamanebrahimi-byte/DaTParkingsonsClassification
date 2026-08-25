"""Simple non-pixel and intensity-distribution features for leakage diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SAFE_METADATA_FEATURES = [
    "shape_x", "shape_y", "shape_z", "spacing_x", "spacing_y", "spacing_z",
    "voxel_volume", "physical_extent_x", "physical_extent_y", "physical_extent_z",
    "nonzero_fraction", "p95_nonzero", "p99_nonzero", "p99_5_nonzero",
]


def metadata_features(frame: pd.DataFrame, columns: Sequence[str] = SAFE_METADATA_FEATURES) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metadata features: {sorted(missing)}")
    values = frame[list(columns)].copy()
    return values.replace([np.inf, -np.inf], np.nan).fillna(values.median(numeric_only=True)).fillna(0.0)

