"""OOF evaluation and fold metric summaries."""

from __future__ import annotations

import pandas as pd

from ..utils.metrics import binary_metrics


def evaluate_oof(frame: pd.DataFrame) -> dict:
    if not {"target", "probability"}.issubset(frame.columns):
        raise ValueError("OOF frame requires target and probability columns")
    overall = binary_metrics(frame["target"], frame["probability"])
    fold_rows = []
    if "fold" in frame:
        for fold, group in frame.groupby("fold"):
            row = {"fold": int(fold), **binary_metrics(group["target"], group["probability"])}
            fold_rows.append(row)
    return {"overall": overall, "folds": fold_rows}

