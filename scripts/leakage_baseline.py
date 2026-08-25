"""Cross-validated acquisition-metadata-only leakage diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.features.quantitative import SAFE_METADATA_FEATURES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.metadata)
    numeric = [column for column in SAFE_METADATA_FEATURES if column in frame.columns]
    categorical = ["orientation"] if "orientation" in frame.columns else []
    transformer = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    model = make_pipeline(transformer, LogisticRegression(max_iter=2000, random_state=args.seed))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    probabilities = cross_val_predict(model, frame, frame["label"].astype(int), cv=splitter, method="predict_proba")[:, 1]
    targets = frame["label"].to_numpy(dtype=int)
    result = {
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "n_samples": int(len(frame)),
        "features": numeric + categorical,
    }
    report = ["# Metadata-only leakage diagnostic", "", "This model uses acquisition and geometry metadata only; no voxel values are used.", "", f"- Samples: {result['n_samples']}", f"- Features: {', '.join(result['features'])}", f"- 5-fold log loss: {result['log_loss']:.6f}", f"- 5-fold AUROC: {result['auroc']:.6f}", "", "Interpretation: compare this diagnostic with the image model. Strong metadata performance is a shortcut-learning warning and supports domain-aware validation."]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

