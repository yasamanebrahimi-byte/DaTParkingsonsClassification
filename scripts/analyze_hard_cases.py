"""Write an OOF hard-case comparison for global, ROI, and ensemble predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def _loss(target: pd.Series, probability: pd.Series) -> np.ndarray:
    values = np.clip(probability.to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    return -(target.to_numpy(dtype=float) * np.log(values) + (1.0 - target.to_numpy(dtype=float)) * np.log(1.0 - values))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-oof", required=True)
    parser.add_argument("--roi-oof", required=True)
    parser.add_argument("--ensemble-oof", required=True, help="CSV emitted alongside the ensemble manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    global_frame = pd.read_csv(args.global_oof)[["uid", "target", "probability", "fold"]].rename(columns={"probability": "global_probability"})
    roi_frame = pd.read_csv(args.roi_oof)[["uid", "probability"]].rename(columns={"probability": "roi_probability"})
    ensemble_frame = pd.read_csv(args.ensemble_oof)[["uid", "probability"]].rename(columns={"probability": "ensemble_probability"})
    result = global_frame.merge(roi_frame, on="uid", validate="one_to_one").merge(ensemble_frame, on="uid", validate="one_to_one")
    result["global_loss"] = _loss(result["target"], result["global_probability"])
    result["roi_loss"] = _loss(result["target"], result["roi_probability"])
    result["ensemble_loss"] = _loss(result["target"], result["ensemble_probability"])
    columns = ["uid", "target", "global_probability", "roi_probability", "ensemble_probability", "global_loss", "roi_loss", "ensemble_loss", "fold"]
    result = result.sort_values("global_loss", ascending=False)[columns]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.head(max(args.limit, 0)).to_csv(output, index=False)
    print(f"Wrote {min(len(result), max(args.limit, 0))} hard cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
