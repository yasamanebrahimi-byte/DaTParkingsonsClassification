"""Create a compact, data-grounded dataset summary from the metadata table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.metadata)
    lines = ["# Dataset summary", "", f"- Scans: {len(frame)}", f"- Normal: {(frame['label'] == 0).sum()}", f"- Pathologic: {(frame['label'] == 1).sum()}", f"- Pathologic fraction: {frame['label'].mean():.4f}", "", "## Shapes", "", frame[["shape_x", "shape_y", "shape_z"]].describe().T.to_string(), "", "## Spacing (mm)", "", frame[["spacing_x", "spacing_y", "spacing_z"]].describe().T.to_string(), "", "## Orientations", "", frame["orientation"].value_counts(dropna=False).to_string(), "", "## Stored dtypes", "", frame["dtype"].value_counts(dropna=False).to_string(), "", "## Intensity and foreground", "", frame[["min_intensity", "max_intensity", "median_nonzero", "p99_5_nonzero", "nonzero_fraction"]].describe().T.to_string(), "", "## Notes", "", "The acquisition geometry is heterogeneous: spacing, matrix shape, physical extent, and intensity scale vary across scans. The model therefore uses affine-aware resampling and per-scan normalization. Metadata-only leakage results are reported separately."]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

