"""Build a metadata table from an extracted image directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.data.metadata import extract_metadata
from datscan.data.validation import read_labels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    labels = read_labels(args.labels).set_index("uid")
    rows = []
    for path in sorted(Path(args.images).rglob("*.nii.gz")):
        uid = path.name[:-7]
        if uid in labels.index:
            rows.append(extract_metadata(path, float(labels.loc[uid, "is_pathologic"]), uid))
    import pandas as pd

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} metadata rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
