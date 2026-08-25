"""Metadata extraction from NIfTI headers and voxel arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .nifti import load_nifti, voxel_spacing


def _percentile_nonzero(data: np.ndarray, percentile: float) -> float:
    positive = data[np.isfinite(data) & (data > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 0.0


def extract_metadata(path: str | Path, label: float | None = None, uid: str | None = None) -> Dict[str, Any]:
    loaded = load_nifti(path, canonical=False)
    data = loaded.data
    spacing = voxel_spacing(loaded.affine)
    finite = data.ravel()
    positive = finite[finite > 0]
    percentiles = np.percentile(positive, [95.0, 99.0, 99.5]) if positive.size else np.zeros(3, dtype=float)
    shape = tuple(int(x) for x in data.shape)
    affine_det = float(np.linalg.det(loaded.affine[:3, :3]))
    row: Dict[str, Any] = {
        "uid": uid or Path(path).name[:-7],
        "label": label,
        "filepath": str(Path(path).resolve()),
        "shape_x": shape[0], "shape_y": shape[1], "shape_z": shape[2],
        "dtype": str(loaded.header.get_data_dtype()),
        "min_intensity": float(finite.min()) if finite.size else float("nan"),
        "max_intensity": float(finite.max()) if finite.size else float("nan"),
        "mean_intensity": float(finite.mean()) if finite.size else float("nan"),
        "median_nonzero": float(np.median(positive)) if positive.size else 0.0,
        "p95_nonzero": float(percentiles[0]),
        "p99_nonzero": float(percentiles[1]),
        "p99_5_nonzero": float(percentiles[2]),
        "spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2],
        "voxel_volume": float(np.prod(spacing)),
        "orientation": loaded.orientation,
        "affine_determinant": affine_det,
        "affine_sign": int(np.sign(affine_det)),
        "nonzero_fraction": float((data > 0).mean()) if data.size else 0.0,
        "physical_extent_x": float(shape[0] * spacing[0]),
        "physical_extent_y": float(shape[1] * spacing[1]),
        "physical_extent_z": float(shape[2] * spacing[2]),
        "qform_code": int(loaded.header.get("qform_code", 0)),
        "sform_code": int(loaded.header.get("sform_code", 0)),
        "xyzt_units": str(loaded.header.get_xyzt_units()),
    }
    return row
