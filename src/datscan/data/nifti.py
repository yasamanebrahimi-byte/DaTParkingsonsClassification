"""NIfTI loading and header-level inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import nibabel as nib
import numpy as np


@dataclass
class LoadedNifti:
    data: np.ndarray
    affine: np.ndarray
    header: Any
    original_shape: Tuple[int, ...]
    canonical_shape: Tuple[int, ...]
    orientation: str
    source_path: Path


def load_nifti(path: str | Path, canonical: bool = True) -> LoadedNifti:
    path = Path(path)
    image = nib.load(str(path))
    original_shape = tuple(int(x) for x in image.shape)
    if len(original_shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {original_shape}: {path}")
    source_orientation = "".join(nib.aff2axcodes(image.affine))
    if canonical:
        image = nib.as_closest_canonical(image)
    data = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    if not np.isfinite(data).all():
        raise ValueError(f"NIfTI contains non-finite values: {path}")
    return LoadedNifti(
        data=np.asarray(data, dtype=np.float32),
        affine=np.asarray(image.affine, dtype=np.float64),
        header=image.header.copy(),
        original_shape=original_shape,
        canonical_shape=tuple(int(x) for x in image.shape),
        orientation=source_orientation,
        source_path=path,
    )


def voxel_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
    values = np.sqrt((np.asarray(affine, dtype=float)[:3, :3] ** 2).sum(axis=0))
    return tuple(float(x) for x in values)


def affine_orientation(affine: np.ndarray) -> str:
    return "".join(nib.aff2axcodes(np.asarray(affine)))


def resample_to_spacing(loaded: LoadedNifti, spacing_mm: float) -> LoadedNifti:
    """Resample in physical space using the NIfTI affine and trilinear interpolation."""
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive")
    source = nib.Nifti1Image(loaded.data, loaded.affine, loaded.header)
    from nibabel.processing import resample_to_output

    resampled = resample_to_output(source, voxel_sizes=(spacing_mm,) * 3, order=1)
    data = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Resampling produced non-finite values: {loaded.source_path}")
    return LoadedNifti(
        data=data,
        affine=np.asarray(resampled.affine, dtype=np.float64),
        header=resampled.header.copy(),
        original_shape=loaded.original_shape,
        canonical_shape=tuple(int(x) for x in data.shape),
        orientation=affine_orientation(resampled.affine),
        source_path=loaded.source_path,
    )

