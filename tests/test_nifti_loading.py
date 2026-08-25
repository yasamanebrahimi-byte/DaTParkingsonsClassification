import numpy as np
import nibabel as nib

from datscan.data.nifti import load_nifti, resample_to_spacing, voxel_spacing


def test_canonical_loading_and_resampling(tmp_path):
    data = np.zeros((8, 10, 12), dtype=np.uint16)
    data[2:6, 3:7, 4:8] = 10
    affine = np.diag([-2.0, 2.0, 2.5, 1.0])
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    loaded = load_nifti(path)
    assert loaded.data.dtype == np.float32
    assert loaded.data.ndim == 3
    resampled = resample_to_spacing(loaded, 3.0)
    assert np.allclose(voxel_spacing(resampled.affine), (3.0, 3.0, 3.0), atol=1e-5)
    assert resampled.data.ndim == 3

