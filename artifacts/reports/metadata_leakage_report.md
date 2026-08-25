# Metadata-only leakage diagnostic

This model uses acquisition and geometry metadata only; no voxel values are used.

- Samples: 1362
- Features: shape_x, shape_y, shape_z, spacing_x, spacing_y, spacing_z, voxel_volume, physical_extent_x, physical_extent_y, physical_extent_z, nonzero_fraction, p95_nonzero, p99_nonzero, p99_5_nonzero, orientation
- 5-fold log loss: 0.661858
- 5-fold AUROC: 0.630548

Interpretation: compare this diagnostic with the image model. Strong metadata performance is a shortcut-learning warning and supports domain-aware validation.
