# DaT-SPECT Classifier

This repository implements a reproducible, probability-first binary classifier for dopamine-transporter SPECT volumes. `is_pathologic=0` means normal and `1` means abnormal. Binary log loss is the primary objective, so calibration and out-of-fold evaluation are part of the baseline rather than afterthoughts.

## Status

The engineering baseline is implemented and wired for the supplied training archive. The first data audit is generated with the commands below. Neural checkpoints and `submission.zip` are intentionally produced only by the training and packaging commands; they are not fabricated before the real labels and images are validated.

## Repository structure

- `src/datscan/data`: archive discovery, NIfTI loading, metadata, validation, physical-space preprocessing, and datasets.
- `src/datscan/models`: compact GroupNorm 3D ResNet-18-style network and extension points for ROI/features/ensembles.
- `src/datscan/training`: folds, CV training, metrics, temperature scaling, and ensemble weights.
- `submission`: minimal offline package copied into the final ZIP.
- `configs`: versioned experiment settings.
- `artifacts`: generated metadata, folds, checkpoints, reports, and OOF predictions.
- `tests`: synthetic NIfTI and submission-contract tests.

## High-resolution ResNet experiment

`configs/highres_resnet.yaml` adds a controlled `resnet3d_highres` experiment while leaving the baseline configuration unchanged. Both variants use the same GroupNorm residual blocks and training loop; the high-resolution model changes only the input resolution and early downsampling strategy.

| Property | Current baseline | High-resolution model |
| --- | --- | --- |
| Input spacing | 3.0 mm | 2.5 mm |
| Input shape | 96³ | 112³ |
| Stem stride | 2 | 1 |
| Initial max pool | Yes, stride 2 | No |
| Total downsampling | 32× | 8× |
| Final feature map | 3³ | 14³ |

The `feature_map_shape()` model diagnostic and tests verify these dimensions without printing tensor contents.

## Scanner-robust augmentation experiment

The historical `MildVolumeAugmentation` remains the default for the baseline,
high-resolution, and ROI experiments. Training configs can now select
`augmentation.name: none`, `mild`, or `scanner_robust` without Python changes.
The scanner-robust transform is applied only after deterministic preprocessing
and cache retrieval; validation, OOF prediction, calibration, and packaged
inference remain deterministic.

The first controlled experiment is `configs/highres_scanner_aug.yaml`. It keeps
the high-resolution architecture, preprocessing, optimizer, folds, seed, and
training schedule fixed while adding moderate blur, resolution, noise,
intensity, and conservative affine perturbations. The companion configs
`highres_blur_noise.yaml` and `highres_blur_noise_resolution.yaml` provide
ablation points without code edits. `scripts/inspect_augmentations.py` saves
orthogonal slice montages and quantitative sanity statistics for explicitly
selected training UIDs.

## Striatum-focused ROI experiment

`configs/roi_resnet.yaml` adds an independent `roi_resnet3d` experiment. It
uses the same canonical orientation, 2.5 mm isotropic resampling, intensity
normalization, folds, and mild augmentation as the high-resolution global
experiment, but the model receives only a deterministic bilateral ROI view.

| Property | Global high-resolution | Striatal ROI |
| --- | --- | --- |
| Input shape | 112 x 112 x 112 | 64 x 64 x 48 |
| Physical FOV | 280 x 280 x 280 mm | 160 x 160 x 120 mm |
| Stem stride | 1 | 1 |
| Initial max pool | No | No |
| Total downsampling | 8x | 8x |
| Final feature map | 14 x 14 x 14 | 8 x 8 x 6 |

After canonical RAS orientation, spatial axis 0 is the left/right axis. The
ROI center is computed from low-threshold foreground support geometry after
resampling, then bounded around the resampled volume center. This keeps both
hemispheres in the crop and prevents a unilateral high-uptake side from
pulling the crop away from the weaker side. It never uses the brightest voxel,
annotations, or test-set information.

Run `scripts/inspect_roi.py` for a compact coverage report. Supplying
`--visualize-dir` additionally saves whole-volume and ROI orthogonal middle
slices for explicitly selected local UIDs.

## Environment setup

Use Python 3.12 in the competition environment. A CPU environment is sufficient for tests and metadata validation; a CUDA PyTorch build is recommended for training.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The inference package has no network step and loads only local assets. It uses the competition runtime's PyTorch, NumPy, Pandas, SciPy, and NiBabel installations.

## Data setup and validation

Place the supplied files in `training_data/` (the current workspace already contains them):

```text
training_data/training_data.zip
training_data/train_labels.csv
```

The filenames may differ; pass the actual paths. The archive is extracted with a path-traversal check and scanned recursively, so its internal folder name is not assumed.

```powershell
python scripts/inspect_data.py `
  --archive training_data/niftis_utCGpHE.zip `
  --labels training_data/train_labels_JNDlMjr.csv `
  --extract-dir data/extracted `
  --metadata artifacts/metadata/train_metadata.csv `
  --report artifacts/reports/dataset_validation.md
```

The report checks missing/extra/duplicate UIDs, invalid labels, unreadable images, finite values, affine and spacing metadata, intensity summaries, orientation, and physical extents. It does not expose patient-identifying data beyond the provided UID.

## Folds and training

Create the canonical stratified five-fold split once:

```powershell
python scripts/create_folds.py `
  --metadata artifacts/metadata/train_metadata.csv `
  --output artifacts/folds/folds.csv `
  --n-splits 5 `
  --seed 20260824
```

Train the baseline 3D ResNet. The pipeline uses canonical orientation, affine-aware 3.0 mm isotropic resampling, per-scan p99.5 normalization, foreground-centered 96³ crop/pad, mild augmentation, GroupNorm, AdamW, BCE-with-logits, early stopping on validation log loss, and saved fold checkpoints.

```powershell
python scripts/train_cv.py `
  --config configs/baseline.yaml `
  --metadata artifacts/metadata/train_metadata.csv `
  --folds artifacts/folds/folds.csv `
  --oof artifacts/metrics/resnet18_oof.csv `
  --checkpoint-dir artifacts/checkpoints
```

Train the scanner-robust high-resolution experiment with the same saved folds:

```powershell
python scripts/train_cv.py `
  --config configs/highres_scanner_aug.yaml `
  --metadata artifacts/metadata/train_metadata.csv `
  --folds artifacts/folds/folds.csv `
  --oof artifacts/metrics/highres_scanner_aug_oof.csv `
  --checkpoint-dir artifacts/checkpoints_highres_scanner_aug
```

The matched high-resolution mild-augmentation control is:

```powershell
python scripts/train_cv.py `
  --config configs/highres_resnet.yaml `
  --metadata artifacts/metadata/train_metadata.csv `
  --folds artifacts/folds/folds.csv `
  --oof artifacts/metrics/highres_resnet_oof.csv `
  --checkpoint-dir artifacts/checkpoints_highres
```

Compare the control and scanner-robust OOF predictions, including fold
stability and per-sample high-confidence errors:

```powershell
python scripts/compare_augmentation_oof.py `
  --mild-oof artifacts/metrics/highres_resnet_oof.csv `
  --scanner-oof artifacts/metrics/highres_scanner_aug_oof.csv `
  --output artifacts/metrics/highres_mild_vs_scanner_aug_errors.csv
```

OOF rows contain `uid`, `fold`, `target`, `logit`, and `probability`. Model selection must use overall OOF log loss; AUROC, Brier score, sensitivity, specificity, and calibration diagnostics are secondary diagnostics.

## Calibration

Fit temperature scaling using OOF logits only:

```powershell
python scripts/calibrate.py `
  --oof artifacts/metrics/resnet18_oof.csv `
  --output artifacts/calibration/temperature.json
```

Calibration is applied to logits before the final numerical epsilon clamp. No public leaderboard score is used to fit calibration or ensemble weights.

Calibrate the scanner-robust OOF logits with the existing OOF-only procedure:

```powershell
python scripts/calibrate.py `
  --oof artifacts/metrics/highres_scanner_aug_oof.csv `
  --output artifacts/calibration/highres_scanner_aug_temperature.json
```

For the global + ROI experiment, first build the OOF grid-search manifest.
The command writes both the JSON manifest and a companion ensemble OOF CSV;
the selected weight is based only on the concatenated OOF rows.

```powershell
python scripts/build_ensemble.py `
  --global-oof artifacts/metrics/highres_resnet_oof.csv `
  --roi-oof artifacts/metrics/roi_resnet_oof.csv `
  --output artifacts/ensemble/global_roi.json
```

The manifest reports global-only, ROI-only, 50/50, and optimized probability
ensembles, AUROC, Brier score, OOF log loss, correlations, classification
disagreement, and disagreement losses. Calibrate the final ensemble after
this step:

```powershell
python scripts/calibrate.py `
  --oof artifacts/ensemble/global_roi_oof.csv `
  --probability-column ensemble_probability `
  --output artifacts/calibration/global_roi_temperature.json
```

Calibration is applied after the probability ensemble only when its OOF log
loss is lower than the uncalibrated ensemble.

## Packaging and local submission simulation

After training, create a root-correct `submission.zip`:

```powershell
python scripts/package_submission.py `
  --checkpoint-dir artifacts/checkpoints `
  --calibration artifacts/calibration/temperature.json `
  --output submission.zip
```

Package the scanner-robust global checkpoints using the normal deterministic
inference package:

```powershell
python scripts/package_submission.py `
  --checkpoint-dir artifacts/checkpoints_highres_scanner_aug `
  --calibration artifacts/calibration/highres_scanner_aug_temperature.json `
  --global-config configs/highres_scanner_aug.yaml `
  --output submission_highres_scanner_aug.zip
```

For the global + ROI package, use the two checkpoint families and the
OOF-derived manifest:

```powershell
python scripts/package_submission.py `
  --global-checkpoint-dir artifacts/checkpoints_highres `
  --roi-checkpoint-dir artifacts/checkpoints_roi `
  --ensemble artifacts/ensemble/global_roi.json `
  --calibration artifacts/calibration/global_roi_temperature.json `
  --global-config configs/highres_resnet.yaml `
  --roi-config configs/roi_resnet.yaml `
  --output submission_global_roi.zip
```

At inference, each NIfTI is loaded and resampled once, both views are derived
from that normalized base volume, fold logits are averaged within each model
family, OOF-derived weights are applied to probabilities, and the saved
post-ensemble calibration is applied. No network access or test-data weight
re-estimation is required.

The package contains `main.py` at its archive root, `datscan_inference/`, and `assets/`. `main.py` reads `/code_execution/data/submission_format.csv`, finds each UID under `/code_execution/data/niftis/`, preserves template ordering, and writes `/code_execution/submission.csv`.

Before packaging a trained solution, run the source tests:

```powershell
python -m pytest
```

A competition-like simulation can be run after a package and test data are available by extracting the ZIP directly into `mock_runtime/`, placing a read-only-style `data/niftis/` and `data/submission_format.csv` beside it, and invoking `python mock_runtime/main.py`. The same `datscan_inference` code is used by the packaged entry point; no training repository import is required.

## Reproducibility and safety

Seeds are set for Python, NumPy, PyTorch, CUDA, and fold generation. Preprocessing is configuration-driven and is duplicated in the submission only as a minimal dependency-free copy; synthetic tests cover orientation, physical resampling, normalization, crop/pad, model output, and submission validation. Corrupt images and missing assets raise explicit errors rather than silently emitting 0.5.

Registration, quantitative feature models, TTA, and other priorities remain
out of scope. The ROI model and learned global/ROI weight are controlled
experiments and should be retained only when OOF log-loss and stability
evidence justify them.
