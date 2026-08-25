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

OOF rows contain `uid`, `fold`, `target`, `logit`, and `probability`. Model selection must use overall OOF log loss; AUROC, Brier score, sensitivity, specificity, and calibration diagnostics are secondary diagnostics.

## Calibration

Fit temperature scaling using OOF logits only:

```powershell
python scripts/calibrate.py `
  --oof artifacts/metrics/resnet18_oof.csv `
  --output artifacts/calibration/temperature.json
```

Calibration is applied to logits before the final numerical epsilon clamp. No public leaderboard score is used to fit calibration or ensemble weights.

## Packaging and local submission simulation

After training, create a root-correct `submission.zip`:

```powershell
python scripts/package_submission.py `
  --checkpoint-dir artifacts/checkpoints `
  --calibration artifacts/calibration/temperature.json `
  --output submission.zip
```

The package contains `main.py` at its archive root, `datscan_inference/`, and `assets/`. `main.py` reads `/code_execution/data/submission_format.csv`, finds each UID under `/code_execution/data/niftis/`, preserves template ordering, and writes `/code_execution/submission.csv`.

Before packaging a trained solution, run the source tests:

```powershell
python -m pytest
```

A competition-like simulation can be run after a package and test data are available by extracting the ZIP directly into `mock_runtime/`, placing a read-only-style `data/niftis/` and `data/submission_format.csv` beside it, and invoking `python mock_runtime/main.py`. The same `datscan_inference` code is used by the packaged entry point; no training repository import is required.

## Reproducibility and safety

Seeds are set for Python, NumPy, PyTorch, CUDA, and fold generation. Preprocessing is configuration-driven and is duplicated in the submission only as a minimal dependency-free copy; synthetic tests cover orientation, physical resampling, normalization, crop/pad, model output, and submission validation. Corrupt images and missing assets raise explicit errors rather than silently emitting 0.5.

Registration, ROI networks, quantitative feature models, TTA, and learned ensemble weights remain extension points. They should be adopted only after controlled OOF log-loss evidence and domain-robustness diagnostics justify them.
