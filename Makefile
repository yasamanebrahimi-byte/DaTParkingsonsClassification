.PHONY: test inspect folds train calibrate package

test:
	python -m pytest

inspect:
	python scripts/inspect_data.py --archive training_data/niftis_utCGpHE.zip --labels training_data/train_labels_JNDlMjr.csv --extract-dir data/extracted --metadata artifacts/metadata/train_metadata.csv --report artifacts/reports/dataset_validation.md

folds:
	python scripts/create_folds.py --metadata artifacts/metadata/train_metadata.csv --output artifacts/folds/folds.csv --n-splits 5 --seed 20260824

train:
	python scripts/train_cv.py --config configs/baseline.yaml --metadata artifacts/metadata/train_metadata.csv --folds artifacts/folds/folds.csv --oof artifacts/metrics/resnet18_oof.csv --checkpoint-dir artifacts/checkpoints

calibrate:
	python scripts/calibrate.py --oof artifacts/metrics/resnet18_oof.csv --output artifacts/calibration/temperature.json

package:
	python scripts/package_submission.py --checkpoint-dir artifacts/checkpoints --calibration artifacts/calibration/temperature.json --output submission.zip

