"""Small, reproducible, OOF-log-loss-first hyperparameter tuning utilities.

The module deliberately keeps the search runner independent of any optimizer
package.  A study is just a deterministic list of resolved trial YAML files,
isolated trial directories, and CSV/JSON results that can be resumed safely.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from ..utils.config import ModelConfig, PreprocessConfig, ROIConfig, load_config, load_yaml
from ..utils.metrics import binary_metrics
from .folds import create_folds, save_folds


DEFAULT_PARAMETER_PATHS = {
    "learning_rate": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "base_channels": "model.base_channels",
    "dropout": "model.dropout",
    "augmentation_severity": "augmentation.severity",
    "batch_size": "training.batch_size",
    "gradient_accumulation_steps": "training.gradient_accumulation_steps",
    "target_spacing_mm": "preprocessing.target_spacing_mm",
    "input_shape": "preprocessing.output_shape",
    "intensity_percentile": "preprocessing.intensity_percentile",
    "clip_max": "preprocessing.clip_max",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = str(path).split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid config path: {path!r}")
    target = config
    for part in parts[:-1]:
        current = target.get(part)
        if not isinstance(current, dict):
            current = {}
            target[part] = current
        target = current
    target[parts[-1]] = copy.deepcopy(value)


def _get_path(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _search_specs(search_space: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name, raw in search_space.items():
        if isinstance(raw, (list, tuple)):
            spec = {"type": "categorical", "values": list(raw)}
        elif isinstance(raw, Mapping):
            spec = dict(raw)
        else:
            raise ValueError(f"Search-space entry {name!r} must be a list or mapping")
        spec.setdefault("path", DEFAULT_PARAMETER_PATHS.get(str(name), str(name)))
        spec.setdefault("type", "categorical")
        kind = str(spec["type"]).lower()
        if kind in {"categorical", "choice"}:
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Categorical search parameter {name!r} needs non-empty values")
        elif kind in {"uniform", "log_uniform", "int", "integer"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"Numeric search parameter {name!r} needs low and high")
            if float(spec["low"]) > float(spec["high"]):
                raise ValueError(f"Search parameter {name!r} has low > high")
        else:
            raise ValueError(f"Unsupported search-space type {spec['type']!r} for {name!r}")
        specs[str(name)] = spec
    return specs


def sample_trial_configs(search_space: Mapping[str, Any], n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Sample trial parameter dictionaries deterministically from a compact space."""
    if int(n_trials) < 0:
        raise ValueError("n_trials must be non-negative")
    specs = _search_specs(search_space)
    rng = random.Random(int(seed))
    trials: list[dict[str, Any]] = []
    for _ in range(int(n_trials)):
        trial: dict[str, Any] = {}
        for name in sorted(specs):
            spec = specs[name]
            kind = str(spec["type"]).lower()
            if kind in {"categorical", "choice"}:
                trial[name] = copy.deepcopy(spec["values"][rng.randrange(len(spec["values"]))])
            elif kind == "uniform":
                trial[name] = rng.uniform(float(spec["low"]), float(spec["high"]))
            elif kind == "log_uniform":
                trial[name] = math.exp(rng.uniform(math.log(float(spec["low"])), math.log(float(spec["high"]))))
            else:
                trial[name] = rng.randint(int(spec["low"]), int(spec["high"]))
        trials.append(trial)
    return trials


def apply_trial_parameters(base_config: Mapping[str, Any], search_space: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with validated parameter overrides applied."""
    specs = _search_specs(search_space)
    unknown = set(parameters) - set(specs)
    if unknown:
        raise ValueError(f"Trial has unknown parameters: {sorted(unknown)}")
    config = copy.deepcopy(dict(base_config))
    for name, value in parameters.items():
        spec = specs[name]
        kind = str(spec["type"]).lower()
        if kind in {"categorical", "choice"} and value not in spec["values"]:
            raise ValueError(f"Value {value!r} is outside search space for {name}")
        if kind in {"uniform", "log_uniform"} and not float(spec["low"]) <= float(value) <= float(spec["high"]):
            raise ValueError(f"Value {value!r} is outside search space for {name}")
        if kind in {"int", "integer"} and not int(spec["low"]) <= int(value) <= int(spec["high"]):
            raise ValueError(f"Value {value!r} is outside search space for {name}")
        overrides = spec.get("overrides")
        if isinstance(overrides, Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"Parameter {name} with overrides must sample a mapping")
            for value_key, path in overrides.items():
                if value_key not in value:
                    raise ValueError(f"Parameter {name} is missing paired value {value_key!r}")
                _set_path(config, str(path), value[value_key])
        else:
            path = str(spec.get("path", DEFAULT_PARAMETER_PATHS.get(name, name)))
            _set_path(config, path, value)
            if spec.get("grouped_preset"):
                # Severity presets must remain meaningful when the base YAML
                # contains explicit values from its original moderate preset.
                # Remove only the grouped severity sections; unrelated custom
                # augmentation settings remain intact.
                augmentation = config.get("augmentation")
                if isinstance(augmentation, dict):
                    for section in ("intensity_scale", "gamma", "gaussian_noise", "gaussian_blur", "resolution_degradation", "poisson_noise", "affine"):
                        augmentation.pop(section, None)
    return config


def physical_fov(config: Mapping[str, Any]) -> list[float] | None:
    spacing = _get_path(config, "preprocessing.target_spacing_mm")
    shape = _get_path(config, "preprocessing.output_shape")
    if spacing is None or not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return None
    return [round(float(spacing) * int(size), 3) for size in shape]


def score_oof_predictions(oof: pd.DataFrame) -> dict[str, Any]:
    """Score concatenated OOF rows, retaining fold metrics as diagnostics."""
    required = {"target", "probability", "fold"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF frame is missing columns: {sorted(missing)}")
    overall = binary_metrics(oof["target"], oof["probability"])
    folds = []
    for fold, group in oof.groupby("fold", sort=True):
        folds.append({"fold": int(fold), **binary_metrics(group["target"], group["probability"])})
    losses = [float(row["log_loss"]) for row in folds]
    return {
        "overall": overall,
        "folds": folds,
        "fold_log_loss_std": float(np.std(losses, ddof=0)) if losses else float("nan"),
        "worst_fold_log_loss": float(max(losses)) if losses else float("nan"),
    }


def resolve_base_config(search_config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    search_path = Path(search_config_path).resolve()
    search = load_yaml(search_path)
    base_value = search.get("base_config")
    if not base_value:
        raise ValueError("Tuning search config requires base_config")
    base_candidate = Path(str(base_value))
    candidates = [
        base_candidate,
        search_path.parent / base_candidate,
        search_path.parent.parent / base_candidate,
    ]
    base_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if base_path is None:
        raise FileNotFoundError(f"Could not resolve base_config {base_value!r} from {search_path}")
    return load_config(base_path), search, base_path


def _folded_metadata(metadata_path: str | Path, folds_path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    folds = pd.read_csv(folds_path)
    if not {"uid", "fold"}.issubset(folds.columns):
        raise ValueError("Fold file requires uid and fold columns")
    if metadata["uid"].duplicated().any() or folds["uid"].duplicated().any():
        raise ValueError("Metadata and folds must not contain duplicate UIDs")
    metadata = metadata.copy()
    folds = folds.copy()
    metadata["uid"] = metadata["uid"].astype(str)
    folds["uid"] = folds["uid"].astype(str)
    fold_columns = [column for column in ["uid", "fold", "domain_group"] if column in folds.columns]
    merged = metadata.merge(folds[fold_columns], on="uid", how="inner", validate="one_to_one")
    if len(merged) != len(metadata):
        raise ValueError("Fold file must contain exactly one assignment for every metadata UID")
    if "label" in folds:
        checked = merged[["uid", "label"]].merge(folds[["uid", "label"]], on="uid", suffixes=("_metadata", "_fold"), validate="one_to_one")
        if not np.allclose(checked["label_metadata"].to_numpy(dtype=float), checked["label_fold"].to_numpy(dtype=float)):
            raise ValueError("Fold labels do not match metadata labels")
    return merged


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(value), sort_keys=False), encoding="utf-8")


def _fold_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _result_rows(study_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(study_dir.glob("trials/trial_*/screening.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
        full_path = path.parent / "full_result.json"
        if full_path.exists():
            rows[-1].update({f"full_{key}": value for key, value in json.loads(full_path.read_text(encoding="utf-8")).items() if key not in {"trial_id", "parameters", "status"}})
        domain_path = path.parent / "domain_aware" / "result.json"
        if domain_path.exists():
            rows[-1].update({f"domain_{key}": value for key, value in json.loads(domain_path.read_text(encoding="utf-8")).items() if key not in {"trial_id", "parameters", "status"}})
    return rows


def _write_results_csv(study_dir: Path) -> None:
    rows = _result_rows(study_dir)
    output = study_dir / "results.csv"
    if not rows:
        pd.DataFrame().to_csv(output, index=False)
        return
    frame = pd.DataFrame(rows)
    if "mean_log_loss" in frame:
        sort_loss = pd.to_numeric(frame["mean_log_loss"], errors="coerce")
        if "full_oof_log_loss" in frame:
            sort_loss = pd.to_numeric(frame["full_oof_log_loss"], errors="coerce").fillna(sort_loss)
        frame["_sort_loss"] = sort_loss
        frame = frame.sort_values(["status", "_sort_loss", "trial_id"], ascending=[True, True, True], na_position="last").drop(columns=["_sort_loss"])
    frame.to_csv(output, index=False)


def _candidate_result(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    stage: str,
    training_seed: int,
    trial_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess = PreprocessConfig.from_mapping(config.get("preprocessing"))
    model = ModelConfig.from_mapping(config.get("model"))
    data_view = str(config.get("data_view", "roi" if model.name.lower().startswith("roi") else "global"))
    roi_config = ROIConfig.from_mapping(config.get("roi")) if data_view == "roi" else None
    training = dict(config.get("training", {}))
    cache_dir = _get_path(config, "preprocessing.cache_dir")
    fold_seed = int(config.get("fold_seed", config.get("seed", training_seed)))
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    max_memory = None
    start = time.perf_counter()
    # Keep sampling, persistence, and report utilities usable in lightweight
    # environments that do not have the optional PyTorch training runtime.
    from . import train as train_module

    for fold in sorted(int(value) for value in metadata["fold"].unique()):
        out, metrics = train_module.train_one_fold(
            metadata,
            fold,
            preprocess,
            model,
            training,
            output_dir / "checkpoints",
            cache_dir=cache_dir,
            data_view=data_view,
            roi_config=roi_config,
            checkpoint_prefix=f"{trial_id}_{stage}",
            augmentation_config=config.get("augmentation"),
            seed=training_seed,
            fold_seed=fold_seed,
            training_seed=training_seed,
            experiment_name=f"{trial_id}_{stage}",
        )
        predictions.append(out)
        fold_metrics.append({"fold": fold, **binary_metrics(out["target"], out["probability"])})
        if metrics.get("max_memory_allocated_mb") is not None:
            max_memory = max(float(max_memory or 0.0), float(metrics["max_memory_allocated_mb"]))
    oof = pd.concat(predictions, ignore_index=True).sort_values("uid").reset_index(drop=True)
    if len(oof) != len(metadata) or oof["uid"].duplicated().any():
        raise RuntimeError("Tuning OOF output must contain every UID exactly once")
    evaluation = score_oof_predictions(oof)
    overall = evaluation["overall"]
    fold_losses = [float(row["log_loss"]) for row in evaluation["folds"]]
    result = {
        "trial_id": trial_id,
        "stage": stage,
        "status": "completed",
        "oof_log_loss": float(overall["log_loss"]),
        "mean_log_loss": float(overall["log_loss"]),
        "std_log_loss": float(evaluation["fold_log_loss_std"]),
        "worst_fold_log_loss": float(evaluation["worst_fold_log_loss"]),
        "auroc": float(overall["auroc"]),
        "brier": float(overall["brier_score"]),
        "runtime_seconds": float(time.perf_counter() - start),
        "max_memory_allocated_mb": max_memory,
        "training_seed": int(training_seed),
        "folds": fold_metrics,
    }
    for row in fold_metrics:
        result[f"fold_{row['fold']}_log_loss"] = float(row["log_loss"])
    oof.to_csv(output_dir / "oof.csv", index=False)
    return result, oof


def _failed_result(trial_id: str, parameters: Mapping[str, Any], started: float, error: Exception, stage: str) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "parameters": _jsonable(parameters),
        "stage": stage,
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _parameter_columns(parameters: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    result = {str(key): _jsonable(value) for key, value in parameters.items()}
    result["spacing"] = _get_path(config, "preprocessing.target_spacing_mm")
    result["input_shape"] = "x".join(str(value) for value in (_get_path(config, "preprocessing.output_shape") or []))
    result["physical_fov_mm"] = "x".join(str(value) for value in (physical_fov(config) or []))
    physical_batch = int(_get_path(config, "training.batch_size", 1))
    accumulation = int(_get_path(config, "training.gradient_accumulation_steps", 1))
    result["physical_batch_size"] = physical_batch
    result["effective_batch_size"] = physical_batch * accumulation
    return result


def _study_metadata(study_dir: Path, search_path: Path, base_path: Path, search: Mapping[str, Any], metadata_path: Path, folds_path: Path, seed: int, n_trials: int) -> None:
    payload = {
        "study": str(study_dir.name),
        "search_config": str(search_path.resolve()),
        "base_config": str(base_path.resolve()),
        "base_experiment": search.get("base_experiment", {"config": str(base_path)}),
        "metadata": str(metadata_path.resolve()),
        "screening_folds": str(folds_path.resolve()),
        "screening_fold_fingerprint": _fold_fingerprint(folds_path),
        "tuning_seed": int(seed),
        "n_trials": int(n_trials),
        "screening": _jsonable(search.get("screening", {})),
        "promotion": _jsonable(search.get("promotion", {})),
        "git_commit": _git_commit(),
    }
    (study_dir / "study.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    _write_yaml(study_dir / "search_config.yaml", search)


def run_screening_study(
    search_config_path: str | Path,
    metadata_path: str | Path,
    folds_path: str | Path,
    output_dir: str | Path,
    *,
    n_trials: int | None = None,
    seed: int | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run or resume Stage A using one fixed screening fold file."""
    search_path = Path(search_config_path).resolve()
    base_config, search, base_path = resolve_base_config(search_path)
    study_dir = Path(output_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    screening = dict(search.get("screening", {}))
    trial_count = int(n_trials if n_trials is not None else search.get("n_trials", 16))
    tuning_seed = int(seed if seed is not None else search.get("seed", base_config.get("seed", 20260824)))
    metadata_file = Path(metadata_path).resolve()
    folds_file = Path(folds_path).resolve()
    if not folds_file.exists():
        folds_file.parent.mkdir(parents=True, exist_ok=True)
        generated = create_folds(pd.read_csv(metadata_file), n_splits=int(screening.get("folds", 3)), seed=tuning_seed)
        save_folds(generated, folds_file)
    _study_metadata(study_dir, search_path, base_path, search, metadata_file, folds_file, tuning_seed, trial_count)
    metadata = _folded_metadata(metadata_file, folds_file)
    trials = sample_trial_configs(search.get("search_space", {}), trial_count, tuning_seed)
    base_training_seed = int(base_config.get("training_seed", base_config.get("seed", tuning_seed)))
    for index, parameters in enumerate(trials):
        trial_id = f"trial_{index:03d}"
        trial_dir = study_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = apply_trial_parameters(base_config, search.get("search_space", {}), parameters)
        config.setdefault("training", {})["epochs"] = int(screening.get("epochs", config["training"].get("epochs", 25)))
        config["training"]["patience"] = int(screening.get("patience", config["training"].get("patience", 6)))
        config["training"]["folds"] = int(screening.get("folds", 3))
        config["_tuning"] = {"study": study_dir.name, "trial_id": trial_id, "stage": "screening", "tuning_seed": tuning_seed, "parameters": _jsonable(parameters), "screening_folds": str(folds_file)}
        _write_yaml(trial_dir / "config.yaml", config)
        existing_path = trial_dir / "screening.json"
        if resume and existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                if existing.get("status") in {"completed", "failed"}:
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        started = time.perf_counter()
        try:
            result, _ = _candidate_result(metadata, config, trial_dir / "screening", stage="screening", training_seed=base_training_seed, trial_id=trial_id)
            result.update(_parameter_columns(parameters, config))
            result["parameters"] = _jsonable(parameters)
        except Exception as error:  # one bad trial must not stop the study
            result = _failed_result(trial_id, parameters, started, error, "screening")
            result.update(_parameter_columns(parameters, config))
        existing_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=True), encoding="utf-8")
        _write_results_csv(study_dir)
    _write_results_csv(study_dir)
    return _result_rows(study_dir)


def _successful_screening(study_dir: Path) -> list[dict[str, Any]]:
    rows = [row for row in _result_rows(study_dir) if row.get("status") == "completed"]
    return sorted(rows, key=lambda row: (float(row.get("mean_log_loss", float("inf"))), str(row.get("trial_id"))))


def promote_trials(
    study_dir: str | Path,
    full_folds_path: str | Path,
    *,
    promote_top: int = 4,
    full_epochs: int | None = None,
    full_patience: int | None = None,
    confirm_seed: int | None = None,
    confirm_top: int = 2,
    domain_folds_path: str | Path | None = None,
    domain_top: int = 1,
) -> list[dict[str, Any]]:
    """Run Stage B on the deterministic top screening rows."""
    study = Path(study_dir).resolve()
    metadata_payload = json.loads((study / "study.json").read_text(encoding="utf-8"))
    metadata = _folded_metadata(metadata_payload["metadata"], full_folds_path)
    selected = _successful_screening(study)[: max(int(promote_top), 0)]
    for rank, screening_row in enumerate(selected, start=1):
        trial_id = str(screening_row["trial_id"])
        trial_dir = study / "trials" / trial_id
        config = load_yaml(trial_dir / "config.yaml")
        config.setdefault("training", {})["folds"] = int(metadata["fold"].nunique())
        if full_epochs is not None:
            config["training"]["epochs"] = int(full_epochs)
        if full_patience is not None:
            config["training"]["patience"] = int(full_patience)
        config.setdefault("_tuning", {}).update({"stage": "full", "promotion_rank": rank, "full_folds": str(Path(full_folds_path).resolve())})
        _write_yaml(trial_dir / "full_config.yaml", config)
        full_path = trial_dir / "full_result.json"
        if not full_path.exists():
            try:
                result, _ = _candidate_result(
                    metadata,
                    config,
                    trial_dir / "full",
                    stage="full",
                    training_seed=int(config.get("training_seed", config.get("seed", metadata_payload["tuning_seed"]))),
                    trial_id=trial_id,
                )
            except Exception as error:
                result = _failed_result(trial_id, screening_row.get("parameters", {}), time.perf_counter(), error, "full")
            result["promotion_rank"] = rank
            full_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=True), encoding="utf-8")
        if confirm_seed is not None and rank <= int(confirm_top) and json.loads(full_path.read_text(encoding="utf-8")).get("status") == "completed":
            confirm_dir = trial_dir / f"confirm_seed_{int(confirm_seed)}"
            confirm_result_path = confirm_dir / "result.json"
            if not confirm_result_path.exists():
                try:
                    result, _ = _candidate_result(metadata, config, confirm_dir, stage=f"confirm_seed_{int(confirm_seed)}", training_seed=int(confirm_seed), trial_id=trial_id)
                except Exception as error:
                    result = _failed_result(trial_id, screening_row.get("parameters", {}), time.perf_counter(), error, f"confirm_seed_{int(confirm_seed)}")
                confirm_result_path.parent.mkdir(parents=True, exist_ok=True)
                confirm_result_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=True), encoding="utf-8")
        if domain_folds_path is not None and rank <= int(domain_top):
            domain_dir = trial_dir / "domain_aware"
            domain_result_path = domain_dir / "result.json"
            if not domain_result_path.exists():
                domain_metadata = _folded_metadata(metadata_payload["metadata"], domain_folds_path)
                try:
                    result, _ = _candidate_result(
                        domain_metadata,
                        config,
                        domain_dir,
                        stage="domain_aware",
                        training_seed=int(config.get("training_seed", config.get("seed", metadata_payload["tuning_seed"]))),
                        trial_id=trial_id,
                    )
                except Exception as error:
                    result = _failed_result(trial_id, screening_row.get("parameters", {}), time.perf_counter(), error, "domain_aware")
                domain_result_path.parent.mkdir(parents=True, exist_ok=True)
                domain_result_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=True), encoding="utf-8")
    _write_results_csv(study)
    return _result_rows(study)


def hyperparameter_importance(study_dir: str | Path) -> dict[str, Any]:
    rows = _successful_screening(Path(study_dir).resolve())
    if not rows:
        return {}
    result: dict[str, Any] = {}
    for name in sorted(rows[0].get("parameters", {})):
        values = [row.get("parameters", {}).get(name) for row in rows]
        scores = np.asarray([float(row["mean_log_loss"]) for row in rows], dtype=float)
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[name] = {"kind": "numeric", "correlation_with_log_loss": float(np.corrcoef(np.asarray(values, dtype=float), scores)[0, 1]) if len(set(values)) > 1 and len(values) > 1 else None}
        else:
            grouped = {}
            for value in sorted({str(item) for item in values}):
                grouped[value] = float(np.mean([score for item, score in zip(values, scores) if str(item) == value]))
            result[name] = {"kind": "categorical", "mean_log_loss_by_value": grouped}
    (Path(study_dir).resolve() / "hyperparameter_importance.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    return result


def write_tuning_report(study_dir: str | Path, output: str | Path | None = None, baseline_oof: str | Path | None = None) -> Path:
    study = Path(study_dir).resolve()
    output_path = Path(output).resolve() if output else study / "hyperparameter_tuning.md"
    metadata = json.loads((study / "study.json").read_text(encoding="utf-8"))
    rows = _result_rows(study)
    successful = sorted([row for row in rows if row.get("status") == "completed"], key=lambda row: (float(row.get("mean_log_loss", float("inf"))), str(row.get("trial_id"))))
    full = sorted([row for row in successful if row.get("full_oof_log_loss") is not None], key=lambda row: (float(row.get("full_oof_log_loss")), str(row.get("trial_id"))))
    domain = sorted([row for row in successful if row.get("domain_oof_log_loss") is not None], key=lambda row: (float(row.get("domain_oof_log_loss")), str(row.get("trial_id"))))
    importance = hyperparameter_importance(study)
    lines = [
        "# Targeted hyperparameter tuning",
        "",
        f"- Base config: `{metadata['base_config']}`",
        f"- Tuning seed: `{metadata['tuning_seed']}`",
        f"- Screening folds: `{metadata['screening_folds']}`",
        f"- Screening fold fingerprint: `{metadata['screening_fold_fingerprint']}`",
        f"- Trials requested: `{metadata['n_trials']}`; completed: `{sum(row.get('status') == 'completed' for row in rows)}`; failed: `{sum(row.get('status') == 'failed' for row in rows)}`",
        "",
        "All search decisions use raw OOF probabilities and concatenated OOF log loss. Leaderboard and test predictions are not used.",
        "",
        "## Screening trials",
        "",
        "| Rank | Trial | Screening LL | LL Std | Worst Fold LL | AUROC | Runtime (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(successful, start=1):
        lines.append(f"| {rank} | {row['trial_id']} | {float(row['mean_log_loss']):.6f} | {float(row['std_log_loss']):.6f} | {float(row['worst_fold_log_loss']):.6f} | {float(row['auroc']):.6f} | {float(row['runtime_seconds']):.1f} |")
    if not successful:
        lines.append("| — | no completed trials | — | — | — | — | — |")
    lines.extend(["", "## Full canonical CV finalists", "", "| Rank | Trial | OOF Log Loss | AUROC | Brier | Fold LL Std |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for rank, row in enumerate(full, start=1):
        lines.append(f"| {rank} | {row['trial_id']} | {float(row['full_oof_log_loss']):.6f} | {float(row['full_auroc']):.6f} | {float(row['full_brier']):.6f} | {float(row['full_std_log_loss']):.6f} |")
    if not full:
        lines.append("| — | promotion not run | — | — | — | — |")
    lines.extend(["", "## Optional domain-aware robustness check", "", "| Rank | Trial | Domain-aware OOF Log Loss | AUROC | Brier | Fold LL Std |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for rank, row in enumerate(domain, start=1):
        lines.append(f"| {rank} | {row['trial_id']} | {float(row['domain_oof_log_loss']):.6f} | {float(row['domain_auroc']):.6f} | {float(row['domain_brier']):.6f} | {float(row['domain_std_log_loss']):.6f} |")
    if not domain:
        lines.append("| — | not requested | — | — | — | — |")
    lines.extend(["", "## Baseline comparison", ""])
    if baseline_oof and Path(baseline_oof).exists():
        baseline = binary_metrics(*(lambda frame: (frame["target"], frame["probability"]))(pd.read_csv(baseline_oof)))
        lines.append(f"Baseline OOF log loss: `{baseline['log_loss']:.6f}`. Full finalist deltas are reported only when full CV exists.")
        if full:
            delta = baseline["log_loss"] - float(full[0]["full_oof_log_loss"])
            relative = delta / baseline["log_loss"]
            lines.append(f"Best full finalist absolute improvement: `{delta:.6f}`; relative improvement: `{relative:.2%}`.")
    else:
        lines.append("No baseline CNN OOF file was available in the repository artifacts; no improvement is claimed until the base and finalists are evaluated on the same canonical folds.")
    lines.extend(["", "## Hyperparameter observations", "", "The following diagnostics are descriptive only and are not a surrogate model:", ""])
    if importance:
        for name, values in importance.items():
            lines.append(f"- `{name}`: {json.dumps(values, sort_keys=True)}")
    else:
        lines.append("- Pending completed screening trials.")
    lines.extend(["", "## Reproducibility", "", f"- Git commit: `{metadata.get('git_commit') or 'not available'}`", "- Each resolved config is saved under `trials/trial_NNN/config.yaml`; checkpoints are isolated under that trial.", "- Confirmation runs, when requested, are saved under `confirm_seed_<seed>/`.", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
