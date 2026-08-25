"""Fold-safe classical models for deterministic striatal features."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.striatal_features import feature_family, select_feature_columns, validate_feature_frame
from ..models.feature_model import FeatureMLP
from ..utils.metrics import binary_metrics, safe_probabilities


class FeatureMLPClassifier:
    """Small optional torch estimator with an sklearn-like interface.

    The scaler is supplied by the surrounding Pipeline, so its parameters are
    fitted independently inside each outer training fold.
    """

    def __init__(self, hidden: int = 64, epochs: int = 100, learning_rate: float = 1e-3, weight_decay: float = 1e-3, random_state: int = 20260824):
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("FeatureMLPClassifier expects a 2-D X aligned with y")
        torch.manual_seed(self.random_state)
        model = FeatureMLP(X.shape[1], self.hidden)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        validation_indices = np.arange(len(X))
        if len(X) >= 20 and np.unique(y).size == 2:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
            train_indices, validation_indices = next(splitter.split(X, y.astype(int)))
        else:
            train_indices = validation_indices
        train_x = torch.from_numpy(X[train_indices])
        train_y = torch.from_numpy(y[train_indices])
        valid_x = torch.from_numpy(X[validation_indices])
        valid_y = torch.from_numpy(y[validation_indices])
        best_state = None
        best_loss = float("inf")
        stale = 0
        for _ in range(max(self.epochs, 1)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                validation_probability = expit(model(valid_x).detach().numpy())
            validation_loss = binary_metrics(valid_y.numpy(), validation_probability)["log_loss"]
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 15:
                    break
        if best_state is None:
            best_state = model.state_dict()
        self.input_features_ = int(X.shape[1])
        self.model_ = FeatureMLP(self.input_features_, self.hidden)
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_validation_log_loss_ = float(best_loss)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("FeatureMLPClassifier must be fitted before prediction")
        values = np.asarray(X, dtype=np.float32)
        with torch.inference_mode():
            probability = expit(self.model_(torch.from_numpy(values)).numpy())
        probability = safe_probabilities(probability)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_feature_estimator(model: str, C: float = 1.0, random_state: int = 20260824, max_iter: int = 2000):
    normalized = str(model).lower().replace("-", "_")
    if normalized in {"logistic", "logistic_regression", "lr"}:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=float(C), penalty="l2", solver="lbfgs", max_iter=int(max_iter), random_state=int(random_state)),
                ),
            ]
        )
    if normalized in {"histgb", "hist_gradient_boosting", "nonlinear"}:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=int(random_state),
        )
    if normalized in {"feature_mlp", "mlp"}:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", FeatureMLPClassifier(random_state=int(random_state))),
            ]
        )
    raise ValueError(f"Unknown feature model: {model}; use logistic, histgb, or feature_mlp")


def _drop_redundant_features(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], list[str]]:
    retained: list[str] = []
    removed: list[str] = []
    signatures: dict[tuple[float, ...], str] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {column} contains NaN or infinity")
        if np.std(values, ddof=0) <= 1.0e-12 or np.unique(values).size <= 1:
            removed.append(f"{column}: zero/near-zero variance")
            continue
        signature = tuple(np.round(values, 12))
        if signature in signatures:
            removed.append(f"{column}: duplicate of {signatures[signature]}")
            continue
        signatures[signature] = column
        retained.append(column)
    if not retained:
        raise ValueError("No non-constant quantitative features remain")
    return retained, removed


def _fit_outer_estimator(estimator, model: str, X: np.ndarray, y: np.ndarray, tune_c: bool, random_state: int):
    if tune_c and str(model).lower().replace("-", "_") in {"logistic", "logistic_regression", "lr"}:
        inner_splits = min(4, int(np.bincount(y.astype(int)).min()))
        if inner_splits >= 2:
            search = GridSearchCV(
                estimator,
                {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
                scoring="neg_log_loss",
                cv=StratifiedKFold(inner_splits, shuffle=True, random_state=int(random_state)),
                refit=True,
                n_jobs=1,
            )
            search.fit(X, y)
            return search, getattr(search, "best_params_", {})
    estimator.fit(X, y)
    return estimator, {}


def train_feature_cv(
    features: pd.DataFrame,
    folds: pd.DataFrame,
    model: str,
    output_path: str | Path,
    output_dir: str | Path,
    feature_families: Sequence[str] | None = None,
    C: float = 1.0,
    tune_c: bool = False,
    random_state: int = 20260824,
) -> tuple[pd.DataFrame, dict]:
    """Train one selected feature model with the canonical outer folds."""
    if "uid" not in features or "label" not in features:
        raise ValueError("Feature CSV requires uid and label columns")
    if features["uid"].duplicated().any() or folds["uid"].duplicated().any():
        raise ValueError("Feature and fold tables must not contain duplicate UIDs")
    if not {"uid", "fold"}.issubset(folds.columns):
        raise ValueError("Fold CSV requires uid and fold columns")
    left = features.copy()
    left["uid"] = left["uid"].astype(str)
    right = folds[["uid", "fold"]].copy()
    right["uid"] = right["uid"].astype(str)
    if set(left["uid"]) != set(right["uid"]):
        raise ValueError("Feature and fold CSVs must contain exactly the same UIDs")
    if "label" in folds:
        labels = folds[["uid", "label"]].copy()
        labels["uid"] = labels["uid"].astype(str)
        checked = left[["uid", "label"]].merge(labels, on="uid", suffixes=("_features", "_fold"), validate="one_to_one")
        if not np.allclose(checked["label_features"].to_numpy(dtype=float), checked["label_fold"].to_numpy(dtype=float)):
            raise ValueError("Feature and fold labels do not match")
    frame = left.merge(right, on="uid", how="inner", validate="one_to_one").sort_values("uid").reset_index(drop=True)
    all_columns = [column for column in frame.columns if column not in {"uid", "label", "fold", "target"}]
    all_columns = select_feature_columns(all_columns, feature_families)
    stats, issues = validate_feature_frame(frame, all_columns)
    invalid = [issue for issue in issues if "NaN" in issue or "infinite" in issue]
    if invalid:
        raise ValueError("Feature validation failed: " + "; ".join(invalid))
    feature_columns, removed = _drop_redundant_features(frame, all_columns)
    X = frame[feature_columns].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=float)
    folds_used = sorted(int(value) for value in frame["fold"].unique())
    predictions = []
    fold_metrics = []
    model_directory = Path(output_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    for fold in folds_used:
        train_mask = frame["fold"].to_numpy(dtype=int) != fold
        valid_mask = ~train_mask
        if np.unique(y[train_mask]).size < 2:
            raise ValueError(f"Training partition for fold {fold} contains only one class")
        estimator = make_feature_estimator(model, C=C, random_state=random_state + fold)
        fitted, best_params = _fit_outer_estimator(estimator, model, X[train_mask], y[train_mask], tune_c, random_state + fold)
        probability = safe_probabilities(fitted.predict_proba(X[valid_mask])[:, 1])
        valid_rows = frame.loc[valid_mask, ["uid", "fold", "label"]].copy()
        valid_rows["target"] = valid_rows.pop("label").astype(float)
        valid_rows["probability"] = probability
        valid_rows["logit"] = logit(probability)
        predictions.append(valid_rows[["uid", "fold", "target", "probability", "logit"]])
        fold_metrics.append({"fold": fold, **binary_metrics(valid_rows["target"], valid_rows["probability"]), "best_params": best_params})
        with (model_directory / f"model_fold{fold}.pkl").open("wb") as handle:
            pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)

    oof = pd.concat(predictions, ignore_index=True).sort_values("uid").reset_index(drop=True)
    if len(oof) != len(frame) or oof["uid"].duplicated().any():
        raise RuntimeError("Feature OOF output must contain every UID exactly once")
    metrics = {
        "model": str(model),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "removed_features": removed,
        "overall": binary_metrics(oof["target"], oof["probability"]),
        "folds": fold_metrics,
        "fold_log_loss_std": float(np.std([row["log_loss"] for row in fold_metrics], ddof=0)),
        "mean_predicted_probability": float(oof["probability"].mean()),
        "actual_positive_fraction": float(oof["target"].mean()),
        "feature_families": {column: feature_family(column) for column in feature_columns},
        "random_state": int(random_state),
    }
    (model_directory / "feature_columns.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    _write_importance(fitted, feature_columns, model_directory / "feature_importance.csv")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output, index=False)
    (model_directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    return oof, metrics


def _write_importance(estimator, feature_columns: Sequence[str], output: Path) -> None:
    base = estimator
    if isinstance(estimator, Pipeline):
        base = estimator.named_steps["classifier"]
    if isinstance(base, LogisticRegression):
        values = base.coef_[0]
        rows = [{"feature": column, "importance": float(value), "absolute_importance": float(abs(value)), "kind": "standardized_coefficient"} for column, value in zip(feature_columns, values)]
    elif hasattr(base, "feature_importances_"):
        values = base.feature_importances_
        rows = [{"feature": column, "importance": float(value), "absolute_importance": float(abs(value)), "kind": "tree_importance"} for column, value in zip(feature_columns, values)]
    else:
        rows = [{"feature": column, "importance": float("nan"), "absolute_importance": float("nan"), "kind": "not_available"} for column in feature_columns]
    pd.DataFrame(rows).sort_values("absolute_importance", ascending=False).to_csv(output, index=False)
