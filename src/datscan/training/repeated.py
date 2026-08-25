"""Repeated-CV OOF aggregation and diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ..utils.metrics import binary_metrics, safe_probabilities


REPEATED_OOF_COLUMNS = ["uid", "target", "repeat", "fold", "logit", "probability"]
REPEATED_OOF_METADATA_COLUMNS = ["fold_seed", "training_seed", "experiment_name"]


def validate_repeated_oof(frame: pd.DataFrame, n_repeats: int | None = None) -> None:
    """Validate the long repeated-OOF contract and fail closed on missing rows."""

    missing = sorted(set(REPEATED_OOF_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Repeated OOF file is missing columns: {missing}")
    key_columns = ["uid", "repeat"]
    if "training_seed" in frame.columns:
        key_columns.append("training_seed")
    if frame.duplicated(key_columns).any():
        raise ValueError(f"Repeated OOF contains duplicate {'/'.join(key_columns)} predictions")
    if not np.isfinite(frame[["target", "repeat", "fold", "logit", "probability"]].to_numpy(dtype=float)).all():
        raise ValueError("Repeated OOF contains non-finite values")
    probabilities = expit(frame["logit"].to_numpy(dtype=float))
    if not np.allclose(probabilities, frame["probability"].to_numpy(dtype=float), atol=1e-5, rtol=1e-5):
        raise ValueError("Repeated OOF probability must be sigmoid(logit)")
    repeat_counts = frame.groupby(["uid", "repeat"], sort=False).size()
    if repeat_counts.empty or not (repeat_counts == repeat_counts.iloc[0]).all():
        bad = repeat_counts[repeat_counts != repeat_counts.iloc[0]].head().to_dict() if not repeat_counts.empty else {}
        raise ValueError(f"Every UID/repeat pair must have the same number of OOF predictions; mismatches={bad}")
    counts = frame.groupby("uid", sort=False)["repeat"].nunique()
    expected_repeats = int(n_repeats if n_repeats is not None else counts.max())
    if expected_repeats < 1 or not (counts == expected_repeats).all():
        bad = counts[counts != expected_repeats].head().to_dict()
        raise ValueError(f"Every UID must have exactly {expected_repeats} repeated OOF partitions; mismatches={bad}")
    repeat_values = set(int(value) for value in frame["repeat"].unique())
    if repeat_values != set(range(expected_repeats)):
        raise ValueError(f"Repeat IDs must be contiguous 0..{expected_repeats - 1}")
    per_repeat = frame.groupby("repeat")["uid"].nunique()
    if per_repeat.nunique() != 1:
        raise ValueError("Each repeated OOF split must cover the same UIDs")
    uid_sets = [set(group["uid"].astype(str)) for _, group in frame.groupby("repeat", sort=True)]
    if uid_sets and any(current != uid_sets[0] for current in uid_sets[1:]):
        raise ValueError("Each repeated OOF split must contain exactly the same UIDs")


def validate_repeated_oof_assignments(
    metadata: pd.DataFrame,
    fold_tables: dict[int, pd.DataFrame],
    repeated_oof: pd.DataFrame,
) -> None:
    """Check that each prediction's held-out fold excludes its UID from training.

    This is an assignment-level leakage check; it does not infer training data
    from a prediction value.  ``train_one_fold`` constructs training rows as
    ``fold != held_out_fold`` and repeated training uses these same tables.
    """

    required = {"uid", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError("Metadata requires uid and label columns")
    validate_repeated_oof(repeated_oof)
    metadata_uids = set(metadata["uid"].astype(str))
    if set(repeated_oof["uid"].astype(str)) != metadata_uids:
        raise ValueError("Repeated OOF UIDs do not match metadata")
    observed_repeats = set(int(value) for value in repeated_oof["repeat"].unique())
    if set(int(value) for value in fold_tables) != observed_repeats:
        raise ValueError("Fold tables must be supplied for every repeated OOF repeat")
    for repeat, table in fold_tables.items():
        assignments = table.copy()
        assignments["uid"] = assignments["uid"].astype(str)
        if assignments["uid"].duplicated().any():
            raise ValueError(f"Repeat {repeat} has duplicate fold assignments")
        observed = repeated_oof[repeated_oof["repeat"] == repeat]
        merged = observed[["uid", "fold"]].drop_duplicates().copy()
        merged["uid"] = merged["uid"].astype(str)
        merged = merged.merge(assignments[["uid", "fold"]], on="uid", how="left", suffixes=("_oof", "_assignment"), validate="one_to_one")
        if merged["fold_assignment"].isna().any() or not np.array_equal(merged["fold_oof"].to_numpy(), merged["fold_assignment"].to_numpy()):
            raise ValueError(f"Repeated OOF fold assignments do not match repeat {repeat}")


def aggregate_repeated_oof(frame: pd.DataFrame, n_repeats: int | None = None) -> pd.DataFrame:
    """Merge long repeated OOF predictions into one ensemble-like row per UID."""

    validate_repeated_oof(frame, n_repeats=n_repeats)
    source = frame.copy()
    source["uid"] = source["uid"].astype(str)
    source["target"] = source["target"].astype(float)
    target_counts = source.groupby("uid")["target"].nunique()
    if (target_counts > 1).any():
        raise ValueError("Repeated OOF targets disagree for at least one UID")
    grouped = source.groupby("uid", sort=True)
    summary = grouped.agg(
        target=("target", "first"),
        n_predictions=("repeat", "size"),
        mean_logit=("logit", "mean"),
        mean_probability=("probability", "mean"),
        median_probability=("probability", "median"),
        prediction_std=("probability", "std"),
        prediction_min=("probability", "min"),
        prediction_max=("probability", "max"),
    ).reset_index()
    summary["prediction_std"] = summary["prediction_std"].fillna(0.0)
    summary["prob_from_mean_logit"] = safe_probabilities(expit(summary["mean_logit"].to_numpy(dtype=float)))
    summary["prediction_mean"] = summary["mean_probability"]
    expected_repeats = int(n_repeats if n_repeats is not None else source["repeat"].nunique())
    predictions_per_partition = int(source.groupby(["uid", "repeat"], sort=False).size().iloc[0])
    expected = expected_repeats * predictions_per_partition
    if not (summary["n_predictions"] == expected).all():
        raise ValueError(f"Every UID must have exactly {expected} repeated OOF predictions")
    return summary[
        [
            "uid",
            "target",
            "n_predictions",
            "mean_logit",
            "prob_from_mean_logit",
            "mean_probability",
            "prediction_mean",
            "median_probability",
            "prediction_std",
            "prediction_min",
            "prediction_max",
        ]
    ]


def repeated_summary_metrics(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    required = {"target", "prob_from_mean_logit", "mean_probability"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Repeated summary is missing columns: {sorted(missing)}")
    result = {
        "mean_logits": binary_metrics(summary["target"], summary["prob_from_mean_logit"]),
        "mean_probabilities": binary_metrics(summary["target"], summary["mean_probability"]),
    }
    if "median_probability" in summary.columns:
        result["median_probabilities"] = binary_metrics(summary["target"], summary["median_probability"])
    return result


def variance_loss_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Return per-sample uncertainty and log-loss diagnostics."""

    required = {"uid", "target", "mean_probability", "prediction_std"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Repeated summary is missing columns: {sorted(missing)}")
    result = summary[["uid", "target", "mean_probability", "prediction_std"]].copy()
    result["per_sample_log_loss"] = -(
        result["target"] * np.log(np.clip(result["mean_probability"], 1e-6, 1.0 - 1e-6))
        + (1.0 - result["target"]) * np.log(np.clip(1.0 - result["mean_probability"], 1e-6, 1.0 - 1e-6))
    )
    return result.sort_values("prediction_std", ascending=False).reset_index(drop=True)


def save_repeated_summary(summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
