"""Reproducible standard and acquisition-domain-aware fold creation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - only applies to old sklearn installs
    StratifiedGroupKFold = None


def create_folds(metadata: pd.DataFrame, n_splits: int = 5, seed: int = 20260824) -> pd.DataFrame:
    """Create the canonical IID label-stratified assignment.

    This remains the original standard validation strategy.  Domain-aware
    validation is provided separately by :func:`create_domain_folds`.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if "uid" not in metadata or "label" not in metadata:
        raise ValueError("Metadata requires uid and label columns")
    if metadata["uid"].duplicated().any():
        raise ValueError("Cannot create folds with duplicate UIDs")
    if metadata["label"].isna().any():
        raise ValueError("Cannot create folds with missing labels")
    result = metadata[["uid", "label"]].reset_index(drop=True).copy()
    result["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(splitter.split(result, result["label"].astype(int))):
        result.iloc[validation_indices, result.columns.get_loc("fold")] = fold
    if (result["fold"] < 0).any():
        raise RuntimeError("Standard fold generation left one or more UIDs unassigned")
    return result.sort_values("uid").reset_index(drop=True)


def _domain_assignments_frame(metadata: pd.DataFrame, assignments: Any) -> pd.DataFrame:
    if assignments is None:
        if "domain_group" not in metadata:
            raise ValueError("Domain-aware folds require --groups or a domain_group column")
        assignments = metadata[["uid", "domain_group"]]
    elif isinstance(assignments, (str, Path)):
        assignments = pd.read_csv(assignments)
    elif isinstance(assignments, pd.Series):
        assignments = pd.DataFrame({"uid": metadata["uid"], "domain_group": assignments})
    else:
        assignments = assignments.copy()
    required = {"uid", "domain_group"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Domain assignments are missing columns: {', '.join(missing)}")
    if assignments["uid"].duplicated().any():
        raise ValueError("Domain assignments contain duplicate UIDs")
    if assignments["domain_group"].isna().any():
        raise ValueError("Domain assignments contain missing domain_group values")
    metadata_uids = set(metadata["uid"].astype(str))
    assignment_uids = set(assignments["uid"].astype(str))
    if metadata_uids != assignment_uids:
        missing_uids = sorted(metadata_uids - assignment_uids)[:5]
        extra_uids = sorted(assignment_uids - metadata_uids)[:5]
        raise ValueError(
            f"Domain assignments must match metadata UIDs exactly; missing={missing_uids}, extra={extra_uids}"
        )
    result = assignments[["uid", "domain_group"]].copy()
    result["uid"] = result["uid"].astype(str)
    result["domain_group"] = result["domain_group"].astype(str)
    return result


def fold_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validation-set size, class balance, and domain counts by fold."""

    required = {"uid", "label", "fold"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fold frame is missing columns: {', '.join(missing)}")
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        row = {
            "fold": int(fold),
            "sample_count": int(len(group)),
            "normal_count": int((group["label"] == 0).sum()),
            "pathologic_count": int((group["label"] == 1).sum()),
            "pathologic_fraction": float(group["label"].mean()),
        }
        if "domain_group" in group:
            counts = group["domain_group"].value_counts().sort_index()
            row["number_of_domains"] = int(counts.size)
            row["domain_names_counts"] = ", ".join(
                f"{name}:{int(count)}" for name, count in counts.items()
            )
        else:
            row["number_of_domains"] = None
            row["domain_names_counts"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def validate_domain_folds(frame: pd.DataFrame) -> None:
    """Raise a clear error if grouped-CV leakage or assignment defects exist."""

    required = {"uid", "label", "fold", "domain_group"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Domain folds are missing columns: {', '.join(missing)}")
    if frame["uid"].duplicated().any():
        raise ValueError("Domain folds contain duplicate UIDs")
    if frame["fold"].isna().any() or (frame["fold"] < 0).any():
        raise ValueError("Every UID must have a valid non-negative fold assignment")
    if frame["domain_group"].isna().any():
        raise ValueError("Every UID must have a domain_group assignment")
    if frame["label"].isna().any():
        raise ValueError("Domain folds contain missing labels")
    all_uids = set(frame["uid"].astype(str))
    if len(all_uids) != len(frame):
        raise ValueError("Every UID must appear exactly once in a fold file")
    fold_values = sorted(int(value) for value in frame["fold"].unique())
    if fold_values != list(range(len(fold_values))):
        raise ValueError("Fold IDs must be contiguous integers starting at zero")
    for fold in sorted(frame["fold"].unique()):
        validation = frame[frame["fold"] == fold]
        training = frame[frame["fold"] != fold]
        train_domains = set(training["domain_group"].astype(str))
        valid_domains = set(validation["domain_group"].astype(str))
        overlap = train_domains.intersection(valid_domains)
        if overlap:
            raise ValueError(
                f"Domain leakage detected in fold {fold}: {sorted(overlap)} appear in train and validation"
            )


def leave_one_domain_out_splits(
    metadata: pd.DataFrame,
    assignments: Any = None,
    min_validation_samples: int = 30,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return train/validation frames for each sufficiently large domain.

    This is intentionally separate from the ordinary fold-file schema because
    a UID can be in the training partition of many leave-one-domain-out runs.
    """

    if min_validation_samples < 1:
        raise ValueError("min_validation_samples must be at least 1")
    if "uid" not in metadata or "label" not in metadata:
        raise ValueError("Metadata requires uid and label columns")
    if metadata["uid"].duplicated().any():
        raise ValueError("Cannot create leave-one-domain-out splits with duplicate UIDs")
    domains = _domain_assignments_frame(metadata, assignments)
    frame = metadata[["uid", "label"]].copy()
    frame["uid"] = frame["uid"].astype(str)
    frame = frame.merge(domains, on="uid", how="inner", validate="one_to_one")
    counts = frame["domain_group"].value_counts().sort_index()
    eligible = [domain for domain, count in counts.items() if int(count) >= min_validation_samples]
    if not eligible:
        raise ValueError(
            f"No domain has at least {min_validation_samples} validation samples"
        )
    result = []
    for domain in eligible:
        validation = frame[frame["domain_group"] == domain].reset_index(drop=True)
        training = frame[frame["domain_group"] != domain].reset_index(drop=True)
        result.append((str(domain), training, validation))
    return result


def _candidate_is_class_balanced(frame: pd.DataFrame) -> bool:
    classes = set(frame["label"].astype(int).unique())
    if len(classes) < 2:
        return True
    for fold in sorted(frame["fold"].unique()):
        validation_classes = set(frame.loc[frame["fold"] == fold, "label"].astype(int).unique())
        training_classes = set(frame.loc[frame["fold"] != fold, "label"].astype(int).unique())
        if validation_classes != classes or training_classes != classes:
            return False
    return True


def create_domain_folds(
    metadata: pd.DataFrame,
    assignments: Any = None,
    n_splits: int = 5,
    seed: int = 20260824,
) -> pd.DataFrame:
    """Create grouped, label-stratified folds with no domain split.

    The requested fold count is reduced when there are fewer domain groups than
    requested or when a smaller count is needed to keep both classes present in
    each validation/training partition.  The chosen count is exposed in
    ``result.attrs`` and should also be reported by callers.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for domain-aware folds")
    if "uid" not in metadata or "label" not in metadata:
        raise ValueError("Metadata requires uid and label columns")
    if metadata["uid"].duplicated().any():
        raise ValueError("Cannot create domain folds with duplicate UIDs")
    if metadata["label"].isna().any():
        raise ValueError("Cannot create domain folds with missing labels")
    if StratifiedGroupKFold is None:
        raise RuntimeError(
            "Domain-aware folds require scikit-learn StratifiedGroupKFold (scikit-learn >= 1.1)"
        )

    domains = _domain_assignments_frame(metadata, assignments)
    base = metadata[["uid", "label"]].copy()
    base["uid"] = base["uid"].astype(str)
    frame = base.merge(domains, on="uid", how="inner", validate="one_to_one")
    group_count = int(frame["domain_group"].nunique())
    if group_count < 2:
        raise ValueError("At least two domain groups are required for domain-aware folds")
    requested = int(n_splits)
    highest_candidate = min(requested, group_count)
    if highest_candidate != requested:
        warnings.warn(
            f"Requested {requested} grouped folds but only {group_count} domains exist; using {highest_candidate}",
            UserWarning,
        )

    chosen = None
    balanced_choice = False
    for candidate in range(highest_candidate, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=candidate,
            shuffle=True,
            random_state=int(seed),
        )
        trial = frame.copy()
        trial["fold"] = -1
        for fold, (_, validation_indices) in enumerate(
            splitter.split(trial, trial["label"].astype(int), trial["domain_group"])
        ):
            trial.iloc[validation_indices, trial.columns.get_loc("fold")] = fold
        if (trial["fold"] < 0).any():
            continue
        validate_domain_folds(trial[["uid", "label", "fold", "domain_group"]])
        if _candidate_is_class_balanced(trial):
            chosen = trial
            balanced_choice = True
            break
        if chosen is None:
            chosen = trial

    if chosen is None:
        raise ValueError("Unable to construct a valid grouped fold assignment")
    if not balanced_choice:
        warnings.warn(
            "No grouped fold count from the requested range preserved both classes in every partition; "
            "returning the highest valid grouped assignment",
            UserWarning,
        )
    chosen = chosen[["uid", "label", "fold", "domain_group"]].sort_values("uid").reset_index(drop=True)
    validate_domain_folds(chosen)
    actual = int(chosen["fold"].nunique())
    quality = fold_quality(chosen)
    global_fraction = float(chosen["label"].mean())
    size_ratio = float(quality["sample_count"].max() / max(quality["sample_count"].min(), 1))
    fraction_gap = float((quality["pathologic_fraction"] - global_fraction).abs().max())
    if size_ratio > 2.0 or fraction_gap > 0.15:
        warnings.warn(
            "Domain-aware folds are materially uneven; inspect fold quality before interpreting model metrics "
            f"(sample-size ratio={size_ratio:.2f}, maximum prevalence gap={fraction_gap:.3f})",
            UserWarning,
        )
    chosen.attrs["strategy"] = "stratified_group"
    chosen.attrs["requested_n_splits"] = requested
    chosen.attrs["n_splits"] = actual
    chosen.attrs["seed"] = int(seed)
    chosen.attrs["balanced_classes"] = balanced_choice
    chosen.attrs["fold_quality"] = quality.to_dict(orient="records")
    return chosen


def save_folds(folds: pd.DataFrame, path: str | Path) -> None:
    if "uid" not in folds or "fold" not in folds:
        raise ValueError("Fold table requires uid and fold columns")
    if folds["uid"].duplicated().any():
        raise ValueError("Cannot save folds with duplicate UIDs")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
