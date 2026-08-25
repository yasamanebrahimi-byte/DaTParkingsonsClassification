"""Deterministic, label-free acquisition-domain construction.

Domain assignments are deliberately kept separate from labels.  The feature
list below contains only image geometry, NIfTI spacing, and foreground
occupancy metadata.  It does not contain outcome or model-derived features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Intensity percentiles are intentionally not used here: unlike geometry and
# foreground occupancy, their relationship to pathology can be substantial.
DOMAIN_FEATURES: Tuple[str, ...] = (
    "shape_x",
    "shape_y",
    "shape_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "physical_extent_x",
    "physical_extent_y",
    "physical_extent_z",
    "nonzero_fraction",
)

DOMAIN_METHOD = "kmeans_standardized_geometry_v1"
DEFAULT_DOMAIN_COUNT = 8
DEFAULT_DOMAIN_SEED = 20260824


def _validate_metadata(metadata: pd.DataFrame, require_label: bool = False) -> None:
    required = {"uid", *DOMAIN_FEATURES}
    if require_label:
        required.add("label")
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(missing)}")
    if metadata["uid"].duplicated().any():
        duplicates = metadata.loc[metadata["uid"].duplicated(), "uid"].astype(str).tolist()[:5]
        raise ValueError(f"Metadata contains duplicate UIDs, including: {duplicates}")


def geometry_signature_table(
    metadata: pd.DataFrame,
    *,
    spacing_decimals: int = 3,
    extent_decimals: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Return interpretable frequency tables for recurring geometry signatures."""

    _validate_metadata(metadata)
    frame = metadata.copy()
    for column in (
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "physical_extent_x",
        "physical_extent_y",
        "physical_extent_z",
    ):
        decimals = spacing_decimals if column.startswith("spacing") else extent_decimals
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(decimals)

    signatures = {
        "shape": ["shape_x", "shape_y", "shape_z"],
        "spacing": ["spacing_x", "spacing_y", "spacing_z"],
        "shape_spacing": [
            "shape_x",
            "shape_y",
            "shape_z",
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ],
        "shape_spacing_extent": [
            "shape_x",
            "shape_y",
            "shape_z",
            "spacing_x",
            "spacing_y",
            "spacing_z",
            "physical_extent_x",
            "physical_extent_y",
            "physical_extent_z",
        ],
    }
    tables: Dict[str, pd.DataFrame] = {}
    for name, columns in signatures.items():
        grouped = frame.groupby(columns, dropna=False, sort=False)
        table = grouped.size().rename("sample_count").reset_index()
        if "label" in frame.columns:
            label_stats = grouped["label"].agg(
                normal_count=lambda values: int((values == 0).sum()),
                pathologic_count=lambda values: int((values == 1).sum()),
                pathologic_fraction="mean",
            ).reset_index()
            table = table.merge(label_stats, on=columns, how="left", validate="one_to_one")
        tables[name] = table.sort_values("sample_count", ascending=False).reset_index(drop=True)
    return tables


def _fit_domain_model(metadata: pd.DataFrame, n_domains: int, seed: int):
    _validate_metadata(metadata)
    if len(metadata) == 0:
        raise ValueError("Cannot assign domains to an empty metadata table")
    if n_domains < 1:
        raise ValueError("n_domains must be at least 1")

    # Sorting before fitting makes the stochastic initialization independent of
    # the input row order while preserving the original order in the result.
    ordered = metadata.sort_values("uid", kind="mergesort")
    raw_values = ordered[list(DOMAIN_FEATURES)].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw_values)
    if not np.isfinite(imputed).all():
        raise ValueError("Domain features contain no finite value for at least one column")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    unique_rows = np.unique(scaled, axis=0).shape[0]
    cluster_count = min(int(n_domains), len(ordered), unique_rows)
    if cluster_count == 1:
        raw_labels = np.zeros(len(ordered), dtype=int)
        centers = np.zeros((1, scaled.shape[1]), dtype=float)
    else:
        model = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=int(seed),
        )
        raw_labels = model.fit_predict(scaled)
        centers = np.asarray(model.cluster_centers_, dtype=float)

    # KMeans cluster IDs are arbitrary.  Canonicalizing by center coordinates
    # makes the persisted domain names reproducible and interpretable.
    order = sorted(range(cluster_count), key=lambda index: tuple(centers[index].tolist()))
    remap = {old: new for new, old in enumerate(order)}
    canonical_labels = np.asarray([remap[int(label)] for label in raw_labels], dtype=int)
    canonical_centers = centers[order]
    labels_by_uid = dict(zip(ordered["uid"].astype(str), canonical_labels.tolist()))
    return labels_by_uid, imputer, scaler, canonical_centers, cluster_count


def assign_domain_groups(
    metadata: pd.DataFrame,
    n_domains: int = DEFAULT_DOMAIN_COUNT,
    seed: int = DEFAULT_DOMAIN_SEED,
) -> pd.DataFrame:
    """Assign each UID to a deterministic acquisition family.

    The returned table contains only ``uid``, ``domain_group`` and
    ``domain_method``.  Labels are never read by this function, which makes it
    safe to use before any label-based fold construction.
    """

    labels_by_uid, _, _, _, _ = _fit_domain_model(metadata, n_domains, seed)
    result = metadata[["uid"]].copy()
    result["domain_group"] = result["uid"].astype(str).map(
        lambda uid: f"domain_{labels_by_uid[uid]:02d}"
    )
    result["domain_method"] = DOMAIN_METHOD
    if result["domain_group"].isna().any():
        raise ValueError("Every UID must receive exactly one domain_group")
    return result


def assign_domain_groups_with_config(
    metadata: pd.DataFrame,
    n_domains: int = DEFAULT_DOMAIN_COUNT,
    seed: int = DEFAULT_DOMAIN_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign domains and return the exact preprocessing/clustering config."""

    labels_by_uid, imputer, scaler, centers, cluster_count = _fit_domain_model(
        metadata, n_domains, seed
    )
    result = metadata[["uid"]].copy()
    result["domain_group"] = result["uid"].astype(str).map(
        lambda uid: f"domain_{labels_by_uid[uid]:02d}"
    )
    result["domain_method"] = DOMAIN_METHOD
    config: Dict[str, Any] = {
        "method": DOMAIN_METHOD,
        "feature_columns": list(DOMAIN_FEATURES),
        "excluded_feature_columns": [
            "label",
            "p95_nonzero",
            "p99_nonzero",
            "p99_5_nonzero",
            "median_nonzero",
            "mean_intensity",
            "min_intensity",
            "max_intensity",
            "uid",
        ],
        "n_domains_requested": int(n_domains),
        "n_domains_fitted": int(cluster_count),
        "random_state": int(seed),
        "kmeans_n_init": 20,
        "kmeans_algorithm": "lloyd",
        "imputation": "median",
        "imputer_statistics": [float(value) for value in imputer.statistics_],
        "scaling": "StandardScaler",
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "cluster_centers_standardized": centers.tolist(),
        "signature_rounding": {"spacing_decimals": 3, "extent_decimals": 1},
        "intensity_features_used": False,
    }
    return result, config


def domain_summary(metadata: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize size and label prevalence after domains are assigned."""

    _validate_metadata(metadata, require_label=True)
    if not {"uid", "domain_group"}.issubset(assignments.columns):
        raise ValueError("Domain assignments require uid and domain_group columns")
    if assignments["uid"].duplicated().any():
        raise ValueError("Domain assignments contain duplicate UIDs")
    if set(assignments["uid"].astype(str)) != set(metadata["uid"].astype(str)):
        raise ValueError("Domain assignments must contain exactly the metadata UIDs")
    left = metadata[["uid", "label"]].copy()
    left["uid"] = left["uid"].astype(str)
    right = assignments[["uid", "domain_group"]].copy()
    right["uid"] = right["uid"].astype(str)
    merged = left.merge(right, on="uid", how="inner", validate="one_to_one")
    result = (
        merged.groupby("domain_group", sort=True)
        .agg(
            sample_count=("uid", "size"),
            normal_count=("label", lambda values: int((values == 0).sum())),
            pathologic_count=("label", lambda values: int((values == 1).sum())),
            pathologic_fraction=("label", "mean"),
        )
        .reset_index()
    )
    return result.sort_values(["sample_count", "domain_group"], ascending=[False, True]).reset_index(drop=True)


def save_domain_config(config: Mapping[str, Any], path: str | Path) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(config), indent=2) + "\n", encoding="utf-8")
