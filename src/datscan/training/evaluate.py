"""OOF evaluation, fold summaries, and domain-level diagnostics."""

from __future__ import annotations

import pandas as pd

from ..utils.metrics import binary_metrics
from ..utils.reporting import markdown_table


def evaluate_oof(frame: pd.DataFrame) -> dict:
    if not {"target", "probability"}.issubset(frame.columns):
        raise ValueError("OOF frame requires target and probability columns")
    overall = binary_metrics(frame["target"], frame["probability"])
    fold_rows = []
    if "fold" in frame:
        for fold, group in frame.groupby("fold"):
            row = {"fold": int(fold), **binary_metrics(group["target"], group["probability"])}
            fold_rows.append(row)
    return {"overall": overall, "folds": fold_rows}


def evaluate_oof_by_domain(oof: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Evaluate OOF probabilities separately for each acquisition domain.

    Log loss and Brier score remain defined for one-class domains.  AUROC is
    returned as NaN there because it is undefined rather than manufacturing a
    value from a single class.
    """

    if "uid" not in oof or "uid" not in assignments:
        raise ValueError("OOF predictions and domain assignments both require uid")
    target_column = "target" if "target" in oof else "label" if "label" in oof else None
    probability_column = (
        "probability"
        if "probability" in oof
        else "prediction"
        if "prediction" in oof
        else "predicted_probability"
        if "predicted_probability" in oof
        else None
    )
    if target_column is None or probability_column is None:
        raise ValueError("OOF predictions require target/label and probability/prediction columns")
    if assignments["uid"].duplicated().any() or oof["uid"].duplicated().any():
        raise ValueError("OOF predictions and domain assignments must not contain duplicate UIDs")
    if "domain_group" not in assignments:
        raise ValueError("Domain assignments require domain_group")
    if assignments["domain_group"].isna().any():
        raise ValueError("Domain assignments contain missing domain_group values")
    if set(oof["uid"].astype(str)) != set(assignments["uid"].astype(str)):
        raise ValueError("OOF predictions and domain assignments must contain the same UIDs")

    left = oof[["uid", target_column, probability_column]].copy()
    left["uid"] = left["uid"].astype(str)
    right = assignments[["uid", "domain_group"]].copy()
    right["uid"] = right["uid"].astype(str)
    merged = left.merge(right, on="uid", how="inner", validate="one_to_one")
    rows = []
    for domain, group in merged.groupby("domain_group", sort=True):
        metrics = binary_metrics(group[target_column], group[probability_column])
        rows.append(
            {
                "domain_group": domain,
                "n": int(len(group)),
                "normal": int((group[target_column] == 0).sum()),
                "pathologic": int((group[target_column] == 1).sum()),
                "log_loss": metrics["log_loss"],
                "auroc": metrics["auroc"],
                "brier": metrics["brier_score"],
                "mean_predicted_probability": float(group[probability_column].mean()),
                "true_pathologic_fraction": float(group[target_column].mean()),
                "calibration_error": float(
                    group[probability_column].mean() - group[target_column].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["log_loss", "domain_group"], ascending=[False, True]
    ).reset_index(drop=True)


def render_domain_validation_report(
    domain_results: pd.DataFrame,
    output: str,
    *,
    standard_oof: pd.DataFrame | None = None,
    domain_oof: pd.DataFrame | None = None,
    fold_quality: pd.DataFrame | None = None,
) -> None:
    """Write a Markdown domain diagnostic and optional CV comparison."""

    lines = [
        "# Domain-aware validation",
        "",
        "Domain-aware CV holds acquisition families out of each validation fold. "
        "It is a robustness diagnostic and does not replace canonical IID StratifiedKFold.",
        "",
    ]
    if standard_oof is not None and domain_oof is not None:
        standard_evaluation = evaluate_oof(standard_oof)
        domain_evaluation = evaluate_oof(domain_oof)
        standard = standard_evaluation["overall"]
        domain = domain_evaluation["overall"]
        lines.extend(
            [
                "## Standard vs domain-aware OOF",
                "",
                "| Validation | Log Loss | AUROC | Brier |",
                "| --- | ---: | ---: | ---: |",
                f"| Standard Stratified CV | {standard['log_loss']:.6f} | {standard['auroc']:.6f} | {standard['brier_score']:.6f} |",
                f"| Domain-Aware CV | {domain['log_loss']:.6f} | {domain['auroc']:.6f} | {domain['brier_score']:.6f} |",
                "",
                "Interpretation: the two rows answer different questions. A domain-aware score is not "
                "automatically better because it is harder; it indicates how performance changes under "
                "acquisition-family shift.",
                "",
            ]
        )
        fold_rows = []
        for validation, evaluation in (
            ("Standard Stratified CV", standard_evaluation),
            ("Domain-Aware CV", domain_evaluation),
        ):
            for row in evaluation["folds"]:
                fold_rows.append({"validation": validation, **row})
        if fold_rows:
            lines.extend(
                [
                    "### Fold-level metric results",
                    "",
                    markdown_table(pd.DataFrame(fold_rows)),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Standard vs domain-aware OOF",
                "",
                "| Validation | Log Loss | AUROC | Brier |",
                "| --- | ---: | ---: | ---: |",
                "| Standard Stratified CV | pending OOF predictions | pending | pending |",
                "| Domain-Aware CV | pending OOF predictions | pending | pending |",
                "",
                "Neural-model OOF predictions were not supplied, so this implementation report "
                "does not fabricate performance values.",
                "",
            ]
        )

    if fold_quality is not None and not fold_quality.empty:
        lines.extend(
            [
                "## Domain-fold distribution",
                "",
                markdown_table(fold_quality),
                "",
            ]
        )

    lines.extend(
        [
            "## Domain-level results",
            "",
            markdown_table(domain_results) if not domain_results.empty else "No domain results.",
            "",
        ]
    )
    if not domain_results.empty:
        highest = domain_results.nlargest(min(5, len(domain_results)), "log_loss")
        lowest = domain_results.nsmallest(min(5, len(domain_results)), "log_loss")
        over = domain_results.sort_values("calibration_error", ascending=False).head(5)
        under = domain_results.sort_values("calibration_error", ascending=True).head(5)
        lines.extend(
            [
                "## Worst and best generalizing domains",
                "",
                "Highest log-loss domains:",
                "",
                markdown_table(highest[["domain_group", "n", "log_loss"]]),
                "",
                "Lowest log-loss domains:",
                "",
                markdown_table(lowest[["domain_group", "n", "log_loss"]]),
                "",
                "Systematic overprediction (mean predicted probability above prevalence):",
                "",
                markdown_table(over[["domain_group", "n", "mean_predicted_probability", "true_pathologic_fraction", "calibration_error"]]),
                "",
                "Systematic underprediction (mean predicted probability below prevalence):",
                "",
                markdown_table(under[["domain_group", "n", "mean_predicted_probability", "true_pathologic_fraction", "calibration_error"]]),
                "",
            ]
        )
    from pathlib import Path

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
