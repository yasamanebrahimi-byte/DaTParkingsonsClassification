"""Analyze genuinely out-of-sample repeated-CV ensemble predictions.

The input is the long artifact written by ``train_repeated_cv.py``.  Every
column used for an ensemble is one held-out prediction per UID from a repeat
and, when present, a distinct training seed.  Ordinary fold checkpoints are
never treated as aligned OOF members here.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.repeated import aggregate_repeated_oof, validate_repeated_oof  # noqa: E402
from datscan.utils.metrics import binary_metrics, safe_probabilities  # noqa: E402


def _loss(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    target = np.asarray(target, dtype=float)
    return -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    values = binary_metrics(target, safe_probabilities(probability))
    return {"log_loss": values["log_loss"], "auroc": values["auroc"], "brier": values["brier_score"]}


def _aligned_members(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame, list[str]]:
    source = frame.copy()
    source["uid"] = source["uid"].astype(str)
    source["member"] = "repeat_" + source["repeat"].astype(int).astype(str)
    if "training_seed" in source.columns:
        source["member"] = source["member"] + "_seed_" + source["training_seed"].astype(int).astype(str)
    if source.duplicated(["uid", "member"]).any():
        raise ValueError("OOF contains duplicate UID/member rows")
    members = sorted(source["member"].unique(), key=lambda value: tuple(int(part) for part in value.replace("repeat_", "").replace("_seed_", "_").split("_")))
    target = source.drop_duplicates("uid").set_index("uid")["target"].astype(float).sort_index()
    probabilities = source.pivot(index="uid", columns="member", values="probability").reindex(columns=members).sort_index()
    logits = source.pivot(index="uid", columns="member", values="logit").reindex(index=probabilities.index, columns=members)
    if probabilities.isna().any().any() or logits.isna().any().any():
        raise ValueError("Every UID must have one aligned prediction for every ensemble member")
    target = target.reindex(probabilities.index)
    probabilities = probabilities.reset_index().rename(columns={"uid": "_uid"})
    uids = probabilities.pop("_uid")
    target = target.reindex(uids).reset_index(drop=True)
    return uids.reset_index(drop=True), target, probabilities.reset_index(drop=True), members


def _subset_indices(n_members: int, size: int, max_combinations: int, seed: int) -> list[tuple[int, ...]]:
    combinations = math_comb(n_members, size)
    if combinations <= max_combinations:
        return list(itertools.combinations(range(n_members), size))
    rng = np.random.default_rng(seed + size)
    selected: set[tuple[int, ...]] = set()
    while len(selected) < max_combinations:
        selected.add(tuple(sorted(rng.choice(n_members, size=size, replace=False).tolist())))
    return sorted(selected)


def math_comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for index in range(1, k + 1):
        result = result * (n - k + index) // index
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_markdown(
    output: Path,
    source: pd.DataFrame,
    target: np.ndarray,
    probabilities: np.ndarray,
    members: list[str],
    per_model: pd.DataFrame,
    sizes: pd.DataFrame,
    diversity: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    mean_probability = probabilities.mean(axis=1)
    mean_logit = expit(np.log(np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0)).mean(axis=1))
    median_probability = np.median(probabilities, axis=1)
    lines = [
        "# Repeated-model ensemble analysis",
        "",
        "This report uses only the genuinely held-out rows in the repeated OOF artifact. "
        "Fold checkpoints are not treated as aligned members for weight fitting.",
        "",
        f"Members: {len(members)}; UIDs: {len(target)}; members per UID: {len(members)}.",
        "",
        "## Individual model/run and raw aggregation",
        "",
        "| Prediction | OOF Log Loss | AUROC | Brier |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, probability in [(member, probabilities[:, index]) for index, member in enumerate(members)]:
        metric = _metrics(target, probability)
        lines.append(f"| {name} | {metric['log_loss']:.6f} | {metric['auroc']:.6f} | {metric['brier']:.6f} |")
    for name, probability in (
        ("Mean probability", mean_probability),
        ("Mean logit", mean_logit),
        ("Median probability", median_probability),
    ):
        metric = _metrics(target, probability)
        lines.append(f"| {name} | {metric['log_loss']:.6f} | {metric['auroc']:.6f} | {metric['brier']:.6f} |")
    lines.extend(["", "## Per-model held-out diagnostics", "", _markdown_table(per_model), "", "## Ensemble-size growth", "", _markdown_table(sizes), "", "## Diversity", "", _markdown_table(diversity) if not diversity.empty else "No pairwise members available.", ""])
    ensemble_loss = _loss(target, mean_probability)
    individual_losses = np.column_stack([_loss(target, probabilities[:, index]) for index in range(probabilities.shape[1])])
    lines.extend([
        "## Extreme-error and variance diagnostics",
        "",
        f"Mean individual log loss: {individual_losses.mean():.6f}",
        f"Std individual log loss: {individual_losses.mean(axis=0).std(ddof=0):.6f}",
        f"Mean ensemble log loss: {ensemble_loss.mean():.6f}",
        f"Samples where ensemble loss < mean individual loss: {int((ensemble_loss < individual_losses.mean(axis=1)).sum())} / {len(target)}",
        f"Samples where ensemble loss > mean individual loss: {int((ensemble_loss > individual_losses.mean(axis=1)).sum())} / {len(target)}",
        f"Mean prediction standard deviation: {probabilities.std(axis=1).mean():.6f}",
        f"Prediction-std / ensemble-loss Pearson correlation: {sample['prediction_std'].corr(sample['ensemble_log_loss'], method='pearson'):.6f}",
        f"Prediction-std / ensemble-loss Spearman correlation: {sample['prediction_std'].corr(sample['ensemble_log_loss'], method='spearman'):.6f}",
        "",
        "The raw ensemble representation should be selected from these OOF results before fitting Priority 6 calibration.",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="Long repeated OOF CSV")
    parser.add_argument("--output", required=True, help="Markdown report path")
    parser.add_argument("--max-ensemble-size", type=int, default=15)
    parser.add_argument("--max-combinations", type=int, default=200)
    parser.add_argument("--analysis-seed", type=int, default=20260824)
    args = parser.parse_args(argv)

    source = pd.read_csv(args.oof, dtype={"uid": str})
    validate_repeated_oof(source)
    uids, target_series, probability_frame, members = _aligned_members(source)
    target = target_series.to_numpy(dtype=float)
    probabilities = probability_frame.to_numpy(dtype=float)
    output = Path(args.output)
    stem = output.with_suffix("")

    per_model_rows = []
    model_columns = source.copy()
    model_columns["member"] = "repeat_" + model_columns["repeat"].astype(int).astype(str)
    if "training_seed" in model_columns.columns:
        model_columns["member"] = model_columns["member"] + "_seed_" + model_columns["training_seed"].astype(int).astype(str)
    for member, group in model_columns.groupby("member", sort=False):
        metric = _metrics(group["target"].to_numpy(dtype=float), group["probability"].to_numpy(dtype=float))
        first = group.iloc[0]
        per_model_rows.append({"member": member, "repeat": int(first["repeat"]), "fold": "mixed", "training_seed": int(first["training_seed"]) if "training_seed" in group else "mixed", "n_samples": len(group), **metric})
    for (repeat, fold, *seed), group in model_columns.groupby(["repeat", "fold"] + (["training_seed"] if "training_seed" in model_columns else []), sort=True):
        metric = _metrics(group["target"].to_numpy(dtype=float), group["probability"].to_numpy(dtype=float))
        per_model_rows.append({"member": "model", "repeat": int(repeat), "fold": int(fold), "training_seed": int(seed[0]) if seed else "", "n_samples": len(group), **metric})
    per_model = pd.DataFrame(per_model_rows)
    per_model.to_csv(stem.with_name(stem.name + "_per_model_metrics.csv"), index=False)

    size_rows = []
    max_size = min(max(int(args.max_ensemble_size), 1), len(members))
    for size in range(1, max_size + 1):
        rows = []
        for indices in _subset_indices(len(members), size, max(int(args.max_combinations), 1), int(args.analysis_seed)):
            selected = probabilities[:, indices]
            rows.append({
                "size": size,
                "members": "+".join(members[index] for index in indices),
                "method": "probability_mean",
                **_metrics(target, selected.mean(axis=1)),
            })
        current = pd.DataFrame(rows)
        size_rows.append({
            "ensemble_size": size,
            "n_subsets": len(current),
            "log_loss": current["log_loss"].mean(),
            "log_loss_std": current["log_loss"].std(ddof=0),
            "auroc": current["auroc"].mean(),
            "brier": current["brier"].mean(),
            "delta_vs_previous": np.nan if size == 1 else current["log_loss"].mean() - size_rows[-1]["log_loss"],
            "relative_inference_cost": float(size),
        })
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(stem.with_name(stem.name + "_ensemble_sizes.csv"), index=False)

    diversity_rows = []
    for first, second in itertools.combinations(range(len(members)), 2):
        left, right = probabilities[:, first], probabilities[:, second]
        diversity_rows.append({
            "member_a": members[first],
            "member_b": members[second],
            "pearson_probability_correlation": pearsonr(left, right).statistic if len(left) > 1 else np.nan,
            "spearman_probability_correlation": spearmanr(left, right).statistic if len(left) > 1 else np.nan,
            "disagreement_rate_at_0_5": float(((left >= 0.5) != (right >= 0.5)).mean()),
            "mean_absolute_prediction_difference": float(np.abs(left - right).mean()),
            "mean_pair_log_loss": float(np.mean([_metrics(target, left)["log_loss"], _metrics(target, right)["log_loss"]])),
            "probability_mean_log_loss": _metrics(target, (left + right) / 2.0)["log_loss"],
        })
    diversity = pd.DataFrame(diversity_rows)
    diversity.to_csv(stem.with_name(stem.name + "_diversity.csv"), index=False)

    mean_probability = probabilities.mean(axis=1)
    individual_losses = np.column_stack([_loss(target, probabilities[:, index]) for index in range(probabilities.shape[1])])
    sample = pd.DataFrame({
        "uid": uids,
        "target": target,
        "mean_probability": mean_probability,
        "prediction_std": probabilities.std(axis=1),
        "prediction_min": probabilities.min(axis=1),
        "prediction_max": probabilities.max(axis=1),
        "mean_individual_loss": individual_losses.mean(axis=1),
        "max_individual_loss": individual_losses.max(axis=1),
        "ensemble_log_loss": _loss(target, mean_probability),
    })
    sample["ensemble_better_than_mean_individual"] = sample["ensemble_log_loss"] < sample["mean_individual_loss"]
    sample.to_csv(stem.with_name(stem.name + "_sample_diagnostics.csv"), index=False)
    _write_markdown(output, source, target, probabilities, members, per_model, sizes, diversity, sample)
    print(f"Wrote ensemble analysis report to {output}")
    print(f"Wrote ensemble-size analysis to {stem.with_name(stem.name + '_ensemble_sizes.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
