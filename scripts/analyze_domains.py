"""Analyze acquisition structure and persist deterministic domain assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datscan.training.domains import (
    DEFAULT_DOMAIN_COUNT,
    DEFAULT_DOMAIN_SEED,
    DOMAIN_FEATURES,
    assign_domain_groups_with_config,
    domain_summary,
    geometry_signature_table,
    save_domain_config,
)
from datscan.utils.reporting import markdown_table


def _write_plots(metadata: pd.DataFrame, assignments: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = domain_summary(metadata, assignments)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["domain_group"], summary["sample_count"], color="#4C78A8")
    ax.set_title("Sample count by acquisition domain")
    ax.set_xlabel("Domain group")
    ax.set_ylabel("Scans")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "domain_size.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#F58518" if value >= 0.5 else "#54A24B" for value in summary["pathologic_fraction"]]
    ax.bar(summary["domain_group"], summary["pathologic_fraction"], color=colors)
    ax.axhline(metadata["label"].mean(), color="black", linestyle="--", label="Global prevalence")
    ax.set_title("Pathologic prevalence by acquisition domain")
    ax.set_xlabel("Domain group")
    ax.set_ylabel("Pathologic fraction")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "domain_label_prevalence.png", dpi=150)
    plt.close(fig)

    features = metadata[list(DOMAIN_FEATURES)].apply(pd.to_numeric, errors="coerce")
    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(features))
    points = PCA(n_components=2, random_state=0).fit_transform(scaled)
    plot_frame = assignments.copy()
    plot_frame["pca_x"] = points[:, 0]
    plot_frame["pca_y"] = points[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, group in plot_frame.groupby("domain_group", sort=True):
        ax.scatter(group["pca_x"], group["pca_y"], s=12, alpha=0.65, label=domain)
    ax.set_title("Acquisition metadata PCA colored by domain")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "domain_pca.png", dpi=150)
    plt.close(fig)


def _write_report(
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict,
    output: Path,
) -> None:
    summary = domain_summary(metadata, assignments)
    signatures = geometry_signature_table(metadata)
    lines = [
        "# Acquisition domain analysis",
        "",
        f"- Scans: {len(metadata)}",
        f"- Domain method: `{config['method']}`",
        f"- Feature columns: {', '.join(config['feature_columns'])}",
        "- Labels, model predictions, OOF loss, and leaderboard values were not used to assign domains.",
        "- Intensity percentiles were excluded because they can be pathology-dependent; foreground occupancy was retained as a coarse acquisition/field-of-view descriptor.",
        "",
        "## Acquisition metadata diagnostics",
        "",
    ]
    numeric_columns = [
        column
        for column in (
            "min_intensity",
            "max_intensity",
            "mean_intensity",
            "median_nonzero",
            "p95_nonzero",
            "p99_nonzero",
            "p99_5_nonzero",
            "nonzero_fraction",
        )
        if column in metadata.columns
    ]
    if numeric_columns:
        numeric_summary = metadata[numeric_columns].describe().T.reset_index()
        numeric_summary = numeric_summary.rename(columns={"index": "metadata_column"})
        lines.extend(
            [
                "## Intensity and foreground distributions",
                "",
                markdown_table(numeric_summary),
                "",
            ]
        )
    lines.extend(["## Geometry signature diagnostics", ""])
    for name, table in signatures.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Unique combinations: {len(table)}",
                f"- Largest group: {int(table['sample_count'].max())}",
                f"- Smallest group: {int(table['sample_count'].min())}",
                f"- Median group size: {float(table['sample_count'].median()):.1f}",
                "",
                "Largest combinations:",
                "",
                markdown_table(table.head(15)),
                "",
                "Smallest combinations:",
                "",
                markdown_table(table.tail(min(10, len(table)))),
                "",
            ]
        )
    lines.extend(
        [
            "## Final domain groups",
            "",
            "Rounded geometry signatures were too fragmented for validation: the combined shape/spacing signature has many tiny groups. The final method therefore uses KMeans on standardized geometry and foreground features with a fixed seed and canonicalized cluster IDs.",
            "",
            f"- Requested groups: {config['n_domains_requested']}",
            f"- Fitted groups: {config['n_domains_fitted']}",
            f"- Random seed: {config['random_state']}",
            f"- Group size range: {int(summary['sample_count'].min())}–{int(summary['sample_count'].max())}",
            f"- Median group size: {float(summary['sample_count'].median()):.1f}",
            "",
            markdown_table(summary),
            "",
            "## Relationship to labels",
            "",
            "Labels are shown only as a post-assignment diagnostic; they did not enter clustering.",
            "",
            markdown_table(
                summary.rename(
                    columns={
                        "sample_count": "N",
                        "normal_count": "Normal",
                        "pathologic_count": "Pathologic",
                        "pathologic_fraction": "Pathologic %",
                    }
                )[["domain_group", "N", "Normal", "Pathologic", "Pathologic %"]]
            ),
            "",
            "No domain has pathologic prevalence below 10% or above 90%; the label relationship remains diagnostic only.",
            "",
            "## Artifacts",
            "",
            "- `domain_groups.csv`: one deterministic domain assignment per UID.",
            "- `domain_config.json`: feature columns, imputation/scaling parameters, cluster centers, and seed.",
            "- `artifacts/plots/domains/`: domain size, prevalence, and PCA diagnostics.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config")
    parser.add_argument("--plots-dir", default="artifacts/plots/domains")
    parser.add_argument("--n-domains", type=int, default=DEFAULT_DOMAIN_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_DOMAIN_SEED)
    args = parser.parse_args(argv)

    metadata = pd.read_csv(args.metadata)
    assignments, config = assign_domain_groups_with_config(metadata, args.n_domains, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output, index=False)
    save_domain_config(config, args.config or output.parent / "domain_config.json")
    _write_plots(metadata, assignments, Path(args.plots_dir))
    _write_report(metadata, assignments, config, Path(args.report))

    summary = domain_summary(metadata, assignments)
    print(f"Wrote {len(assignments)} domain assignments to {output}")
    print(f"Created {len(summary)} domains; group sizes: {summary['sample_count'].tolist()}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
