"""Small dependency-free Markdown reporting helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a simple Markdown table without tabulate."""

    if frame.empty:
        return "| (no rows) |\n| --- |"
    columns = [str(column) for column in frame.columns]

    def render(value: Any, column: str) -> str:
        if pd.isna(value):
            return "NaN"
        if pd.api.types.is_integer_dtype(frame[column]):
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[column]) else "---" for column in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[column], column) for column in frame.columns) + " |")
    return "\n".join(lines)
