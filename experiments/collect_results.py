"""Aggregate CSV experiment outputs into summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_escape(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    lines = ["% Escape rates (mean over seeds)"]
    group_cols = [c for c in ["type", "n", "k", "nu"] if c in df.columns]
    if group_cols:
        g = df.groupby(group_cols, dropna=False).mean(numeric_only=True)
        lines.append(g.to_string())
    else:
        lines.append(df.describe().to_string())
    return "\n".join(lines)


def summarize_ils(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    succ = [c for c in df.columns if c.startswith("success_")]
    lines = ["ILS success rates (mean %)"]
    group_cols = [c for c in ["type", "n", "k", "nu"] if c in df.columns]
    if group_cols:
        g = df.groupby(group_cols)[succ].mean()
        lines.append(g.to_string())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out_lines = []
    rd = args.results_dir
    if (rd / "escape_rates.csv").exists():
        out_lines.append(summarize_escape(rd / "escape_rates.csv"))
    if (rd / "ils_comparison.csv").exists():
        out_lines.append(summarize_ils(rd / "ils_comparison.csv"))
    if (rd / "fitness_shuffle.csv").exists():
        df = pd.read_csv(rd / "fitness_shuffle.csv")
        out_lines.append("Fitness shuffle ablation:\n" + df.describe().to_string())

    text = "\n\n".join(out_lines)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
