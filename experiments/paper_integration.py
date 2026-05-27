"""Generate LaTeX table fragments from experiment CSVs for revised paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def escape_table(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    group = [c for c in ["n", "k", "nu"] if c in df.columns]
    if not group:
        out_path.write_text("% No grouping columns\n", encoding="utf-8")
        return
    agg = df.groupby(group).agg({
        "escape_orc_pct": "mean",
        "escape_mingap_pct": "mean",
        "escape_random_pct": "mean",
        "n_optima": "mean",
    }).reset_index()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{ORC escape rates (batch reproduction).}",
        "\\begin{tabular}{l rrr r}",
        "\\toprule",
        "Config & \\%ORC & \\%MinGap & \\%Rand & \\#Opt \\\\",
        "\\midrule",
    ]
    for _, r in agg.iterrows():
        label = " ".join(f"{c}={r[c]}" for c in group)
        lines.append(
            f"{label} & {r['escape_orc_pct']:.1f} & {r['escape_mingap_pct']:.1f} "
            f"& {r['escape_random_pct']:.1f} & {int(r['n_optima'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def shuffle_table(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Fitness-shuffle ablation: real vs permuted fitness ORC escape.}",
        "\\begin{tabular}{l rr r}",
        "\\toprule",
        "Instance & Real \\% & Shuffled \\% & Ratio \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"seed {int(r['seed'])} & {r['orc_escape_real_pct']:.1f} & "
            f"{r['orc_escape_shuffled_mean_pct']:.1f} & {r['real_over_shuffled_ratio']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.results_dir / "escape_rates.csv").exists():
        escape_table(args.results_dir / "escape_rates.csv", args.output_dir / "tab_escape.tex")
    if (args.results_dir / "fitness_shuffle.csv").exists():
        shuffle_table(args.results_dir / "fitness_shuffle.csv", args.output_dir / "tab_shuffle.tex")
    print(f"Wrote LaTeX fragments to {args.output_dir}")


if __name__ == "__main__":
    main()
