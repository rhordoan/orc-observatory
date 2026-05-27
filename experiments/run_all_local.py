"""Run all experiment configs sequentially into results/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CONFIGS = [
    ("exp_a_quick.yaml", "quick"),
    ("exp_d_shuffle.yaml", "exp-d"),
    ("exp_e_boltzmann.yaml", "exp-e"),
    ("exp_a_reproduce.yaml", "exp-a"),
    ("exp_b_large_scale.yaml", "exp-b"),
    ("exp_c_structured.yaml", "exp-c"),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for cfg_name, out_name in CONFIGS:
        if only and out_name not in only:
            continue
        cfg = root / "experiments" / "configs" / cfg_name
        out = root / "results" / out_name
        print(f"\n=== {cfg_name} -> {out} ===")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.runner",
                "--config",
                str(cfg),
                "--output",
                str(out),
            ],
            cwd=str(root),
            check=True,
        )
        if (out / "otg_features.csv").exists() and (out / "ils_comparison.csv").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.algorithm_selector",
                    "--features",
                    str(out / "otg_features.csv"),
                    "--ils",
                    str(out / "ils_comparison.csv"),
                    "--output",
                    str(out.parent / "exp-f"),
                ],
                cwd=str(root),
                check=False,
            )


if __name__ == "__main__":
    main()
