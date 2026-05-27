"""Batch experiment runner: python -m experiments.runner --config configs/nk_escape.yaml"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.common import make_space, collect_optima, write_csv, write_json
from experiments.metrics import escape_rate, otg_lon_metrics, ils_success_rate
from experiments.ablations import fitness_shuffle_ablation
from lib.hill_climb import LocalOptimum
from lib.otg import build_otg


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sweep_instances(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand grid into list of instance specs."""
    if "instances" in cfg:
        return cfg["instances"]
    grid = cfg.get("grid", {})
    out = []
    for item in grid.get("items", []):
        base = {k: v for k, v in item.items() if k != "seeds"}
        seeds = item.get("seeds", [0])
        for seed in seeds:
            spec = {**base, "seed": seed}
            out.append(spec)
    return out


def run_escape(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    for spec in _sweep_instances(cfg):
        t0 = time.time()
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu, "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima = collect_optima(space, inst_cfg)
        row = {
            "type": spec["type"],
            "seed": spec.get("seed", 0),
            "n_optima": len(optima),
            **{k: spec[k] for k in ("n", "k", "nu", "instance") if k in spec},
        }
        for strat in cfg.get("strategies", ["orc", "mingap", "random"]):
            er = escape_rate(
                space,
                optima,
                strat,
                gamma=cfg.get("gamma", 1.0),
                n_random_trials=cfg.get("n_random_trials", 30),
            )
            row[f"escape_{strat}_pct"] = er["escape_pct"]
        if "orc" in cfg.get("strategies", []) and "mingap" in cfg.get("strategies", []):
            row["orc_over_mingap"] = row.get("escape_orc_pct", 0) / max(
                row.get("escape_mingap_pct", 1), 1e-9
            )
        row["elapsed_s"] = time.time() - t0
        rows.append(row)
    write_csv(output_dir / "escape_rates.csv", rows)


def run_otg_lon(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    for spec in _sweep_instances(cfg):
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu, "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima = collect_optima(space, inst_cfg)
        m = otg_lon_metrics(space, optima, gamma=cfg.get("gamma", 1.0))
        row = {"type": spec["type"], "seed": spec.get("seed", 0), **m}
        for k in ("n", "k", "nu", "instance"):
            if k in spec:
                row[k] = spec[k]
        rows.append(row)
    write_csv(output_dir / "otg_lon.csv", rows)


def run_ils(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    algos = cfg.get("algorithms", ["orc_pert", "random", "rrhc", "boltzmann"])
    budget = cfg.get("budget", 5000)
    n_trials = cfg.get("n_trials", 30)
    for spec in _sweep_instances(cfg):
        space = make_space(spec, use_gpu=cfg.get("use_gpu", False))
        row = {"type": spec["type"], "seed": spec.get("seed", 0)}
        for k in ("n", "k", "nu", "instance"):
            if k in spec:
                row[k] = spec[k]
        for algo in algos:
            row[f"success_{algo}_pct"] = ils_success_rate(
                space,
                algo,
                budget=budget,
                n_trials=n_trials,
                seed=spec.get("seed", 0) * 1000,
            )
        rows.append(row)
    write_csv(output_dir / "ils_comparison.csv", rows)


def run_shuffle(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    for spec in _sweep_instances(cfg):
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu, "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima = collect_optima(space, inst_cfg)
        ab = fitness_shuffle_ablation(
            space,
            optima,
            n_shuffles=cfg.get("n_shuffles", 5),
            base_seed=spec.get("seed", 0),
        )
        row = {"type": spec["type"], "seed": spec.get("seed", 0), **ab}
        for k in ("n", "k", "nu", "instance"):
            if k in spec:
                row[k] = spec[k]
        rows.append(row)
    write_csv(output_dir / "fitness_shuffle.csv", rows)


def run_otg_features(cfg: dict[str, Any], output_dir: Path) -> None:
    """Export per-instance OTG features for algorithm selection."""
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    for spec in _sweep_instances(cfg):
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu, "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima = collect_optima(space, inst_cfg)
        otg = build_otg(space, optima, gamma=cfg.get("gamma", 1.0))
        cycle_frac = sum(1 for f in otg.funnels if f.is_cycle) / max(len(otg.funnels), 1)
        kappas = []
        for i in range(len(optima)):
            for v in otg.orc_values.get(i, {}).values():
                kappas.append(v)
        row = {
            "type": spec["type"],
            "seed": spec.get("seed", 0),
            "n_optima": len(optima),
            "otg_compression": otg.compression_ratio,
            "otg_mean_terminal_rank": otg.mean_terminal_rank,
            "otg_top5_reach": otg.top5_reachability,
            "otg_dag_depth": otg.dag_depth,
            "otg_has_cycles": float(otg.has_cycles),
            "otg_cycle_fraction": cycle_frac,
            "mean_orc": float(np.mean(kappas)) if kappas else 0.0,
            "std_orc": float(np.std(kappas)) if kappas else 0.0,
        }
        for k in ("n", "k", "nu"):
            if k in spec:
                row[k] = spec[k]
        rows.append(row)
    write_csv(output_dir / "otg_features.csv", rows)


EXPERIMENTS = {
    "escape": run_escape,
    "otg_lon": run_otg_lon,
    "ils": run_ils,
    "shuffle": run_shuffle,
    "otg_features": run_otg_features,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ORC batch experiments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", cfg)

    for exp_name in cfg.get("experiments", ["escape"]):
        fn = EXPERIMENTS.get(exp_name)
        if fn is None:
            raise ValueError(f"Unknown experiment: {exp_name}")
        print(f"Running {exp_name} -> {output_dir}")
        fn(cfg, output_dir)
        print(f"  done {exp_name}")


if __name__ == "__main__":
    main()
