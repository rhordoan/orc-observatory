"""Batch experiment runner: python -m experiments.runner --config configs/xxx.yaml"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.common import make_space, collect_optima, write_csv, write_json
from experiments.metrics import (
    unified_escape_rate, compute_statistical_tests,
    otg_lon_metrics, ils_performance, time_orc_computation,
)
from lib.hill_climb import LocalOptimum
from lib.otg import build_otg


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sweep_instances(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "instances" in cfg:
        return cfg["instances"]
    grid = cfg.get("grid", {})
    out = []
    for item in grid.get("items", []):
        base = {k: v for k, v in item.items() if k != "seeds"}
        for seed in item.get("seeds", [0]):
            out.append({**base, "seed": seed})
    return out


def _collect_with_attractor(space, inst_cfg):
    result = collect_optima(space, inst_cfg, return_attractor=True)
    if isinstance(result, tuple):
        return result
    return result, None


def _instance_key(spec: dict) -> dict:
    row = {"type": spec["type"], "seed": spec.get("seed", 0)}
    for k in ("n", "k", "nu", "instance"):
        if k in spec:
            row[k] = spec[k]
    return row


# ── RQ1+RQ2: Unified escape + shuffle ───────────────────────────────

def run_escape(cfg: dict[str, Any], output_dir: Path) -> None:
    """Unified escape rate: ORC, MinGap, MaxGap, Steepest, Random, Shuffled.
    All measured on the SAME optima with the SAME methodology."""
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    instances = _sweep_instances(cfg)
    for ii, spec in enumerate(instances):
        t0 = time.time()
        label = spec.get("instance", spec.get("type", "?"))
        print(f"  escape [{ii+1}/{len(instances)}] {label} seed={spec.get('seed',0)} ...", flush=True)
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu,
                    "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima, attractor = _collect_with_attractor(space, inst_cfg)

        er = unified_escape_rate(
            space, optima,
            gamma=cfg.get("gamma", 1.0),
            n_random_trials=cfg.get("n_random_trials", 30),
            n_shuffles=cfg.get("n_shuffles", 5),
            seed=spec.get("seed", 0),
            use_gpu=use_gpu,
            attractor=attractor,
        )

        row = {**_instance_key(spec), **{k: v for k, v in er.items() if not k.startswith("_")}}
        row["elapsed_s"] = time.time() - t0
        rows.append(row)

    # Statistical tests across all instances
    stats = compute_statistical_tests(rows)
    write_json(output_dir / "statistical_tests.json", stats)
    write_csv(output_dir / "escape_rates.csv", rows)


# ── OTG / LON structural metrics ────────────────────────────────────

def run_otg_lon(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    instances = _sweep_instances(cfg)
    for ii, spec in enumerate(instances):
        label = spec.get("instance", spec.get("type", "?"))
        print(f"  otg_lon [{ii+1}/{len(instances)}] {label} seed={spec.get('seed',0)} ...", flush=True)
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu,
                    "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima, attractor = _collect_with_attractor(space, inst_cfg)
        m = otg_lon_metrics(space, optima, gamma=cfg.get("gamma", 1.0),
                            use_gpu=use_gpu, attractor=attractor)
        rows.append({**_instance_key(spec), **m})
    write_csv(output_dir / "otg_lon.csv", rows)


# ── ILS portfolio comparison ─────────────────────────────────────────

def run_ils(cfg: dict[str, Any], output_dir: Path) -> None:
    algos = cfg.get("algorithms", ["orc_pert", "random", "rrhc", "boltzmann",
                                    "mingap", "sa", "tabu", "ea11", "vns"])
    budgets = cfg.get("budgets", [cfg.get("budget", 5000)])
    if isinstance(budgets, int):
        budgets = [budgets]
    n_trials = cfg.get("n_trials", 50)
    instances = _sweep_instances(cfg)

    for budget in budgets:
        rows = []
        all_trials: dict[str, list[list[float]]] = {a: [] for a in algos}
        for ii, spec in enumerate(instances):
            label = spec.get("instance", spec.get("type", "?"))
            print(f"  ils(b={budget}) [{ii+1}/{len(instances)}] {label} seed={spec.get('seed',0)} ...", flush=True)
            space = make_space(spec, use_gpu=cfg.get("use_gpu", False))
            row = _instance_key(spec)
            for algo in algos:
                perf = ils_performance(
                    space, algo, budget=budget, n_trials=n_trials,
                    seed=spec.get("seed", 0) * 1000,
                )
                row[f"perf_{algo}"] = perf["mean"]
                row[f"std_{algo}"] = perf["std"]
                all_trials[algo].append(perf["trials"])
            rows.append(row)

        suffix = f"_b{budget}" if len(budgets) > 1 else ""
        write_csv(output_dir / f"ils_comparison{suffix}.csv", rows)

        # Friedman test across algorithms
        _friedman_test(rows, algos, output_dir / f"friedman{suffix}.json")


def _friedman_test(rows: list[dict], algos: list[str], out_path: Path) -> None:
    """Friedman test + pairwise Wilcoxon with Holm correction."""
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except ImportError:
        return

    perf_matrix = np.array([[r[f"perf_{a}"] for a in algos] for r in rows])
    n_inst, n_algo = perf_matrix.shape
    if n_inst < 10 or n_algo < 3:
        return

    # Rank per instance (higher perf = rank 1)
    ranks = np.zeros_like(perf_matrix)
    for i in range(n_inst):
        order = np.argsort(-perf_matrix[i])
        for rank, idx in enumerate(order):
            ranks[i, idx] = rank + 1
    mean_ranks = ranks.mean(axis=0)

    try:
        stat, p = friedmanchisquare(*[perf_matrix[:, j] for j in range(n_algo)])
    except Exception:
        stat, p = 0.0, 1.0

    # Pairwise Wilcoxon (ORC vs each)
    orc_idx = algos.index("orc_pert") if "orc_pert" in algos else 0
    pairwise = {}
    for j, algo in enumerate(algos):
        if j == orc_idx:
            continue
        diff = perf_matrix[:, orc_idx] - perf_matrix[:, j]
        diff = diff[np.abs(diff) > 1e-12]
        if len(diff) < 10:
            pairwise[algo] = {"p": float("nan"), "direction": "insufficient"}
            continue
        try:
            w_stat, w_p = wilcoxon(diff)
            direction = "orc_better" if np.mean(diff) > 0 else "baseline_better"
            pairwise[algo] = {"p": float(w_p), "direction": direction}
        except Exception:
            pairwise[algo] = {"p": float("nan"), "direction": "error"}

    result = {
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "mean_ranks": {a: float(r) for a, r in zip(algos, mean_ranks)},
        "pairwise_vs_orc": pairwise,
    }
    write_json(out_path, result)


# ── OTG + FLA features for algorithm selection ───────────────────────

def run_otg_features(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    fla_rows = []
    use_gpu = cfg.get("use_gpu", False)
    instances = _sweep_instances(cfg)
    for ii, spec in enumerate(instances):
        label = spec.get("instance", spec.get("type", "?"))
        print(f"  otg_features [{ii+1}/{len(instances)}] {label} seed={spec.get('seed',0)} ...", flush=True)
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu,
                    "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima, attractor = _collect_with_attractor(space, inst_cfg)
        otg = build_otg(space, optima, gamma=cfg.get("gamma", 1.0),
                        attractor=attractor, use_gpu=use_gpu)

        cycle_frac = sum(1 for f in otg.funnels if f.is_cycle) / max(len(otg.funnels), 1)
        kappas = []
        for i in range(len(optima)):
            for v in otg.orc_values.get(i, {}).values():
                kappas.append(v)

        row = {
            **_instance_key(spec),
            "n_optima": len(optima),
            "degree": space.degree,
            "otg_compression": otg.compression_ratio,
            "otg_mean_terminal_rank": otg.mean_terminal_rank,
            "otg_top5_reach": otg.top5_reachability,
            "otg_dag_depth": otg.dag_depth,
            "otg_has_cycles": float(otg.has_cycles),
            "otg_cycle_fraction": cycle_frac,
            "mean_orc": float(np.mean(kappas)) if kappas else 0.0,
            "std_orc": float(np.std(kappas)) if kappas else 0.0,
        }

        from experiments.fla_features import compute_fla_features
        fla = compute_fla_features(space, optima, seed=spec.get("seed", 0))
        fla_row = {**_instance_key(spec), **fla}
        row.update(fla)

        rows.append(row)
        fla_rows.append(fla_row)
    write_csv(output_dir / "otg_features.csv", rows)
    write_csv(output_dir / "fla_features.csv", fla_rows)


# ── RQ5: Scalability ────────────────────────────────────────────────

def run_scalability(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = []
    use_gpu = cfg.get("use_gpu", False)
    instances = _sweep_instances(cfg)
    for ii, spec in enumerate(instances):
        label = spec.get("instance", spec.get("type", "?"))
        print(f"  scalability [{ii+1}/{len(instances)}] {label} ...", flush=True)
        space = make_space(spec, use_gpu=use_gpu)
        inst_cfg = {**spec, "use_gpu": use_gpu,
                    "optima_mode": spec.get("optima_mode", cfg.get("optima_mode", "enumerate"))}
        optima, _ = _collect_with_attractor(space, inst_cfg)

        t0 = time.perf_counter()
        timing = time_orc_computation(space, optima)
        otg_t0 = time.perf_counter()
        build_otg(space, optima, gamma=cfg.get("gamma", 1.0), use_gpu=use_gpu)
        otg_time = time.perf_counter() - otg_t0

        row = {
            **_instance_key(spec),
            "n_optima": len(optima),
            "degree": space.degree,
            "size": space.size,
            "orc_per_optimum_ms": timing["per_optimum_ms"],
            "orc_total_50_s": timing["mean_total_s"],
            "otg_build_s": otg_time,
        }
        rows.append(row)
    write_csv(output_dir / "scalability.csv", rows)


# ── Dispatch ─────────────────────────────────────────────────────────

EXPERIMENTS = {
    "escape": run_escape,
    "otg_lon": run_otg_lon,
    "ils": run_ils,
    "otg_features": run_otg_features,
    "scalability": run_scalability,
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
