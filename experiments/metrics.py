"""Experiment metrics: escape rates, OTG/LON, ILS success."""

from __future__ import annotations

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.hill_climb import LocalOptimum, hill_climb
from lib.orc import compute_all_orc, min_orc_neighbor
from lib.otg import build_otg
from lib.lon import build_lon_d1


def min_gap_neighbor(space: SearchSpace, x: int) -> int:
    """Neighbor with smallest fitness gap |f(y) - f(x)|."""
    nbrs = space.neighbors(x)
    fx = space.fitness(x)
    best = int(nbrs[0])
    best_gap = abs(space.fitness(best) - fx)
    for n in nbrs[1:]:
        g = abs(space.fitness(int(n)) - fx)
        if g < best_gap:
            best_gap = g
            best = int(n)
    return best


def escape_rate(
    space: SearchSpace,
    optima: list[LocalOptimum],
    strategy: str,
    gamma: float = 1.0,
    n_random_trials: int = 30,
    rng: np.random.Generator | None = None,
    use_gpu: bool = False,
    attractor: np.ndarray | None = None,
) -> dict[str, float]:
    """Fraction of optima where one perturbation + HC reaches a better optimum.

    For "orc": uses Algorithm 1 (OTG resolution with curvature-ranked fallback),
    matching the thesis definition of %ORC.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    global_best = max(o.fitness for o in optima)
    n = len(optima)
    if n == 0:
        return {"escape_pct": 0.0, "n_optima": 0}

    if strategy == "orc":
        return _orc_escape_via_otg(space, optima, gamma, global_best,
                                    use_gpu=use_gpu, attractor=attractor)

    successes = 0.0
    for opt in optima:
        if opt.fitness >= global_best - 1e-12:
            continue

        if strategy == "mingap":
            y = min_gap_neighbor(space, opt.idx)
            dest = hill_climb(space, y)
            if space.fitness(dest) > opt.fitness:
                successes += 1
        elif strategy == "random":
            trial_hits = 0
            for _ in range(n_random_trials):
                nbrs = space.neighbors(opt.idx)
                y = int(rng.choice(nbrs))
                dest = hill_climb(space, y)
                if space.fitness(dest) > opt.fitness:
                    trial_hits += 1
            successes += trial_hits / n_random_trials
        else:
            raise ValueError(strategy)

    eligible = sum(1 for o in optima if o.fitness < global_best - 1e-12)
    pct = 100.0 * successes / max(eligible, 1)
    return {
        "escape_pct": pct,
        "n_optima": n,
        "n_eligible": eligible,
        "n_success": int(successes) if isinstance(successes, int) else successes,
    }


def _orc_escape_via_otg(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float,
    global_best: float,
    use_gpu: bool = False,
    attractor: np.ndarray | None = None,
) -> dict[str, float]:
    """ORC escape rate using full OTG resolution (Algorithm 1 with fallback)."""
    otg = build_otg(space, optima, gamma=gamma,
                     use_gpu=use_gpu, attractor=attractor)
    n = len(optima)
    successes = 0
    eligible = 0
    for edge in otg.edges:
        src_fit = optima[edge.source].fitness
        if src_fit >= global_best - 1e-12:
            continue
        eligible += 1
        tgt_fit = optima[edge.target].fitness
        if tgt_fit > src_fit:
            successes += 1
    pct = 100.0 * successes / max(eligible, 1)
    return {
        "escape_pct": pct,
        "n_optima": n,
        "n_eligible": eligible,
        "n_success": successes,
    }


def otg_lon_metrics(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float = 1.0,
    use_gpu: bool = False,
    attractor: np.ndarray | None = None,
) -> dict[str, float]:
    """Structural metrics for OTG and LON-d1."""
    otg = build_otg(space, optima, gamma=gamma,
                     use_gpu=use_gpu, attractor=attractor)
    lon = build_lon_d1(space, optima)
    n = len(optima)
    return {
        "n_optima": n,
        "otg_mean_terminal_rank": otg.mean_terminal_rank,
        "otg_top5_reach": otg.top5_reachability * 100,
        "otg_compression_pct": otg.compression_ratio * 100,
        "otg_dag_depth": otg.dag_depth,
        "otg_has_cycles": float(otg.has_cycles),
        "lon_mean_terminal_rank": _lon_mean_rank(lon, optima),
        "lon_compression_pct": lon.singleton_fraction * 100,
        "lon_self_loops_pct": 100.0 * lon.n_self_loops / max(n, 1),
    }


def _lon_mean_rank(lon, optima: list[LocalOptimum]) -> float:
    n = len(optima)
    if n == 0:
        return 0.5
    fitness_sorted = sorted(range(n), key=lambda i: optima[i].fitness, reverse=True)
    rank_map = {idx: rank for rank, idx in enumerate(fitness_sorted)}
    targets = {e.target for e in lon.edges}
    ranks = [rank_map[t] / max(n - 1, 1) for t in targets]
    return float(np.mean(ranks)) if ranks else 0.5


def ils_success_rate(
    space: SearchSpace,
    algo: str,
    budget: int = 5000,
    n_trials: int = 30,
    seed: int = 0,
    **kwargs,
) -> float:
    """Success rate (finding global optimum under sampling) for one ILS variant."""
    from lib.ils import orc_ils, random_ils, random_restart_hc, mingap_ils, orc_only_ils
    from experiments.boltzmann_ils import boltzmann_orc_ils

    global_opt = max(space.fitness(s) for s in range(space.size))
    successes = 0
    for t in range(n_trials):
        trial_seed = seed + t
        if algo == "orc_pert":
            gen = orc_ils(space, budget=budget, d_r=kwargs.get("d_r", 2), seed=trial_seed)
        elif algo == "random":
            gen = random_ils(space, budget=budget, d_r_total=kwargs.get("d_r_total", 3), seed=trial_seed)
        elif algo == "rrhc":
            gen = random_restart_hc(space, budget=budget, seed=trial_seed)
        elif algo == "orc_only":
            gen = orc_only_ils(space, budget=budget, seed=trial_seed)
        elif algo == "mingap":
            gen = mingap_ils(space, budget=budget, seed=trial_seed)
        elif algo == "boltzmann":
            gen = boltzmann_orc_ils(
                space, budget=budget, d_r=kwargs.get("d_r", 2), seed=trial_seed
            )
        else:
            raise ValueError(algo)
        best = -float("inf")
        for ev in gen:
            best = max(best, ev.best_fitness)
        if best >= global_opt - 1e-9:
            successes += 1
    return 100.0 * successes / n_trials
