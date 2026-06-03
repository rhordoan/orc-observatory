"""Experiment metrics: escape rates, OTG/LON, ILS performance.

All escape-related measurements use the same unified pipeline to
guarantee identical methodology across real/shuffled/baseline conditions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.hill_climb import LocalOptimum, hill_climb
from lib.orc import compute_all_orc
from lib.otg import build_otg
from lib.lon import build_lon_d1


# ── Escape strategy helpers ──────────────────────────────────────────

def _orc_direction(orc_vals: dict[int, float]) -> int:
    """Most-negative ORC neighbor."""
    return min(orc_vals, key=orc_vals.get)


def _mingap_direction(space: SearchSpace, x: int) -> int:
    """Neighbor with smallest |f(y) - f(x)|."""
    nbrs = space.neighbors(x)
    fx = space.fitness(x)
    best, best_gap = int(nbrs[0]), abs(space.fitness(int(nbrs[0])) - fx)
    for n in nbrs[1:]:
        g = abs(space.fitness(int(n)) - fx)
        if g < best_gap:
            best_gap, best = g, int(n)
    return best


def _maxgap_direction(space: SearchSpace, x: int) -> int:
    """Neighbor with largest |f(y) - f(x)| (opposite of MinGap)."""
    nbrs = space.neighbors(x)
    fx = space.fitness(x)
    best, best_gap = int(nbrs[0]), abs(space.fitness(int(nbrs[0])) - fx)
    for n in nbrs[1:]:
        g = abs(space.fitness(int(n)) - fx)
        if g > best_gap:
            best_gap, best = g, int(n)
    return best


def _steepest_direction(space: SearchSpace, x: int) -> int:
    """Neighbor with highest fitness (greedy ascent)."""
    nbrs = space.neighbors(x)
    best, best_f = int(nbrs[0]), space.fitness(int(nbrs[0]))
    for n in nbrs[1:]:
        f = space.fitness(int(n))
        if f > best_f:
            best_f, best = f, int(n)
    return best


# ── Precompute ORC ───────────────────────────────────────────────────

def _precompute_orc(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float,
    use_gpu: bool = False,
    attractor: np.ndarray | None = None,
    return_topology: bool = False,
    topology: dict | None = None,
) -> dict[int, dict[int, float]] | tuple:
    from lib.orc import compute_all_orc, batch_orc_gpu, batch_orc_reuse_topology

    max_nbrs = 60 if space.degree > 100 else None
    mem_budget = len(optima) * (max_nbrs or space.degree) * space.degree
    safe_for_batch = mem_budget < 200_000_000

    if space.degree > 30 and hasattr(space, "neighbor_table") and safe_for_batch:
        idx_arr = np.array([o.idx for o in optima], dtype=np.int64)
        mn = max_nbrs or space.degree

        if topology is not None:
            return batch_orc_reuse_topology(space, idx_arr, gamma, topology)

        return batch_orc_gpu(
            space, idx_arr, gamma,
            max_neighbors=mn,
            return_topology=return_topology,
        )
    orc_values: dict[int, dict[int, float]] = {}
    for i, opt in enumerate(optima):
        orc_values[i] = compute_all_orc(space, opt.idx, gamma, max_neighbors=max_nbrs)
    if return_topology:
        return orc_values, None
    return orc_values


# ── Per-optimum escape result ────────────────────────────────────────

@dataclass
class EscapeResult:
    """Result for a single optimum under a single strategy."""
    escaped: bool
    fitness_improvement: float  # (f(dest) - f(x*)) / (f_global - f(x*))


# ── Unified escape measurement ───────────────────────────────────────

def _measure_escape_for_optimum(
    real_space: SearchSpace,
    opt: LocalOptimum,
    neighbor_idx: int,
    global_best: float,
) -> EscapeResult:
    """Hill-climb from neighbor_idx in real_space, check real fitness improvement."""
    dest = hill_climb(real_space, neighbor_idx)
    dest_fit = real_space.fitness(dest)
    improved = dest_fit > opt.fitness + 1e-12
    gap = global_best - opt.fitness
    norm_improvement = (dest_fit - opt.fitness) / gap if gap > 1e-12 else 0.0
    return EscapeResult(escaped=improved, fitness_improvement=norm_improvement)


def unified_escape_rate(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float = 1.0,
    n_random_trials: int = 30,
    n_shuffles: int = 5,
    seed: int = 0,
    use_gpu: bool = False,
    attractor: np.ndarray | None = None,
) -> dict[str, Any]:
    """Unified escape measurement: all strategies use the SAME optima,
    SAME hill_climb, SAME fitness check.

    Strategies: orc, mingap, maxgap, steepest, random, shuffled_orc (x n_shuffles).
    Returns per-strategy: escape_pct, mean_improvement, per_optimum results.
    """
    rng = np.random.default_rng(seed)
    global_best = max(o.fitness for o in optima)

    eligible_optima = [(i, opt) for i, opt in enumerate(optima)
                       if opt.fitness < global_best - 1e-12]
    n_eligible = len(eligible_optima)
    if n_eligible == 0:
        return {"n_optima": len(optima), "n_eligible": 0}

    # Precompute real ORC (with topology for structured spaces)
    is_structured = space.degree > 30 and hasattr(space, "neighbor_table")
    if is_structured:
        real_orc, topology = _precompute_orc(
            space, optima, gamma, use_gpu, attractor, return_topology=True)
    else:
        real_orc = _precompute_orc(space, optima, gamma, use_gpu, attractor)
        topology = None

    # Shuffled ablation on dynamic spaces requires pre-cached neighbor arrays.
    # Only skip for instances where pre-caching would exceed ~80GB (e.g. ch130).
    if is_structured:
        n_unique_est = len(optima) * min(60, space.degree)
        sol_size = getattr(space, '_n', getattr(space, '_n_jobs', space.degree))
        est_bytes = n_unique_est * space.degree * sol_size * 16
        effective_n_shuffles = 0 if est_bytes > 80 * (1024 ** 3) else n_shuffles
    else:
        effective_n_shuffles = n_shuffles

    # Precompute shuffled ORCs (reuse topology when available)
    from experiments.ablations import ShuffledFitnessSpace, make_shuffle_perm
    shuffled_orcs = []
    for s in range(effective_n_shuffles):
        perm = make_shuffle_perm(space, seed + s + 1)
        shuf_space = ShuffledFitnessSpace(space, perm)
        shuf_orc = _precompute_orc(
            shuf_space, optima, gamma, use_gpu=False, topology=topology)
        shuffled_orcs.append(shuf_orc)

    strategies = ["orc", "mingap", "maxgap", "steepest", "random"]
    for s in range(effective_n_shuffles):
        strategies.append(f"shuffled_{s}")

    results_per_opt: dict[str, list[EscapeResult]] = {s: [] for s in strategies}

    for i, opt in eligible_optima:
        # ORC direction
        orc_nbr = _orc_direction(real_orc[i])
        results_per_opt["orc"].append(
            _measure_escape_for_optimum(space, opt, orc_nbr, global_best))

        # MinGap direction
        mg_nbr = _mingap_direction(space, opt.idx)
        results_per_opt["mingap"].append(
            _measure_escape_for_optimum(space, opt, mg_nbr, global_best))

        # MaxGap direction
        mxg_nbr = _maxgap_direction(space, opt.idx)
        results_per_opt["maxgap"].append(
            _measure_escape_for_optimum(space, opt, mxg_nbr, global_best))

        # Steepest-ascent direction
        sa_nbr = _steepest_direction(space, opt.idx)
        results_per_opt["steepest"].append(
            _measure_escape_for_optimum(space, opt, sa_nbr, global_best))

        # Random direction (average over n_random_trials)
        nbrs = space.neighbors(opt.idx)
        random_escaped = 0
        random_improvement = 0.0
        for _ in range(n_random_trials):
            y = int(rng.choice(nbrs))
            r = _measure_escape_for_optimum(space, opt, y, global_best)
            random_escaped += r.escaped
            random_improvement += r.fitness_improvement
        results_per_opt["random"].append(EscapeResult(
            escaped=random_escaped / n_random_trials > 0.5,
            fitness_improvement=random_improvement / n_random_trials,
        ))

        # Shuffled ORC directions
        for s in range(effective_n_shuffles):
            shuf_nbr = _orc_direction(shuffled_orcs[s][i])
            results_per_opt[f"shuffled_{s}"].append(
                _measure_escape_for_optimum(space, opt, shuf_nbr, global_best))

    # Aggregate
    out: dict[str, Any] = {
        "n_optima": len(optima),
        "n_eligible": n_eligible,
    }

    for strat in ["orc", "mingap", "maxgap", "steepest", "random"]:
        res = results_per_opt[strat]
        esc_pct = 100.0 * sum(r.escaped for r in res) / n_eligible
        mean_imp = float(np.mean([r.fitness_improvement for r in res]))
        out[f"escape_{strat}_pct"] = esc_pct
        out[f"improvement_{strat}"] = mean_imp

    # Shuffled: aggregate across all shuffle seeds
    shuf_esc_per_seed = []
    shuf_imp_per_seed = []
    for s in range(effective_n_shuffles):
        res = results_per_opt[f"shuffled_{s}"]
        shuf_esc_per_seed.append(100.0 * sum(r.escaped for r in res) / n_eligible)
        shuf_imp_per_seed.append(float(np.mean([r.fitness_improvement for r in res])))
    if effective_n_shuffles:
        out["escape_shuffled_mean_pct"] = float(np.mean(shuf_esc_per_seed))
        out["escape_shuffled_std_pct"] = float(np.std(shuf_esc_per_seed))
        out["improvement_shuffled_mean"] = float(np.mean(shuf_imp_per_seed))
    else:
        out["escape_shuffled_mean_pct"] = float("nan")
        out["escape_shuffled_std_pct"] = float("nan")
        out["improvement_shuffled_mean"] = float("nan")
    out["n_shuffles"] = effective_n_shuffles

    # Ratios
    out["orc_over_random"] = out["escape_orc_pct"] / max(out["escape_random_pct"], 1e-9)
    out["orc_over_shuffled"] = out["escape_orc_pct"] / max(out["escape_shuffled_mean_pct"], 1e-9)
    out["orc_over_mingap"] = out["escape_orc_pct"] / max(out["escape_mingap_pct"], 1e-9)

    # Per-optimum arrays for statistical tests
    out["_per_opt_orc"] = [r.escaped for r in results_per_opt["orc"]]
    out["_per_opt_mingap"] = [r.escaped for r in results_per_opt["mingap"]]
    out["_per_opt_random"] = [r.escaped for r in results_per_opt["random"]]
    out["_per_opt_steepest"] = [r.escaped for r in results_per_opt["steepest"]]

    return out


def compute_statistical_tests(rows: list[dict]) -> dict[str, Any]:
    """Wilcoxon signed-rank tests and A12 effect size across instances."""
    from scipy.stats import wilcoxon

    orc_escs = np.array([r["escape_orc_pct"] for r in rows])
    mg_escs = np.array([r["escape_mingap_pct"] for r in rows])
    rand_escs = np.array([r["escape_random_pct"] for r in rows])
    shuf_escs = np.array([r["escape_shuffled_mean_pct"] for r in rows])
    steep_escs = np.array([r["escape_steepest_pct"] for r in rows])
    mxg_escs = np.array([r["escape_maxgap_pct"] for r in rows])

    def safe_wilcoxon(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        diff = a - b
        diff = diff[np.abs(diff) > 1e-12]
        if len(diff) < 10:
            return float("nan")
        try:
            return wilcoxon(diff, alternative="greater").pvalue
        except Exception:
            return float("nan")

    def a12(a, b):
        """Vargha-Delaney A12: P(a > b) + 0.5 * P(a == b)."""
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        n = len(a)
        if n == 0:
            return float("nan")
        wins = sum((ai > bi) + 0.5 * (ai == bi) for ai, bi in zip(a, b))
        return wins / n

    tests = {}
    for name, baseline in [("mingap", mg_escs), ("random", rand_escs),
                           ("shuffled", shuf_escs), ("steepest", steep_escs),
                           ("maxgap", mxg_escs)]:
        tests[f"wilcoxon_orc_vs_{name}_p"] = safe_wilcoxon(orc_escs, baseline)
        tests[f"a12_orc_vs_{name}"] = a12(orc_escs, baseline)

    return tests


# ── OTG / LON structural metrics ────────────────────────────────────

def otg_lon_metrics(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float = 1.0,
    use_gpu: bool = False,
    attractor: np.ndarray | None = None,
) -> dict[str, float]:
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


# ── ILS performance ─────────────────────────────────────────────────

def _make_ils_generator(space, algo, budget, trial_seed, kwargs):
    from lib.ils import (orc_ils, random_ils, random_restart_hc,
                         mingap_ils, orc_only_ils, simulated_annealing,
                         tabu_search, one_plus_one_ea,
                         variable_neighborhood_search)
    from experiments.boltzmann_ils import boltzmann_orc_ils
    from experiments.orc_advanced_ils import orc_walk_ils, orc_adaptive_ils, orc_sa

    dispatch = {
        "orc_pert": lambda: orc_ils(space, budget=budget, d_r=kwargs.get("d_r", 2), seed=trial_seed),
        "random": lambda: random_ils(space, budget=budget, d_r_total=kwargs.get("d_r_total", 3), seed=trial_seed),
        "rrhc": lambda: random_restart_hc(space, budget=budget, seed=trial_seed),
        "orc_only": lambda: orc_only_ils(space, budget=budget, seed=trial_seed),
        "mingap": lambda: mingap_ils(space, budget=budget, seed=trial_seed),
        "boltzmann": lambda: boltzmann_orc_ils(space, budget=budget, d_r=kwargs.get("d_r", 2), seed=trial_seed),
        "sa": lambda: simulated_annealing(space, budget=budget, seed=trial_seed),
        "tabu": lambda: tabu_search(space, budget=budget, seed=trial_seed),
        "ea11": lambda: one_plus_one_ea(space, budget=budget, seed=trial_seed),
        "vns": lambda: variable_neighborhood_search(space, budget=budget, seed=trial_seed),
        "orc_walk": lambda: orc_walk_ils(space, budget=budget, walk_length=kwargs.get("walk_length", 3), seed=trial_seed),
        "orc_adaptive": lambda: orc_adaptive_ils(space, budget=budget, seed=trial_seed),
        "orc_sa": lambda: orc_sa(space, budget=budget, seed=trial_seed),
    }
    if algo not in dispatch:
        raise ValueError(f"Unknown algorithm: {algo}")
    return dispatch[algo]()


def _run_single_ils_trial(args: tuple) -> float:
    space, algo, budget, trial_seed, kwargs = args
    gen = _make_ils_generator(space, algo, budget, trial_seed, kwargs)
    best = -float("inf")
    for ev in gen:
        best = max(best, ev.best_fitness)
    return best


def _run_single_ils_trial_curve(args: tuple) -> list[tuple[int, float]]:
    """Run one ILS trial; returns convergence curve [(evals, best_fitness), ...]."""
    space, algo, budget, trial_seed, kwargs = args
    gen = _make_ils_generator(space, algo, budget, trial_seed, kwargs)
    curve = []
    for ev in gen:
        curve.append((ev.evals, ev.best_fitness))
    return curve


def ils_performance(
    space: SearchSpace,
    algo: str,
    budget: int = 2000,
    n_trials: int = 50,
    seed: int = 0,
    **kwargs,
) -> dict[str, float]:
    """Mean best fitness + per-trial results for statistical tests."""
    from concurrent.futures import ThreadPoolExecutor
    import os

    trial_args = [
        (space, algo, budget, seed + t, kwargs)
        for t in range(n_trials)
    ]

    is_stateful = hasattr(space, "neighbor_table")
    if is_stateful:
        best_fitnesses = [_run_single_ils_trial(a) for a in trial_args]
    else:
        n_workers = min(os.cpu_count() or 4, n_trials, 64)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            best_fitnesses = list(pool.map(_run_single_ils_trial, trial_args))

    return {
        "mean": float(np.mean(best_fitnesses)),
        "std": float(np.std(best_fitnesses)),
        "median": float(np.median(best_fitnesses)),
        "trials": best_fitnesses,
    }


def ils_convergence(
    space: SearchSpace,
    algo: str,
    budget: int = 5000,
    n_trials: int = 50,
    seed: int = 0,
    **kwargs,
) -> list[list[tuple[int, float]]]:
    """Collect convergence curves for all trials."""
    trial_args = [
        (space, algo, budget, seed + t, kwargs)
        for t in range(n_trials)
    ]
    return [_run_single_ils_trial_curve(a) for a in trial_args]


# ── Scalability timing ──────────────────────────────────────────────

def time_orc_computation(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float = 1.0,
    n_repeats: int = 3,
) -> dict[str, float]:
    """Time ORC computation for scalability analysis."""
    from lib.orc import batch_orc_gpu

    is_structured = space.degree > 30 and hasattr(space, "neighbor_table")
    n_computed = min(50, len(optima))
    idx_arr = np.array([o.idx for o in optima[:n_computed]], dtype=np.int64)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        if is_structured:
            batch_orc_gpu(space, idx_arr, gamma,
                          max_neighbors=min(60, space.degree))
        else:
            for opt in optima[:n_computed]:
                compute_all_orc(space, opt.idx, gamma)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {
        "mean_total_s": float(np.mean(times)),
        "per_optimum_ms": float(np.mean(times)) / max(n_computed, 1) * 1000,
        "degree": space.degree,
        "n_optima_timed": n_computed,
    }
