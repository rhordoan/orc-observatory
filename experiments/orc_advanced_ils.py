"""Advanced ORC-guided ILS variants for Track 2: continuous navigation.

Three new algorithms that use ORC as a continuous navigation signal
rather than a one-shot escape mechanism:

1. ORC Walk ILS: follow the min-kappa chain for k steps before HC
2. ORC Adaptive ILS: adapt perturbation strength based on curvature magnitude
3. ORC-SA: simulated annealing with curvature-biased proposals
"""

from __future__ import annotations

from typing import Generator

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.orc import compute_all_orc, saddle_orc_neighbor
from lib.ils import CountingSpace, ILSEvent, _hill_climb_counted


def orc_walk_ils(
    space: SearchSpace,
    budget: int = 5000,
    walk_length: int = 3,
    gamma: float = 1.0,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """ORC Walk ILS: follow the min-kappa chain for multiple steps.

    Instead of 1 ORC step + d_r random steps, take walk_length consecutive
    ORC-directed steps (recomputing ORC at each intermediate point).
    This follows the "most negative curvature path" through the landscape.
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="orc_walk", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        current = x_star
        for _ in range(walk_length):
            orc_vals = compute_all_orc(space, current, gamma)
            current = min(orc_vals, key=orc_vals.get)

        x_star = _hill_climb_counted(cs, current)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star

        yield ILSEvent(algo="orc_walk", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def saddle_orc_walk_ils(
    space: SearchSpace,
    budget: int = 5000,
    walk_length: int = 3,
    gamma: float = 1.0,
    keep_frac: float = 0.5,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """Saddle-ORC Walk: multi-step walk using saddle-filtered min-kappa."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="saddle_walk", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        current = x_star
        for _ in range(walk_length):
            current, _ = saddle_orc_neighbor(space, current, gamma, keep_frac)

        x_star = _hill_climb_counted(cs, current)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star

        yield ILSEvent(algo="saddle_walk", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def orc_adaptive_ils(
    space: SearchSpace,
    budget: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """ORC Adaptive ILS: adapt perturbation strength from curvature magnitude.

    Uses the spread of ORC values at the current optimum to decide how
    aggressively to perturb:
    - High |mean_orc| (strongly negative) -> landscape is rugged,
      take a short ORC step (the signal is clear)
    - Low |mean_orc| (near zero) -> landscape is flat/neutral,
      add more random diversification (ORC can't distinguish directions well)

    The random walk length d_r is set to:
      d_r = clamp(round(2 + 3 * (1 - curvature_strength)), 1, 6)
    where curvature_strength = |min_kappa - mean_kappa| / max(std_kappa, eps)
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="orc_adaptive", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        orc_vals = compute_all_orc(space, x_star, gamma)
        kappas = np.array(list(orc_vals.values()), dtype=np.float64)

        min_k = float(kappas.min())
        mean_k = float(kappas.mean())
        std_k = float(kappas.std())

        curvature_strength = abs(min_k - mean_k) / max(std_k, 1e-8)
        signal_clarity = min(curvature_strength / 3.0, 1.0)
        d_r = int(np.clip(round(1 + 4 * (1 - signal_clarity)), 1, 6))

        y_star = min(orc_vals, key=orc_vals.get)
        current = y_star
        for _ in range(d_r):
            nbrs = cs.neighbors(current)
            current = int(rng.choice(nbrs))

        x_star = _hill_climb_counted(cs, current)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star

        yield ILSEvent(algo="orc_adaptive", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def orc_sa(
    space: SearchSpace,
    budget: int = 5000,
    gamma: float = 1.0,
    t_init: float = 2.0,
    t_min: float = 0.01,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """ORC-biased Simulated Annealing: curvature-weighted proposals.

    Standard SA picks a random neighbor uniformly. ORC-SA weights the
    proposal distribution by exp(-beta * kappa), preferring neighbors
    with more negative curvature (bridges to new regions).

    Recomputes ORC every `recompute_interval` steps to amortize cost.
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    current = int(rng.integers(0, cs.size))
    current_fit = cs.fitness(current)
    best, best_fit = current, current_fit

    alpha = (t_min / t_init) ** (1.0 / max(budget - 1, 1))
    temp = t_init
    recompute_interval = max(space.degree // 2, 10)
    step_since_recompute = recompute_interval  # force first computation

    orc_keys: list[int] = []
    orc_probs: np.ndarray = np.array([])

    yield ILSEvent(algo="orc_sa", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=current)

    while cs.eval_count < budget:
        if step_since_recompute >= recompute_interval:
            orc_vals = compute_all_orc(space, current, gamma)
            orc_keys = list(orc_vals.keys())
            kappas = np.array([orc_vals[k] for k in orc_keys], dtype=np.float64)
            beta = 1.0 / max(float(np.std(kappas)), 1e-8)
            logits = -beta * kappas
            logits -= logits.max()
            orc_probs = np.exp(logits)
            orc_probs /= orc_probs.sum()
            step_since_recompute = 0

        candidate = int(rng.choice(orc_keys, p=orc_probs))
        cand_fit = cs.fitness(candidate)
        delta = cand_fit - current_fit

        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-15)):
            current, current_fit = candidate, cand_fit
            step_since_recompute = recompute_interval  # recompute at new position

        if current_fit > best_fit:
            best, best_fit = current, current_fit

        step_since_recompute += 1
        temp *= alpha

        yield ILSEvent(algo="orc_sa", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=current)
