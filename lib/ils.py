"""Iterated Local Search variants for algorithm comparison.

Implements Algorithm 2 (ORC-Guided ILS) from the thesis alongside
random-perturbation ILS and random-restart hill climbing baselines.
All three are generators that yield per-iteration events for streaming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

from .search_spaces.protocol import SearchSpace
from .orc import compute_all_orc, min_orc_neighbor, saddle_orc_neighbor


@dataclass
class ILSEvent:
    """One iteration snapshot emitted by an ILS generator."""

    algo: str
    evals: int
    best_fitness: float
    current_optimum: int

    def __post_init__(self):
        self.evals = int(self.evals)
        self.best_fitness = float(self.best_fitness)
        self.current_optimum = int(self.current_optimum)


class CountingSpace:
    """Wraps a SearchSpace to count fitness evaluations honestly."""

    def __init__(self, space: SearchSpace) -> None:
        self._space = space
        self.eval_count = 0

    @property
    def name(self) -> str:
        return self._space.name

    @property
    def size(self) -> int:
        return self._space.size

    @property
    def degree(self) -> int:
        return self._space.degree

    @property
    def fitnesses(self):
        return getattr(self._space, "fitnesses", None)

    def fitness(self, idx: int) -> float:
        self.eval_count += 1
        return self._space.fitness(idx)

    def neighbors(self, idx: int) -> np.ndarray:
        return self._space.neighbors(idx)

    def solution_label(self, idx: int) -> str:
        return self._space.solution_label(idx)


_BITFLIP_MASKS: dict[int, np.ndarray] = {}


def _get_masks(n_bits: int) -> np.ndarray:
    """Cached one-hot bit masks for vectorized neighbor generation."""
    if n_bits not in _BITFLIP_MASKS:
        _BITFLIP_MASKS[n_bits] = np.int64(1) << np.arange(n_bits, dtype=np.int64)
    return _BITFLIP_MASKS[n_bits]


def _hill_climb_vec(fitnesses: np.ndarray, n_bits: int, start: int) -> tuple[int, int]:
    """Vectorized best-improvement hill climbing. Returns (optimum, n_evals)."""
    masks = _get_masks(n_bits)
    current = np.int64(start)
    current_fit = fitnesses[current]
    evals = 1
    while True:
        neighbors = current ^ masks
        fits = fitnesses[neighbors]
        evals += n_bits
        best_idx = fits.argmax()
        if fits[best_idx] <= current_fit:
            return int(current), evals
        current = neighbors[best_idx]
        current_fit = fits[best_idx]


def _hill_climb_counted(cs: CountingSpace, start: int) -> int:
    """Best-improvement hill climbing using the counting wrapper.

    Uses vectorized NumPy path for bit-flip spaces; delta-eval
    hill_climb_from for TSP/QAP; falls back to scalar loop otherwise.
    """
    if not hasattr(cs._space, "neighbor_table"):
        f = cs.fitnesses
        if f is not None:
            opt, n_evals = _hill_climb_vec(f, cs.degree, start)
            cs.eval_count += n_evals
            return opt
    hcf = getattr(cs._space, "hill_climb_from", None)
    if hcf is not None:
        opt = hcf(start)
        cs.eval_count += cs.degree * 5
        return opt
    current = start
    while True:
        nbrs = cs.neighbors(current)
        current_fit = cs.fitness(current)
        best_nbr = -1
        best_fit = current_fit
        for n in nbrs:
            f = cs.fitness(n)
            if f > best_fit:
                best_fit = f
                best_nbr = n
        if best_nbr == -1:
            return current
        current = best_nbr


def orc_ils(
    space: SearchSpace,
    budget: int = 5000,
    d_r: int = 2,
    gamma: float = 1.0,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """ORC-Guided ILS (Algorithm 2).

    Perturbation: 1 ORC-directed move (min-kappa neighbor) + d_r random moves.
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="orc", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        orc_vals = compute_all_orc(space, x_star, gamma)
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

        yield ILSEvent(algo="orc", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def saddle_orc_ils(
    space: SearchSpace,
    budget: int = 5000,
    d_r: int = 2,
    gamma: float = 1.0,
    keep_frac: float = 0.5,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """Saddle-ORC ILS: pre-filter to saddle zone, then min-kappa + random walk.

    Fixes the curvature-gap bias where raw min-kappa selects deep-basin
    neighbors. Pre-filters to the closest-fitness fraction of the
    neighborhood before ORC selection.
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="saddle_orc", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        y_star, _ = saddle_orc_neighbor(space, x_star, gamma, keep_frac)

        current = y_star
        for _ in range(d_r):
            nbrs = cs.neighbors(current)
            current = int(rng.choice(nbrs))

        x_star = _hill_climb_counted(cs, current)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star

        yield ILSEvent(algo="saddle_orc", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def random_ils(
    space: SearchSpace,
    budget: int = 5000,
    d_r_total: int = 3,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """Random-perturbation ILS baseline.

    Perturbation: d_r_total random neighbor moves (matched budget with ORC+Pert).
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="random", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        current = x_star
        for _ in range(d_r_total):
            nbrs = cs.neighbors(current)
            current = int(rng.choice(nbrs))

        x_star = _hill_climb_counted(cs, current)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star

        yield ILSEvent(algo="random", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def random_restart_hc(
    space: SearchSpace,
    budget: int = 5000,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """Random-restart hill climbing baseline.

    Each iteration picks a uniformly random starting point and hill-climbs.
    """
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    best = -1
    best_fit = float("-inf")

    while cs.eval_count < budget:
        start = int(rng.integers(0, cs.size))
        opt = _hill_climb_counted(cs, start)
        f = cs.fitness(opt)
        if f > best_fit:
            best_fit = f
            best = opt

        yield ILSEvent(algo="rrhc", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=opt)


def _min_gap_neighbor(space: SearchSpace, x: int) -> int:
    f = getattr(space, "fitnesses", None)
    if f is not None and not hasattr(space, "neighbor_table"):
        masks = _get_masks(space.degree)
        neighbors = x ^ masks
        gaps = np.abs(f[neighbors] - f[x])
        return int(neighbors[gaps.argmin()])
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


def orc_only_ils(
    space: SearchSpace,
    budget: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """ORC-ILS ablation: only the min-kappa directed bit (no random diversification)."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="orc_only", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        y_star, _ = min_orc_neighbor(space, x_star, gamma)
        x_star = _hill_climb_counted(cs, y_star)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star
        yield ILSEvent(algo="orc_only", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


def mingap_ils(
    space: SearchSpace,
    budget: int = 5000,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """MG-ILS ablation: single MinGap-directed bit per iteration."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="mingap", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    while cs.eval_count < budget:
        y_star = _min_gap_neighbor(space, x_star)
        x_star = _hill_climb_counted(cs, y_star)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star
        yield ILSEvent(algo="mingap", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)


# ── Additional metaheuristics for algorithm selection ────────────────


def simulated_annealing(
    space: SearchSpace,
    budget: int = 5000,
    t_init: float = 2.0,
    t_min: float = 0.01,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """Simulated Annealing with geometric cooling."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    current = int(rng.integers(0, cs.size))
    current_fit = cs.fitness(current)
    best, best_fit = current, current_fit

    alpha = (t_min / t_init) ** (1.0 / max(budget - 1, 1))
    temp = t_init

    yield ILSEvent(algo="sa", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=current)

    while cs.eval_count < budget:
        nbrs = cs.neighbors(current)
        candidate = int(rng.choice(nbrs))
        cand_fit = cs.fitness(candidate)
        delta = cand_fit - current_fit
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-15)):
            current, current_fit = candidate, cand_fit
        if current_fit > best_fit:
            best, best_fit = current, current_fit
        temp *= alpha
        yield ILSEvent(algo="sa", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=current)


def tabu_search(
    space: SearchSpace,
    budget: int = 5000,
    tabu_tenure: int | None = None,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """Tabu Search with fixed-length recency memory."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    if tabu_tenure is None:
        tabu_tenure = max(7, space.degree // 3)

    current = int(rng.integers(0, cs.size))
    current_fit = cs.fitness(current)
    best, best_fit = current, current_fit

    from collections import deque
    tabu_list: deque[int] = deque(maxlen=tabu_tenure)
    tabu_set: set[int] = set()

    yield ILSEvent(algo="tabu", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=current)

    while cs.eval_count < budget:
        nbrs = cs.neighbors(current)
        best_cand, best_cand_fit = -1, float("-inf")
        for n in nbrs:
            n = int(n)
            f = cs.fitness(n)
            if cs.eval_count >= budget:
                break
            is_tabu = n in tabu_set
            aspiration = f > best_fit
            if (not is_tabu or aspiration) and f > best_cand_fit:
                best_cand, best_cand_fit = n, f

        if best_cand == -1:
            break

        if len(tabu_list) == tabu_tenure:
            evicted = tabu_list[0]
            tabu_set.discard(evicted)
        tabu_list.append(current)
        tabu_set.add(current)

        current, current_fit = best_cand, best_cand_fit
        if current_fit > best_fit:
            best, best_fit = current, current_fit

        yield ILSEvent(algo="tabu", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=current)


def one_plus_one_ea(
    space: SearchSpace,
    budget: int = 5000,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """(1+1) EA: mutate one random neighbor, accept if not worse."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    current = int(rng.integers(0, cs.size))
    current_fit = cs.fitness(current)
    best, best_fit = current, current_fit

    yield ILSEvent(algo="ea11", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=current)

    while cs.eval_count < budget:
        nbrs = cs.neighbors(current)
        mutant = int(rng.choice(nbrs))
        mutant_fit = cs.fitness(mutant)
        if mutant_fit >= current_fit:
            current, current_fit = mutant, mutant_fit
        if current_fit > best_fit:
            best, best_fit = current, current_fit
        yield ILSEvent(algo="ea11", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=current)


def variable_neighborhood_search(
    space: SearchSpace,
    budget: int = 5000,
    k_max: int = 5,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """VNS: shake with increasing random walk length, then hill-climb."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(algo="vns", evals=cs.eval_count,
                   best_fitness=best_fit, current_optimum=x_star)

    k = 1
    while cs.eval_count < budget:
        current = x_star
        for _ in range(k):
            nbrs = cs.neighbors(current)
            current = int(rng.choice(nbrs))

        x_prime = _hill_climb_counted(cs, current)
        f = cs.fitness(x_prime)

        if f > best_fit:
            best_fit = f
            best = x_prime
            x_star = x_prime
            k = 1
        else:
            k = k + 1 if k < k_max else 1

        yield ILSEvent(algo="vns", evals=cs.eval_count,
                       best_fitness=best_fit, current_optimum=x_star)
