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
from .orc import compute_all_orc, min_orc_neighbor


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

    def fitness(self, idx: int) -> float:
        self.eval_count += 1
        return self._space.fitness(idx)

    def neighbors(self, idx: int) -> np.ndarray:
        return self._space.neighbors(idx)

    def solution_label(self, idx: int) -> str:
        return self._space.solution_label(idx)


def _hill_climb_counted(cs: CountingSpace, start: int) -> int:
    """Best-improvement hill climbing using the counting wrapper."""
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
