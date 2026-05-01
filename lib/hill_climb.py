"""Hill climbing and local optima enumeration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .search_spaces.protocol import SearchSpace


@dataclass
class LocalOptimum:
    """A local optimum with its basin of attraction."""

    idx: int
    fitness: float
    basin: list[int]

    @property
    def basin_size(self) -> int:
        return len(self.basin)


def hill_climb(space: SearchSpace, start: int) -> int:
    """Best-improvement hill climbing from *start*. Returns local optimum index.

    Ties among improving neighbors are broken by choosing the fittest.
    Stops when no neighbor has strictly better fitness.
    """
    current = start
    while True:
        nbrs = space.neighbors(current)
        current_fit = space.fitness(current)
        best_nbr = -1
        best_fit = current_fit
        for n in nbrs:
            f = space.fitness(n)
            if f > best_fit:
                best_fit = f
                best_nbr = n
        if best_nbr == -1:
            return current
        current = best_nbr


def enumerate_local_optima(space: SearchSpace) -> list[LocalOptimum]:
    """Exhaustive enumeration: hill-climb from every solution.

    Only practical for small spaces (|S| <= 2^16 or so).
    Returns local optima sorted by fitness (descending).
    """
    # Map each solution to its local optimum via hill climbing
    attractor = np.full(space.size, -1, dtype=np.intp)
    for s in range(space.size):
        if attractor[s] == -1:
            path = []
            current = s
            while attractor[current] == -1:
                path.append(current)
                attractor[current] = -2  # mark as visiting
                current = hill_climb(space, current)
                if current in path:
                    # Already visited in this path, it's the optimum
                    break
            opt = current
            for node in path:
                attractor[node] = opt

    # Group solutions by their attractor
    basins: dict[int, list[int]] = {}
    for s in range(space.size):
        opt = attractor[s]
        basins.setdefault(opt, []).append(s)

    optima = [
        LocalOptimum(idx=opt, fitness=space.fitness(opt), basin=basin)
        for opt, basin in basins.items()
    ]
    optima.sort(key=lambda o: o.fitness, reverse=True)
    return optima


def random_restart_optima(
    space: SearchSpace, n_restarts: int = 1000, seed: int | None = None
) -> list[LocalOptimum]:
    """Sampling-based local optima collection for large spaces.

    Runs *n_restarts* independent hill climbs from random starting points.
    Returns deduplicated local optima sorted by fitness (descending).
    """
    rng = np.random.default_rng(seed)
    found: dict[int, list[int]] = {}

    for _ in range(n_restarts):
        start = int(rng.integers(0, space.size))
        opt = hill_climb(space, start)
        found.setdefault(opt, []).append(start)

    optima = [
        LocalOptimum(idx=opt, fitness=space.fitness(opt), basin=starts)
        for opt, starts in found.items()
    ]
    optima.sort(key=lambda o: o.fitness, reverse=True)
    return optima
