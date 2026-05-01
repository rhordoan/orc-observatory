"""LON-d1 (deterministic 1-hop Local Optima Network) for comparison.

LON-d1 connects each local optimum to the optimum reached by hill-climbing
from its best-fitness neighbor. This is the deterministic, parameter-free
LON variant that serves as a baseline against the OTG.
"""

from __future__ import annotations

from dataclasses import dataclass

from .search_spaces.protocol import SearchSpace
from .hill_climb import LocalOptimum, hill_climb


@dataclass
class LONEdge:
    source: int     # index into optima list
    target: int     # index into optima list
    via_neighbor: int
    neighbor_fitness: float


@dataclass
class LONResult:
    optima: list[LocalOptimum]
    edges: list[LONEdge]
    n_self_loops: int
    singleton_fraction: float


def build_lon_d1(
    space: SearchSpace,
    optima: list[LocalOptimum],
) -> LONResult:
    """Build LON-d1: from each optimum, follow the best-fitness neighbor.

    LON-d1 typically produces 97-98% self-loops on standard benchmarks
    because the best neighbor of a local optimum usually hill-climbs
    back to the same optimum. This is the key limitation that the OTG
    addresses with distributional (ORC) information.
    """
    opt_idx_map = {o.idx: i for i, o in enumerate(optima)}
    edges: list[LONEdge] = []
    n_self_loops = 0

    for i, opt in enumerate(optima):
        nbrs = space.neighbors(opt.idx)

        # Find the best-fitness neighbor
        best_nbr = -1
        best_fit = -float("inf")
        for n in nbrs:
            f = space.fitness(n)
            if f > best_fit:
                best_fit = f
                best_nbr = n

        # Hill-climb from that neighbor
        dest_solution = hill_climb(space, best_nbr)
        dest_idx = opt_idx_map.get(dest_solution, i)

        if dest_idx == i:
            n_self_loops += 1

        edges.append(LONEdge(
            source=i,
            target=dest_idx,
            via_neighbor=best_nbr,
            neighbor_fitness=best_fit,
        ))

    n = len(optima)
    singleton_frac = n_self_loops / n if n > 0 else 1.0

    return LONResult(
        optima=optima,
        edges=edges,
        n_self_loops=n_self_loops,
        singleton_fraction=singleton_frac,
    )
