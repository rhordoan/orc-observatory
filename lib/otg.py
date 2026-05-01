"""ORC Transition Graph construction and funnel analysis.

Implements Algorithm 1 from the thesis: for each local optimum, follow
the most negatively curved neighbor and hill-climb to the destination.
The result is a functional graph (out-degree 1) whose connected components
reveal the funnel structure of the landscape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .search_spaces.protocol import SearchSpace
from .hill_climb import LocalOptimum, hill_climb
from .orc import compute_all_orc, min_orc_neighbor


@dataclass
class OTGEdge:
    source: int         # index into the optima list
    target: int         # index into the optima list
    min_kappa: float    # ORC value of the chosen escape direction
    via_neighbor: int   # the solution-space neighbor that was followed


@dataclass
class Funnel:
    attractor_idx: int          # index into the optima list
    member_indices: list[int]   # indices into the optima list
    attractor_fitness: float
    is_cycle: bool = False


@dataclass
class OTGResult:
    """Complete OTG analysis result."""

    optima: list[LocalOptimum]
    edges: list[OTGEdge]
    funnels: list[Funnel]

    # Per-optimum data for the frontend
    orc_values: dict[int, dict[int, float]]  # opt_list_idx -> {neighbor: kappa}

    # Structural metrics
    compression_ratio: float
    mean_terminal_rank: float
    top5_reachability: float
    dag_depth: int
    has_cycles: bool


def build_otg(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float = 1.0,
    on_edge: callable = None,
) -> OTGResult:
    """Build the ORC Transition Graph over the given local optima.

    Args:
        space: the search space instance
        optima: list of local optima (from hill_climb.enumerate_local_optima)
        gamma: ORC sensitivity parameter
        on_edge: optional callback(edge: OTGEdge) for streaming progress

    Returns:
        OTGResult with edges, funnels, and structural metrics.
    """
    # Map solution-space index to optima-list index
    opt_idx_map = {o.idx: i for i, o in enumerate(optima)}
    n = len(optima)

    edges: list[OTGEdge] = []
    orc_values: dict[int, dict[int, float]] = {}
    # successor[i] = j means optimum i points to optimum j in the OTG
    successor = np.full(n, -1, dtype=np.intp)

    for i, opt in enumerate(optima):
        # Compute ORC to all neighbors
        all_orc = compute_all_orc(space, opt.idx, gamma)
        orc_values[i] = all_orc

        # Find the most negatively curved neighbor
        escape_nbr = min(all_orc, key=all_orc.get)
        escape_kappa = all_orc[escape_nbr]

        # Hill-climb from the escape neighbor to reach the destination optimum
        dest_solution = hill_climb(space, escape_nbr)

        if dest_solution in opt_idx_map:
            dest_idx = opt_idx_map[dest_solution]
        else:
            # Destination is an optimum we haven't seen (shouldn't happen
            # with exhaustive enumeration, but handle gracefully)
            dest_idx = i  # self-loop

        successor[i] = dest_idx

        edge = OTGEdge(
            source=i,
            target=dest_idx,
            min_kappa=escape_kappa,
            via_neighbor=escape_nbr,
        )
        edges.append(edge)

        if on_edge is not None:
            on_edge(edge)

    # Detect funnels: follow successor chains until we hit a cycle
    funnels = _detect_funnels(successor, optima)
    has_cycles = any(f.is_cycle for f in funnels)

    # Compute structural metrics
    compression = len(funnels) / n if n > 0 else 1.0

    # Mean terminal rank: average normalized rank of funnel attractors
    fitness_sorted = sorted(range(n), key=lambda i: optima[i].fitness, reverse=True)
    rank_map = {idx: rank for rank, idx in enumerate(fitness_sorted)}
    attractor_ranks = [rank_map[f.attractor_idx] / max(n - 1, 1) for f in funnels]
    mean_rank = float(np.mean(attractor_ranks)) if attractor_ranks else 0.5

    # Top-5% reachability
    top5_threshold = max(1, int(0.05 * n))
    top5_set = set(fitness_sorted[:top5_threshold])
    top5_reach = sum(
        1 for f in funnels if f.attractor_idx in top5_set
    ) / max(len(funnels), 1)

    # DAG depth: longest chain from any node to its attractor
    dag_depth = _compute_dag_depth(successor, funnels)

    return OTGResult(
        optima=optima,
        edges=edges,
        funnels=funnels,
        orc_values=orc_values,
        compression_ratio=compression,
        mean_terminal_rank=mean_rank,
        top5_reachability=top5_reach,
        dag_depth=dag_depth,
        has_cycles=has_cycles,
    )


def _detect_funnels(
    successor: np.ndarray, optima: list[LocalOptimum]
) -> list[Funnel]:
    """Detect funnels by following successor chains to terminal attractors."""
    n = len(successor)
    funnel_id = np.full(n, -1, dtype=np.intp)
    funnels: list[Funnel] = []

    for start in range(n):
        if funnel_id[start] != -1:
            continue

        # Trace the chain from start
        path = []
        visited_in_path = {}
        current = start
        while current not in visited_in_path and funnel_id[current] == -1:
            visited_in_path[current] = len(path)
            path.append(current)
            current = int(successor[current])

        if funnel_id[current] != -1:
            # Reached a node already assigned to a funnel
            fid = funnel_id[current]
            for node in path:
                funnel_id[node] = fid
                funnels[fid].member_indices.append(node)
        elif current in visited_in_path:
            # Found a cycle; the cycle members form the attractor
            cycle_start_pos = visited_in_path[current]
            cycle = path[cycle_start_pos:]
            is_cycle = len(cycle) > 1

            # The attractor is the best-fitness node in the cycle
            attractor = min(cycle, key=lambda i: -optima[i].fitness)

            fid = len(funnels)
            funnel = Funnel(
                attractor_idx=attractor,
                member_indices=list(path),
                attractor_fitness=optima[attractor].fitness,
                is_cycle=is_cycle,
            )
            funnels.append(funnel)
            for node in path:
                funnel_id[node] = fid

    return funnels


def _compute_dag_depth(successor: np.ndarray, funnels: list[Funnel]) -> int:
    """Longest path from any node to its funnel attractor."""
    attractor_set = {f.attractor_idx for f in funnels}
    max_depth = 0

    for start in range(len(successor)):
        depth = 0
        current = start
        seen = set()
        while current not in attractor_set and current not in seen:
            seen.add(current)
            current = int(successor[current])
            depth += 1
        max_depth = max(max_depth, depth)

    return max_depth
