"""Fitness-lifted Ollivier-Ricci curvature on discrete search graphs.

Implements the O(k^2) computation from Section 3.2 of the thesis, exploiting
the disjoint-neighborhood property to decompose optimal transport into
shared self-matches and an exclusive-neighbor assignment problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .search_spaces.protocol import SearchSpace


@dataclass
class ORCExplainerData:
    """Detailed breakdown of one ORC computation for visualization.

    Returned by compute_orc_explained() to power the "why this direction?"
    modal in the frontend.
    """

    x_idx: int
    y_idx: int
    kappa: float
    w1: float

    shared: list[int]
    x_exclusive: list[int]
    y_exclusive: list[int]

    # The optimal matching between exclusive neighbors
    # matching[i] = j means x_exclusive[i] is matched to y_exclusive[j]
    matching: list[tuple[int, int]]
    pair_costs: list[float]

    shared_cost: float  # always 0 for self-matches


def compute_orc(
    space: SearchSpace,
    x: int,
    y: int,
    gamma: float = 1.0,
) -> float:
    """Compute fitness-lifted ORC for edge (x, y).

    Uses the disjoint-neighborhood decomposition:
    1. Identify shared elements {x, y} in both supports
    2. Self-match shared elements (cost 0)
    3. Solve assignment on exclusive neighbors via the Hungarian algorithm

    Returns kappa in [-1, +1] (typically negative at local optima).
    """
    return compute_orc_explained(space, x, y, gamma).kappa


def compute_orc_explained(
    space: SearchSpace,
    x: int,
    y: int,
    gamma: float = 1.0,
) -> ORCExplainerData:
    """ORC computation with full explainer data for visualization."""

    nbrs_x = set(space.neighbors(x).tolist())
    nbrs_y = set(space.neighbors(y).tolist())

    support_x = nbrs_x | {x}
    support_y = nbrs_y | {y}

    shared = sorted(support_x & support_y)
    x_exclusive = sorted(support_x - support_y)
    y_exclusive = sorted(support_y - support_x)

    k = space.degree
    n_shared = len(shared)
    n_excl = len(x_exclusive)

    assert n_excl == len(y_exclusive), (
        f"Exclusive neighbor counts differ: {n_excl} vs {len(y_exclusive)}"
    )

    # Shared elements self-match at cost 0
    shared_cost = 0.0

    if n_excl == 0:
        # Fully overlapping neighborhoods (unusual)
        w1 = 0.0
    else:
        # Build cost matrix for exclusive neighbors
        cost_matrix = np.zeros((n_excl, n_excl))
        for i, a in enumerate(x_exclusive):
            f_a = space.fitness(a)
            for j, b in enumerate(y_exclusive):
                f_b = space.fitness(b)
                # Graph distance between exclusive neighbors is always 2
                # (by the disjoint-neighborhood property)
                structural = 2.0
                fitness_penalty = gamma * abs(f_a - f_b)
                cost_matrix[i, j] = structural + fitness_penalty

        # Solve the assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        excl_total = cost_matrix[row_ind, col_ind].sum()

        matching = list(zip(row_ind.tolist(), col_ind.tolist()))
        pair_costs = [float(cost_matrix[r, c]) for r, c in zip(row_ind, col_ind)]

        # W1 = (shared_cost + exclusive_cost) / (k + 1)
        w1 = (shared_cost + excl_total) / (k + 1)

    if n_excl == 0:
        matching = []
        pair_costs = []

    # kappa = 1 - W1 / d(x, y), and d(x, y) = 1 for adjacent nodes
    kappa = 1.0 - w1

    return ORCExplainerData(
        x_idx=x,
        y_idx=y,
        kappa=kappa,
        w1=w1,
        shared=shared,
        x_exclusive=x_exclusive,
        y_exclusive=y_exclusive,
        matching=matching,
        pair_costs=pair_costs,
        shared_cost=shared_cost,
    )


def compute_all_orc(
    space: SearchSpace,
    x: int,
    gamma: float = 1.0,
) -> dict[int, float]:
    """Compute ORC from node x to all its neighbors.

    Returns {neighbor_idx: kappa_value} dict.
    """
    nbrs = space.neighbors(x)
    return {int(y): compute_orc(space, x, y, gamma) for y in nbrs}


def min_orc_neighbor(
    space: SearchSpace,
    x: int,
    gamma: float = 1.0,
) -> tuple[int, float]:
    """Return the neighbor with the most negative ORC (escape direction).

    Returns (neighbor_idx, kappa_value).
    """
    orc_values = compute_all_orc(space, x, gamma)
    best = min(orc_values, key=orc_values.get)
    return best, orc_values[best]
