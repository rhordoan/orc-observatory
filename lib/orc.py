"""Fitness-lifted Ollivier-Ricci curvature on discrete search graphs.

All search spaces in this project (bit-flip, 2-opt, transposition swap) have
the disjoint-neighborhood property, so optimal transport decomposes into
sorted matching — O(k log k) per edge instead of O(k³) Hungarian.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .search_spaces.protocol import SearchSpace


@dataclass
class ORCExplainerData:
    """Detailed breakdown of one ORC computation for visualization."""

    x_idx: int
    y_idx: int
    kappa: float
    w1: float

    shared: list[int]
    x_exclusive: list[int]
    y_exclusive: list[int]

    matching: list[tuple[int, int]]
    pair_costs: list[float]

    shared_cost: float


def compute_orc(
    space: SearchSpace,
    x: int,
    y: int,
    gamma: float = 1.0,
) -> float:
    """Compute ORC for edge (x, y) via sorted matching."""
    return _orc_sorted_pair(space, x, y, gamma)


def _orc_sorted_pair(space: SearchSpace, x: int, y: int, gamma: float) -> float:
    """O(k log k) sorted-matching ORC for any disjoint-neighborhood graph."""
    nbrs_x = space.neighbors(x)
    nbrs_y = space.neighbors(y)
    k = len(nbrs_x)

    fx = np.empty(k - 1, dtype=np.float64)
    fy = np.empty(k - 1, dtype=np.float64)
    j = 0
    for n in nbrs_x:
        n = int(n)
        if n == y:
            continue
        fx[j] = space.fitness(n)
        j += 1
    j = 0
    for n in nbrs_y:
        n = int(n)
        if n == x:
            continue
        fy[j] = space.fitness(n)
        j += 1

    fx.sort()
    fy.sort()

    cost = (k - 1) * 2.0 + gamma * float(np.abs(fx - fy).sum())
    w1 = cost / (k + 1)
    return 1.0 - w1


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
    n_excl = len(x_exclusive)

    assert n_excl == len(y_exclusive), (
        f"Exclusive neighbor counts differ: {n_excl} vs {len(y_exclusive)}"
    )

    shared_cost = 0.0

    if n_excl == 0:
        w1 = 0.0
    else:
        f_x = np.array([space.fitness(a) for a in x_exclusive])
        f_y = np.array([space.fitness(b) for b in y_exclusive])
        cost_matrix = 2.0 + gamma * np.abs(f_x[:, None] - f_y[None, :])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        excl_total = cost_matrix[row_ind, col_ind].sum()

        matching = list(zip(row_ind.tolist(), col_ind.tolist()))
        pair_costs = [float(cost_matrix[r, c]) for r, c in zip(row_ind, col_ind)]

        w1 = (shared_cost + excl_total) / (k + 1)

    if n_excl == 0:
        matching = []
        pair_costs = []

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
    max_neighbors: int | None = None,
) -> dict[int, float]:
    """Compute ORC from node x to its neighbors.

    Uses sorted matching O(k log k) per edge for all disjoint-neighborhood
    spaces. Bit-flip spaces use a specialized vectorized path; other spaces
    (2-opt, transposition swap) use the generic sorted-pair path.

    Returns {neighbor_idx: kappa_value} dict.
    """
    if not hasattr(space, "neighbor_table") and hasattr(space, "fitnesses"):
        return _compute_all_orc_bitflip(space, x, gamma)
    return _compute_all_orc_generic(space, x, gamma, max_neighbors)


def _compute_all_orc_bitflip(space, x: int, gamma: float) -> dict[int, float]:
    """Vectorized sorted-matching ORC for bit-flip graphs."""
    from .gpu_orc import cpu_orc_bitflip_sorted
    n_bits = space.degree
    idx_arr = np.array([x], dtype=np.int64)
    orc_row = cpu_orc_bitflip_sorted(space.fitnesses, idx_arr, n_bits, gamma)
    return {x ^ (1 << b): float(orc_row[0, b]) for b in range(n_bits)}


def _compute_all_orc_generic(
    space: SearchSpace, x: int, gamma: float, max_neighbors: int | None,
) -> dict[int, float]:
    """Sorted-matching ORC for table-based / sampling-based spaces.

    Pre-fetches all neighbor arrays, then vectorizes the sort+match step.
    """
    nbrs_x = space.neighbors(x)
    k = len(nbrs_x)

    if max_neighbors is not None and k > max_neighbors:
        fx_val = space.fitness(x)
        gaps = np.array([abs(space.fitness(int(n)) - fx_val) for n in nbrs_x])
        top_k = np.argsort(gaps)[:max_neighbors]
        nbrs_x = nbrs_x[top_k]

    fx_all = np.array([space.fitness(int(n)) for n in nbrs_x])
    result: dict[int, float] = {}

    for idx_j, y_val in enumerate(nbrs_x):
        y = int(y_val)
        nbrs_y = space.neighbors(y)
        ky = len(nbrs_y)

        fy_excl = np.empty(ky - 1, dtype=np.float64)
        j = 0
        for n in nbrs_y:
            n_int = int(n)
            if n_int == x:
                continue
            fy_excl[j] = space.fitness(n_int)
            j += 1

        fx_excl = np.delete(fx_all, idx_j)

        fx_s = np.sort(fx_excl)
        fy_s = np.sort(fy_excl)

        n_excl = min(len(fx_s), len(fy_s))
        cost = n_excl * 2.0 + gamma * float(np.abs(fx_s[:n_excl] - fy_s[:n_excl]).sum())
        w1 = cost / (k + 1)
        result[y] = 1.0 - w1

    return result


def batch_orc_gpu(
    space: SearchSpace,
    optima_indices: np.ndarray,
    gamma: float = 1.0,
    max_neighbors: int = 60,
) -> dict[int, dict[int, float]]:
    """Batch ORC computation for all optima, GPU-accelerated via CuPy.

    Phase 1: Pre-fetch all neighbor arrays and fitness values (CPU).
    Phase 2: Batch sort + match on GPU.
    """
    try:
        import cupy as cp
        has_gpu = True
    except Exception:
        has_gpu = False

    n_opt = len(optima_indices)
    k = space.degree

    print(f"    batch_orc: {n_opt} optima, degree={k}, max_nbrs={max_neighbors}", flush=True)

    nbr_arrays: dict[int, np.ndarray] = {}
    selected_nbrs: list[np.ndarray] = []

    for x in optima_indices:
        x = int(x)
        nbrs = space.neighbors(x)
        nbr_arrays[x] = nbrs
        if k > max_neighbors:
            fx = space.fitness(x)
            gaps = np.array([abs(space.fitness(int(n)) - fx) for n in nbrs])
            top_k = np.argsort(gaps)[:max_neighbors]
            selected_nbrs.append(nbrs[top_k])
        else:
            selected_nbrs.append(nbrs)

    unique_ys = set()
    for sel in selected_nbrs:
        for y in sel:
            unique_ys.add(int(y))
    print(f"    batch_orc: pre-fetching {len(unique_ys)} neighbor arrays...", flush=True)
    for y in unique_ys:
        if y not in nbr_arrays:
            nbr_arrays[y] = space.neighbors(y)

    all_nodes: set[int] = set()
    for arr in nbr_arrays.values():
        for n in arr:
            all_nodes.add(int(n))
    for x in optima_indices:
        all_nodes.add(int(x))

    node_list = sorted(all_nodes)
    node_to_pos = {n: i for i, n in enumerate(node_list)}
    fit_arr = np.array([space.fitness(n) for n in node_list], dtype=np.float64)

    m = min(max_neighbors, k)
    n_excl = k - 1

    total_pairs = n_opt * m
    fx_block = np.empty((total_pairs, n_excl), dtype=np.float64)
    fy_block = np.empty((total_pairs, n_excl), dtype=np.float64)

    print(f"    batch_orc: building {total_pairs} fitness blocks...", flush=True)
    pair_idx = 0
    pair_map: list[tuple[int, int]] = []
    for i, x_val in enumerate(optima_indices):
        x = int(x_val)
        sel = selected_nbrs[i]
        nbrs_x_full = nbr_arrays[x]
        fx_full = fit_arr[[node_to_pos[int(n)] for n in nbrs_x_full]]

        for y_val in sel:
            y = int(y_val)
            mask = nbrs_x_full != y
            fx_excl = fx_full[mask]
            if len(fx_excl) < n_excl:
                fx_excl = np.pad(fx_excl, (0, n_excl - len(fx_excl)))
            fx_block[pair_idx] = fx_excl[:n_excl]

            nbrs_y_full = nbr_arrays[y]
            fy_full = fit_arr[[node_to_pos[int(n)] for n in nbrs_y_full]]
            mask_y = nbrs_y_full != x
            fy_excl = fy_full[mask_y]
            if len(fy_excl) < n_excl:
                fy_excl = np.pad(fy_excl, (0, n_excl - len(fy_excl)))
            fy_block[pair_idx] = fy_excl[:n_excl]

            pair_map.append((i, y))
            pair_idx += 1

    fx_block = fx_block[:pair_idx]
    fy_block = fy_block[:pair_idx]

    if has_gpu and pair_idx > 100:
        print(f"    batch_orc: GPU sort+match on {pair_idx} pairs...", flush=True)
        fx_g = cp.asarray(fx_block)
        fy_g = cp.asarray(fy_block)
        fx_g.sort(axis=1)
        fy_g.sort(axis=1)
        costs = n_excl * 2.0 + gamma * cp.abs(fx_g - fy_g).sum(axis=1)
        kappas = 1.0 - costs / (k + 1)
        kappas_cpu = cp.asnumpy(kappas)
    else:
        print(f"    batch_orc: CPU sort+match on {pair_idx} pairs...", flush=True)
        fx_block.sort(axis=1)
        fy_block.sort(axis=1)
        costs = n_excl * 2.0 + gamma * np.abs(fx_block - fy_block).sum(axis=1)
        kappas_cpu = 1.0 - costs / (k + 1)

    orc_values: dict[int, dict[int, float]] = {}
    for p, (opt_i, y) in enumerate(pair_map):
        if opt_i not in orc_values:
            orc_values[opt_i] = {}
        orc_values[opt_i][y] = float(kappas_cpu[p])

    print(f"    batch_orc: done.", flush=True)
    return orc_values


def min_orc_neighbor(
    space: SearchSpace,
    x: int,
    gamma: float = 1.0,
) -> tuple[int, float]:
    """Return the neighbor with the most negative ORC (escape direction)."""
    orc_values = compute_all_orc(space, x, gamma)
    best = min(orc_values, key=orc_values.get)
    return best, orc_values[best]
