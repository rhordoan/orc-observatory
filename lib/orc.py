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
    return_topology: bool = False,
) -> dict[int, dict[int, float]] | tuple:
    """Batch ORC for all optima, GPU-accelerated via CuPy.

    Uses neighbor_fitnesses() (O(1) delta per move) when available to
    avoid materializing neighbor tour tuples. For N(y)\\{x} exclusion,
    finds x by matching fitness value f(x). Then batch sort+match on GPU.

    When return_topology=True, returns (orc_values, topology_dict) so
    shuffled ORCs can reuse the same edge set without re-materializing
    neighbor tours.
    """
    try:
        import cupy as cp
        has_gpu = True
    except Exception:
        has_gpu = False

    import time as _time

    n_opt = len(optima_indices)
    k = space.degree
    has_delta = hasattr(space, "neighbor_fitnesses")
    n_excl = k - 1
    print(f"    batch_orc: {n_opt} optima, degree={k}, max_nbrs={max_neighbors}, delta={has_delta}", flush=True)
    t0 = _time.time()

    opt_nbr_nodes: list[np.ndarray] = []
    opt_nbr_fit: list[np.ndarray] = []
    selected_move_idx: list[np.ndarray] = []

    for x in optima_indices:
        x = int(x)
        nbrs = space.neighbors(x)
        opt_nbr_nodes.append(nbrs)
        nf = space.neighbor_fitnesses(x) if has_delta else np.array([space.fitness(int(n)) for n in nbrs])
        opt_nbr_fit.append(nf)
        if k > max_neighbors:
            gaps = np.abs(nf - space.fitness(x))
            selected_move_idx.append(np.argsort(gaps)[:max_neighbors])
        else:
            selected_move_idx.append(np.arange(k))

    print(f"    batch_orc: optima arrays in {_time.time()-t0:.1f}s", flush=True)
    t1 = _time.time()

    y_fit_cache: dict[int, np.ndarray] = {}
    pair_list: list[tuple[int, int, int]] = []

    for i in range(n_opt):
        nbrs = opt_nbr_nodes[i]
        for move_idx in selected_move_idx[i]:
            move_idx = int(move_idx)
            y_node = int(nbrs[move_idx])
            pair_list.append((i, y_node, move_idx))

    unique_ys = {y for _, y, _ in pair_list}
    print(f"    batch_orc: computing {len(unique_ys)} neighbor fitness arrays (delta)...", flush=True)
    for y in unique_ys:
        if y not in y_fit_cache:
            y_fit_cache[y] = space.neighbor_fitnesses(y) if has_delta else np.array([space.fitness(int(n)) for n in space.neighbors(y)])

    print(f"    batch_orc: neighbor fitness in {_time.time()-t1:.1f}s", flush=True)

    if return_topology:
        est_registrations = len(unique_ys) * k
        solution_size = getattr(space, '_n', getattr(space, '_n_jobs', k))
        est_bytes = est_registrations * solution_size * 16
        mem_limit = 80 * (1024 ** 3)
        if est_bytes < mem_limit:
            t_cache = _time.time()
            n_cached = 0
            for y in unique_ys:
                _ = space.neighbors(y)
                n_cached += 1
            print(f"    batch_orc: pre-cached {n_cached} neighbor arrays "
                  f"(~{est_bytes / 1e9:.1f}GB est) in {_time.time()-t_cache:.1f}s", flush=True)
        else:
            print(f"    batch_orc: skipping pre-cache (~{est_bytes / 1e9:.1f}GB > 80GB limit)", flush=True)

    t2 = _time.time()

    total = len(pair_list)
    fx_block = np.empty((total, n_excl), dtype=np.float64)
    fy_block = np.empty((total, n_excl), dtype=np.float64)

    for p, (opt_i, y_node, move_idx) in enumerate(pair_list):
        fx_block[p] = np.delete(opt_nbr_fit[opt_i], move_idx)[:n_excl]

        fy_full = y_fit_cache[y_node]
        x_node = int(optima_indices[opt_i])
        fx_val = space.fitness(x_node)
        x_pos = int(np.argmin(np.abs(fy_full - fx_val)))
        fy_block[p] = np.delete(fy_full, x_pos)[:n_excl]

    print(f"    batch_orc: blocks in {_time.time()-t2:.1f}s", flush=True)
    t3 = _time.time()

    if has_gpu and total > 50:
        print(f"    batch_orc: GPU sort+match {total} pairs...", flush=True)
        fx_g = cp.asarray(fx_block)
        fy_g = cp.asarray(fy_block)
        fx_g.sort(axis=1)
        fy_g.sort(axis=1)
        costs = n_excl * 2.0 + gamma * cp.abs(fx_g - fy_g).sum(axis=1)
        kappas_cpu = cp.asnumpy(1.0 - costs / (k + 1))
    else:
        print(f"    batch_orc: CPU sort+match {total} pairs...", flush=True)
        fx_block.sort(axis=1)
        fy_block.sort(axis=1)
        costs = n_excl * 2.0 + gamma * np.abs(fx_block - fy_block).sum(axis=1)
        kappas_cpu = 1.0 - costs / (k + 1)

    print(f"    batch_orc: sort+match in {_time.time()-t3:.1f}s, total {_time.time()-t0:.1f}s", flush=True)

    orc_values: dict[int, dict[int, float]] = {}
    for p, (opt_i, y_node, _) in enumerate(pair_list):
        if opt_i not in orc_values:
            orc_values[opt_i] = {}
        orc_values[opt_i][y_node] = float(kappas_cpu[p])

    if return_topology:
        topology = {
            "pair_list": pair_list,
            "unique_ys": unique_ys,
            "opt_nbr_nodes": opt_nbr_nodes,
            "selected_move_idx": selected_move_idx,
        }
        return orc_values, topology
    return orc_values


def batch_orc_reuse_topology(
    space: SearchSpace,
    optima_indices: np.ndarray,
    gamma: float,
    topology: dict,
) -> dict[int, dict[int, float]]:
    """Recompute ORC on the same edges with different fitness values.

    Reuses pair_list and unique_ys from a prior batch_orc_gpu call,
    avoiding neighbor-tour materialization entirely.
    """
    import time as _time

    pair_list = topology["pair_list"]
    unique_ys = topology["unique_ys"]

    k = space.degree
    n_excl = k - 1
    has_delta = hasattr(space, "neighbor_fitnesses")

    t0 = _time.time()

    opt_nbr_fit: list[np.ndarray] = []
    for x in optima_indices:
        nf = space.neighbor_fitnesses(int(x)) if has_delta else np.array(
            [space.fitness(int(n)) for n in space.neighbors(int(x))])
        opt_nbr_fit.append(nf)

    print(f"    batch_orc_reuse: optima fitnesses in {_time.time()-t0:.1f}s", flush=True)
    t1 = _time.time()

    y_fit_cache: dict[int, np.ndarray] = {}
    for y in unique_ys:
        y_fit_cache[y] = space.neighbor_fitnesses(y) if has_delta else np.array(
            [space.fitness(int(n)) for n in space.neighbors(y)])

    print(f"    batch_orc_reuse: unique_y fitnesses in {_time.time()-t1:.1f}s", flush=True)
    t2 = _time.time()

    total = len(pair_list)
    fx_block = np.empty((total, n_excl), dtype=np.float64)
    fy_block = np.empty((total, n_excl), dtype=np.float64)

    for p, (opt_i, y_node, move_idx) in enumerate(pair_list):
        fx_block[p] = np.delete(opt_nbr_fit[opt_i], move_idx)[:n_excl]

        fy_full = y_fit_cache[y_node]
        x_node = int(optima_indices[opt_i])
        fx_val = space.fitness(x_node)
        x_pos = int(np.argmin(np.abs(fy_full - fx_val)))
        fy_block[p] = np.delete(fy_full, x_pos)[:n_excl]

    print(f"    batch_orc_reuse: blocks in {_time.time()-t2:.1f}s", flush=True)
    t3 = _time.time()

    fx_block.sort(axis=1)
    fy_block.sort(axis=1)
    costs = n_excl * 2.0 + gamma * np.abs(fx_block - fy_block).sum(axis=1)
    kappas = 1.0 - costs / (k + 1)

    print(f"    batch_orc_reuse: sort+match in {_time.time()-t3:.1f}s, total {_time.time()-t0:.1f}s", flush=True)

    orc_values: dict[int, dict[int, float]] = {}
    for p, (opt_i, y_node, _) in enumerate(pair_list):
        if opt_i not in orc_values:
            orc_values[opt_i] = {}
        orc_values[opt_i][y_node] = float(kappas[p])

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
