"""GPU-accelerated ORC computation for disjoint-neighborhood graphs.

Key insight: on graphs with the disjoint-neighborhood property (bit-flip,
2-opt, transposition swap), the ORC optimal transport cost matrix has the
form c_ij = 2 + gamma * |f(a_i) - f(b_j)|.  The constant structural
distance means the optimal assignment is the *sorted matching* (pair the
i-th smallest exclusive-neighbor fitness of x with the i-th smallest of y).
This reduces per-edge cost from O(k^3) Hungarian to O(k log k) sort, and
enables massive GPU parallelism via one CUDA thread per edge.
"""

from __future__ import annotations

import numpy as np

_NUMBA_OK: bool | None = None


def _check_numba() -> bool:
    global _NUMBA_OK
    if _NUMBA_OK is None:
        try:
            from numba import cuda
            _NUMBA_OK = cuda.is_available()
        except Exception:
            _NUMBA_OK = False
    return _NUMBA_OK


# ── Numba CUDA kernel: batch ORC for bit-flip neighborhoods ──────────────

def gpu_orc_bitflip(
    fitnesses: np.ndarray,
    optima_indices: np.ndarray,
    n_bits: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute ORC from each optimum to all k=n_bits neighbors on GPU.

    Returns (n_optima, n_bits) array of kappa values.
    """
    if not _check_numba() or _orc_bitflip_kernel is None:
        return cpu_orc_bitflip_sorted(fitnesses, optima_indices, n_bits, gamma)

    try:
        from numba import cuda

        n_opt = len(optima_indices)
        d_fit = cuda.to_device(fitnesses.astype(np.float64))
        d_idx = cuda.to_device(optima_indices.astype(np.int64))
        d_out = cuda.device_array((n_opt, n_bits), dtype=np.float64)

        total = n_opt * n_bits
        threads = 256
        blocks = (total + threads - 1) // threads
        _orc_bitflip_kernel[blocks, threads](d_fit, d_idx, n_bits, gamma, n_opt, d_out)
        cuda.synchronize()

        return d_out.copy_to_host()
    except Exception:
        return cpu_orc_bitflip_sorted(fitnesses, optima_indices, n_bits, gamma)


def _make_orc_bitflip_kernel():
    from numba import cuda, float64, int64

    @cuda.jit
    def kernel(fitnesses, optima_indices, n_bits, gamma, n_opt, orc_out):
        tid = cuda.grid(1)
        total = n_opt * n_bits
        if tid >= total:
            return

        oi = tid // n_bits
        bj = tid % n_bits

        x = optima_indices[oi]
        n_excl = n_bits - 1

        xf = cuda.local.array(32, float64)
        yf = cuda.local.array(32, float64)

        y = x ^ (int64(1) << int64(bj))
        idx = 0
        for b in range(n_bits):
            if b == bj:
                continue
            xf[idx] = fitnesses[x ^ (int64(1) << int64(b))]
            yf[idx] = fitnesses[y ^ (int64(1) << int64(b))]
            idx += 1

        # Insertion sort xf
        for i in range(1, n_excl):
            key = xf[i]
            j = i - 1
            while j >= 0 and xf[j] > key:
                xf[j + 1] = xf[j]
                j -= 1
            xf[j + 1] = key

        # Insertion sort yf
        for i in range(1, n_excl):
            key = yf[i]
            j = i - 1
            while j >= 0 and yf[j] > key:
                yf[j + 1] = yf[j]
                j -= 1
            yf[j + 1] = key

        cost = float64(0.0)
        for i in range(n_excl):
            diff = xf[i] - yf[i]
            if diff < 0.0:
                diff = -diff
            cost += float64(2.0) + gamma * diff

        w1 = cost / float64(n_bits + 1)
        orc_out[oi, bj] = float64(1.0) - w1

    return kernel


try:
    _orc_bitflip_kernel = _make_orc_bitflip_kernel()
except Exception:
    _orc_bitflip_kernel = None


# ── CPU fast path: sorted matching (still much faster than Hungarian) ────

def cpu_orc_bitflip_sorted(
    fitnesses: np.ndarray,
    optima_indices: np.ndarray,
    n_bits: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Vectorized NumPy ORC using sorted matching -- no Python loops.

    Processes all (optimum, neighbor) pairs in one shot via advanced indexing.
    O(n_opt * k * k*log(k)) total, dominated by the sort.
    """
    xs = optima_indices.astype(np.int64)
    n_opt = len(xs)

    masks = np.array([1 << b for b in range(n_bits)], dtype=np.int64)
    other_masks = np.array(
        [[1 << b for b in range(n_bits) if b != j] for j in range(n_bits)],
        dtype=np.int64,
    )

    x_nbrs = xs[:, None, None] ^ other_masks[None, :, :]
    ys = xs[:, None] ^ masks[None, :]
    y_nbrs = ys[:, :, None] ^ other_masks[None, :, :]

    x_fits = fitnesses[x_nbrs]
    y_fits = fitnesses[y_nbrs]

    x_fits.sort(axis=-1)
    y_fits.sort(axis=-1)

    abs_diff_sum = np.abs(x_fits - y_fits).sum(axis=-1)
    total_cost = 2.0 * (n_bits - 1) + gamma * abs_diff_sum
    return 1.0 - total_cost / (n_bits + 1)


# ── Full GPU OTG pipeline ────────────────────────────────────────────────

def gpu_build_otg_edges(
    orc_matrix: np.ndarray,
    optima_indices: np.ndarray,
    attractor: np.ndarray,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve OTG edges using precomputed ORC + attractor array.

    For each optimum, walk through neighbors in curvature order (most
    negative first) and pick the first that escapes to a different basin.
    The attractor array gives O(1) hill-climb resolution.

    Returns (targets, kappas, via_neighbors) arrays of shape (n_optima,).
    """
    opt_set = set(int(v) for v in attractor)
    idx_map = {int(optima_indices[i]): i for i in range(len(optima_indices))}
    n_opt = len(optima_indices)

    targets = np.full(n_opt, -1, dtype=np.intp)
    kappas = np.zeros(n_opt, dtype=np.float64)
    via = np.zeros(n_opt, dtype=np.intp)

    for i in range(n_opt):
        x = int(optima_indices[i])
        order = np.argsort(orc_matrix[i])

        for rank in order:
            y = x ^ (1 << int(rank))
            dest_sol = int(attractor[y])
            dest_i = idx_map.get(dest_sol, i)
            if dest_i != i:
                targets[i] = dest_i
                kappas[i] = orc_matrix[i, rank]
                via[i] = y
                break
        else:
            rank0 = int(order[0])
            y = x ^ (1 << rank0)
            dest_sol = int(attractor[y])
            targets[i] = idx_map.get(dest_sol, i)
            kappas[i] = orc_matrix[i, rank0]
            via[i] = y

    return targets, kappas, via
