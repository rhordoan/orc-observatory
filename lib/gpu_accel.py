"""Optional GPU acceleration for fitness precomputation and hill climbing.

Uses CuPy for vectorized array operations and Numba CUDA for the parallel
hill climbing kernel.  Falls back gracefully to vectorized NumPy when no
GPU is available.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# GPU availability detection
# ---------------------------------------------------------------------------

_GPU_AVAILABLE: bool | None = None


def is_gpu_available() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import cupy as cp  # noqa: F401
            cp.cuda.Device(0).compute_capability
            _GPU_AVAILABLE = True
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def _get_xp():
    """Return cupy if GPU available, else numpy."""
    if is_gpu_available():
        import cupy as cp
        return cp
    return np


# ---------------------------------------------------------------------------
# Vectorized fitness computation (works on both NumPy and CuPy arrays)
# ---------------------------------------------------------------------------

def _all_bit_vectors(n: int, xp) -> np.ndarray:
    """Generate (2^N, N) matrix of all bit vectors."""
    size = 1 << n
    indices = xp.arange(size, dtype=xp.int64)
    bits = xp.zeros((size, n), dtype=xp.int32)
    for bit in range(n):
        bits[:, bit] = (indices >> bit) & 1
    return bits


def gpu_compute_fitness_nk(
    n: int, k: int, deps: np.ndarray, tables: np.ndarray
) -> np.ndarray:
    """Vectorized NK fitness for all 2^N solutions."""
    xp = _get_xp()
    deps_d = xp.asarray(deps)       # (N, K)
    tables_d = xp.asarray(tables)    # (N, 2^(K+1))
    bits = _all_bit_vectors(n, xp)   # (2^N, N)

    size = 1 << n
    keys = bits.copy()  # (2^N, N) -- start with bit_i for each locus
    for j in range(k):
        dep_bits = xp.take(bits, deps_d[:, j], axis=1)  # (2^N, N)
        keys = keys | (dep_bits << (j + 1))

    locus_idx = xp.arange(n, dtype=xp.int64)
    contributions = tables_d[locus_idx, keys]  # (2^N, N) fancy indexing
    fitnesses = contributions.mean(axis=1)

    if xp is not np:
        return xp.asnumpy(fitnesses)
    return fitnesses.astype(np.float64)


def gpu_compute_fitness_wmodel(
    n: int, mu: int, nu: int, ruggedness_perm: np.ndarray
) -> np.ndarray:
    """Vectorized W-model fitness for all 2^N solutions."""
    xp = _get_xp()
    n_eff = n // mu
    bits = _all_bit_vectors(n, xp)  # (2^N, N)
    perm = xp.asarray(ruggedness_perm)

    # Neutrality: majority-vote groups of mu bits
    grouped = bits[:, :n_eff * mu].reshape(-1, n_eff, mu)
    group_sums = grouped.sum(axis=2)
    reduced = (group_sums > mu / 2).astype(xp.int32)  # (2^N, n_eff)

    # Epistasis: XOR each bit with its predecessor in blocks of nu
    if nu > 1:
        n_blocks = n_eff // nu
        for b in range(n_blocks):
            start = b * nu
            for j in range(nu - 1, 0, -1):
                reduced[:, start + j] ^= reduced[:, start + j - 1]

    # OneMax + ruggedness permutation
    ones = reduced.sum(axis=1).astype(xp.int64)
    fitnesses = perm[ones] / n_eff

    if xp is not np:
        return xp.asnumpy(fitnesses)
    return fitnesses.astype(np.float64)


def gpu_compute_fitness_maxsat(
    n: int, clause_vars: np.ndarray, clause_signs: np.ndarray, n_clauses: int
) -> np.ndarray:
    """Vectorized MAX-SAT fitness for all 2^N solutions."""
    xp = _get_xp()
    bits = _all_bit_vectors(n, xp)  # (2^N, N)
    cvars = xp.asarray(clause_vars)    # (M, clause_length)
    csigns = xp.asarray(clause_signs)  # (M, clause_length)

    # Gather the variable bits for each clause literal: (2^N, M, clause_length)
    lit_bits = bits[:, cvars.ravel()].reshape(bits.shape[0], cvars.shape[0], cvars.shape[1])
    matches = lit_bits == csigns[xp.newaxis, :, :]
    clause_sat = matches.any(axis=2)  # (2^N, M)
    fitnesses = clause_sat.sum(axis=1).astype(xp.float64) / n_clauses

    if xp is not np:
        return xp.asnumpy(fitnesses)
    return fitnesses.astype(np.float64)


# ---------------------------------------------------------------------------
# Parallel hill climbing
# ---------------------------------------------------------------------------

_NUMBA_AVAILABLE: bool | None = None


def _has_numba_cuda() -> bool:
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            from numba import cuda  # noqa: F401
            if cuda.is_available():
                _NUMBA_AVAILABLE = True
            else:
                _NUMBA_AVAILABLE = False
        except Exception:
            _NUMBA_AVAILABLE = False
    return _NUMBA_AVAILABLE


def gpu_enumerate_optima(fitnesses: np.ndarray, n: int) -> np.ndarray:
    """Parallel hill climbing returning attractor[solution] -> local optimum.

    Uses Numba CUDA kernel when available, otherwise falls back to
    vectorized NumPy iteration.
    """
    if _has_numba_cuda() and is_gpu_available():
        return _cuda_hill_climb(fitnesses, n)
    return _vectorized_hill_climb(fitnesses, n)


def _cuda_hill_climb(fitnesses: np.ndarray, n: int) -> np.ndarray:
    """Run hill climbing on GPU with one CUDA thread per solution."""
    from numba import cuda

    @cuda.jit
    def _kernel(fit, n_bits, attractor):
        tid = cuda.grid(1)
        if tid >= fit.shape[0]:
            return
        current = tid
        while True:
            best = current
            best_f = fit[current]
            for bit in range(n_bits):
                nbr = current ^ (1 << bit)
                f = fit[nbr]
                if f > best_f:
                    best = nbr
                    best_f = f
            if best == current:
                break
            current = best
        attractor[tid] = current

    size = len(fitnesses)
    d_fit = cuda.to_device(fitnesses.astype(np.float64))
    d_attr = cuda.device_array(size, dtype=np.int64)

    threads = 256
    blocks = (size + threads - 1) // threads
    _kernel[blocks, threads](d_fit, n, d_attr)

    return d_attr.copy_to_host()


def _vectorized_hill_climb(fitnesses: np.ndarray, n: int) -> np.ndarray:
    """Vectorized NumPy hill climbing (CPU fallback).

    All 2^N solutions climb simultaneously; each iteration is O(N * 2^N)
    but uses fast array operations instead of Python loops.
    """
    size = len(fitnesses)
    current = np.arange(size, dtype=np.int64)

    max_steps = size
    for _ in range(max_steps):
        best = current.copy()
        best_f = fitnesses[current]

        for bit in range(n):
            nbrs = current ^ (1 << bit)
            nbr_f = fitnesses[nbrs]
            improving = nbr_f > best_f
            best = np.where(improving, nbrs, best)
            best_f = np.where(improving, nbr_f, best_f)

        if np.array_equal(best, current):
            break
        current = best

    return current
