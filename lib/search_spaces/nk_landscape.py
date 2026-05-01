"""NK landscape search space with bit-flip neighborhood."""

from __future__ import annotations

import numpy as np


class NKSearchSpace:
    """Kauffman NK landscape on the binary hypercube {0,1}^N.

    Each of the N loci contributes to fitness based on its own value and
    the values of K epistatic partners.  The neighborhood is single-bit
    flip, giving degree k = N.
    """

    def __init__(self, n: int, k: int, seed: int | None = None) -> None:
        if not 1 <= k < n:
            raise ValueError(f"Need 1 <= K < N, got K={k}, N={n}")
        self._n = n
        self._k = k
        self._rng = np.random.default_rng(seed)

        # Epistatic interactions: for each locus i, pick K other loci
        self._deps = np.zeros((n, k), dtype=np.intp)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            self._deps[i] = self._rng.choice(others, size=k, replace=False)

        # Lookup tables: for each locus, 2^(K+1) random contributions
        self._tables = self._rng.random((n, 2 ** (k + 1)))

        # Pre-compute all fitness values for fast lookup
        self._size = 2**n
        self._fitnesses = np.array(
            [self._compute_fitness(idx) for idx in range(self._size)]
        )

    # -- Protocol implementation ------------------------------------------

    @property
    def name(self) -> str:
        return f"NK Landscape (N={self._n}, K={self._k})"

    @property
    def size(self) -> int:
        return self._size

    @property
    def degree(self) -> int:
        return self._n

    def fitness(self, idx: int) -> float:
        return float(self._fitnesses[idx])

    def neighbors(self, idx: int) -> np.ndarray:
        """Flip each of the N bits to get N neighbors."""
        return np.array([idx ^ (1 << bit) for bit in range(self._n)], dtype=np.intp)

    def solution_label(self, idx: int) -> str:
        return format(idx, f"0{self._n}b")

    # -- Internals --------------------------------------------------------

    def _compute_fitness(self, idx: int) -> float:
        bits = np.array(
            [(idx >> bit) & 1 for bit in range(self._n)], dtype=np.intp
        )
        total = 0.0
        for i in range(self._n):
            # Build the index into the lookup table for locus i:
            # concatenate (bit_i, bit_dep0, bit_dep1, ..., bit_depK)
            key = bits[i]
            for j, dep in enumerate(self._deps[i]):
                key |= bits[dep] << (j + 1)
            total += self._tables[i, key]
        return total / self._n

    @property
    def fitnesses(self) -> np.ndarray:
        """Direct access to the full fitness array (for testing/metrics)."""
        return self._fitnesses
