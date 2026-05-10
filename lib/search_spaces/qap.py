"""Quadratic Assignment Problem search space with transposition-swap neighborhood."""

from __future__ import annotations

from itertools import permutations

import numpy as np


class QAPSearchSpace:
    """Random QAP instance with transposition-swap neighborhood.

    Given *n* facilities and *n* locations with random distance and flow
    matrices, fitness is the negated assignment cost normalized to [0, 1].
    The neighborhood is all pairwise transpositions, giving degree n(n-1)/2.
    """

    def __init__(
        self, n: int, seed: int | None = None, use_gpu: bool = False
    ) -> None:
        if n < 4:
            raise ValueError(f"Need n >= 4, got {n}")
        self._n = n
        rng = np.random.default_rng(seed)

        self._dist = rng.integers(1, 100, size=(n, n)).astype(np.float64)
        self._flow = rng.integers(1, 100, size=(n, n)).astype(np.float64)
        np.fill_diagonal(self._dist, 0)
        np.fill_diagonal(self._flow, 0)
        self._dist = (self._dist + self._dist.T) / 2
        self._flow = (self._flow + self._flow.T) / 2

        self._perms: list[tuple[int, ...]] = list(permutations(range(n)))
        self._perm_to_idx: dict[tuple[int, ...], int] = {
            p: i for i, p in enumerate(self._perms)
        }
        self._size = len(self._perms)  # n!

        self._swap_pairs: list[tuple[int, int]] = [
            (i, j) for i in range(n) for j in range(i + 1, n)
        ]
        self._degree = len(self._swap_pairs)  # n*(n-1)/2

        perms_arr = np.asarray(self._perms, dtype=np.intp)
        raw_costs = np.zeros(self._size, dtype=np.float64)
        for s in range(self._size):
            p = perms_arr[s]
            raw_costs[s] = float(np.sum(self._dist * self._flow[np.ix_(p, p)]))

        lo, hi = raw_costs.min(), raw_costs.max()
        span = hi - lo
        if span < 1e-12:
            self._fitnesses = np.ones(self._size)
        else:
            self._fitnesses = (hi - raw_costs) / span

        self._neighbor_table = np.empty(
            (self._size, self._degree), dtype=np.intp
        )
        for idx, perm in enumerate(self._perms):
            for k, (i, j) in enumerate(self._swap_pairs):
                swapped = list(perm)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                self._neighbor_table[idx, k] = self._perm_to_idx[tuple(swapped)]

    # -- SearchSpace protocol ------------------------------------------------

    @property
    def name(self) -> str:
        return f"QAP Swap (n={self._n})"

    @property
    def size(self) -> int:
        return self._size

    @property
    def degree(self) -> int:
        return self._degree

    def fitness(self, idx: int) -> float:
        return float(self._fitnesses[idx])

    def neighbors(self, idx: int) -> np.ndarray:
        return self._neighbor_table[idx].copy()

    def solution_label(self, idx: int) -> str:
        return "[" + ",".join(str(v) for v in self._perms[idx]) + "]"

    # -- Extra attributes ----------------------------------------------------

    @property
    def fitnesses(self) -> np.ndarray:
        return self._fitnesses

    @property
    def neighbor_table(self) -> np.ndarray:
        return self._neighbor_table

    def solution_distance(self, a: int, b: int) -> int:
        """Cayley distance: minimum transpositions to transform a into b."""
        pa, pb = self._perms[a], self._perms[b]
        n = self._n
        inv_a = [0] * n
        for i, v in enumerate(pa):
            inv_a[v] = i
        composed = [pb[inv_a[i]] for i in range(n)]
        visited = [False] * n
        n_cycles = 0
        for i in range(n):
            if not visited[i]:
                n_cycles += 1
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = composed[j]
        return n - n_cycles
