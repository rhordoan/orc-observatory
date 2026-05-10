"""Asymmetric TSP search space with 2-opt neighborhood."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np


class TSPSearchSpace:
    """Random asymmetric TSP with 2-opt neighborhood.

    Uses a random asymmetric distance matrix (non-Euclidean) to produce
    rugged landscapes with many local optima.  City coordinates are still
    generated for the detail-panel tour mini-map but do not determine
    the actual distances.

    City 0 is fixed at position 0 in every tour, giving (n-1)! distinct tours.
    The 2-opt move reverses a contiguous sub-segment, yielding degree n(n-3)/2.
    """

    def __init__(
        self, n_cities: int, seed: int | None = None, use_gpu: bool = False
    ) -> None:
        if n_cities < 5:
            raise ValueError(f"Need >= 5 cities, got {n_cities}")
        self._n = n_cities
        rng = np.random.default_rng(seed)

        self.coords = rng.random((n_cities, 2))

        d = rng.integers(1, 100, size=(n_cities, n_cities)).astype(np.float64)
        np.fill_diagonal(d, 0)
        self._dist = d

        rest = list(range(1, n_cities))
        self._tours: list[tuple[int, ...]] = [
            (0, *p) for p in permutations(rest)
        ]
        self._tour_to_idx: dict[tuple[int, ...], int] = {
            t: i for i, t in enumerate(self._tours)
        }
        self._size = len(self._tours)  # (n-1)!
        self._degree = n_cities * (n_cities - 3) // 2

        self._moves: list[tuple[int, int]] = []
        for i in range(n_cities):
            for j in range(i + 2, n_cities):
                if i == 0 and j == n_cities - 1:
                    continue
                self._moves.append((i, j))

        tours_arr = np.asarray(self._tours, dtype=np.intp)
        rolled = np.roll(tours_arr, -1, axis=1)
        raw_lengths = self._dist[tours_arr, rolled].sum(axis=1)

        lo, hi = raw_lengths.min(), raw_lengths.max()
        span = hi - lo
        if span < 1e-12:
            self._fitnesses = np.ones(self._size)
        else:
            self._fitnesses = (hi - raw_lengths) / span

        self._neighbor_table = np.empty(
            (self._size, self._degree), dtype=np.intp
        )
        for idx, tour in enumerate(self._tours):
            for k, (i, j) in enumerate(self._moves):
                new_tour = tour[: i + 1] + tour[j : i : -1] + tour[j + 1 :]
                self._neighbor_table[idx, k] = self._tour_to_idx[new_tour]

    # -- SearchSpace protocol ------------------------------------------------

    @property
    def name(self) -> str:
        return f"ATSP 2-opt (n={self._n})"

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
        return "\u2192".join(str(c) for c in self._tours[idx])

    # -- Extra attributes used by generic stack ------------------------------

    @property
    def fitnesses(self) -> np.ndarray:
        return self._fitnesses

    @property
    def neighbor_table(self) -> np.ndarray:
        return self._neighbor_table

    def tour_for_idx(self, idx: int) -> tuple[int, ...]:
        """Return the tour tuple for a given solution index."""
        return self._tours[idx]

    def solution_distance(self, a: int, b: int) -> int:
        """Number of directed edges present in tour *a* but absent from tour *b*."""
        n = self._n
        ta, tb = self._tours[a], self._tours[b]
        ea = {(ta[i], ta[(i + 1) % n]) for i in range(n)}
        eb = {(tb[i], tb[(i + 1) % n]) for i in range(n)}
        return len(ea - eb)
