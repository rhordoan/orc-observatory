"""TSPLIB instances as implicit 2-opt tour graphs (sampling-based)."""

from __future__ import annotations

import time

import numpy as np

from experiments.loaders.tsplib import load_tsplib, default_data_dir


class TSPLIBSearchSpace:
    """Large TSP from TSPLIB: dynamic tour registry + 2-opt neighborhood.

    Does not enumerate (n-1)! tours. Hill climbing and ORC operate on
    tours discovered via random-restart 2-opt.
    """

    def __init__(
        self,
        instance_name: str,
        data_dir: str | None = None,
        n_restarts: int = 1500,
        seed: int | None = None,
        use_gpu: bool = False,
    ) -> None:
        del use_gpu
        ddir = default_data_dir() if data_dir is None else __import__("pathlib").Path(data_dir)
        self.coords, self._dist, self._n = load_tsplib(instance_name, ddir)
        self._name = f"TSPLIB {instance_name} (n={self._n})"
        self._rng = np.random.default_rng(seed)

        self._tours: list[tuple[int, ...]] = []
        self._tour_to_idx: dict[tuple[int, ...], int] = {}
        self._moves = [
            (i, j)
            for i in range(self._n)
            for j in range(i + 2, self._n)
            if not (i == 0 and j == self._n - 1)
        ]
        self._degree = len(self._moves)
        self._fitness_cache: dict[int, float] = {}

        t0 = time.time()
        for _ in range(n_restarts):
            tour = self._random_tour()
            opt = self._hill_climb_tour(tour)
            self._register(opt)
        print(f"    TSPLIB {instance_name}: {n_restarts} restarts -> {len(self._tours)} optima in {time.time()-t0:.1f}s", flush=True)

    def _register(self, tour: tuple[int, ...]) -> int:
        if tour not in self._tour_to_idx:
            idx = len(self._tours)
            self._tour_to_idx[tour] = idx
            self._tours.append(tour)
            self._fitness_cache[idx] = self._tour_fitness(tour)
        return self._tour_to_idx[tour]

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return len(self._tours)

    @property
    def degree(self) -> int:
        return self._degree

    def fitness(self, idx: int) -> float:
        return self._fitness_cache[idx]

    def neighbors(self, idx: int) -> np.ndarray:
        tour = self._tours[idx]
        nbrs = []
        for i, j in self._moves:
            new_tour = tour[: i + 1] + tour[j : i : -1] + tour[j + 1 :]
            nbrs.append(self._register(new_tour))
        return np.array(nbrs, dtype=np.intp)

    def solution_label(self, idx: int) -> str:
        return "\u2192".join(str(c) for c in self._tours[idx][:8]) + "..."

    def _tour_length_np(self, tour_arr: np.ndarray) -> float:
        return float(self._dist[tour_arr, np.roll(tour_arr, -1)].sum())

    def _tour_fitness(self, tour: tuple[int, ...]) -> float:
        return -self._tour_length_np(np.asarray(tour, dtype=np.intp))

    def _random_tour(self) -> tuple[int, ...]:
        rest = list(range(1, self._n))
        self._rng.shuffle(rest)
        return (0, *rest)

    def _apply_2opt(self, tour: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
        return tour[: i + 1] + tour[j : i : -1] + tour[j + 1 :]

    def _hill_climb_tour(self, tour: tuple[int, ...]) -> tuple[int, ...]:
        """Best-improvement 2-opt with O(1) delta evaluation per move."""
        t = list(tour)
        n = self._n
        d = self._dist
        length = float(self._dist[np.asarray(t, dtype=np.intp),
                                   np.roll(np.asarray(t, dtype=np.intp), -1)].sum())
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_i = best_j = -1
            for i, j in self._moves:
                ci, cj = t[i], t[j]
                ci1 = t[(i + 1) % n]
                cj1 = t[(j + 1) % n]
                old = d[ci, ci1] + d[cj, cj1]
                new = d[ci, cj] + d[ci1, cj1]
                delta = new - old
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
            if best_i >= 0:
                t[best_i + 1 : best_j + 1] = t[best_i + 1 : best_j + 1][::-1]
                length += best_delta
                improved = True
        return tuple(t)

    @property
    def neighbor_table(self):
        return None

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([self._fitness_cache[i] for i in range(len(self._tours))])
