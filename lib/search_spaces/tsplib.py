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
        self._nbr_cache: dict[int, np.ndarray] = {}

        moves_arr = np.array(self._moves, dtype=np.intp)
        self._moves_i = moves_arr[:, 0]
        self._moves_j = moves_arr[:, 1]
        self._moves_i1 = (self._moves_i + 1) % self._n
        self._moves_j1 = (self._moves_j + 1) % self._n

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
        cached = self._nbr_cache.get(idx)
        if cached is not None:
            return cached.copy()
        tour = self._tours[idx]
        nbrs = np.empty(self._degree, dtype=np.intp)
        for k, (i, j) in enumerate(self._moves):
            new_tour = tour[: i + 1] + tour[j : i : -1] + tour[j + 1 :]
            nbrs[k] = self._register(new_tour)
        self._nbr_cache[idx] = nbrs
        return nbrs.copy()

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
        """Best-improvement 2-opt with vectorized delta evaluation."""
        t = np.asarray(tour, dtype=np.intp)
        d = self._dist
        improved = True
        while improved:
            ci = t[self._moves_i]
            cj = t[self._moves_j]
            ci1 = t[self._moves_i1]
            cj1 = t[self._moves_j1]
            delta = d[ci, cj] + d[ci1, cj1] - d[ci, ci1] - d[cj, cj1]
            best_k = int(np.argmin(delta))
            if delta[best_k] < -1e-12:
                bi = int(self._moves_i[best_k])
                bj = int(self._moves_j[best_k])
                t[bi + 1 : bj + 1] = t[bi + 1 : bj + 1][::-1]
            else:
                improved = False
        return tuple(t)

    def hill_climb_from(self, idx: int) -> int:
        """Vectorized delta-evaluation hill climb."""
        t = np.asarray(self._tours[idx], dtype=np.intp)
        d = self._dist
        improved = True
        while improved:
            ci = t[self._moves_i]
            cj = t[self._moves_j]
            ci1 = t[self._moves_i1]
            cj1 = t[self._moves_j1]
            delta = d[ci, cj] + d[ci1, cj1] - d[ci, ci1] - d[cj, cj1]
            best_k = int(np.argmin(delta))
            if delta[best_k] < -1e-12:
                bi = int(self._moves_i[best_k])
                bj = int(self._moves_j[best_k])
                t[bi + 1 : bj + 1] = t[bi + 1 : bj + 1][::-1]
            else:
                improved = False
        return self._register(tuple(t))

    def neighbor_fitnesses(self, idx: int) -> np.ndarray:
        """Vectorized fitness of all 2-opt neighbors via numpy fancy indexing."""
        tour_arr = np.asarray(self._tours[idx], dtype=np.intp)
        base_length = -self._fitness_cache[idx]
        ci = tour_arr[self._moves_i]
        cj = tour_arr[self._moves_j]
        ci1 = tour_arr[self._moves_i1]
        cj1 = tour_arr[self._moves_j1]
        delta = self._dist[ci, cj] + self._dist[ci1, cj1] - self._dist[ci, ci1] - self._dist[cj, cj1]
        return -(base_length + delta)

    @property
    def neighbor_table(self):
        return None

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([self._fitness_cache[i] for i in range(len(self._tours))])
