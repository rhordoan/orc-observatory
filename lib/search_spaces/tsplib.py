"""TSPLIB instances as implicit 2-opt tour graphs (sampling-based)."""

from __future__ import annotations

import time

import numpy as np

from experiments.loaders.tsplib import load_tsplib, default_data_dir


class TSPLIBSearchSpace:
    """Large TSP from TSPLIB: dynamic tour registry + 2-opt neighborhood.

    Does not enumerate (n-1)! tours. Hill climbing and ORC operate on
    tours discovered via random-restart 2-opt.

    Uses bytes-key hashing and in-place numpy reversal for fast neighbor
    generation. Fitness computation is deferred (lazy) for neighbor tours
    and batch-computed on demand via vectorized numpy.
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
        self._key_to_idx: dict[bytes, int] = {}
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
            tour_arr = self._random_tour_arr()
            opt_arr = self._hill_climb_tour_arr(tour_arr)
            self._register(opt_arr, compute_fitness=True)
        print(f"    TSPLIB {instance_name}: {n_restarts} restarts -> {len(self._tours)} optima in {time.time()-t0:.1f}s", flush=True)

    def _register(self, tour_arr: np.ndarray, compute_fitness: bool = True) -> int:
        key = tour_arr.tobytes()
        existing = self._key_to_idx.get(key)
        if existing is not None:
            return existing
        idx = len(self._tours)
        self._key_to_idx[key] = idx
        self._tours.append(tuple(tour_arr.tolist()))
        if compute_fitness:
            self._fitness_cache[idx] = -self._tour_length_np(tour_arr)
        return idx

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
        f = self._fitness_cache.get(idx)
        if f is not None:
            return f
        f = self._tour_fitness(self._tours[idx])
        self._fitness_cache[idx] = f
        return f

    def neighbors(self, idx: int) -> np.ndarray:
        cached = self._nbr_cache.get(idx)
        if cached is not None:
            return cached.copy()
        t = np.asarray(self._tours[idx], dtype=np.intp)
        nbrs = np.empty(self._degree, dtype=np.intp)
        for k in range(self._degree):
            i, j = self._moves[k]
            t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
            nbrs[k] = self._register(t, compute_fitness=False)
            t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
        self._nbr_cache[idx] = nbrs
        return nbrs.copy()

    def solution_label(self, idx: int) -> str:
        return "\u2192".join(str(c) for c in self._tours[idx][:8]) + "..."

    def _tour_length_np(self, tour_arr: np.ndarray) -> float:
        return float(self._dist[tour_arr, np.roll(tour_arr, -1)].sum())

    def _tour_fitness(self, tour: tuple[int, ...]) -> float:
        return -self._tour_length_np(np.asarray(tour, dtype=np.intp))

    def _random_tour_arr(self) -> np.ndarray:
        rest = np.arange(1, self._n, dtype=np.intp)
        self._rng.shuffle(rest)
        tour = np.empty(self._n, dtype=np.intp)
        tour[0] = 0
        tour[1:] = rest
        return tour

    def _random_tour(self) -> tuple[int, ...]:
        return tuple(self._random_tour_arr().tolist())

    def _apply_2opt(self, tour: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
        return tour[: i + 1] + tour[j : i : -1] + tour[j + 1 :]

    def _hill_climb_tour_arr(self, t: np.ndarray) -> np.ndarray:
        """Best-improvement 2-opt with vectorized delta evaluation."""
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
        return t

    def _hill_climb_tour(self, tour: tuple[int, ...]) -> tuple[int, ...]:
        """Best-improvement 2-opt (tuple interface for backward compat)."""
        t = np.asarray(tour, dtype=np.intp)
        return tuple(self._hill_climb_tour_arr(t).tolist())

    def hill_climb_from(self, idx: int) -> int:
        """Vectorized delta-evaluation hill climb."""
        t = np.asarray(self._tours[idx], dtype=np.intp).copy()
        self._hill_climb_tour_arr(t)
        return self._register(t, compute_fitness=True)

    def neighbor_fitnesses(self, idx: int) -> np.ndarray:
        """Vectorized fitness of all 2-opt neighbors via numpy fancy indexing."""
        tour_arr = np.asarray(self._tours[idx], dtype=np.intp)
        base_length = -self.fitness(idx)
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
        self._batch_compute_missing_fitnesses()
        return np.array([self._fitness_cache[i] for i in range(len(self._tours))])

    def _batch_compute_missing_fitnesses(self) -> None:
        missing = [i for i in range(len(self._tours)) if i not in self._fitness_cache]
        if not missing:
            return
        batch_size = 10000
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            tours = np.array([self._tours[i] for i in batch], dtype=np.intp)
            rolled = np.roll(tours, -1, axis=1)
            lengths = self._dist[tours, rolled].sum(axis=1)
            for i_b, idx in enumerate(batch):
                self._fitness_cache[idx] = -float(lengths[i_b])
