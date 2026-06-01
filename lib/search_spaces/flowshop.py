"""Permutation flowshop scheduling as implicit swap-neighborhood graph (sampling-based).

Taillard-style benchmark instances: n jobs on m machines, minimize makespan.
Neighborhood: pairwise job swap (same structure as QAP transposition).
Disjoint neighborhoods guarantee ORC sorted-matching is exact.
"""

from __future__ import annotations

import time

import numpy as np

from experiments.loaders.flowshop import load_flowshop, default_data_dir


class FlowshopSearchSpace:
    """Permutation flowshop with pairwise swap neighborhood.

    Fitness = -makespan (maximize = minimize makespan).
    Same sampling approach as TSPLIB/QAPLIB: random-restart hill climb
    to discover local optima, dynamic registry for lazy neighbor expansion.
    """

    def __init__(
        self,
        instance_name: str,
        data_dir: str | None = None,
        n_restarts: int = 1000,
        seed: int | None = None,
        use_gpu: bool = False,
    ) -> None:
        del use_gpu
        ddir = default_data_dir() if data_dir is None else __import__("pathlib").Path(data_dir)
        self._pt, self._n_jobs, self._n_machines = load_flowshop(instance_name, ddir)
        self._name = f"Flowshop {instance_name} ({self._n_jobs}x{self._n_machines})"
        self._rng = np.random.default_rng(seed)

        self._perms: list[tuple[int, ...]] = []
        self._perm_to_idx: dict[tuple[int, ...], int] = {}
        self._swap_pairs = [
            (i, j) for i in range(self._n_jobs) for j in range(i + 1, self._n_jobs)
        ]
        self._degree = len(self._swap_pairs)
        self._fitness_cache: dict[int, float] = {}
        self._nbr_cache: dict[int, np.ndarray] = {}

        swap_arr = np.array(self._swap_pairs, dtype=np.intp)
        self._swap_r = swap_arr[:, 0]
        self._swap_s = swap_arr[:, 1]

        t0 = time.time()
        for _ in range(n_restarts):
            perm = tuple(self._rng.permutation(self._n_jobs).tolist())
            opt = self._hill_climb_perm(perm)
            self._register(opt)
        print(
            f"    Flowshop {instance_name}: {n_restarts} restarts -> "
            f"{len(self._perms)} optima in {time.time()-t0:.1f}s",
            flush=True,
        )

    def _register(self, perm: tuple[int, ...]) -> int:
        if perm not in self._perm_to_idx:
            idx = len(self._perms)
            self._perm_to_idx[perm] = idx
            self._perms.append(perm)
            self._fitness_cache[idx] = float(-self._makespan(perm))
        return self._perm_to_idx[perm]

    def _makespan(self, perm: tuple[int, ...] | list[int] | np.ndarray) -> int:
        """Compute makespan for a job permutation.

        Uses a tight loop over the completion-time matrix.
        """
        pt = self._pt
        n_j = len(perm)
        n_m = self._n_machines
        c = np.zeros((n_j + 1, n_m), dtype=np.int64)
        for i in range(n_j):
            job = perm[i] if isinstance(perm[i], int) else int(perm[i])
            c[i + 1, 0] = c[i, 0] + pt[job, 0]
            for m in range(1, n_m):
                c[i + 1, m] = max(c[i + 1, m - 1], c[i, m]) + pt[job, m]
        return int(c[n_j, n_m - 1])

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return len(self._perms)

    @property
    def degree(self) -> int:
        return self._degree

    def fitness(self, idx: int) -> float:
        return self._fitness_cache[idx]

    def neighbors(self, idx: int) -> np.ndarray:
        cached = self._nbr_cache.get(idx)
        if cached is not None:
            return cached.copy()
        perm = list(self._perms[idx])
        nbrs = np.empty(self._degree, dtype=np.intp)
        for k, (i, j) in enumerate(self._swap_pairs):
            perm[i], perm[j] = perm[j], perm[i]
            nbrs[k] = self._register(tuple(perm))
            perm[i], perm[j] = perm[j], perm[i]
        self._nbr_cache[idx] = nbrs
        return nbrs.copy()

    def solution_label(self, idx: int) -> str:
        return str(self._perms[idx][: min(8, self._n_jobs)])

    def hill_climb_from(self, idx: int) -> int:
        """Best-improvement swap hill climb."""
        p = list(self._perms[idx])
        base_ms = self._makespan(p)
        improved = True
        while improved:
            best_ms = base_ms
            best_k = -1
            for k, (i, j) in enumerate(self._swap_pairs):
                p[i], p[j] = p[j], p[i]
                ms = self._makespan(p)
                p[i], p[j] = p[j], p[i]
                if ms < best_ms:
                    best_ms = ms
                    best_k = k
            if best_k >= 0:
                i, j = self._swap_pairs[best_k]
                p[i], p[j] = p[j], p[i]
                base_ms = best_ms
            else:
                improved = False
        return self._register(tuple(p))

    def neighbor_fitnesses(self, idx: int) -> np.ndarray:
        """Compute fitness of all swap neighbors."""
        perm = list(self._perms[idx])
        fits = np.empty(self._degree, dtype=np.float64)
        for k, (i, j) in enumerate(self._swap_pairs):
            perm[i], perm[j] = perm[j], perm[i]
            fits[k] = float(-self._makespan(perm))
            perm[i], perm[j] = perm[j], perm[i]
        return fits

    @property
    def neighbor_table(self):
        return None

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([self._fitness_cache[i] for i in range(len(self._perms))])

    def _hill_climb_perm(self, perm: tuple[int, ...]) -> tuple[int, ...]:
        """Best-improvement swap hill climb."""
        p = list(perm)
        base_ms = self._makespan(p)
        improved = True
        while improved:
            best_ms = base_ms
            best_k = -1
            for k, (i, j) in enumerate(self._swap_pairs):
                p[i], p[j] = p[j], p[i]
                ms = self._makespan(p)
                p[i], p[j] = p[j], p[i]
                if ms < best_ms:
                    best_ms = ms
                    best_k = k
            if best_k >= 0:
                i, j = self._swap_pairs[best_k]
                p[i], p[j] = p[j], p[i]
                base_ms = best_ms
            else:
                improved = False
        return tuple(p)
