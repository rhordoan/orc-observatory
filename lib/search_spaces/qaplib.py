"""QAPLIB instances as implicit swap-neighborhood graphs (sampling-based)."""

from __future__ import annotations

import time

import numpy as np

from experiments.loaders.qaplib import load_qaplib, default_data_dir


class QAPLIBSearchSpace:
    """QAP from QAPLIB: dynamic permutation registry + transposition swaps."""

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
        self._dist, self._flow, self._n = load_qaplib(instance_name, ddir)
        self._name = f"QAPLIB {instance_name} (n={self._n})"
        self._rng = np.random.default_rng(seed)

        self._perms: list[tuple[int, ...]] = []
        self._perm_to_idx: dict[tuple[int, ...], int] = {}
        self._swap_pairs = [(i, j) for i in range(self._n) for j in range(i + 1, self._n)]
        self._degree = len(self._swap_pairs)
        self._fitness_cache: dict[int, float] = {}
        self._nbr_cache: dict[int, np.ndarray] = {}

        t0 = time.time()
        for _ in range(n_restarts):
            p = tuple(self._rng.permutation(self._n))
            opt = self._hill_climb_perm(p)
            self._register(opt)
        print(f"    QAPLIB {instance_name}: {n_restarts} restarts -> {len(self._perms)} optima in {time.time()-t0:.1f}s", flush=True)

    def _register(self, perm: tuple[int, ...]) -> int:
        if perm not in self._perm_to_idx:
            idx = len(self._perms)
            self._perm_to_idx[perm] = idx
            self._perms.append(perm)
            self._fitness_cache[idx] = self._perm_fitness(perm)
        return self._perm_to_idx[perm]

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
        return str(self._perms[idx][: min(8, self._n)])

    def neighbor_fitnesses(self, idx: int) -> np.ndarray:
        """Fitness of all swap neighbors via O(n) delta per move."""
        p = list(self._perms[idx])
        base_cost = -self._fitness_cache[idx]
        result = np.empty(self._degree, dtype=np.float64)
        for k, (r, s) in enumerate(self._swap_pairs):
            delta = self._swap_delta(p, r, s)
            result[k] = -(base_cost + delta)
        return result

    @property
    def neighbor_table(self):
        return None

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([self._fitness_cache[i] for i in range(len(self._perms))])

    def _perm_cost(self, perm: tuple[int, ...]) -> float:
        p = np.asarray(perm, dtype=np.intp)
        return float(np.sum(self._dist * self._flow[np.ix_(p, p)]))

    def _perm_fitness(self, perm: tuple[int, ...]) -> float:
        return -self._perm_cost(perm)

    def _swap_delta(self, p: list[int], r: int, s: int) -> float:
        """O(n) delta for swapping positions r and s in permutation p."""
        n = self._n
        d, f = self._dist, self._flow
        pr, ps = p[r], p[s]
        delta = 0.0
        for k in range(n):
            if k == r or k == s:
                continue
            pk = p[k]
            delta += (d[r, k] - d[s, k]) * (f[ps, pk] - f[pr, pk])
            delta += (d[k, r] - d[k, s]) * (f[pk, ps] - f[pk, pr])
        delta += (d[r, s] - d[s, r]) * (f[ps, pr] - f[pr, ps])
        return delta

    def _hill_climb_perm(self, perm: tuple[int, ...]) -> tuple[int, ...]:
        """Best-improvement swap with O(n) delta evaluation per move."""
        p = list(perm)
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_r = best_s = -1
            for r, s in self._swap_pairs:
                delta = self._swap_delta(p, r, s)
                if delta < best_delta:
                    best_delta = delta
                    best_r, best_s = r, s
            if best_r >= 0:
                p[best_r], p[best_s] = p[best_s], p[best_r]
                improved = True
        return tuple(p)
