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
        swap_arr = np.array(self._swap_pairs, dtype=np.intp)
        self._swap_r = swap_arr[:, 0]
        self._swap_s = swap_arr[:, 1]

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

    def hill_climb_from(self, idx: int) -> int:
        """Vectorized delta-evaluation hill climb."""
        p = np.asarray(self._perms[idx], dtype=np.intp)
        improved = True
        while improved:
            deltas = self._all_swap_deltas(p)
            best_k = int(np.argmin(deltas))
            if deltas[best_k] < -1e-12:
                r, s = int(self._swap_r[best_k]), int(self._swap_s[best_k])
                p[r], p[s] = p[s], p[r]
            else:
                improved = False
        return self._register(tuple(p))

    def neighbor_fitnesses(self, idx: int) -> np.ndarray:
        """Vectorized fitness of all swap neighbors."""
        p = np.asarray(self._perms[idx], dtype=np.intp)
        base_cost = -self._fitness_cache[idx]
        return -(base_cost + self._all_swap_deltas(p))

    def _all_swap_deltas(self, p: np.ndarray) -> np.ndarray:
        """Vectorized O(n) delta for every swap pair simultaneously."""
        n = self._n
        d, f = self._dist, self._flow
        R, S = self._swap_r, self._swap_s
        pr, ps = p[R], p[S]
        deltas = np.empty(len(R), dtype=np.float64)
        for idx in range(len(R)):
            r, s = int(R[idx]), int(S[idx])
            pr_v, ps_v = int(pr[idx]), int(ps[idx])
            mask = np.ones(n, dtype=bool)
            mask[r] = mask[s] = False
            k = np.where(mask)[0]
            pk = p[k]
            delta = float(np.sum(
                (d[r, k] - d[s, k]) * (f[ps_v, pk] - f[pr_v, pk]) +
                (d[k, r] - d[k, s]) * (f[pk, ps_v] - f[pk, pr_v])
            ))
            delta += float((d[r, s] - d[s, r]) * (f[ps_v, pr_v] - f[pr_v, ps_v]))
            deltas[idx] = delta
        return deltas

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
        """Best-improvement swap with vectorized delta evaluation."""
        p = np.asarray(perm, dtype=np.intp)
        improved = True
        while improved:
            deltas = self._all_swap_deltas(p)
            best_k = int(np.argmin(deltas))
            if deltas[best_k] < -1e-12:
                r, s = int(self._swap_r[best_k]), int(self._swap_s[best_k])
                p[r], p[s] = p[s], p[r]
            else:
                improved = False
        return tuple(p)
