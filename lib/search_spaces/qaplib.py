"""QAPLIB instances as implicit swap-neighborhood graphs (sampling-based)."""

from __future__ import annotations

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

        for _ in range(n_restarts):
            p = tuple(self._rng.permutation(self._n))
            opt = self._hill_climb_perm(p)
            self._register(opt)

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
        perm = list(self._perms[idx])
        nbrs = []
        for i, j in self._swap_pairs:
            swapped = perm.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            nbrs.append(self._register(tuple(swapped)))
        return np.array(nbrs, dtype=np.intp)

    def solution_label(self, idx: int) -> str:
        return str(self._perms[idx][: min(8, self._n)])

    def _perm_cost(self, perm: tuple[int, ...]) -> float:
        p = np.asarray(perm, dtype=np.intp)
        return float(np.sum(self._dist * self._flow[np.ix_(p, p)]))

    def _perm_fitness(self, perm: tuple[int, ...]) -> float:
        return -self._perm_cost(perm)

    def _hill_climb_perm(self, perm: tuple[int, ...]) -> tuple[int, ...]:
        current = perm
        current_fit = self._perm_fitness(current)
        improved = True
        while improved:
            improved = False
            best = current
            best_fit = current_fit
            pl = list(current)
            for i, j in self._swap_pairs:
                pl[i], pl[j] = pl[j], pl[i]
                nbr = tuple(pl)
                f = self._perm_fitness(nbr)
                pl[i], pl[j] = pl[j], pl[i]
                if f > best_fit:
                    best_fit = f
                    best = nbr
                    improved = True
            current = best
            current_fit = best_fit
        return current
