"""Fitness-shuffle ablation: topology vs fitness signal."""

from __future__ import annotations

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.hill_climb import LocalOptimum
from experiments.metrics import escape_rate


class ShuffledFitnessSpace:
    """Wraps a space with permuted fitness values (topology unchanged).

    For dynamically growing spaces (TSP/QAP), solutions registered after
    the permutation was created fall back to their real fitness.  ORC
    escape from the *original* optima still sees shuffled values.
    """

    def __init__(self, base: SearchSpace, perm: np.ndarray) -> None:
        self._base = base
        self._perm = perm.astype(np.intp)

    @property
    def name(self) -> str:
        return self._base.name + " (shuffled fitness)"

    @property
    def size(self) -> int:
        return self._base.size

    @property
    def degree(self) -> int:
        return self._base.degree

    def fitness(self, idx: int) -> float:
        idx = int(idx)
        if idx < len(self._perm):
            return self._base.fitness(int(self._perm[idx]))
        return self._base.fitness(idx)

    def neighbors(self, idx: int) -> np.ndarray:
        return self._base.neighbors(idx)

    def solution_label(self, idx: int) -> str:
        return self._base.solution_label(idx)


def make_shuffle_perm(space: SearchSpace, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(space.size, dtype=np.intp)
    rng.shuffle(perm)
    return perm


def fitness_shuffle_ablation(
    space: SearchSpace,
    optima: list[LocalOptimum],
    n_shuffles: int = 5,
    base_seed: int = 0,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Compare ORC escape rate on real vs shuffled fitness."""
    real = escape_rate(space, optima, "orc", gamma=gamma)
    shuffled_rates = []
    for s in range(n_shuffles):
        perm = make_shuffle_perm(space, base_seed + s + 1)
        shuf_space = ShuffledFitnessSpace(space, perm)
        shuf_optima = [
            LocalOptimum(idx=o.idx, fitness=shuf_space.fitness(o.idx), basin=o.basin)
            for o in optima
        ]
        sh = escape_rate(shuf_space, shuf_optima, "orc", gamma=gamma)
        shuffled_rates.append(sh["escape_pct"])

    mean_shuf = float(np.mean(shuffled_rates))
    ratio = real["escape_pct"] / max(mean_shuf, 1e-9)
    return {
        "orc_escape_real_pct": real["escape_pct"],
        "orc_escape_shuffled_mean_pct": mean_shuf,
        "real_over_shuffled_ratio": ratio,
        "n_shuffles": n_shuffles,
    }
