"""Fitness-shuffle ablation: topology vs fitness signal.

Tests whether ORC's escape direction depends on the real fitness-topology
correlation.  All three conditions (real ORC, shuffled ORC, random) use
the *same* escape mechanism:

    1. Pick a direction (neighbor) using the respective strategy
    2. Hill-climb from that neighbor in the REAL space
    3. Check if REAL fitness improved

Only the direction-selection differs; this isolates the ORC signal.
"""

from __future__ import annotations

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.hill_climb import LocalOptimum, hill_climb
from lib.orc import compute_all_orc, batch_orc_gpu


class ShuffledFitnessSpace:
    """Wraps a space with permuted fitness values (topology unchanged)."""

    def __init__(self, base: SearchSpace, perm: np.ndarray) -> None:
        self._base = base
        self._perm = perm.astype(np.intp)
        self._fit_arr: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self._base.size

    @property
    def degree(self) -> int:
        return self._base.degree

    @property
    def neighbor_table(self):
        return getattr(self._base, "neighbor_table", None)

    def fitness(self, idx: int) -> float:
        idx = int(idx)
        if idx < len(self._perm):
            return self._base.fitness(int(self._perm[idx]))
        return self._base.fitness(idx)

    def neighbors(self, idx: int) -> np.ndarray:
        return self._base.neighbors(idx)

    def _ensure_fit_arr(self, min_size: int) -> None:
        if self._fit_arr is None or len(self._fit_arr) < min_size:
            batch_fn = getattr(self._base, "_batch_compute_missing_fitnesses", None)
            if batch_fn is not None:
                batch_fn()
            self._fit_arr = np.array(
                [self._base.fitness(i) for i in range(self._base.size)]
            )

    def neighbor_fitnesses(self, idx: int) -> np.ndarray:
        nbrs = self._base.neighbors(idx)
        self._ensure_fit_arr(int(nbrs.max()) + 1)
        perm = self._perm
        n_perm = len(perm)
        in_range = nbrs < n_perm
        permuted = np.empty_like(nbrs)
        permuted[in_range] = perm[nbrs[in_range]]
        permuted[~in_range] = nbrs[~in_range]
        safe = permuted < len(self._fit_arr)
        if np.all(safe):
            return self._fit_arr[permuted]
        result = np.empty(len(nbrs), dtype=np.float64)
        result[safe] = self._fit_arr[permuted[safe]]
        for i in np.where(~safe)[0]:
            result[i] = self._base.fitness(int(permuted[i]))
        return result

    def solution_label(self, idx: int) -> str:
        return self._base.solution_label(idx)


def make_shuffle_perm(space: SearchSpace, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(space.size, dtype=np.intp)
    rng.shuffle(perm)
    return perm


def _precompute_orc_for_space(
    space: SearchSpace,
    optima: list[LocalOptimum],
    gamma: float,
) -> dict[int, dict[int, float]]:
    """Batch ORC values for all optima on the given space."""
    max_nbrs = 60 if space.degree > 100 else None
    if space.degree > 30 and hasattr(space, "neighbor_table"):
        idx_arr = np.array([o.idx for o in optima], dtype=np.int64)
        return batch_orc_gpu(space, idx_arr, gamma,
                             max_neighbors=max_nbrs or space.degree)
    orc_vals: dict[int, dict[int, float]] = {}
    for i, opt in enumerate(optima):
        orc_vals[i] = compute_all_orc(space, opt.idx, gamma,
                                       max_neighbors=max_nbrs)
    return orc_vals


def _direction_escape_rate(
    real_space: SearchSpace,
    optima: list[LocalOptimum],
    orc_cache: dict[int, dict[int, float]],
) -> float:
    """Escape rate: for each optimum, follow the most-negative-ORC
    neighbor, hill-climb in the REAL space, check REAL fitness improvement."""
    global_best = max(o.fitness for o in optima)
    eligible = 0
    successes = 0
    for i, opt in enumerate(optima):
        if opt.fitness >= global_best - 1e-12:
            continue
        eligible += 1
        all_orc = orc_cache[i]
        best_nbr = min(all_orc, key=all_orc.get)
        dest = hill_climb(real_space, best_nbr)
        if real_space.fitness(dest) > opt.fitness:
            successes += 1
    return 100.0 * successes / max(eligible, 1)


def _random_escape_rate(
    real_space: SearchSpace,
    optima: list[LocalOptimum],
    n_trials: int = 30,
    seed: int = 0,
) -> float:
    """Control: random neighbor, hill-climb in real space."""
    rng = np.random.default_rng(seed)
    global_best = max(o.fitness for o in optima)
    eligible = 0
    successes = 0.0
    for opt in optima:
        if opt.fitness >= global_best - 1e-12:
            continue
        eligible += 1
        hits = 0
        nbrs = real_space.neighbors(opt.idx)
        for _ in range(n_trials):
            y = int(rng.choice(nbrs))
            dest = hill_climb(real_space, y)
            if real_space.fitness(dest) > opt.fitness:
                hits += 1
        successes += hits / n_trials
    return 100.0 * successes / max(eligible, 1)


def fitness_shuffle_ablation(
    space: SearchSpace,
    optima: list[LocalOptimum],
    n_shuffles: int = 3,
    base_seed: int = 0,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Compare ORC escape rate with real vs shuffled fitness.

    All conditions hill-climb in the REAL space and check REAL fitness.
    Only the direction-selection (which neighbor to perturb toward) differs:
      - real_orc: ORC computed on real fitness
      - shuffled_orc: ORC computed on shuffled fitness (mean of n_shuffles)
      - random: random neighbor (control)
    """
    real_orc = _precompute_orc_for_space(space, optima, gamma)
    real_pct = _direction_escape_rate(space, optima, real_orc)

    shuffled_pcts = []
    for s in range(n_shuffles):
        perm = make_shuffle_perm(space, base_seed + s + 1)
        shuf_space = ShuffledFitnessSpace(space, perm)
        shuf_orc = _precompute_orc_for_space(shuf_space, optima, gamma)
        shuf_pct = _direction_escape_rate(space, optima, shuf_orc)
        shuffled_pcts.append(shuf_pct)

    random_pct = _random_escape_rate(space, optima, seed=base_seed)
    mean_shuf = float(np.mean(shuffled_pcts))

    return {
        "orc_escape_real_pct": real_pct,
        "orc_escape_shuffled_mean_pct": mean_shuf,
        "orc_escape_random_pct": random_pct,
        "real_over_shuffled_ratio": real_pct / max(mean_shuf, 1e-9),
        "real_over_random_ratio": real_pct / max(random_pct, 1e-9),
        "n_shuffles": n_shuffles,
    }
