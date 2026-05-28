"""Classical Fitness Landscape Analysis features for algorithm selection.

Provides the standard scalar FLA features as a baseline to compare
against OTG-derived structural features.
"""

from __future__ import annotations

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.hill_climb import LocalOptimum


def fdc(space: SearchSpace, optima: list[LocalOptimum]) -> float:
    """Fitness-Distance Correlation: Pearson correlation between fitness
    and Hamming/swap distance to the best-known optimum."""
    if len(optima) < 3:
        return 0.0
    best = max(optima, key=lambda o: o.fitness)
    fitnesses = np.array([o.fitness for o in optima])
    distances = np.array([_distance(space, o.idx, best.idx) for o in optima])
    if distances.std() < 1e-12 or fitnesses.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(fitnesses, distances)[0, 1])


def _distance(space: SearchSpace, a: int, b: int) -> float:
    """Hamming distance for bit-flip spaces, position distance for permutations."""
    if hasattr(space, "solution_label"):
        la = space.solution_label(a)
        lb = space.solution_label(b)
        if all(c in "01" for c in la) and all(c in "01" for c in lb):
            return sum(ca != cb for ca, cb in zip(la, lb))
    return abs(a - b)


def autocorrelation_length(
    space: SearchSpace,
    walk_length: int = 500,
    seed: int = 0,
) -> float:
    """Autocorrelation length from a random walk.

    Longer autocorrelation = smoother landscape = easier to search.
    """
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, space.size))
    fitnesses = _random_walk(space, start, walk_length, rng)
    return _acl_from_series(fitnesses)


def _random_walk(
    space: SearchSpace, start: int, length: int, rng: np.random.Generator,
) -> np.ndarray:
    """Perform a random walk, returning the fitness sequence."""
    f = np.empty(length, dtype=np.float64)
    current = start
    for i in range(length):
        f[i] = space.fitness(current)
        nbrs = space.neighbors(current)
        current = int(rng.choice(nbrs))
    return f


def _acl_from_series(f: np.ndarray) -> float:
    """Autocorrelation length: first lag where autocorrelation drops below 1/e."""
    n = len(f)
    if n < 10:
        return 1.0
    f_centered = f - f.mean()
    var = f_centered.var()
    if var < 1e-15:
        return float(n)
    threshold = 1.0 / np.e
    for lag in range(1, n // 2):
        acf = np.mean(f_centered[:n - lag] * f_centered[lag:]) / var
        if acf < threshold:
            return float(lag)
    return float(n // 2)


def information_content(
    space: SearchSpace,
    walk_length: int = 500,
    epsilon: float = 0.0,
    seed: int = 0,
) -> tuple[float, float]:
    """Information content H(eps) and partial information content M(eps).

    H measures the entropy of the slope-sign sequence along a random walk.
    M measures the number of slope changes (ruggedness indicator).
    """
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, space.size))
    fitnesses = _random_walk(space, start, walk_length, rng)

    n = len(fitnesses)
    if n < 3:
        return 0.0, 0.0

    symbols = []
    for i in range(n - 1):
        diff = fitnesses[i + 1] - fitnesses[i]
        if diff > epsilon:
            symbols.append(1)
        elif diff < -epsilon:
            symbols.append(-1)
        else:
            symbols.append(0)

    if len(symbols) < 2:
        return 0.0, 0.0

    pair_counts: dict[tuple[int, int], int] = {}
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    total = sum(pair_counts.values())

    h = 0.0
    for count in pair_counts.values():
        p = count / total
        if p > 0:
            h -= p * np.log2(p)

    n_changes = sum(1 for i in range(len(symbols) - 1)
                    if symbols[i] != symbols[i + 1])
    m = n_changes / max(len(symbols) - 1, 1)

    return float(h), float(m)


def neutrality_ratio(
    space: SearchSpace,
    walk_length: int = 500,
    seed: int = 0,
) -> float:
    """Fraction of neutral steps (equal fitness) in a random walk."""
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, space.size))
    fitnesses = _random_walk(space, start, walk_length, rng)
    if len(fitnesses) < 2:
        return 0.0
    neutral = sum(1 for i in range(len(fitnesses) - 1)
                  if abs(fitnesses[i + 1] - fitnesses[i]) < 1e-12)
    return neutral / (len(fitnesses) - 1)


def compute_fla_features(
    space: SearchSpace,
    optima: list[LocalOptimum],
    seed: int = 0,
) -> dict[str, float]:
    """Compute all classical FLA features for one instance."""
    fdc_val = fdc(space, optima)
    acl = autocorrelation_length(space, seed=seed)
    h_eps, m_eps = information_content(space, seed=seed)
    neut = neutrality_ratio(space, seed=seed)

    return {
        "fdc": fdc_val,
        "autocorrelation_length": acl,
        "information_content_h": h_eps,
        "partial_info_content_m": m_eps,
        "neutrality_ratio": neut,
    }
