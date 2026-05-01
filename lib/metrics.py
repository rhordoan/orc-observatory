"""Classical fitness landscape analysis metrics.

FDC, autocorrelation length, and information content -- the scalar metrics
that the ORC-based approach improves upon.
"""

from __future__ import annotations

import numpy as np

from .search_spaces.protocol import SearchSpace
from .hill_climb import LocalOptimum


def fitness_distance_correlation(
    space: SearchSpace,
    optima: list[LocalOptimum],
) -> float:
    """FDC: Pearson correlation between fitness and distance to global optimum.

    High positive FDC indicates a "big valley" structure favorable to
    local search. Values near zero suggest a deceptive landscape.
    """
    if len(optima) < 2:
        return 0.0

    global_opt = optima[0]  # optima are sorted by fitness, descending

    fitnesses = np.array([o.fitness for o in optima])
    distances = np.array([
        _hamming_distance(o.idx, global_opt.idx, _bit_length(space))
        for o in optima
    ])

    if np.std(fitnesses) == 0 or np.std(distances) == 0:
        return 0.0

    return float(np.corrcoef(fitnesses, distances)[0, 1])


def autocorrelation_length(
    space: SearchSpace,
    walk_length: int = 1000,
    seed: int | None = None,
) -> float:
    """Autocorrelation length from a random walk on the search graph.

    Measures how quickly fitness values decorrelate. Longer correlation
    lengths indicate smoother, easier landscapes.
    """
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, space.size))

    # Perform random walk
    fitnesses_walk = np.zeros(walk_length)
    current = start
    for t in range(walk_length):
        fitnesses_walk[t] = space.fitness(current)
        nbrs = space.neighbors(current)
        current = int(rng.choice(nbrs))

    # Compute autocorrelation
    mean = np.mean(fitnesses_walk)
    var = np.var(fitnesses_walk)
    if var == 0:
        return float(walk_length)

    centered = fitnesses_walk - mean

    # Find the lag where autocorrelation drops below 1/e
    for tau in range(1, walk_length // 2):
        r = np.mean(centered[:walk_length - tau] * centered[tau:]) / var
        if r < 1 / np.e:
            return float(tau)

    return float(walk_length // 2)


def information_content(
    space: SearchSpace,
    walk_length: int = 1000,
    epsilon: float = 0.0,
    seed: int | None = None,
) -> float:
    """Information content H(epsilon) of fitness-change signs along a random walk.

    Quantifies landscape ruggedness via the entropy of the sign sequence.
    High H indicates complex, rugged landscapes.
    """
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, space.size))

    # Random walk, record fitness changes
    current = start
    signs = []
    prev_fitness = space.fitness(current)

    for _ in range(walk_length):
        nbrs = space.neighbors(current)
        current = int(rng.choice(nbrs))
        f = space.fitness(current)
        diff = f - prev_fitness
        if diff > epsilon:
            signs.append("+")
        elif diff < -epsilon:
            signs.append("-")
        else:
            signs.append("0")
        prev_fitness = f

    # Compute pairwise transition entropy
    symbols = ["+", "-", "0"]
    counts = {}
    total = 0
    for a in symbols:
        for b in symbols:
            counts[(a, b)] = 0

    for i in range(len(signs) - 1):
        pair = (signs[i], signs[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
        total += 1

    if total == 0:
        return 0.0

    entropy = 0.0
    for pair, count in counts.items():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return float(entropy)


def _hamming_distance(a: int, b: int, n_bits: int) -> int:
    """Hamming distance between two binary strings."""
    xor = a ^ b
    return bin(xor).count("1")


def _bit_length(space: SearchSpace) -> int:
    """Infer bit length from space size."""
    return int(np.log2(space.size))
