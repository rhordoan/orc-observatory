"""W-model search space with neutrality, epistasis, and ruggedness layers."""

from __future__ import annotations

import numpy as np

from .nk_landscape import NKSearchSpace


class WModelSearchSpace:
    """W-model wrapper that applies transformations on top of a base landscape.

    Layers applied in order:
      1. Neutrality (mu): groups of mu bits are majority-voted into one
      2. Epistasis (nu): blocks of nu bits are XOR-permuted
      3. Ruggedness (gamma): fitness values are permuted

    The underlying base is an NK landscape with K=0 (no epistasis in the
    base), so all complexity comes from the W-model layers.
    """

    def __init__(
        self,
        n: int,
        mu: int = 1,
        nu: int = 1,
        gamma: int = 0,
        seed: int | None = None,
    ) -> None:
        if mu < 1 or nu < 1:
            raise ValueError("mu and nu must be >= 1")

        self._n = n
        self._mu = mu
        self._nu = nu
        self._gamma = gamma
        self._rng = np.random.default_rng(seed)

        # Effective dimension after neutrality reduction
        self._n_eff = n // mu

        # Build a simple base landscape (OneMax on the effective bits)
        # then apply ruggedness permutation to fitness values
        self._size = 2**n

        # Ruggedness permutation: maps integer fitness ranks to shuffled ranks
        max_ones = self._n_eff
        perm = np.arange(max_ones + 1, dtype=np.float64)
        if gamma > 0:
            for i in range(min(gamma, max_ones)):
                j = max_ones - i
                if j > i:
                    perm[i], perm[j] = perm[j], perm[i]
        self._ruggedness_perm = perm

        # Pre-compute all fitness values
        self._fitnesses = np.array(
            [self._compute_fitness(idx) for idx in range(self._size)]
        )

    @property
    def name(self) -> str:
        return f"W-Model (N={self._n}, mu={self._mu}, nu={self._nu}, gamma={self._gamma})"

    @property
    def size(self) -> int:
        return self._size

    @property
    def degree(self) -> int:
        return self._n

    def fitness(self, idx: int) -> float:
        return float(self._fitnesses[idx])

    def neighbors(self, idx: int) -> np.ndarray:
        return np.array([idx ^ (1 << bit) for bit in range(self._n)], dtype=np.intp)

    def solution_label(self, idx: int) -> str:
        return format(idx, f"0{self._n}b")

    def _compute_fitness(self, idx: int) -> float:
        bits = np.array(
            [(idx >> bit) & 1 for bit in range(self._n)], dtype=np.intp
        )

        # Step 1: Neutrality -- majority-vote groups of mu bits
        reduced = np.zeros(self._n_eff, dtype=np.intp)
        for i in range(self._n_eff):
            group = bits[i * self._mu : (i + 1) * self._mu]
            reduced[i] = 1 if np.sum(group) > self._mu / 2 else 0

        # Step 2: Epistasis -- XOR-permute blocks of nu bits
        if self._nu > 1:
            n_blocks = self._n_eff // self._nu
            for b in range(n_blocks):
                block = reduced[b * self._nu : (b + 1) * self._nu]
                # Simple epistasis: XOR each bit with its predecessor
                for j in range(len(block) - 1, 0, -1):
                    block[j] ^= block[j - 1]

        # Step 3: Compute base fitness (OneMax on reduced bits)
        ones = int(np.sum(reduced))

        # Step 4: Ruggedness permutation
        fitness = self._ruggedness_perm[ones] / self._n_eff

        return float(fitness)

    @property
    def fitnesses(self) -> np.ndarray:
        return self._fitnesses
