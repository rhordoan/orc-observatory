"""Random MAX-SAT search space with bit-flip neighborhood."""

from __future__ import annotations

import numpy as np


class MaxSATSearchSpace:
    """Random k-SAT instance used as a MAX-SAT optimization problem.

    Each clause is a disjunction of *clause_length* literals drawn uniformly.
    Fitness is the fraction of satisfied clauses (maximization, but we store
    it so higher = better; the ORC framework handles direction).
    """

    def __init__(
        self,
        n_vars: int,
        n_clauses: int | None = None,
        clause_length: int = 3,
        seed: int | None = None,
    ) -> None:
        self._n = n_vars
        self._clause_length = clause_length
        self._rng = np.random.default_rng(seed)

        if n_clauses is None:
            # Default ratio alpha ~= 4.27 for 3-SAT phase transition
            n_clauses = int(4.27 * n_vars)
        self._n_clauses = n_clauses

        # Generate random clauses: each clause is an array of signed literals
        # Positive = variable appears positive, negative = negated
        self._clauses = []
        for _ in range(n_clauses):
            variables = self._rng.choice(n_vars, size=clause_length, replace=False)
            signs = self._rng.choice([-1, 1], size=clause_length)
            self._clauses.append(variables * signs + signs)
            # Store as (var_index, is_positive) pairs for clarity
        # Re-generate in a cleaner format
        self._clause_vars = np.zeros((n_clauses, clause_length), dtype=np.intp)
        self._clause_signs = np.zeros((n_clauses, clause_length), dtype=np.intp)
        for i in range(n_clauses):
            self._clause_vars[i] = self._rng.choice(n_vars, size=clause_length, replace=False)
            self._clause_signs[i] = self._rng.choice([0, 1], size=clause_length)

        self._size = 2**n_vars

        # Pre-compute fitness for all assignments
        self._fitnesses = np.array(
            [self._compute_fitness(idx) for idx in range(self._size)]
        )

    @property
    def name(self) -> str:
        return f"MAX-SAT (n={self._n}, m={self._n_clauses}, k={self._clause_length})"

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
        satisfied = 0
        for c in range(self._n_clauses):
            clause_sat = False
            for j in range(self._clause_length):
                var = self._clause_vars[c, j]
                want_positive = self._clause_signs[c, j]
                if bits[var] == want_positive:
                    clause_sat = True
                    break
            if clause_sat:
                satisfied += 1
        return satisfied / self._n_clauses

    @property
    def fitnesses(self) -> np.ndarray:
        return self._fitnesses
