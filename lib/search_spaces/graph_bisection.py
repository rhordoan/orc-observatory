"""Graph bisection as a bit-flip search space.

Partition n vertices into two equal sets A (bit=0) and B (bit=1),
minimizing the cut (edges between A and B).
Fitness = -cut, neighborhood = flip one bit (move one vertex).

Supports planted-partition (strong community structure = clear funnels)
and Erdos-Renyi (random = weak structure) graph models.

Properties:
  - Dense fitness signal: every flip changes the cut
  - Degree = n (same as NK bit-flip)
  - Planted partition creates strong funnel structure ideal for ORC
"""

from __future__ import annotations

import numpy as np


class GraphBisectionSearchSpace:
    """Graph bisection on {0,1}^n with bit-flip neighborhood.

    Fitness = -(number of edges between partition halves).
    Maximizing fitness = minimizing cut.
    """

    def __init__(
        self,
        n: int = 20,
        edge_prob_within: float = 0.7,
        edge_prob_between: float = 0.1,
        model: str = "planted",
        seed: int | None = None,
        use_gpu: bool = False,
    ) -> None:
        self._n = n
        self._rng = np.random.default_rng(seed)

        if model == "planted":
            self._adj = self._planted_partition(
                n, edge_prob_within, edge_prob_between)
            self._model_desc = (f"planted p_in={edge_prob_within} "
                                f"p_out={edge_prob_between}")
        elif model == "er":
            p = edge_prob_within
            self._adj = self._erdos_renyi(n, p)
            self._model_desc = f"ER p={p}"
        else:
            raise ValueError(f"Unknown model: {model}")

        self._n_edges = int(self._adj.sum()) // 2
        self._size = 2 ** n
        self._fitnesses = self._compute_all_fitnesses_vec()

    def _planted_partition(self, n: int, p_in: float, p_out: float) -> np.ndarray:
        """Two communities: vertices [0..n/2-1] and [n/2..n-1]."""
        adj = np.zeros((n, n), dtype=np.float64)
        half = n // 2
        for i in range(n):
            for j in range(i + 1, n):
                same_community = (i < half) == (j < half)
                p = p_in if same_community else p_out
                if self._rng.random() < p:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    def _erdos_renyi(self, n: int, p: float) -> np.ndarray:
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if self._rng.random() < p:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    def _compute_all_fitnesses_vec(self) -> np.ndarray:
        """Vectorized fitness computation for all 2^n solutions."""
        n = self._n
        size = self._size
        bits = np.array([[(idx >> b) & 1 for b in range(n)]
                         for idx in range(size)], dtype=np.float64)
        # cut(x) = Σ_{(i,j)∈E} |x_i - x_j| = x^T * L * x (Laplacian-like)
        # More directly: for each edge (i,j), it contributes to cut if bits differ
        cuts = np.zeros(size, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if self._adj[i, j] > 0:
                    cuts += np.abs(bits[:, i] - bits[:, j])
        return -cuts

    @property
    def name(self) -> str:
        return f"Graph Bisection (n={self._n}, {self._model_desc}, {self._n_edges} edges)"

    @property
    def size(self) -> int:
        return self._size

    @property
    def degree(self) -> int:
        return self._n

    def fitness(self, idx: int) -> float:
        return float(self._fitnesses[idx])

    def neighbors(self, idx: int) -> np.ndarray:
        return np.array([idx ^ (1 << b) for b in range(self._n)], dtype=np.intp)

    def solution_label(self, idx: int) -> str:
        return format(idx, f"0{self._n}b")

    @property
    def fitnesses(self) -> np.ndarray:
        return self._fitnesses
