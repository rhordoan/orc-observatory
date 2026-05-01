"""Tests for ORC computation.

Validates the fitness-lifted ORC against the hand-computed example from
the thesis (Example 3.1): a 4-bit binary landscape with x*=0000 and y=0001.
"""

import numpy as np
import pytest

from lib.search_spaces.nk_landscape import NKSearchSpace
from lib.orc import compute_orc, compute_orc_explained, compute_all_orc, min_orc_neighbor


class TinyBinarySpace:
    """Minimal 4-bit space with hand-set fitness values for testing.

    Solutions are 4-bit strings. Neighborhood is single-bit flip (degree 4).
    Fitness values are set so x*=0000 is a local optimum with known ORC.
    """

    def __init__(self):
        self._n = 4
        self._size = 16
        self._fitnesses = np.zeros(16)

        # x* = 0000 (idx 0) is a local optimum with fitness 10.0
        self._fitnesses[0] = 10.0
        # y = 0001 (idx 1) has fitness 8.0
        self._fitnesses[1] = 8.0
        # Neighbors of x* (one-bit flips of 0000): 0001, 0010, 0100, 1000
        self._fitnesses[0b0010] = 7.0
        self._fitnesses[0b0100] = 6.0
        self._fitnesses[0b1000] = 5.0
        # Neighbors of y=0001 (one-bit flips of 0001): 0000, 0011, 0101, 1001
        self._fitnesses[0b0011] = 4.0
        self._fitnesses[0b0101] = 3.0
        self._fitnesses[0b1001] = 2.0

        # Fill remaining with low fitness
        for i in range(16):
            if self._fitnesses[i] == 0.0 and i not in (0, 1, 2, 4, 8, 3, 5, 9):
                self._fitnesses[i] = 1.0

    @property
    def name(self):
        return "Test 4-bit space"

    @property
    def size(self):
        return 16

    @property
    def degree(self):
        return 4

    def fitness(self, idx):
        return float(self._fitnesses[idx])

    def neighbors(self, idx):
        return np.array([idx ^ (1 << b) for b in range(4)], dtype=np.intp)

    def solution_label(self, idx):
        return format(idx, "04b")


@pytest.fixture
def tiny_space():
    return TinyBinarySpace()


class TestDisjointNeighborhoodProperty:
    """Verify the disjoint-neighborhood property holds for bit-flip graphs."""

    def test_shared_elements(self, tiny_space):
        """x* and y share exactly {x*, y} in their supports."""
        s = tiny_space
        support_x = set(s.neighbors(0).tolist()) | {0}
        support_y = set(s.neighbors(1).tolist()) | {1}
        shared = support_x & support_y
        assert shared == {0, 1}

    def test_exclusive_count(self, tiny_space):
        """Each side has exactly k-1 = 3 exclusive neighbors."""
        s = tiny_space
        support_x = set(s.neighbors(0).tolist()) | {0}
        support_y = set(s.neighbors(1).tolist()) | {1}
        x_excl = support_x - support_y
        y_excl = support_y - support_x
        assert len(x_excl) == 3
        assert len(y_excl) == 3


class TestORCComputation:
    """Verify ORC values match the thesis hand computation."""

    def test_orc_is_negative_at_local_optimum(self, tiny_space):
        """Theorem 3.1: ORC is guaranteed negative at local optima with k>=4."""
        kappa = compute_orc(tiny_space, 0, 1, gamma=1.0)
        assert kappa < 0, f"Expected negative ORC, got {kappa}"

    def test_orc_upper_bound(self, tiny_space):
        """ORC should satisfy kappa <= (3-k)/(k+1) = -1/5 for k=4."""
        kappa = compute_orc(tiny_space, 0, 1, gamma=1.0)
        upper_bound = (3 - 4) / (4 + 1)  # -0.2
        assert kappa <= upper_bound + 1e-10, (
            f"ORC {kappa} exceeds theoretical bound {upper_bound}"
        )

    def test_self_matches_at_zero_cost(self, tiny_space):
        """Shared elements x* and y self-match with cost 0."""
        result = compute_orc_explained(tiny_space, 0, 1, gamma=1.0)
        assert result.shared_cost == 0.0
        assert 0 in result.shared
        assert 1 in result.shared

    def test_exclusive_distance_is_two(self, tiny_space):
        """All exclusive neighbor pairs should have structural distance 2."""
        result = compute_orc_explained(tiny_space, 0, 1, gamma=1.0)
        # Each pair cost >= 2.0 (structural component)
        for cost in result.pair_costs:
            assert cost >= 2.0, f"Pair cost {cost} < 2.0"


class TestExplainerData:
    """Verify the explainer data is complete and consistent."""

    def test_partition_completeness(self, tiny_space):
        """Shared + exclusive partitions cover the full support."""
        result = compute_orc_explained(tiny_space, 0, 1, gamma=1.0)
        all_in_x = set(result.shared) | set(result.x_exclusive)
        all_in_y = set(result.shared) | set(result.y_exclusive)
        support_x = set(tiny_space.neighbors(0).tolist()) | {0}
        support_y = set(tiny_space.neighbors(1).tolist()) | {1}
        assert all_in_x == support_x
        assert all_in_y == support_y

    def test_matching_is_valid_permutation(self, tiny_space):
        """The matching should be a valid bijection between exclusive sets."""
        result = compute_orc_explained(tiny_space, 0, 1, gamma=1.0)
        rows = [m[0] for m in result.matching]
        cols = [m[1] for m in result.matching]
        assert sorted(rows) == list(range(len(result.x_exclusive)))
        assert sorted(cols) == list(range(len(result.y_exclusive)))

    def test_w1_and_kappa_consistency(self, tiny_space):
        """kappa = 1 - W1 for adjacent nodes (d(x,y) = 1)."""
        result = compute_orc_explained(tiny_space, 0, 1, gamma=1.0)
        assert abs(result.kappa - (1.0 - result.w1)) < 1e-10


class TestNKLandscapeORC:
    """Test ORC on an actual NK landscape (not hand-crafted)."""

    def test_all_orc_negative_at_local_optimum(self):
        """Every edge incident to a local optimum should have negative ORC."""
        space = NKSearchSpace(n=8, k=2, seed=42)
        from lib.hill_climb import enumerate_local_optima
        optima = enumerate_local_optima(space)

        # Check the first local optimum
        opt = optima[0]
        orc_vals = compute_all_orc(space, opt.idx, gamma=1.0)
        for nbr, kappa in orc_vals.items():
            assert kappa < 0, (
                f"ORC to neighbor {nbr} is {kappa}, expected negative"
            )

    def test_min_orc_neighbor_is_most_negative(self):
        space = NKSearchSpace(n=8, k=2, seed=42)
        from lib.hill_climb import enumerate_local_optima
        optima = enumerate_local_optima(space)

        opt = optima[0]
        best_nbr, best_kappa = min_orc_neighbor(space, opt.idx)
        all_orc = compute_all_orc(space, opt.idx)
        assert best_kappa == min(all_orc.values())
