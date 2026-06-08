from pathlib import Path

import numpy as np
import pytest

from lib.search_spaces.qaplib import QAPLIBSearchSpace
from lib.search_spaces.tsplib import TSPLIBSearchSpace


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _assert_permutation(values, size):
    assert len(values) == size
    assert sorted(int(v) for v in values) == list(range(size))


@pytest.fixture(scope="module")
def tsplib_space():
    fixture = DATA_DIR / "tsplib" / "eil51.tsp"
    assert fixture.exists(), "expected bundled TSPLIB fixture"
    return TSPLIBSearchSpace(
        "eil51",
        data_dir=str(fixture.parent),
        n_restarts=1,
        seed=123,
    )


@pytest.fixture(scope="module")
def qaplib_space():
    fixture = DATA_DIR / "qaplib" / "nug12.dat"
    assert fixture.exists(), "expected bundled QAPLIB fixture"
    return QAPLIBSearchSpace(
        "nug12",
        data_dir=str(fixture.parent),
        n_restarts=1,
        seed=123,
    )


class TestTSPLIBSearchSpace:
    def test_neighbors_are_valid_two_opt_tours(self, tsplib_space):
        space = tsplib_space
        base_tour = space._tours[0]

        _assert_permutation(base_tour, space._n)
        assert base_tour[0] == 0

        neighbors = space.neighbors(0)

        assert len(neighbors) == space.degree
        assert len(set(neighbors.tolist())) == space.degree
        for neighbor_idx in neighbors:
            tour = space._tours[int(neighbor_idx)]
            _assert_permutation(tour, space._n)
            assert tour[0] == 0

    def test_neighbor_fitnesses_match_direct_two_opt_evaluation(self, tsplib_space):
        space = tsplib_space

        neighbors = space.neighbors(0)
        vectorized = space.neighbor_fitnesses(0)
        direct = np.array([space.fitness(int(neighbor_idx)) for neighbor_idx in neighbors])

        np.testing.assert_allclose(vectorized, direct, rtol=0, atol=1e-9)
        assert np.all(vectorized <= space.fitness(0) + 1e-9)


class TestQAPLIBSearchSpace:
    def test_neighbors_are_valid_swap_permutations(self, qaplib_space):
        space = qaplib_space
        base_perm = space._perms[0]

        _assert_permutation(base_perm, space._n)

        neighbors = space.neighbors(0)

        assert len(neighbors) == space.degree
        assert len(set(neighbors.tolist())) == space.degree
        for neighbor_idx in neighbors:
            _assert_permutation(space._perms[int(neighbor_idx)], space._n)

    def test_neighbor_fitnesses_match_direct_and_scalar_delta_evaluation(self, qaplib_space):
        space = qaplib_space

        neighbors = space.neighbors(0)
        vectorized = space.neighbor_fitnesses(0)
        direct = np.array([space.fitness(int(neighbor_idx)) for neighbor_idx in neighbors])

        base_cost = -space.fitness(0)
        base_perm = list(space._perms[0])
        scalar_deltas = np.array(
            [space._swap_delta(base_perm, r, s) for r, s in space._swap_pairs]
        )

        np.testing.assert_allclose(vectorized, direct, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            vectorized,
            -(base_cost + scalar_deltas),
            rtol=0,
            atol=1e-9,
        )
        assert np.all(vectorized <= space.fitness(0) + 1e-9)
