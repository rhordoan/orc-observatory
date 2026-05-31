import numpy as np
import pytest

from lib.search_spaces.protocol import SearchSpace
from experiments.ablations import ShuffledFitnessSpace, make_shuffle_perm

class DummySpace(SearchSpace):
    def __init__(self, size=10, degree=3):
        self._size = size
        self._degree = degree
        # Simple deterministic fitness
        self._fitness_vals = np.array([float(i * 1.5) for i in range(size)])
        # Simple cyclic neighbors
        self._neighbors = {
            i: np.array([(i - 1) % size, (i + 1) % size, (i + 2) % size], dtype=np.intp)
            for i in range(size)
        }

    @property
    def size(self) -> int:
        return self._size

    @property
    def degree(self) -> int:
        return self._degree

    def fitness(self, idx: int) -> float:
        return self._fitness_vals[idx]

    def neighbors(self, idx: int) -> np.ndarray:
        return self._neighbors[idx]

    def solution_label(self, idx: int) -> str:
        return str(idx)


def test_make_shuffle_perm():
    space = DummySpace(10)
    perm = make_shuffle_perm(space, seed=42)
    assert len(perm) == 10
    assert set(perm) == set(range(10))
    
    # Check deterministic seed
    perm2 = make_shuffle_perm(space, seed=42)
    np.testing.assert_array_equal(perm, perm2)


def test_shuffled_fitness_space_properties():
    space = DummySpace(10, 3)
    perm = np.arange(10)[::-1]  # reverse permutation
    shuf = ShuffledFitnessSpace(space, perm)

    assert shuf.size == space.size
    assert shuf.degree == space.degree


def test_shuffled_fitness_space_fitness():
    space = DummySpace(10)
    perm = np.arange(10)[::-1]  # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    shuf = ShuffledFitnessSpace(space, perm)

    # fitness of idx 0 should be base.fitness(perm[0]) -> base.fitness(9)
    assert shuf.fitness(0) == space.fitness(9)
    assert shuf.fitness(1) == space.fitness(8)
    assert shuf.fitness(9) == space.fitness(0)

    # Check out of bounds handles gracefully if perm is shorter (based on implementation logic)
    # The implementation says: if idx < len(perm): return base.fitness(perm[idx])
    # else: return base.fitness(idx)
    # Testing out-of-bounds requires a DummySpace that supports it, skipping here.


def test_shuffled_fitness_space_neighbors():
    space = DummySpace(10)
    perm = np.arange(10)[::-1]
    shuf = ShuffledFitnessSpace(space, perm)

    np.testing.assert_array_equal(shuf.neighbors(0), space.neighbors(0))
    np.testing.assert_array_equal(shuf.neighbors(5), space.neighbors(5))


def test_shuffled_fitness_space_neighbor_fitnesses():
    space = DummySpace(10)
    perm = np.arange(10)[::-1]
    shuf = ShuffledFitnessSpace(space, perm)

    # idx = 1
    # space.neighbors(1) = [0, 2, 3]
    # base fitness for these: 0.0, 3.0, 4.5
    # For ShuffledFitnessSpace:
    # shuf.fitness(0) -> base.fitness(perm[0]) -> base.fitness(9) = 13.5
    # shuf.fitness(2) -> base.fitness(perm[2]) -> base.fitness(7) = 10.5
    # shuf.fitness(3) -> base.fitness(perm[3]) -> base.fitness(6) = 9.0
    
    expected_fitnesses = np.array([shuf.fitness(n) for n in space.neighbors(1)])
    actual_fitnesses = shuf.neighbor_fitnesses(1)
    
    np.testing.assert_array_equal(actual_fitnesses, expected_fitnesses)

def test_shuffled_fitness_space_solution_label():
    space = DummySpace(10)
    perm = np.arange(10)[::-1]
    shuf = ShuffledFitnessSpace(space, perm)

    assert shuf.solution_label(5) == space.solution_label(5)

