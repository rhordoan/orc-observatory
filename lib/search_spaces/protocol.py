"""SearchSpace protocol -- Strategy pattern interface for problem types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SearchSpace(Protocol):
    """Every problem type implements this interface.

    Adding a new search space requires only implementing these methods;
    the API routers, services, and frontend need no changes (NFR6).
    """

    @property
    def name(self) -> str:
        """Human-readable name, e.g. 'NK Landscape (N=12, K=4)'."""
        ...

    @property
    def size(self) -> int:
        """Total number of candidate solutions |S|."""
        ...

    @property
    def degree(self) -> int:
        """Neighborhood degree k (number of neighbors per solution)."""
        ...

    def fitness(self, idx: int) -> float:
        """Return f(x) for the solution at index *idx*."""
        ...

    def neighbors(self, idx: int) -> np.ndarray:
        """Return array of neighbor indices for solution *idx*."""
        ...

    def solution_label(self, idx: int) -> str:
        """Human-readable label for solution *idx*, e.g. '0110'."""
        ...
