"""In-memory instance cache (Repository pattern).

Stores generated search space instances and their local optima so that
subsequent OTG/LON/explain requests do not need to regenerate them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from lib.search_spaces.protocol import SearchSpace
from lib.hill_climb import LocalOptimum


@dataclass
class CachedInstance:
    instance_id: str
    space: SearchSpace
    optima: list[LocalOptimum]
    problem_type: str


_store: dict[str, CachedInstance] = {}


def put(space: SearchSpace, optima: list[LocalOptimum], problem_type: str) -> str:
    """Cache an instance, return its ID."""
    iid = uuid.uuid4().hex[:12]
    _store[iid] = CachedInstance(
        instance_id=iid,
        space=space,
        optima=optima,
        problem_type=problem_type,
    )
    return iid


def get(instance_id: str) -> CachedInstance | None:
    return _store.get(instance_id)


def clear() -> None:
    _store.clear()
