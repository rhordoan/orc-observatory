from .protocol import SearchSpace
from .nk_landscape import NKSearchSpace
from .wmodel import WModelSearchSpace
from .maxsat import MaxSATSearchSpace
from .tsp import TSPSearchSpace
from .qap import QAPSearchSpace

__all__ = [
    "SearchSpace",
    "NKSearchSpace",
    "WModelSearchSpace",
    "MaxSATSearchSpace",
    "TSPSearchSpace",
    "QAPSearchSpace",
]
