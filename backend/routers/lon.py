"""LON-d1 construction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.schemas import LONRequest, LONResponse, LONEdgeInfo
from backend import cache
from lib.lon import build_lon_d1

router = APIRouter(tags=["lon"])


@router.post("/api/lon/build", response_model=LONResponse)
def build_lon(req: LONRequest):
    """Build LON-d1 for comparison with the OTG."""
    cached = cache.get(req.instance_id)
    if cached is None:
        raise HTTPException(404, "Instance not found")

    result = build_lon_d1(cached.space, cached.optima)

    return LONResponse(
        instance_id=req.instance_id,
        edges=[
            LONEdgeInfo(
                source=e.source, target=e.target,
                via_neighbor=e.via_neighbor,
                neighbor_fitness=e.neighbor_fitness,
            )
            for e in result.edges
        ],
        n_self_loops=result.n_self_loops,
        singleton_fraction=result.singleton_fraction,
    )
