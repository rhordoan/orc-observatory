"""ORC explainer endpoint -- returns the transport plan for one edge."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import ORCExplainResponse
from backend import cache
from lib.orc import compute_orc_explained

router = APIRouter(tags=["orc"])


@router.get("/api/orc/explain", response_model=ORCExplainResponse)
def explain_orc(
    instance_id: str = Query(...),
    from_optimum: int = Query(..., description="Solution-space index of the optimum"),
    to_neighbor: int = Query(..., description="Solution-space index of the neighbor"),
    gamma: float = Query(default=1.0, gt=0),
):
    """Return detailed ORC breakdown for one edge.

    Powers the 'why this direction?' modal in the frontend by returning
    the transport plan, exclusive neighbor matching, and per-pair costs.
    """
    cached = cache.get(instance_id)
    if cached is None:
        raise HTTPException(404, "Instance not found")

    space = cached.space
    result = compute_orc_explained(space, from_optimum, to_neighbor, gamma)

    return ORCExplainResponse(
        x_idx=result.x_idx,
        y_idx=result.y_idx,
        kappa=result.kappa,
        w1=result.w1,
        shared=result.shared,
        x_exclusive=result.x_exclusive,
        y_exclusive=result.y_exclusive,
        matching=[list(m) for m in result.matching],
        pair_costs=result.pair_costs,
        x_exclusive_fitness=[space.fitness(s) for s in result.x_exclusive],
        y_exclusive_fitness=[space.fitness(s) for s in result.y_exclusive],
        shared_labels=[space.solution_label(s) for s in result.shared],
        x_exclusive_labels=[space.solution_label(s) for s in result.x_exclusive],
        y_exclusive_labels=[space.solution_label(s) for s in result.y_exclusive],
    )
