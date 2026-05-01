"""Instance generation and retrieval endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    InstanceRequest, InstanceResponse, OptimumInfo, ProblemType,
)
from backend import cache
from lib.search_spaces import NKSearchSpace, WModelSearchSpace, MaxSATSearchSpace
from lib.hill_climb import enumerate_local_optima

router = APIRouter(prefix="/api/instances", tags=["instances"])


@router.post("", response_model=InstanceResponse)
def create_instance(req: InstanceRequest):
    """Generate a landscape instance and cache it."""
    if req.problem_type == ProblemType.NK:
        if req.k >= req.n:
            raise HTTPException(422, f"K must be < N, got K={req.k}, N={req.n}")
        space = NKSearchSpace(n=req.n, k=req.k, seed=req.seed)
    elif req.problem_type == ProblemType.WMODEL:
        space = WModelSearchSpace(
            n=req.n, mu=req.mu, nu=req.nu, gamma=req.gamma_wmodel, seed=req.seed,
        )
    elif req.problem_type == ProblemType.MAXSAT:
        space = MaxSATSearchSpace(
            n_vars=req.n, n_clauses=req.n_clauses,
            clause_length=req.clause_length, seed=req.seed,
        )
    else:
        raise HTTPException(422, f"Unknown problem type: {req.problem_type}")

    optima = enumerate_local_optima(space)
    iid = cache.put(space, optima, req.problem_type.value)

    return InstanceResponse(
        instance_id=iid,
        problem_type=req.problem_type.value,
        name=space.name,
        space_size=space.size,
        degree=space.degree,
        n_optima=len(optima),
        optima=[
            OptimumInfo(
                list_idx=i,
                solution_idx=o.idx,
                label=space.solution_label(o.idx),
                fitness=o.fitness,
                basin_size=o.basin_size,
            )
            for i, o in enumerate(optima)
        ],
    )


@router.get("/{instance_id}", response_model=InstanceResponse)
def get_instance(instance_id: str):
    """Retrieve a cached instance."""
    cached = cache.get(instance_id)
    if cached is None:
        raise HTTPException(404, "Instance not found")

    space = cached.space
    optima = cached.optima

    return InstanceResponse(
        instance_id=instance_id,
        problem_type=cached.problem_type,
        name=space.name,
        space_size=space.size,
        degree=space.degree,
        n_optima=len(optima),
        optima=[
            OptimumInfo(
                list_idx=i,
                solution_idx=o.idx,
                label=space.solution_label(o.idx),
                fitness=o.fitness,
                basin_size=o.basin_size,
            )
            for i, o in enumerate(optima)
        ],
    )
