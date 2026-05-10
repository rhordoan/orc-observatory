"""Difficulty prediction metrics endpoint (FR6.1-FR6.2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend import cache
from backend.models.schemas import MetricsResponse
from lib.metrics import (
    fitness_distance_correlation,
    autocorrelation_length,
    information_content,
    mean_orc,
)

router = APIRouter(tags=["metrics"])


@router.get("/api/metrics/{instance_id}", response_model=MetricsResponse)
def get_metrics(instance_id: str):
    """Compute classical FLA metrics and mean ORC for a cached instance."""
    cached = cache.get(instance_id)
    if cached is None:
        raise HTTPException(404, "Instance not found")

    return MetricsResponse(
        instance_id=instance_id,
        fdc=round(fitness_distance_correlation(cached.space, cached.optima), 4),
        autocorrelation_length=round(
            autocorrelation_length(cached.space, seed=0), 4
        ),
        information_content=round(
            information_content(cached.space, seed=0), 4
        ),
        mean_orc=round(mean_orc(cached.space, cached.optima), 4),
    )
