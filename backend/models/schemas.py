"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum


class ProblemType(str, Enum):
    NK = "nk"
    WMODEL = "wmodel"
    MAXSAT = "maxsat"


# -- Requests ----------------------------------------------------------------

class InstanceRequest(BaseModel):
    problem_type: ProblemType
    n: int = Field(ge=4, le=18, description="Problem size (bits or variables)")
    k: int = Field(default=2, ge=1, description="NK epistasis or W-model param")
    mu: int = Field(default=1, ge=1, description="W-model neutrality")
    nu: int = Field(default=1, ge=1, description="W-model epistasis")
    gamma_wmodel: int = Field(default=0, ge=0, description="W-model ruggedness")
    n_clauses: int | None = Field(default=None, description="MAX-SAT clause count")
    clause_length: int = Field(default=3, ge=2, description="MAX-SAT clause length")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class OTGRequest(BaseModel):
    instance_id: str
    gamma: float = Field(default=1.0, gt=0, description="ORC sensitivity parameter")


class LONRequest(BaseModel):
    instance_id: str


class ORCExplainRequest(BaseModel):
    instance_id: str
    from_optimum: int
    to_neighbor: int
    gamma: float = Field(default=1.0, gt=0)


# -- Responses ---------------------------------------------------------------

class OptimumInfo(BaseModel):
    list_idx: int
    solution_idx: int
    label: str
    fitness: float
    basin_size: int


class OTGEdgeInfo(BaseModel):
    source: int
    target: int
    min_kappa: float
    via_neighbor: int


class FunnelInfo(BaseModel):
    attractor_idx: int
    member_indices: list[int]
    attractor_fitness: float
    is_cycle: bool


class InstanceResponse(BaseModel):
    instance_id: str
    problem_type: str
    name: str
    space_size: int
    degree: int
    n_optima: int
    optima: list[OptimumInfo]


class OTGResponse(BaseModel):
    instance_id: str
    edges: list[OTGEdgeInfo]
    funnels: list[FunnelInfo]
    orc_values: dict[int, dict[int, float]]
    compression_ratio: float
    mean_terminal_rank: float
    top5_reachability: float
    dag_depth: int
    has_cycles: bool


class LONEdgeInfo(BaseModel):
    source: int
    target: int
    via_neighbor: int
    neighbor_fitness: float


class LONResponse(BaseModel):
    instance_id: str
    edges: list[LONEdgeInfo]
    n_self_loops: int
    singleton_fraction: float


class ORCExplainResponse(BaseModel):
    x_idx: int
    y_idx: int
    kappa: float
    w1: float
    shared: list[int]
    x_exclusive: list[int]
    y_exclusive: list[int]
    matching: list[list[int]]
    pair_costs: list[float]
    x_exclusive_fitness: list[float]
    y_exclusive_fitness: list[float]
    shared_labels: list[str]
    x_exclusive_labels: list[str]
    y_exclusive_labels: list[str]
