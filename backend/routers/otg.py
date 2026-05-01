"""OTG construction endpoints -- both sync REST and streaming WebSocket."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend.models.schemas import (
    OTGRequest, OTGResponse, OTGEdgeInfo, FunnelInfo,
)
from backend import cache
from lib.otg import build_otg, OTGEdge

router = APIRouter(tags=["otg"])


@router.post("/api/otg/build", response_model=OTGResponse)
def build_otg_sync(req: OTGRequest):
    """Synchronous OTG construction. Returns the full graph at once."""
    cached = cache.get(req.instance_id)
    if cached is None:
        raise HTTPException(404, "Instance not found")

    result = build_otg(cached.space, cached.optima, gamma=req.gamma)

    return OTGResponse(
        instance_id=req.instance_id,
        edges=[
            OTGEdgeInfo(
                source=e.source, target=e.target,
                min_kappa=e.min_kappa, via_neighbor=e.via_neighbor,
            )
            for e in result.edges
        ],
        funnels=[
            FunnelInfo(
                attractor_idx=f.attractor_idx,
                member_indices=f.member_indices,
                attractor_fitness=f.attractor_fitness,
                is_cycle=f.is_cycle,
            )
            for f in result.funnels
        ],
        orc_values={
            str(k): {str(nk): nv for nk, nv in v.items()}
            for k, v in result.orc_values.items()
        },
        compression_ratio=result.compression_ratio,
        mean_terminal_rank=result.mean_terminal_rank,
        top5_reachability=result.top5_reachability,
        dag_depth=result.dag_depth,
        has_cycles=result.has_cycles,
    )


@router.websocket("/ws/otg/stream")
async def stream_otg(ws: WebSocket):
    """Stream OTG construction step by step.

    Client sends: {"instance_id": "...", "gamma": 1.0}
    Server sends events: computing_orc, edge_added, funnel_formed, complete
    """
    await ws.accept()
    try:
        msg = await ws.receive_json()
        instance_id = msg.get("instance_id")
        gamma = msg.get("gamma", 1.0)

        cached = cache.get(instance_id)
        if cached is None:
            await ws.send_json({"type": "error", "message": "Instance not found"})
            await ws.close()
            return

        n_optima = len(cached.optima)
        edges_so_far: list[dict] = []

        async def on_edge(edge: OTGEdge):
            edges_so_far.append({
                "source": edge.source,
                "target": edge.target,
                "min_kappa": edge.min_kappa,
                "via_neighbor": edge.via_neighbor,
            })
            await ws.send_json({
                "type": "edge_added",
                "source": edge.source,
                "target": edge.target,
                "min_kappa": round(edge.min_kappa, 4),
                "dest_fitness": round(cached.optima[edge.target].fitness, 4),
                "progress": f"{len(edges_so_far)}/{n_optima}",
            })

        # Build OTG with streaming callback
        # Since build_otg is sync, we wrap the callback
        result = build_otg(cached.space, cached.optima, gamma=gamma)

        # Stream edges one by one (replay from result)
        for i, edge in enumerate(result.edges):
            await ws.send_json({
                "type": "computing_orc",
                "optimum_idx": edge.source,
                "progress": f"{i + 1}/{n_optima}",
            })
            await ws.send_json({
                "type": "edge_added",
                "source": edge.source,
                "target": edge.target,
                "min_kappa": round(edge.min_kappa, 4),
                "via_neighbor": edge.via_neighbor,
                "dest_fitness": round(cached.optima[edge.target].fitness, 4),
                "progress": f"{i + 1}/{n_optima}",
            })

        # Send funnel events
        for funnel in result.funnels:
            await ws.send_json({
                "type": "funnel_formed",
                "attractor": funnel.attractor_idx,
                "members": funnel.member_indices,
                "attractor_fitness": funnel.attractor_fitness,
                "size": len(funnel.member_indices),
                "is_cycle": funnel.is_cycle,
            })

        # Send completion
        await ws.send_json({
            "type": "complete",
            "num_funnels": len(result.funnels),
            "compression": round(result.compression_ratio, 4),
            "has_cycles": result.has_cycles,
            "mean_terminal_rank": round(result.mean_terminal_rank, 4),
            "top5_reachability": round(result.top5_reachability, 4),
            "dag_depth": result.dag_depth,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
