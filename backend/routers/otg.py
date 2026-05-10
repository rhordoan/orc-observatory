"""OTG construction endpoints -- both sync REST and streaming WebSocket."""

from __future__ import annotations

import asyncio

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
    """Stream OTG construction incrementally.

    Client sends: {"instance_id": "...", "gamma": 1.0}
    Server sends events as ORC is computed and edges are resolved in real time.
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
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_computing(i: int):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "computing_orc", "optimum_idx": i, "progress": f"{i + 1}/{n_optima}"},
            )

        def on_edge(edge: OTGEdge):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "edge_added",
                    "source": edge.source,
                    "target": edge.target,
                    "min_kappa": round(edge.min_kappa, 4),
                    "via_neighbor": edge.via_neighbor,
                    "dest_fitness": round(cached.optima[edge.target].fitness, 4),
                    "progress": f"{edge.source + 1}/{n_optima}",
                },
            )

        _DONE = object()

        async def run_build():
            result = await asyncio.to_thread(
                build_otg, cached.space, cached.optima,
                gamma=gamma, on_edge=on_edge, on_computing=on_computing,
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))

        build_task = asyncio.create_task(run_build())

        result = None
        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "result":
                result = item[1]
                break
            await ws.send_json(item)

        await build_task

        for funnel in result.funnels:
            await ws.send_json({
                "type": "funnel_formed",
                "attractor": funnel.attractor_idx,
                "members": funnel.member_indices,
                "attractor_fitness": funnel.attractor_fitness,
                "size": len(funnel.member_indices),
                "is_cycle": funnel.is_cycle,
            })

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
