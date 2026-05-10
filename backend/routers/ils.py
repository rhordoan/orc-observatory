"""ILS race endpoint -- streams iteration events for all three variants concurrently."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend import cache
from lib.ils import orc_ils, random_ils, random_restart_hc, ILSEvent

router = APIRouter(tags=["ils"])


@dataclass
class _AlgoDone:
    """Sentinel pushed to queue when one generator finishes."""
    algo: str
    last_event: ILSEvent | None
    trajectory: list[int]


def _drain_generator(
    gen,
    queue: asyncio.Queue,
    cancel: threading.Event,
    loop: asyncio.AbstractEventLoop,
    idx_map: dict[int, int],
):
    """Run a single ILS generator in a worker thread, pushing events to the async queue."""
    trajectory: list[int] = []
    last_event: ILSEvent | None = None
    algo_name = ""
    try:
        for event in gen:
            if cancel.is_set():
                break
            last_event = event
            algo_name = event.algo
            mapped_idx = idx_map.get(event.current_optimum, -1)
            trajectory.append(mapped_idx)
            loop.call_soon_threadsafe(queue.put_nowait, (event, mapped_idx))
    except Exception:
        pass
    loop.call_soon_threadsafe(
        queue.put_nowait,
        _AlgoDone(algo=algo_name, last_event=last_event, trajectory=trajectory),
    )


@router.websocket("/ws/ils/stream")
async def stream_ils(ws: WebSocket):
    """Run ORC+Pert, Random-ILS, and RR-HC concurrently on the same instance.

    Client sends:
        {"instance_id": "...", "budget": 5000, "d_r": 2, "seed": 42}

    Server streams interleaved events from all three algorithms:
        {"type": "iteration", "algo": "orc"|"random"|"rrhc", ...}
        {"type": "complete", "winner": "...", "results": [...]}
    """
    await ws.accept()
    cancel = threading.Event()
    try:
        msg = await ws.receive_json()
        instance_id = msg.get("instance_id")
        budget = msg.get("budget", 5000)
        d_r = msg.get("d_r", 2)
        seed = msg.get("seed")

        cached = cache.get(instance_id)
        if cached is None:
            await ws.send_json({"type": "error", "message": "Instance not found"})
            await ws.close()
            return

        space = cached.space
        idx_map = {opt.idx: i for i, opt in enumerate(cached.optima)}

        generators = [
            orc_ils(space, budget=budget, d_r=d_r, gamma=1.0, seed=seed),
            random_ils(space, budget=budget, d_r_total=1 + d_r, seed=seed),
            random_restart_hc(space, budget=budget, seed=seed),
        ]

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        async def listen_for_cancel():
            try:
                while not cancel.is_set():
                    data = await ws.receive_json()
                    if data.get("type") == "cancel":
                        cancel.set()
                        return
            except (WebSocketDisconnect, Exception):
                cancel.set()

        cancel_task = asyncio.create_task(listen_for_cancel())

        with ThreadPoolExecutor(max_workers=3) as pool:
            for gen in generators:
                pool.submit(_drain_generator, gen, queue, cancel, loop, idx_map)

            done_count = 0
            results: list[dict] = []

            while done_count < 3:
                item = await queue.get()

                if isinstance(item, _AlgoDone):
                    done_count += 1
                    if item.last_event is not None:
                        results.append({
                            "algo": item.algo,
                            "best_fitness": round(item.last_event.best_fitness, 6),
                            "total_evals": item.last_event.evals,
                            "trajectory": item.trajectory,
                        })
                    continue

                event, mapped_idx = item
                try:
                    await ws.send_json({
                        "type": "iteration",
                        "algo": event.algo,
                        "evals": event.evals,
                        "best_fitness": round(event.best_fitness, 6),
                        "current_optimum": mapped_idx,
                    })
                except Exception:
                    cancel.set()
                    break

        cancel_task.cancel()

        if cancel.is_set():
            await ws.send_json({"type": "cancelled", "results": results})
        else:
            winner = max(results, key=lambda r: r["best_fitness"])["algo"] if results else ""
            await ws.send_json({
                "type": "complete",
                "winner": winner,
                "results": results,
            })

    except WebSocketDisconnect:
        cancel.set()
    except Exception as e:
        cancel.set()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
