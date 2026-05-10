"""ILS race endpoint -- streams iteration events for all three variants."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend import cache
from lib.ils import orc_ils, random_ils, random_restart_hc, ILSEvent

router = APIRouter(tags=["ils"])


def _collect_events(gen: Generator[ILSEvent, None, None]) -> list[ILSEvent]:
    """Drain a generator in a thread-safe way."""
    return list(gen)


@router.websocket("/ws/ils/stream")
async def stream_ils(ws: WebSocket):
    """Run ORC+Pert, Random-ILS, and RR-HC on the same instance.

    Client sends:
        {"instance_id": "...", "budget": 5000, "d_r": 2, "seed": 42, "pace_ms": 50}

    Server streams:
        {"type": "iteration", "algo": "orc"|"random"|"rrhc", "evals": n,
         "best_fitness": f, "current_optimum": idx}
        {"type": "complete", "winner": "...", "results": [...]}
    """
    await ws.accept()
    try:
        msg = await ws.receive_json()
        instance_id = msg.get("instance_id")
        budget = msg.get("budget", 5000)
        d_r = msg.get("d_r", 2)
        seed = msg.get("seed")
        pace_ms = msg.get("pace_ms", 50)

        cached = cache.get(instance_id)
        if cached is None:
            await ws.send_json({"type": "error", "message": "Instance not found"})
            await ws.close()
            return

        space = cached.space

        algo_specs = [
            ("orc", orc_ils(space, budget=budget, d_r=d_r, gamma=1.0, seed=seed)),
            ("random", random_ils(space, budget=budget, d_r_total=1 + d_r, seed=seed)),
            ("rrhc", random_restart_hc(space, budget=budget, seed=seed)),
        ]

        results = []
        cancelled = False

        for algo_name, gen in algo_specs:
            if cancelled:
                break

            all_events = await asyncio.to_thread(_collect_events, gen)

            trajectory: list[int] = []
            last_event = None

            for event in all_events:
                trajectory.append(event.current_optimum)
                last_event = event

                await ws.send_json({
                    "type": "iteration",
                    "algo": event.algo,
                    "evals": event.evals,
                    "best_fitness": round(event.best_fitness, 6),
                    "current_optimum": event.current_optimum,
                })

                if pace_ms > 0:
                    await asyncio.sleep(pace_ms / 1000.0)

            if last_event is not None:
                results.append({
                    "algo": algo_name,
                    "best_fitness": round(last_event.best_fitness, 6),
                    "total_evals": last_event.evals,
                    "trajectory": trajectory,
                })

        if cancelled:
            await ws.send_json({"type": "cancelled", "results": results})
        else:
            winner = max(results, key=lambda r: r["best_fitness"])["algo"] if results else ""
            await ws.send_json({
                "type": "complete",
                "winner": winner,
                "results": results,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
