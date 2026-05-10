"""FastAPI application entrypoint."""

import sys
from pathlib import Path

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import instances, otg, lon, orc_explain, ils, metrics

app = FastAPI(
    title="ORC Observatory API",
    version="0.1.0",
    description="REST and WebSocket API for fitness landscape analysis via Ollivier-Ricci curvature.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(instances.router)
app.include_router(otg.router)
app.include_router(lon.router)
app.include_router(orc_explain.router)
app.include_router(ils.router)
app.include_router(metrics.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/gpu/status")
def gpu_status():
    from lib.gpu_accel import is_gpu_available
    return {"available": is_gpu_available()}
