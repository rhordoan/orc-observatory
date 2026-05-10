# ORC Observatory

Interactive web platform for fitness landscape analysis using Ollivier-Ricci curvature.

## What it does

The ORC Observatory lets researchers explore combinatorial fitness landscapes through a geometric lens. Given a problem instance (NK landscape, W-model, MAX-SAT), it:

1. Enumerates local optima and their basins of attraction
2. Computes fitness-lifted Ollivier-Ricci curvature on the search graph
3. Builds the **ORC Transition Graph (OTG)**, a deterministic, parameter-free directed graph that reveals funnel structure and escape directions
4. Visualizes the OTG interactively with animated construction, curvature-colored edges, and funnel highlighting
5. Compares the OTG against classical Local Optima Networks (LON-d1) side by side
6. Races three ILS variants (ORC-guided, random perturbation, random-restart HC) with live convergence streaming and algorithm recommendation based on OTG cycle structure

## Architecture

Three-tier design:

```
frontend/    Next.js 16 + shadcn/ui + Canvas2D + Recharts
backend/     FastAPI + Pydantic + WebSocket streaming
lib/         Pure Python computation (NumPy, SciPy) + optional GPU (CuPy, Numba)
```

Single-command startup via Docker Compose.

## Quick start

```bash
docker compose up
```

Then open http://localhost:3000.

## Development

### Backend

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### GPU acceleration (optional)

If a CUDA-compatible GPU is available, install the optional dependencies for accelerated fitness precomputation and parallel hill climbing:

```bash
pip install cupy-cuda12x numba
```

The frontend detects GPU availability automatically and shows a toggle in the sidebar. With GPU enabled, the problem size slider extends from N=14 to N=20.

### Tests

```bash
pytest tests/ -v
```

## Project structure

```
orc-observatory/
  lib/                    Pure Python computation library
    search_spaces/        SearchSpace protocol + NK, W-model, MAX-SAT
    orc.py                Fitness-lifted ORC computation
    otg.py                OTG construction, funnel analysis, parallelized ORC
    lon.py                LON-d1 construction for comparison
    hill_climb.py         Hill climbing and local optima enumeration
    metrics.py            FDC, autocorrelation, information content
    ils.py                ILS race generators (ORC-guided, random, RR-HC)
    gpu_accel.py          Optional CuPy/Numba GPU acceleration
  backend/                FastAPI API tier
    routers/              REST + WebSocket endpoint definitions
    models/               Pydantic request/response schemas
    main.py               App entrypoint + GPU status endpoint
  frontend/               Next.js 16 frontend
    src/components/
      graph-canvas.tsx    Canvas2D force-directed graph renderer
      race-view.tsx       Full-screen algorithm race tab with convergence chart
      sidebar.tsx         Problem configuration + GPU toggle
      detail-panel.tsx    Node ORC detail + transport decomposition
      metrics-bar.tsx     OTG/LON structural metrics strip
  tests/                  Unit and integration tests
  docker-compose.yml      Orchestration
```

## License

MIT
