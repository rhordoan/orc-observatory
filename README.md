# ORC Observatory

Interactive web platform for fitness landscape analysis using Ollivier-Ricci curvature.

## What it does

The ORC Observatory lets researchers explore combinatorial fitness landscapes through a geometric lens. Given a problem instance (NK landscape, W-model, MAX-SAT), it:

1. Enumerates local optima and their basins of attraction
2. Computes fitness-lifted Ollivier-Ricci curvature on the search graph
3. Builds the **ORC Transition Graph (OTG)**, a deterministic, parameter-free directed graph that reveals funnel structure and escape directions
4. Visualizes the OTG interactively with animated construction, curvature-colored edges, and funnel highlighting
5. Compares the OTG against classical Local Optima Networks (LON-d1) side by side

## Architecture

Three-tier design:

```
frontend/    Next.js 14 + shadcn/ui + D3.js
backend/     FastAPI + Pydantic + WebSocket streaming
lib/         Pure Python computation (NumPy, SciPy, NetworkX)
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
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

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
    otg.py                OTG construction and funnel analysis
    lon.py                LON-d1 construction for comparison
    hill_climb.py         Hill climbing and local optima enumeration
    metrics.py            FDC, autocorrelation, information content
  backend/                FastAPI API tier
    routers/              REST + WebSocket endpoint definitions
    services/             Business logic layer
    models/               Pydantic request/response schemas
    main.py               App entrypoint
  frontend/               Next.js frontend
  tests/                  Unit, integration, and e2e tests
  docker-compose.yml      Orchestration
```

## License

MIT
