# ORC Observatory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/)

A Python library and interactive web platform for **fitness landscape analysis** using Ollivier–Ricci curvature (ORC).

ORC Observatory computes curvature on combinatorial search graphs and uses it to build the **ORC Transition Graph (OTG)** — a deterministic, parameter-free directed graph that reveals funnel structure and escape directions in fitness landscapes. The OTG consistently outperforms classical Local Optima Networks (LONs) at predicting search difficulty and guiding metaheuristic perturbation.

## Library usage

The core library (`lib/`) is a standalone Python package with no web dependencies.

```python
from lib.search_spaces.nk import NKLandscape
from lib.hill_climb import enumerate_optima
from lib.orc import compute_all_orc
from lib.otg import build_otg

space = NKLandscape(n=14, k=4, seed=42)
optima = enumerate_optima(space)

orc_values = compute_all_orc(space, optima[0].idx, gamma=1.0)
otg = build_otg(space, optima)

print(f"{len(optima)} local optima, OTG depth = {otg.dag_depth}")
print(f"Funnel compression: {otg.compression_ratio:.0%}")
```

### Supported search spaces

| Space | Neighborhood | Module |
|-------|-------------|--------|
| NK landscape | Bit-flip | `lib.search_spaces.nk` |
| W-model | Bit-flip (neutrality, epistasis, ruggedness) | `lib.search_spaces.wmodel` |
| MAX-SAT | Bit-flip | `lib.search_spaces.maxsat` |
| TSPLIB | 2-opt | `lib.search_spaces.tsplib` |
| QAPLIB | Transposition swap | `lib.search_spaces.qaplib` |

New search spaces only need to implement the `SearchSpace` protocol (see `lib/search_spaces/protocol.py`).

### ILS variants

The library ships several Iterated Local Search variants that can be used independently:

```python
from lib.ils import orc_ils, random_ils, simulated_annealing

for event in orc_ils(space, budget=5000, seed=0):
    print(f"evals={event.evals}  best={event.best_fitness:.2f}")
```

Available: `orc_ils`, `random_ils`, `random_restart_hc`, `mingap_ils`, `simulated_annealing`, `tabu_search`, `one_plus_one_ea`, `variable_neighborhood_search`.

## Web platform

The interactive frontend visualizes OTG construction, curvature-colored edges, funnel structure, and live algorithm races with convergence streaming.

```bash
docker compose up        # http://localhost:3000
```

### Architecture

```
lib/         Pure Python computation (NumPy, SciPy) + optional GPU (CuPy, Numba)
backend/     FastAPI + Pydantic + WebSocket streaming
frontend/    Next.js 16 + shadcn/ui + Canvas2D + Recharts
```

## Quick start

### As a library

```bash
pip install -r requirements.txt
python -c "from lib.search_spaces.nk import NKLandscape; print(NKLandscape(14,4).name)"
```

### Full platform (Docker)

```bash
docker compose up
```

### Development

```bash
# Backend
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev

# GPU acceleration (optional)
pip install cupy-cuda12x numba
```

### Tests

```bash
pytest tests/ -v
```

## Experiments

The `experiments/` directory contains the full benchmark suite: escape-rate measurement, fitness-shuffle ablation, ILS comparison, OTG-based algorithm selection, and scalability analysis. See [`experiments/README.md`](experiments/README.md) for configuration and cluster deployment.

```bash
pip install -r experiments/requirements-experiments.txt
python -m experiments.runner --config experiments/configs/exp_a_reproduce.yaml --output results/
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Good first issues are labeled [`good first issue`](https://github.com/rhordoan/orc-observatory/labels/good%20first%20issue) on the tracker.

## License

[MIT](LICENSE)
