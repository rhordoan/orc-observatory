"""TSPLIB instance download and parsing."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np

# Common instances for paper experiments
DEFAULT_INSTANCES = ("eil51", "berlin52", "kroA100", "ch150")

_BASE_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "tsplib"


def download_tsplib(
    names: tuple[str, ...] = DEFAULT_INSTANCES,
    data_dir: Path | None = None,
) -> Path:
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = data_dir / f"{name}.tsp"
        if path.exists():
            continue
        url = f"{_BASE_URL}{name}.tsp"
        urllib.request.urlretrieve(url, path)
    return data_dir


def load_tsplib(
    name: str,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load TSPLIB instance. Returns (coords, dist_matrix, n_cities)."""
    data_dir = data_dir or default_data_dir()
    path = data_dir / f"{name}.tsp"
    if not path.exists():
        download_tsplib((name,), data_dir)

    coords: list[tuple[float, float]] = []
    dimension = None
    edge_weight_type = None

    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DIMENSION"):
                dimension = int(re.split(r"[: ]+", line)[-1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[-1].strip()
            elif line == "NODE_COORD_SECTION":
                break
        if dimension is None:
            raise ValueError(f"Could not parse DIMENSION from {path}")
        for _ in range(dimension):
            parts = f.readline().split()
            if len(parts) < 3:
                break
            coords.append((float(parts[1]), float(parts[2])))

    n = len(coords)
    coords_arr = np.array(coords, dtype=np.float64)

    if edge_weight_type == "EUC_2D":
        dist = _euclidean_dist(coords_arr)
    else:
        dist = _euclidean_dist(coords_arr)

    return coords_arr, dist.astype(np.float64), n


def _euclidean_dist(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            val = int(round(np.sqrt(dx * dx + dy * dy)))
            d[i, j] = d[j, i] = val
    return d
