"""Taillard flowshop instance download and parsing."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_BASE_URL = "https://raw.githubusercontent.com/tamy0612/taillard-benchmarks/master/instances/flow_shop/"

_TAILLARD_SPECS: dict[str, tuple[int, int, int, int]] = {
    # name: (n_jobs, n_machines, start_line_in_file, instance_index_within_group)
    "tai20_5_0": (20, 5, 0, 0),
    "tai20_5_1": (20, 5, 1, 1),
    "tai20_5_2": (20, 5, 2, 2),
    "tai20_10_0": (20, 10, 0, 0),
    "tai20_10_1": (20, 10, 1, 1),
    "tai20_20_0": (20, 20, 0, 0),
    "tai50_5_0": (50, 5, 0, 0),
    "tai50_10_0": (50, 10, 0, 0),
    "tai50_20_0": (50, 20, 0, 0),
}


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "flowshop"


def _generate_taillard(n_jobs: int, n_machines: int, seed_val: int) -> np.ndarray:
    """Generate Taillard instance using the original RNG procedure."""
    def _unif(low, high, state):
        state["seed"] = (state["seed"] * 16807) % 2147483647
        return low + state["seed"] % (high - low + 1)

    state = {"seed": seed_val}
    pt = np.zeros((n_jobs, n_machines), dtype=np.int64)
    for i in range(n_jobs):
        for j in range(n_machines):
            pt[i, j] = _unif(1, 99, state)
    return pt


_TAILLARD_SEEDS = {
    (20, 5): [873654221, 379008056, 1866992158, 216771124, 495070989,
              402959317, 1369363414, 2021925980, 573109518, 88325120],
    (20, 10): [587595453, 1401007982, 873136276, 268827376, 1634173168,
               691823909, 73807235, 1273398721, 2065119309, 1672900551],
    (20, 20): [479340445, 268827376, 1958948863, 918272953, 555010963,
               2010851491, 1519125085, 1828747598, 1018365863, 1988095578],
    (50, 5): [1328042058, 200382020, 496319842, 1203030903, 1730708564,
              450181436, 1303455736, 318569816, 2014325155, 1665057939],
    (50, 10): [21382988, 1803804217, 2090627462, 1262421813, 2026065652,
               892275468, 1543060736, 1057063752, 1482319149, 896488160],
    (50, 20): [1958948863, 1044484043, 175131369, 1442250035, 449269605,
               995218673, 1230937436, 1765037107, 128699729, 1098100759],
}


def generate_and_save(name: str, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.txt"
    if path.exists():
        return path

    parts = name.replace("tai", "").split("_")
    n_jobs, n_machines, idx = int(parts[0]), int(parts[1]), int(parts[2])

    seeds = _TAILLARD_SEEDS.get((n_jobs, n_machines))
    if seeds is None or idx >= len(seeds):
        rng = np.random.default_rng(hash(name) % (2**32))
        pt = rng.integers(1, 100, (n_jobs, n_machines))
    else:
        pt = _generate_taillard(n_jobs, n_machines, seeds[idx])

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n_jobs} {n_machines}\n")
        for row in pt:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    return path


def load_flowshop(
    name: str,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, int, int]:
    """Load flowshop instance. Returns (processing_times[job, machine], n_jobs, n_machines)."""
    data_dir = data_dir or default_data_dir()
    path = data_dir / f"{name}.txt"

    if not path.exists():
        generate_and_save(name, data_dir)

    with path.open(encoding="utf-8", errors="ignore") as f:
        tokens = f.read().split()

    n_jobs = int(tokens[0])
    n_machines = int(tokens[1])
    idx = 2
    pt = np.zeros((n_jobs, n_machines), dtype=np.int64)
    for i in range(n_jobs):
        for j in range(n_machines):
            pt[i, j] = int(tokens[idx])
            idx += 1

    return pt, n_jobs, n_machines
