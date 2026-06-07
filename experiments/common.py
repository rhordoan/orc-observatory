"""Shared utilities for batch experiments."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is on path when running as script
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.search_spaces import (
    NKSearchSpace,
    WModelSearchSpace,
    MaxSATSearchSpace,
    TSPSearchSpace,
    QAPSearchSpace,
    GraphBisectionSearchSpace,
)
from lib.search_spaces.protocol import SearchSpace


def make_space(spec: dict[str, Any], use_gpu: bool = False) -> SearchSpace:
    """Instantiate a search space from a config dict."""
    kind = spec["type"]
    seed = spec.get("seed", 0)

    if kind == "nk":
        return NKSearchSpace(
            n=spec["n"],
            k=spec["k"],
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "wmodel":
        return WModelSearchSpace(
            n=spec["n"],
            mu=spec.get("mu", 1),
            nu=spec["nu"],
            gamma=spec.get("gamma", 0),
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "maxsat":
        n = spec["n"]
        alpha = spec.get("alpha", 4.27)
        return MaxSATSearchSpace(
            n_vars=n,
            n_clauses=int(alpha * n) if spec.get("n_clauses") is None else spec["n_clauses"],
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "tsp":
        return TSPSearchSpace(
            n_cities=spec["n"],
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "qap":
        return QAPSearchSpace(
            n=spec["n"],
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "tsplib":
        from lib.search_spaces.tsplib import TSPLIBSearchSpace

        return TSPLIBSearchSpace(
            instance_name=spec["instance"],
            data_dir=spec.get("data_dir"),
            n_restarts=spec.get("n_restarts", 500),
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "qaplib":
        from lib.search_spaces.qaplib import QAPLIBSearchSpace

        return QAPLIBSearchSpace(
            instance_name=spec["instance"],
            data_dir=spec.get("data_dir"),
            n_restarts=spec.get("n_restarts", 500),
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "flowshop":
        from lib.search_spaces.flowshop import FlowshopSearchSpace

        return FlowshopSearchSpace(
            instance_name=spec["instance"],
            data_dir=spec.get("data_dir"),
            n_restarts=spec.get("n_restarts", 500),
            seed=seed,
            use_gpu=use_gpu,
        )
    if kind == "bisection":
        return GraphBisectionSearchSpace(
            n=spec["n"],
            edge_prob_within=spec.get("p_in", 0.7),
            edge_prob_between=spec.get("p_out", 0.1),
            model=spec.get("model", "planted"),
            seed=seed,
            use_gpu=use_gpu,
        )
    raise ValueError(f"Unknown space type: {kind}")


def collect_optima(
    space: SearchSpace, cfg: dict[str, Any], return_attractor: bool = False,
):
    """Enumerate or sample local optima per config.

    When *return_attractor* is True (and enumeration is used), returns
    (optima, attractor_array) for GPU OTG construction.
    """
    from lib.hill_climb import LocalOptimum, enumerate_local_optima, random_restart_optima

    kind = cfg.get("type", "")
    if kind in ("tsplib", "qaplib", "flowshop"):
        optima = [
            LocalOptimum(idx=i, fitness=space.fitness(i), basin=[i])
            for i in range(space.size)
        ]
        return (optima, None) if return_attractor else optima

    mode = cfg.get("optima_mode", "enumerate")
    if mode == "enumerate":
        result = enumerate_local_optima(
            space,
            use_gpu=cfg.get("use_gpu", False),
            return_attractor=return_attractor,
        )
        return result
    n_restarts = int(cfg.get("n_restarts", 1000))
    optima = random_restart_optima(
        space, n_restarts=n_restarts, seed=cfg.get("seed")
    )
    return (optima, None) if return_attractor else optima


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
