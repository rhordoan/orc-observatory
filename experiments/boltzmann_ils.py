"""Boltzmann (softmax) ORC-guided ILS with self-adaptive temperature."""

from __future__ import annotations

from typing import Generator

import numpy as np

from lib.search_spaces.protocol import SearchSpace
from lib.orc import compute_all_orc
from lib.ils import CountingSpace, ILSEvent, _hill_climb_counted


def _boltzmann_neighbor(
    space: SearchSpace,
    x: int,
    gamma: float,
    rng: np.random.Generator,
) -> int:
    orc_vals = compute_all_orc(space, x, gamma)
    keys = list(orc_vals.keys())
    kappas = np.array([orc_vals[k] for k in keys], dtype=np.float64)
    std = float(np.std(kappas))
    if std < 1e-12:
        return int(rng.choice(keys))
    temp = std
    logits = -kappas / temp
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(keys, p=probs))


def boltzmann_orc_ils(
    space: SearchSpace,
    budget: int = 5000,
    d_r: int = 2,
    gamma: float = 1.0,
    seed: int | None = None,
) -> Generator[ILSEvent, None, None]:
    """ORC escape via softmax over curvature + random diversification."""
    rng = np.random.default_rng(seed)
    cs = CountingSpace(space)

    start = int(rng.integers(0, cs.size))
    x_star = _hill_climb_counted(cs, start)
    best = x_star
    best_fit = cs.fitness(best)

    yield ILSEvent(
        algo="boltzmann",
        evals=cs.eval_count,
        best_fitness=best_fit,
        current_optimum=x_star,
    )

    while cs.eval_count < budget:
        y_star = _boltzmann_neighbor(space, x_star, gamma, rng)
        current = y_star
        for _ in range(d_r):
            nbrs = cs.neighbors(current)
            current = int(rng.choice(nbrs))

        x_star = _hill_climb_counted(cs, current)
        f = cs.fitness(x_star)
        if f > best_fit:
            best_fit = f
            best = x_star

        yield ILSEvent(
            algo="boltzmann",
            evals=cs.eval_count,
            best_fitness=best_fit,
            current_optimum=x_star,
        )
