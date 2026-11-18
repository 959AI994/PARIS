"""Candidate indexing, statistics, and DOPP-style simulated annealing."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import DesignCache
from .errors import ValidationError
from .io import Candidate, load_candidates, write_json


@dataclass
class CandidateStats:
    design: str
    candidate_count: int
    object_count: int
    unique_assignments: int
    die_count: int
    generator_counts: dict[str, int]


def index_candidates(path: str | Path, output: str | Path) -> dict[str, Any]:
    candidates = load_candidates(path)
    rows = []
    generator_counts: dict[str, int] = {}
    for candidate in candidates:
        generator = str(candidate.metadata.get("generator", "unknown"))
        generator_counts[generator] = generator_counts.get(generator, 0) + 1
        rows.append(
            {
                "design": candidate.design,
                "candidate_id": candidate.candidate_id,
                "assignment_hash": candidate.assignment_hash,
                "granularity": candidate.granularity,
                "die_count": candidate.die_count,
                "generator": generator,
                "seed": candidate.metadata.get("seed"),
                "metadata": json.dumps(candidate.metadata, sort_keys=True),
            }
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(target, index=False, compression="zstd")
    stats = CandidateStats(
        design=candidates[0].design,
        candidate_count=len(candidates),
        object_count=len(candidates[0].assignment),
        unique_assignments=len({candidate.assignment_hash for candidate in candidates}),
        die_count=candidates[0].die_count,
        generator_counts=generator_counts,
    )
    write_json(target.with_suffix(target.suffix + ".stats.json"), stats.__dict__)
    return stats.__dict__


def _proxy(cache: DesignCache, assignment: np.ndarray) -> tuple[float, float]:
    areas = cache.nodes["area"].to_numpy(dtype=float)
    total = np.asarray(cache.cell_area_by_die, dtype=float).copy()
    for die in range(cache.die_count):
        total[die] += areas[assignment == die].sum()
    imbalance = (total.max() - total.min()) / max(total.sum(), 1e-12)
    weights = cache.nets["weight"].to_numpy(dtype=float)
    if len(weights) == 0:
        return 0.0, float(imbalance)
    dies = assignment[cache.net_pins]
    starts = cache.net_offsets[:-1]
    # Empty nets are impossible in a valid cache; reduceat covers each net's pin slice.
    dmin = np.minimum.reduceat(dies, starts)
    dmax = np.maximum.reduceat(dies, starts)
    cut = float(weights[dmin != dmax].sum() / max(float(weights.sum()), 1e-12))
    return cut, float(imbalance)


def _area_balanced_assignment(cache: DesignCache, rng: random.Random) -> np.ndarray:
    """Greedy area-aware init so SA starts inside the feasible imbalance band."""
    areas = cache.nodes["area"].to_numpy(dtype=float)
    assignment = np.zeros(len(areas), dtype=np.int16)
    totals = np.asarray(cache.cell_area_by_die, dtype=float).copy()
    order = list(range(len(areas)))
    rng.shuffle(order)
    order.sort(key=lambda index: (areas[index], rng.random()), reverse=True)
    for index in order:
        if abs(totals[0] - totals[1]) < 1e-9:
            die = rng.randrange(2)
        else:
            die = 0 if totals[0] <= totals[1] else 1
        # occasional exploratory flip while still roughly balanced
        if rng.random() < 0.15:
            die = 1 - die
        assignment[index] = die
        totals[die] += areas[index]
    return assignment


def _random_feasible_assignment(
    cache: DesignCache, rng: random.Random, area_tolerance: float
) -> np.ndarray:
    """Sample random partitions until area imbalance is acceptable."""
    n_objects = len(cache.object_names)
    best = _area_balanced_assignment(cache, rng)
    _, best_imbalance = _proxy(cache, best)
    for _ in range(200):
        assignment = np.asarray([rng.randrange(2) for _ in range(n_objects)], dtype=np.int16)
        _, imbalance = _proxy(cache, assignment)
        if imbalance < best_imbalance:
            best, best_imbalance = assignment, imbalance
        if imbalance <= area_tolerance:
            return assignment
    return best


def _grid_key(cut: float, imbalance: float, grid_size: float) -> tuple[int, int]:
    return (
        int(math.floor(math.log1p(cut) / grid_size)),
        int(math.floor(math.log1p(imbalance) / grid_size)),
    )


def generate_dopp_candidates(
    cache: DesignCache,
    output: str | Path,
    *,
    seed: int = 3,
    iterations: int = 10000,
    grid_size: float = 0.02,
    temperature: float = 1.0,
    cooling: float = 0.999,
    area_tolerance: float = 0.25,
    max_candidates: int | None = None,
) -> list[Candidate]:
    if cache.die_count != 2:
        raise ValidationError("DOPP SA currently supports two dies")
    rng = random.Random(seed)
    names = cache.object_names
    if len(names) <= 40:
        assignment = _random_feasible_assignment(cache, rng, area_tolerance)
    else:
        assignment = _area_balanced_assignment(cache, rng)
    current_cut, current_imbalance = _proxy(cache, assignment)
    archive: dict[tuple[int, int], tuple[float, float, np.ndarray, int]] = {}

    def keep(candidate_assignment: np.ndarray, cut: float, imbalance: float, step: int) -> None:
        if imbalance > area_tolerance:
            return
        key = _grid_key(cut, imbalance, grid_size)
        previous = archive.get(key)
        if previous is None or (cut <= previous[0] and imbalance <= previous[1]):
            archive[key] = (cut, imbalance, candidate_assignment.copy(), step)

    keep(assignment, current_cut, current_imbalance, 0)
    current_temperature = temperature
    for step in range(1, iterations + 1):
        proposal = assignment.copy()
        if rng.random() < 0.35:
            # single-object die flip explores more of the partition space
            index = rng.randrange(len(names))
            proposal[index] = 1 - int(proposal[index])
        else:
            first = rng.randrange(len(names))
            other_die = 1 - int(assignment[first])
            choices = np.flatnonzero(assignment == other_die)
            if not len(choices):
                proposal[first] = other_die
            else:
                second = int(choices[rng.randrange(len(choices))])
                proposal[first], proposal[second] = proposal[second], proposal[first]
        cut, imbalance = _proxy(cache, proposal)
        current_objective = current_cut + current_imbalance
        proposal_objective = cut + imbalance
        accept = proposal_objective <= current_objective or rng.random() < math.exp(
            -(proposal_objective - current_objective) / max(current_temperature, 1e-12)
        )
        if accept:
            assignment = proposal
            current_cut, current_imbalance = cut, imbalance
        keep(proposal, cut, imbalance, step)
        current_temperature *= cooling
    entries = sorted(archive.values(), key=lambda item: (item[0] + item[1], item[0], item[1]))
    if max_candidates is not None:
        entries = entries[:max_candidates]
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, (cut, imbalance, values, step) in enumerate(entries):
        candidate = Candidate(
            design=cache.name,
            candidate_id=f"{cache.name}_dopp_{seed}_{index:05d}",
            assignment={name: int(values[position]) for position, name in enumerate(names)},
            metadata={
                "generator": "dopp_sa_grid",
                "seed": seed,
                "iteration": step,
                "proxy_cut": cut,
                "proxy_area_imbalance": imbalance,
                "grid_size": grid_size,
            },
        )
        candidate.dump(directory / f"{candidate.candidate_id}.json")
        candidates.append(candidate)
    write_json(
        directory / "manifest.json",
        {
            "design": cache.name,
            "generator": "dopp_sa_grid",
            "seed": seed,
            "iterations": iterations,
            "candidate_count": len(candidates),
            "grid_size": grid_size,
        },
    )
    return candidates
