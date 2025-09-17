"""Multi-objective optimization helpers."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import random
from dataclasses import dataclass

import numpy as np

Bounds = Dict[str, Tuple[float, float]]


def random_pareto_search(
    evaluate: Callable[[np.ndarray], Tuple[float, float]],
    bounds: Bounds,
    n_samples: int = 1000,
    seed: int | None = None,
) -> List[Dict[str, float]]:
    """Approximate the Pareto front for yield and spot size.

    This routine performs a random search over the parameter space defined by
    ``bounds``.  The ``evaluate`` callable should return ``(yield, spot_size)``
    for a given parameter vector.  Solutions with higher ``yield`` and lower
    ``spot_size`` are preferred.  The method returns the set of nondominated
    parameter dictionaries representing the estimated Pareto front.

    Parameters
    ----------
    evaluate:
        Callable accepting an array of parameters and returning
        ``(yield, spot_size)``.
    bounds:
        Mapping of parameter names to ``(min, max)`` bounds.
    n_samples:
        Number of random samples to draw.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    List[Dict[str, float]]
        Parameter dictionaries corresponding to the estimated Pareto front.
    """

    rng = random.Random(seed)
    names = list(bounds)
    lower = [bounds[n][0] for n in names]
    upper = [bounds[n][1] for n in names]

    params = [
        [rng.uniform(l, u) for l, u in zip(lower, upper)] for _ in range(n_samples)
    ]
    scores = [evaluate(np.array(p)) for p in params]

    yields = [s[0] for s in scores]
    spots = [s[1] for s in scores]
    pareto_mask = [True] * n_samples

    for i in range(n_samples):
        if not pareto_mask[i]:
            continue
        for j in range(n_samples):
            if j == i:
                continue
            if (
                yields[j] >= yields[i]
                and spots[j] <= spots[i]
                and (yields[j] > yields[i] or spots[j] < spots[i])
            ):
                pareto_mask[i] = False
                break

    pareto_params = [p for p, keep in zip(params, pareto_mask) if keep]
    return [
        {name: float(p[idx]) for idx, name in enumerate(names)} for p in pareto_params
    ]


@dataclass
class ConvergenceRecord:
    """Simple container recording solver progress."""

    generation: int
    best_yield: float
    min_spot_size: float


def nsga2(
    evaluate: Callable[[np.ndarray], Tuple[float, float]],
    bounds: Bounds,
    n_generations: int = 50,
    pop_size: int = 100,
    seed: int | None = None,
    *,
    constraint: Callable[[np.ndarray], bool] | None = None,
    return_history: bool = False,
) -> List[Dict[str, float]] | Tuple[List[Dict[str, float]], List[ConvergenceRecord]]:
    """Estimate the Pareto front using a lightweight NSGA-II implementation.

    Parameters
    ----------
    evaluate:
        Callable accepting an array of parameters and returning
        ``(yield, spot_size)``.
    bounds:
        Mapping of parameter names to ``(min, max)`` bounds.
    n_generations:
        Number of evolutionary generations to execute.
    pop_size:
        Population size per generation.
    seed:
        Optional random seed.
    constraint:
        Optional callable returning ``True`` when a candidate parameter vector
        satisfies hardware limits.  Candidates violating the constraint are
        ignored.
    return_history:
        When ``True`` a list of :class:`ConvergenceRecord` instances is
        returned along with the final Pareto parameters.
    """

    rng = random.Random(seed)
    names = list(bounds)
    lower = np.array([bounds[n][0] for n in names], dtype=float)
    upper = np.array([bounds[n][1] for n in names], dtype=float)
    dim = len(names)

    def _random_vector() -> np.ndarray:
        while True:
            vec = np.array([rng.uniform(l, u) for l, u in zip(lower, upper)])
            if constraint is None or constraint(vec):
                return vec

    def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _nondominated_sort(objs: List[Tuple[float, float]]):
        S = [[] for _ in objs]
        n = [0] * len(objs)
        rank = [0] * len(objs)
        fronts: List[List[int]] = [[]]
        for p in range(len(objs)):
            for q in range(len(objs)):
                if _dominates(objs[p], objs[q]):
                    S[p].append(q)
                elif _dominates(objs[q], objs[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts, rank

    def _crowding_distance(
        objs: List[Tuple[float, float]], front: List[int]
    ) -> Dict[int, float]:
        distance = {i: 0.0 for i in front}
        if not front:
            return distance
        for m in range(2):
            front_sorted = sorted(front, key=lambda i: objs[i][m])
            distance[front_sorted[0]] = distance[front_sorted[-1]] = float("inf")
            f_min = objs[front_sorted[0]][m]
            f_max = objs[front_sorted[-1]][m]
            if f_max == f_min:
                continue
            for j in range(1, len(front_sorted) - 1):
                prev_f = objs[front_sorted[j - 1]][m]
                next_f = objs[front_sorted[j + 1]][m]
                distance[front_sorted[j]] += (next_f - prev_f) / (f_max - f_min)
        return distance

    population = [_random_vector() for _ in range(pop_size)]
    scores = [evaluate(p) for p in population]
    objectives = [(-s[0], s[1]) for s in scores]
    history: List[ConvergenceRecord] = []

    for gen in range(n_generations):
        fronts, rank = _nondominated_sort(objectives)
        best_y = max(scores[i][0] for i in fronts[0])
        min_s = min(scores[i][1] for i in fronts[0])
        history.append(ConvergenceRecord(gen, best_y, min_s))
        crowd: Dict[int, float] = {}
        for f in fronts:
            crowd.update(_crowding_distance(objectives, f))

        def _tournament() -> np.ndarray:
            i, j = rng.randrange(pop_size), rng.randrange(pop_size)
            if rank[i] < rank[j]:
                return population[i]
            if rank[j] < rank[i]:
                return population[j]
            return (
                population[i]
                if crowd.get(i, 0.0) > crowd.get(j, 0.0)
                else population[j]
            )

        offspring: List[np.ndarray] = []
        while len(offspring) < pop_size:
            p1 = _tournament()
            p2 = _tournament()
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            for k in range(dim):
                if rng.random() < 0.1:
                    scale = 0.1 * (upper[k] - lower[k])
                    child[k] += rng.gauss(0.0, scale)
            child = np.clip(child, lower, upper)
            if constraint is not None and not constraint(child):
                continue
            offspring.append(child)

        off_scores = [evaluate(c) for c in offspring]
        off_objs = [(-s[0], s[1]) for s in off_scores]

        population.extend(offspring)
        scores.extend(off_scores)
        objectives.extend(off_objs)

        fronts, rank = _nondominated_sort(objectives)
        new_population: List[np.ndarray] = []
        new_scores: List[Tuple[float, float]] = []
        new_objs: List[Tuple[float, float]] = []
        for f in fronts:
            if len(new_population) + len(f) <= pop_size:
                new_population.extend(population[i] for i in f)
                new_scores.extend(scores[i] for i in f)
                new_objs.extend(objectives[i] for i in f)
            else:
                crowd = _crowding_distance(objectives, f)
                sorted_f = sorted(f, key=lambda i: crowd[i], reverse=True)
                remaining = pop_size - len(new_population)
                new_population.extend(population[i] for i in sorted_f[:remaining])
                new_scores.extend(scores[i] for i in sorted_f[:remaining])
                new_objs.extend(objectives[i] for i in sorted_f[:remaining])
                break
        population, scores, objectives = new_population, new_scores, new_objs

    fronts, _ = _nondominated_sort(objectives)
    pareto_params = [population[i] for i in fronts[0]]
    result = [
        {name: float(p[idx]) for idx, name in enumerate(names)} for p in pareto_params
    ]
    if return_history:
        return result, history
    return result


__all__ = ["random_pareto_search", "nsga2", "ConvergenceRecord"]
