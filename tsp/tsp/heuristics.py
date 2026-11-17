"""Heuristic TSP: nearest neighbor construction + 2-opt improvement."""
from __future__ import annotations

from tsp.exact import TourResult
from tsp.instance import Instance


def nearest_neighbor(instance: Instance, *, start: int = 0) -> TourResult:
    """Greedy nearest-neighbor tour starting at `start`."""
    n = instance.n
    if start < 0 or start >= n:
        raise ValueError("invalid start city")
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: instance.dist(cur, j))
        unvisited.remove(nxt)
        tour.append(nxt)
        cur = nxt
    length = instance.tour_length(tour)
    return TourResult(tour=tour, length=length, method="nearest-neighbor")


def two_opt(instance: Instance, tour: list[int], *, max_passes: int = 100) -> TourResult:
    """
    Improve a tour with 2-opt local search (first-improvement).

    Reverses segments while it reduces Euclidean length.
    """
    n = len(tour)
    if n < 4:
        return TourResult(tour=list(tour), length=instance.tour_length(tour), method="2-opt")

    best = list(tour)
    best_len = instance.tour_length(best)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for k in range(i + 2, n if i > 0 else n - 1):
                # edges (i, i+1) and (k, k+1 mod)
                a, b = best[i], best[(i + 1) % n]
                c, d = best[k], best[(k + 1) % n]
                before = instance.dist(a, b) + instance.dist(c, d)
                after = instance.dist(a, c) + instance.dist(b, d)
                if after + 1e-12 < before:
                    # reverse segment i+1 .. k
                    best[i + 1 : k + 1] = reversed(best[i + 1 : k + 1])
                    best_len += after - before
                    improved = True
                    break
            if improved:
                break

    # normalize: rotate so smallest index city is not required; keep start of input
    return TourResult(tour=best, length=instance.tour_length(best), method="nearest-neighbor+2-opt")


def nearest_neighbor_two_opt(instance: Instance, *, start: int = 0) -> TourResult:
    nn = nearest_neighbor(instance, start=start)
    improved = two_opt(instance, nn.tour)
    improved.method = "nearest-neighbor+2-opt"
    return improved
