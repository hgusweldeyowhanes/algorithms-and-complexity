"""Exact TSP via Held–Karp dynamic programming. O(n² 2ⁿ)."""
from __future__ import annotations

from dataclasses import dataclass

from tsp.instance import Instance


@dataclass
class TourResult:
    tour: list[int]  # city indices, starting at 0, not repeating start at end
    length: float
    method: str


def held_karp(instance: Instance, *, start: int = 0) -> TourResult:
    """
    Exact minimum tour (Euclidean) using Held–Karp DP.

    Practical for n ≲ 18–20 depending on machine; raises if n is too large.
    """
    n = instance.n
    if n > 20:
        raise ValueError(f"Held–Karp exact solve limited to n<=20 (got n={n})")
    if start < 0 or start >= n:
        raise ValueError("invalid start city")

    dist = instance.distance_matrix()
    # dp[mask][j] = min cost to reach j from start, visiting exactly cities in mask
    full = 1 << n
    INF = float("inf")
    dp = [[INF] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]

    start_bit = 1 << start
    dp[start_bit][start] = 0.0

    for mask in range(full):
        if (mask & start_bit) == 0:
            continue
        for j in range(n):
            if (mask & (1 << j)) == 0:
                continue
            cost_j = dp[mask][j]
            if cost_j == INF:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                cand = cost_j + dist[j][k]
                if cand < dp[new_mask][k]:
                    dp[new_mask][k] = cand
                    parent[new_mask][k] = j

    all_mask = (1 << n) - 1
    best_len = INF
    best_end = -1
    for j in range(n):
        if j == start:
            continue
        cand = dp[all_mask][j] + dist[j][start]
        if cand < best_len:
            best_len = cand
            best_end = j

    if best_end < 0 or best_len == INF:
        # n == 1 edge case
        return TourResult(tour=[start], length=0.0, method="held-karp")

    # reconstruct path end -> start
    tour_rev = [best_end]
    mask = all_mask
    cur = best_end
    while cur != start:
        prev = parent[mask][cur]
        if prev < 0:
            break
        tour_rev.append(prev)
        mask ^= 1 << cur
        cur = prev
    tour = list(reversed(tour_rev))
    if tour[0] != start:
        # ensure start first
        if start in tour:
            i = tour.index(start)
            tour = tour[i:] + tour[:i]
    return TourResult(tour=tour, length=best_len, method="held-karp")
