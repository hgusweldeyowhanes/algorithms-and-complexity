"""
Metric TSP approximation algorithms.

For Euclidean (metric) instances:
  - MST double-tree: 2-approximation (tour ≤ 2 · OPT)
"""
from __future__ import annotations

from tsp.exact import TourResult
from tsp.instance import Instance


def _mst_prim(instance: Instance) -> list[tuple[int, int]]:
    """Return MST edges using Prim's algorithm from city 0."""
    n = instance.n
    in_tree = [False] * n
    parent = [-1] * n
    key = [float("inf")] * n
    key[0] = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: key[i])
        in_tree[u] = True
        for v in range(n):
            if not in_tree[v]:
                d = instance.dist(u, v)
                if d < key[v]:
                    key[v] = d
                    parent[v] = u
    edges: list[tuple[int, int]] = []
    for v in range(1, n):
        if parent[v] >= 0:
            edges.append((parent[v], v))
    return edges


def _adjacency(n: int, edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _euler_tour(adj: dict[int, list[int]], start: int = 0) -> list[int]:
    """Hierholzer on an undirected multigraph (adjacency lists, edges removed as used)."""
    # work on mutable copy of multisets
    graph = {u: list(vs) for u, vs in adj.items()}
    stack = [start]
    path: list[int] = []
    while stack:
        u = stack[-1]
        if graph[u]:
            v = graph[u].pop()
            graph[v].remove(u)
            stack.append(v)
        else:
            path.append(stack.pop())
    path.reverse()
    return path


def _shortcut(euler: list[int], n: int) -> list[int]:
    """Shortcutting: keep first occurrence of each vertex (metric inequality)."""
    seen = [False] * n
    tour: list[int] = []
    for v in euler:
        if not seen[v]:
            seen[v] = True
            tour.append(v)
    return tour


def mst_double_approx(instance: Instance, *, start: int = 0) -> TourResult:
    """
    Classic **2-approximation** for metric TSP:

    1. Build MST
    2. Double every edge → Eulerian multigraph
    3. Euler tour, then shortcut repeated cities

    Guarantee (metric): length(tour) ≤ 2 · OPT.
    """
    n = instance.n
    if start < 0 or start >= n:
        raise ValueError("invalid start city")
    if n == 1:
        return TourResult(tour=[start], length=0.0, method="mst-double-2approx")

    mst = _mst_prim(instance)
    # double edges
    doubled = mst + mst
    adj = _adjacency(n, doubled)
    # start Euler from requested city if possible
    euler_start = start if adj[start] else 0
    euler = _euler_tour(adj, euler_start)
    tour = _shortcut(euler, n)
    # rotate so `start` is first
    if start in tour:
        i = tour.index(start)
        tour = tour[i:] + tour[:i]
    return TourResult(
        tour=tour,
        length=instance.tour_length(tour),
        method="mst-double-2approx",
    )
