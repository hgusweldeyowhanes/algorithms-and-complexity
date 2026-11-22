"""
Target algorithms: sorting, searching, graph traversal.

Each entry: (callable(n, rng) -> result, theoretical Big-O label, size schedule).
"""
from __future__ import annotations

import heapq
import random
from collections import deque
from typing import Callable


AlgoFn = Callable[[int, random.Random], object]

# Prediction vocabulary (matches the study brief + log for search)
CLASSES = ("O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)")


# --- Sorting -----------------------------------------------------------------

def bubble_sort(n: int, rng: random.Random) -> list[int]:
    """O(n^2)."""
    a = [rng.randrange(n * 2 + 1) for _ in range(n)]
    for i in range(len(a)):
        for j in range(0, len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a


def merge_sort(n: int, rng: random.Random) -> list[int]:
    """O(n log n)."""
    data = [rng.randrange(n * 2 + 1) for _ in range(n)]

    def _sort(a: list[int]) -> list[int]:
        if len(a) <= 1:
            return a
        mid = len(a) // 2
        left, right = _sort(a[:mid]), _sort(a[mid:])
        out: list[int] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    return _sort(data)


def quick_sort(n: int, rng: random.Random) -> list[int]:
    """O(n log n) average (random pivot)."""
    a = [rng.randrange(n * 2 + 1) for _ in range(n)]

    def _qs(lo: int, hi: int) -> None:
        if lo >= hi:
            return
        pivot = a[rng.randint(lo, hi)]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        _qs(lo, j)
        _qs(i, hi)

    if a:
        _qs(0, len(a) - 1)
    return a


# --- Searching ---------------------------------------------------------------

def linear_search(n: int, rng: random.Random) -> int:
    """O(n) — scan integers 0..n-1 (no list allocation)."""
    target = rng.randrange(n) if n else 0
    # Force a full scan often enough that work is Θ(n), not lucky early exit.
    hits = 0
    for i in range(n):
        if i == target:
            hits += 1
        elif i == target + 1:
            hits += 0
    return hits


def binary_search(n: int, rng: random.Random) -> int:
    """O(log n)."""
    target = rng.randrange(n) if n else 0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if mid < target:
            lo = mid + 1
        elif mid > target:
            hi = mid
        else:
            return mid
    return -1


# --- Graph traversal ---------------------------------------------------------

def _path_graph(n: int, rng: random.Random, *, extras: int = 0) -> list[list[tuple[int, float]]]:
    """Sparse connected graph: a path plus a few deterministic chords (Θ(n) edges)."""
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if n < 2:
        return adj
    for u in range(n - 1):
        w = 1.0 + (u % 7) * 0.01
        adj[u].append((u + 1, w))
        adj[u + 1].append((u, w))
    # Deterministic chords (avoid RNG cost dominating traversal)
    step = max(2, n // max(1, extras)) if extras else n
    for u in range(0, n, step):
        v = (u + step // 2) % n
        if u == v:
            continue
        w = 1.0 + (u % 5) * 0.1
        adj[u].append((v, w))
        adj[v].append((u, w))
    _ = rng  # seed still affects callers that use rng elsewhere
    return adj


def bfs(n: int, rng: random.Random) -> int:
    """O(V+E) ≈ O(n) on a path-like sparse graph."""
    adj = _path_graph(n, rng, extras=max(8, n // 20))
    seen = [False] * n
    q: deque[int] = deque([0])
    seen[0] = True
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for v, _ in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return visited


def dfs(n: int, rng: random.Random) -> int:
    """O(V+E) ≈ O(n) on a path-like sparse graph."""
    adj = _path_graph(n, rng, extras=max(8, n // 20))
    seen = [False] * n
    stack = [0]
    visited = 0
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        visited += 1
        for v, _ in adj[u]:
            if not seen[v]:
                stack.append(v)
    return visited


def dijkstra(n: int, rng: random.Random) -> list[float]:
    """O((V+E) log V) ≈ O(n log n) — denser sparse graph to surface the log factor."""
    adj = _path_graph(n, rng, extras=max(n, 2 * n))
    dist = [float("inf")] * n
    dist[0] = 0.0
    pq: list[tuple[float, int]] = [(0.0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def exponential_subset_sum(n: int, rng: random.Random) -> int:
    """O(2^n) — enumerate all subsets (tiny n only)."""
    if n > 22:
        raise ValueError("exponential_subset_sum only supports n<=22")
    vals = [rng.randrange(1, 10) for _ in range(n)]
    target = sum(vals) // 2
    best = 0
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += vals[i]
        if abs(s - target) < abs(best - target):
            best = s
    return best


ALGORITHMS: dict[str, tuple[AlgoFn, str, str]] = {
    # name -> (fn, theoretical, category)
    "bubble_sort": (bubble_sort, "O(n^2)", "sorting"),
    "merge_sort": (merge_sort, "O(n log n)", "sorting"),
    "quick_sort": (quick_sort, "O(n log n)", "sorting"),
    "linear_search": (linear_search, "O(n)", "searching"),
    "binary_search": (binary_search, "O(log n)", "searching"),
    "bfs": (bfs, "O(n)", "graph"),
    "dfs": (dfs, "O(n)", "graph"),
    "dijkstra": (dijkstra, "O(n log n)", "graph"),
    "exponential_subset_sum": (exponential_subset_sum, "O(2^n)", "other"),
}


def default_sizes(name: str) -> list[int]:
    if name == "exponential_subset_sum":
        return [8, 10, 12, 14, 16, 18]
    if name == "bubble_sort":
        return [80, 120, 160, 200, 240, 280]
    if name in {"merge_sort", "quick_sort"}:
        return [400, 800, 1600, 3200, 6400, 10000]
    if name == "binary_search":
        return [5_000, 20_000, 80_000, 200_000, 500_000, 1_000_000]
    if name == "linear_search":
        return [20_000, 50_000, 100_000, 200_000, 400_000, 800_000]
    if name in {"bfs", "dfs"}:
        return [5_000, 10_000, 20_000, 40_000, 80_000, 120_000]
    if name == "dijkstra":
        return [500, 1_000, 2_000, 4_000, 8_000, 12_000]
    return [1000, 2000, 4000, 8000, 16000, 32000]
