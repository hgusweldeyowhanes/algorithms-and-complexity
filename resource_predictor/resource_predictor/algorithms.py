"""Workload algorithms spanning sorting, searching, and graph work."""
from __future__ import annotations

import random
from collections import deque
from typing import Callable


AlgoFn = Callable[[int, random.Random], object]

# Stable id order for one-hot encoding
ALGORITHM_NAMES: tuple[str, ...] = (
    "bubble_sort",
    "merge_sort",
    "linear_search",
    "binary_search",
    "bfs",
    "matrix_multiply",
)


def bubble_sort(n: int, rng: random.Random) -> list[int]:
    a = [rng.randrange(n * 2 + 1) for _ in range(n)]
    for i in range(len(a)):
        for j in range(0, len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a


def merge_sort(n: int, rng: random.Random) -> list[int]:
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


def linear_search(n: int, rng: random.Random) -> int:
    target = rng.randrange(n) if n else 0
    hits = 0
    for i in range(n):
        if i == target:
            hits += 1
    return hits


def binary_search(n: int, rng: random.Random) -> int:
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


def bfs(n: int, rng: random.Random) -> int:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u in range(n - 1):
        adj[u].append(u + 1)
        adj[u + 1].append(u)
    for _ in range(max(0, n // 5)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    seen = [False] * n
    q: deque[int] = deque([0])
    seen[0] = True
    count = 0
    while q:
        u = q.popleft()
        count += 1
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return count


def matrix_multiply(n: int, rng: random.Random) -> float:
    """O(n³) dense multiply — good memory + CPU signal. n is matrix side length."""
    side = max(2, min(n, 120))
    a = [[rng.random() for _ in range(side)] for _ in range(side)]
    b = [[rng.random() for _ in range(side)] for _ in range(side)]
    total = 0.0
    for i in range(side):
        for j in range(side):
            s = 0.0
            for k in range(side):
                s += a[i][k] * b[k][j]
            total += s
    return total


ALGORITHMS: dict[str, AlgoFn] = {
    "bubble_sort": bubble_sort,
    "merge_sort": merge_sort,
    "linear_search": linear_search,
    "binary_search": binary_search,
    "bfs": bfs,
    "matrix_multiply": matrix_multiply,
}


def size_grid(name: str) -> list[int]:
    if name == "bubble_sort":
        return [40, 60, 80, 100, 120, 160, 200]
    if name == "merge_sort":
        return [200, 400, 800, 1600, 3200, 5000, 8000]
    if name == "linear_search":
        return [20_000, 50_000, 100_000, 200_000, 400_000, 600_000]
    if name == "binary_search":
        return [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
    if name == "bfs":
        return [500, 1000, 2000, 4000, 8000, 12000]
    if name == "matrix_multiply":
        return [20, 30, 40, 50, 60, 70, 80]
    return [100, 200, 400, 800]
