"""Built-in algorithms with known asymptotic behavior (for demos)."""
from __future__ import annotations

import random
from typing import Callable


AlgoFn = Callable[[int, random.Random], object]


def linear_sum(n: int, rng: random.Random) -> int:
    """O(n) — sum 0..n-1."""
    total = 0
    for i in range(n):
        total += i
    return total


def binary_search(n: int, rng: random.Random) -> int:
    """O(log n) — search in a sorted range [0, n)."""
    target = rng.randrange(n) if n > 0 else 0
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


def merge_sort(n: int, rng: random.Random) -> list[int]:
    """O(n log n) — sort a random list of length n."""
    data = [rng.randrange(n * 2 + 1) for _ in range(n)]

    def _sort(a: list[int]) -> list[int]:
        if len(a) <= 1:
            return a
        mid = len(a) // 2
        left = _sort(a[:mid])
        right = _sort(a[mid:])
        return _merge(left, right)

    def _merge(a: list[int], b: list[int]) -> list[int]:
        out: list[int] = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                out.append(a[i])
                i += 1
            else:
                out.append(b[j])
                j += 1
        out.extend(a[i:])
        out.extend(b[j:])
        return out

    return _sort(data)


def bubble_sort(n: int, rng: random.Random) -> list[int]:
    """O(n²) — classic bubble sort on a random list."""
    data = [rng.randrange(n * 2 + 1) for _ in range(n)]
    a = list(data)
    for i in range(len(a)):
        for j in range(0, len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a


def fib_recursive(n: int, rng: random.Random) -> int:
    """O(φⁿ) — naive recursive Fibonacci (use small n only)."""
    # Cap to keep demos safe; CLI uses a small size schedule for this algo
    if n > 28:
        raise ValueError("fib_recursive only supports n<=28 in this demo")

    def fib(k: int) -> int:
        if k <= 1:
            return k
        return fib(k - 1) + fib(k - 2)

    return fib(n)


ALGORITHMS: dict[str, tuple[AlgoFn, str]] = {
    "linear_sum": (linear_sum, "O(n)"),
    "binary_search": (binary_search, "O(log n)"),
    "merge_sort": (merge_sort, "O(n log n)"),
    "bubble_sort": (bubble_sort, "O(n^2)"),
    "fib_recursive": (fib_recursive, "O(phi^n)"),
}


def default_sizes(name: str) -> list[int]:
    if name == "fib_recursive":
        return [5, 10, 15, 18, 20, 22]
    if name == "bubble_sort":
        return [50, 100, 150, 200, 250, 300]
    if name == "binary_search":
        return [1000, 5000, 20000, 80000, 200000, 500000]
    if name == "merge_sort":
        return [500, 1000, 2000, 4000, 8000, 12000]
    # linear_sum
    return [50_000, 100_000, 200_000, 400_000, 800_000, 1_200_000]
