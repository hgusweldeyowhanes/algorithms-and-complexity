"""Timing harness across input sizes."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from algo_benchmarker.algorithms import AlgoFn


@dataclass
class SizeSample:
    n: int
    mean_seconds: float
    runs: int


@dataclass
class BenchmarkResult:
    name: str
    expected: str
    samples: list[SizeSample]


def _time_once(fn: AlgoFn, n: int, rng: random.Random) -> float:
    t0 = time.perf_counter()
    fn(n, rng)
    return time.perf_counter() - t0


def run_benchmark(
    name: str,
    fn: AlgoFn,
    sizes: list[int],
    *,
    expected: str = "",
    runs: int = 3,
    warmup: int = 1,
    seed: int = 1,
) -> BenchmarkResult:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if not sizes:
        raise ValueError("sizes must be non-empty")

    samples: list[SizeSample] = []
    for n in sizes:
        rng = random.Random(seed + n)
        for _ in range(warmup):
            _time_once(fn, n, random.Random(seed))
        times: list[float] = []
        for r in range(runs):
            times.append(_time_once(fn, n, random.Random(seed + n * 17 + r)))
        samples.append(SizeSample(n=n, mean_seconds=sum(times) / len(times), runs=runs))

    return BenchmarkResult(name=name, expected=expected, samples=samples)


def format_table(result: BenchmarkResult) -> str:
    lines = [
        f"algorithm: {result.name}"
        + (f"  (expected {result.expected})" if result.expected else ""),
        f"{'n':>10} {'mean_s':>12} {'ops_hint':>14}",
        "-" * 40,
    ]
    for s in result.samples:
        # rough throughput hint
        rate = s.n / s.mean_seconds if s.mean_seconds > 0 else float("inf")
        lines.append(f"{s.n:10d} {s.mean_seconds:12.6f} {rate:14.1f}/s")
    return "\n".join(lines)
