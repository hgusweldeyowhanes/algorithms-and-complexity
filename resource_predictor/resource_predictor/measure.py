"""Measure execution time and peak Python memory (tracemalloc)."""
from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from resource_predictor.algorithms import AlgoFn
from resource_predictor.system_info import SystemFeatures, collect_system_features


@dataclass
class ResourceSample:
    algorithm: str
    n: int
    time_s: float
    peak_memory_kb: float
    system: SystemFeatures


def _run_once(fn: AlgoFn, n: int, seed: int) -> tuple[float, float]:
    """Return (seconds, peak_kibibytes via tracemalloc)."""
    rng = random.Random(seed)
    tracemalloc.start()
    t0 = time.perf_counter()
    fn(n, rng)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024.0


def measure_sample(
    algorithm: str,
    fn: AlgoFn,
    n: int,
    *,
    runs: int = 2,
    seed: int = 1,
    system: SystemFeatures | None = None,
) -> ResourceSample:
    if runs < 1:
        raise ValueError("runs >= 1")
    times: list[float] = []
    peaks: list[float] = []
    for r in range(runs):
        t, peak = _run_once(fn, n, seed=seed + n * 13 + r)
        times.append(t)
        peaks.append(peak)
    sysf = system or collect_system_features()
    return ResourceSample(
        algorithm=algorithm,
        n=n,
        time_s=sum(times) / len(times),
        peak_memory_kb=max(peaks),
        system=sysf,
    )
