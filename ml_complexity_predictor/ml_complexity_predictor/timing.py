"""High-precision timing across input sizes."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from ml_complexity_predictor.algorithms import AlgoFn


@dataclass
class SizeSample:
    n: int
    mean_seconds: float
    runs: int


@dataclass
class Curve:
    """Empirical (n, T(n)) curve for one algorithm run."""

    name: str
    theoretical: str
    samples: list[SizeSample]

    @property
    def ns(self) -> list[int]:
        return [s.n for s in self.samples]

    @property
    def times(self) -> list[float]:
        return [s.mean_seconds for s in self.samples]


def _time_once(fn: AlgoFn, n: int, rng: random.Random) -> float:
    t0 = time.perf_counter()
    fn(n, rng)
    return time.perf_counter() - t0


def measure_curve(
    name: str,
    fn: AlgoFn,
    sizes: list[int],
    *,
    theoretical: str = "",
    runs: int = 3,
    warmup: int = 1,
    seed: int = 1,
) -> Curve:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if len(sizes) < 3:
        raise ValueError("need at least 3 sizes for a usable curve")

    samples: list[SizeSample] = []
    for n in sizes:
        for _ in range(warmup):
            _time_once(fn, n, random.Random(seed))
        times: list[float] = []
        for r in range(runs):
            times.append(_time_once(fn, n, random.Random(seed + n * 17 + r)))
        samples.append(SizeSample(n=n, mean_seconds=sum(times) / len(times), runs=runs))
    return Curve(name=name, theoretical=theoretical, samples=samples)


def format_curve(curve: Curve) -> str:
    lines = [
        f"algorithm: {curve.name}"
        + (f"  (theoretical {curve.theoretical})" if curve.theoretical else ""),
        f"{'n':>10} {'mean_s':>12}",
        "-" * 24,
    ]
    for s in curve.samples:
        lines.append(f"{s.n:10d} {s.mean_seconds:12.6f}")
    return "\n".join(lines)
