"""
Guess asymptotic class by comparing growth to model curves.

We estimate c for T(n) ≈ c * f(n) via least squares on positive samples,
then score by relative residual. Lowest residual wins.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from algo_benchmarker.benchmark import BenchmarkResult


@dataclass
class ComplexityFit:
    label: str
    residual: float
    constant: float


def _models() -> dict[str, Callable[[int], float]]:
    return {
        "O(1)": lambda n: 1.0,
        "O(log n)": lambda n: math.log2(n) if n > 1 else 1.0,
        "O(n)": lambda n: float(n),
        "O(n log n)": lambda n: n * math.log2(n) if n > 1 else 1.0,
        "O(n^2)": lambda n: float(n * n),
        "O(n^3)": lambda n: float(n**3),
        "O(2^n)": lambda n: float(2**n) if n <= 40 else float("inf"),
        "O(phi^n)": lambda n: float((1.6180339887) ** n) if n <= 40 else float("inf"),
    }


def fit_complexity(result: BenchmarkResult) -> list[ComplexityFit]:
    xs = [s.n for s in result.samples]
    ys = [s.mean_seconds for s in result.samples]
    if len(xs) < 2:
        return []

    fits: list[ComplexityFit] = []
    for label, f in _models().items():
        fs = [f(n) for n in xs]
        if any(not math.isfinite(v) or v <= 0 for v in fs):
            continue
        # least squares: c = sum(y f) / sum(f^2)
        num = sum(y * fv for y, fv in zip(ys, fs))
        den = sum(fv * fv for fv in fs)
        if den <= 0:
            continue
        c = num / den
        if c <= 0:
            continue
        # mean relative residual
        rel = []
        for y, fv in zip(ys, fs):
            pred = c * fv
            if y <= 0:
                continue
            rel.append(abs(pred - y) / y)
        if not rel:
            continue
        residual = sum(rel) / len(rel)
        fits.append(ComplexityFit(label=label, residual=residual, constant=c))

    fits.sort(key=lambda x: x.residual)
    return fits


def format_fits(fits: list[ComplexityFit], *, top: int = 4) -> str:
    if not fits:
        return "no fit"
    lines = [f"{'model':<12} {'rel_err':>10} {'c':>12}", "-" * 36]
    for fit in fits[:top]:
        lines.append(f"{fit.label:<12} {fit.residual:10.4f} {fit.constant:12.3e}")
    lines.append(f"best guess: {fits[0].label}")
    return "\n".join(lines)
