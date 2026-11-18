"""Sweep clause/variable ratio α and estimate P(SAT)."""
from __future__ import annotations

from dataclasses import dataclass

from sat_phase_transition.generate import random_ksat
from sat_phase_transition.solver_bridge import solve_formula


@dataclass
class SweepPoint:
    alpha: float
    n: int
    m: int
    trials: int
    sat_count: int
    mean_time: float

    @property
    def sat_probability(self) -> float:
        return self.sat_count / self.trials if self.trials else 0.0


def run_sweep(
    *,
    n: int = 20,
    k: int = 3,
    alpha_min: float = 3.0,
    alpha_max: float = 5.5,
    steps: int = 11,
    trials: int = 20,
    seed: int = 1,
) -> list[SweepPoint]:
    """
    For each α in [alpha_min, alpha_max], generate `trials` random k-SAT
    instances with m = round(α * n) and measure fraction satisfiable.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    points: list[SweepPoint] = []
    for i in range(steps):
        t = i / (steps - 1)
        alpha = alpha_min + t * (alpha_max - alpha_min)
        m = max(0, int(round(alpha * n)))
        sat_count = 0
        total_time = 0.0
        for t_i in range(trials):
            formula = random_ksat(n, m, k=k, seed=seed + 10007 * i + t_i)
            is_sat, elapsed = solve_formula(formula)
            if is_sat:
                sat_count += 1
            total_time += elapsed
        points.append(
            SweepPoint(
                alpha=m / n if n else alpha,
                n=n,
                m=m,
                trials=trials,
                sat_count=sat_count,
                mean_time=total_time / trials,
            )
        )
    return points


def format_table(points: list[SweepPoint]) -> str:
    lines = [
        f"{'alpha':>8} {'m':>5} {'P(SAT)':>8} {'sat/trials':>12} {'mean_s':>10}",
        "-" * 48,
    ]
    for p in points:
        lines.append(
            f"{p.alpha:8.3f} {p.m:5d} {p.sat_probability:8.3f} "
            f"{p.sat_count:4d}/{p.trials:<5d} {p.mean_time:10.4f}"
        )
    return "\n".join(lines)


def format_csv(points: list[SweepPoint]) -> str:
    rows = ["alpha,n,m,trials,sat_count,sat_probability,mean_time_s"]
    for p in points:
        rows.append(
            f"{p.alpha:.6f},{p.n},{p.m},{p.trials},{p.sat_count},"
            f"{p.sat_probability:.6f},{p.mean_time:.6f}"
        )
    return "\n".join(rows) + "\n"
