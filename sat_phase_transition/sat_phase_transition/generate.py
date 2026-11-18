"""Random k-SAT instance generation."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RandomFormula:
    num_vars: int
    clauses: list[list[int]]  # each clause: list of nonzero DIMACS lits
    k: int

    @property
    def num_clauses(self) -> int:
        return len(self.clauses)

    @property
    def alpha(self) -> float:
        return self.num_clauses / self.num_vars if self.num_vars else 0.0

    def to_dimacs(self) -> str:
        lines = [f"c random {self.k}-SAT n={self.num_vars} m={self.num_clauses} alpha={self.alpha:.4f}"]
        lines.append(f"p cnf {self.num_vars} {self.num_clauses}")
        for cl in self.clauses:
            lines.append(" ".join(str(x) for x in cl) + " 0")
        return "\n".join(lines) + "\n"


def random_ksat(
    n: int,
    m: int,
    *,
    k: int = 3,
    seed: int | None = None,
    allow_duplicate_clauses: bool = True,
) -> RandomFormula:
    """
    Generate a uniform random k-SAT formula with n variables and m clauses.

    Each clause: k distinct variables, each negated independently with prob 1/2.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 1 or k > n:
        raise ValueError(f"need 1 <= k <= n (got k={k}, n={n})")
    if m < 0:
        raise ValueError("m must be >= 0")

    rng = random.Random(seed)
    clauses: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    attempts = 0
    max_attempts = max(1000, m * 50)
    while len(clauses) < m and attempts < max_attempts:
        attempts += 1
        vars_ = rng.sample(range(1, n + 1), k)
        lits = [v if rng.random() < 0.5 else -v for v in vars_]
        # canonical key ignores order but keeps polarity of each var
        key = tuple(sorted(lits, key=lambda x: (abs(x), x)))
        # reject tautologies
        if any(-lit in lits for lit in lits):
            continue
        if not allow_duplicate_clauses and key in seen:
            continue
        seen.add(key)
        clauses.append(lits)

    if len(clauses) < m:
        raise RuntimeError(f"could only generate {len(clauses)}/{m} clauses")

    return RandomFormula(num_vars=n, clauses=clauses, k=k)
