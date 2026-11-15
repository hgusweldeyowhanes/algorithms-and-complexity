"""Literal / clause helpers.

Literals are nonzero ints: +v is True, -v is False (DIMACS convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def var_of(lit: int) -> int:
    return abs(lit)


def neg(lit: int) -> int:
    return -lit


@dataclass
class Clause:
    """A CNF clause (disjunction of literals)."""

    lits: list[int]
    learnt: bool = False
    # Activity for clause deletion / VSIDS bumping of involved vars
    activity: float = 0.0

    def __post_init__(self) -> None:
        # Deduplicate while preserving order; drop tautologies later at attach time
        seen: set[int] = set()
        out: list[int] = []
        for lit in self.lits:
            if lit == 0:
                continue
            if lit in seen:
                continue
            if neg(lit) in seen:
                # tautology — mark empty after clear
                self.lits = []
                self._tautology = True
                return
            seen.add(lit)
            out.append(lit)
        self.lits = out
        self._tautology = False

    @property
    def is_tautology(self) -> bool:
        return bool(getattr(self, "_tautology", False))

    def __len__(self) -> int:
        return len(self.lits)

    def __repr__(self) -> str:
        return f"Clause({self.lits!r}, learnt={self.learnt})"


@dataclass
class Formula:
    num_vars: int
    clauses: list[Clause] = field(default_factory=list)
