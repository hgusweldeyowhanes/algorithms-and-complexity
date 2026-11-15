"""DIMACS CNF parser."""
from __future__ import annotations

from pathlib import Path

from sat_solver.types import Clause, Formula


class DimacsError(ValueError):
    """Malformed DIMACS input."""


def parse_dimacs(text: str) -> Formula:
    """Parse DIMACS CNF from a string."""
    num_vars: int | None = None
    expected_clauses: int | None = None
    clauses: list[Clause] = []
    current: list[int] = []
    saw_header = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("c") or line.startswith("%"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1].lower() != "cnf":
                raise DimacsError(f"line {lineno}: expected 'p cnf <vars> <clauses>'")
            try:
                num_vars = int(parts[2])
                expected_clauses = int(parts[3])
            except ValueError as exc:
                raise DimacsError(f"line {lineno}: invalid header numbers") from exc
            if num_vars < 0 or expected_clauses < 0:
                raise DimacsError(f"line {lineno}: negative counts in header")
            saw_header = True
            continue

        if not saw_header:
            raise DimacsError(f"line {lineno}: clause before 'p cnf' header")

        try:
            tokens = [int(tok) for tok in line.split()]
        except ValueError as exc:
            raise DimacsError(f"line {lineno}: non-integer token") from exc

        for tok in tokens:
            if tok == 0:
                if not current:
                    raise DimacsError(f"line {lineno}: empty clause")
                for lit in current:
                    if abs(lit) > (num_vars or 0):
                        raise DimacsError(
                            f"line {lineno}: literal {lit} exceeds declared vars={num_vars}"
                        )
                clauses.append(Clause(list(current)))
                current = []
            else:
                current.append(tok)

    if current:
        raise DimacsError("unterminated clause (missing trailing 0)")

    if not saw_header or num_vars is None:
        raise DimacsError("missing 'p cnf' header")

    # Header clause count is advisory; warn via mismatch only if wildly off is not fatal
    _ = expected_clauses
    return Formula(num_vars=num_vars, clauses=clauses)


def parse_dimacs_file(path: str | Path) -> Formula:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CNF file not found: {p}")
    return parse_dimacs(p.read_text(encoding="utf-8", errors="replace"))
