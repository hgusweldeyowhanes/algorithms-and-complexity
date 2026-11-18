"""3-SAT formulas and brute-force solving (small instances)."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import re


@dataclass(frozen=True)
class Literal:
    var: str
    positive: bool

    def __str__(self) -> str:
        return self.var if self.positive else f"~{self.var}"

    def negate(self) -> Literal:
        return Literal(self.var, not self.positive)

    def eval(self, assignment: dict[str, bool]) -> bool:
        val = assignment[self.var]
        return val if self.positive else not val


@dataclass
class Formula3SAT:
    clauses: list[tuple[Literal, Literal, Literal]]

    @property
    def variables(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for cl in self.clauses:
            for lit in cl:
                if lit.var not in seen:
                    seen.add(lit.var)
                    out.append(lit.var)
        return out

    def is_satisfied(self, assignment: dict[str, bool]) -> bool:
        for cl in self.clauses:
            if not any(lit.eval(assignment) for lit in cl):
                return False
        return True

    def solve(self) -> dict[str, bool] | None:
        """Brute-force; OK for teaching-size formulas."""
        vars_ = self.variables
        for bits in itertools.product([False, True], repeat=len(vars_)):
            asn = dict(zip(vars_, bits))
            if self.is_satisfied(asn):
                return asn
        return None

    def __str__(self) -> str:
        parts = []
        for cl in self.clauses:
            parts.append("(" + " ∨ ".join(str(l) for l in cl) + ")")
        return " ∧ ".join(parts)


_LIT_RE = re.compile(r"~?([A-Za-z_]\w*)")


def _parse_lit(tok: str) -> Literal:
    tok = tok.strip()
    if tok.startswith("~") or tok.startswith("!"):
        return Literal(tok[1:].strip(), False)
    if tok.startswith("-") and len(tok) > 1 and tok[1].isalpha():
        return Literal(tok[1:].strip(), False)
    m = _LIT_RE.fullmatch(tok)
    if not m:
        raise ValueError(f"bad literal: {tok!r}")
    return Literal(m.group(1), True)


def parse_formula(text: str) -> Formula3SAT:
    """
    Parse a simple 3-SAT text form:

        (x ∨ ~y ∨ z) ∧ (~x ∨ y ∨ w)

    Also accepts 'v' or '|' for OR, and '^' or '&' for AND.
    """
    cleaned = text.strip()
    cleaned = cleaned.replace("|", "∨")
    cleaned = cleaned.replace("^", "∧").replace("&", "∧")
    # allow spaced ' v ' as OR without eating variable names
    cleaned = re.sub(r"\s+v\s+", " ∨ ", cleaned, flags=re.I)
    # split clauses on ∧
    raw_clauses = re.split(r"\s*∧\s*", cleaned)
    clauses: list[tuple[Literal, Literal, Literal]] = []
    for raw in raw_clauses:
        if not raw.strip():
            continue
        body = raw.strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        lits = re.split(r"\s*∨\s*", body)
        if len(lits) != 3:
            raise ValueError(f"each clause must have exactly 3 literals, got {lits!r}")
        a, b, c = (_parse_lit(t) for t in lits)
        clauses.append((a, b, c))
    if not clauses:
        raise ValueError("empty formula")
    return Formula3SAT(clauses=clauses)
