from __future__ import annotations

from pathlib import Path

from sat_solver.dimacs import parse_dimacs, parse_dimacs_file
from sat_solver.solver import Solver, luby
from sat_solver.types import Formula, Clause

ROOT = Path(__file__).resolve().parents[1]


def _satisfies(clauses: list[list[int]], model: dict[int, bool]) -> bool:
    for lits in clauses:
        ok = False
        for lit in lits:
            v = abs(lit)
            val = model.get(v)
            if val is None:
                continue
            if (lit > 0 and val) or (lit < 0 and not val):
                ok = True
                break
        if not ok:
            return False
    return True


def test_luby_prefix():
    assert [luby(i) for i in range(1, 8)] == [1, 1, 2, 1, 1, 2, 4]


def test_unsat_contradiction():
    f = parse_dimacs("p cnf 1 2\n1 0\n-1 0\n")
    r = Solver(f).solve()
    assert r.sat is False


def test_sat_unit_chain():
    # x1, ~x1 \/ x2, ~x2 \/ x3  => all true
    f = parse_dimacs("p cnf 3 3\n1 0\n-1 2 0\n-2 3 0\n")
    r = Solver(f).solve()
    assert r.sat is True
    assert r.model is not None
    assert _satisfies([[1], [-1, 2], [-2, 3]], r.model)


def test_sat_simple_3sat():
    f = parse_dimacs(
        """
p cnf 3 4
1 2 3 0
-1 -2 0
-1 -3 0
-2 -3 0
"""
    )
    r = Solver(f).solve()
    assert r.sat is True
    assert r.model is not None
    clauses = [[1, 2, 3], [-1, -2], [-1, -3], [-2, -3]]
    assert _satisfies(clauses, r.model)


def test_examples_sat_tiny():
    f = parse_dimacs_file(ROOT / "examples" / "sat_tiny.cnf")
    r = Solver(f, seed=1).solve()
    assert r.sat is True
    assert r.model is not None
    assert _satisfies([c.lits for c in f.clauses], r.model)


def test_examples_unsat_tiny():
    f = parse_dimacs_file(ROOT / "examples" / "unsat_tiny.cnf")
    r = Solver(f).solve()
    assert r.sat is False


def test_randomish_smoke():
    # Small crafted formula: (a\/b) /\ (~a\/c) /\ (~b\/~c) /\ (a\/~c) — SAT
    f = Formula(
        num_vars=3,
        clauses=[
            Clause([1, 2]),
            Clause([-1, 3]),
            Clause([-2, -3]),
            Clause([1, -3]),
        ],
    )
    r = Solver(f, seed=7).solve()
    assert r.sat is True
    assert r.model is not None
    assert _satisfies([c.lits for c in f.clauses], r.model)


def test_assignment_lits_cover_all_vars():
    f = parse_dimacs("p cnf 2 1\n1 2 0\n")
    r = Solver(f).solve()
    assert r.sat
    lits = r.assignment_lits(2)
    assert sorted(abs(x) for x in lits) == [1, 2]
