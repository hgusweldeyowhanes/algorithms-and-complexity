"""Solve formulas using sibling sat_solver CDCL, with a tiny DPLL fallback."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from sat_phase_transition.generate import RandomFormula


def _ensure_sat_solver_path() -> None:
    here = Path(__file__).resolve()
    # .../algorithms-and-complexity/sat_phase_transition/sat_phase_transition/solver_bridge.py
    algo_root = here.parents[2]
    sat_root = algo_root / "sat_solver"
    if sat_root.is_dir():
        p = str(sat_root)
        if p not in sys.path:
            sys.path.insert(0, p)


def solve_formula(formula: RandomFormula) -> tuple[bool, float]:
    """
    Return (is_sat, elapsed_seconds).
    Prefers CDCL from ../sat_solver; falls back to recursive DPLL.
    """
    _ensure_sat_solver_path()
    t0 = time.perf_counter()
    try:
        from sat_solver.solver import Solver
        from sat_solver.types import Clause, Formula
    except ImportError:
        sat = _dpll(formula.num_vars, [list(c) for c in formula.clauses], {})
        return sat, time.perf_counter() - t0

    f = Formula(
        num_vars=formula.num_vars,
        clauses=[Clause(list(cl)) for cl in formula.clauses],
    )
    result = Solver(f, seed=1).solve()
    return result.sat, time.perf_counter() - t0


def _dpll(n: int, clauses: list[list[int]], assign: dict[int, bool]) -> bool:
    """Simple DPLL with unit propagation (fallback)."""
    clauses = _simplify(clauses, assign)
    if clauses is None:
        return False
    if not clauses:
        return True

    # unit propagate
    changed = True
    while changed:
        changed = False
        units = [c[0] for c in clauses if len(c) == 1]
        if not units:
            break
        for lit in units:
            v = abs(lit)
            val = lit > 0
            if v in assign and assign[v] != val:
                return False
            if v not in assign:
                assign = dict(assign)
                assign[v] = val
                changed = True
        clauses = _simplify(clauses, assign)
        if clauses is None:
            return False
        if not clauses:
            return True

    # pick unassigned var
    used = set()
    for c in clauses:
        for lit in c:
            used.add(abs(lit))
    var = None
    for v in range(1, n + 1):
        if v not in assign and v in used:
            var = v
            break
    if var is None:
        return True

    for val in (True, False):
        nxt = dict(assign)
        nxt[var] = val
        if _dpll(n, [list(c) for c in clauses], nxt):
            return True
    return False


def _simplify(clauses: list[list[int]], assign: dict[int, bool]) -> list[list[int]] | None:
    out: list[list[int]] = []
    for cl in clauses:
        new_cl: list[int] = []
        sat = False
        for lit in cl:
            v = abs(lit)
            if v in assign:
                if (lit > 0) == assign[v]:
                    sat = True
                    break
                continue
            new_cl.append(lit)
        if sat:
            continue
        if not new_cl:
            return None
        out.append(new_cl)
    return out
