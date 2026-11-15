"""CLI: python -m sat_solver file.cnf [--seed N]"""
from __future__ import annotations

import argparse
import sys

from sat_solver.dimacs import DimacsError, parse_dimacs_file
from sat_solver.solver import Solver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CDCL SAT solver (DIMACS CNF)")
    parser.add_argument("cnf", help="Path to DIMACS .cnf file")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for VSIDS tie-breaks")
    args = parser.parse_args(argv)

    try:
        formula = parse_dimacs_file(args.cnf)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except DimacsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    solver = Solver(formula, seed=args.seed)
    result = solver.solve()

    if result.sat:
        print("s SATISFIABLE")
        lits = result.assignment_lits(formula.num_vars)
        # wrap v-line reasonably
        line = "v " + " ".join(str(x) for x in lits) + " 0"
        print(line)
        return 10

    print("s UNSATISFIABLE")
    return 20


if __name__ == "__main__":
    raise SystemExit(main())
