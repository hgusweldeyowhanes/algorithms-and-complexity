"""CLI: random k-SAT phase transition sweep."""
from __future__ import annotations

import argparse
import sys

from sat_phase_transition.experiment import format_csv, format_table, run_sweep
from sat_phase_transition.generate import random_ksat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate the random k-SAT phase transition (P(SAT) vs alpha=m/n)"
    )
    parser.add_argument("--n", type=int, default=20, help="number of variables (default 20)")
    parser.add_argument("--k", type=int, default=3, help="clause width (default 3)")
    parser.add_argument("--alpha-min", type=float, default=3.0, help="min alpha m/n")
    parser.add_argument("--alpha-max", type=float, default=5.5, help="max alpha m/n")
    parser.add_argument("--steps", type=int, default=11, help="number of alpha samples")
    parser.add_argument("--trials", type=int, default=20, help="instances per alpha")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed base")
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="optional path to write CSV results",
    )
    parser.add_argument(
        "--demo-formula",
        action="store_true",
        help="print one sample random 3-SAT DIMACS and exit",
    )
    args = parser.parse_args(argv)

    if args.demo_formula:
        m = int(round(4.2 * args.n))
        f = random_ksat(args.n, m, k=args.k, seed=args.seed)
        sys.stdout.write(f.to_dimacs())
        return 0

    print(
        f"random {args.k}-SAT phase transition  n={args.n}  "
        f"alpha=[{args.alpha_min}, {args.alpha_max}]  "
        f"steps={args.steps}  trials={args.trials}"
    )
    print("(3-SAT threshold is near alpha ≈ 4.26 for large n)\n")

    try:
        points = run_sweep(
            n=args.n,
            k=args.k,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            steps=args.steps,
            trials=args.trials,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_table(points))

    if args.csv:
        path = args.csv
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(format_csv(points))
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
