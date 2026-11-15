"""CLI: benchmark algorithms and guess complexity class."""
from __future__ import annotations

import argparse
import sys

from algo_benchmarker.algorithms import ALGORITHMS, default_sizes
from algo_benchmarker.benchmark import format_table, run_benchmark
from algo_benchmarker.fit import fit_complexity, format_fits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time algorithms vs input size and compare to complexity classes"
    )
    parser.add_argument(
        "algorithm",
        nargs="?",
        default="",
        help=f"one of: {', '.join(ALGORITHMS)} (default: all)",
    )
    parser.add_argument("--runs", type=int, default=3, help="timed runs per size")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs per size")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="comma-separated sizes, e.g. 100,200,400 (overrides defaults)",
    )
    parser.add_argument("--list", action="store_true", help="list algorithms and exit")
    args = parser.parse_args(argv)

    if args.list or args.algorithm == "list":
        for name, (_, expected) in ALGORITHMS.items():
            print(f"{name:16} expected {expected}")
        return 0

    names = list(ALGORITHMS) if not args.algorithm else [args.algorithm]
    for name in names:
        if name not in ALGORITHMS:
            print(f"error: unknown algorithm {name!r}", file=sys.stderr)
            print(f"choose from: {', '.join(ALGORITHMS)}", file=sys.stderr)
            return 1

        fn, expected = ALGORITHMS[name]
        if args.sizes.strip():
            sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
        else:
            sizes = default_sizes(name)

        result = run_benchmark(
            name,
            fn,
            sizes,
            expected=expected,
            runs=args.runs,
            warmup=args.warmup,
            seed=args.seed,
        )
        print(format_table(result))
        print()
        fits = fit_complexity(result)
        print(format_fits(fits))
        if result.expected and fits:
            match = fits[0].label == expected
            print(f"match expected {expected}? {'yes' if match else 'no (noisy / too few sizes)'}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
