"""CLI: collect resource samples, train RF/XGBoost/NN, report accuracy."""
from __future__ import annotations

import argparse
import sys

from resource_predictor.algorithms import ALGORITHMS
from resource_predictor.train_eval import format_sample_table, format_study, run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict algorithm CPU time and memory with RF, XGBoost, Neural Net"
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        help="limit dataset to these algorithms (repeatable)",
    )
    parser.add_argument("--runs", type=int, default=2, help="timing repeats per (algo, n)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="print an excerpt of measured samples",
    )
    parser.add_argument("--list", action="store_true", help="list algorithms and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in ALGORITHMS:
            print(name)
        return 0

    if args.algorithms:
        unknown = [a for a in args.algorithms if a not in ALGORITHMS]
        if unknown:
            print(f"unknown algorithms: {unknown}", file=sys.stderr)
            return 1

    print("Collecting time/memory samples and training models...")
    study = run_study(
        runs=args.runs,
        seed=args.seed,
        test_size=args.test_size,
        algorithms=args.algorithms,
    )
    if args.show_samples:
        print(format_sample_table(study.samples))
        print()
    print(format_study(study))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
