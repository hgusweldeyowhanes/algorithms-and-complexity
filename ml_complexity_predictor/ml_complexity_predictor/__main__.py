"""CLI: train ML complexity predictor and compare theoretical vs predicted."""
from __future__ import annotations

import argparse
import sys

from ml_complexity_predictor.algorithms import ALGORITHMS
from ml_complexity_predictor.predict import (
    format_report,
    format_summary,
    predict_algorithm,
    study_all,
    train_default_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ML-based Big-O predictor from execution-time curves"
    )
    parser.add_argument(
        "algorithm",
        nargs="?",
        default=None,
        help="algorithm to measure (default: study all). "
        f"choices: {', '.join(sorted(ALGORITHMS))}",
    )
    parser.add_argument(
        "--category",
        choices=["sorting", "searching", "graph", "other"],
        action="append",
        default=None,
        help="filter study by category (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2, help="timing repeats per size")
    parser.add_argument(
        "--per-class",
        type=int,
        default=80,
        help="synthetic training curves per Big-O class",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip real-algorithm samples in training (synthetic only)",
    )
    parser.add_argument(
        "--show-curve",
        action="store_true",
        help="print timing table for single-algorithm mode",
    )
    parser.add_argument("--list", action="store_true", help="list algorithms and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, (_fn, theory, cat) in sorted(ALGORITHMS.items()):
            print(f"{name:<24} {cat:<10} theory={theory}")
        return 0

    print("Training softmax classifier on synthetic (+ optional real) curves...")
    model, metrics = train_default_model(
        per_class=args.per_class,
        include_real=not args.fast,
        seed=args.seed,
        epochs=400 if args.fast else 500,
    )
    print(
        f"  train accuracy: {metrics['train_accuracy']:.1%} "
        f"(n={int(metrics['n_train'])})"
    )
    print(
        f"  test  accuracy: {metrics['test_accuracy']:.1%} "
        f"(n={int(metrics['n_test'])})"
    )
    print()

    if args.algorithm:
        if args.algorithm not in ALGORITHMS:
            print(f"unknown algorithm: {args.algorithm}", file=sys.stderr)
            print("use --list to see options", file=sys.stderr)
            return 1
        report = predict_algorithm(
            model, args.algorithm, runs=args.runs, seed=args.seed
        )
        print(format_report(report, show_curve=args.show_curve))
        return 0

    cats = set(args.category) if args.category else None
    reports = study_all(model, runs=args.runs, seed=args.seed, categories=cats)
    print(format_summary(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
