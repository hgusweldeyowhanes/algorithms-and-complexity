"""CLI: compare exact vs approximate vs heuristic TSP."""
from __future__ import annotations

import argparse
import sys
import time

from tsp.approximate import mst_double_approx
from tsp.exact import held_karp
from tsp.heuristics import nearest_neighbor, nearest_neighbor_two_opt
from tsp.instance import load_instance


def _fmt_tour(instance, tour: list[int]) -> str:
    return " -> ".join(instance.cities[i].name for i in tour) + f" -> {instance.cities[tour[0]].name}"


def _print_result(label: str, result, instance, ms: float) -> None:
    print(f"{label} ({result.method})")
    print(f"  length: {result.length:.6f}")
    print(f"  time:   {ms:.3f} ms")
    print(f"  tour:   {_fmt_tour(instance, result.tour)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TSP: exact (Held–Karp) vs approximate (MST 2-approx) vs heuristic (NN+2-opt)"
    )
    parser.add_argument("instance", help="Path to cities file (name x y per line)")
    parser.add_argument("--start", type=int, default=0, help="Start city index (default 0)")
    parser.add_argument(
        "--no-exact",
        action="store_true",
        help="Skip Held–Karp (useful for larger n)",
    )
    parser.add_argument(
        "--no-approx",
        action="store_true",
        help="Skip MST double-tree 2-approximation",
    )
    parser.add_argument(
        "--no-heuristic",
        action="store_true",
        help="Skip nearest-neighbor + 2-opt",
    )
    parser.add_argument(
        "--no-2opt",
        action="store_true",
        help="Heuristic: nearest neighbor only (no 2-opt)",
    )
    args = parser.parse_args(argv)

    try:
        instance = load_instance(args.instance)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"cities: {instance.n}  (Euclidean / metric TSP)")
    print()

    approx = None
    heur = None
    exact = None

    if not args.no_approx:
        t0 = time.perf_counter()
        approx = mst_double_approx(instance, start=args.start)
        _print_result("approximate", approx, instance, (time.perf_counter() - t0) * 1000)

    if not args.no_heuristic:
        t0 = time.perf_counter()
        if args.no_2opt:
            heur = nearest_neighbor(instance, start=args.start)
        else:
            heur = nearest_neighbor_two_opt(instance, start=args.start)
        _print_result("heuristic", heur, instance, (time.perf_counter() - t0) * 1000)

    if not args.no_exact:
        if instance.n > 20:
            print("exact: skipped (n > 20). Re-run with --no-exact or shrink the instance.")
        else:
            t0 = time.perf_counter()
            try:
                exact = held_karp(instance, start=args.start)
            except ValueError as exc:
                print(f"exact: {exc}", file=sys.stderr)
                return 1
            _print_result("exact", exact, instance, (time.perf_counter() - t0) * 1000)

    if exact is not None and exact.length > 0:
        print("gaps vs exact OPT")
        if approx is not None:
            gap = approx.length - exact.length
            pct = 100.0 * gap / exact.length
            ratio = approx.length / exact.length
            print(f"  approx:    +{gap:.6f} ({pct:.2f}%)  ratio={ratio:.4f}  (guarantee ≤ 2)")
        if heur is not None:
            gap = heur.length - exact.length
            pct = 100.0 * gap / exact.length
            ratio = heur.length / exact.length
            print(f"  heuristic: +{gap:.6f} ({pct:.2f}%)  ratio={ratio:.4f}  (no worst-case guarantee)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
