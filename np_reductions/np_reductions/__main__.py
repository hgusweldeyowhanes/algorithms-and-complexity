"""CLI: visualize NP reductions and verify bidirectional solution maps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from np_reductions.sat import parse_formula
from np_reductions.verify import verify_clique_reduction, verify_vertex_cover_reduction


DEFAULT_SAT = "(x ∨ ~y ∨ z) ∧ (~x ∨ y ∨ z) ∧ (x ∨ y ∨ ~z)"
DEFAULT_UNSAT = "(x ∨ x ∨ x) ∧ (~x ∨ ~x ∨ ~x)"


def _load_formula(args: argparse.Namespace):
    if args.formula:
        return parse_formula(args.formula)
    if args.file:
        return parse_formula(Path(args.file).read_text(encoding="utf-8"))
    if args.unsat_demo:
        return parse_formula(DEFAULT_UNSAT)
    return parse_formula(DEFAULT_SAT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NP-completeness reduction visualizer (3-SAT → Clique / Vertex Cover)"
    )
    parser.add_argument(
        "reduction",
        nargs="?",
        default="both",
        choices=["clique", "vertex-cover", "vc", "both"],
        help="which reduction to run (default both)",
    )
    parser.add_argument("--formula", type=str, default="", help="3-SAT formula text")
    parser.add_argument("--file", type=str, default="", help="path to formula text file")
    parser.add_argument(
        "--unsat-demo",
        action="store_true",
        help="use a tiny unsatisfiable demo formula",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="print the constructed graph (edges may be truncated)",
    )
    args = parser.parse_args(argv)

    try:
        formula = _load_formula(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Formula φ (3-SAT)")
    print(f"  {formula}")
    asn = formula.solve()
    print(f"  satisfiable? {'YES' if asn else 'NO'}")
    if asn:
        pretty = ", ".join(f"{v}={'T' if asn[v] else 'F'}" for v in formula.variables)
        print(f"  witness: {pretty}")
    print()

    targets = []
    red = args.reduction
    if red in {"clique", "both"}:
        targets.append("clique")
    if red in {"vertex-cover", "vc", "both"}:
        targets.append("vertex-cover")

    all_ok = True
    for kind in targets:
        if kind == "clique":
            print("=" * 60)
            print("Reduction: 3-SAT  ≤p  Clique")
            print("  vertices: one per literal occurrence in each clause")
            print("  edges: between consistent literals from different clauses")
            print("  ask: clique of size k = #clauses")
            print()
            red_obj, check = verify_clique_reduction(formula)
            print(f"  |V|={len(red_obj.graph.vertices)}  |E|={len(red_obj.graph.edges)}  k={red_obj.k}")
            if args.graph:
                print(red_obj.graph.ascii())
            print("Verification")
            for msg in check.messages:
                print(f"  • {msg}")
            print(f"  result: {'PASS' if check.ok else 'FAIL'}")
            print()
            all_ok &= check.ok
        else:
            print("=" * 60)
            print("Reduction: 3-SAT  ≤p  Vertex Cover")
            print("  gadgets: {x,~x} edges + clause triangles linked to literals")
            print("  ask: vertex cover of size k = n + 2m")
            print()
            red_obj, check = verify_vertex_cover_reduction(formula)
            print(f"  |V|={len(red_obj.graph.vertices)}  |E|={len(red_obj.graph.edges)}  k={red_obj.k}")
            if args.graph:
                print(red_obj.graph.ascii())
            print("Verification")
            for msg in check.messages:
                print(f"  • {msg}")
            print(f"  result: {'PASS' if check.ok else 'FAIL'}")
            print()
            all_ok &= check.ok

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
