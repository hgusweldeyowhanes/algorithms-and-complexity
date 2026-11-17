"""CLI: approximation algorithms study (VC 2-approx + greedy Set Cover)."""
from __future__ import annotations

import argparse
import sys

from approx_algorithms.study import format_study, study_set_cover, study_vertex_cover


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Study Vertex Cover 2-approx and greedy Set Cover vs OPT"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="both",
        choices=["vertex-cover", "vc", "set-cover", "sc", "both"],
        help="which study to run (default both)",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    args = parser.parse_args(argv)

    topic = args.topic
    run_vc = topic in {"vertex-cover", "vc", "both"}
    run_sc = topic in {"set-cover", "sc", "both"}

    if run_vc:
        rows = study_vertex_cover(seed=args.seed)
        print(format_study("Vertex Cover — maximal-matching 2-approximation vs OPT", rows))
        print()
        print("Notes:")
        print("  • Guarantee: |C| ≤ 2 · OPT")
        print("  • Disjoint edges / star instances approach ratio 2 (tight examples)")
        print()

    if run_sc:
        rows = study_set_cover(seed=args.seed)
        print(format_study("Set Cover — greedy vs OPT", rows))
        print()
        print("Notes:")
        print("  • Guarantee: |C| ≤ H_|U| · OPT  (H_n ≈ ln n + 1)")
        print("  • 'bound' column is H_|U| for that instance")
        print("  • Adversarial row searches for a small instance with greedy > OPT")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
