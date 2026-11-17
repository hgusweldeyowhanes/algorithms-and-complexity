"""CLI: run DFA / NFA / Turing machine on an input string."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from automata.loader import load_dfa, load_nfa, load_tm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate DFA, NFA, or Turing machine")
    parser.add_argument("machine", help="Path to machine JSON")
    parser.add_argument("input", nargs="?", default="", help="Input string (default empty)")
    parser.add_argument("--trace", action="store_true", help="Print execution trace")
    parser.add_argument("--max-steps", type=int, default=10_000, help="TM step limit")
    args = parser.parse_args(argv)

    path = Path(args.machine)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 1

    mtype = (meta.get("type") or "").lower()
    try:
        if mtype == "dfa":
            machine = load_dfa(path)
            if args.trace:
                ok, trace = machine.run(args.input)
                print("trace:", " -> ".join(trace))
            else:
                ok = machine.accepts(args.input)
            print("ACCEPT" if ok else "REJECT")
            return 0 if ok else 2

        if mtype == "nfa":
            machine = load_nfa(path)
            if args.trace:
                ok, trace = machine.run(args.input)
                for i, states in enumerate(trace):
                    label = "start" if i == 0 else f"after {args.input[i-1]!r}"
                    print(f"{label}: {{{', '.join(sorted(states))}}}")
            else:
                ok = machine.accepts(args.input)
            print("ACCEPT" if ok else "REJECT")
            return 0 if ok else 2

        if mtype == "tm":
            machine = load_tm(path)
            result = machine.run(
                args.input,
                max_steps=args.max_steps,
                keep_trace=args.trace,
            )
            if args.trace:
                for line in result.trace:
                    print(line)
            status = (
                "ACCEPT"
                if result.accepted
                else ("REJECT" if result.halted else "TIMEOUT")
            )
            print(status)
            print(f"steps={result.steps} state={result.final_state} tape={result.tape!r}")
            if result.accepted:
                return 0
            return 3 if not result.halted else 2

        print(f"error: unknown machine type {mtype!r} (use dfa|nfa|tm)", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
