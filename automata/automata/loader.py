"""Load machine definitions from simple JSON files."""
from __future__ import annotations

import json
from pathlib import Path

from automata.dfa import DFA
from automata.nfa import EPS, NFA
from automata.tm import TuringMachine


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dfa(path: str | Path) -> DFA:
    data = load_json(path)
    delta: dict[tuple[str, str], str] = {}
    for row in data["transitions"]:
        delta[(row["from"], row["symbol"])] = row["to"]
    return DFA(
        states=set(data["states"]),
        alphabet=set(data["alphabet"]),
        start=data["start"],
        accept=set(data["accept"]),
        delta=delta,
    )


def load_nfa(path: str | Path) -> NFA:
    data = load_json(path)
    delta: dict[tuple[str, str], set[str]] = {}
    for row in data["transitions"]:
        sym = EPS if row.get("symbol") in (None, "", "eps", "ε", "epsilon") else row["symbol"]
        key = (row["from"], sym)
        delta.setdefault(key, set()).add(row["to"])
    return NFA(
        states=set(data["states"]),
        alphabet=set(data["alphabet"]),
        start=data["start"],
        accept=set(data["accept"]),
        delta=delta,
    )


def load_tm(path: str | Path) -> TuringMachine:
    data = load_json(path)
    delta: dict[tuple[str, str], tuple[str, str, str]] = {}
    for row in data["transitions"]:
        delta[(row["state"], row["read"])] = (row["write"], row["move"], row["next"])
    return TuringMachine(
        states=set(data["states"]),
        input_alphabet=set(data["input_alphabet"]),
        tape_alphabet=set(data["tape_alphabet"]),
        blank=data.get("blank", "_"),
        start=data["start"],
        accept=data["accept"],
        reject=data["reject"],
        delta=delta,
    )
