"""Deterministic Finite Automaton."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DFA:
    states: set[str]
    alphabet: set[str]
    start: str
    accept: set[str]
    # (state, symbol) -> state
    delta: dict[tuple[str, str], str]

    def __post_init__(self) -> None:
        if self.start not in self.states:
            raise ValueError(f"start state {self.start!r} not in states")
        bad = self.accept - self.states
        if bad:
            raise ValueError(f"accept states not in states: {bad}")

    def step(self, state: str, symbol: str) -> str:
        key = (state, symbol)
        if key not in self.delta:
            raise ValueError(f"no transition from {state!r} on {symbol!r}")
        return self.delta[key]

    def accepts(self, word: str) -> bool:
        state = self.start
        for ch in word:
            if ch not in self.alphabet:
                raise ValueError(f"symbol {ch!r} not in alphabet")
            state = self.step(state, ch)
        return state in self.accept

    def run(self, word: str) -> tuple[bool, list[str]]:
        """Return (accepted, state trace including start)."""
        trace = [self.start]
        state = self.start
        for ch in word:
            if ch not in self.alphabet:
                raise ValueError(f"symbol {ch!r} not in alphabet")
            state = self.step(state, ch)
            trace.append(state)
        return state in self.accept, trace
