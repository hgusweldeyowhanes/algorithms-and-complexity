"""Nondeterministic Finite Automaton (with epsilon transitions)."""
from __future__ import annotations

from dataclasses import dataclass


EPS = ""  # epsilon marker in transition keys


@dataclass
class NFA:
    states: set[str]
    alphabet: set[str]  # should not include epsilon
    start: str
    accept: set[str]
    # (state, symbol_or_eps) -> set of states
    delta: dict[tuple[str, str], set[str]]

    def __post_init__(self) -> None:
        if self.start not in self.states:
            raise ValueError(f"start state {self.start!r} not in states")
        bad = self.accept - self.states
        if bad:
            raise ValueError(f"accept states not in states: {bad}")

    def _epsilon_closure(self, states: set[str]) -> set[str]:
        stack = list(states)
        closure = set(states)
        while stack:
            s = stack.pop()
            for nxt in self.delta.get((s, EPS), set()):
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
        return closure

    def _move(self, states: set[str], symbol: str) -> set[str]:
        out: set[str] = set()
        for s in states:
            out |= self.delta.get((s, symbol), set())
        return out

    def accepts(self, word: str) -> bool:
        current = self._epsilon_closure({self.start})
        for ch in word:
            if ch not in self.alphabet:
                raise ValueError(f"symbol {ch!r} not in alphabet")
            current = self._epsilon_closure(self._move(current, ch))
            if not current:
                return False
        return bool(current & self.accept)

    def run(self, word: str) -> tuple[bool, list[set[str]]]:
        """Return (accepted, list of state-sets after each symbol, starting with ε-closure)."""
        trace: list[set[str]] = []
        current = self._epsilon_closure({self.start})
        trace.append(set(current))
        for ch in word:
            if ch not in self.alphabet:
                raise ValueError(f"symbol {ch!r} not in alphabet")
            current = self._epsilon_closure(self._move(current, ch))
            trace.append(set(current))
            if not current:
                return False, trace
        return bool(current & self.accept), trace
