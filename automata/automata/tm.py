"""Single-tape deterministic Turing machine."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TMResult:
    accepted: bool
    halted: bool
    steps: int
    final_state: str
    tape: str
    head: int
    trace: list[str] = field(default_factory=list)


@dataclass
class TuringMachine:
    states: set[str]
    input_alphabet: set[str]
    tape_alphabet: set[str]
    blank: str
    start: str
    accept: str
    reject: str
    # (state, read) -> (write, move L|R|S, next_state)
    delta: dict[tuple[str, str], tuple[str, str, str]]

    def __post_init__(self) -> None:
        if self.blank not in self.tape_alphabet:
            raise ValueError("blank must be in tape alphabet")
        if self.input_alphabet - self.tape_alphabet:
            raise ValueError("input alphabet must be subset of tape alphabet")
        for name in (self.start, self.accept, self.reject):
            if name not in self.states:
                raise ValueError(f"state {name!r} not in states")
        for (state, read), (write, move, nxt) in self.delta.items():
            if move not in {"L", "R", "S"}:
                raise ValueError(f"invalid move {move!r}")
            if write not in self.tape_alphabet:
                raise ValueError(f"write symbol {write!r} not in tape alphabet")
            if nxt not in self.states:
                raise ValueError(f"next state {nxt!r} not in states")

    def run(
        self,
        input_word: str,
        *,
        max_steps: int = 10_000,
        keep_trace: bool = False,
    ) -> TMResult:
        for ch in input_word:
            if ch not in self.input_alphabet:
                raise ValueError(f"symbol {ch!r} not in input alphabet")

        tape: dict[int, str] = {i: ch for i, ch in enumerate(input_word)}
        head = 0
        state = self.start
        trace: list[str] = []

        def snapshot() -> str:
            lo = min(tape.keys()) if tape else 0
            hi = max(tape.keys()) if tape else 0
            lo = min(lo, head)
            hi = max(hi, head)
            cells = []
            for i in range(lo, hi + 1):
                sym = tape.get(i, self.blank)
                cells.append(f"[{sym}]" if i == head else f" {sym} ")
            return f"q={state} head={head} tape={''.join(cells)}"

        if keep_trace:
            trace.append(snapshot())

        for step in range(max_steps):
            if state == self.accept:
                return TMResult(True, True, step, state, self._tape_str(tape), head, trace)
            if state == self.reject:
                return TMResult(False, True, step, state, self._tape_str(tape), head, trace)

            read = tape.get(head, self.blank)
            key = (state, read)
            if key not in self.delta:
                # implicit reject if no transition
                return TMResult(False, True, step, state, self._tape_str(tape), head, trace)

            write, move, nxt = self.delta[key]
            tape[head] = write
            if move == "L":
                head -= 1
            elif move == "R":
                head += 1
            state = nxt
            if keep_trace:
                trace.append(snapshot())

        return TMResult(False, False, max_steps, state, self._tape_str(tape), head, trace)

    def _tape_str(self, tape: dict[int, str]) -> str:
        if not tape:
            return self.blank
        lo, hi = min(tape), max(tape)
        return "".join(tape.get(i, self.blank) for i in range(lo, hi + 1)).strip(self.blank) or self.blank

    def accepts(self, word: str, *, max_steps: int = 10_000) -> bool:
        return self.run(word, max_steps=max_steps).accepted
