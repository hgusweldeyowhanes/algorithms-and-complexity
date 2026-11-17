# Turing Machine / Automata Simulator

Simulate classic models of computation:

| Model | Class | Notes |
|-------|--------|--------|
| **DFA** | Regular languages | Deterministic finite automaton |
| **NFA** | Regular languages | Nondeterminism + ε-transitions |
| **TM** | Recursively enumerable | Single-tape deterministic Turing machine |

Machines are defined in JSON; the CLI runs a string and prints `ACCEPT` / `REJECT` (or `TIMEOUT` for TMs).

## Usage

```bash
cd automata

# DFA: even number of 0s
python -m automata examples/dfa_even_zeros.json 0100
python -m automata examples/dfa_even_zeros.json 00 --trace

# NFA: ends with ab
python -m automata examples/nfa_ends_with_ab.json babab

# TM: a^n b^n
python -m automata examples/tm_anbn.json aabb --trace
python -m automata examples/tm_anbn.json aaabbb
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Accept |
| 2 | Reject |
| 3 | TM timeout |
| 1 | Error |

## JSON sketches

**DFA transition:** `{ "from": "E", "symbol": "0", "to": "O" }`

**NFA epsilon:** `{ "from": "q0", "symbol": "eps", "to": "q1" }`

**TM transition:** `{ "state": "q0", "read": "a", "write": "X", "move": "R", "next": "q1" }`  
Moves: `L`, `R`, `S` (stay).

## Library API

```python
from automata.loader import load_dfa, load_tm

dfa = load_dfa("examples/dfa_even_zeros.json")
print(dfa.accepts("00"))

tm = load_tm("examples/tm_anbn.json")
print(tm.run("aabb"))
```

## Layout

```
automata/     # package
examples/     # sample machines
tests/
```
