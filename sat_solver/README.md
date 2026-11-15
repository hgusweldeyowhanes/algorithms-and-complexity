# CDCL SAT Solver

A from-scratch **CDCL** SAT solver in Python with a **DIMACS CNF** CLI.

## Features

- 2-watched-literal unit propagation
- 1-UIP conflict analysis and clause learning
- VSIDS variable activity heuristic
- Luby restart schedule
- Phase saving

## Requirements

- Python 3.10+
- No runtime dependencies

Optional for tests:

```bash
pip install pytest
```

## Usage

From this directory:

```bash
python -m sat_solver examples/sat_tiny.cnf
python -m sat_solver examples/unsat_tiny.cnf
python -m sat_solver path/to/formula.cnf --seed 42
```

### Output

Satisfiable:

```
s SATISFIABLE
v 1 2 ... 0
```

Unsatisfiable:

```
s UNSATISFIABLE
```

### Exit codes

| Code | Meaning        |
|------|----------------|
| 10   | Satisfiable    |
| 20   | Unsatisfiable  |
| 1    | I/O / parse error |

## Library API

```python
from sat_solver import parse_dimacs, Solver

formula = parse_dimacs("""
p cnf 1 2
1 0
-1 0
""")
result = Solver(formula).solve()
print(result.sat)  # False
```

## Layout

```
sat_solver/          # package
examples/            # sample CNF files
tests/               # pytest suite
```

## Notes

This is an educational / practical CDCL implementation, not a competition-tuned binary.
It is suitable for small and medium CNFs; very large industrial instances will be slow in pure Python.
