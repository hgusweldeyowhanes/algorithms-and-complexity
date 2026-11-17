# TSP — Exact vs Approximate

Compare **exact**, **approximation-algorithm**, and **heuristic** solvers for the
metric Traveling Salesman Problem (Euclidean distances).

| Method | Kind | Idea | Guarantee |
|--------|------|------|-----------|
| **Held–Karp** | Exact | DP over subsets | Optimal; \(O(n^2 2^n)\) |
| **MST double-tree** | Approximate | MST → double edges → Euler → shortcut | **≤ 2 · OPT** (metric) |
| **Nearest neighbor + 2-opt** | Heuristic | Greedy tour + local search | None (often good in practice) |

Exact is optimal but exponential. The MST method is a true **2-approximation**.
Heuristics have no worst-case bound but are fast — the CLI prints gaps vs OPT.

## Requirements

- Python 3.10+
- No runtime dependencies

```bash
pip install pytest   # optional, for tests
```

## Usage

```bash
cd tsp
python -m tsp examples/tiny4.txt
python -m tsp examples/cities8.txt
python -m tsp examples/cities8.txt --no-heuristic
python -m tsp examples/cities8.txt --no-exact   # approx + heuristic only
```

### Instance format

```text
# comment
Name x y
A 0 0
B 1 0
```

Distances are **Euclidean** (metric).

## Library API

```python
from tsp import load_instance, held_karp, mst_double_approx, nearest_neighbor, two_opt

inst = load_instance("examples/tiny4.txt")
exact = held_karp(inst)
approx = mst_double_approx(inst)
nn = nearest_neighbor(inst)
improved = two_opt(inst, nn.tour)
print(exact.length, approx.length, improved.length)
```

## Layout

```
tsp/           # package
examples/      # sample city files
tests/         # pytest
```
