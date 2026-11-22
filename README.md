# Algorithms and Complexity

Implementations and experiments in algorithms and computational complexity.

## Projects

| Project | Description |
|---------|-------------|
| [`sat_solver/`](sat_solver/) | CDCL SAT solver with DIMACS CNF CLI |
| [`sat_phase_transition/`](sat_phase_transition/) | Random k-SAT phase transition (\(P(\mathrm{SAT})\) vs \(\alpha=m/n\)) |
| [`tsp/`](tsp/) | TSP exact vs approximate (MST 2-approx) vs heuristic |
| [`algo_benchmarker/`](algo_benchmarker/) | Time algorithms vs \(n\) and guess complexity class |
| [`automata/`](automata/) | DFA / NFA / Turing machine simulator |
| [`np_reductions/`](np_reductions/) | NP-completeness reductions (3-SAT → Clique / Vertex Cover) |
| [`approx_algorithms/`](approx_algorithms/) | Approximation algorithms study (VC 2-approx + greedy Set Cover vs OPT) |
| [`ml_complexity_predictor/`](ml_complexity_predictor/) | ML-based Big-O predictor from execution-time curves |
| [`resource_predictor/`](resource_predictor/) | Predict CPU time & memory (RF, XGBoost, Neural Net) |
| [`../container-resource-allocation/`](../container-resource-allocation/) | Hybrid ML + optimization container scheduler study |
| [`../algorithm-complexity-research/`](../algorithm-complexity-research/) | Shortest-path algorithm selection study |

## Quick start

**SAT solver**

```bash
cd sat_solver
python -m sat_solver examples/sat_tiny.cnf
```

**SAT phase transition**

```bash
cd sat_phase_transition
python -m sat_phase_transition --n 15 --trials 10 --steps 9
```

**TSP (exact vs heuristic)**

```bash
cd tsp
python -m tsp examples/cities8.txt
```

**Algorithm complexity benchmarker**

```bash
cd algo_benchmarker
python -m algo_benchmarker bubble_sort
```

**Automata / Turing machine**

```bash
cd automata
python -m automata examples/tm_anbn.json aabb --trace
```

**NP-completeness reductions**

```bash
cd np_reductions
python -m np_reductions both --graph
```

**Approximation algorithms study**

```bash
cd approx_algorithms
python -m approx_algorithms
```

**ML complexity predictor**

```bash
cd ml_complexity_predictor
python -m ml_complexity_predictor --fast
```

**Resource predictor (time & memory)**

```bash
cd resource_predictor
python -m resource_predictor --show-samples
```

## Adding a new project

Create a new folder under this repo (e.g. `graphs/`, `dynamic_programming/`) with its own README, code, and tests.
