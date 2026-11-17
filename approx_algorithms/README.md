# Approximation Algorithms Study

Compare classical **approximation algorithms** to **OPT** on random and adversarial instances.

| Problem | Algorithm | Guarantee |
|---------|-----------|-----------|
| **Vertex Cover** | Maximal-matching 2-approx (pick both ends of matched edges) | \(\|C\| \le 2\cdot OPT\) |
| **Set Cover** | Greedy (pick set covering most uncovered elements) | \(\|C\| \le H_{\|U\|}\cdot OPT\) |

Exact OPT is computed by brute force on small instances so we can measure the **empirical ratio** `approx / OPT`.

## Usage

```bash
cd approx_algorithms
python -m approx_algorithms
python -m approx_algorithms vertex-cover
python -m approx_algorithms set-cover --seed 7
```

### What to look for

- **Vertex Cover:** star and disjoint-edge graphs push the ratio toward **2** (tight for this algorithm).
- **Set Cover:** random instances usually sit well below \(H_n\); the adversarial row searches for a case with `greedy > OPT`.

## Library API

```python
from approx_algorithms import vertex_cover_2approx, greedy_set_cover
from approx_algorithms.graph import random_graph
from approx_algorithms.set_cover import random_set_cover

g = random_graph(12, 0.2, seed=1)
cover = vertex_cover_2approx(g)

inst = random_set_cover(7, 10, set_size=3, seed=1)
chosen = greedy_set_cover(inst)
```

## Layout

```
approx_algorithms/   # package
tests/
```
