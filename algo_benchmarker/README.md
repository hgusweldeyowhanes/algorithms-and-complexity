# Algorithm Complexity Benchmarker

Time algorithms on growing input sizes \(n\), then **guess** the asymptotic class
by fitting \(T(n) \approx c \cdot f(n)\) for common \(f\):

`O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n²)`, `O(n³)`, `O(2ⁿ)`, `O(φⁿ)`.

## Built-in demos

| Name | Expected |
|------|----------|
| `linear_sum` | \(O(n)\) |
| `binary_search` | \(O(\log n)\) |
| `merge_sort` | \(O(n \log n)\) |
| `bubble_sort` | \(O(n^2)\) |
| `fib_recursive` | \(O(\varphi^n)\) |

## Usage

```bash
cd algo_benchmarker

python -m algo_benchmarker --list
python -m algo_benchmarker bubble_sort
python -m algo_benchmarker merge_sort --runs 5
python -m algo_benchmarker linear_sum --sizes 100000,200000,400000,800000
```

### Example

```text
algorithm: bubble_sort  (expected O(n^2))
         n       mean_s
------------------------------
        50     0.00012
       ...
best guess: O(n^2)
```

Fitting is heuristic (noise, constants, cache effects). Use it as a teaching aid, not a proof.

## Library API

```python
from algo_benchmarker.algorithms import ALGORITHMS, default_sizes
from algo_benchmarker import run_benchmark, fit_complexity

fn, expected = ALGORITHMS["bubble_sort"]
result = run_benchmark("bubble_sort", fn, default_sizes("bubble_sort"), expected=expected)
print(fit_complexity(result)[0].label)
```

## Layout

```
algo_benchmarker/   # package
tests/
```
