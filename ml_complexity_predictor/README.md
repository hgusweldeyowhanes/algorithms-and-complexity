# ML Complexity Predictor

**Machine Learning-Based Algorithm Complexity Predictor** — train a classifier on execution-time curves, then predict Big-O and compare to theory.

| | |
|-|-|
| **Classes** | \(O(\log n)\), \(O(n)\), \(O(n\log n)\), \(O(n^2)\), \(O(2^n)\) |
| **Model** | Softmax regression on curve-shape features (pure Python, no sklearn) |
| **Targets** | Sorting · Searching · Graph traversal |

## Idea

1. Run an algorithm at several input sizes \(n\) and record mean wall time \(T(n)\).
2. Extract **growth features** (normalized ratios, log–log slope, residuals vs each Big-O model).
3. Classify the feature vector with a trained multiclass model.
4. Report **theoretical** vs **ML-predicted** complexity.

Training data is mostly **synthetic** noisy curves \(T(n)\approx c\cdot f(n)\); optional real timings refine the model.

Prediction **blends** the softmax classifier with a classical residual fit (same idea as curve-fitting): when one Big-O model clearly wins on relative residual, that label is preferred. Expect occasional mismatches on **\(O(n)\) vs \(O(n\log n)\)** — those are hard to tell apart from short wall-clock curves (an intended part of the study).

## Usage

```bash
cd ml_complexity_predictor

# Train + study all built-in algorithms
python -m ml_complexity_predictor

# Faster training (synthetic only)
python -m ml_complexity_predictor --fast

# One algorithm (+ timing table)
python -m ml_complexity_predictor merge_sort --show-curve

# Filter by category
python -m ml_complexity_predictor --category sorting --category searching

python -m ml_complexity_predictor --list
```

## Built-in algorithms

| Algorithm | Category | Theory |
|-----------|----------|--------|
| `bubble_sort` | sorting | \(O(n^2)\) |
| `merge_sort`, `quick_sort` | sorting | \(O(n\log n)\) |
| `linear_search` | searching | \(O(n)\) |
| `binary_search` | searching | \(O(\log n)\) |
| `bfs`, `dfs` | graph | \(O(n)\) (sparse) |
| `dijkstra` | graph | \(O(n\log n)\) (sparse) |
| `exponential_subset_sum` | other | \(O(2^n)\) |

## Library API

```python
from ml_complexity_predictor import train_default_model, predict_algorithm

model, metrics = train_default_model(per_class=80, include_real=False, seed=1)
report = predict_algorithm(model, "bubble_sort", runs=2)
print(report.theoretical, report.predicted, report.confidence)
```

## Layout

```
ml_complexity_predictor/
  algorithms.py   # sorting / searching / graph / exponential
  timing.py       # (n, T(n)) measurement
  features.py     # curve → feature vector
  dataset.py      # synthetic + real labeled curves
  model.py        # softmax classifier
  predict.py      # train / predict / compare
```
