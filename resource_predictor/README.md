# Resource Predictor

**Predicting CPU and Memory Consumption Using ML** — estimate an algorithm’s wall time and peak memory from input size, algorithm type, and system features.

| | |
|-|-|
| **Targets** | Execution time (s), peak memory (KB via `tracemalloc`) |
| **Features** | \(n\), \(\log n\), algorithm one-hot, CPU count, OS, Python version, pointer width, RAM |
| **Models** | **Random Forest**, **XGBoost**, **Neural Net** (sklearn MLP) |
| **Metrics** | MAE, MAPE %, R² on a held-out test split |

## Usage

```bash
cd resource_predictor

# Works out of the box (pure-Python RF / GBDT / MLP).
# Optional acceleration:
#   pip install -e ".[ml]"   # numpy + scikit-learn + xgboost

python -m resource_predictor
python -m resource_predictor --show-samples
python -m resource_predictor --algorithm bubble_sort --algorithm matrix_multiply
python -m resource_predictor --list
```

## What it does

1. Runs built-in workloads (sort / search / BFS / matrix multiply) across size grids.
2. Records **time** and **peak Python memory** plus host features.
3. Trains three regressors (separate heads for time and memory).
4. Prints a comparison table so you can see which model predicts best.

## Library API

```python
from resource_predictor import run_study

study = run_study(runs=2, seed=1)
for s in study.scores:
    print(s.name, s.time.r2, s.memory.r2)
```

## Layout

```
resource_predictor/
  algorithms.py    # workloads
  system_info.py   # host features
  measure.py       # time + tracemalloc
  features.py      # X / y matrices
  dataset.py       # sample collection
  models.py        # RF, XGBoost, NeuralNet
  train_eval.py    # train / score / report
```
