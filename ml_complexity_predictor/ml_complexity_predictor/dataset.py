"""
Build a labeled training set from:
  1) synthetic noisy Big-O curves (bulk of the data)
  2) short real-algorithm timings (optional, for realism)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ml_complexity_predictor.algorithms import (
    ALGORITHMS,
    CLASSES,
    default_sizes,
)
from ml_complexity_predictor.features import MODEL_FNS, extract_features
from ml_complexity_predictor.timing import measure_curve


@dataclass
class LabeledCurve:
    label: str
    ns: list[int]
    times: list[float]
    source: str


def _synthetic_sizes(label: str, rng: random.Random) -> list[int]:
    if label == "O(2^n)":
        base = rng.randint(6, 10)
        return [base + 2 * i for i in range(6)]
    if label == "O(log n)":
        base = rng.choice([1000, 2000, 5000])
        return [base * (3**i) for i in range(6)]
    if label == "O(n^2)":
        base = rng.randint(40, 80)
        return [base + i * base // 2 for i in range(6)]
    if label == "O(n log n)":
        base = rng.randint(200, 500)
        return [base * (2**i) for i in range(6)]
    # O(n)
    base = rng.randint(5_000, 20_000)
    return [base * (i + 1) for i in range(6)]


def generate_synthetic(
    *,
    per_class: int = 80,
    seed: int = 0,
    noise: float = 0.08,
) -> list[LabeledCurve]:
    """
    T(n) = c * f(n) * (1 + ε), with random c and multiplicative noise.
    """
    rng = random.Random(seed)
    out: list[LabeledCurve] = []
    for label in CLASSES:
        f = MODEL_FNS[label]
        for _ in range(per_class):
            ns = _synthetic_sizes(label, rng)
            c = 10 ** rng.uniform(-9, -5)
            times: list[float] = []
            for n in ns:
                fv = f(float(n))
                if not math.isfinite(fv) or fv <= 0:
                    fv = 1.0
                eps = 1.0 + rng.gauss(0.0, noise)
                times.append(max(1e-12, c * fv * max(0.2, eps)))
            out.append(LabeledCurve(label=label, ns=ns, times=times, source="synthetic"))
    return out


def measure_labeled_algorithms(
    *,
    names: list[str] | None = None,
    runs: int = 2,
    seed: int = 1,
) -> list[LabeledCurve]:
    """Time built-in algorithms and label by their theoretical class."""
    names = names or list(ALGORITHMS.keys())
    out: list[LabeledCurve] = []
    for i, name in enumerate(names):
        fn, theoretical, _cat = ALGORITHMS[name]
        sizes = default_sizes(name)
        # Use smaller schedules for training speed when possible
        if name == "bubble_sort":
            sizes = [40, 60, 80, 100, 120, 140]
        elif name in {"merge_sort", "quick_sort"}:
            sizes = [200, 400, 800, 1600, 3200, 5000]
        elif name in {"bfs", "dfs"}:
            sizes = [500, 1000, 2000, 4000, 8000, 12000]
        elif name == "dijkstra":
            sizes = [200, 400, 800, 1600, 3200, 5000]
        elif name == "linear_search":
            sizes = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
        elif name == "binary_search":
            sizes = [10_000, 40_000, 100_000, 250_000, 500_000, 1_000_000]
        elif name == "exponential_subset_sum":
            sizes = [8, 10, 12, 14, 16, 18]

        curve = measure_curve(
            name,
            fn,
            sizes,
            theoretical=theoretical,
            runs=runs,
            warmup=1,
            seed=seed + i,
        )
        out.append(
            LabeledCurve(
                label=theoretical,
                ns=curve.ns,
                times=curve.times,
                source=f"algo:{name}",
            )
        )
    return out


def build_feature_matrix(
    curves: list[LabeledCurve],
) -> tuple[list[list[float]], list[str]]:
    X: list[list[float]] = []
    y: list[str] = []
    for c in curves:
        try:
            X.append(extract_features(c.ns, c.times))
            y.append(c.label)
        except ValueError:
            continue
    return X, y


def train_test_split(
    X: list[list[float]],
    y: list[str],
    *,
    test_frac: float = 0.25,
    seed: int = 0,
) -> tuple[list[list[float]], list[str], list[list[float]], list[str]]:
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_frac))
    test_idx = set(idx[:n_test])
    Xtr, ytr, Xte, yte = [], [], [], []
    for i, (x, lab) in enumerate(zip(X, y)):
        if i in test_idx:
            Xte.append(x)
            yte.append(lab)
        else:
            Xtr.append(x)
            ytr.append(lab)
    return Xtr, ytr, Xte, yte
