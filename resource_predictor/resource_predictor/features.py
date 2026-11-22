"""Build numeric feature matrices from resource samples."""
from __future__ import annotations

import math

from resource_predictor.algorithms import ALGORITHM_NAMES
from resource_predictor.measure import ResourceSample
from resource_predictor.system_info import FEATURE_NAMES_SYSTEM

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False


def _algo_one_hot(name: str) -> list[float]:
    return [1.0 if name == a else 0.0 for a in ALGORITHM_NAMES]


FEATURE_NAMES: tuple[str, ...] = (
    "log_n",
    "n",
    *tuple(f"algo_{a}" for a in ALGORITHM_NAMES),
    *FEATURE_NAMES_SYSTEM,
)


def sample_to_features(s: ResourceSample) -> list[float]:
    return [
        math.log2(s.n) if s.n > 1 else 0.0,
        float(s.n),
        *_algo_one_hot(s.algorithm),
        *s.system.as_vector(),
    ]


def build_xy(
    samples: list[ResourceSample],
) -> tuple[list[list[float]], list[float], list[float]]:
    """
    Returns X, y_time, y_memory_kb.
    Targets are log1p-scaled for stabler regression; invert with expm1.
    """
    X = [sample_to_features(s) for s in samples]
    y_time = [math.log1p(s.time_s) for s in samples]
    y_mem = [math.log1p(s.peak_memory_kb) for s in samples]
    return X, y_time, y_mem


def invert_target(values: list[float]) -> list[float]:
    return [math.expm1(v) for v in values]


def as_numpy(X: list[list[float]], y_time: list[float], y_mem: list[float]):
    if not HAS_NUMPY:
        return X, y_time, y_mem
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y_time, dtype=np.float64),
        np.asarray(y_mem, dtype=np.float64),
    )
