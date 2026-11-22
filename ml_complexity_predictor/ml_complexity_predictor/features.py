"""
Feature extraction from empirical timing curves.

Features capture growth shape (not absolute speed), so a slow O(n) and a fast
O(n) land in the same region of feature space.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from ml_complexity_predictor.algorithms import CLASSES


def _f_log(n: float) -> float:
    return math.log2(n) if n > 1 else 1.0


def _f_n(n: float) -> float:
    return float(n)


def _f_nlogn(n: float) -> float:
    return n * math.log2(n) if n > 1 else 1.0


def _f_n2(n: float) -> float:
    return float(n * n)


def _f_2n(n: float) -> float:
    if n > 40:
        return float("inf")
    return float(2**n)


MODEL_FNS = {
    "O(log n)": _f_log,
    "O(n)": _f_n,
    "O(n log n)": _f_nlogn,
    "O(n^2)": _f_n2,
    "O(2^n)": _f_2n,
}


def _rel_residual(ns: Sequence[int], ys: Sequence[float], label: str) -> float:
    f = MODEL_FNS[label]
    fs = [f(float(n)) for n in ns]
    if any(not math.isfinite(v) or v <= 0 for v in fs):
        return 10.0
    num = sum(y * fv for y, fv in zip(ys, fs))
    den = sum(fv * fv for fv in fs)
    if den <= 0:
        return 10.0
    c = num / den
    if c <= 0:
        return 10.0
    rel = []
    for y, fv in zip(ys, fs):
        if y <= 0:
            continue
        rel.append(abs(c * fv - y) / y)
    return sum(rel) / len(rel) if rel else 10.0


def _loglog_slope(ns: Sequence[int], ys: Sequence[float]) -> float:
    """Least-squares slope of log T vs log n (≈ degree for polynomial growth)."""
    pts = [(math.log(n), math.log(t)) for n, t in zip(ns, ys) if n > 1 and t > 0]
    if len(pts) < 2:
        return 0.0
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    mz = sum(zs) / len(zs)
    num = sum((x - mx) * (z - mz) for x, z in zip(xs, zs))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _growth_ratios(ns: Sequence[int], ys: Sequence[float]) -> list[float]:
    """T(n_{i+1})/T(n_i) normalized by size jump — growth fingerprint."""
    ratios: list[float] = []
    for i in range(len(ns) - 1):
        n0, n1 = ns[i], ns[i + 1]
        t0, t1 = ys[i], ys[i + 1]
        if t0 <= 0 or n0 <= 0 or n1 <= n0:
            ratios.append(1.0)
            continue
        # (T1/T0) / (n1/n0) — ≈1 for linear, ≈ n1/n0 for quadratic, etc.
        ratios.append((t1 / t0) / (n1 / n0))
    # pad / trim to fixed length 5
    while len(ratios) < 5:
        ratios.append(ratios[-1] if ratios else 1.0)
    return ratios[:5]


FEATURE_DIM = 5 + 1 + len(CLASSES) + len(CLASSES) + 2  # growth+slope+resid+aff+gaps


def residual_vector(ns: Sequence[int], times: Sequence[float]) -> list[float]:
    return [_rel_residual(ns, times, label) for label in CLASSES]


def best_residual_label(ns: Sequence[int], times: Sequence[float]) -> tuple[str, float, float]:
    """Return (best_label, best_residual, margin_over_second)."""
    res = residual_vector(ns, times)
    ranked = sorted(zip(CLASSES, res), key=lambda kv: kv[1])
    best_lab, best_r = ranked[0]
    second_r = ranked[1][1] if len(ranked) > 1 else best_r
    return best_lab, best_r, second_r - best_r


def extract_features(ns: Sequence[int], times: Sequence[float]) -> list[float]:
    if len(ns) != len(times) or len(ns) < 3:
        raise ValueError("need >=3 matched (n, time) points")
    # Normalize times so absolute machine speed drops out
    tmax = max(times)
    ys = [t / tmax if tmax > 0 else 0.0 for t in times]

    feats: list[float] = []
    feats.extend(_growth_ratios(ns, ys))
    feats.append(_loglog_slope(ns, ys))
    residuals = residual_vector(ns, ys)
    feats.extend(residuals)
    # Affinity = 1/residual — amplifies the winning model for the classifier
    aff = [1.0 / (1e-3 + r) for r in residuals]
    s = sum(aff) or 1.0
    feats.extend([a / s for a in aff])
    r_n = residuals[CLASSES.index("O(n)")]
    r_nl = residuals[CLASSES.index("O(n log n)")]
    r_n2 = residuals[CLASSES.index("O(n^2)")]
    feats.append(r_n - r_nl)
    feats.append(r_nl - r_n2)
    return feats


def feature_names() -> list[str]:
    names = [f"growth_{i}" for i in range(5)]
    names.append("loglog_slope")
    names.extend(f"resid_{c}" for c in CLASSES)
    names.extend(f"aff_{c}" for c in CLASSES)
    names.extend(["gap_n_nlogn", "gap_nlogn_n2"])
    return names
