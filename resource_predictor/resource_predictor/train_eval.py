"""Train RF / XGBoost / Neural Net and evaluate time & memory predictions."""
from __future__ import annotations

import random
from dataclasses import dataclass

from resource_predictor.dataset import collect_dataset
from resource_predictor.features import as_numpy, build_xy, invert_target
from resource_predictor.measure import ResourceSample
from resource_predictor.models import BACKEND, DualTargetModel, build_model_zoo
from resource_predictor.system_info import collect_system_features, system_report


@dataclass
class TargetScores:
    mae: float
    mape: float
    r2: float


@dataclass
class ModelScores:
    name: str
    time: TargetScores
    memory: TargetScores


@dataclass
class ResourceStudy:
    n_samples: int
    n_train: int
    n_test: int
    backend: str
    scores: list[ModelScores]
    samples: list[ResourceSample]


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def _mape(y_true: list[float], y_pred: list[float], *, floor: float = 1e-3) -> float:
    """Mean absolute percentage error with a floor to avoid tiny-denominator blowups."""
    return (
        100.0
        * sum(abs(a - b) / max(abs(a), floor) for a, b in zip(y_true, y_pred))
        / max(1, len(y_true))
    )


def _r2(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return 0.0
    mean = sum(y_true) / len(y_true)
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    if ss_tot <= 1e-18:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _score(y_true: list[float], y_pred: list[float], *, mape_floor: float) -> TargetScores:
    return TargetScores(
        mae=_mae(y_true, y_pred),
        mape=_mape(y_true, y_pred, floor=mape_floor),
        r2=_r2(y_true, y_pred),
    )


def _to_list(pred) -> list[float]:
    try:
        return [float(x) for x in pred]
    except TypeError:
        return [float(pred)]


def train_test_split(
    X: list[list[float]],
    y_time: list[float],
    y_mem: list[float],
    *,
    test_size: float,
    seed: int,
):
    idx = list(range(len(X)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_size))
    te = set(idx[:n_test])
    Xtr, Xte, yt_tr, yt_te, ym_tr, ym_te = [], [], [], [], [], []
    for i, row in enumerate(X):
        if i in te:
            Xte.append(row)
            yt_te.append(y_time[i])
            ym_te.append(y_mem[i])
        else:
            Xtr.append(row)
            yt_tr.append(y_time[i])
            ym_tr.append(y_mem[i])
    return Xtr, Xte, yt_tr, yt_te, ym_tr, ym_te


def evaluate_model(
    model: DualTargetModel,
    X_test: list[list[float]],
    y_time_log: list[float],
    y_mem_log: list[float],
) -> ModelScores:
    X_in, _, _ = as_numpy(X_test, y_time_log, y_mem_log)
    pred_t_log, pred_m_log = model.predict(X_in)
    yt = invert_target(y_time_log)
    ym = invert_target(y_mem_log)
    pt = [max(0.0, v) for v in invert_target(_to_list(pred_t_log))]
    pm = [max(0.0, v) for v in invert_target(_to_list(pred_m_log))]
    return ModelScores(
        name=model.name,
        time=_score(yt, pt, mape_floor=1e-3),
        memory=_score(ym, pm, mape_floor=1.0),
    )


def run_study(
    *,
    runs: int = 2,
    seed: int = 1,
    test_size: float = 0.25,
    algorithms: list[str] | None = None,
) -> ResourceStudy:
    samples = collect_dataset(algorithms=algorithms, runs=runs, seed=seed)
    X, y_time, y_mem = build_xy(samples)
    Xtr, Xte, yt_tr, yt_te, ym_tr, ym_te = train_test_split(
        X, y_time, y_mem, test_size=test_size, seed=seed
    )
    scores: list[ModelScores] = []
    zoo = build_model_zoo(seed=seed)
    Xtr_in, yt_in, ym_in = as_numpy(Xtr, yt_tr, ym_tr)
    for model in zoo:
        model.fit(Xtr_in, yt_in, ym_in)
        scores.append(evaluate_model(model, Xte, yt_te, ym_te))
    return ResourceStudy(
        n_samples=len(samples),
        n_train=len(Xtr),
        n_test=len(Xte),
        backend=BACKEND,
        scores=scores,
        samples=samples,
    )


def format_study(study: ResourceStudy) -> str:
    lines = [
        "Predicting CPU time & memory — model comparison",
        f"backend: {study.backend}  samples: {study.n_samples}  "
        f"train: {study.n_train}  test: {study.n_test}",
        "system features:",
        system_report(collect_system_features()),
        "",
        f"{'model':<14} {'time MAE(s)':>12} {'time MAPE%':>11} {'time R²':>8} "
        f"{'mem MAE(KB)':>12} {'mem MAPE%':>10} {'mem R²':>8}",
        "-" * 88,
    ]
    for s in study.scores:
        lines.append(
            f"{s.name:<14} {s.time.mae:12.6f} {s.time.mape:11.1f} {s.time.r2:8.3f} "
            f"{s.memory.mae:12.1f} {s.memory.mape:10.1f} {s.memory.r2:8.3f}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  • Features: input size, algorithm one-hot, CPU/OS/Python/RAM")
    lines.append("  • Targets: wall time (s) and peak tracemalloc memory (KB)")
    lines.append("  • Models: Random Forest, XGBoost (or GBDT fallback), Neural Net")
    return "\n".join(lines)


def format_sample_table(samples: list[ResourceSample], *, limit: int = 12) -> str:
    lines = [
        "Measured resource samples (excerpt)",
        f"{'algorithm':<16} {'n':>8} {'time_s':>10} {'peak_KB':>10}",
        "-" * 48,
    ]
    for s in samples[:limit]:
        lines.append(
            f"{s.algorithm:<16} {s.n:8d} {s.time_s:10.6f} {s.peak_memory_kb:10.1f}"
        )
    if len(samples) > limit:
        lines.append(f"... ({len(samples) - limit} more)")
    return "\n".join(lines)
