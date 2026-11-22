"""Train the classifier and compare theoretical vs ML-predicted complexity."""
from __future__ import annotations

from dataclasses import dataclass

from ml_complexity_predictor.algorithms import ALGORITHMS, default_sizes
from ml_complexity_predictor.dataset import (
    build_feature_matrix,
    generate_synthetic,
    measure_labeled_algorithms,
    train_test_split,
)
from ml_complexity_predictor.features import best_residual_label, extract_features
from ml_complexity_predictor.model import ComplexityClassifier
from ml_complexity_predictor.timing import Curve, format_curve, measure_curve


@dataclass
class PredictionReport:
    name: str
    category: str
    theoretical: str
    predicted: str
    confidence: float
    match: bool
    curve: Curve
    all_probs: dict[str, float]


def train_default_model(
    *,
    per_class: int = 100,
    include_real: bool = True,
    seed: int = 0,
    epochs: int = 500,
) -> tuple[ComplexityClassifier, dict[str, float]]:
    """
    Train on synthetic Big-O curves (+ optional short real timings).
    Returns (model, metrics) with train/test accuracy.
    """
    curves = generate_synthetic(per_class=per_class, seed=seed)
    if include_real:
        # Fast subset for training augmentation
        real_names = [
            "bubble_sort",
            "merge_sort",
            "linear_search",
            "binary_search",
            "bfs",
            "exponential_subset_sum",
        ]
        curves.extend(measure_labeled_algorithms(names=real_names, runs=1, seed=seed + 7))

    X, y = build_feature_matrix(curves)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.2, seed=seed + 1)
    model = ComplexityClassifier()
    train_metrics = model.fit(Xtr, ytr, epochs=epochs, lr=0.2, seed=seed + 2)
    test_metrics = model.evaluate(Xte, yte)
    metrics = {
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "n_train": train_metrics["n"],
        "n_test": test_metrics["n"],
    }
    return model, metrics


def predict_curve(
    model: ComplexityClassifier,
    curve: Curve,
    *,
    residual_margin: float = 0.05,
) -> tuple[str, float, dict[str, float]]:
    """
    Blend ML softmax with classical residual fit.

    When one Big-O model fits clearly better (margin on relative residual),
    trust that label; otherwise use the trained classifier.
    """
    feats = extract_features(curve.ns, curve.times)
    probs = model.predict_proba(feats)
    ml_label, ml_conf = model.predict(feats)
    classical, _best_r, margin = best_residual_label(curve.ns, curve.times)

    if margin >= residual_margin:
        # Soft-override: boost classical class in the reported distribution
        blended = dict(probs)
        blended[classical] = max(blended.get(classical, 0.0), 0.55 + min(0.4, margin))
        total = sum(blended.values()) or 1.0
        blended = {k: v / total for k, v in blended.items()}
        return classical, blended[classical], blended
    return ml_label, ml_conf, probs


def predict_algorithm(
    model: ComplexityClassifier,
    name: str,
    *,
    runs: int = 2,
    seed: int = 1,
    sizes: list[int] | None = None,
) -> PredictionReport:
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    fn, theoretical, category = ALGORITHMS[name]
    sizes = sizes or default_sizes(name)
    curve = measure_curve(
        name, fn, sizes, theoretical=theoretical, runs=runs, seed=seed
    )
    predicted, conf, probs = predict_curve(model, curve)
    return PredictionReport(
        name=name,
        category=category,
        theoretical=theoretical,
        predicted=predicted,
        confidence=conf,
        match=predicted == theoretical,
        curve=curve,
        all_probs=probs,
    )


def study_all(
    model: ComplexityClassifier,
    *,
    runs: int = 2,
    seed: int = 1,
    categories: set[str] | None = None,
) -> list[PredictionReport]:
    reports: list[PredictionReport] = []
    for name, (_fn, _th, cat) in ALGORITHMS.items():
        if categories and cat not in categories:
            continue
        reports.append(predict_algorithm(model, name, runs=runs, seed=seed))
    return reports


def format_report(r: PredictionReport, *, show_curve: bool = False) -> str:
    mark = "OK" if r.match else "DIFF"
    lines = [
        f"[{r.category}] {r.name}",
        f"  theoretical : {r.theoretical}",
        f"  predicted   : {r.predicted}  (confidence {r.confidence:.1%})  [{mark}]",
        "  class probabilities:",
    ]
    for lab, p in sorted(r.all_probs.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {lab:<12} {p:6.1%}")
    if show_curve:
        lines.append(format_curve(r.curve))
    return "\n".join(lines)


def format_summary(reports: list[PredictionReport]) -> str:
    ok = sum(1 for r in reports if r.match)
    lines = [
        "Theoretical vs ML-predicted complexity",
        f"{'algorithm':<24} {'category':<10} {'theory':<12} {'predicted':<12} {'conf':>6} {'match':>5}",
        "-" * 78,
    ]
    for r in reports:
        lines.append(
            f"{r.name:<24} {r.category:<10} {r.theoretical:<12} {r.predicted:<12} "
            f"{r.confidence:5.0%} {'yes' if r.match else 'no':>5}"
        )
    lines.append("-" * 78)
    lines.append(f"agreement: {ok}/{len(reports)} ({ok / len(reports):.0%})" if reports else "no reports")
    return "\n".join(lines)
