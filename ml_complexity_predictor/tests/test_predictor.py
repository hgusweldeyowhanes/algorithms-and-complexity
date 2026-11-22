from __future__ import annotations

from ml_complexity_predictor.algorithms import ALGORITHMS, CLASSES
from ml_complexity_predictor.dataset import (
    build_feature_matrix,
    generate_synthetic,
    train_test_split,
)
from ml_complexity_predictor.features import FEATURE_DIM, extract_features
from ml_complexity_predictor.model import ComplexityClassifier
from ml_complexity_predictor.predict import predict_algorithm, train_default_model
from ml_complexity_predictor.timing import measure_curve


def test_feature_dim():
    ns = [10, 20, 40, 80, 160, 320]
    times = [0.01 * n for n in ns]  # linear-ish
    feats = extract_features(ns, times)
    assert len(feats) == FEATURE_DIM


def test_synthetic_train_high_accuracy():
    curves = generate_synthetic(per_class=40, seed=3, noise=0.05)
    X, y = build_feature_matrix(curves)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.25, seed=4)
    model = ComplexityClassifier()
    model.fit(Xtr, ytr, epochs=350, lr=0.2, seed=5)
    metrics = model.evaluate(Xte, yte)
    assert metrics["accuracy"] >= 0.85


def test_classes_cover_brief():
    for needed in ("O(n)", "O(n log n)", "O(n^2)", "O(2^n)"):
        assert needed in CLASSES


def test_algorithms_categories():
    cats = {c for _n, (_f, _t, c) in ALGORITHMS.items()}
    assert {"sorting", "searching", "graph"} <= cats


def test_measure_and_predict_linear_search():
    model, _ = train_default_model(
        per_class=50, include_real=False, seed=9, epochs=300
    )
    report = predict_algorithm(model, "linear_search", runs=1, seed=2)
    assert report.theoretical == "O(n)"
    assert report.predicted in CLASSES
    assert abs(sum(report.all_probs.values()) - 1.0) < 1e-6


def test_curve_timing_smoke():
    fn, theory, _ = ALGORITHMS["binary_search"]
    curve = measure_curve(
        "binary_search",
        fn,
        [1000, 5000, 20000, 80000],
        theoretical=theory,
        runs=1,
        warmup=0,
        seed=1,
    )
    assert len(curve.samples) == 4
    assert all(s.mean_seconds >= 0 for s in curve.samples)
