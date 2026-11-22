from __future__ import annotations

from resource_predictor.algorithms import ALGORITHMS
from resource_predictor.dataset import collect_dataset
from resource_predictor.features import FEATURE_NAMES, build_xy, sample_to_features
from resource_predictor.measure import measure_sample
from resource_predictor.models import build_model_zoo
from resource_predictor.system_info import collect_system_features
from resource_predictor.train_eval import run_study


def test_system_features():
    s = collect_system_features()
    assert s.cpu_count >= 1
    assert len(s.as_vector()) == 5


def test_measure_sample():
    sample = measure_sample("merge_sort", ALGORITHMS["merge_sort"], 200, runs=1, seed=2)
    assert sample.time_s >= 0
    assert sample.peak_memory_kb > 0


def test_feature_dim_matches_names():
    sample = measure_sample("bfs", ALGORITHMS["bfs"], 100, runs=1, seed=3)
    feats = sample_to_features(sample)
    assert len(feats) == len(FEATURE_NAMES)


def test_models_fit_predict_shapes():
    samples = collect_dataset(
        algorithms=["merge_sort", "binary_search", "matrix_multiply"],
        runs=1,
        seed=4,
    )
    X, yt, ym = build_xy(samples)
    assert len(X) == len(samples)
    for model in build_model_zoo(seed=0):
        model.fit(X, yt, ym)
        pt, pm = model.predict(X[:3])
        assert len(list(pt)) == 3
        assert len(list(pm)) == 3


def test_study_runs():
    study = run_study(
        runs=1,
        seed=5,
        test_size=0.3,
        algorithms=["bubble_sort", "linear_search", "matrix_multiply"],
    )
    assert study.n_samples >= 10
    assert len(study.scores) == 3
    names = {s.name for s in study.scores}
    assert names == {"RandomForest", "XGBoost", "NeuralNet"}
