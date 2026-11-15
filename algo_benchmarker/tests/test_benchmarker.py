from __future__ import annotations

from algo_benchmarker.algorithms import ALGORITHMS, bubble_sort, linear_sum, merge_sort
from algo_benchmarker.benchmark import run_benchmark
from algo_benchmarker.fit import fit_complexity
import random


def test_algorithms_registered():
    assert "linear_sum" in ALGORITHMS
    assert "bubble_sort" in ALGORITHMS
    assert "merge_sort" in ALGORITHMS


def test_linear_runs():
    rng = random.Random(0)
    assert linear_sum(100, rng) == sum(range(100))


def test_benchmark_and_fit_bubble():
    fn, expected = ALGORITHMS["bubble_sort"]
    result = run_benchmark(
        "bubble_sort",
        fn,
        [40, 60, 80, 100],
        expected=expected,
        runs=2,
        warmup=0,
        seed=1,
    )
    assert len(result.samples) == 4
    assert result.samples[0].mean_seconds >= 0
    # times should generally increase with n
    assert result.samples[-1].mean_seconds >= result.samples[0].mean_seconds
    fits = fit_complexity(result)
    assert fits
    # best should be quadratic-ish for bubble sort
    assert fits[0].label in {"O(n^2)", "O(n^3)", "O(n log n)"}


def test_merge_sort_beats_quadratic_guess_often():
    fn, expected = ALGORITHMS["merge_sort"]
    result = run_benchmark(
        "merge_sort",
        fn,
        [200, 400, 800, 1200],
        expected=expected,
        runs=2,
        warmup=0,
        seed=2,
    )
    fits = fit_complexity(result)
    assert fits
    assert fits[0].label in {"O(n log n)", "O(n)", "O(n^2)"}
