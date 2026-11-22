"""Collect labeled (features → time/memory) samples by running algorithms."""
from __future__ import annotations

from resource_predictor.algorithms import ALGORITHMS, size_grid
from resource_predictor.measure import ResourceSample, measure_sample
from resource_predictor.system_info import SystemFeatures, collect_system_features


def collect_dataset(
    *,
    algorithms: list[str] | None = None,
    runs: int = 2,
    seed: int = 1,
    system: SystemFeatures | None = None,
) -> list[ResourceSample]:
    names = algorithms or list(ALGORITHMS.keys())
    sysf = system or collect_system_features()
    samples: list[ResourceSample] = []
    for i, name in enumerate(names):
        fn = ALGORITHMS[name]
        for n in size_grid(name):
            samples.append(
                measure_sample(
                    name,
                    fn,
                    n,
                    runs=runs,
                    seed=seed + i * 1009,
                    system=sysf,
                )
            )
    return samples
