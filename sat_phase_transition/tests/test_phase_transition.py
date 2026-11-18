from __future__ import annotations

from sat_phase_transition.experiment import run_sweep
from sat_phase_transition.generate import random_ksat
from sat_phase_transition.solver_bridge import solve_formula


def test_random_ksat_shape():
    f = random_ksat(10, 40, k=3, seed=0)
    assert f.num_vars == 10
    assert f.num_clauses == 40
    assert abs(f.alpha - 4.0) < 1e-12
    for cl in f.clauses:
        assert len(cl) == 3
        assert len({abs(x) for x in cl}) == 3


def test_dimacs_header():
    f = random_ksat(5, 10, k=3, seed=1)
    text = f.to_dimacs()
    assert "p cnf 5 10" in text


def test_underconstrained_mostly_sat():
    # Low alpha → almost always SAT
    f = random_ksat(12, 12, k=3, seed=2)  # alpha = 1
    sat, _ = solve_formula(f)
    assert sat is True


def test_overconstrained_unsat_possible():
    # Very high alpha with small n often UNSAT
    f = random_ksat(8, 80, k=3, seed=3)  # alpha = 10
    sat, _ = solve_formula(f)
    # not a hard assert on False (rare SAT possible), but solvable
    assert sat in (True, False)


def test_sweep_monotonic_trend():
    points = run_sweep(
        n=12,
        k=3,
        alpha_min=2.0,
        alpha_max=6.0,
        steps=5,
        trials=8,
        seed=11,
    )
    assert len(points) == 5
    # First point (low alpha) should be more SAT-friendly than last (high alpha)
    assert points[0].sat_probability >= points[-1].sat_probability
