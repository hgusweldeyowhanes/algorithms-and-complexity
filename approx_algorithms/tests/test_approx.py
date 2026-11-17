from __future__ import annotations

from approx_algorithms.graph import (
    adversarial_vc_disjoint_edges,
    adversarial_vc_star,
    random_graph,
)
from approx_algorithms.set_cover import (
    greedy_set_cover,
    harmonic,
    is_set_cover,
    random_set_cover,
    set_cover_exact,
)
from approx_algorithms.study import study_set_cover, study_vertex_cover
from approx_algorithms.vertex_cover import (
    is_vertex_cover,
    vertex_cover_2approx,
    vertex_cover_exact,
)


def test_vc_2approx_is_cover_and_within_2opt():
    g = random_graph(12, 0.2, seed=3)
    approx = vertex_cover_2approx(g, seed=3)
    opt = vertex_cover_exact(g)
    assert is_vertex_cover(g, approx)
    assert is_vertex_cover(g, opt)
    assert len(approx) <= 2 * len(opt) + 0  # integral
    assert len(approx) >= len(opt)


def test_vc_star_ratio_two():
    g = adversarial_vc_star(8)
    approx = vertex_cover_2approx(g, seed=0)
    opt = vertex_cover_exact(g)
    assert len(opt) == 1
    assert len(approx) == 2
    assert len(approx) == 2 * len(opt)


def test_vc_disjoint_ratio_two():
    g = adversarial_vc_disjoint_edges(4)
    approx = vertex_cover_2approx(g, seed=1)
    opt = vertex_cover_exact(g)
    assert len(opt) == 4
    assert len(approx) == 8


def test_greedy_set_cover_valid_and_at_least_opt():
    inst = random_set_cover(6, 9, set_size=2, seed=5)
    gre = greedy_set_cover(inst)
    opt = set_cover_exact(inst)
    assert is_set_cover(inst, gre)
    assert is_set_cover(inst, opt)
    assert len(gre) >= len(opt)
    assert len(gre) <= harmonic(len(inst.universe)) * len(opt) + 1e-9


def test_studies_run():
    vc = study_vertex_cover(seed=2)
    sc = study_set_cover(seed=2)
    assert vc and sc
    assert all(r.ratio + 1e-9 >= 1.0 for r in vc)
    assert all(r.approx_size >= r.opt_size for r in sc)
