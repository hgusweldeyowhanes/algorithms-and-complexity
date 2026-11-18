from __future__ import annotations

from pathlib import Path

from np_reductions.clique import reduce_3sat_to_clique
from np_reductions.sat import parse_formula
from np_reductions.verify import verify_clique_reduction, verify_vertex_cover_reduction
from np_reductions.vertex_cover import reduce_3sat_to_vertex_cover

ROOT = Path(__file__).resolve().parents[1]


def test_parse_and_sat():
    f = parse_formula("(x ∨ ~y ∨ z) ∧ (~x ∨ y ∨ z)")
    assert len(f.clauses) == 2
    assert f.solve() is not None


def test_clique_reduction_sat_roundtrip():
    f = parse_formula(Path(ROOT / "examples" / "sat_demo.txt").read_text(encoding="utf-8"))
    red, check = verify_clique_reduction(f)
    assert check.ok, check.messages
    asn = f.solve()
    assert asn is not None
    clique = red.assignment_to_clique(asn)
    assert red.graph.is_clique(clique)
    assert len(clique) == red.k
    back = red.clique_to_assignment(clique)
    assert f.is_satisfied(back)


def test_clique_reduction_unsat():
    f = parse_formula("(x ∨ x ∨ x) ∧ (~x ∨ ~x ∨ ~x)")
    red, check = verify_clique_reduction(f)
    assert check.ok, check.messages
    assert red.graph.find_clique(red.k) is None


def test_vertex_cover_reduction_sat_roundtrip():
    f = parse_formula(Path(ROOT / "examples" / "sat_demo.txt").read_text(encoding="utf-8"))
    red, check = verify_vertex_cover_reduction(f)
    assert check.ok, check.messages
    asn = f.solve()
    assert asn is not None
    cover = red.assignment_to_cover(asn)
    assert red.graph.is_vertex_cover(cover)
    assert len(cover) == red.k
    back = red.cover_to_assignment(cover)
    assert f.is_satisfied(back)


def test_vertex_cover_reduction_unsat():
    f = parse_formula("(x ∨ x ∨ x) ∧ (~x ∨ ~x ∨ ~x)")
    red, check = verify_vertex_cover_reduction(f)
    assert check.ok, check.messages
    assert red.graph.find_vertex_cover(red.k) is None


def test_k_sizes():
    f = parse_formula("(a ∨ b ∨ c) ∧ (~a ∨ ~b ∨ c)")
    cl = reduce_3sat_to_clique(f)
    assert cl.k == 2
    assert len(cl.graph.vertices) == 6
    vc = reduce_3sat_to_vertex_cover(f)
    # n=3, m=2 → k = 3 + 4 = 7
    assert vc.k == 7
