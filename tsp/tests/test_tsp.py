from __future__ import annotations

from pathlib import Path

from tsp.approximate import mst_double_approx
from tsp.exact import held_karp
from tsp.heuristics import nearest_neighbor, nearest_neighbor_two_opt, two_opt
from tsp.instance import City, Instance, load_instance

ROOT = Path(__file__).resolve().parents[1]


def test_load_tiny4():
    inst = load_instance(ROOT / "examples" / "tiny4.txt")
    assert inst.n == 4
    assert inst.cities[0].name == "A"


def test_held_karp_square():
    # Unit square: optimal tour length = 4
    inst = Instance(
        cities=[
            City("A", 0, 0),
            City("B", 1, 0),
            City("C", 1, 1),
            City("D", 0, 1),
        ]
    )
    r = held_karp(inst, start=0)
    assert abs(r.length - 4.0) < 1e-9
    assert len(r.tour) == 4
    assert set(r.tour) == {0, 1, 2, 3}


def test_nearest_neighbor_visits_all():
    inst = load_instance(ROOT / "examples" / "tiny4.txt")
    r = nearest_neighbor(inst, start=0)
    assert len(r.tour) == 4
    assert set(r.tour) == set(range(4))
    assert abs(r.length - inst.tour_length(r.tour)) < 1e-12


def test_two_opt_not_worse():
    inst = load_instance(ROOT / "examples" / "cities8.txt")
    nn = nearest_neighbor(inst, start=0)
    improved = two_opt(inst, nn.tour)
    assert improved.length <= nn.length + 1e-9


def test_heuristic_vs_exact_gap_nonnegative():
    inst = load_instance(ROOT / "examples" / "cities8.txt")
    exact = held_karp(inst, start=0)
    heur = nearest_neighbor_two_opt(inst, start=0)
    assert heur.length + 1e-9 >= exact.length
    assert set(heur.tour) == set(range(inst.n))
    assert set(exact.tour) == set(range(inst.n))


def test_held_karp_rejects_large_n():
    cities = [City(str(i), float(i), 0.0) for i in range(21)]
    inst = Instance(cities=cities)
    try:
        held_karp(inst)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "n<=20" in str(exc)


def test_mst_double_approx_visits_all_and_within_2opt():
    inst = load_instance(ROOT / "examples" / "cities8.txt")
    exact = held_karp(inst, start=0)
    approx = mst_double_approx(inst, start=0)
    assert set(approx.tour) == set(range(inst.n))
    assert len(approx.tour) == inst.n
    # 2-approximation guarantee (metric)
    assert approx.length <= 2 * exact.length + 1e-6
    assert approx.length + 1e-9 >= exact.length


def test_mst_double_square():
    inst = Instance(
        cities=[
            City("A", 0, 0),
            City("B", 1, 0),
            City("C", 1, 1),
            City("D", 0, 1),
        ]
    )
    approx = mst_double_approx(inst, start=0)
    assert set(approx.tour) == {0, 1, 2, 3}
    assert abs(approx.length - 4.0) < 1e-9 or approx.length <= 8.0 + 1e-9

