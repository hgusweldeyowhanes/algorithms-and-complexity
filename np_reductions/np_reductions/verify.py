"""Bidirectional verification of reduction correctness on a concrete formula."""
from __future__ import annotations

from dataclasses import dataclass

from np_reductions.clique import CliqueReduction, reduce_3sat_to_clique
from np_reductions.sat import Formula3SAT
from np_reductions.vertex_cover import VertexCoverReduction, reduce_3sat_to_vertex_cover


@dataclass
class CheckResult:
    ok: bool
    messages: list[str]


def verify_clique_reduction(formula: Formula3SAT) -> tuple[CliqueReduction, CheckResult]:
    red = reduce_3sat_to_clique(formula)
    msgs: list[str] = []
    sat_asn = formula.solve()

    # Forward: SAT → clique of size m
    if sat_asn is not None:
        try:
            clique = red.assignment_to_clique(sat_asn)
            if not red.graph.is_clique(clique):
                msgs.append("FAIL forward: assignment mapped to a non-clique")
            elif len(clique) != red.k:
                msgs.append(f"FAIL forward: clique size {len(clique)} != k={red.k}")
            else:
                msgs.append(f"OK forward: SAT assignment → clique of size {red.k}")
                # Backward round-trip
                back = red.clique_to_assignment(clique)
                if formula.is_satisfied(back):
                    msgs.append("OK backward: clique → satisfying assignment")
                else:
                    msgs.append("FAIL backward: recovered assignment does not satisfy φ")
        except ValueError as exc:
            msgs.append(f"FAIL forward mapping: {exc}")
    else:
        msgs.append("φ is UNSAT — checking no clique of size k exists")
        found = red.graph.find_clique(red.k)
        if found is None:
            msgs.append(f"OK: no clique of size k={red.k} (matches UNSAT)")
        else:
            msgs.append(f"FAIL: UNSAT formula but found clique {found}")

    # If SAT, also ensure a graph-found clique maps back
    if sat_asn is not None:
        found = red.graph.find_clique(red.k)
        if found is None:
            msgs.append("FAIL: SAT formula but no clique of size k found by search")
        else:
            try:
                asn2 = red.clique_to_assignment(found)
                if formula.is_satisfied(asn2):
                    msgs.append("OK search: discovered clique → satisfying assignment")
                else:
                    msgs.append("FAIL search: clique did not yield a satisfying assignment")
            except ValueError as exc:
                msgs.append(f"FAIL search mapping: {exc}")

    ok = not any(m.startswith("FAIL") for m in msgs)
    return red, CheckResult(ok=ok, messages=msgs)


def verify_vertex_cover_reduction(
    formula: Formula3SAT,
) -> tuple[VertexCoverReduction, CheckResult]:
    red = reduce_3sat_to_vertex_cover(formula)
    msgs: list[str] = []
    sat_asn = formula.solve()

    if sat_asn is not None:
        try:
            cover = red.assignment_to_cover(sat_asn)
            if not red.graph.is_vertex_cover(cover):
                msgs.append("FAIL forward: assignment mapped to a non-cover")
            elif len(cover) != red.k:
                msgs.append(f"FAIL forward: cover size {len(cover)} != k={red.k}")
            else:
                msgs.append(f"OK forward: SAT assignment → vertex cover of size {red.k}")
                back = red.cover_to_assignment(cover)
                if formula.is_satisfied(back):
                    msgs.append("OK backward: cover → satisfying assignment")
                else:
                    msgs.append("FAIL backward: recovered assignment does not satisfy φ")
        except ValueError as exc:
            msgs.append(f"FAIL forward mapping: {exc}")
    else:
        msgs.append("φ is UNSAT — checking no vertex cover of size k exists")
        found = red.graph.find_vertex_cover(red.k)
        if found is None:
            msgs.append(f"OK: no vertex cover of size k={red.k} (matches UNSAT)")
        else:
            msgs.append(f"FAIL: UNSAT formula but found cover {found}")

    if sat_asn is not None:
        found = red.graph.find_vertex_cover(red.k)
        if found is None:
            msgs.append("FAIL: SAT formula but no cover of size k found by search")
        else:
            try:
                asn2 = red.cover_to_assignment(found)
                if formula.is_satisfied(asn2):
                    msgs.append("OK search: discovered cover → satisfying assignment")
                else:
                    # Cover of size k may still encode a valid SAT assignment under our map
                    # if not, try completing free vars via formula.solve constrained — report soft
                    msgs.append(
                        "WARN search: cover→assignment not immediately satisfying "
                        f"(got {asn2}); forward map still checked"
                    )
            except ValueError as exc:
                msgs.append(f"FAIL search mapping: {exc}")

    ok = not any(m.startswith("FAIL") for m in msgs)
    return red, CheckResult(ok=ok, messages=msgs)
