"""Vertex Cover: maximal-matching 2-approximation and exact OPT."""
from __future__ import annotations

import random

from approx_algorithms.graph import Graph, exact_vertex_cover


def vertex_cover_2approx(g: Graph, *, seed: int = 0) -> set[int]:
    """
    Classic **2-approximation**:
      repeatedly pick an arbitrary remaining edge (u,v),
      add both endpoints to the cover, delete incident edges.

    Equivalent to taking both ends of a maximal matching.
    Guarantee: |C| ≤ 2 · OPT.
    """
    rng = random.Random(seed)
    remaining = list(g.edges)
    rng.shuffle(remaining)
    cover: set[int] = set()
    deleted: set[tuple[int, int]] = set()

    for u, v in remaining:
        if (u, v) in deleted:
            continue
        cover.add(u)
        cover.add(v)
        for e in remaining:
            if e in deleted:
                continue
            a, b = e
            if a == u or a == v or b == u or b == v:
                deleted.add(e)

    return cover


def vertex_cover_exact(g: Graph) -> set[int]:
    return exact_vertex_cover(g)


def is_vertex_cover(g: Graph, cover: set[int]) -> bool:
    return all(u in cover or v in cover for u, v in g.edges)
