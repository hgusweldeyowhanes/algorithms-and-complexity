"""Undirected graphs for Vertex Cover."""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import random


@dataclass
class Graph:
    n: int
    edges: set[tuple[int, int]] = field(default_factory=set)

    def add_edge(self, u: int, v: int) -> None:
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if 0 <= a < self.n and 0 <= b < self.n:
            self.edges.add((a, b))


def random_graph(n: int, p: float, *, seed: int = 0) -> Graph:
    """Erdős–Rényi G(n, p)."""
    rng = random.Random(seed)
    g = Graph(n=n)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                g.add_edge(u, v)
    return g


def adversarial_vc_star(n: int) -> Graph:
    """
    Star: OPT = 1 (center). Matching 2-approx takes both ends of one edge → 2.
    """
    if n < 2:
        raise ValueError("n>=2")
    g = Graph(n=n)
    for v in range(1, n):
        g.add_edge(0, v)
    return g


def adversarial_vc_disjoint_edges(m: int) -> Graph:
    """
    m disjoint edges. OPT = m. 2-approx takes 2m vertices. Ratio = 2 (tight).
    """
    g = Graph(n=2 * m)
    for i in range(m):
        g.add_edge(2 * i, 2 * i + 1)
    return g


def exact_vertex_cover(g: Graph) -> set[int]:
    """Brute-force minimum vertex cover (small n only)."""
    if g.n > 20:
        raise ValueError("exact VC only for n<=20")
    nodes = list(range(g.n))
    for k in range(0, g.n + 1):
        for combo in itertools.combinations(nodes, k):
            cover = set(combo)
            if all(u in cover or v in cover for u, v in g.edges):
                return cover
    return set(nodes)
