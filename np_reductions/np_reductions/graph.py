"""Simple undirected graph helpers + brute solvers for small instances."""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools


@dataclass
class Graph:
    vertices: list[str]
    edges: set[tuple[str, str]] = field(default_factory=set)
    # optional labels for visualization
    meta: dict[str, str] = field(default_factory=dict)

    def add_edge(self, u: str, v: str) -> None:
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        self.edges.add((a, b))

    def neighbors(self, u: str) -> set[str]:
        out: set[str] = set()
        for a, b in self.edges:
            if a == u:
                out.add(b)
            elif b == u:
                out.add(a)
        return out

    def is_clique(self, nodes: set[str]) -> bool:
        nodes_l = list(nodes)
        for i, u in enumerate(nodes_l):
            for v in nodes_l[i + 1 :]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in self.edges:
                    return False
        return True

    def is_vertex_cover(self, cover: set[str]) -> bool:
        for u, v in self.edges:
            if u not in cover and v not in cover:
                return False
        return True

    def find_clique(self, k: int) -> set[str] | None:
        for combo in itertools.combinations(self.vertices, k):
            s = set(combo)
            if self.is_clique(s):
                return s
        return None

    def find_vertex_cover(self, k: int) -> set[str] | None:
        for combo in itertools.combinations(self.vertices, k):
            s = set(combo)
            if self.is_vertex_cover(s):
                return s
        return None

    def ascii(self, *, max_edges: int = 40) -> str:
        lines = [f"V = {{{', '.join(self.vertices)}}}", f"|V|={len(self.vertices)}  |E|={len(self.edges)}"]
        shown = sorted(self.edges)[:max_edges]
        lines.append("E = {")
        for a, b in shown:
            lines.append(f"  {a} — {b}")
        if len(self.edges) > max_edges:
            lines.append(f"  ... ({len(self.edges) - max_edges} more)")
        lines.append("}")
        return "\n".join(lines)
