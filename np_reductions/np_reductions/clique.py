"""
3-SAT ≤p Clique

Construction (standard textbook gadget):
  • One vertex v_{i,j} for literal j in clause i (3m vertices for m clauses).
  • Edge between v_{i,j} and v_{i',j'} iff i ≠ i' and the two literals are
    consistent (not complements).

Claim: φ is satisfiable  ⟺  G has a clique of size m
  (exactly one vertex from each clause-group).
"""
from __future__ import annotations

from dataclasses import dataclass

from np_reductions.graph import Graph
from np_reductions.sat import Formula3SAT, Literal


@dataclass
class CliqueReduction:
    formula: Formula3SAT
    graph: Graph
    k: int
    # vertex id -> (clause_index, literal)
    vertex_lit: dict[str, tuple[int, Literal]]

    def assignment_to_clique(self, assignment: dict[str, bool]) -> set[str]:
        """Map a satisfying assignment to an m-clique (forward)."""
        if not self.formula.is_satisfied(assignment):
            raise ValueError("assignment does not satisfy the formula")
        clique: set[str] = set()
        for i, clause in enumerate(self.formula.clauses):
            picked = None
            for j, lit in enumerate(clause):
                if lit.eval(assignment):
                    picked = f"c{i}_l{j}"
                    break
            if picked is None:
                raise ValueError(f"clause {i} has no true literal under assignment")
            clique.add(picked)
        return clique

    def clique_to_assignment(self, clique: set[str]) -> dict[str, bool]:
        """Map an m-clique back to a satisfying assignment (backward)."""
        if len(clique) != self.k:
            raise ValueError(f"clique must have size k={self.k}")
        if not self.graph.is_clique(clique):
            raise ValueError("set is not a clique")

        # one vertex per clause
        by_clause: dict[int, Literal] = {}
        for v in clique:
            if v not in self.vertex_lit:
                raise ValueError(f"unknown vertex {v}")
            ci, lit = self.vertex_lit[v]
            if ci in by_clause:
                raise ValueError("clique picks two vertices from the same clause")
            by_clause[ci] = lit
        if len(by_clause) != len(self.formula.clauses):
            raise ValueError("clique must hit every clause group")

        assignment: dict[str, bool] = {}
        for lit in by_clause.values():
            val = lit.positive
            if lit.var in assignment and assignment[lit.var] != val:
                raise ValueError("clique encodes inconsistent literals")
            assignment[lit.var] = val
        # free vars default False
        for var in self.formula.variables:
            assignment.setdefault(var, False)
        return assignment


def reduce_3sat_to_clique(formula: Formula3SAT) -> CliqueReduction:
    m = len(formula.clauses)
    vertices: list[str] = []
    vertex_lit: dict[str, tuple[int, Literal]] = {}
    for i, clause in enumerate(formula.clauses):
        for j, lit in enumerate(clause):
            vid = f"c{i}_l{j}"
            vertices.append(vid)
            vertex_lit[vid] = (i, lit)
            # store readable meta
    g = Graph(vertices=vertices)
    for v, (i, lit) in vertex_lit.items():
        g.meta[v] = f"C{i}:{lit}"

    verts = list(vertex_lit.items())
    for a in range(len(verts)):
        va, (ia, lita) = verts[a]
        for b in range(a + 1, len(verts)):
            vb, (ib, litb) = verts[b]
            if ia == ib:
                continue
            # consistent: not complements of same variable
            if lita.var == litb.var and lita.positive != litb.positive:
                continue
            g.add_edge(va, vb)

    return CliqueReduction(formula=formula, graph=g, k=m, vertex_lit=vertex_lit)
